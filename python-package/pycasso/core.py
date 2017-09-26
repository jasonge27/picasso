# coding: utf-8
"""
Main Interface of the package
"""

import math
import numpy as np
import scipy.stats as ss
import ctypes

from .libpath import find_lib_path

class PycassoError(Exception):
    """Error thrown by pycasso solver."""
    pass

def _load_lib():
    """Load picasso library."""
    lib_path = find_lib_path()
    if not lib_path:
        PycassoError("Can not find picasso Library. Please install pycasso correctly.")
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    return lib

# load the PICASSO library globally
_PICASSO_LIB = _load_lib()

class Solver:
    """
    The PICASSO Solver For GLM.

    :param x: An `n*m` design matrix where n is the sample size and d is the data dimension.
    :param y: The *n* dimensional response vector. `y` is numeric vector for `gaussian` and `sqrtlasso`,
            or a two-level factor for `binomial`, or a non-negative integer vector representing counts
            for `gaussian`.
    :param lambdas: A sequence of decreasing positive values to control the regularization. Typical usage
            is to leave the input `lambda = None` and have the program compute its own `lambda` sequence
            based on `nlambda` and `lambda_min_ratio`. Users can also specify a sequence to override this.
            Default value is from `lambda_max` to `lambda_min_ratio*lambda_max`. The default value of
            `lambda_max` is the minimum regularization parameter which yields an all-zero estimates.
    :param nlambda: The number of values used in lambdas. Default value is 100. (useless when `lambdas` is specified)
    :param lambda_min_ratio: The smallest value for lambdas, as a fraction of the upper-bound (`MAX`) of the
            regularization parameter. The program can automatically generate `lambda` as a sequence of
            `length = nlambda` starting from `MAX` to `lambda_min_ratio` * `MAX` in log scale. The
            default value is `0.05`. **Caution**: logistic and poisson regression can be ill-conditioned
            if lambda is too small for nonconvex penalty. We suggest the user to avoid using any
            `lambda_min_raito` smaller than 0.05 for logistic/poisson regression under nonconvex penalty.
    :param lambda_min: The smallest value for `lambda`. If `lambda_min_ratio` is provided, then it is set to
            `lambda.min.ratio*MAX`, where `MAX` is the uppperbound of the regularization parameter. The default
            value is `0.05*MAX`.
    :param family: Options for model. Sparse linear regression and sparse multivariate regression is applied if
            `family = "gaussian"`, sqrt lasso is applied if `family = "sqrtlasso"`, sparse logistic regression is
            applied if `family = "binomial"` and sparse poisson regression is applied if `family = "poisson"`.
            The default value is `"gaussian"`.
    :param penalty: Options for regularization. Lasso is applied if `method = "l1"`, MCP is applied if `
            method = "mcp"` and SCAD Lasso is applied if `method = "scad"`. The default value is `"l1"`.
    :param type_gaussian: Options for updating residuals in sparse linear regression. The naive update rule is
            applied if `opt = "naive"`, and the covariance update rule is applied if `opt = "covariance"`. The
            default value is `"naive"`.
    :param gamma: The concavity parameter for MCP and SCAD. The default value is `3`.
    :param df: Maximum degree of freedom for the covariance update. The default value is `m`.
    :param standardize: Design matrix X will be standardized to have mean zero and unit standard deviation if
            `standardize = TRUE`. The default value is `TRUE`.
    :param prec: Stopping precision. The default value is 1e-7.
    :param max_ite: The iteration limit. The default value is 1000.
    :param verbose: Tracing information is disabled if `verbose = FALSE`. The default value is `FALSE`.
    """
    def __init__(self, x, y, lambdas = None, nlambda = 100, lambda_min_ratio = None,
                 lambda_min = None, family = "gaussian", penalty = "l1",
                 type_gaussian = "naive", gamma = 3, df = None, standardize = True,
                 prec = 1e-7, max_ite = 1000,  verbose = False):

        # Define the model
        if family not in ("gaussian", "binomial", "poisson", "sqrtlasso"):
            raise RuntimeError(r' Wrong "family" input. "family" should be one of "gaussian", "binomial", "poisson" and "sqrtlasso".')
        self.family = family
        if penalty not in ("l1", "mcp", "scad"):
            raise RuntimeError(r' Wrong "penalty" input. "penalty" should be one of "l1", "mcp" and "scad".')
        self.penalty = penalty

        # Define the data
        self.x = np.array(x, dtype = 'float64')
        self.y = np.array(y, dtype = 'float64')
        self.num_sample = self.x.shape[0]
        self.num_feature = self.x.shape[1]
        if self.x.size is 0:
            raise RuntimeError("Wrong: no input!")
        self.standardize = standardize
        if standardize:
            self.x_mean = np.mean(self.x, axis = 0)
            self.x_std = np.mean(self.x, axis = 0)
            self.x = ss.zscore(self.x, axis = 0, ddof = 0)
            self.y_mean = np.mean(self.y)
            self.y -= self.y_mean
        self.y = np.array(y)
        if self.x.shape[0] is not self.y.shape[0]:
            raise RuntimeError(r' the size of data "x" and label "y" does not match'+ \
                               "/nx: %i * %i, y: %i"%(self.x.shape[0],self.x.shape[1],self.y.shape[0]))

        # Define the parameters
        if df is None:
            self.df = min(self.num_feature, self.num_sample)
        else:
            self.df = df
        self.type_gaussian = type_gaussian
        self.gamma = gamma
        if self.penalty is "mcp":
            if self.gamma <= 1:
                print ("gamma have to be greater than 1 for MCP. Set to the default value 3.")
                self.gamma = 3
        if self.penalty is "scad":
            if self.gamma <= 2:
                print ("gamma have to be greater than 2 for SCAD. Set to the default value 3.")
                self.gamma = 3
        self.max_ite = max_ite
        self.prec = prec
        self.verbose = verbose

        if lambdas is not None:
            self.lambdas = lambdas
            self.nlambda = lamdas.size
        else:
            lambda_max = np.max( np.abs( np.matmul(self.x.T, self.y) ) )
            if lambda_min_ratio is None:
                if lambda_min is None:
                    lambda_min_ratio = 0.05
                else:
                    lambda_min_ratio = 1. * lambda_min/lambda_max
            self.nlambda = nlambda
            self.lambdas = np.linspace(1,math.log(lambda_min_ratio),nlambda)
            self.lambdas = lambda_max * np.exp(self.lambdas)

        # register trainer
        self.trainer = getattr(self, '_'+self.family+'_wrapper')()
        self.result = {'beta': [1,1,1,1,1]}

    def __del__(self):
        pass

    def _gaussian_wrapper(self):
        """
        A wrapper for linear regression, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
        if self.verbose:
            print("Sparse linear regression.")
            print(self.penalty.upper() + "regularization via active set identification and coordinate descent. \n")
        if self.type_gaussian not in ("covariance", "naive"):
            print(r'Automatically set "type_gaussian", since "type_gaussian" is not one of "covariance", "naive"'+'\n')
            if self.num_sample < 500:
                self.type_gaussian = "covariance"
            else:
                self.type_gaussian = "naive"

        return lambda: _PICASSO_LIB.test()

    def _binomial_wrapper(self):
        """
        A wrapper for logistic regression, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
        if self.verbose:
            print("Sparse logistic regression. \n")
            print(self.penalty.upper() + "regularization via active set identification and coordinate descent. \n")
        return lambda:0

    def _poisson_wrapper(self):
        """
        A wrapper for poisson regression, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
        if self.verbose:
            print("Sparse poisson regression. \n")
            print(self.penalty.upper() + "regularization via active set identification and coordinate descent. \n")
        return lambda:0

    def _sqrtlasso_wrapper(self):
        """
        A wrapper for sqrt lasso, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
        if self.verbose:
            print("Sparse logistic regression. \n")
            print(self.penalty.upper() + "regularization via active set identification and coordinate descent. \n")
        return lambda:0

    def train(self):
        """
        The trigger function for training the model
        """
        self.result = self.trainer()
        # TODO: deal wth error
        # if (out$err == 1)
        # cat("Error! Parameters are too dense. Please choose larger \"lambda\". \n")
        # if (out$err == 2)
        # cat("Warning! \"df\" may be too small. You may choose larger \"df\". \n")

    def coef(self):
        """
        Extract model coefficients.

        :return: a dictionary of the model coefficients.
        :rtype: dict{name : value}
        """
        pass
        return {}

    def plot(self):
        """
        Visualize the solution path of regression estimate corresponding to regularization parameters.
        """
        pass

    def predict(self, newdata = None):
        """
        Predicting responses of the new data.

        :param newdata: An optional data frame in which to look for variables with which to predict.
                        If omitted, the training data of the model are used.
        :return: The predicted response vectors based on the estimated models.
        """
        pass
        return 0

    def __str__(self):
        """
        A summary of the information about an object

        :return: a summary string
        :rtype: string
        """
        return  "Model Type: " + self.family + "\n" +\
                "Penalty Type: " + self.penalty + "\n" + \
                "Sample Number: " + str(self.num_sample) + "\n" + \
                "Feature Number: " + str(self.num_sample) + "\n"