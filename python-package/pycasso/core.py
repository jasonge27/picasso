# coding: utf-8
"""
Main Interface of the package
"""

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

class Solver:
    """
    The PICASSO Solver For GLM.

    :param x: An `n*m` design matrix where n is the sample size and d is the data dimension.
    :param y: The *n* dimensional response vector. `y` is numeric vector for `gaussian` and `sqrtlasso`,
            or a two-level factor for `binomial`, or a non-negative integer vector representing counts
            for `gaussian`.
    :param lambdas: A sequence of decreasing positive values to control the regularization. Typical usage
            is to leave the input `lambda = ()` and have the program compute its own `lambda` sequence
            based on `nlambda` and `lambda_min_ratio`. Users can also specify a sequence to override this.
            Default value is from `lambda_max` to `lambda_min_ratio*lambda_max`. The default value of
            `lambda_max` is the minimum regularization parameter which yields an all-zero estimates.
    :param nlambda: The number of values used in lambdas. Default value is 100.
    :param lambda_min_ratio: The smallest value for lambdas, as a fraction of the upper-bound (`MAX`) of the
            regularization parameter. The program can automatically generate `lambda` as a sequence of
            `length = nlambda` starting from `MAX` to `lambda_min_ratio` * `MAX` in log scale. The
            default value is `0.05`. **Caution**: logistic and poisson regression can be ill-conditioned
            if lambda is too small for nonconvex penalty. We suggest the user to avoid using any
            `lambda_min_raito` smaller than 0.05 for logistic/poisson regression under nonconvex penalty.
    :param lambda_min: The smallest value for `lambda`. If `lambda_min_ratio` is provided, then it is set to
            `lambda.min.ratio*MAX`, where `MAX` is the uppperbound of the regularization parameter. The default
            value is `0.1*MAX`.
    :param family: Options for model. Sparse linear regression and sparse multivariate regression is applied if
            `family = "gaussian"`, sqrt lasso is applied if `family = "sqrtlasso"`, sparse logistic regression is
            applied if `family = "binomial"` and sparse poisson regression is applied if `family = "poisson"`.
            The default value is `"gaussian"`.
    :param method: Options for regularization. Lasso is applied if `method = "l1"`, MCP is applied if `
            method = "mcp"` and SCAD Lasso is applied if `method = "scad"`. The default value is `"l1"`.
    :param type_gaussian: Options for updating residuals in sparse linear regression. The naive update rule is
            applied if `opt = "naive"`, and the covariance update rule is applied if `opt = "covariance"`. The
            default value is `"naive"`.
    :param gamma: The concavity parameter for MCP and SCAD. The default value is `3`.
    :param df: Maximum degree of freedom for the covariance update. The default value is `2*n`.
    :param standardize: Design matrix X will be standardized to have mean zero and unit standard deviation if
            `standardize = TRUE`. The default value is `TRUE`.
    :param prec: Stopping precision. The default value is 1e-7.
    :param max_ite: The iteration limit. The default value is 1000.
    :param verbose: Tracing information is disabled if `verbose = FALSE`. The default value is `FALSE`.
    """
    def __init__(self,x, y, lambdas = (), nlambda = 100, lambda_min_ratio = 0.05,
                 lambda_min = (), family = "gaussian", method = "l1",
                 type_gaussian = "naive", gamma = 3, df = (), standardize = True,
                 prec = 1e-7, max_ite = 1000,  verbose = False):
        self.x = x
        self.y = y

    def __del__(self):
        pass

    def coef(self):
        """
        Extract model coefficients.

        :return: a dictionary of the model coefficients.
        :rtype: dict{name : value}
        """
        pass
        return {}

    def lalala(self,a,b,g,s,e):
        """

        :param a:
        :param b:
        :param g:
        :param s:
        :param e:
        :return sth: val
        :type sth: int
        """
        return 1