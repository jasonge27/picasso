# coding: utf-8
"""
Primary Python interface for PICASSO solvers.
"""

import time
import math
import numpy as np
import scipy.stats as ss
import ctypes
from numpy.ctypeslib import ndpointer

from .libpath import find_lib_path

__all__ = ["Solver"]

class PycassoError(Exception):
  """Error thrown by pycasso solver."""
  pass


def _load_lib():
  """Load picasso library."""
  lib_path = find_lib_path()
  if not lib_path:
    PycassoError(
        "Can not find picasso Library. Please install pycasso correctly.")
  lib = ctypes.cdll.LoadLibrary(lib_path[0])
  return lib


# load the PICASSO library globally
_PICASSO_LIB = _load_lib()


class Solver:
  """
    PICASSO solver for sparse GLM and square-root lasso.

    :param x: Numeric design matrix with shape ``(n_samples, n_features)``.
    :param y: Response vector of length ``n_samples``.
              Use numeric values for ``gaussian``/``sqrtlasso``,
              binary values in ``{0, 1}`` for ``binomial``,
              and non-negative integers for ``poisson``.
    :param lambdas: Regularization path specification. Use one of:
              (1) tuple ``(nlambda, lambda_min_ratio)`` to auto-generate
              a logarithmic path from ``lambda_max`` to
              ``lambda_min_ratio * lambda_max``;
              (2) explicit decreasing positive sequence.
    :param family: One of ``"gaussian"``, ``"binomial"``,
              ``"poisson"``, or ``"sqrtlasso"``.
    :param penalty: One of ``"l1"``, ``"mcp"``, or ``"scad"``.
    :param gamma: Concavity parameter for MCP/SCAD penalties.
    :param useintercept: Whether to include an intercept term.
    :param prec: Optimization stopping tolerance.
    :param max_ite: Maximum number of iterations per lambda.
    :param verbose: Whether to print training diagnostics.

    Notes
    -----
    For nonconvex logistic/poisson fits (MCP/SCAD), very small lambda values
    can make optimization unstable in practice. A common default is
    ``lambda_min_ratio >= 0.05``.
    """

  def __init__(self,
               x,
               y,
               lambdas=(100,0.05),
               family="gaussian",
               penalty="l1",
               gamma=3,
               useintercept=False,
               prec=1e-4,
               max_ite=1000,
               verbose=False):

    # Define the model
    if family not in ("gaussian", "binomial", "poisson", "sqrtlasso"):
      raise RuntimeError(
          r' Wrong "family" input. "family" should be one of "gaussian", "binomial", "poisson" and "sqrtlasso".'
      )
    self.family = family
    if penalty not in ("l1", "mcp", "scad"):
      raise RuntimeError(
          r' Wrong "penalty" input. "penalty" should be one of "l1", "mcp" and "scad".'
      )
    self.penalty = penalty
    self.use_intercept = useintercept

    # Define the data
    self.x = np.asfortranarray(x, dtype='double')
    self.y = np.ascontiguousarray(y, dtype='double')
    self.num_sample = self.x.shape[0]
    self.num_feature = self.x.shape[1]
    if self.x.size == 0:
      raise RuntimeError("Wrong: no input!")
    if self.x.shape[0] != self.y.shape[0]:
      raise RuntimeError(r' the size of data "x" and label "y" does not match'+ \
                         "/nx: %i * %i, y: %i"%(self.x.shape[0],self.x.shape[1],self.y.shape[0]))

    # Define the parameters
    self.gamma = gamma
    if self.penalty == "mcp":
      self.penaltyflag = 2
      if self.gamma <= 1:
        print(
            "gamma have to be greater than 1 for MCP. Set to the default value 3."
        )
        self.gamma = 3
    elif self.penalty == "scad":
      self.penaltyflag = 3
      if self.gamma <= 2:
        print(
            "gamma have to be greater than 2 for SCAD. Set to the default value 3."
        )
        self.gamma = 3
    else:  # self.penalty is "l1":
      self.penaltyflag = 1
    self.max_ite = max_ite
    self.prec = prec
    self.verbose = verbose

    if len(lambdas) > 2:
      self.lambdas = np.array(lambdas, dtype='double')
      self.nlambda = len(lambdas)
    else:
      nlambda = int(lambdas[0])
      lambda_min_ratio = lambdas[1]
      if self.family == 'poisson':
        lambda_max = np.max(
            np.abs(np.matmul(self.x.T,
                             (self.y - np.mean(self.y))/ self.num_sample )))
      elif self.family == 'sqrtlasso':
        lambda_max = np.max( np.abs( np.matmul(self.x.T, self.y) ) ) /self.num_sample \
                     /np.sqrt(np.sum(self.y**2)/self.num_sample)
      else:
        lambda_max = np.max(
            np.abs(np.matmul(self.x.T, self.y))) / self.num_sample
      if lambda_min_ratio > 1:
        raise RuntimeError(r'"lambda_min_ratio" is too small.')
      self.nlambda = nlambda
      self.lambdas = np.linspace(
          math.log(1), math.log(lambda_min_ratio), self.nlambda, dtype='double')
      self.lambdas = lambda_max * np.exp(self.lambdas)
      self.lambdas = np.array(self.lambdas, dtype='double')

    # register trainer
    self.trainer = getattr(self, '_' + self.family + '_wrapper')()
    self.result = {
        'beta': np.zeros((self.nlambda, self.num_feature), dtype='double'),
        'intercept': np.zeros(self.nlambda, dtype='double'),
        'ite_lamb': np.zeros(self.nlambda, dtype='int32'),
        'size_act': np.zeros((self.nlambda, self.num_feature), dtype='int32'),
        'df': np.zeros(self.nlambda, dtype='int32'),
        'train_time': np.zeros(self.nlambda, dtype='double'),
        'total_train_time': 0,
        'state': 'not trained'
    }

  def __del__(self):
    pass

  def _decor_cinterface(self, _function):
    """
        Since all c functions take the same input, we can provide a unified decorator for defining the C interface

        :param _function: the raw c function
        :return: the decorated function
        :rtype: function
        """
    """
        C Function parameters:
            double *Y,       // input: model response
            double *X,       // input: model covariates
            int nn,          // input: number of samples
            int dd,          // input: dimension
            double *lambda,  // input: regularization parameter
            int nnlambda,    // input: number of lambda on the regularization path
            double gamma,    // input: gamma for SCAD or MCP penalty
            int mmax_ite,    // input: max number of interations
            double pprec,    // input: optimization precision
            int reg_type,    // input: type of regularization
            bool intercept,  // input: to have intercept term or not
            double *beta,    // output: an nlambda * d dim matrix
                             //         saving the coefficients for each lambda
            double *intcpt,  // output: an nlambda dim array
                             //         saving the model intercept for each lambda
            int *ite_lamb,   // output: number of iterations for each lambda
            int *size_act,   // output: an array of solution sparsity (model df)
            double *runt     // output: runtime
        """
    FDoubleArray = ndpointer(ctypes.c_double, flags='F_CONTIGUOUS')
    CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
    CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
    _function.argtypes = [
        CDoubleArray, FDoubleArray, ctypes.c_int, ctypes.c_int, CDoubleArray,
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_bool, CDoubleArray, CDoubleArray, CIntArray,
        CIntArray, CDoubleArray
    ]

    def wrapper():
      time_start = time.time()
      _function(self.y, self.x, self.num_sample, self.num_feature, self.lambdas,
                self.nlambda, self.gamma, self.max_ite, self.prec,
                self.penaltyflag, self.use_intercept, self.result['beta'],
                self.result['intercept'], self.result['ite_lamb'],
                self.result['size_act'], self.result['train_time'])
      time_end = time.time()
      self.result['total_train_time'] = time_end - time_start
      self.result['df'] = sum(self.result['beta'].T!=0)

    return wrapper

  def _gaussian_wrapper(self):
    """
        A wrapper for linear regression, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
    if self.verbose:
      print("Sparse linear regression.")
      print(self.penalty.upper(
      ) + "regularization via active set identification and coordinate descent. \n"
           )

    return self._decor_cinterface(_PICASSO_LIB.SolveLinearRegressionNaiveUpdate)

  def _binomial_wrapper(self):
    """
        A wrapper for logistic regression, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
    if self.verbose:
      print("Sparse logistic regression. \n")
      print(self.penalty.upper(
      ) + "regularization via active set identification and coordinate descent. \n"
           )
    levels = np.unique(self.y)
    if (levels.size != 2) or (1 not in levels) or (0 not in levels):
      raise RuntimeError("Response vector should contains 0s and 1s.")
    return self._decor_cinterface(_PICASSO_LIB.SolveLogisticRegression)

  def _poisson_wrapper(self):
    """
        A wrapper for poisson regression, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
    if self.verbose:
      print("Sparse poisson regression. \n")
      print(self.penalty.upper(
      ) + "regularization via active set identification and coordinate descent. \n"
           )
    if np.any(np.less(self.y, 0)):
      raise RuntimeError("The response vector should be non-negative.")
    if not np.allclose(self.y, np.round(self.y)):
      raise RuntimeError("The response vector should be integers.")
    if np.allclose(self.y, 0):
      raise RuntimeError(
          "The response vector is an all-zero vector. The problem is ill-conditioned."
      )
    self.y = np.round(self.y)

    return self._decor_cinterface(_PICASSO_LIB.SolvePoissonRegression)

  def _sqrtlasso_wrapper(self):
    """
        A wrapper for sqrt lasso, including some specialized parameter checks.

        :return: A function which can be used for training
        :rtype: function
        """
    if self.verbose:
      print("Sparse logistic regression. \n")
      print(self.penalty.upper(
      ) + "regularization via active set identification and coordinate descent. \n"
           )

    return self._decor_cinterface(_PICASSO_LIB.SolveSqrtLinearRegression)

  def train(self):
    """
        Train the model along the lambda path.
        """
    self.result['state'] = 'trained'
    self.trainer()
    if self.verbose:
      print('Training is over.')

  def coef(self):
    """
        Return fitted coefficients and training diagnostics.

        :return: dictionary containing model path results.
        :rtype: dict

        Main fields in the returned dictionary:

            - **beta**: coefficient path with shape ``(nlambda, n_features)``
            - **intercept**: intercept path with shape ``(nlambda,)``
            - **ite_lamb**: iteration count for each lambda
            - **size_act**: active-set size over iterations
            - **df**: number of nonzero coefficients per lambda
            - **train_time**: runtime per lambda
            - **total_train_time**: end-to-end runtime
            - **state**: training state string

        """
    if self.result['state'] == 'not trained':
      print(r'Warning: The model has not been trained yet! ')
    return self.result

  def plot(self):
    """
        Visualize the solution path of regression estimate corresponding to regularization parameters.
        """
    import matplotlib.pyplot as plt
    plt.plot(self.lambdas, self.result['beta'])
    plt.ylabel('Coefficient')
    plt.xlabel('Regularization Parameter')
    plt.suptitle('Regularization Path')
    plt.show()

  def predict(self, newdata=None, lambdidx=None):
    """
        Predict responses for new data.

        :param newdata: Optional feature matrix with shape
                        ``(n_new_samples, n_features)``.
                        If omitted, the training matrix is used.
        :param lambdidx: Index of the lambda model to use.
                         Defaults to the last lambda in the path.

        :return: Predicted response vector.
        :rtype: np.array
        """
    if lambdidx is None:
      lambdidx = self.nlambda - 1

    _beta = np.copy(self.result['beta'][lambdidx,])
    _intercept = np.copy(self.result['intercept'][lambdidx])
    if newdata is None:
      y_pred = np.matmul(self.x, _beta) + _intercept
    else:
      y_pred = np.matmul(newdata, _beta) + _intercept

    return y_pred

  def __str__(self):
    """
        A summary of the information about an object

        :return: a summary string
        :rtype: string
        """
    return_str = "Model Type: " + self.family + "\n" + \
                 "Penalty Type: " + self.penalty + "\n" + \
                 "Sample Number: " + str(self.num_sample) + "\n" + \
                 "Feature Number: " + str(self.num_feature) + "\n" + \
                 "Lambda Number: " + str(self.nlambda) + "\n"
    if self.result['state']:
      return_str += "Training Time (ms): " + str(
          self.result['total_train_time']) + "\n"

    return return_str
