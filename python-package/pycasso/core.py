# coding: utf-8
"""
Main Interface of the package
"""

import time
import math
import numpy as np
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
    raise PycassoError(
        "Can not find picasso Library. Please install pycasso correctly.")
  lib = ctypes.cdll.LoadLibrary(lib_path[0])
  return lib


# load the PICASSO library globally
_PICASSO_LIB = _load_lib()


def _standardize(x):
  """Standardize design matrix: center and scale each column.

  Matches the R picasso standardization:
    xm[j] = mean of column j
    xx[j] = (x[j] - xm[j]) * xinvc[j]
    xinvc[j] = 1 / sqrt( sum((x[j]-xm[j])^2) / (n-1) )   (i.e. 1/sd with ddof=1)

  :param x: design matrix (n, d)
  :return: (xx, xm, xinvc) where xx is standardized, xm is column means,
           xinvc is 1/sd scaling factors
  """
  n = x.shape[0]
  xm = np.mean(x, axis=0)               # (d,)
  xx = x - xm                            # centered
  col_ss = np.sum(xx ** 2, axis=0)       # sum of squares per column
  xinvc = np.where(col_ss > 0, 1.0 / np.sqrt(col_ss / (n - 1)), 0.0)
  xx = xx * xinvc                         # scaled
  return xx, xm, xinvc


def _rescale_solution(beta, intercept, xinvc, xm):
  """Rescale coefficients from standardized space back to original space.

  :param beta: (nlambda, d) coefficient matrix from solver
  :param intercept: (nlambda,) intercept array from solver
  :param xinvc: (d,) scaling factors from standardization
  :param xm: (d,) column means from standardization
  :return: (beta_rescaled, intercept_rescaled)
  """
  beta_rescaled = beta * xinvc  # broadcast (nlambda, d) * (d,)
  intercept_rescaled = intercept - beta_rescaled @ xm
  return beta_rescaled, intercept_rescaled


class Solver:
  """
    The PICASSO Solver For GLM.

    :param x: An ``n*d`` design matrix where n is the sample size and d is the data dimension.
    :param y: The *n* dimensional response vector. ``y`` is numeric vector for ``gaussian`` and ``sqrtlasso``,
            or a two-level factor for ``binomial``, or a non-negative integer vector representing counts
            for ``poisson``.
    :param lambdas: The parameters of controlling regularization. Can be one of the following two cases:

            **Case1 (default)**: A tuple of 2 variables ``(n, lambda_min_ratio)``, where the default values are
            ``(100, 0.05)``. The program calculates ``lambdas`` as an array of ``n`` elements starting from
            ``lambda_max`` to ``lambda_min_ratio * lambda_max`` in log scale. ``lambda_max`` is the minimum
            regularization parameter which yields an all-zero estimate.

            **Case2**: A manually specified sequence (size > 2) of decreasing positive values to control
            the regularization.
    :param family: Options for model. Sparse linear regression is applied if
            ``family = "gaussian"``, sqrt lasso is applied if ``family = "sqrtlasso"``, sparse logistic
            regression is applied if ``family = "binomial"`` and sparse poisson regression is applied if
            ``family = "poisson"``. The default value is ``"gaussian"``.
    :param penalty: Options for regularization. Lasso is applied if ``penalty = "l1"``, MCP is applied if
            ``penalty = "mcp"`` and SCAD is applied if ``penalty = "scad"``. The default value is ``"l1"``.
    :param gamma: The concavity parameter for MCP and SCAD. The default value is ``3``.
    :param useintercept: Whether or not to include intercept term. Default value is ``True``.
    :param standardize: Whether or not to standardize the design matrix. Default value is ``True``.
    :param type_gaussian: The type of Gaussian solver. ``"naive"`` uses naive update, ``"covariance"``
            uses covariance update. Only used when ``family = "gaussian"``. Default is ``"naive"``.
    :param dfmax: Maximum number of nonzero coefficients for early stopping. ``-1`` means no limit.
    :param prec: Stopping precision. The default value is ``1e-7``.
    :param max_ite: The iteration limit. The default value is ``1000``.
    :param verbose: Tracing information is disabled if ``verbose = False``. The default value is ``False``.
    """

  def __init__(self,
               x,
               y,
               lambdas=(100, 0.05),
               family="gaussian",
               penalty="l1",
               gamma=3,
               useintercept=True,
               standardize=True,
               type_gaussian="naive",
               dfmax=-1,
               prec=1e-7,
               max_ite=1000,
               verbose=False):

    # Validate model
    if family not in ("gaussian", "binomial", "poisson", "sqrtlasso"):
      raise ValueError(
          'Invalid "family". Must be one of "gaussian", "binomial", "poisson", "sqrtlasso".'
      )
    self.family = family
    if penalty not in ("l1", "mcp", "scad"):
      raise ValueError(
          'Invalid "penalty". Must be one of "l1", "mcp", "scad".'
      )
    self.penalty = penalty
    self.use_intercept = useintercept
    self.standardize = standardize
    self.dfmax = int(dfmax)

    if family == "gaussian":
      if type_gaussian not in ("naive", "covariance"):
        raise ValueError(
            'Invalid "type_gaussian". Must be one of "naive", "covariance".'
        )
    self.type_gaussian = type_gaussian

    # Validate and store data
    x_raw = np.asarray(x, dtype='double')
    self.y = np.ascontiguousarray(y, dtype='double')
    self.num_sample = x_raw.shape[0]
    self.num_feature = x_raw.shape[1]
    if x_raw.size == 0:
      raise ValueError("No data input.")
    if x_raw.shape[0] != self.y.shape[0]:
      raise ValueError(
          'The size of "x" and "y" does not match: '
          'x: %d * %d, y: %d' % (x_raw.shape[0], x_raw.shape[1], self.y.shape[0]))

    # Family-specific validation
    if self.family == "binomial":
      levels = np.unique(self.y)
      if (levels.size != 2) or (1 not in levels) or (0 not in levels):
        raise ValueError("Response vector should contain only 0s and 1s.")
    elif self.family == "poisson":
      if np.any(self.y < 0):
        raise ValueError("The response vector should be non-negative.")
      if not np.allclose(self.y, np.round(self.y)):
        raise ValueError("The response vector should be integers.")
      if np.allclose(self.y, 0):
        raise ValueError(
            "The response vector is an all-zero vector. The problem is ill-conditioned.")
      self.y = np.round(self.y).astype('double')

    # Standardize design matrix
    if self.standardize:
      xx, self._xm, self._xinvc = _standardize(x_raw)
      self.x = np.ascontiguousarray(xx, dtype='double')
    else:
      self.x = np.ascontiguousarray(x_raw, dtype='double')
      self._xm = np.zeros(self.num_feature)
      self._xinvc = np.ones(self.num_feature)

    # For Gaussian, center Y when standardizing
    self._ym = 0.0
    if self.family == "gaussian" and self.standardize:
      self._ym = np.mean(self.y)
      self._y_fit = np.ascontiguousarray(self.y - self._ym, dtype='double')
    else:
      self._y_fit = self.y

    # Penalty parameters
    self.gamma = gamma
    if self.penalty == "mcp":
      self.penaltyflag = 2
      if self.gamma <= 1:
        print("gamma must be greater than 1 for MCP. Set to the default value 3.")
        self.gamma = 3
    elif self.penalty == "scad":
      self.penaltyflag = 3
      if self.gamma <= 2:
        print("gamma must be greater than 2 for SCAD. Set to the default value 3.")
        self.gamma = 3
    else:
      self.penaltyflag = 1
    self.max_ite = max_ite
    self.prec = prec
    self.verbose = verbose

    # Compute lambda path
    n = self.num_sample
    if len(lambdas) > 2:
      self.lambdas = np.array(lambdas, dtype='double')
      self.nlambda = len(lambdas)
    else:
      nlambda = int(lambdas[0])
      lambda_min_ratio = lambdas[1]
      if self.family == 'poisson':
        lambda_max = np.max(
            np.abs(self.x.T @ ((self._y_fit - np.mean(self._y_fit)) / n)))
      elif self.family == 'sqrtlasso':
        L0 = np.sqrt(np.sum(self._y_fit ** 2) / n)
        lambda_max = np.max(np.abs(self.x.T @ self._y_fit)) / n / L0
      else:
        lambda_max = np.max(np.abs(self.x.T @ self._y_fit)) / n
      if lambda_min_ratio > 1:
        raise ValueError('"lambda_min_ratio" must be <= 1.')
      self.nlambda = nlambda
      self.lambdas = np.exp(np.linspace(
          math.log(lambda_max), math.log(lambda_min_ratio * lambda_max),
          self.nlambda)).astype('double')

    # Store original x for predict
    self._x_orig = np.asarray(x_raw, dtype='double')

    # Register trainer
    self.trainer = getattr(self, '_' + self.family + '_wrapper')()
    self.result = {
        'beta': np.zeros((self.nlambda, self.num_feature), dtype='double'),
        'intercept': np.zeros(self.nlambda, dtype='double'),
        'ite_lamb': np.zeros(self.nlambda, dtype='int32'),
        'size_act': np.zeros(self.nlambda, dtype='int32'),
        'df': np.zeros(self.nlambda, dtype='int32'),
        'train_time': np.zeros(self.nlambda, dtype='double'),
        'num_fit': np.zeros(1, dtype='int32'),
        'total_train_time': 0,
        'state': 'not trained'
    }

  def _decor_cinterface(self, _function):
    """Decorate a C API function with correct ctypes signatures and result handling.

    :param _function: the raw c function
    :return: the decorated function
    :rtype: function
    """
    CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
    CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
    _function.argtypes = [
        CDoubleArray, CDoubleArray, ctypes.c_int, ctypes.c_int, CDoubleArray,
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_bool,
        ctypes.c_int,        # dfmax
        CDoubleArray, CDoubleArray, CIntArray,
        CIntArray, CDoubleArray,
        CIntArray,           # num_fit
        ctypes.c_bool        # usePython
    ]

    def wrapper():
      time_start = time.time()
      _function(self._y_fit, self.x, self.num_sample, self.num_feature, self.lambdas,
                self.nlambda, self.gamma, self.max_ite, self.prec,
                self.penaltyflag, self.use_intercept,
                self.dfmax,
                self.result['beta'],
                self.result['intercept'], self.result['ite_lamb'],
                self.result['size_act'], self.result['train_time'],
                self.result['num_fit'],
                True)  # usePython=True
      time_end = time.time()
      self.result['total_train_time'] = time_end - time_start

      # Truncate to actual number of lambdas fit (early stopping)
      nfit = int(self.result['num_fit'][0])
      if 0 < nfit < self.nlambda:
        self.result['beta'] = self.result['beta'][:nfit, :]
        self.result['intercept'] = self.result['intercept'][:nfit]
        self.result['ite_lamb'] = self.result['ite_lamb'][:nfit]
        self.result['size_act'] = self.result['size_act'][:nfit]
        self.result['train_time'] = self.result['train_time'][:nfit]
        self.nlambda = nfit
        self.lambdas = self.lambdas[:nfit]

      # Rescale from standardized space back to original space
      if self.standardize:
        self.result['beta'], self.result['intercept'] = _rescale_solution(
            self.result['beta'], self.result['intercept'],
            self._xinvc, self._xm)

      # Gaussian: add back Y mean to intercept
      if self.family == 'gaussian' and self._ym != 0.0:
        self.result['intercept'] = self.result['intercept'] + self._ym

      self.result['df'] = np.sum(self.result['beta'] != 0, axis=1).astype('int32')

    return wrapper

  def _gaussian_wrapper(self):
    """A wrapper for linear regression.

    :return: A function which can be used for training
    :rtype: function
    """
    if self.verbose:
      print("Sparse linear regression.")
      print(self.penalty.upper()
            + " regularization via active set identification and coordinate descent.\n")

    if self.type_gaussian == "covariance":
      return self._decor_cinterface(_PICASSO_LIB.SolveLinearRegressionCovUpdate)
    return self._decor_cinterface(_PICASSO_LIB.SolveLinearRegressionNaiveUpdate)

  def _binomial_wrapper(self):
    """A wrapper for logistic regression.

    :return: A function which can be used for training
    :rtype: function
    """
    if self.verbose:
      print("Sparse logistic regression.")
      print(self.penalty.upper()
            + " regularization via active set identification and coordinate descent.\n")
    return self._decor_cinterface(_PICASSO_LIB.SolveLogisticRegression)

  def _poisson_wrapper(self):
    """A wrapper for poisson regression.

    :return: A function which can be used for training
    :rtype: function
    """
    if self.verbose:
      print("Sparse poisson regression.")
      print(self.penalty.upper()
            + " regularization via active set identification and coordinate descent.\n")
    return self._decor_cinterface(_PICASSO_LIB.SolvePoissonRegression)

  def _sqrtlasso_wrapper(self):
    """A wrapper for sqrt lasso regression.

    :return: A function which can be used for training
    :rtype: function
    """
    if self.verbose:
      print("Sparse sqrt lasso regression.")
      print(self.penalty.upper()
            + " regularization via active set identification and coordinate descent.\n")
    return self._decor_cinterface(_PICASSO_LIB.SolveSqrtLinearRegression)

  def train(self):
    """Train the model along the regularization path."""
    self.result['state'] = 'trained'
    self.trainer()
    if self.verbose:
      print('Training is over.')

  def coef(self):
    """Extract model coefficients.

    :return: a dictionary of the model coefficients.
    :rtype: dict

    The detail of returned dict:

        - **beta** - A matrix of regression estimates whose columns correspond to regularization parameters.
        - **intercept** - The value of intercepts corresponding to regularization parameters.
        - **ite_lamb** - Number of iterations for each lambda.
        - **size_act** - An array of solution sparsity (model degree of freedom).
        - **train_time** - The training time on each lambda.
        - **total_train_time** - The total training time.
        - **state** - The training state.
        - **df** - The number of nonzero coefficients.
        - **num_fit** - The number of lambdas actually fit (may be less than requested if early stopping via ``dfmax``).

    """
    if self.result['state'] == 'not trained':
      print('Warning: The model has not been trained yet!')
    return self.result

  def plot(self):
    """Visualize the solution path of regression estimate corresponding to regularization parameters."""
    import matplotlib.pyplot as plt
    plt.plot(self.lambdas, self.result['beta'])
    plt.ylabel('Coefficient')
    plt.xlabel('Regularization Parameter')
    plt.suptitle('Regularization Path')
    plt.show()

  def predict(self, newdata=None, lambdidx=None):
    """Predict responses for new data.

    :param newdata: An optional data matrix for prediction.
                    If omitted, the original (unstandardized) training data are used.
    :param lambdidx: Use the model coefficient corresponding to the ``lambdidx`` th lambda.
                     Defaults to the last lambda.

    :return: The predicted response vectors based on the estimated models.
    :rtype: np.array
    """
    if lambdidx is None:
      lambdidx = self.nlambda - 1

    _beta = self.result['beta'][lambdidx, :]
    _intercept = self.result['intercept'][lambdidx]
    if newdata is None:
      y_pred = self._x_orig @ _beta + _intercept
    else:
      y_pred = np.asarray(newdata, dtype='double') @ _beta + _intercept

    return y_pred

  def __str__(self):
    """A summary of the model.

    :return: a summary string
    :rtype: string
    """
    return_str = "Model Type: " + self.family + "\n" + \
                 "Penalty Type: " + self.penalty + "\n" + \
                 "Sample Number: " + str(self.num_sample) + "\n" + \
                 "Feature Number: " + str(self.num_feature) + "\n" + \
                 "Lambda Number: " + str(self.nlambda) + "\n"
    if self.result['state'] == 'trained':
      return_str += "Training Time (s): " + str(
          self.result['total_train_time']) + "\n"

    return return_str
