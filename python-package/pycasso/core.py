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
    """Standardize design matrix: center and scale each column."""
    n = x.shape[0]
    xm = np.mean(x, axis=0)
    xx = x - xm
    col_ss = np.sum(xx ** 2, axis=0)
    xinvc = np.where(col_ss > 0, 1.0 / np.sqrt(col_ss / (n - 1)), 0.0)
    xx = xx * xinvc
    return xx, xm, xinvc


def _rescale_solution(beta, intercept, xinvc, xm):
    """Rescale coefficients from standardized space back to original space."""
    beta_rescaled = beta * xinvc
    intercept_rescaled = intercept - beta_rescaled @ xm
    return beta_rescaled, intercept_rescaled


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(lp):
    """lp: (n, K) array. Returns (n, K) probabilities."""
    lp_shifted = lp - lp.max(axis=1, keepdims=True)
    ep = np.exp(lp_shifted)
    return ep / ep.sum(axis=1, keepdims=True)


def _poisson_dev(y, mu):
    """Poisson deviance D = 2*mean(y*log(y/mu) - (y-mu)), always >= 0.
    Convention: 0*log(0) = 0.
    """
    mu = np.maximum(mu, 1e-15)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(y > 0, np.log(y / mu), 0.0)
    term = np.where(y > 0, y * log_ratio - (y - mu), mu - y)
    return 2.0 * np.mean(term)


def _null_deviance(y, family, offset=None):
    """Compute null-model deviance (intercept only, optionally with offset)."""
    n = len(y)
    off = offset if offset is not None else np.zeros(n)
    if family in ("gaussian", "sqrtlasso"):
        return np.sum((y - np.mean(y)) ** 2) / (2.0 * n)
    elif family == "binomial":
        p0 = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0))
    elif family == "poisson":
        # Null model with offset: log(mu_i) = offset_i + c
        # MLE: sum(exp(offset_i + c)) = sum(y_i) → c = log(mean(y) / mean(exp(offset)))
        exp_off = np.exp(np.clip(off, -500, 500))
        mean_y = np.mean(y)
        mean_eoff = np.mean(exp_off)
        c = np.log(mean_y / mean_eoff) if (mean_eoff > 0 and mean_y > 0) else 0.0
        mu0 = exp_off * np.exp(c)
        return _poisson_dev(y, mu0)
    elif family == "multinomial":
        return None  # handled separately
    return None


def _fit_deviances(y, x, beta, intercept, family, offset=None):
    """Compute per-lambda deviance (NLL) for non-multinomial families.

    beta: (nlambda, d), intercept: (nlambda,)
    offset: (n,) optional per-observation offset (binomial/poisson only)
    Returns (nlambda,) array.
    """
    nlambda = beta.shape[0]
    devs = np.zeros(nlambda)
    off = offset if offset is not None else 0.0
    for i in range(nlambda):
        eta = x @ beta[i] + intercept[i] + off
        if family in ("gaussian", "sqrtlasso"):
            devs[i] = np.mean((y - eta) ** 2) / 2.0
        elif family == "binomial":
            p = np.clip(_sigmoid(eta), 1e-15, 1 - 1e-15)
            devs[i] = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        elif family == "poisson":
            mu = np.exp(np.clip(eta, -500, 500))
            devs[i] = _poisson_dev(y, mu)
    return devs


def _mn_null_deviance(y_codes, K):
    """Null deviance for multinomial (intercept-only softmax)."""
    n = len(y_codes)
    p0 = np.bincount(y_codes, minlength=K).astype(float) / n
    p0 = np.clip(p0, 1e-15, None)
    return -np.mean(np.log(p0[y_codes]))


def _mn_fit_deviances(y_codes, x, beta, intercept):
    """Per-lambda deviance for multinomial.

    beta: (nlambda, K, d), intercept: (nlambda, K)
    Returns (nlambda,) array.
    """
    nlambda = beta.shape[0]
    devs = np.zeros(nlambda)
    for i in range(nlambda):
        lp = x @ beta[i].T + intercept[i]  # (n, K)
        P = _softmax(lp)
        devs[i] = -np.mean(np.log(np.maximum(P[np.arange(len(y_codes)), y_codes], 1e-15)))
    return devs


class Solver:
    """
    The PICASSO Solver For GLM.

    :param x: An ``n*d`` design matrix.
    :param y: The *n* dimensional response vector.
    :param lambdas: Lambda path specification: tuple ``(n, lambda_min_ratio)``
            or explicit sequence (size > 2).
    :param family: ``"gaussian"``, ``"sqrtlasso"``, ``"binomial"``,
            ``"poisson"``, or ``"multinomial"``. Default ``"gaussian"``.
    :param penalty: ``"l1"``, ``"mcp"``, or ``"scad"``. Default ``"l1"``.
    :param gamma: Concavity parameter for MCP/SCAD. Default ``3``.
    :param useintercept: Include intercept term. Default ``True``.
    :param standardize: Standardize design matrix. Default ``True``.
    :param type_gaussian: ``"naive"`` or ``"covariance"`` (gaussian only).
    :param dfmax: Max nonzero coefficients for early stopping (``-1`` = no limit).
    :param prec: Stopping precision. Default ``1e-7``.
    :param max_ite: Iteration limit. Default ``1000``.
    :param offset: Per-observation offset vector (binomial/poisson only).
    :param verbose: Print tracing info. Default ``False``.
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
                 offset=None,
                 verbose=False):

        # Validate model
        if family not in ("gaussian", "binomial", "poisson", "sqrtlasso", "multinomial"):
            raise ValueError(
                'Invalid "family". Must be one of "gaussian", "binomial", '
                '"poisson", "sqrtlasso", "multinomial".'
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

        # Offset handling
        if offset is not None and family not in ("binomial", "poisson"):
            raise ValueError("offset is only supported for 'binomial' and 'poisson' families.")
        if offset is not None:
            self._offset = np.ascontiguousarray(offset, dtype='double')
            if self._offset.shape[0] != self.num_sample:
                raise ValueError("offset length must equal number of samples.")
        else:
            self._offset = np.zeros(self.num_sample, dtype='double')

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
        elif self.family == "multinomial":
            y_raw = np.asarray(y)
            uniq, y_codes = np.unique(y_raw, return_inverse=True)
            self._K = len(uniq)
            if self._K < 3:
                raise ValueError("multinomial requires >= 3 classes.")
            self._y_mn = y_codes.astype('double')
            self._y_codes = y_codes  # int array for dev computations
            self._mn_levels = uniq

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
            elif self.family == 'multinomial':
                K = self._K
                p0 = np.bincount(self._y_codes, minlength=K).astype(float) / n
                lambda_max = max(
                    np.max(np.abs(self.x.T @ ((self._y_codes == k).astype(float) - p0[k]))) / n
                    for k in range(K))
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

        # Register trainer and initialize result
        if self.family == "multinomial":
            K = self._K
            d = self.num_feature
            self.result = {
                'beta': np.zeros((self.nlambda, K, d), dtype='double'),
                'intercept': np.zeros((self.nlambda, K), dtype='double'),
                'ite_lamb': np.zeros(self.nlambda, dtype='int32'),
                'size_act': np.zeros(self.nlambda, dtype='int32'),
                'df': np.zeros(self.nlambda, dtype='int32'),
                'train_time': np.zeros(self.nlambda, dtype='double'),
                'num_fit': np.zeros(1, dtype='int32'),
                'total_train_time': 0,
                'state': 'not trained'
            }
            self.trainer = self._multinomial_wrapper()
        else:
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
            self.trainer = getattr(self, '_' + self.family + '_wrapper')()

    # ------------------------------------------------------------------
    # C interface decorators
    # ------------------------------------------------------------------

    def _decor_cinterface(self, _function):
        """Decorate a C API function (gaussian / sqrtlasso, no offset)."""
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
                      True)
            time_end = time.time()
            self.result['total_train_time'] = time_end - time_start
            self._finalize_result()
            self.result['nulldev'] = _null_deviance(self.y, self.family)
            devs = _fit_deviances(self.y, self._x_orig,
                                  self.result['beta'], self.result['intercept'],
                                  self.family)
            nd = self.result['nulldev']
            if nd is not None and nd > 0:
                self.result['dev_ratio'] = np.clip(1.0 - devs / nd, 0, 1)
            else:
                self.result['dev_ratio'] = np.zeros(self.nlambda)

        return wrapper

    def _decor_cinterface_glm(self, _function):
        """Decorate a C API function (binomial / poisson, with offset)."""
        CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
        _function.argtypes = [
            CDoubleArray, CDoubleArray, ctypes.c_int, ctypes.c_int, CDoubleArray,
            ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
            ctypes.c_int, ctypes.c_bool,
            ctypes.c_int,        # dfmax
            CDoubleArray,        # offset
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
                      self._offset,
                      self.result['beta'],
                      self.result['intercept'], self.result['ite_lamb'],
                      self.result['size_act'], self.result['train_time'],
                      self.result['num_fit'],
                      True)
            time_end = time.time()
            self.result['total_train_time'] = time_end - time_start
            self._finalize_result()
            off = self._offset if np.any(self._offset != 0) else None
            self.result['nulldev'] = _null_deviance(self.y, self.family, offset=off)
            devs = _fit_deviances(self.y, self._x_orig,
                                  self.result['beta'], self.result['intercept'],
                                  self.family, offset=off)
            nd = self.result['nulldev']
            if nd is not None and nd > 0:
                self.result['dev_ratio'] = np.clip(1.0 - devs / nd, 0, 1)
            else:
                self.result['dev_ratio'] = np.zeros(self.nlambda)

        return wrapper

    def _finalize_result(self):
        """Truncate to actual fit count, rescale, add df."""
        nfit = int(self.result['num_fit'][0])
        if 0 < nfit < self.nlambda:
            self.result['beta'] = self.result['beta'][:nfit]
            self.result['intercept'] = self.result['intercept'][:nfit]
            self.result['ite_lamb'] = self.result['ite_lamb'][:nfit]
            self.result['size_act'] = self.result['size_act'][:nfit]
            self.result['train_time'] = self.result['train_time'][:nfit]
            self.nlambda = nfit
            self.lambdas = self.lambdas[:nfit]

        if self.standardize:
            self.result['beta'], self.result['intercept'] = _rescale_solution(
                self.result['beta'], self.result['intercept'],
                self._xinvc, self._xm)

        if self.family == 'gaussian' and self._ym != 0.0:
            self.result['intercept'] = self.result['intercept'] + self._ym

        self.result['df'] = np.sum(self.result['beta'] != 0, axis=1).astype('int32')

    # ------------------------------------------------------------------
    # Family wrappers
    # ------------------------------------------------------------------

    def _gaussian_wrapper(self):
        if self.verbose:
            print("Sparse linear regression.")
            print(self.penalty.upper()
                  + " regularization via active set identification and coordinate descent.\n")
        if self.type_gaussian == "covariance":
            return self._decor_cinterface(_PICASSO_LIB.SolveLinearRegressionCovUpdate)
        return self._decor_cinterface(_PICASSO_LIB.SolveLinearRegressionNaiveUpdate)

    def _binomial_wrapper(self):
        if self.verbose:
            print("Sparse logistic regression.")
            print(self.penalty.upper()
                  + " regularization via active set identification and coordinate descent.\n")
        return self._decor_cinterface_glm(_PICASSO_LIB.SolveLogisticRegression)

    def _poisson_wrapper(self):
        if self.verbose:
            print("Sparse poisson regression.")
            print(self.penalty.upper()
                  + " regularization via active set identification and coordinate descent.\n")
        return self._decor_cinterface_glm(_PICASSO_LIB.SolvePoissonRegression)

    def _sqrtlasso_wrapper(self):
        if self.verbose:
            print("Sparse sqrt lasso regression.")
            print(self.penalty.upper()
                  + " regularization via active set identification and coordinate descent.\n")
        return self._decor_cinterface(_PICASSO_LIB.SolveSqrtLinearRegression)

    def _multinomial_wrapper(self):
        if self.verbose:
            print("Sparse multinomial regression.")
            print(self.penalty.upper()
                  + " regularization via coordinate descent.\n")

        K = self._K
        d = self.num_feature
        nlambda = self.nlambda

        CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
        func = _PICASSO_LIB.SolveMultinomialRegression
        func.argtypes = [
            CDoubleArray,        # Y_int (n,)
            CDoubleArray,        # X (n*d)
            ctypes.c_int,        # n
            ctypes.c_int,        # d
            ctypes.c_int,        # K
            CDoubleArray,        # lambda
            ctypes.c_int,        # nlambda
            ctypes.c_double,     # gamma
            ctypes.c_int,        # max_ite
            ctypes.c_double,     # prec
            ctypes.c_int,        # reg_type
            ctypes.c_bool,       # intercept
            ctypes.c_int,        # dfmax
            CDoubleArray,        # beta_out (d*K*nlambda)
            CDoubleArray,        # intcpt_out (K*nlambda)
            CIntArray,           # ite_lamb
            CIntArray,           # size_act
            CDoubleArray,        # runt
            CIntArray,           # num_fit
            ctypes.c_bool,       # usePython
        ]

        beta_flat = np.zeros(d * K * nlambda, dtype='double')
        intcpt_flat = np.zeros(K * nlambda, dtype='double')

        def wrapper():
            time_start = time.time()
            func(self._y_mn, self.x, self.num_sample, d, K,
                 self.lambdas, nlambda, self.gamma, self.max_ite, self.prec,
                 self.penaltyflag, self.use_intercept, self.dfmax,
                 beta_flat, intcpt_flat,
                 self.result['ite_lamb'], self.result['size_act'],
                 self.result['train_time'], self.result['num_fit'],
                 True)
            time_end = time.time()
            self.result['total_train_time'] = time_end - time_start

            nfit = int(self.result['num_fit'][0])
            actual_nl = nfit if (0 < nfit < nlambda) else nlambda

            # beta_flat layout: [lambda * K * d + class_k * d + feat_j]
            raw = beta_flat[:actual_nl * K * d].reshape(actual_nl, K, d)
            raw_intcpt = intcpt_flat[:actual_nl * K].reshape(actual_nl, K)

            if 0 < nfit < nlambda:
                self.result['ite_lamb'] = self.result['ite_lamb'][:nfit]
                self.result['size_act'] = self.result['size_act'][:nfit]
                self.result['train_time'] = self.result['train_time'][:nfit]
                self.nlambda = nfit
                self.lambdas = self.lambdas[:nfit]

            # Rescale: for each class k
            # beta[li, k, :] *= xinvc  => beta_rescaled[li, k, :]
            # intcpt[li, k]  -= beta_rescaled[li, k, :] @ xm
            if self.standardize:
                beta_r = raw * self._xinvc  # broadcasts (nl, K, d) * (d,)
                intcpt_r = raw_intcpt - (beta_r * self._xm).sum(axis=2)
            else:
                beta_r = raw
                intcpt_r = raw_intcpt

            self.result['beta'] = beta_r
            self.result['intercept'] = intcpt_r
            self.result['df'] = np.array(
                [np.sum(beta_r[i] != 0) for i in range(actual_nl)], dtype='int32')

            # Deviance
            null_dev = _mn_null_deviance(self._y_codes, K)
            self.result['nulldev'] = null_dev
            devs = _mn_fit_deviances(self._y_codes, self._x_orig,
                                     beta_r, intcpt_r)
            if null_dev > 0:
                self.result['dev_ratio'] = np.clip(1.0 - devs / null_dev, 0, 1)
            else:
                self.result['dev_ratio'] = np.zeros(actual_nl)

        return wrapper

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def train(self):
        """Train the model along the regularization path."""
        self.result['state'] = 'trained'
        self.trainer()
        if self.verbose:
            print('Training is over.')

    def coef(self):
        """Extract model coefficients dict."""
        if self.result['state'] == 'not trained':
            print('Warning: The model has not been trained yet!')
        return self.result

    def _resolve_lam(self, lam):
        """Find lambda index/interpolation for a given lambda value.

        Returns (beta, intercept) interpolated to the requested lambda.
        Prints a note if interpolation is needed.
        """
        lambdas = self.lambdas
        beta = self.result['beta']
        intercept = self.result['intercept']

        if lam >= lambdas[0]:
            return beta[0], intercept[0]
        if lam <= lambdas[-1]:
            return beta[-1], intercept[-1]

        # Find bracket
        idx = np.searchsorted(-lambdas, -lam)  # lambdas is decreasing
        idx = int(np.clip(idx, 1, len(lambdas) - 1))
        lo, hi = idx - 1, idx
        lam_lo, lam_hi = lambdas[lo], lambdas[hi]

        if abs(lam - lam_lo) < 1e-12 * max(1, abs(lam_lo)):
            return beta[lo], intercept[lo]
        if abs(lam - lam_hi) < 1e-12 * max(1, abs(lam_hi)):
            return beta[hi], intercept[hi]

        # Linear interpolation
        t = (lam - lam_lo) / (lam_hi - lam_lo)
        print(f"Note: lambda={lam:.6g} is between lambdas[{lo}]={lam_lo:.6g} "
              f"and lambdas[{hi}]={lam_hi:.6g}; results are linearly interpolated.")
        b_interp = (1 - t) * beta[lo] + t * beta[hi]
        i_interp = (1 - t) * intercept[lo] + t * intercept[hi]
        return b_interp, i_interp

    def predict(self, newdata=None, lambdidx=None, type="response", lam=None):
        """Predict responses for new data.

        :param newdata: Data matrix for prediction. Defaults to training data.
        :param lambdidx: Index into lambda path. Defaults to last lambda.
        :param type: ``"response"`` (default), ``"link"``, ``"class"``
                     (binomial/multinomial only), or ``"nonzero"``.
        :param lam: Lambda value (scalar). If provided, overrides ``lambdidx``
                    and linearly interpolates if needed.
        :return: Predicted values.
        """
        x_pred = self._x_orig if newdata is None else np.asarray(newdata, dtype='double')

        # Determine beta/intercept to use
        if lam is not None:
            _beta, _intercept = self._resolve_lam(lam)
        else:
            if lambdidx is None:
                lambdidx = self.nlambda - 1
            _beta = self.result['beta'][lambdidx]
            _intercept = self.result['intercept'][lambdidx]

        # nonzero: feature indices with nonzero coefficients
        if type == "nonzero":
            if self.family == "multinomial":
                return [list(np.where(np.abs(_beta[k]) > 1e-8)[0])
                        for k in range(self._K)]
            return np.where(np.abs(_beta) > 1e-8)[0]

        if self.family == "multinomial":
            lp = x_pred @ _beta.T + _intercept  # (n, K)
            if type == "link":
                return lp
            if type == "class":
                return np.argmax(lp, axis=1)
            # response: softmax
            return _softmax(lp)

        eta = x_pred @ _beta + _intercept

        if type == "link":
            return eta

        if self.family in ("gaussian", "sqrtlasso"):
            return eta  # same as link

        if self.family == "binomial":
            if type == "class":
                return (eta > 0).astype(int)
            return _sigmoid(eta)

        if self.family == "poisson":
            return np.exp(np.clip(eta, -500, 500))

        return eta

    def assess(self, newx=None, newy=None):
        """Compute evaluation metrics over the full lambda path.

        :param newx: Data matrix. Defaults to training data.
        :param newy: Response vector. Defaults to training response.
        :return: Dict with keys ``"lambda"``, ``"deviance"``, and family-specific metrics.
        """
        x = self._x_orig if newx is None else np.asarray(newx, dtype='double')
        if newy is None:
            y = self.y
        else:
            y = np.asarray(newy, dtype='double')

        result = {'lambda': self.lambdas}

        if self.family == "multinomial":
            y_codes = np.searchsorted(self._mn_levels, y).astype(int)
            devs = _mn_fit_deviances(y_codes, x,
                                     self.result['beta'], self.result['intercept'])
            result['deviance'] = devs
            preds = np.array([np.argmax(
                _softmax(x @ self.result['beta'][i].T + self.result['intercept'][i]),
                axis=1)
                for i in range(self.nlambda)])
            result['class_error'] = np.mean(preds != y_codes[np.newaxis, :], axis=1)
            return result

        beta = self.result['beta']       # (nlambda, d)
        intercept = self.result['intercept']  # (nlambda,)
        off = self._offset if np.any(self._offset != 0) else None
        devs = _fit_deviances(y, x, beta, intercept, self.family, offset=off)
        result['deviance'] = devs

        if self.family in ("gaussian", "sqrtlasso"):
            mse_arr = np.zeros(self.nlambda)
            mae_arr = np.zeros(self.nlambda)
            for i in range(self.nlambda):
                eta = x @ beta[i] + intercept[i]
                mse_arr[i] = np.mean((y - eta) ** 2)
                mae_arr[i] = np.mean(np.abs(y - eta))
            result['mse'] = mse_arr
            result['mae'] = mae_arr

        elif self.family == "binomial":
            cerr = np.zeros(self.nlambda)
            for i in range(self.nlambda):
                eta = x @ beta[i] + intercept[i]
                pred_class = (eta > 0).astype(int)
                cerr[i] = np.mean(pred_class != y.astype(int))
            result['class_error'] = cerr

        elif self.family == "poisson":
            mse_arr = np.zeros(self.nlambda)
            for i in range(self.nlambda):
                eta = x @ beta[i] + intercept[i]
                mu = np.exp(np.clip(eta, -500, 500))
                mse_arr[i] = np.mean((y - mu) ** 2)
            result['mse'] = mse_arr

        return result

    def confusion(self, newx, newy, lambdidx=None):
        """Compute confusion matrices for binomial family.

        :param newx: Data matrix.
        :param newy: True binary labels (0/1).
        :param lambdidx: List of lambda indices. Defaults to all lambdas.
        :return: List of 2x2 numpy arrays.
        """
        if self.family != "binomial":
            raise ValueError("confusion() is only supported for 'binomial' family.")
        x = np.asarray(newx, dtype='double')
        y = np.asarray(newy, dtype='double')
        if lambdidx is None:
            lambdidx = list(range(self.nlambda))
        matrices = []
        for i in lambdidx:
            eta = x @ self.result['beta'][i] + self.result['intercept'][i]
            pred = (eta > 0).astype(int)
            ytrue = y.astype(int)
            cm = np.zeros((2, 2), dtype=int)
            for t in range(2):
                for p in range(2):
                    cm[t, p] = int(np.sum((ytrue == t) & (pred == p)))
            matrices.append(cm)
        return matrices

    def cross_validate(self, nfolds=10, foldid=None, type_measure="default"):
        """K-fold cross-validation to select lambda.

        :param nfolds: Number of folds. Default 10.
        :param foldid: Optional array of fold assignments (0-indexed, length n).
        :param type_measure: ``"default"``, ``"deviance"``, ``"mse"``, ``"mae"``,
                             or ``"class"``.
        :return: Dict with ``"lambda"``, ``"cvm"``, ``"cvsd"``, ``"cvup"``,
                 ``"cvlo"``, ``"nzero"``, ``"lambda_min"``, ``"lambda_1se"``,
                 ``"foldid"``, ``"name"``.
        """
        n = self.num_sample
        if foldid is None:
            foldid = np.zeros(n, dtype=int)
            idx = np.random.permutation(n)
            for i, j in enumerate(idx):
                foldid[j] = i % nfolds
        else:
            foldid = np.asarray(foldid, dtype=int)
            nfolds = int(foldid.max()) + 1

        # Determine measure name
        if type_measure == "default":
            if self.family == "binomial":
                type_measure = "class"
            else:
                type_measure = "deviance"
        measure_name = type_measure

        losses = np.full((nfolds, self.nlambda), np.nan)

        for fold in range(nfolds):
            test_idx = np.where(foldid == fold)[0]
            train_idx = np.where(foldid != fold)[0]
            x_tr = self._x_orig[train_idx]
            y_tr = self.y[train_idx]
            x_te = self._x_orig[test_idx]
            y_te = self.y[test_idx]

            # Build fold solver with same lambdas
            kw = dict(
                lambdas=self.lambdas,
                family=self.family,
                penalty=self.penalty,
                gamma=self.gamma,
                useintercept=self.use_intercept,
                standardize=self.standardize,
                type_gaussian=self.type_gaussian,
                dfmax=self.dfmax,
                prec=self.prec,
                max_ite=self.max_ite,
                verbose=False,
            )
            if self.family in ("binomial", "poisson"):
                offset_tr = self._offset[train_idx] if np.any(self._offset != 0) else None
                kw['offset'] = offset_tr

            try:
                fold_solver = Solver(x_tr, y_tr, **kw)
                fold_solver.train()
            except Exception:
                continue

            fl = fold_solver.nlambda
            beta_f = fold_solver.result['beta']
            intcpt_f = fold_solver.result['intercept']

            for li in range(min(fl, self.nlambda)):
                if self.family == "multinomial":
                    y_codes_te = np.searchsorted(self._mn_levels, y_te).astype(int)
                    lp = x_te @ beta_f[li].T + intcpt_f[li]
                    P = _softmax(lp)
                    loss_val = -np.mean(np.log(np.maximum(
                        P[np.arange(len(y_codes_te)), y_codes_te], 1e-15)))
                else:
                    eta = x_te @ beta_f[li] + intcpt_f[li]
                    if type_measure == "mse":
                        if self.family == "poisson":
                            mu = np.exp(np.clip(eta, -500, 500))
                            loss_val = np.mean((y_te - mu) ** 2)
                        else:
                            loss_val = np.mean((y_te - eta) ** 2)
                    elif type_measure == "mae":
                        loss_val = np.mean(np.abs(y_te - eta))
                    elif type_measure == "class":
                        pred = (eta > 0).astype(int)
                        loss_val = np.mean(pred != y_te.astype(int))
                    else:  # deviance
                        if self.family in ("gaussian", "sqrtlasso"):
                            loss_val = np.mean((y_te - eta) ** 2) / 2.0
                        elif self.family == "binomial":
                            p = np.clip(_sigmoid(eta), 1e-15, 1 - 1e-15)
                            loss_val = -np.mean(y_te * np.log(p) + (1 - y_te) * np.log(1 - p))
                        elif self.family == "poisson":
                            mu = np.exp(np.clip(eta, -500, 500))
                            loss_val = _poisson_dev(y_te, mu)
                        else:
                            loss_val = np.mean((y_te - eta) ** 2) / 2.0
                losses[fold, li] = loss_val

        cvm = np.nanmean(losses, axis=0)
        cvsd = np.nanstd(losses, axis=0, ddof=1)
        cvsd = np.where(np.isnan(cvsd), 0.0, cvsd)

        best_idx = int(np.argmin(cvm))
        lambda_min = self.lambdas[best_idx]
        threshold = cvm[best_idx] + cvsd[best_idx]
        lse_candidates = np.where((cvm <= threshold) & (self.lambdas >= lambda_min))[0]
        lambda_1se = self.lambdas[lse_candidates[0]] if len(lse_candidates) > 0 else lambda_min

        nzero = self.result['df'] if self.result['state'] == 'trained' else np.zeros(self.nlambda, dtype=int)

        return {
            'lambda': self.lambdas,
            'cvm': cvm,
            'cvsd': cvsd,
            'cvup': cvm + cvsd,
            'cvlo': cvm - cvsd,
            'nzero': nzero,
            'lambda_min': lambda_min,
            'lambda_1se': lambda_1se,
            'foldid': foldid,
            'name': measure_name,
        }

    def plot(self, log_scale=True, max_features=None, ax=None):
        """Visualize the solution path.

        :param log_scale: Use log scale on x-axis (default True).
        :param max_features: Only plot top-N features by max |coef|.
        :param ax: Matplotlib axes to plot on (optional). If None, a new
                   figure is created and shown.
        """
        import matplotlib.pyplot as plt

        show = ax is None
        if show:
            fig, ax = plt.subplots()

        beta = self.result['beta']
        if self.family == "multinomial":
            # Sum coefficient magnitudes across classes for display
            beta_display = np.sum(np.abs(beta), axis=1)  # (nlambda, d)
        else:
            beta_display = beta  # (nlambda, d)

        if max_features is not None:
            importance = np.max(np.abs(beta_display), axis=0)
            top_idx = np.argsort(importance)[-max_features:]
            beta_display = beta_display[:, top_idx]

        x_axis = np.log(self.lambdas) if log_scale else self.lambdas
        ax.plot(x_axis, beta_display)
        ax.set_ylabel('Coefficient')
        ax.set_xlabel('log(lambda)' if log_scale else 'lambda')
        ax.set_title('Regularization Path')

        # Add df annotation on top axis
        df_vals = self.result.get('df', np.zeros(self.nlambda, dtype=int))
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_indices = np.linspace(0, self.nlambda - 1, min(6, self.nlambda), dtype=int)
        ax2.set_xticks([x_axis[i] for i in tick_indices])
        ax2.set_xticklabels([str(df_vals[i]) for i in tick_indices])
        ax2.set_xlabel('df')

        if show:
            plt.tight_layout()
            plt.show()

    def __str__(self):
        """Tabular summary of the model."""
        lines = [
            f"Model Type: {self.family:<12}  Penalty: {self.penalty}",
            f"n_samples: {self.num_sample}    n_features: {self.num_feature}"
            f"    nlambda: {self.nlambda}",
        ]
        if self.result['state'] == 'trained':
            lines.append("")
            has_dr = 'dev_ratio' in self.result
            if has_dr:
                lines.append(f"{'idx':>5}  {'lambda':>10}  {'df':>5}  {'dev_ratio':>10}")
            else:
                lines.append(f"{'idx':>5}  {'lambda':>10}  {'df':>5}")
            df_arr = self.result.get('df', np.zeros(self.nlambda, dtype=int))
            dr_arr = self.result.get('dev_ratio', None)
            indices = list(range(min(3, self.nlambda))) + \
                      list(range(max(3, self.nlambda - 2), self.nlambda))
            indices = sorted(set(indices))
            prev = -1
            for i in indices:
                if i > prev + 1:
                    lines.append("  ...")
                prev = i
                lam_str = f"{self.lambdas[i]:.2e}"
                df_str = str(int(df_arr[i]))
                if has_dr and dr_arr is not None:
                    dr_str = f"{dr_arr[i]:.3f}"
                    lines.append(f"{i+1:>5}  {lam_str:>10}  {df_str:>5}  {dr_str:>10}")
                else:
                    lines.append(f"{i+1:>5}  {lam_str:>10}  {df_str:>5}")
        return "\n".join(lines) + "\n"
