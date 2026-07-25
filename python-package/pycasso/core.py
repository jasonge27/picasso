# coding: utf-8
"""Public Python interface to the PICASSO native solvers."""

import os
import time
import math
import numbers
import warnings
import functools
import threading
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

from .libpath import find_lib_path

__all__ = ["Solver", "PycassoError"]


class PycassoError(Exception):
    """Error thrown by pycasso solver."""
    pass


_CTYPES_BIND_LOCK = threading.RLock()
_CTYPES_SIGNATURE_ATTR = '_pycasso_ctypes_signature'
_CTYPES_SIGNATURE_MISSING = object()


def _bind_ctypes_signature(function, argtypes, restype):
    """Bind one shared ctypes function exactly once and reject ABI drift."""
    signature = (tuple(argtypes), restype)
    with _CTYPES_BIND_LOCK:
        current = getattr(
            function, _CTYPES_SIGNATURE_ATTR, _CTYPES_SIGNATURE_MISSING)
        if current is not _CTYPES_SIGNATURE_MISSING:
            if current != signature:
                function_name = getattr(function, '__name__', repr(function))
                raise PycassoError(
                    f'Conflicting ctypes signature for {function_name}.')
            return function

        function.argtypes = list(signature[0])
        function.restype = restype
        setattr(function, _CTYPES_SIGNATURE_ATTR, signature)
    return function


def _serialize_solver_operation(method):
    """Serialize one buffer-mutating operation on a Solver instance."""
    @functools.wraps(method)
    def serialized(self, *args, **kwargs):
        with self._operation_lock:
            return method(self, *args, **kwargs)
    return serialized


def _load_lib():
    """Load picasso library."""
    lib_path = find_lib_path()
    if not lib_path:
        # Sphinx imports the module to extract signatures and docstrings but
        # never constructs a Solver.  libpath deliberately returns an empty
        # list in this documentation-only mode.
        if os.environ.get('PICASSO_BUILD_DOC'):
            return None
        raise PycassoError(
            "Can not find picasso Library. Please install pycasso correctly.")
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    return lib


# load the PICASSO library globally
_PICASSO_LIB = _load_lib()


_MULTINOMIAL_STATUS_NAMES = {
    0: 'completed',
    1: 'dfmax_reached',
    2: 'invalid_input',
    3: 'outer_iteration_limit',
    4: 'inner_iteration_limit',
    5: 'line_search_failed',
    6: 'no_descent_direction',
    7: 'numerical_failure',
    8: 'lla_majorization_failed',
    9: 'exception',
    10: 'lla_stationarity_limit',
    11: 'interrupted',
}


_SCALAR_LLA_STATUS_NAMES = {
    0: 'completed',
    1: 'dfmax_reached',
    2: 'invalid_input',
    3: 'subproblem_failed',
    4: 'inner_iteration_limit',
    5: 'line_search_failed',
    6: 'no_descent_direction',
    7: 'numerical_failure',
    8: 'lla_majorization_failed',
    9: 'exception',
    10: 'lla_stationarity_limit',
    11: 'interrupted',
}


# Bound the temporary linear-predictor block used when evaluating a scalar
# solution path.  Metrics consume one column at a time, so this is the only
# path-sized working array retained by the evaluator.
_SCALAR_PATH_BLOCK_BYTES = 8 * 1024 * 1024

# Cross-validation keeps independent fold solvers live, and folds may run in
# parallel.  Use a smaller predictor block there so batched path scoring stays
# within a modest per-worker budget while still using BLAS-3 operations.
_SCALAR_CV_PATH_BLOCK_BYTES = 1 * 1024 * 1024

# Bound the temporary used to adjust multinomial intercepts after native
# coefficients are rescaled in place.  The full coefficient path can be
# hundreds of MiB for wide multi-class problems.
_MULTINOMIAL_RESCALE_BLOCK_BYTES = 8 * 1024 * 1024

# User-facing precision presets.  PICASSO and glmnet use different stopping
# statistics, so the fast value is a benchmark-calibrated PICASSO tolerance,
# not glmnet's raw ``thresh`` argument.
_HIGH_PRECISION = 1e-7
_FAST_MODE_PRECISION = 1e-4
_FAST_MODE_POISSON_PRECISION = 4e-4

# Automatic Gaussian covariance updates are limited to an 8 MiB worst-case
# lazy Gram cache (1024^2 doubles). Explicit requests are never overridden.
_GAUSSIAN_AUTO_MAX_FEATURES = 1024
_GAUSSIAN_AUTO_MIN_LAMBDAS = 8
_GAUSSIAN_AUTO_SMALL_FEATURES = 160
_GAUSSIAN_AUTO_SPARSE_PATH_RATIO = 0.10
_GAUSSIAN_AUTO_DEFAULT_PATH_RATIO = 0.05
_GAUSSIAN_AUTO_MODERATE_N_OVER_D = 4
_GAUSSIAN_AUTO_VERY_TALL_N_OVER_D = 16


def _fast_precision(family):
    """Return the glmnet-like achieved-accuracy preset for a family."""
    # Gaussian already uses glmnet's scaled objective-change convention.
    # Newton/IRLS families use an approximately absolute KKT tolerance.
    if family == "gaussian":
        return _HIGH_PRECISION
    if family == "poisson":
        return _FAST_MODE_POISSON_PRECISION
    return _FAST_MODE_PRECISION


def _resolve_gaussian_type(type_gaussian, n, d, lambdas):
    """Resolve the public Gaussian backend without changing explicit modes."""
    if type_gaussian not in ("auto", "naive", "covariance"):
        raise ValueError(
            'Invalid "type_gaussian". Must be one of '
            '"auto", "naive", "covariance".')
    if type_gaussian != "auto":
        return type_gaussian
    nlambda = len(lambdas)
    if (d <= 0 or nlambda < _GAUSSIAN_AUTO_MIN_LAMBDAS or n < d or
            d > _GAUSSIAN_AUTO_MAX_FEATURES):
        return "naive"
    lambda_ratio = (float(lambdas[-1]) / float(lambdas[0])
                    if nlambda and lambdas[0] > 0 else 0.0)
    ratio_tolerance = 1e-12
    small_design = d <= _GAUSSIAN_AUTO_SMALL_FEATURES
    sparse_path = (
        lambda_ratio + ratio_tolerance >= _GAUSSIAN_AUTO_SPARSE_PATH_RATIO)
    moderately_tall_default_path = (
        lambda_ratio + ratio_tolerance >= _GAUSSIAN_AUTO_DEFAULT_PATH_RATIO
        and n >= _GAUSSIAN_AUTO_MODERATE_N_OVER_D * d)
    very_tall = n >= _GAUSSIAN_AUTO_VERY_TALL_N_OVER_D * d
    if (small_design or sparse_path or moderately_tall_default_path or
            very_tall):
        return "covariance"
    return "naive"


def _scaled_design(x, center):
    """Scale a finite design using the required output as its workspace."""
    n, d = x.shape
    xx = np.empty((n, d), dtype='double', order='C')
    xm = np.zeros(d, dtype='double')
    xinvc = np.zeros(d, dtype='double')

    # xx is the only n-by-d allocation required by the result. Use it first
    # for column maxima, then overwrite it with the scaled design.
    np.abs(x, out=xx, casting='unsafe')
    maximum = np.max(xx, axis=0)
    safe_maximum = maximum.copy()
    safe_maximum[safe_maximum == 0.0] = 1.0
    np.divide(x, safe_maximum, out=xx, casting='unsafe')

    if center:
        scaled_mean = np.mean(xx, axis=0)
        # Scaled finite entries are in [-1, 1]. Guard the final multiplication
        # against a last-bit reduction overshoot near DBL_MAX.
        np.clip(scaled_mean, -1.0, 1.0, out=scaled_mean)
        xm[:] = maximum * scaled_mean
        np.subtract(xx, scaled_mean, out=xx)

    divisor = max(n - 1, 1)
    scaled_norm = np.sqrt(
        np.einsum('ij,ij->j', xx, xx, optimize=False) / divisor)
    scalable = (maximum > 0.0) & (scaled_norm > 0.0)
    safe_norm = scaled_norm.copy()
    safe_norm[~scalable] = 1.0
    np.divide(xx, safe_norm, out=xx)
    if np.any(~scalable):
        xx[:, ~scalable] = 0.0

    xinvc[scalable] = (
        (1.0 / maximum[scalable]) / scaled_norm[scalable])

    return xx, xm, xinvc


def _standardize(x):
    """Standardize design matrix: center and scale each column."""
    return _scaled_design(x, center=True)


def _scale_without_centering(x):
    """Scale columns for a no-intercept model without changing its origin."""
    return _scaled_design(x, center=False)


def _compact_and_rescale_scalar_solution(beta, intercept, nfit,
                                         standardize, use_intercept,
                                         xinvc, xm):
    """Return committed scalar outputs without a second full coefficient path.

    Full paths retain the native output owners and are rescaled in place.
    Truncated standardized paths fuse prefix compaction with scaling in one
    allocation; unstandardized prefixes receive ordinary compact copies.
    """
    if nfit < beta.shape[0]:
        intercept = intercept[:nfit].copy()
        if standardize:
            beta = np.multiply(beta[:nfit], xinvc)
        else:
            beta = beta[:nfit].copy()
    elif standardize:
        np.multiply(beta, xinvc, out=beta)

    if standardize and use_intercept:
        np.subtract(intercept, beta @ xm, out=intercept)
    return beta, intercept


def _compact_prefix(values, size):
    """Return ``values`` unchanged when full, otherwise an owning prefix."""
    if values.shape[0] == size:
        return values
    return values[:size].copy()


def _rescale_multinomial_solution_in_place(beta, intercept, xinvc, xm):
    """Rescale one multinomial path with bounded temporary storage."""
    np.multiply(beta, xinvc, out=beta)
    if beta.size == 0:
        return beta, intercept

    feature_count = beta.shape[-1]
    beta_rows = beta.reshape(-1, feature_count)
    intercept_rows = intercept.reshape(-1)
    bytes_per_row = feature_count * np.dtype('double').itemsize
    rows_per_block = max(
        1, _MULTINOMIAL_RESCALE_BLOCK_BYTES // max(bytes_per_row, 1))
    rows_per_block = min(rows_per_block, beta_rows.shape[0])
    for start in range(0, beta_rows.shape[0], rows_per_block):
        stop = min(start + rows_per_block, beta_rows.shape[0])
        adjustment = beta_rows[start:stop] @ xm
        np.subtract(intercept_rows[start:stop], adjustment,
                    out=intercept_rows[start:stop])
    return beta, intercept


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _reject_masked_array(values, name):
    """Reject masked arrays before NumPy can silently discard their mask."""
    if np.ma.isMaskedArray(values):
        raise ValueError(f'"{name}" must not be a masked array.')


def _real_numeric_array(values, name, copy=False, order='K'):
    """Return a real numeric array without parsing or lossy coercion."""
    _reject_masked_array(values, name)
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'"{name}" must be a real numeric array.') from exc
    if array.dtype.kind not in ('i', 'u', 'f'):
        raise ValueError(f'"{name}" must be a real numeric array.')
    if copy:
        return np.array(array, dtype='double', copy=True, order=order)
    return np.asarray(array, dtype='double')


def _softmax(lp):
    """lp: (n, K) array. Returns (n, K) probabilities."""
    _reject_masked_array(lp, 'logits')
    lp = np.asarray(lp, dtype='double')
    if lp.ndim != 2 or not np.all(np.isfinite(lp)):
        raise ValueError('Multinomial logits must be a finite matrix.')
    with np.errstate(over='ignore', invalid='ignore'):
        lp_shifted = lp - lp.max(axis=1, keepdims=True)
        ep = np.exp(lp_shifted)
        probabilities = ep / ep.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('Multinomial probabilities are not finite.')
    return probabilities


def _binomial_nll_from_eta(y, eta):
    """Return mean binomial negative log-likelihood on the link scale."""
    _reject_masked_array(y, 'y')
    _reject_masked_array(eta, 'eta')
    response = np.asarray(y, dtype='double')
    linear_predictor = np.asarray(eta, dtype='double')
    if (response.shape != linear_predictor.shape or
            not np.all(np.isfinite(response)) or
            not np.all(np.isfinite(linear_predictor))):
        raise ValueError(
            'Binomial response and linear predictor must be finite and have '
            'matching shapes.')
    loss = np.mean(
        np.logaddexp(0.0, linear_predictor) -
        response * linear_predictor)
    if not np.isfinite(loss):
        raise ValueError('Binomial negative log-likelihood is not finite.')
    return float(loss)


def _multinomial_nll_from_logits(y_codes, logits):
    """Return mean multinomial NLL without forming clipped probabilities."""
    _reject_masked_array(y_codes, 'y_codes')
    _reject_masked_array(logits, 'logits')
    try:
        numeric_codes = np.asarray(y_codes, dtype='double')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            'Multinomial label codes must be finite integers.') from exc
    scores = np.asarray(logits, dtype='double')
    if (scores.ndim != 2 or numeric_codes.ndim != 1 or
            scores.shape[0] != numeric_codes.size or
            not np.all(np.isfinite(scores))):
        raise ValueError(
            'Multinomial labels and logits must be finite and have matching '
            'rows.')
    if (not np.all(np.isfinite(numeric_codes)) or
            not np.all(numeric_codes == np.floor(numeric_codes))):
        raise ValueError('Multinomial label codes must be finite integers.')
    if (np.any(numeric_codes < 0) or
            np.any(numeric_codes >= scores.shape[1])):
        raise ValueError('Multinomial label code is outside the class range.')
    codes = numeric_codes.astype(np.intp, copy=False)
    row_max = np.max(scores, axis=1)
    shifted = scores - row_max[:, np.newaxis]
    loss = np.mean(
        np.log(np.sum(np.exp(shifted), axis=1)) +
        (row_max - scores[np.arange(codes.size), codes]))
    if not np.isfinite(loss):
        raise ValueError('Multinomial negative log-likelihood is not finite.')
    return float(loss)


def _response_vector(values, name, dtype=None):
    """Return a contiguous one-dimensional response vector."""
    _reject_masked_array(values, name)
    if dtype is not None and np.dtype(dtype) == np.dtype('double'):
        array = _real_numeric_array(values, name)
    else:
        try:
            array = np.asarray(values, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f'"{name}" must be a one-dimensional vector.') from exc
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f'"{name}" must be a one-dimensional vector.')
    return np.ascontiguousarray(array)


def _finite_real_scalar(value, name):
    """Validate and return one finite real scalar."""
    _reject_masked_array(value, name)
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f'"{name}" must be a finite numeric scalar.')
    scalar = array.item()
    if (isinstance(scalar, (bool, np.bool_)) or
            not isinstance(scalar, numbers.Real)):
        raise ValueError(f'"{name}" must be a finite numeric scalar.')
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'"{name}" must be a finite numeric scalar.') from exc
    if not np.isfinite(result):
        raise ValueError(f'"{name}" must be a finite numeric scalar.')
    return result


def _boolean_scalar(value, name):
    """Validate and return one strict boolean scalar."""
    _reject_masked_array(value, name)
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f'"{name}" must be True or False.')
    scalar = array.item()
    if not isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f'"{name}" must be True or False.')
    return bool(scalar)


def _positive_integer(value, name):
    """Validate a positive integer representable by the native C API."""
    result = _finite_real_scalar(value, name)
    if result <= 0 or result != math.floor(result):
        raise ValueError(f'"{name}" must be a positive integer.')
    if result > np.iinfo(np.int32).max:
        raise ValueError(f'"{name}" exceeds the native integer limit.')
    return int(result)


def _path_index(value, size, name='lambdidx'):
    """Validate one zero-based solution-path index."""
    result = _finite_real_scalar(value, name)
    if result != math.floor(result):
        raise ValueError(f'"{name}" must be an integer.')
    index = int(result)
    if index < 0 or index >= size:
        raise ValueError(
            f'"{name}" must be between 0 and {size - 1}.')
    return index


def _native_count_within_limit(value, name):
    """Reject dimensions/counts that the C ABI cannot represent."""
    if value > np.iinfo(np.int32).max:
        raise ValueError(
            f'"{name}" exceeds the native 32-bit indexing limit.')


def _validate_native_fit_count(value, requested, family_label,
                               zero_message=None):
    """Validate the committed fit count returned through the native ABI."""
    nfit = int(value)
    if 1 <= nfit <= requested:
        return nfit
    if nfit == 0 and zero_message is not None:
        raise PycassoError(zero_message)
    raise PycassoError(
        f'{family_label} solver returned an invalid fit count '
        f'(num_fit={nfit}, requested={requested}).')


def _validate_multinomial_native_counts(n, d, K=None, nlambda=None):
    """Validate every flattened count checked by the native V2 solver."""
    _native_count_within_limit(n, 'number of samples')
    _native_count_within_limit(d, 'number of features')
    _native_count_within_limit(n * d, 'n*d design size')
    if K is None:
        return
    _native_count_within_limit(K, 'number of classes')
    _native_count_within_limit(n * K, 'n*K probability size')
    coefficient_count = d * K
    _native_count_within_limit(
        coefficient_count, 'd*K coefficient size')
    if nlambda is None:
        return
    _native_count_within_limit(nlambda, 'lambda path length')
    _native_count_within_limit(
        coefficient_count * nlambda, 'd*K*nlambda output size')
    _native_count_within_limit(
        K * nlambda, 'K*nlambda intercept output size')


def _label_key(value):
    """Convert a numpy scalar label to its hashable Python counterpart."""
    return value.item() if isinstance(value, np.generic) else value


def _categorical_label_array(values, name):
    """Preserve heterogeneous sequence scalars before NumPy string coercion."""
    _reject_masked_array(values, name)
    try:
        labels = np.asarray(values)
        # A plain mixed Python sequence such as ["no", np.nan] is otherwise
        # silently converted to ["no", "nan"]. Re-read only inferred string
        # sequences as objects; homogeneous numeric lists keep their fast
        # vectorized path, and an explicit NumPy string array remains a string
        # array because its original scalar types are already unavailable.
        if (not isinstance(values, np.ndarray) and
                labels.dtype.kind in ('U', 'S')):
            labels = np.asarray(values, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f'"{name}" must be a one-dimensional categorical label vector.') \
            from exc
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError(
            f'"{name}" must be a one-dimensional categorical label vector.')
    return labels


def _binomial_label_vector(values, name='y'):
    """Validate one homogeneous numeric, string, or Boolean label vector."""
    labels = _categorical_label_array(values, name)
    if labels.size == 0:
        raise ValueError(f'"{name}" must contain at least one label.')

    kind = labels.dtype.kind
    if kind in ('i', 'u', 'f'):
        if kind == 'f' and not np.all(np.isfinite(labels)):
            raise ValueError(
                f'"{name}" numeric labels must be finite and non-missing.')
        category = 'numeric'
    elif kind == 'U':
        category = 'string'
    elif kind == 'b':
        category = 'boolean'
    elif kind == 'O':
        category = None
        for value in labels:
            scalar = _label_key(value)
            if scalar is None:
                raise ValueError(f'"{name}" contains missing labels.')
            if isinstance(scalar, (bool, np.bool_)):
                scalar_category = 'boolean'
            elif isinstance(
                    scalar, (numbers.Real, np.integer, np.floating)):
                try:
                    is_finite = bool(np.isfinite(scalar))
                except TypeError:
                    is_finite = isinstance(scalar, numbers.Integral)
                if not is_finite:
                    raise ValueError(
                        f'"{name}" numeric labels must be finite and '
                        'non-missing.')
                scalar_category = 'numeric'
            elif isinstance(scalar, (str, np.str_)):
                scalar_category = 'string'
            else:
                raise ValueError(
                    f'"{name}" labels must be finite numbers, strings, or '
                    'Booleans.')
            if category is None:
                category = scalar_category
            elif category != scalar_category:
                raise ValueError(
                    f'"{name}" must not mix numeric, string, and Boolean '
                    'labels.')
    else:
        raise ValueError(
            f'"{name}" labels must be finite numbers, strings, or Booleans.')
    return labels, category


def _encode_binomial_labels(values, levels=None, name='y'):
    """Encode two categorical levels as native double zero/one values."""
    labels, category = _binomial_label_vector(values, name)
    if levels is None:
        # Preserve and accelerate the long-standing 0/1 path. Integer and
        # Boolean bounds are sufficient; floating input also needs an exact
        # membership check to exclude interior values such as 0.5.
        kind = labels.dtype.kind
        is_encoded = False
        if kind in ('i', 'u', 'b'):
            is_encoded = bool(np.min(labels) == 0 and np.max(labels) == 1)
        elif kind == 'f':
            is_encoded = bool(
                np.min(labels) == 0 and np.max(labels) == 1 and
                np.all((labels == 0) | (labels == 1)))
        if is_encoded:
            discovered = np.asarray([0, 1], dtype=labels.dtype)
            codes = np.array(
                labels, dtype='double', copy=True, order='C')
            return discovered, codes

        try:
            discovered = np.unique(labels)
        except TypeError as exc:
            raise ValueError(
                f'"{name}" contains labels that cannot be ordered.') from exc
        if discovered.size != 2:
            raise ValueError(
                f'"{name}" must contain exactly two observed levels; found '
                f'{discovered.size}.')
        discovered = np.asarray(discovered).copy()
        codes = np.ascontiguousarray(
            np.equal(labels, discovered[1]), dtype='double')
        return discovered, codes

    fixed_levels, level_category = _binomial_label_vector(
        levels, 'binomial levels')
    if fixed_levels.size != 2:
        raise ValueError('The fitted binomial model has no two-level map.')

    matched = None
    if category == level_category:
        second = np.equal(labels, fixed_levels[1])
        matched = np.equal(labels, fixed_levels[0])
        np.logical_or(matched, second, out=matched)
        if np.all(matched):
            return fixed_levels, np.ascontiguousarray(second, dtype='double')

    # Match R's compatibility convention: original labels take precedence,
    # but a vector containing only finite numeric 0/1 is also accepted as an
    # already encoded response when any original-label match is missing.
    if category == 'numeric':
        try:
            numeric = np.asarray(labels, dtype='double')
        except (TypeError, ValueError, OverflowError):
            numeric = None
        if (numeric is not None and np.all(np.isfinite(numeric)) and
                np.all((numeric == 0) | (numeric == 1))):
            return fixed_levels, np.ascontiguousarray(numeric)

    if matched is None:
        unseen_values = labels[:3]
    else:
        unseen_values = labels[np.logical_not(matched)][:3]
    preview = ', '.join(repr(_label_key(value)) for value in unseen_values)
    raise ValueError(
        f'"{name}" contains unseen binomial label(s): {preview}.')


def _multinomial_label_vector(values, name='y'):
    """Validate numeric/string categorical labels without changing them."""
    labels = _categorical_label_array(values, name)
    if labels.size == 0:
        raise ValueError(f'"{name}" must contain at least one label.')

    for value in labels:
        scalar = _label_key(value)
        if scalar is None:
            raise ValueError(f'"{name}" contains missing labels.')
        if isinstance(scalar, (numbers.Real, np.integer, np.floating)):
            try:
                is_finite = bool(np.isfinite(scalar))
            except TypeError:
                # Arbitrarily large Python integers are still valid finite
                # category labels even when NumPy cannot safely coerce them.
                is_finite = isinstance(scalar, numbers.Integral)
            if not is_finite:
                raise ValueError(
                    f'"{name}" numeric labels must be finite and non-missing.')
        elif not isinstance(scalar, (str, np.str_)):
            raise ValueError(
                f'"{name}" labels must be finite numbers or strings.')
    return labels


def _encode_multinomial_labels(values, levels=None, name='y'):
    """Encode labels using either newly discovered or fixed class levels."""
    labels = _multinomial_label_vector(values, name)
    if levels is None:
        try:
            discovered, codes = np.unique(labels, return_inverse=True)
            return labels, discovered, codes.astype(int, copy=False)
        except TypeError:
            # Mixed numeric/string object arrays are valid categorical data but
            # are not orderable on recent NumPy versions. Preserve first use.
            discovered_values = []
            level_index = {}
            codes = np.empty(labels.size, dtype=int)
            for index, value in enumerate(labels):
                key = _label_key(value)
                if key not in level_index:
                    level_index[key] = len(discovered_values)
                    discovered_values.append(key)
                codes[index] = level_index[key]
            return labels, np.asarray(discovered_values, dtype=object), codes

    fixed_levels = np.asarray(levels)
    level_index = {
        _label_key(value): index for index, value in enumerate(fixed_levels)
    }
    codes = np.empty(labels.size, dtype=int)
    unseen = []
    for index, value in enumerate(labels):
        key = _label_key(value)
        if key not in level_index:
            unseen.append(key)
        else:
            codes[index] = level_index[key]
    if unseen:
        preview = ', '.join(repr(value) for value in unseen[:3])
        raise ValueError(
            f'"{name}" contains unseen multinomial label(s): {preview}.')
    return labels, fixed_levels, codes


def _poisson_dev(y, mu):
    """Poisson deviance D = 2*mean(y*log(y/mu) - (y-mu)), always >= 0.
    Convention: 0*log(0) = 0.
    """
    mu = np.maximum(mu, 1e-15)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(y > 0, np.log(y / mu), 0.0)
    term = np.where(y > 0, y * log_ratio - (y - mu), mu - y)
    return 2.0 * np.mean(term)


def _poisson_mean(eta):
    """Apply the Poisson inverse link without silently changing the model."""
    eta = np.asarray(eta, dtype='double')
    with np.errstate(over='ignore', invalid='ignore'):
        mu = np.exp(eta)
    if not np.all(np.isfinite(mu)):
        raise ValueError(
            'Poisson linear predictor is too large for a finite response mean.')
    return mu


def _poisson_deviance_from_eta(y, eta, mean=None):
    """Return mean Poisson deviance without clipping an underflowed mean."""
    _reject_masked_array(y, 'y')
    _reject_masked_array(eta, 'eta')
    response = np.asarray(y, dtype='double')
    linear_predictor = np.asarray(eta, dtype='double')
    if (response.shape != linear_predictor.shape or
            not np.all(np.isfinite(response)) or np.any(response < 0) or
            not np.all(np.isfinite(linear_predictor))):
        raise ValueError(
            'Poisson response and linear predictor must be finite, '
            'nonnegative where applicable, and have matching shapes.')
    if mean is None:
        fitted_mean = _poisson_mean(linear_predictor)
    else:
        _reject_masked_array(mean, 'mean')
        fitted_mean = np.asarray(mean, dtype='double')
        if (fitted_mean.shape != response.shape or
                not np.all(np.isfinite(fitted_mean)) or
                np.any(fitted_mean < 0)):
            raise ValueError(
                'Poisson fitted mean must be finite, nonnegative, and match '
                'the response shape.')
    log_response = np.zeros_like(response)
    positive = response > 0
    log_response[positive] = np.log(response[positive])
    terms = fitted_mean.copy()
    terms[positive] = (
        response[positive] *
        (log_response[positive] - linear_predictor[positive]) -
        response[positive] + fitted_mean[positive])
    deviance = 2.0 * np.mean(terms)
    if not np.isfinite(deviance):
        raise ValueError('Poisson deviance is not finite.')
    return float(max(deviance, 0.0))


def _scalar_null_linear_predictor(y, family, offset=None,
                                  include_intercept=True):
    """Return the scalar-family null-model linear predictor.

    The no-intercept null model fixes the fitted intercept at zero.  With an
    intercept, Poisson has a closed form and binomial uses a monotone bisection
    solve so an observation offset is handled exactly rather than discarded.
    """
    y = np.asarray(y, dtype='double')
    n = len(y)
    off = (np.asarray(offset, dtype='double') if offset is not None
           else np.zeros(n, dtype='double'))
    if family in ("gaussian", "sqrtlasso"):
        intercept = float(np.mean(y)) if include_intercept else 0.0
        return np.full(n, intercept, dtype='double')
    if not include_intercept:
        return off.copy()
    if family == "poisson":
        mean_y = float(np.mean(y))
        maximum_offset = float(np.max(off))
        scaled_mean = float(np.mean(np.exp(off - maximum_offset)))
        log_mean_exp_offset = maximum_offset + math.log(scaled_mean)
        intercept = (math.log(mean_y) - log_mean_exp_offset
                     if mean_y > 0.0 else 0.0)
        return off + intercept
    if family == "binomial":
        target = float(np.mean(y))
        # The public binomial validation guarantees 0 < target < 1.  These
        # bounds force every fitted probability below/above the target even
        # when offsets are widely separated.
        lower = -float(np.max(off)) - 40.0
        upper = -float(np.min(off)) + 40.0
        for _ in range(80):
            midpoint = lower + 0.5 * (upper - lower)
            if float(np.mean(_sigmoid(off + midpoint))) < target:
                lower = midpoint
            else:
                upper = midpoint
        return off + lower + 0.5 * (upper - lower)
    raise ValueError(f'Unsupported scalar family {family!r}.')


def _null_deviance(y, family, offset=None, include_intercept=True):
    """Compute the matching intercept or no-intercept null deviance."""
    eta0 = _scalar_null_linear_predictor(
        y, family, offset=offset, include_intercept=include_intercept)
    if family in ("gaussian", "sqrtlasso"):
        return np.mean((y - eta0) ** 2) / 2.0
    elif family == "binomial":
        return _binomial_nll_from_eta(y, eta0)
    elif family == "poisson":
        return _poisson_deviance_from_eta(y, eta0)
    elif family == "multinomial":
        return None  # handled separately
    return None


def _scalar_linear_predictor_blocks(x, beta, intercept, offset=None,
                                    block_bytes=None):
    """Yield memory-bounded ``(start, stop, eta)`` path blocks."""
    sample_count = x.shape[0]
    path_size = beta.shape[0]
    if path_size == 0:
        return
    if block_bytes is None:
        block_bytes = _SCALAR_PATH_BLOCK_BYTES
    bytes_per_column = max(sample_count, 1) * np.dtype('double').itemsize
    block_size = max(1, block_bytes // bytes_per_column)
    block_size = min(path_size, block_size)
    offset_column = (None if offset is None else
                     np.asarray(offset, dtype='double')[np.newaxis, :])

    for start in range(0, path_size, block_size):
        stop = min(start + block_size, path_size)
        if stop - start == 1:
            # Retain the BLAS-2 path when a block contains only one model.
            eta = (x @ beta[start]).reshape(1, sample_count)
        else:
            eta = beta[start:stop] @ x.T
        eta += intercept[start:stop, np.newaxis]
        if offset_column is not None:
            eta += offset_column
        yield start, stop, eta


def _scalar_cv_fold_losses(y, x, beta, intercept, family, measure,
                           offset=None, path_size=None,
                           block_bytes=_SCALAR_CV_PATH_BLOCK_BYTES):
    """Score one scalar CV fold with only the requested metric."""
    if path_size is None:
        path_size = beta.shape[0]
    path_size = min(path_size, beta.shape[0])
    losses = np.empty(path_size, dtype='double')
    integer_response = (y.astype(int) if family == "binomial" and
                        measure == "class" else None)

    # Class prediction has a strict eta > 0 boundary. BLAS-3 path batching can
    # round an exact BLAS-2 zero to either side of that boundary, so preserve
    # the established one-model-at-a-time arithmetic for this discrete metric.
    if integer_response is not None:
        for path_index in range(path_size):
            eta = x @ beta[path_index] + intercept[path_index]
            if offset is not None:
                eta = eta + offset
            losses[path_index] = np.mean(
                (eta > 0).astype(int) != integer_response)
        return losses

    for start, stop, eta_block in _scalar_linear_predictor_blocks(
            x, beta[:path_size], intercept[:path_size], offset=offset,
            block_bytes=block_bytes):
        for local_index, path_index in enumerate(range(start, stop)):
            eta = eta_block[local_index]
            if family in ("gaussian", "sqrtlasso"):
                residual = y - eta
                if measure == "mae":
                    loss = np.mean(np.abs(residual))
                else:
                    mse = np.mean(residual ** 2)
                    loss = mse if measure == "mse" else mse / 2.0
            elif family == "binomial":
                if measure == "deviance":
                    loss = _binomial_nll_from_eta(y, eta)
                else:
                    residual = y - _sigmoid(eta)
                    loss = (np.mean(residual ** 2) if measure == "mse"
                            else np.mean(np.abs(residual)))
            elif family == "poisson":
                fitted_mean = _poisson_mean(eta)
                if measure == "deviance":
                    loss = _poisson_deviance_from_eta(
                        y, eta, mean=fitted_mean)
                else:
                    residual = y - fitted_mean
                    loss = (np.mean(residual ** 2) if measure == "mse"
                            else np.mean(np.abs(residual)))
            losses[path_index] = loss
    return losses


def _scalar_path_metrics(y, x, beta, intercept, family, offset=None,
                         include_assessment=False):
    """Evaluate a scalar path while reusing each predictor block."""
    path_size = beta.shape[0]
    metrics = {'deviance': np.zeros(path_size)}
    integer_response = (y.astype(int) if include_assessment and
                        family == "binomial" else None)
    if include_assessment:
        if family in ("gaussian", "sqrtlasso"):
            metrics['mse'] = np.zeros(path_size)
            metrics['mae'] = np.zeros(path_size)
        elif family == "binomial":
            metrics['class_error'] = np.zeros(path_size)
        elif family == "poisson":
            metrics['mse'] = np.zeros(path_size)

    for start, stop, eta_block in _scalar_linear_predictor_blocks(
            x, beta, intercept, offset=offset):
        for local_index, path_index in enumerate(range(start, stop)):
            eta = eta_block[local_index]
            if family in ("gaussian", "sqrtlasso"):
                residual = y - eta
                mse = np.mean(residual ** 2)
                metrics['deviance'][path_index] = mse / 2.0
                if include_assessment:
                    metrics['mse'][path_index] = mse
                    metrics['mae'][path_index] = np.mean(np.abs(residual))
            elif family == "binomial":
                metrics['deviance'][path_index] = \
                    _binomial_nll_from_eta(y, eta)
                if include_assessment:
                    metrics['class_error'][path_index] = np.mean(
                        (eta > 0).astype(int) != integer_response)
            elif family == "poisson":
                mean = _poisson_mean(eta)
                metrics['deviance'][path_index] = \
                    _poisson_deviance_from_eta(y, eta, mean=mean)
                if include_assessment:
                    metrics['mse'][path_index] = np.mean((y - mean) ** 2)
    return metrics


def _fit_deviances(y, x, beta, intercept, family, offset=None):
    """Compute per-lambda deviance (NLL) for non-multinomial families.

    beta: (nlambda, d), intercept: (nlambda,)
    offset: (n,) optional per-observation offset (binomial/poisson only)
    Returns (nlambda,) array.
    """
    return _scalar_path_metrics(
        y, x, beta, intercept, family, offset=offset)['deviance']


def _native_scalar_deviances(y, family, smooth_objective):
    """Convert native smooth objectives to public deviance conventions."""
    smooth = np.asarray(smooth_objective, dtype='double')
    if family == 'gaussian':
        return 0.5 * smooth
    if family == 'sqrtlasso':
        return 0.5 * smooth ** 2
    if family == 'binomial':
        return smooth
    if family == 'poisson':
        response = np.asarray(y, dtype='double')
        positive = response > 0
        saturated_terms = np.zeros_like(response)
        saturated_terms[positive] = (
            response[positive] * np.log(response[positive]) -
            response[positive])
        saturated_constant = float(np.mean(saturated_terms))
        return np.maximum(0.0, 2.0 * (smooth + saturated_constant))
    raise ValueError(
        f'Native smooth objective is unavailable for family {family!r}.')


def _mn_null_deviance(y_codes, K, include_intercept=True):
    """Null multinomial NLL for an intercept or no-intercept model."""
    n = len(y_codes)
    if include_intercept:
        p0 = np.bincount(y_codes, minlength=K).astype(float) / n
    else:
        p0 = np.full(K, 1.0 / K, dtype='double')
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
        devs[i] = _multinomial_nll_from_logits(y_codes, lp)
    return devs


def _mn_assessment_logits(x, beta, intercept):
    """Form one multinomial linear-predictor matrix."""
    return x @ beta.T + intercept


def _mn_assessment_metrics(y_codes, x, beta, intercept):
    """Compute multinomial path metrics from one logits matrix per lambda."""
    nlambda = beta.shape[0]
    deviances = np.zeros(nlambda)
    class_errors = np.zeros(nlambda)
    for i in range(nlambda):
        logits = _mn_assessment_logits(x, beta[i], intercept[i])
        deviances[i] = _multinomial_nll_from_logits(y_codes, logits)
        # Softmax is monotone in each finite logit. Classify directly on the
        # link scale to avoid probability rounding and an n-by-K temporary.
        predictions = np.argmax(logits, axis=1)
        class_errors[i] = np.mean(predictions != y_codes)
    return deviances, class_errors


def _plot_lambda_values(lambdas, log_scale):
    """Return a finite path axis, falling back to lambda when zero is fit."""
    values = np.asarray(lambdas, dtype='double')
    if log_scale and np.all(values > 0):
        return np.log(values), 'log(lambda)'
    return values, 'lambda'


class Solver:
    """Fit a sparse regularization path with PICASSO.

    :param x: Finite design matrix with shape
        ``(n_samples, n_features)``.
    :param y: Response vector. Gaussian and square-root-lasso require finite
        numeric values; binomial accepts exactly two observed numeric, string,
        or Boolean levels; Poisson requires nonnegative integer counts with at
        least one positive count; multinomial accepts numeric or string labels
        with at least three observed classes.
    :param lambdas: Path specification. A two-element non-NumPy sequence is
        interpreted as ``(count, lambda_min_ratio)``. Every NumPy array,
        including arrays of length one or two, is an explicit path. Other
        explicit sequences therefore need a length other than two. Explicit
        paths must be nonempty, finite, nonnegative, and strictly decreasing.
        Default ``(100, 0.05)``.
    :param family: One of ``"gaussian"``, ``"binomial"``, ``"poisson"``,
        ``"sqrtlasso"``, or ``"multinomial"``. Default ``"gaussian"``.
    :param penalty: One of ``"l1"``, ``"mcp"``, or ``"scad"``.
        Default ``"l1"``.
    :param gamma: Concavity parameter. MCP requires ``gamma > 1`` and
        SCAD requires ``gamma > 2``. Default ``3``.
    :param useintercept: Include an intercept. Default ``True``.
    :param standardize: Scale predictors before fitting; predictors are
        centered only when an intercept is included. Default ``True``.
    :param type_gaussian: Gaussian backend: ``"auto"``, ``"naive"``, or
        ``"covariance"``. Automatic covariance updates require at least
        eight lambdas, ``n_samples >= n_features``, and at most 1024
        features. Within those guards, they are selected when
        ``d <= 160``, the requested lambda-path ratio is at least 0.10,
        the ratio is at least 0.05 with ``n >= 4d``, or ``n >= 16d``. The
        resolved choice is available as ``solver.type_gaussian``.
    :param dfmax: Stop after a committed model exceeds this number of nonzero
        coefficients; ``-1`` disables the limit. The crossing model is
        retained, so this is not a hard coefficient cap. Multinomial counts
        nonzero class-feature entries.
    :param prec: Positive stopping/KKT tolerance when
        ``fast_mode=False``. Default ``1e-7``.
    :param max_ite: Native iteration budget within a path subproblem. Default
        ``1000``.
    :param offset: Optional finite link-scale offset per observation for
        binomial or Poisson models. Prediction-derived operations then require
        a corresponding ``newoffset``, except support extraction with
        ``predict(type="nonzero")``.
    :param verbose: Print native tracing information. Default ``False``.
    :param lla_max_stages: Total LLA stage budget for MCP/SCAD binomial,
        Poisson, square-root-lasso, and multinomial fits, including the lasso
        master. Every fit validates an integer of at least three, but only
        non-Gaussian MCP/SCAD optimization consumes the budget. Default
        ``3``.
    :param fast_mode: Use benchmark-calibrated stopping/KKT tolerances.
        Default ``False``. Poisson uses ``4e-4``; binomial,
        square-root-lasso, and multinomial use ``1e-4``; Gaussian stays
        at ``1e-7``. A different custom ``prec`` cannot be combined
        with fast mode.
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
                 type_gaussian="auto",
                 dfmax=-1,
                 prec=1e-7,
                 max_ite=1000,
                 offset=None,
                 verbose=False,
                 lla_max_stages=3,
                 fast_mode=False):

        # ctypes releases the GIL while native solvers write into result
        # buffers. Keep train/CV atomic per Solver without limiting independent
        # Solver instances. RLock is required because fresh CV calls train().
        self._operation_lock = threading.RLock()

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
        self.use_intercept = _boolean_scalar(
            useintercept, 'useintercept')
        self.standardize = _boolean_scalar(standardize, 'standardize')
        if dfmax is None:
            self.dfmax = -1
        else:
            dfmax_value = _finite_real_scalar(dfmax, 'dfmax')
            if dfmax_value != math.floor(dfmax_value) or dfmax_value < -1:
                raise ValueError('"dfmax" must be -1 or a nonnegative integer.')
            if dfmax_value > np.iinfo(np.int32).max:
                raise ValueError('"dfmax" exceeds the native integer limit.')
            self.dfmax = int(dfmax_value)

        self.gamma = _finite_real_scalar(gamma, 'gamma')
        if self.penalty == "mcp":
            self.penaltyflag = 2
            if self.gamma <= 1:
                raise ValueError('"gamma" must be greater than 1 for MCP.')
        elif self.penalty == "scad":
            self.penaltyflag = 3
            if self.gamma <= 2:
                raise ValueError('"gamma" must be greater than 2 for SCAD.')
        else:
            self.penaltyflag = 1

        requested_precision = _finite_real_scalar(prec, 'prec')
        if requested_precision <= 0:
            raise ValueError('"prec" must be positive.')
        self.fast_mode = _boolean_scalar(fast_mode, 'fast_mode')
        fast_precision = _fast_precision(self.family)
        matches_preset = any(math.isclose(
            requested_precision, preset, rel_tol=1e-7, abs_tol=0.0)
            for preset in set((_HIGH_PRECISION, fast_precision)))
        if self.fast_mode and not matches_preset:
            raise ValueError(
                f'"fast_mode=True" fixes prec at {fast_precision:g} for '
                f'family="{self.family}"; remove the custom prec value or '
                'set fast_mode=False.')
        self.prec = (fast_precision if self.fast_mode
                     else requested_precision)
        self.max_ite = _positive_integer(max_ite, 'max_ite')
        self.lla_max_stages = _positive_integer(
            lla_max_stages, 'lla_max_stages')
        if self.lla_max_stages < 3:
            raise ValueError(
                '"lla_max_stages" must be an integer of at least 3.')
        self.verbose = _boolean_scalar(verbose, 'verbose')

        if family == "gaussian" and type_gaussian not in (
                "auto", "naive", "covariance"):
            raise ValueError(
                'Invalid "type_gaussian". Must be one of '
                '"auto", "naive", "covariance".')
        self.type_gaussian_requested = type_gaussian
        self.type_gaussian = type_gaussian

        # Validate and store data
        # Training is deferred, and prediction/assessment/CV retain the
        # original design. Own one C-contiguous copy so caller mutation cannot
        # change any later Solver operation.
        x_raw = _real_numeric_array(x, 'x', copy=True, order='C')
        if x_raw.ndim != 2:
            raise ValueError('"x" must be a two-dimensional matrix.')
        if x_raw.size == 0 or x_raw.shape[0] == 0 or x_raw.shape[1] == 0:
            raise ValueError("No data input.")
        if self.family == "multinomial":
            _validate_multinomial_native_counts(
                x_raw.shape[0], x_raw.shape[1])
        if not np.all(np.isfinite(x_raw)):
            raise ValueError('"x" must contain only finite values.')

        self.num_sample = x_raw.shape[0]
        self.num_feature = x_raw.shape[1]
        if self.family == "multinomial":
            # Keep categorical labels stable after construction.  The encoded
            # native response already owns its storage, but default assessment
            # and class restoration also use these original labels.
            self.y = _multinomial_label_vector(y).copy()
        elif self.family == "binomial":
            levels, codes = _encode_binomial_labels(y)
            # The native stack and every path calculation consume only the
            # owned double codes. Retain just two private levels rather than a
            # second O(n) copy of categorical labels.
            self._binomial_levels = np.asarray(levels).copy()
            self.y = np.ascontiguousarray(codes, dtype='double')
        else:
            # A generated lambda path is computed during construction and the
            # native solver reads the response later in train().  Own this
            # inexpensive O(n) vector so caller mutation cannot make those two
            # stages observe different responses.
            response = _response_vector(y, 'y', dtype='double')
            self.y = response.copy()
            if not np.all(np.isfinite(self.y)):
                raise ValueError('"y" must contain only finite values.')
        if x_raw.shape[0] != self.y.shape[0]:
            raise ValueError(
                'The size of "x" and "y" does not match: '
                'x: %d * %d, y: %d' % (x_raw.shape[0], x_raw.shape[1], self.y.shape[0]))

        # Offset handling
        if offset is not None and family not in ("binomial", "poisson"):
            raise ValueError("offset is only supported for 'binomial' and 'poisson' families.")
        self._offset_supplied = offset is not None
        if offset is not None:
            self._offset = _response_vector(
                offset, 'offset', dtype='double').copy()
            if self._offset.shape[0] != self.num_sample:
                raise ValueError("offset length must equal number of samples.")
            if not np.all(np.isfinite(self._offset)):
                raise ValueError("offset must contain only finite values.")
        else:
            self._offset = np.zeros(self.num_sample, dtype='double')

        # Family-specific validation
        if self.family == "poisson":
            if np.any(self.y < 0):
                raise ValueError("The response vector should be non-negative.")
            if np.any(self.y != np.floor(self.y)):
                raise ValueError("The response vector should be integers.")
            if np.all(self.y == 0):
                raise ValueError(
                    "The response vector is an all-zero vector. The problem is ill-conditioned.")
        elif self.family == "multinomial":
            y_raw, uniq, y_codes = _encode_multinomial_labels(self.y)
            self._K = len(uniq)
            if self._K < 3:
                raise ValueError("multinomial requires >= 3 classes.")
            _validate_multinomial_native_counts(
                self.num_sample, self.num_feature, self._K)
            self.y = np.ascontiguousarray(y_raw)
            self._y_mn = np.ascontiguousarray(y_codes, dtype='double')
            self._y_codes = np.ascontiguousarray(y_codes, dtype=int)
            self._mn_levels = np.asarray(uniq).copy()

        # Centering changes the origin and therefore introduces an implicit
        # intercept.  No-intercept models may scale columns, but must retain
        # the original origin (including nonzero constant columns).
        if self.standardize:
            if self.use_intercept:
                xx, self._xm, self._xinvc = _standardize(x_raw)
            else:
                xx, self._xm, self._xinvc = \
                    _scale_without_centering(x_raw)
            self.x = np.ascontiguousarray(xx, dtype='double')
        else:
            self.x = np.ascontiguousarray(x_raw, dtype='double')
            self._xm = np.zeros(self.num_feature)
            self._xinvc = np.ones(self.num_feature)

        # For Gaussian, center Y when standardizing
        self._ym = 0.0
        if (self.family == "gaussian" and self.standardize and
                self.use_intercept):
            self._ym = np.mean(self.y)
            self._y_fit = np.ascontiguousarray(self.y - self._ym, dtype='double')
        else:
            self._y_fit = self.y

        # Compute lambda path
        n = self.num_sample
        _reject_masked_array(lambdas, 'lambdas')
        try:
            lambda_spec_length = len(lambdas)
        except TypeError as exc:
            raise ValueError(
                '"lambdas" must be a (count, ratio) specification or a '
                'one-dimensional explicit path.') from exc

        # Retain the documented two-element count/ratio convention. NumPy
        # arrays are always explicit paths, including arrays of length 1 or 2.
        generated_path = (
            lambda_spec_length == 2 and not isinstance(lambdas, np.ndarray))
        if not generated_path:
            explicit_lambdas = _real_numeric_array(lambdas, 'lambdas')
            if explicit_lambdas.ndim != 1 or explicit_lambdas.size == 0:
                raise ValueError(
                    'Explicit lambdas must be a nonempty one-dimensional path.')
            if explicit_lambdas.size > np.iinfo(np.int32).max:
                raise ValueError('The lambda path exceeds the native integer limit.')
            if self.family == 'multinomial':
                _validate_multinomial_native_counts(
                    self.num_sample, self.num_feature, self._K,
                    int(explicit_lambdas.size))
            if (not np.all(np.isfinite(explicit_lambdas)) or
                    np.any(explicit_lambdas < 0)):
                raise ValueError(
                    'Explicit lambdas must be finite and nonnegative.')
            if (explicit_lambdas.size > 1 and
                    not np.all(np.diff(explicit_lambdas) < 0)):
                raise ValueError(
                    'Explicit lambdas must be strictly decreasing.')
            self.lambdas = np.ascontiguousarray(explicit_lambdas)
            self.nlambda = int(explicit_lambdas.size)
        else:
            nlambda = _positive_integer(lambdas[0], 'nlambda')
            lambda_min_ratio = _finite_real_scalar(
                lambdas[1], 'lambda_min_ratio')
            if lambda_min_ratio <= 0:
                raise ValueError('"lambda_min_ratio" must be positive.')
            if lambda_min_ratio > 1:
                raise ValueError('"lambda_min_ratio" must be <= 1.')
            if nlambda > 1 and lambda_min_ratio == 1:
                raise ValueError(
                    '"lambda_min_ratio" must be < 1 when nlambda > 1.')
            if self.family == 'multinomial':
                _validate_multinomial_native_counts(
                    self.num_sample, self.num_feature, self._K, nlambda)

            if self.family in ('binomial', 'poisson'):
                eta0 = _scalar_null_linear_predictor(
                    self.y, self.family, offset=self._offset,
                    include_intercept=self.use_intercept)
                fitted0 = (_sigmoid(eta0) if self.family == 'binomial'
                           else _poisson_mean(eta0))
                lambda_max = np.max(
                    np.abs(self.x.T @ (self.y - fitted0))) / n
            elif self.family == 'sqrtlasso':
                eta0 = _scalar_null_linear_predictor(
                    self.y, self.family,
                    include_intercept=self.use_intercept)
                residual0 = self.y - eta0
                L0 = np.sqrt(np.sum(residual0 ** 2) / n)
                lambda_max = (0.0 if L0 == 0 else
                              np.max(np.abs(self.x.T @ residual0)) / n / L0)
            elif self.family == 'multinomial':
                K = self._K
                if self.use_intercept:
                    p0 = (np.bincount(self._y_codes, minlength=K)
                          .astype(float) / n)
                else:
                    p0 = np.full(K, 1.0 / K, dtype='double')
                lambda_max = max(
                    np.max(np.abs(self.x.T @ ((self._y_codes == k).astype(float) - p0[k]))) / n
                    for k in range(K))
            else:
                eta0 = _scalar_null_linear_predictor(
                    self.y, self.family,
                    include_intercept=self.use_intercept)
                lambda_max = np.max(
                    np.abs(self.x.T @ (self.y - eta0))) / n
            if not np.isfinite(lambda_max) or lambda_max < 0:
                raise ValueError("Could not construct a finite lambda path.")
            self.nlambda = nlambda
            if lambda_max == 0:
                # Every feature gradient vanishes at the null fit. A tiny path
                # ending at zero is numerically meaningful and remains usable
                # by prediction/CV routines that require decreasing lambdas.
                if self.nlambda == 1:
                    self.lambdas = np.zeros(1, dtype='double')
                else:
                    self.lambdas = np.linspace(
                        np.finfo(float).eps, 0.0, self.nlambda,
                        dtype='double')
            else:
                self.lambdas = np.exp(np.linspace(
                    math.log(lambda_max),
                    math.log(lambda_min_ratio * lambda_max),
                    self.nlambda)).astype('double')
            self.lambdas = np.ascontiguousarray(self.lambdas)

        if self.family == "gaussian":
            self.type_gaussian = _resolve_gaussian_type(
                self.type_gaussian_requested, self.num_sample,
                self.num_feature, self.lambdas)

        # Store original x for predict
        self._x_orig = np.asarray(x_raw, dtype='double')
        if self.family == "multinomial":
            _validate_multinomial_native_counts(
                self.num_sample, self.num_feature, self._K, self.nlambda)
        self._generated_lambda_path = bool(generated_path)
        # Keep the requested path immutable. Native dfmax/failure handling
        # exposes only a fitted prefix, but a later train() must retry the
        # original request instead of silently treating that prefix as a new
        # complete path.
        self._requested_lambdas = self.lambdas.copy()
        self._requested_nlambda = int(self.nlambda)

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
                'runtime': np.zeros(self.nlambda, dtype='double'),
                'num_fit': np.zeros(1, dtype='int32'),
                'total_train_time': 0,
                'levels': self._mn_levels.copy(),
                'status_code': None,
                'status': 'not_run',
                'failed_lambda': -1,
                'stage': -1,
                'failed_stage': -1,
                'outer_ite': np.zeros(self.nlambda, dtype='int32'),
                'inner_sweeps': np.zeros(self.nlambda, dtype='int64'),
                'coordinate_updates': np.zeros(self.nlambda, dtype='int64'),
                'objective': np.full(self.nlambda, np.nan, dtype='double'),
                'kkt': np.full(self.nlambda, np.nan, dtype='double'),
                'stationarity': np.full(
                    self.nlambda, np.nan, dtype='double'),
                'failure_diagnostics': None,
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
            if self.family == "binomial":
                self.result['levels'] = self._binomial_levels.copy()
            if self.family == "gaussian":
                self.result.update({
                    'status_code': None,
                    'status': 'not_run',
                    'failed_lambda': -1,
                    'failure_diagnostics': None,
                })
            if self.family in ("binomial", "poisson", "sqrtlasso"):
                lla_stages = np.zeros(self.nlambda, dtype='int32')
                self.result.update({
                    'runtime': self.result['train_time'],
                    'status_code': None,
                    'status': 'not_run',
                    'failed_lambda': -1,
                    'stage': -1,
                    'failed_stage': -1,
                    'lla_stages': lla_stages,
                    # Short alias retained for users comparing native output
                    # names across language interfaces; no second allocation.
                    'stages': lla_stages,
                    'objective': np.full(
                        self.nlambda, np.nan, dtype='double'),
                    'kkt': np.full(self.nlambda, np.nan, dtype='double'),
                    'stationarity': np.full(
                        self.nlambda, np.nan, dtype='double'),
                    'failure_diagnostics': None,
                })
            self.trainer = getattr(self, '_' + self.family + '_wrapper')()

        self.result.update({
            'fast_mode': self.fast_mode,
            'precision': self.prec,
            'lla_max_stages': self.lla_max_stages,
        })

    # ------------------------------------------------------------------
    # C interface decorators
    # ------------------------------------------------------------------

    def _decor_cinterface(self, _function, abi_version=1):
        """Decorate a Gaussian C API with versioned status fallback."""
        has_native_loss = abi_version >= 2
        has_status = abi_version >= 3
        CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
        argtypes = [
            CDoubleArray, CDoubleArray, ctypes.c_int, ctypes.c_int, CDoubleArray,
            ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
            ctypes.c_int, ctypes.c_bool,
            ctypes.c_int,        # dfmax
            CDoubleArray, CDoubleArray, CIntArray,
            CIntArray, CDoubleArray,
            CIntArray,           # num_fit
            ctypes.c_bool        # usePython
        ]
        if has_native_loss:
            argtypes.append(CDoubleArray)
        if has_status:
            argtypes.append(CIntArray)
        _bind_ctypes_signature(
            _function, argtypes, ctypes.c_int if has_status else None)

        def wrapper():
            requested_nlambda = int(self.nlambda)
            smooth_objective = np.full(
                requested_nlambda, np.nan, dtype='double')
            failed_lambda = np.full(1, -1, dtype='int32')
            call_args = [
                self._y_fit, self.x, self.num_sample, self.num_feature,
                self.lambdas, requested_nlambda, self.gamma, self.max_ite,
                self.prec, self.penaltyflag, self.use_intercept, self.dfmax,
                self.result['beta'], self.result['intercept'],
                self.result['ite_lamb'], self.result['size_act'],
                self.result['train_time'], self.result['num_fit'], True,
            ]
            if has_native_loss:
                call_args.append(smooth_objective)
            if has_status:
                call_args.append(failed_lambda)
            time_start = time.time()
            if has_status:
                status_code = _function(*call_args)
            else:
                _function(*call_args)
                status_code = None
            time_end = time.time()
            self.result['total_train_time'] = time_end - time_start
            nfit = int(self.result['num_fit'][0])
            if nfit < 0 or nfit > requested_nlambda:
                raise PycassoError(
                    'Gaussian solver returned an invalid fit count '
                    f'(num_fit={nfit}, requested={requested_nlambda}).')
            if has_status:
                status_code = int(status_code)
                status = _SCALAR_LLA_STATUS_NAMES.get(
                    status_code, f'unknown_{status_code}')
                failed_index = int(failed_lambda[0])
                self.result.update({
                    'status_code': status_code,
                    'status': status,
                    'failed_lambda': failed_index,
                    'failure_diagnostics': None,
                })
                usable_status = status_code in (0, 1)
                if not usable_status and nfit == 0:
                    raise PycassoError(
                        'Gaussian solver stopped before completing a lambda '
                        f'value: status={status!r} (code {status_code}), '
                        f'failed_lambda={failed_index}.')
                if usable_status and nfit == 0:
                    raise PycassoError(
                        'Gaussian solver returned no usable lambda values.')
            else:
                _validate_native_fit_count(
                    nfit, requested_nlambda, 'Gaussian')
                status = 'legacy_unknown'
                failed_index = -1
                self.result.update({
                    'status_code': None,
                    'status': status,
                    'failed_lambda': failed_index,
                    'failure_diagnostics': None,
                })
            self._finalize_result()
            self.result['nulldev'] = _null_deviance(
                self.y, self.family,
                include_intercept=self.use_intercept)
            if has_native_loss:
                committed_smooth = smooth_objective[:self.nlambda]
                if not np.all(np.isfinite(committed_smooth)):
                    raise PycassoError(
                        'Gaussian V2 returned a non-finite smooth objective '
                        'for a committed lambda.')
                self.result['smooth_objective'] = _compact_prefix(
                    smooth_objective, self.nlambda)
                devs = _native_scalar_deviances(
                    self.y, self.family,
                    self.result['smooth_objective'])
            else:
                devs = _fit_deviances(
                    self.y, self._x_orig, self.result['beta'],
                    self.result['intercept'], self.family)
            nd = self.result['nulldev']
            if nd is not None and nd > 0:
                self.result['dev_ratio'] = np.clip(1.0 - devs / nd, 0, 1)
            else:
                self.result['dev_ratio'] = np.zeros(self.nlambda)
            if has_status and status_code not in (0, 1):
                self.result['state'] = 'partially trained'
                warnings.warn(
                    'Gaussian solver stopped after retaining a fitted prefix '
                    f'of {self.nlambda}/{requested_nlambda} lambdas: '
                    f'status={status!r} (code {status_code}), '
                    f'failed_lambda={failed_index}.',
                    RuntimeWarning, stacklevel=2)

        return wrapper

    def _decor_cinterface_glm(self, _function):
        """Decorate a C API function (binomial / poisson, with offset)."""
        CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
        argtypes = [
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
        _bind_ctypes_signature(_function, argtypes, None)

        def wrapper():
            requested_nlambda = int(self.nlambda)
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
            _validate_native_fit_count(
                self.result['num_fit'][0], requested_nlambda,
                self.family.capitalize())
            self._finalize_result()
            off = self._offset if np.any(self._offset != 0) else None
            self.result['nulldev'] = _null_deviance(
                self.y, self.family, offset=off,
                include_intercept=self.use_intercept)
            devs = _fit_deviances(self.y, self._x_orig,
                                  self.result['beta'], self.result['intercept'],
                                  self.family, offset=off)
            nd = self.result['nulldev']
            if nd is not None and nd > 0:
                self.result['dev_ratio'] = np.clip(1.0 - devs / nd, 0, 1)
            else:
                self.result['dev_ratio'] = np.zeros(self.nlambda)

        return wrapper

    def _decor_scalar_lla_cinterface(self, legacy_name, v2_name, v3_name,
                                     has_offset=False):
        """Bind a scalar-family adaptive-LLA API with legacy fallback."""
        CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
        base_argtypes = [
            CDoubleArray, CDoubleArray, ctypes.c_int, ctypes.c_int,
            CDoubleArray, ctypes.c_int, ctypes.c_double, ctypes.c_int,
            ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        ]
        if has_offset:
            base_argtypes.append(CDoubleArray)
        base_argtypes.extend([
            CDoubleArray, CDoubleArray, CIntArray, CIntArray, CDoubleArray,
            CIntArray, ctypes.c_bool,
        ])
        diagnostic_argtypes = [
            ctypes.c_int,  # lla_max_stages
            CIntArray,     # failed_lambda (scalar)
            CIntArray,     # failed_stage (scalar)
            CIntArray,     # lla_stages (nlambda)
            CDoubleArray,  # objective (nlambda)
            CDoubleArray,  # weighted-L1 KKT (nlambda)
            CDoubleArray,  # nonconvex stationarity (nlambda)
        ]

        try:
            func = getattr(_PICASSO_LIB, v3_name)
            abi_version = 3
        except AttributeError:
            try:
                func = getattr(_PICASSO_LIB, v2_name)
                abi_version = 2
            except AttributeError:
                func = getattr(_PICASSO_LIB, legacy_name)
                abi_version = 1

        has_diagnostics = abi_version >= 2
        has_native_loss = abi_version >= 3
        if has_diagnostics:
            argtypes = base_argtypes + diagnostic_argtypes
            if has_native_loss:
                argtypes.append(CDoubleArray)
            _bind_ctypes_signature(func, argtypes, ctypes.c_int)
        else:
            _bind_ctypes_signature(func, base_argtypes, None)
            if self.penaltyflag in (2, 3) and self.lla_max_stages != 3:
                raise PycassoError(
                    f'{self.family} MCP/SCAD with nondefault '
                    'lla_max_stages requires the versioned native API '
                    f'{v2_name}; the loaded legacy backend can honor only '
                    'the default value 3. Rebuild or reinstall PICASSO.')

        family_label = {
            'binomial': 'Binomial',
            'poisson': 'Poisson',
            'sqrtlasso': 'Sqrt-lasso',
        }[self.family]

        def wrapper():
            requested_nlambda = int(self.nlambda)
            failed_lambda = np.full(1, -1, dtype='int32')
            failed_stage = np.full(1, -1, dtype='int32')
            lla_stages = self.result['lla_stages']
            objective = self.result['objective']
            kkt = self.result['kkt']
            stationarity = self.result['stationarity']
            smooth_objective = np.full(
                requested_nlambda, np.nan, dtype='double')

            call_args = [
                self._y_fit, self.x, self.num_sample, self.num_feature,
                self.lambdas, requested_nlambda, self.gamma, self.max_ite,
                self.prec, self.penaltyflag, self.use_intercept, self.dfmax,
            ]
            if has_offset:
                call_args.append(self._offset)
            call_args.extend([
                self.result['beta'], self.result['intercept'],
                self.result['ite_lamb'], self.result['size_act'],
                self.result['train_time'], self.result['num_fit'], True,
            ])

            time_start = time.perf_counter()
            if has_diagnostics:
                diagnostic_args = [
                    self.lla_max_stages, failed_lambda, failed_stage,
                    lla_stages, objective, kkt, stationarity,
                ]
                if has_native_loss:
                    diagnostic_args.append(smooth_objective)
                status_code = int(func(*call_args, *diagnostic_args))
                status = _SCALAR_LLA_STATUS_NAMES.get(
                    status_code, 'unknown')
            else:
                func(*call_args)
                status_code = None
                status = 'legacy_unknown'
            self.result['total_train_time'] = (
                time.perf_counter() - time_start)

            nfit = int(self.result['num_fit'][0])
            failed_index = (int(failed_lambda[0])
                            if has_diagnostics else -1)
            stage = int(failed_stage[0]) if has_diagnostics else -1
            self.result['status_code'] = status_code
            self.result['status'] = status
            self.result['failed_lambda'] = failed_index
            self.result['stage'] = stage
            self.result['failed_stage'] = stage
            self.result['failure_diagnostics'] = None

            # Keep a failed point inspectable even though only successfully
            # committed models remain in the public path arrays below.
            if has_diagnostics and 0 <= failed_index < requested_nlambda:
                self.result['failure_diagnostics'] = {
                    'lambda': float(self.lambdas[failed_index]),
                    'train_time': float(
                        self.result['train_time'][failed_index]),
                    'runtime': float(
                        self.result['train_time'][failed_index]),
                    'lla_stages': int(lla_stages[failed_index]),
                    'stages': int(lla_stages[failed_index]),
                    'objective': float(objective[failed_index]),
                    'kkt': float(kkt[failed_index]),
                    'stationarity': float(stationarity[failed_index]),
                }

            # The public result owns the native buffers, so diagnostics remain
            # inspectable after a raised error without another full-path copy.
            self.result['runtime'] = self.result['train_time']
            self.result['stages'] = self.result['lla_stages']

            zero_message = (
                f'{family_label} solver stopped before fitting any lambda: '
                f'status={status!r} (code {status_code}), '
                f'failed_lambda={failed_index}, stage={stage}.'
                if has_diagnostics else
                f'Legacy {family_label.lower()} solver did not fit any '
                'lambda values (termination status is unavailable).')
            nfit = _validate_native_fit_count(
                nfit, requested_nlambda, family_label,
                zero_message=zero_message)

            successful_status = (
                has_diagnostics and status_code in (0, 1, 10))
            self._finalize_result(extra_path_fields=(
                'lla_stages', 'objective', 'kkt', 'stationarity'))
            actual_nlambda = int(self.nlambda)
            if has_native_loss:
                committed_smooth = smooth_objective[:actual_nlambda]
                if not np.all(np.isfinite(committed_smooth)):
                    raise PycassoError(
                        f'{family_label} V3 returned a non-finite smooth '
                        'objective for a committed lambda.')
                self.result['smooth_objective'] = _compact_prefix(
                    smooth_objective, actual_nlambda)

            offset = (self._offset if has_offset and
                      np.any(self._offset != 0) else None)
            self.result['nulldev'] = _null_deviance(
                self.y, self.family, offset=offset,
                include_intercept=self.use_intercept)
            if has_native_loss:
                devs = _native_scalar_deviances(
                    self.y, self.family,
                    self.result['smooth_objective'])
            else:
                devs = _fit_deviances(
                    self.y, self._x_orig, self.result['beta'],
                    self.result['intercept'], self.family, offset=offset)
            null_deviance = self.result['nulldev']
            if null_deviance is not None and null_deviance > 0:
                self.result['dev_ratio'] = np.clip(
                    1.0 - devs / null_deviance, 0, 1)
            else:
                self.result['dev_ratio'] = np.zeros(actual_nlambda)

            if has_diagnostics and not successful_status:
                self.result['state'] = 'partially trained'
                warnings.warn(
                    f'{family_label} solver stopped after retaining a fitted '
                    f'prefix of {actual_nlambda}/{requested_nlambda} lambdas: '
                    f'status={status!r} (code {status_code}), '
                    f'failed_lambda={failed_index}, stage={stage}.',
                    RuntimeWarning, stacklevel=2)
            elif not has_diagnostics and actual_nlambda < requested_nlambda:
                self.result['state'] = 'partially trained'
                warnings.warn(
                    f'Legacy {family_label.lower()} ABI returned a truncated '
                    f'lambda path ({actual_nlambda}/{requested_nlambda}); the '
                    'termination reason is unknown. Rebuild PICASSO for '
                    'versioned status diagnostics.',
                    RuntimeWarning, stacklevel=2)

        return wrapper

    def _finalize_result(self, extra_path_fields=()):
        """Truncate to actual fit count, rescale, add df."""
        nfit = int(self.result['num_fit'][0])
        if 0 < nfit < self.nlambda:
            for field in (
                    'ite_lamb', 'size_act', 'train_time') + tuple(
                        extra_path_fields):
                self.result[field] = _compact_prefix(
                    self.result[field], nfit)
            self.nlambda = nfit
            self.lambdas = _compact_prefix(self.lambdas, nfit)

        self.result['beta'], self.result['intercept'] = \
            _compact_and_rescale_scalar_solution(
                self.result['beta'], self.result['intercept'], nfit,
                self.standardize, self.use_intercept,
                self._xinvc, self._xm)

        if not self.use_intercept:
            self.result['intercept'].fill(0.0)

        if self.family == 'gaussian' and self._ym != 0.0:
            np.add(self.result['intercept'], self._ym,
                   out=self.result['intercept'])

        self.result['df'] = np.count_nonzero(
            self.result['beta'], axis=1).astype('int32', copy=False)
        if 'runtime' in self.result:
            self.result['runtime'] = self.result['train_time']
        if 'lla_stages' in self.result:
            self.result['stages'] = self.result['lla_stages']

    # ------------------------------------------------------------------
    # Family wrappers
    # ------------------------------------------------------------------

    def _gaussian_wrapper(self):
        if self.verbose:
            print("Sparse linear regression.")
            print(self.penalty.upper()
                  + " regularization via active set identification and coordinate descent.\n")
        if self.type_gaussian == "covariance":
            legacy_name = 'SolveLinearRegressionCovUpdate'
            v2_name = 'SolveLinearRegressionCovUpdateV2'
            v3_name = 'SolveLinearRegressionCovUpdateV3'
        else:
            legacy_name = 'SolveLinearRegressionNaiveUpdate'
            v2_name = 'SolveLinearRegressionNaiveUpdateV2'
            v3_name = 'SolveLinearRegressionNaiveUpdateV3'
        try:
            func = getattr(_PICASSO_LIB, v3_name)
            abi_version = 3
        except AttributeError:
            try:
                func = getattr(_PICASSO_LIB, v2_name)
                abi_version = 2
            except AttributeError:
                func = getattr(_PICASSO_LIB, legacy_name)
                abi_version = 1
        self._gaussian_abi_version = abi_version
        return self._decor_cinterface(func, abi_version=abi_version)

    def _binomial_wrapper(self):
        if self.verbose:
            print("Sparse logistic regression.")
            print(self.penalty.upper()
                  + " regularization via Proximal Newton/IRLS.\n")
        return self._decor_scalar_lla_cinterface(
            'SolveLogisticRegression', 'SolveLogisticRegressionV2',
            'SolveLogisticRegressionV3',
            has_offset=True)

    def _poisson_wrapper(self):
        if self.verbose:
            print("Sparse poisson regression.")
            print(self.penalty.upper()
                  + " regularization via Proximal Newton/IRLS.\n")
        return self._decor_scalar_lla_cinterface(
            'SolvePoissonRegression', 'SolvePoissonRegressionV2',
            'SolvePoissonRegressionV3',
            has_offset=True)

    def _sqrtlasso_wrapper(self):
        if self.verbose:
            print("Sparse sqrt lasso regression.")
            print(self.penalty.upper()
                  + " regularization via adaptive active-set quadratic-MM"
                  + " updates.\n")
        return self._decor_scalar_lla_cinterface(
            'SolveSqrtLinearRegression', 'SolveSqrtLinearRegressionV2',
            'SolveSqrtLinearRegressionV3')

    def _multinomial_wrapper(self):
        if self.verbose:
            print("Sparse multinomial regression.")
            print(self.penalty.upper()
                  + " regularization via Proximal Newton/IRLS.\n")

        K = self._K
        d = self.num_feature

        CDoubleArray = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        CIntArray = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
        CInt64Array = ndpointer(ctypes.c_longlong, flags='C_CONTIGUOUS')
        base_argtypes = [
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
        try:
            func = _PICASSO_LIB.SolveMultinomialRegressionV5
            abi_version = 5
        except AttributeError:
            try:
                func = _PICASSO_LIB.SolveMultinomialRegressionV4
                abi_version = 4
            except AttributeError:
                try:
                    func = _PICASSO_LIB.SolveMultinomialRegressionV3
                    abi_version = 3
                except AttributeError:
                    try:
                        func = _PICASSO_LIB.SolveMultinomialRegressionV2
                        abi_version = 2
                    except AttributeError:
                        # Keep old wheels usable, but never imply that an
                        # unversioned ABI reported a successful termination
                        # status or diagnostics.
                        func = _PICASSO_LIB.SolveMultinomialRegression
                        abi_version = 1
        self._multinomial_abi_version = abi_version

        has_diagnostics = abi_version >= 2
        diagnostic_argtypes = [
            CIntArray,       # failed_lambda (scalar)
            CIntArray,       # failed_stage (scalar)
            CIntArray,       # outer_ite (nlambda)
            CInt64Array,     # inner_sweeps (nlambda)
            CInt64Array,     # coordinate_updates (nlambda)
            CDoubleArray,    # objective (nlambda)
            CDoubleArray,    # kkt (nlambda)
            CDoubleArray,    # stationarity (nlambda)
        ]
        if abi_version == 5:
            argtypes = (base_argtypes + [ctypes.c_int, ctypes.c_bool] +
                        diagnostic_argtypes + [CDoubleArray])
            restype = ctypes.c_int
        elif abi_version == 4:
            argtypes = (base_argtypes + [ctypes.c_int, ctypes.c_bool] +
                        diagnostic_argtypes)
            restype = ctypes.c_int
        elif abi_version == 3:
            argtypes = base_argtypes + [ctypes.c_int] + diagnostic_argtypes
            restype = ctypes.c_int
        elif abi_version == 2:
            argtypes = base_argtypes + diagnostic_argtypes
            restype = ctypes.c_int
        else:
            argtypes = base_argtypes
            restype = None
        _bind_ctypes_signature(func, argtypes, restype)

        if (abi_version < 3 and self.penalty in ("mcp", "scad") and
                self.lla_max_stages != 3):
            raise PycassoError(
                'The installed PICASSO native library does not expose the V3 '
                'multinomial API required to honor a nondefault '
                '"lla_max_stages" value. Rebuild or upgrade PICASSO.')

        def wrapper():
            nlambda = int(self.nlambda)
            # train() has just reset these owning, C-contiguous buffers.
            # Passing them directly avoids staging and then copying a second
            # full multinomial coefficient path. Resolve the aliases on every
            # call so retraining writes fresh buffers, not a captured old fit.
            beta_out = self.result['beta']
            intcpt_out = self.result['intercept']
            ite_lamb = self.result['ite_lamb']
            size_act = self.result['size_act']
            train_time = self.result['train_time']
            runtime = self.result['runtime']
            num_fit = self.result['num_fit']
            failed_lambda = np.full(1, -1, dtype='int32')
            failed_stage = np.full(1, -1, dtype='int32')
            outer_ite = self.result['outer_ite']
            inner_sweeps = self.result['inner_sweeps']
            coordinate_updates = self.result['coordinate_updates']
            objective = self.result['objective']
            kkt = self.result['kkt']
            stationarity = self.result['stationarity']
            smooth_nll = (self.result['smooth_nll']
                          if abi_version >= 5 else None)

            call_args = (
                self._y_mn, self.x, self.num_sample, d, K,
                self.lambdas, nlambda, self.gamma, self.max_ite, self.prec,
                self.penaltyflag, self.use_intercept, self.dfmax,
                beta_out, intcpt_out, ite_lamb, size_act, train_time,
                num_fit, True)
            diagnostic_args = (
                failed_lambda, failed_stage, outer_ite, inner_sweeps,
                coordinate_updates, objective, kkt, stationarity)
            time_start = time.perf_counter()
            if abi_version == 5:
                status_code = int(func(
                    *call_args, self.lla_max_stages,
                    self._generated_lambda_path, *diagnostic_args,
                    smooth_nll))
                status = _MULTINOMIAL_STATUS_NAMES.get(
                    status_code, 'unknown')
            elif abi_version == 4:
                status_code = int(func(
                    *call_args, self.lla_max_stages,
                    self._generated_lambda_path, *diagnostic_args))
                status = _MULTINOMIAL_STATUS_NAMES.get(
                    status_code, 'unknown')
            elif abi_version == 3:
                status_code = int(func(
                    *call_args, self.lla_max_stages, *diagnostic_args))
                status = _MULTINOMIAL_STATUS_NAMES.get(
                    status_code, 'unknown')
            elif abi_version == 2:
                status_code = int(func(*call_args, *diagnostic_args))
                status = _MULTINOMIAL_STATUS_NAMES.get(
                    status_code, 'unknown')
            else:
                func(*call_args)
                status_code = None
                status = 'legacy_unknown'
                # The legacy iteration output is the only detailed count it
                # can supply; all other missing diagnostics stay explicit.
                np.copyto(inner_sweeps, ite_lamb, casting='unsafe')
            self.result['total_train_time'] = (
                time.perf_counter() - time_start)
            # Preserve the existing public distinction between runtime and
            # train_time without allocating a third path-sized array.
            np.copyto(runtime, train_time)

            nfit = int(num_fit[0])
            failed_index = int(failed_lambda[0]) if has_diagnostics else -1
            stage = int(failed_stage[0]) if has_diagnostics else -1
            self.result['status_code'] = status_code
            self.result['status'] = status
            self.result['failed_lambda'] = failed_index
            self.result['stage'] = stage
            self.result['failed_stage'] = stage
            self.result['failure_diagnostics'] = None
            self.result['requested_nlambda'] = int(nlambda)

            if 0 <= failed_index < nlambda:
                self.result['failure_diagnostics'] = {
                    'lambda': float(self.lambdas[failed_index]),
                    'train_time': float(train_time[failed_index]),
                    'runtime': float(train_time[failed_index]),
                    'outer_ite': int(outer_ite[failed_index]),
                    'inner_sweeps': int(inner_sweeps[failed_index]),
                    'coordinate_updates': int(
                        coordinate_updates[failed_index]),
                    'objective': float(objective[failed_index]),
                    'kkt': float(kkt[failed_index]),
                    'stationarity': float(stationarity[failed_index]),
                }

            successful_status = (
                has_diagnostics and status_code in (0, 1, 10))
            self.result['path_early_stopped'] = bool(
                has_diagnostics and status_code in (0, 10) and
                0 < nfit < nlambda)
            zero_message = (
                'Multinomial solver stopped before fitting any lambda: '
                f'status={status!r} (code {status_code}), '
                f'failed_lambda={failed_index}, stage={stage}.'
                if has_diagnostics else
                'Legacy multinomial solver did not fit any lambda values '
                '(termination status is unavailable).')
            nfit = _validate_native_fit_count(
                nfit, nlambda, 'Multinomial', zero_message=zero_message)

            actual_nl = nfit
            if abi_version >= 5:
                committed_smooth_nll = smooth_nll[:actual_nl]
                if not np.all(np.isfinite(committed_smooth_nll)):
                    raise PycassoError(
                        'Multinomial V5 returned a non-finite smooth NLL for '
                        'a committed lambda.')

            if actual_nl < nlambda:
                compact_fields = [
                    'beta', 'intercept', 'ite_lamb', 'size_act', 'df',
                    'train_time', 'runtime', 'outer_ite', 'inner_sweeps',
                    'coordinate_updates', 'objective', 'kkt',
                    'stationarity',
                ]
                if abi_version >= 5:
                    compact_fields.append('smooth_nll')
                for field in compact_fields:
                    self.result[field] = \
                        self.result[field][:actual_nl].copy()
                self.nlambda = actual_nl
                self.lambdas = self.lambdas[:actual_nl].copy()

            raw = self.result['beta']
            raw_intcpt = self.result['intercept']

            # Rescale: for each class k
            # beta[li, k, :] *= xinvc  => beta_rescaled[li, k, :]
            # intcpt[li, k]  -= beta_rescaled[li, k, :] @ xm
            if self.standardize:
                beta_r, intcpt_r = _rescale_multinomial_solution_in_place(
                    raw, raw_intcpt, self._xinvc, self._xm)
            else:
                beta_r = raw
                intcpt_r = raw_intcpt
            if not self.use_intercept:
                intcpt_r.fill(0.0)

            # Keep the working set at one K-by-d model rather than creating
            # an abs/threshold temporary as large as the entire path.
            for path_index in range(actual_nl):
                self.result['df'][path_index] = np.count_nonzero(
                    np.abs(beta_r[path_index]) > 1e-8)

            # Deviance
            null_dev = _mn_null_deviance(
                self._y_codes, K, include_intercept=self.use_intercept)
            self.result['nulldev'] = null_dev
            if abi_version >= 5:
                devs = self.result['smooth_nll']
            else:
                devs = _mn_fit_deviances(
                    self._y_codes, self._x_orig, beta_r, intcpt_r)
            if null_dev > 0:
                self.result['dev_ratio'] = np.clip(1.0 - devs / null_dev, 0, 1)
            else:
                self.result['dev_ratio'] = np.zeros(actual_nl)

            if has_diagnostics and not successful_status:
                self.result['state'] = 'partially trained'
                warnings.warn(
                    'Multinomial solver stopped after retaining a fitted '
                    f'prefix of {actual_nl}/{nlambda} lambdas: '
                    f'status={status!r} (code {status_code}), '
                    f'failed_lambda={failed_index}, stage={stage}.',
                    RuntimeWarning, stacklevel=2)
            elif not has_diagnostics and actual_nl < nlambda:
                self.result['state'] = 'partially trained'
                warnings.warn(
                    'Legacy multinomial ABI returned a truncated lambda path '
                    f'({actual_nl}/{nlambda}); the termination reason is '
                    'unknown. Rebuild PICASSO for versioned status '
                    'diagnostics.',
                    RuntimeWarning, stacklevel=2)

        return wrapper

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def _reset_result_for_training(self):
        """Restore the requested path and allocate fresh native outputs."""
        self.lambdas = self._requested_lambdas.copy()
        self.nlambda = self._requested_nlambda
        self.result.pop('nulldev', None)
        self.result.pop('dev_ratio', None)
        self.result.pop('smooth_objective', None)
        self.result.pop('smooth_nll', None)
        if self.family == "multinomial":
            K = self._K
            d = self.num_feature
            self.result.update({
                'beta': np.zeros((self.nlambda, K, d), dtype='double'),
                'intercept': np.zeros((self.nlambda, K), dtype='double'),
                'ite_lamb': np.zeros(self.nlambda, dtype='int32'),
                'size_act': np.zeros(self.nlambda, dtype='int32'),
                'df': np.zeros(self.nlambda, dtype='int32'),
                'train_time': np.zeros(self.nlambda, dtype='double'),
                'runtime': np.zeros(self.nlambda, dtype='double'),
                'num_fit': np.zeros(1, dtype='int32'),
                'total_train_time': 0,
                'status_code': None,
                'status': 'not_run',
                'failed_lambda': -1,
                'stage': -1,
                'failed_stage': -1,
                'outer_ite': np.zeros(self.nlambda, dtype='int32'),
                'inner_sweeps': np.zeros(self.nlambda, dtype='int64'),
                'coordinate_updates': np.zeros(
                    self.nlambda, dtype='int64'),
                'objective': np.full(
                    self.nlambda, np.nan, dtype='double'),
                'kkt': np.full(self.nlambda, np.nan, dtype='double'),
                'stationarity': np.full(
                    self.nlambda, np.nan, dtype='double'),
                'failure_diagnostics': None,
                'levels': self._mn_levels.copy(),
                'state': 'not trained',
            })
            if getattr(self, '_multinomial_abi_version', 0) >= 5:
                self.result['smooth_nll'] = np.full(
                    self.nlambda, np.nan, dtype='double')
            return

        self.result.update({
            'beta': np.zeros(
                (self.nlambda, self.num_feature), dtype='double'),
            'intercept': np.zeros(self.nlambda, dtype='double'),
            'ite_lamb': np.zeros(self.nlambda, dtype='int32'),
            'size_act': np.zeros(self.nlambda, dtype='int32'),
            'df': np.zeros(self.nlambda, dtype='int32'),
            'train_time': np.zeros(self.nlambda, dtype='double'),
            'num_fit': np.zeros(1, dtype='int32'),
            'total_train_time': 0,
            'state': 'not trained',
        })
        if self.family == "binomial":
            self.result['levels'] = self._binomial_levels.copy()
        if self.family == "gaussian":
            self.result.update({
                'status_code': None,
                'status': 'not_run',
                'failed_lambda': -1,
                'failure_diagnostics': None,
            })
        if self.family in ("binomial", "poisson", "sqrtlasso"):
            lla_stages = np.zeros(self.nlambda, dtype='int32')
            self.result.update({
                'runtime': self.result['train_time'],
                'status_code': None,
                'status': 'not_run',
                'failed_lambda': -1,
                'stage': -1,
                'failed_stage': -1,
                'lla_stages': lla_stages,
                'stages': lla_stages,
                'objective': np.full(
                    self.nlambda, np.nan, dtype='double'),
                'kkt': np.full(self.nlambda, np.nan, dtype='double'),
                'stationarity': np.full(
                    self.nlambda, np.nan, dtype='double'),
                'failure_diagnostics': None,
            })

    @_serialize_solver_operation
    def train(self):
        """Train or retrain the requested regularization path.

        A retraining call starts again from the originally requested lambda
        path, not from a previously truncated prefix. Usable termination
        leaves ``result["state"]`` equal to ``"trained"``. A hard failure
        after at least one committed lambda retains that prefix as
        ``"partially trained"`` and emits ``RuntimeWarning``.

        :return: ``None``.
        """
        self._reset_result_for_training()
        try:
            self.trainer()
        except Exception:
            self.result['state'] = 'not trained'
            raise
        if self.result['state'] != 'partially trained':
            self.result['state'] = 'trained'
        if self.verbose:
            print('Training is over.')

    def coef(self):
        """Return the live result dictionary.

        Scalar-family ``beta`` has shape
        ``(fitted_nlambda, n_features)`` and ``intercept`` has shape
        ``(fitted_nlambda,)``. Multinomial ``beta`` has shape
        ``(fitted_nlambda, n_classes, n_features)``, ``intercept`` has
        shape ``(fitted_nlambda, n_classes)``, and ``levels`` gives the
        class-axis labels. Binomial results also expose their two labels in
        encoded 0/1 order through ``levels``. All trained results include path sizes,
        coefficients, intercepts, degrees of freedom, timing, state,
        ``fast_mode``, ``precision``, ``lla_max_stages``, null deviance, and
        deviance ratio.

        Non-Gaussian results also expose status/failure metadata and
        per-lambda ``runtime``, ``objective``, ``kkt``, and
        ``stationarity``. Binomial, Poisson, and square-root-lasso
        results add ``lla_stages`` (alias ``stages``); multinomial results
        add ``outer_ite``, ``inner_sweeps``, and
        ``coordinate_updates``.

        The returned dictionary is not a copy.
        """
        if self.result['state'] == 'not trained':
            print('Warning: The model has not been trained yet!')
        return self.result

    def _resolve_lam(self, lam):
        """Find lambda index/interpolation for a given lambda value.

        Returns (beta, intercept) interpolated to the requested lambda.
        Prints a note if interpolation is needed.
        """
        lam = _finite_real_scalar(lam, 'lam')
        if lam < 0:
            raise ValueError('"lam" must be nonnegative.')
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

    def _nearest_lam_index(self, lam):
        """Return the nearest fitted lambda index for support extraction."""
        lam = _finite_real_scalar(lam, 'lam')
        if lam < 0:
            raise ValueError('"lam" must be nonnegative.')
        # np.argmin returns the first match, so an exact distance tie selects
        # the earlier (larger) lambda on the decreasing path, as in R.
        return int(np.argmin(np.abs(self.lambdas - lam)))

    def _resolve_prediction_offset(self, newoffset, sample_count):
        """Validate an offset for prediction-derived operations."""
        if self.family not in ("binomial", "poisson"):
            if newoffset is not None:
                raise ValueError(
                    '"newoffset" is only supported for binomial and Poisson '
                    'prediction.')
            return None
        if newoffset is None:
            if self._offset_supplied:
                raise ValueError(
                    '"newoffset" must be provided when using a model fitted '
                    'with offset.')
            return None
        prediction_offset = _response_vector(
            newoffset, 'newoffset', dtype='double')
        if prediction_offset.shape[0] != sample_count:
            raise ValueError(
                '"newoffset" length must equal the number of prediction rows.')
        if not np.all(np.isfinite(prediction_offset)):
            raise ValueError('"newoffset" must contain only finite values.')
        return prediction_offset

    def predict(self, newdata=None, lambdidx=None, type="response", lam=None,
                newoffset=None):
        """Predict responses for new data.

        :param newdata: Finite matrix with ``n_features`` columns. Defaults
            to the training design. Explicit prediction data must contain at
            least one row.
        :param lambdidx: Zero-based fitted-path index. Defaults to the last
            fitted lambda.
        :param type: ``"response"`` (default), ``"link"``, ``"class"``
            (binomial/multinomial only), or ``"nonzero"``.
        :param lam: Optional finite nonnegative lambda value. It overrides
            ``lambdidx``, interpolates coefficients inside the fitted path,
            and clamps values outside the path to an endpoint. Because a
            support set cannot be interpolated, ``type="nonzero"`` instead
            uses the nearest fitted lambda; distance ties select the earlier,
            larger lambda.
        :param newoffset: Per-observation offset for binomial or Poisson
            prediction. It is required when the model was fitted with an
            offset and otherwise defaults to zero. It is not needed for
            ``type="nonzero"``.
        :return: Scalar response/link/class predictions have shape
            ``(n_new,)``. Multinomial response/link predictions have
            shape ``(n_new, n_classes)`` and class prediction has shape
            ``(n_new,)`` using the fitted labels. Binomial class prediction
            returns 0/1 codes; ``result["levels"]`` maps those codes back to
            the fitted labels. Nonzero prediction returns one zero-based index
            array, or one list per multinomial class.
        """
        if self.result['state'] not in ('trained', 'partially trained'):
            raise PycassoError("The model must be trained before prediction.")
        valid_types = {"response", "link", "nonzero"}
        if self.family in ("binomial", "multinomial"):
            valid_types.add("class")
        if type not in valid_types:
            raise ValueError(
                f'Invalid prediction type {type!r}; expected one of '
                f'{sorted(valid_types)}.')
        if (newoffset is not None and
                self.family not in ("binomial", "poisson")):
            self._resolve_prediction_offset(newoffset, 0)

        # Support extraction does not evaluate observations.  Preserve the
        # established validation of explicitly supplied newdata, but avoid an
        # otherwise unnecessary O(n*d) finite-value scan of the training design
        # when it is omitted.
        skip_default_prediction_data = type == "nonzero" and newdata is None
        if not skip_default_prediction_data:
            if newdata is not None:
                _reject_masked_array(newdata, 'newdata')
            x_pred = (self._x_orig if newdata is None else
                      _real_numeric_array(newdata, 'newdata'))
            if x_pred.ndim != 2 or x_pred.shape[1] != self.num_feature:
                raise ValueError(
                    f'"newdata" must be a matrix with '
                    f'{self.num_feature} columns.')
            if x_pred.shape[0] == 0:
                raise ValueError('"newdata" must contain at least one row.')
            # The omitted-data path uses the constructor-owned design, which
            # was validated before it was stored. Explicit data must always be
            # checked, even when the caller passes self._x_orig by identity.
            if (newdata is not None and
                    not np.all(np.isfinite(x_pred))):
                raise ValueError('"newdata" must contain only finite values.')

        # Determine beta/intercept to use
        if lam is not None:
            if type == "nonzero":
                nearest = self._nearest_lam_index(lam)
                _beta = self.result['beta'][nearest]
                _intercept = self.result['intercept'][nearest]
            else:
                _beta, _intercept = self._resolve_lam(lam)
        else:
            if lambdidx is None:
                lambdidx = self.nlambda - 1
            lambdidx = _path_index(lambdidx, self.nlambda)
            _beta = self.result['beta'][lambdidx]
            _intercept = self.result['intercept'][lambdidx]

        # nonzero: feature indices with nonzero coefficients
        if type == "nonzero":
            if self.family == "multinomial":
                return [list(np.where(np.abs(_beta[k]) > 1e-8)[0])
                        for k in range(self._K)]
            return np.where(np.abs(_beta) > 1e-8)[0]

        prediction_offset = self._resolve_prediction_offset(
            newoffset, x_pred.shape[0])

        if self.family == "multinomial":
            lp = x_pred @ _beta.T + _intercept  # (n, K)
            if not np.all(np.isfinite(lp)):
                raise ValueError('Multinomial logits must be finite.')
            if type == "link":
                return lp
            if type == "class":
                return self._mn_levels[np.argmax(lp, axis=1)]
            # response: softmax
            return _softmax(lp)

        eta = x_pred @ _beta + _intercept
        if prediction_offset is not None:
            eta += prediction_offset

        if type == "link":
            return eta

        if self.family in ("gaussian", "sqrtlasso"):
            return eta  # same as link

        if self.family == "binomial":
            if type == "class":
                return (eta > 0).astype(int)
            return _sigmoid(eta)

        if self.family == "poisson":
            return _poisson_mean(eta)

        return eta

    def assess(self, newx=None, newy=None, newoffset=None):
        """Compute evaluation metrics over the full lambda path.

        :param newx: Nonempty finite data matrix. Defaults to training data.
        :param newy: Response vector. Defaults to the training response.
        :param newoffset: Offset for binomial or Poisson assessment. Required
            for a model fitted with an offset.
        :return: Dictionary of ``(fitted_nlambda,)`` arrays. Every family
            returns ``lambda`` and ``deviance``. Gaussian and
            square-root-lasso add ``mse`` and ``mae``; Poisson adds
            ``mse``; binomial and multinomial add ``class_error``.
        """
        if self.result['state'] not in ('trained', 'partially trained'):
            raise PycassoError("The model must be trained before assessment.")
        if newx is not None:
            _reject_masked_array(newx, 'newx')
        x = (self._x_orig if newx is None else
             _real_numeric_array(newx, 'newx'))
        if x.ndim != 2 or x.shape[1] != self.num_feature:
            raise ValueError(
                f'"newx" must be a matrix with {self.num_feature} columns.')
        if x.shape[0] == 0:
            raise ValueError('"newx" must contain at least one row.')
        # The constructor owns and validated the default design. Preserve the
        # full validation contract for every explicitly supplied matrix.
        if newx is not None and not np.all(np.isfinite(x)):
            raise ValueError('"newx" must contain only finite values.')
        prediction_offset = self._resolve_prediction_offset(
            newoffset, x.shape[0])

        # Assessment results are independent snapshots.  Returning the live
        # fitted path here would let a metrics consumer mutate solver state.
        result = {'lambda': self.lambdas.copy()}

        if self.family == "multinomial":
            label_values = self.y if newy is None else newy
            y, _, y_codes = _encode_multinomial_labels(
                label_values, self._mn_levels, name='newy')
            if y.shape[0] != x.shape[0]:
                raise ValueError(
                    'The number of rows in "newx" must equal the length of '
                    '"newy".')
            result['deviance'], result['class_error'] = \
                _mn_assessment_metrics(
                    y_codes, x, self.result['beta'],
                    self.result['intercept'])
            return result

        if newy is None:
            y = self.y
        elif self.family == "binomial":
            _, y = _encode_binomial_labels(
                newy, self._binomial_levels, name='newy')
        else:
            y = _response_vector(newy, 'newy', dtype='double')
            if not np.all(np.isfinite(y)):
                raise ValueError('"newy" must contain only finite values.')
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                'The number of rows in "newx" must equal the length of '
                '"newy".')
        if self.family == "poisson" and np.any(y < 0):
            raise ValueError(
                '"newy" must contain nonnegative values for Poisson assessment.')

        beta = self.result['beta']       # (nlambda, d)
        intercept = self.result['intercept']  # (nlambda,)
        result.update(_scalar_path_metrics(
            y, x, beta, intercept, self.family,
            offset=prediction_offset, include_assessment=True))

        return result

    def confusion(self, newx, newy, lambdidx=None, newoffset=None):
        """Compute confusion matrices for binomial or multinomial models.

        :param newx: Nonempty finite data matrix.
        :param newy: True binomial labels (or encoded zero/one values) or
            fitted multinomial labels.
        :param lambdidx: Scalar or iterable of zero-based fitted-path indices.
            Defaults to every fitted lambda.
        :param newoffset: Offset for binomial confusion matrices. Required for
            a binomial model fitted with an offset.
        :return: List of integer square matrices with predicted classes in
            rows and observed classes in columns. Multinomial axes follow
            ``result["levels"]``.
        """
        if self.result['state'] not in ('trained', 'partially trained'):
            raise PycassoError("The model must be trained before confusion().")
        if self.family not in ("binomial", "multinomial"):
            raise ValueError(
                "confusion() supports only binomial or multinomial family.")
        _reject_masked_array(newx, 'newx')
        x = _real_numeric_array(newx, 'newx')
        if x.ndim != 2 or x.shape[1] != self.num_feature:
            raise ValueError(
                f'"newx" must be a matrix with {self.num_feature} columns.')
        if x.shape[0] == 0:
            raise ValueError('"newx" must contain at least one row.')
        if not np.all(np.isfinite(x)):
            raise ValueError('"newx" must contain only finite values.')
        prediction_offset = self._resolve_prediction_offset(
            newoffset, x.shape[0])
        if lambdidx is None:
            lambdidx = list(range(self.nlambda))
        else:
            _reject_masked_array(lambdidx, 'lambdidx')
            if np.asarray(lambdidx).ndim == 0:
                lambdidx = [lambdidx]
        indices = [_path_index(index, self.nlambda) for index in lambdidx]
        if not indices:
            raise ValueError('"lambdidx" must contain at least one index.')

        if self.family == "multinomial":
            labels, _, y_codes = _encode_multinomial_labels(
                newy, self._mn_levels, name='newy')
            if labels.shape[0] != x.shape[0]:
                raise ValueError(
                    'The number of rows in "newx" must equal the length of '
                    '"newy".')
            matrices = []
            for index in indices:
                logits = (x @ self.result['beta'][index].T +
                          self.result['intercept'][index])
                if not np.all(np.isfinite(logits)):
                    raise ValueError('Multinomial logits must be finite.')
                predictions = np.argmax(logits, axis=1)
                matrix = np.zeros((self._K, self._K), dtype=int)
                np.add.at(matrix, (predictions, y_codes), 1)
                matrices.append(matrix)
            return matrices

        _, y = _encode_binomial_labels(
            newy, self._binomial_levels, name='newy')
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                'The number of rows in "newx" must equal the length of '
                '"newy".')
        matrices = []
        for i in indices:
            eta = x @ self.result['beta'][i] + self.result['intercept'][i]
            if prediction_offset is not None:
                eta += prediction_offset
            pred = (eta > 0).astype(int)
            ytrue = y.astype(int)
            cm = np.zeros((2, 2), dtype=int)
            for p in range(2):
                for t in range(2):
                    cm[p, t] = int(np.sum((pred == p) & (ytrue == t)))
            matrices.append(cm)
        return matrices

    @_serialize_solver_operation
    def cross_validate(self, nfolds=10, foldid=None, type_measure="default",
                       n_jobs=1):
        """K-fold cross-validation to select lambda.

        The full-data path is trained first if necessary. Generated
        categorical folds are stratified; training complements must retain
        every class. Binomial and Poisson offsets are subset automatically.
        The initial full-data generated multinomial path may stop normally
        after saturation; its retained prefix then becomes the fixed path for
        every fold. Any shortened, partially trained, or otherwise unusable
        fold is rejected. Concurrent folds own separate designs and outputs,
        so memory use rises with ``n_jobs``; cap BLAS threads separately to
        avoid oversubscription.

        :param nfolds: Number of folds. Default ``10``.
        :param foldid: Optional length-``n`` vector of zero-based,
            contiguous fold IDs starting at zero. (The R
            ``cv.picasso`` counterpart instead expects one-based labels,
            matching its one-based lambda indices.)
        :param type_measure: One of ``"default"``, ``"deviance"``, ``"mse"``,
            ``"mae"``, or ``"class"``. The default is class error for
            binomial/multinomial and deviance otherwise. Multinomial supports
            only class error and deviance. MSE/MAE use response-scale fits.
        :param n_jobs: Number of fold-fitting threads. Default ``1`` preserves
            serial execution. Values above one run independent fold solvers
            concurrently, capped at the number of folds.
        :return: Dictionary with equal-length ``lambda``, ``cvm``,
            ``cvsd``, ``cvup``, ``cvlo``, and ``nzero`` arrays;
            scalar ``lambda_min`` and ``lambda_1se``; plus ``foldid``,
            ``name``, ``fast_mode``, and effective ``precision``.
        """
        # Validate before training, generating folds, or otherwise changing
        # this Solver. Parallelism is deliberately opt-in.
        n_jobs = _positive_integer(n_jobs, 'n_jobs')
        n = self.num_sample
        valid_measures = {"default", "deviance", "mse", "mae", "class"}
        if type_measure not in valid_measures:
            raise ValueError(
                'Invalid "type_measure". Expected one of "default", '
                '"deviance", "mse", "mae", or "class".')
        if type_measure == "default":
            type_measure = ("class" if self.family in
                            ("binomial", "multinomial") else "deviance")
        if (self.family == "multinomial" and
                type_measure not in ("deviance", "class")):
            raise ValueError(
                'Multinomial cross-validation supports only "deviance" '
                'and "class" measures.')
        if (type_measure == "class" and
                self.family not in ("binomial", "multinomial")):
            raise ValueError(
                'Class loss is available only for binomial or multinomial '
                'models.')
        measure_name = type_measure

        if self.result['state'] == 'not trained':
            # Establish the common full-data path and truthful nzero values
            # before fitting folds, matching the R interface's lifecycle.
            self.train()
        if self.result['state'] == 'partially trained':
            raise PycassoError(
                f'Cannot cross-validate a partially trained {self.family} '
                'path; relax the convergence settings first.')

        if foldid is None:
            nfolds = _positive_integer(nfolds, 'nfolds')
            if nfolds < 2:
                raise ValueError('"nfolds" must be at least 2.')
            if nfolds > n:
                raise ValueError(
                    '"nfolds" cannot exceed the number of observations.')

            foldid = np.empty(n, dtype=int)
            if self.family in ("binomial", "multinomial"):
                class_codes = (self._y_codes if self.family == "multinomial"
                               else self.y.astype(int))
                number_classes = self._K if self.family == "multinomial" else 2
                class_counts = np.bincount(
                    class_codes, minlength=number_classes)
                if np.any(class_counts < 2):
                    smallest = int(class_counts.min())
                    raise ValueError(
                        'Each categorical class must contain at least two '
                        'observations for cross-validation; the smallest class '
                        f'contains {smallest}.')
                # Round-robin assignment ensures no held-out fold contains an
                # entire class, so every training complement retains it.
                next_fold = int(np.random.randint(nfolds))
                for klass in range(number_classes):
                    indices = np.flatnonzero(class_codes == klass)
                    indices = np.random.permutation(indices)
                    foldid[indices] = (
                        np.arange(indices.size) + next_fold) % nfolds
                    next_fold = (next_fold + indices.size) % nfolds
            else:
                indices = np.random.permutation(n)
                foldid[indices] = np.arange(n) % nfolds
        else:
            _reject_masked_array(foldid, 'foldid')
            raw_foldid = np.asarray(foldid)
            if raw_foldid.ndim != 1:
                raise ValueError('"foldid" must be a one-dimensional vector.')
            if raw_foldid.size != n:
                raise ValueError(
                    f'"foldid" length must equal the number of observations '
                    f'({n}).')
            if (np.issubdtype(raw_foldid.dtype, np.bool_) or
                    any(isinstance(_label_key(value), (bool, np.bool_))
                        for value in raw_foldid)):
                raise ValueError('"foldid" values must be nonnegative integers.')
            try:
                numeric_foldid = _real_numeric_array(raw_foldid, 'foldid')
            except ValueError as exc:
                raise ValueError(
                    '"foldid" values must be nonnegative integers.') from exc
            if (not np.all(np.isfinite(numeric_foldid)) or
                    np.any(numeric_foldid < 0) or
                    np.any(numeric_foldid != np.floor(numeric_foldid))):
                raise ValueError(
                    '"foldid" values must be finite nonnegative integers.')
            if np.any(numeric_foldid > np.iinfo(np.int32).max):
                raise ValueError('"foldid" exceeds the native integer limit.')
            foldid = numeric_foldid.astype(int)
            folds = np.unique(foldid)
            if folds.size < 2:
                raise ValueError('"foldid" must define at least two folds.')
            if folds[0] != 0 or np.any(np.diff(folds) != 1):
                raise ValueError(
                    '"foldid" values must be contiguous and start at 0.')
            nfolds = int(folds.size)

            if self.family in ("binomial", "multinomial"):
                class_codes = (self._y_codes if self.family == "multinomial"
                               else self.y.astype(int))
                number_classes = self._K if self.family == "multinomial" else 2
                for fold in range(nfolds):
                    present = np.bincount(
                        class_codes[foldid != fold],
                        minlength=number_classes)
                    if np.any(present == 0):
                        missing_indices = np.flatnonzero(present == 0)
                        missing = (self._mn_levels[missing_indices].tolist()
                                   if self.family == "multinomial"
                                   else self._binomial_levels[
                                       missing_indices].tolist())
                        raise ValueError(
                            f'Every {self.family} training fold must retain '
                            f'all classes; fold {fold} is missing {missing}.')

        losses = np.full((nfolds, self.nlambda), np.nan)
        common_nlambda = self.nlambda

        def fit_fold(fold, loss_row=None):
            """Fit and score one fold without writing shared parallel state."""
            if loss_row is None:
                loss_row = np.full(self.nlambda, np.nan)
            test_idx = np.where(foldid == fold)[0]
            train_idx = np.where(foldid != fold)[0]
            x_tr = self._x_orig[train_idx]
            if self.family == "multinomial":
                # Fit with full-data integer codes. Since every class is in
                # every fold, the fold solver's class axis is exactly 0..K-1.
                y_tr = self._y_codes[train_idx]
            else:
                y_tr = self.y[train_idx]
            x_te = self._x_orig[test_idx]
            y_te = (self._y_codes[test_idx] if self.family == "multinomial"
                    else self.y[test_idx])

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
                lla_max_stages=self.lla_max_stages,
                fast_mode=self.fast_mode,
                verbose=False,
            )
            if self.family in ("binomial", "poisson"):
                offset_tr = (self._offset[train_idx]
                             if self._offset_supplied else None)
                kw['offset'] = offset_tr

            fold_solver = Solver(x_tr, y_tr, **kw)
            fold_solver.train()

            fl = fold_solver.nlambda
            fold_status = (
                fold_solver.result['state'],
                bool(fold_solver.result.get('path_early_stopped', False)),
                fold_solver.result.get('status', 'completed'),
            )
            if (fold_status[0] == 'partially trained' or
                    (fl != self.nlambda and not fold_status[1])):
                return loss_row, fl, fold_status

            beta_f = fold_solver.result['beta']
            intcpt_f = fold_solver.result['intercept']

            score_size = min(fl, self.nlambda)
            if self.family == "multinomial":
                for li in range(score_size):
                    lp = x_te @ beta_f[li].T + intcpt_f[li]
                    if not np.all(np.isfinite(lp)):
                        raise ValueError('Multinomial logits must be finite.')
                    if type_measure == "class":
                        loss_val = np.mean(np.argmax(lp, axis=1) != y_te)
                    else:
                        loss_val = _multinomial_nll_from_logits(y_te, lp)
                    loss_row[li] = loss_val
            else:
                score_offset = None
                if (self.family in ("binomial", "poisson") and
                        self._offset_supplied):
                    score_offset = self._offset[test_idx]
                loss_row[:score_size] = _scalar_cv_fold_losses(
                    y_te, x_te, beta_f, intcpt_f, self.family,
                    type_measure, offset=score_offset,
                    path_size=score_size)

            return loss_row, fl, fold_status

        def record_fold(fold, fold_result, copy_loss):
            """Commit one ordered fold result and preserve serial errors."""
            nonlocal common_nlambda
            loss_row, fl, fold_status = fold_result
            fold_state, path_early_stopped, status = fold_status
            if fold_state == 'partially trained':
                raise PycassoError(
                    f'{self.family} cross-validation fold {fold} returned a '
                    'partially trained path; relax the convergence settings '
                    'before cross-validation.')
            if fl != self.nlambda:
                if not path_early_stopped:
                    raise PycassoError(
                        f'{self.family} cross-validation requires every fold '
                        'to cover a usable common lambda path; fold '
                        f'{fold} covered {fl}/{self.nlambda} lambdas '
                        f'(status={status!r}).')
                common_nlambda = min(common_nlambda, fl)
            # Serial workers receive this row as their output view, avoiding
            # a per-fold allocation and preserving the old default path.
            if copy_loss:
                losses[fold] = loss_row

        if n_jobs == 1:
            for fold in range(nfolds):
                record_fold(fold, fit_fold(fold, losses[fold]), False)
        else:
            # ctypes releases the GIL during native fitting. Threads avoid
            # serializing design matrices and keep exception order identical
            # to fold order through Executor.map's ordered iterator.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                    max_workers=min(n_jobs, nfolds)) as executor:
                fold_results = executor.map(fit_fold, range(nfolds))
                for fold, fold_result in enumerate(fold_results):
                    record_fold(fold, fold_result, True)

        # A supported successful early stop produces a prefix of the same
        # ordered path. Evaluate CV only on that certified common prefix.
        if common_nlambda < self.nlambda:
            losses = losses[:, :common_nlambda]
        # CV output must not expose a writable view of the solver's fitted
        # lambda path.
        cv_lambdas = self.lambdas[:common_nlambda].copy()

        finite_counts = np.sum(np.isfinite(losses), axis=0)
        if np.any(finite_counts != nfolds):
            first = int(np.flatnonzero(finite_counts != nfolds)[0])
            raise PycassoError(
                'Cross-validation covered only '
                f'{int(finite_counts[first])}/{nfolds} folds at lambda '
                f'index {first}.')
        if np.any(finite_counts == 0):
            missing = np.flatnonzero(finite_counts == 0).tolist()
            raise PycassoError(
                'Cross-validation produced no finite loss for lambda '
                f'index/indices {missing}.')

        cvm = np.zeros(common_nlambda)
        cvsd = np.zeros(common_nlambda)
        for li in range(common_nlambda):
            finite_losses = losses[np.isfinite(losses[:, li]), li]
            cvm[li] = np.mean(finite_losses)
            if finite_losses.size > 1:
                cvsd[li] = (np.std(finite_losses, ddof=1) /
                            np.sqrt(finite_losses.size))

        best_idx = int(np.argmin(cvm))
        lambda_min = cv_lambdas[best_idx]
        threshold = cvm[best_idx] + cvsd[best_idx]
        lse_candidates = np.where(
            (cvm <= threshold) & (cv_lambdas >= lambda_min))[0]
        lambda_1se = (cv_lambdas[lse_candidates[0]]
                      if len(lse_candidates) > 0 else lambda_min)

        nzero = (self.result['df'][:common_nlambda].copy()
                 if self.result['state'] in ('trained', 'partially trained')
                 else np.zeros(common_nlambda, dtype=int))

        return {
            'lambda': cv_lambdas,
            'cvm': cvm,
            'cvsd': cvsd,
            'cvup': cvm + cvsd,
            'cvlo': cvm - cvsd,
            'nzero': nzero,
            'lambda_min': lambda_min,
            'lambda_1se': lambda_1se,
            'foldid': foldid,
            'name': measure_name,
            'fast_mode': self.fast_mode,
            'precision': self.prec,
        }

    def plot(self, log_scale=True, max_features=None, ax=None):
        """Visualize the solution path.

        Multinomial paths display the sum of absolute class-specific
        coefficients for each feature.

        :param log_scale: Plot log-lambda on the x-axis. Default ``True``.
            A path containing zero automatically uses a finite linear-lambda
            axis because ``log(0)`` is undefined.
        :param max_features: Optional count of features with largest maximum
            absolute path coefficient to display.
        :param ax: Optional Matplotlib axes. If omitted, create and show a new
            figure.
        :return: ``None``.
        """
        if self.result['state'] not in ('trained', 'partially trained'):
            raise PycassoError("The model must be trained before plotting.")
        log_scale = _boolean_scalar(log_scale, 'log_scale')
        if max_features is not None:
            max_features = _positive_integer(
                max_features, 'max_features')
            if max_features > self.num_feature:
                raise ValueError(
                    '"max_features" cannot exceed the fitted feature count '
                    f'({self.num_feature}).')

        show = ax is None
        if show:
            import matplotlib.pyplot as plt
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

        x_axis, x_label = _plot_lambda_values(self.lambdas, log_scale)
        ax.plot(x_axis, beta_display)
        ax.set_ylabel('Coefficient')
        ax.set_xlabel(x_label)
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
        if self.result['state'] in ('trained', 'partially trained'):
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
                    lines.append(f"{i:>5}  {lam_str:>10}  {df_str:>5}  {dr_str:>10}")
                else:
                    lines.append(f"{i:>5}  {lam_str:>10}  {df_str:>5}")
        return "\n".join(lines) + "\n"
