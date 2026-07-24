import numpy as np
import sys
import os
import warnings
import contextlib
import concurrent.futures
import io
import builtins
import ctypes
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import pycasso
import pycasso.core as pycasso_core
from pycasso.core import PycassoError

assert pycasso.PycassoError is PycassoError, \
    "PycassoError is not exported from the top-level package"

np.random.seed(42)
n, d = 200, 50
X = np.random.randn(n, d)
b = np.array([1.5]*5 + [0]*(d-5))
Y_g = X @ b + np.random.randn(n)
Y_b = (np.random.rand(n) < 1/(1+np.exp(-X @ b))).astype(float)
# Y_p generated with offset so that true model is log(mu) = log(exposure) + X@b
_exposure = np.random.poisson(5, n) + 1
Y_p = np.random.poisson(_exposure * np.exp(X[:,:3] @ [0.4,-0.3,0.2])).astype(float)

# Step 1: dev_ratio
print("=== Step 1: dev_ratio ===")
s = pycasso.Solver(X, Y_g)
s.train()
assert 'dev_ratio' in s.result, "dev_ratio missing"
assert 'nulldev' in s.result, "nulldev missing"
assert np.all(s.result['dev_ratio'] >= 0), "dev_ratio negative"
assert np.all(s.result['dev_ratio'] <= 1), "dev_ratio > 1"
print(f"  gaussian dev_ratio range: [{s.result['dev_ratio'].min():.3f}, {s.result['dev_ratio'].max():.3f}]")
print("  PASS")

# Step 1b: fast-mode precision preset
print("\n=== Step 1b: fast mode ===")
fast_path = np.array([0.24, 0.14, 0.08], dtype=float)
high_solver = pycasso.Solver(X, Y_g, lambdas=fast_path)
assert high_solver.fast_mode is False and high_solver.prec == 1e-7, \
    "fast mode must default to high accuracy"
assert high_solver.result['fast_mode'] is False and \
    high_solver.result['precision'] == 1e-7, \
    "high-accuracy metadata is missing"

fast_solver = pycasso.Solver(
    X, Y_g, lambdas=fast_path, fast_mode=True)
fast_reference = pycasso.Solver(
    X, Y_g, lambdas=fast_path, prec=1e-7)
assert fast_solver.fast_mode is True and fast_solver.prec == 1e-7, \
    "Gaussian fast mode did not retain its glmnet-aligned prec=1e-7"
assert fast_solver.result['fast_mode'] is True and \
    fast_solver.result['precision'] == 1e-7, \
    "fast-mode metadata is missing"
fast_solver.train()
fast_reference.train()
assert np.array_equal(
    fast_solver.result['beta'], fast_reference.result['beta']), \
    "Gaussian fast mode differs from an explicit prec=1e-7 fit"
assert np.array_equal(
    fast_solver.result['intercept'], fast_reference.result['intercept']), \
    "Gaussian fast-mode intercept differs from explicit prec=1e-7"

custom_solver = pycasso.Solver(
    X, Y_g, lambdas=fast_path, prec=1e-6)
assert custom_solver.fast_mode is False and custom_solver.prec == 1e-6, \
    "high-accuracy mode did not preserve a custom precision"
for invalid_fast_mode in (1, 0.0, "yes", [True]):
    try:
        pycasso.Solver(X, Y_g, fast_mode=invalid_fast_mode)
        assert False, f"invalid fast_mode accepted: {invalid_fast_mode!r}"
    except ValueError as exc:
        assert "fast_mode" in str(exc)

for boolean_name in ('useintercept', 'standardize', 'verbose'):
    for invalid_boolean in (1, 0.0, "False", [True]):
        try:
            pycasso.Solver(
                X, Y_g, **{boolean_name: invalid_boolean})
            assert False, \
                f"invalid {boolean_name} accepted: {invalid_boolean!r}"
        except ValueError as exc:
            assert boolean_name in str(exc)
    boolean_solver = pycasso.Solver(
        X, Y_g, **{boolean_name: np.bool_(True)})
    expected_attribute = {
        'useintercept': 'use_intercept',
        'standardize': 'standardize',
        'verbose': 'verbose',
    }[boolean_name]
    assert getattr(boolean_solver, expected_attribute) is True, \
        f"{boolean_name} rejected np.bool_"
try:
    pycasso.Solver(X, Y_g, fast_mode=True, prec=1e-6)
    assert False, "fast mode accepted a conflicting custom precision"
except ValueError as exc:
    assert "fixes prec" in str(exc)
try:
    pycasso.Solver(X, Y_g, fast_mode=True, prec=1e-4)
    assert False, "Gaussian fast mode accepted the Newton/IRLS preset"
except ValueError as exc:
    assert 'family="gaussian"' in str(exc)
float32_fast = pycasso.Solver(
    X, Y_b, lambdas=fast_path, family="binomial",
    fast_mode=True, prec=np.float32(1e-4))
assert float32_fast.prec == 1e-4, \
    "fast mode rejected a numerically equivalent precision preset"
poisson_float32_fast = pycasso.Solver(
    X, Y_p, lambdas=fast_path, family="poisson",
    fast_mode=True, prec=np.float32(4e-4))
assert poisson_float32_fast.prec == 4e-4, \
    "Poisson fast mode rejected its calibrated precision preset"
try:
    pycasso.Solver(
        X, Y_p, family="poisson", fast_mode=True, prec=1e-4)
    assert False, "Poisson fast mode accepted the old precision preset"
except ValueError as exc:
    assert 'family="poisson"' in str(exc)

fixed_foldid = np.arange(n) % 3
fast_cv = fast_solver.cross_validate(
    foldid=fixed_foldid, type_measure="deviance")
reference_cv = fast_reference.cross_validate(
    foldid=fixed_foldid, type_measure="deviance")
assert fast_cv['fast_mode'] is True and fast_cv['precision'] == 1e-7, \
    "cross-validation lost fast-mode metadata"
assert np.array_equal(fast_cv['cvm'], reference_cv['cvm']) and \
    np.array_equal(fast_cv['cvsd'], reference_cv['cvsd']), \
    "cross-validation did not propagate fast mode to its folds"

for family_name, response in (
        ("binomial", Y_b), ("poisson", Y_p), ("sqrtlasso", Y_g)):
    family_precision = 4e-4 if family_name == "poisson" else 1e-4
    family_solver = pycasso.Solver(
        X, response, lambdas=fast_path, family=family_name,
        fast_mode=True)
    family_reference = pycasso.Solver(
        X, response, lambdas=fast_path, family=family_name,
        prec=family_precision)
    family_solver.train()
    family_reference.train()
    assert (family_solver.fast_mode is True and
            family_solver.prec == family_precision), \
        f"{family_name} did not inherit fast mode"
    assert np.array_equal(
        family_solver.result['beta'], family_reference.result['beta']) and \
        np.array_equal(
            family_solver.result['intercept'],
            family_reference.result['intercept']), \
        f"{family_name} fast mode differs from its explicit precision"

fast_binomial_cv = pycasso.Solver(
    X, Y_b, lambdas=fast_path, family="binomial", fast_mode=True)
reference_binomial_cv = pycasso.Solver(
    X, Y_b, lambdas=fast_path, family="binomial", prec=1e-4)
fast_binomial_result = fast_binomial_cv.cross_validate(
    foldid=fixed_foldid, type_measure="class")
reference_binomial_result = reference_binomial_cv.cross_validate(
    foldid=fixed_foldid, type_measure="class")
assert np.array_equal(
    fast_binomial_result['cvm'], reference_binomial_result['cvm']) and \
    np.array_equal(
        fast_binomial_result['cvsd'], reference_binomial_result['cvsd']), \
    "binomial CV did not propagate fast mode"
print("  PASS")

# Gaussian auto selection uses a benchmark-calibrated, memory-bounded policy.
print("\n=== Step 1c: Gaussian automatic backend ===")
path_05 = np.geomspace(1.0, 0.05, 8)
path_03 = np.geomspace(1.0, 0.03, 8)
path_20 = np.geomspace(1.0, 0.20, 8)
resolve_gaussian = pycasso_core._resolve_gaussian_type
assert resolve_gaussian('auto', 120, 120, path_03) == 'covariance'
assert resolve_gaussian('auto', 250, 250, path_20) == 'covariance'
assert resolve_gaussian('auto', 1000, 250, path_05) == 'covariance'
assert resolve_gaussian('auto', 2000, 250, path_03) == 'naive'
assert resolve_gaussian('auto', 4000, 250, path_03) == 'covariance'
assert resolve_gaussian('auto', 10000, 1025, path_20) == 'naive'
assert resolve_gaussian('auto', 100, 0, path_20) == 'naive'
assert resolve_gaussian('auto', 1000, 100, path_05[:7]) == 'naive'
assert resolve_gaussian('naive', 4000, 250, path_03) == 'naive'
assert resolve_gaussian('covariance', 10, 250, path_03) == 'covariance'

# Gaussian V3 must never present an iteration-limited lambda as converged.
iteration_x = np.array([
    [1.0, 1.1, 0.9], [2.0, 2.1, 1.8], [3.0, 3.2, 2.7],
    [4.0, 4.1, 3.7], [-1.0, -1.2, -0.8], [-2.0, -2.1, -1.7],
    [-3.0, -3.1, -2.8], [-4.0, -4.2, -3.6],
])
iteration_y = np.array([3.0, 5.9, 9.2, 12.1, -3.2, -6.1, -9.0, -12.2])
for gaussian_backend in ('naive', 'covariance'):
    limited = pycasso.Solver(
        iteration_x, iteration_y, lambdas=np.array([50.0, 0.05]),
        family='gaussian', type_gaussian=gaussian_backend,
        standardize=False, useintercept=False, max_ite=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        limited.train()
    assert len(caught) == 1 and issubclass(
        caught[0].category, RuntimeWarning), \
        f'{gaussian_backend} did not expose its partial Gaussian path'
    assert (limited.nlambda == 1 and
            limited.result['state'] == 'partially trained' and
            limited.result['status'] == 'inner_iteration_limit' and
            limited.result['status_code'] == 4 and
            limited.result['failed_lambda'] == 1), \
        f'{gaussian_backend} lost Gaussian iteration-limit metadata'
    assert np.array_equal(limited.result['beta'], np.zeros((1, 3))), \
        f'{gaussian_backend} changed the retained null prefix'

    zero_prefix = pycasso.Solver(
        iteration_x, iteration_y, lambdas=np.array([0.05]),
        family='gaussian', type_gaussian=gaussian_backend,
        standardize=False, useintercept=False, max_ite=1)
    try:
        zero_prefix.train()
        raise AssertionError(
            f'{gaussian_backend} accepted a zero-prefix iteration failure')
    except PycassoError as exc:
        assert 'inner_iteration_limit' in str(exc) and \
            'failed_lambda=0' in str(exc)
    assert zero_prefix.result['state'] == 'not trained'

    completed = pycasso.Solver(
        iteration_x, iteration_y, lambdas=np.array([50.0, 0.05]),
        family='gaussian', type_gaussian=gaussian_backend,
        standardize=False, useintercept=False, max_ite=1000)
    completed.train()
    assert (completed.nlambda == 2 and
            completed.result['status'] == 'completed' and
            completed.result['status_code'] == 0 and
            completed.result['failed_lambda'] == -1), \
        f'{gaussian_backend} rejected a converged Gaussian path'

auto_rng = np.random.default_rng(20260718)
auto_x = auto_rng.normal(size=(80, 8))
auto_y = (0.6 + auto_x[:, 0] - 0.4 * auto_x[:, 1] +
          auto_rng.normal(size=80))
automatic_gaussian = pycasso.Solver(
    auto_x, auto_y, lambdas=(8, 0.05), family='gaussian')
explicit_covariance = pycasso.Solver(
    auto_x, auto_y, lambdas=(8, 0.05), family='gaussian',
    type_gaussian='covariance')
assert automatic_gaussian.type_gaussian_requested == 'auto'
assert automatic_gaussian.type_gaussian == 'covariance'
automatic_gaussian.train()
explicit_covariance.train()
assert np.array_equal(automatic_gaussian.result['beta'],
                      explicit_covariance.result['beta'])
assert np.array_equal(automatic_gaussian.result['intercept'],
                      explicit_covariance.result['intercept'])
for nonconvex_penalty in ('mcp', 'scad'):
    automatic_nonconvex = pycasso.Solver(
        auto_x, auto_y, lambdas=(8, 0.05), family='gaussian',
        penalty=nonconvex_penalty)
    explicit_nonconvex = pycasso.Solver(
        auto_x, auto_y, lambdas=(8, 0.05), family='gaussian',
        penalty=nonconvex_penalty, type_gaussian='covariance')
    automatic_nonconvex.train()
    explicit_nonconvex.train()
    assert np.array_equal(automatic_nonconvex.result['beta'],
                          explicit_nonconvex.result['beta'])
    assert np.array_equal(automatic_nonconvex.result['intercept'],
                          explicit_nonconvex.result['intercept'])
short_gaussian = pycasso.Solver(
    auto_x, auto_y, lambdas=(7, 0.05), family='gaussian')
assert short_gaussian.type_gaussian == 'naive'

# A two-fold training split falls below n/d=4, so matching the explicit
# covariance result verifies that CV propagates the full-data decision.
cv_x = auto_rng.normal(size=(644, 161))
cv_y = cv_x[:, 0] - 0.5 * cv_x[:, 1] + auto_rng.normal(size=644)
cv_foldid = np.arange(644) % 2
auto_gaussian_cv = pycasso.Solver(
    cv_x, cv_y, lambdas=(8, 0.05), family='gaussian')
explicit_gaussian_cv = pycasso.Solver(
    cv_x, cv_y, lambdas=(8, 0.05), family='gaussian',
    type_gaussian='covariance')
assert auto_gaussian_cv.type_gaussian == 'covariance'
auto_gaussian_cv_result = auto_gaussian_cv.cross_validate(
    foldid=cv_foldid, type_measure='deviance')
explicit_gaussian_cv_result = explicit_gaussian_cv.cross_validate(
    foldid=cv_foldid, type_measure='deviance')
assert np.array_equal(auto_gaussian_cv_result['cvm'],
                      explicit_gaussian_cv_result['cvm'])
assert np.array_equal(auto_gaussian_cv_result['cvsd'],
                      explicit_gaussian_cv_result['cvsd'])
try:
    pycasso.Solver(
        auto_x, auto_y, lambdas=(8, 0.05), family='gaussian',
        type_gaussian='other')
    assert False, "invalid Gaussian backend should fail"
except ValueError as exc:
    assert 'auto' in str(exc) and 'covariance' in str(exc)
print("  PASS")

# Modern native ABIs return the already-evaluated training loss. Training must
# not replay every model through X merely to construct deviance ratios.
print("\n=== Step 1d: native training loss ===")
native_mn_y = (np.arange(n) + (X[:, 0] > 0).astype(int)) % 3
native_loss_cases = [
    ("gaussian", Y_g, {}),
    ("gaussian", Y_g, {"type_gaussian": "covariance"}),
    ("binomial", Y_b, {}),
    ("binomial", Y_b, {"useintercept": False}),
    ("poisson", Y_p, {"offset": np.log(_exposure)}),
    ("sqrtlasso", Y_g, {}),
    ("multinomial", native_mn_y, {}),
]
old_fit_deviances = pycasso_core._fit_deviances
old_mn_fit_deviances = pycasso_core._mn_fit_deviances


def forbid_training_loss_replay(*args, **kwargs):
    raise AssertionError("modern native loss ABI replayed the training path")


native_loss_solvers = []
try:
    pycasso_core._fit_deviances = forbid_training_loss_replay
    pycasso_core._mn_fit_deviances = forbid_training_loss_replay
    for native_family, native_y, native_kwargs in native_loss_cases:
        native_solver = pycasso.Solver(
            X, native_y, lambdas=fast_path, family=native_family,
            standardize=True, **native_kwargs)
        native_solver.train()
        native_loss_solvers.append(
            (native_family, native_y, native_kwargs, native_solver))
finally:
    pycasso_core._fit_deviances = old_fit_deviances
    pycasso_core._mn_fit_deviances = old_mn_fit_deviances

for native_family, native_y, native_kwargs, native_solver in native_loss_solvers:
    native_result = native_solver.result
    if native_family == "multinomial":
        assert np.all(np.isfinite(native_result["smooth_nll"])), \
            "multinomial V5 smooth NLL is missing or non-finite"
        replay_deviance = old_mn_fit_deviances(
            native_solver._y_codes, native_solver._x_orig,
            native_result["beta"], native_result["intercept"])
    else:
        assert np.all(np.isfinite(native_result["smooth_objective"])), \
            f"{native_family} native smooth objective is missing or non-finite"
        replay_offset = native_kwargs.get("offset")
        replay_deviance = old_fit_deviances(
            native_y, native_solver._x_orig, native_result["beta"],
            native_result["intercept"], native_family,
            offset=replay_offset)
    null_deviance = native_result["nulldev"]
    expected_ratio = (np.clip(1.0 - replay_deviance / null_deviance, 0, 1)
                      if null_deviance > 0 else
                      np.zeros(native_solver.nlambda))
    assert np.allclose(
        native_result["dev_ratio"], expected_ratio, rtol=0, atol=5e-12), \
        f"{native_family} native-loss deviance ratio differs from replay"
print("  PASS")

# Responses, labels, and offsets are construction-time inputs.  Mutating the
# caller's arrays later must not change native training or default assessment.
print("\n=== Step 1e: response and offset ownership ===")
ownership_rng = np.random.default_rng(20260719)
ownership_x = ownership_rng.normal(size=(24, 4))
ownership_offset = np.linspace(-0.25, 0.2, ownership_x.shape[0])
ownership_responses = {
    'gaussian': (0.4 + 0.7 * ownership_x[:, 0] -
                 0.2 * ownership_x[:, 1]),
    'binomial': (ownership_x[:, 0] + 0.2 * ownership_x[:, 1] > 0).astype(float),
    'poisson': (1 + np.arange(ownership_x.shape[0]) % 4).astype(float),
    'sqrtlasso': (-0.3 + 0.5 * ownership_x[:, 2] +
                  0.05 * np.arange(ownership_x.shape[0])),
}

for ownership_family, ownership_response in ownership_responses.items():
    response_input = ownership_response.copy()
    response_snapshot = ownership_response.copy()
    constructor_args = {}
    reference_args = {}
    offset_input = None
    if ownership_family in ('binomial', 'poisson'):
        offset_input = ownership_offset.copy()
        constructor_args['offset'] = offset_input
        reference_args['offset'] = ownership_offset.copy()

    ownership_solver = pycasso.Solver(
        ownership_x, response_input, lambdas=(2, 0.5),
        family=ownership_family, max_ite=300, **constructor_args)
    ownership_reference = pycasso.Solver(
        ownership_x, response_snapshot, lambdas=(2, 0.5),
        family=ownership_family, max_ite=300, **reference_args)
    assert ownership_solver.y.flags.owndata and \
        not np.shares_memory(ownership_solver.y, response_input), \
        f"{ownership_family} retained the caller's response storage"
    if offset_input is not None:
        assert ownership_solver._offset.flags.owndata and \
            not np.shares_memory(ownership_solver._offset, offset_input), \
            f"{ownership_family} retained the caller's offset storage"

    response_input.fill(-1000.0)
    if offset_input is not None:
        offset_input.fill(1000.0)
    ownership_solver.train()
    ownership_reference.train()
    assert np.allclose(
        ownership_solver.result['beta'],
        ownership_reference.result['beta'], rtol=1e-12, atol=1e-13) and \
        np.allclose(
            ownership_solver.result['intercept'],
            ownership_reference.result['intercept'],
            rtol=1e-12, atol=1e-13), \
        f"{ownership_family} training observed caller mutation"

    assessment_args = {}
    if ownership_family in ('binomial', 'poisson'):
        assessment_args['newoffset'] = ownership_offset
    ownership_assessment = ownership_solver.assess(**assessment_args)
    reference_assessment = ownership_reference.assess(**assessment_args)
    for metric in ownership_assessment:
        assert np.allclose(
            ownership_assessment[metric], reference_assessment[metric],
            rtol=1e-12, atol=1e-13), \
            f"{ownership_family} assessment observed caller mutation in {metric}"

label_input = np.array(['red', 'green', 'blue'] * 8)
label_snapshot = label_input.copy()
label_solver = pycasso.Solver(
    ownership_x, label_input, lambdas=np.array([0.2]),
    family='multinomial', max_ite=300)
label_reference = pycasso.Solver(
    ownership_x, label_snapshot, lambdas=np.array([0.2]),
    family='multinomial', max_ite=300)
assert label_solver.y.flags.owndata and \
    not np.shares_memory(label_solver.y, label_input), \
    "multinomial retained the caller's label storage"
label_input.fill('other')
label_solver.train()
label_reference.train()
assert np.allclose(
    label_solver.result['beta'], label_reference.result['beta'],
    rtol=1e-12, atol=1e-13) and \
    np.allclose(
        label_solver.result['intercept'], label_reference.result['intercept'],
        rtol=1e-12, atol=1e-13), \
    "multinomial training observed caller label mutation"
label_assessment = label_solver.assess()
label_reference_assessment = label_reference.assess()
for metric in label_assessment:
    assert np.allclose(
        label_assessment[metric], label_reference_assessment[metric],
        rtol=1e-12, atol=1e-13), \
        f"multinomial assessment observed caller label mutation in {metric}"

# The design is read during train(), default prediction/assessment, and CV.
# Solver must therefore own it just as it owns responses and offsets.
for ownership_standardize in (False, True):
    design_input = np.ascontiguousarray(ownership_x.copy())
    design_snapshot = design_input.copy()
    design_path = np.array([0.5, 0.25])
    design_solver = pycasso.Solver(
        design_input, ownership_responses['gaussian'], lambdas=design_path,
        family='gaussian', standardize=ownership_standardize,
        type_gaussian='naive', max_ite=300)
    design_reference = pycasso.Solver(
        design_snapshot, ownership_responses['gaussian'], lambdas=design_path,
        family='gaussian', standardize=ownership_standardize,
        type_gaussian='naive', max_ite=300)
    assert design_solver._x_orig.flags.owndata and \
        not np.shares_memory(design_solver._x_orig, design_input), \
        f"standardize={ownership_standardize} retained caller design storage"
    assert not np.shares_memory(design_solver.x, design_input), \
        f"standardize={ownership_standardize} native design aliases caller storage"

    design_input.fill(1000.0)
    design_solver.train()
    design_reference.train()
    assert np.allclose(
        design_solver.result['beta'], design_reference.result['beta'],
        rtol=1e-12, atol=1e-13) and np.allclose(
            design_solver.result['intercept'],
            design_reference.result['intercept'],
            rtol=1e-12, atol=1e-13), \
        f"standardize={ownership_standardize} training observed design mutation"
    assert np.allclose(
        design_solver.predict(), design_reference.predict(),
        rtol=1e-12, atol=1e-13), \
        f"standardize={ownership_standardize} prediction observed design mutation"
    design_assessment = design_solver.assess()
    reference_design_assessment = design_reference.assess()
    for metric in design_assessment:
        assert np.allclose(
            design_assessment[metric], reference_design_assessment[metric],
            rtol=1e-12, atol=1e-13), \
            (f"standardize={ownership_standardize} assessment observed "
             f"design mutation in {metric}")
    foldid = np.arange(design_input.shape[0]) % 3
    design_cv = design_solver.cross_validate(foldid=foldid)
    reference_design_cv = design_reference.cross_validate(foldid=foldid)
    assert np.allclose(
        design_cv['cvm'], reference_design_cv['cvm'],
        rtol=1e-12, atol=1e-13), \
        f"standardize={ownership_standardize} CV observed design mutation"

# Default prediction and assessment reuse the owned, constructor-validated
# design. They must not rescan it, while every explicit matrix retains the
# public finite-value validation contract, even if it is the same object.
default_data_x = np.ascontiguousarray(X[:30, :6])
default_data_signal = default_data_x @ np.array(
    [0.5, -0.35, 0.2, 0.0, 0.0, 0.0])
default_data_responses = {
    'gaussian': default_data_signal + np.linspace(-0.1, 0.1, 30),
    'sqrtlasso': Y_g[:30].copy(),
    'binomial': (np.arange(30) % 2).astype(float),
    'poisson': (1 + np.arange(30) % 4).astype(float),
    'multinomial': (np.arange(30) % 3).astype(float),
}
default_data_solvers = {}
for default_data_family, default_data_y in default_data_responses.items():
    caller_design = default_data_x.copy()
    default_data_solver = pycasso.Solver(
        caller_design, default_data_y, family=default_data_family,
        lambdas=np.array([0.4, 0.2]), standardize=False, max_ite=300)
    assert default_data_solver._x_orig.flags.owndata and not np.shares_memory(
        default_data_solver._x_orig, caller_design), \
        f"{default_data_family} default-data probe did not own its design"
    caller_design.fill(np.nan)
    default_data_solver.train()
    default_prediction = default_data_solver.predict()
    explicit_prediction = default_data_solver.predict(
        default_data_solver._x_orig)
    assert np.array_equal(default_prediction, explicit_prediction), \
        (f"{default_data_family} default prediction differs from explicit "
         "training-data prediction")
    default_assessment = default_data_solver.assess()
    explicit_assessment = default_data_solver.assess(
        default_data_solver._x_orig, default_data_y)
    assert default_assessment.keys() == explicit_assessment.keys() and all(
        np.array_equal(default_assessment[key], explicit_assessment[key])
        for key in default_assessment), \
        (f"{default_data_family} default assessment differs from explicit "
         "training-data assessment")
    default_data_solvers[default_data_family] = default_data_solver

scan_probe_solver = default_data_solvers['gaussian']
scan_probe_events = []
original_isfinite = pycasso_core.np.isfinite


def tracking_isfinite(values, *args, **kwargs):
    if values is scan_probe_solver._x_orig:
        scan_probe_events.append(values)
    return original_isfinite(values, *args, **kwargs)


try:
    pycasso_core.np.isfinite = tracking_isfinite
    scan_probe_solver.predict()
    scan_probe_solver.assess()
    assert not scan_probe_events, \
        "default prediction or assessment rescanned the owned training design"
    scan_probe_solver.predict(scan_probe_solver._x_orig)
    assert len(scan_probe_events) == 1, \
        "explicit prediction skipped training-design finite validation"
    scan_probe_solver.assess(
        scan_probe_solver._x_orig,
        default_data_responses['gaussian'])
    assert len(scan_probe_events) == 2, \
        "explicit assessment skipped training-design finite validation"
finally:
    pycasso_core.np.isfinite = original_isfinite

for nan_family in ('gaussian', 'multinomial'):
    nan_solver = default_data_solvers[nan_family]
    nan_design = np.zeros((1, nan_solver.num_feature), dtype='double')
    nan_design[0, 0] = np.nan
    nan_response = np.array([
        default_data_responses[nan_family][0]])
    for nan_operation, nan_call in (
            ('prediction', lambda: nan_solver.predict(nan_design)),
            ('assessment', lambda: nan_solver.assess(
                nan_design, nan_response))):
        try:
            nan_call()
            assert False, \
                f"explicit NaN {nan_family} {nan_operation} was accepted"
        except ValueError as exc:
            assert "finite" in str(exc), \
                (f"explicit NaN {nan_family} {nan_operation} returned the "
                 f"wrong error: {exc}")
print("  PASS")

# Native ctypes calls release the GIL. Operations that replace a Solver's
# output buffers must therefore serialize per instance, while separate Solver
# instances must remain able to run concurrently.
print("\n=== Step 1f: per-Solver operation serialization ===")


def _run_together(callables):
    start = threading.Barrier(len(callables) + 1)

    def invoke(function):
        start.wait()
        return function()

    with ThreadPoolExecutor(max_workers=len(callables)) as executor:
        futures = [executor.submit(invoke, function)
                   for function in callables]
        start.wait()
        return [future.result(timeout=10) for future in futures]


counter_guard = threading.Lock()
shared_active = [0]
shared_peak = [0]


def _shared_probe_trainer():
    with counter_guard:
        shared_active[0] += 1
        shared_peak[0] = max(shared_peak[0], shared_active[0])
    time.sleep(0.04)
    with counter_guard:
        shared_active[0] -= 1


shared_operation_solver = pycasso.Solver(
    ownership_x, ownership_responses['gaussian'], lambdas=np.array([0.3]),
    standardize=False, type_gaussian='naive')
shared_operation_solver.trainer = _shared_probe_trainer
_run_together([
    shared_operation_solver.train,
    shared_operation_solver.train,
])
assert shared_peak[0] == 1, \
    "two train calls entered the same Solver operation concurrently"
assert shared_operation_solver.result['state'] == 'trained', \
    "serialized retraining did not retain a trained state"

# Verify that cross_validate() uses the same instance lock. An invalid request
# would return immediately without the decorator, so it is a cheap probe that
# does not construct fold solvers.
blocking_entered = threading.Event()
blocking_release = threading.Event()


def _blocking_trainer():
    blocking_entered.set()
    if not blocking_release.wait(timeout=5):
        raise AssertionError("operation-lock probe timed out")


blocking_solver = pycasso.Solver(
    ownership_x, ownership_responses['gaussian'], lambdas=np.array([0.3]),
    standardize=False, type_gaussian='naive')
blocking_solver.trainer = _blocking_trainer
with ThreadPoolExecutor(max_workers=2) as executor:
    train_future = executor.submit(blocking_solver.train)
    assert blocking_entered.wait(timeout=5), \
        "blocking trainer did not start"
    cv_started = threading.Event()

    def _invalid_cv_probe():
        cv_started.set()
        try:
            blocking_solver.cross_validate(type_measure='invalid')
        except ValueError:
            return 'rejected'
        raise AssertionError("invalid CV measure was accepted")

    cv_future = executor.submit(_invalid_cv_probe)
    assert cv_started.wait(timeout=5), "CV lock probe did not start"
    time.sleep(0.02)
    cv_was_blocked = not cv_future.done()
    blocking_release.set()
    train_future.result(timeout=5)
    assert cv_future.result(timeout=5) == 'rejected'
assert cv_was_blocked, \
    "cross_validate entered while train held the same Solver lock"

exception_solver = pycasso.Solver(
    ownership_x, ownership_responses['gaussian'], lambdas=np.array([0.3]),
    standardize=False, type_gaussian='naive')


def _raise_from_trainer():
    raise RuntimeError("intentional operation-lock probe")


exception_solver.trainer = _raise_from_trainer
try:
    exception_solver.train()
    assert False, "trainer exception was swallowed"
except RuntimeError as exc:
    assert "operation-lock probe" in str(exc)
exception_solver.trainer = lambda: None
exception_solver.train()
assert exception_solver.result['state'] == 'trained', \
    "trainer exception left the per-Solver operation lock held"

independent_active = [0]
independent_peak = [0]


def _independent_probe_trainer():
    with counter_guard:
        independent_active[0] += 1
        independent_peak[0] = max(
            independent_peak[0], independent_active[0])
    time.sleep(0.04)
    with counter_guard:
        independent_active[0] -= 1


independent_solvers = [
    pycasso.Solver(
        ownership_x, ownership_responses['gaussian'],
        lambdas=np.array([0.3]), standardize=False,
        type_gaussian='naive')
    for _ in range(2)
]
for independent_solver in independent_solvers:
    independent_solver.trainer = _independent_probe_trainer
_run_together([solver.train for solver in independent_solvers])
assert independent_peak[0] == 2, \
    "operation serialization unexpectedly used a process-wide lock"

# Exercise the real native buffers as well as the deterministic entry probe.
concurrent_path = np.array([0.5, 0.25])
concurrent_reference = pycasso.Solver(
    ownership_x, ownership_responses['gaussian'], lambdas=concurrent_path,
    standardize=False, type_gaussian='naive')
concurrent_reference.train()
concurrent_solver = pycasso.Solver(
    ownership_x, ownership_responses['gaussian'], lambdas=concurrent_path,
    standardize=False, type_gaussian='naive')
_run_together([concurrent_solver.train, concurrent_solver.train])
assert concurrent_solver.result['state'] == 'trained' and np.allclose(
    concurrent_solver.result['beta'], concurrent_reference.result['beta'],
    rtol=1e-12, atol=1e-13), \
    "serialized native retraining differs from the serial reference"
assert np.allclose(
    concurrent_solver.result['intercept'],
    concurrent_reference.result['intercept'], rtol=1e-12, atol=1e-13), \
    "serialized native intercepts differ from the serial reference"

mixed_solver = pycasso.Solver(
    ownership_x, ownership_responses['gaussian'], lambdas=concurrent_path,
    standardize=False, type_gaussian='naive')
mixed_foldid = np.arange(ownership_x.shape[0]) % 2
mixed_results = _run_together([
    mixed_solver.train,
    lambda: mixed_solver.cross_validate(
        foldid=mixed_foldid, type_measure='mse'),
])
mixed_reference = pycasso.Solver(
    ownership_x, ownership_responses['gaussian'], lambdas=concurrent_path,
    standardize=False, type_gaussian='naive')
mixed_reference.train()
mixed_cv_reference = mixed_reference.cross_validate(
    foldid=mixed_foldid, type_measure='mse')
assert mixed_solver.result['state'] == 'trained' and np.all(np.isfinite(
    mixed_results[1]['cvm'])), \
    "concurrent train/CV serialization did not produce a usable result"
assert np.allclose(
    mixed_results[1]['cvm'], mixed_cv_reference['cvm'],
    rtol=1e-12, atol=1e-13) and np.allclose(
        mixed_solver.result['beta'], mixed_reference.result['beta'],
        rtol=1e-12, atol=1e-13), \
    "serialized train/CV differs from the serial reference"
print("  PASS")

# Step 2: predict types
print("\n=== Step 2: predict types ===")
# gaussian link == response
p_link = s.predict(X[:5], type="link")
p_resp = s.predict(X[:5], type="response")
assert p_link.shape == (5,), f"link shape wrong: {p_link.shape}"
assert np.allclose(p_link, p_resp), "gaussian link != response"

# nonzero
nz = s.predict(X[:5], type="nonzero")
assert isinstance(nz, np.ndarray), f"nonzero not array: {type(nz)}"
saved_training_design = s._x_orig
try:
    # No observation is evaluated for support extraction.  An omitted
    # newdata must therefore avoid even a finite-value scan of training X.
    s._x_orig = np.full_like(saved_training_design, np.nan)
    default_nz = s.predict(type="nonzero")
    assert np.array_equal(default_nz, nz), \
        "default nonzero prediction unexpectedly evaluated training X"
finally:
    s._x_orig = saved_training_design
try:
    s.predict(np.full((1, d), np.nan), type="nonzero")
    assert False, "explicit nonzero newdata skipped finite-value validation"
except ValueError as exc:
    assert "finite" in str(exc)
try:
    s.predict(type="nonzero", newoffset=np.zeros(n))
    assert False, "Gaussian nonzero prediction accepted newoffset"
except ValueError as exc:
    assert "only supported" in str(exc)

# binomial
sb = pycasso.Solver(X, Y_b, family="binomial")
sb.train()
probs = sb.predict(X[:5], type="response")
assert probs.shape == (5,), f"binomial probs shape: {probs.shape}"
assert np.all(probs >= 0) and np.all(probs <= 1), "binomial probs out of range"
cls = sb.predict(X[:5], type="class")
assert set(cls).issubset({0, 1}), f"binomial class not binary: {set(cls)}"
link_b = sb.predict(X[:5], type="link")
assert link_b.shape == (5,), "binomial link shape"
prediction_shift = np.linspace(-0.4, 0.5, 5)
assert np.allclose(
    sb.predict(X[:5], type="link", newoffset=prediction_shift),
    link_b + prediction_shift), \
    "binomial newoffset was not added on the link scale"
assert np.allclose(
    sb.predict(X[:5], type="link", newoffset=np.zeros(5)), link_b), \
    "no-offset model should default to a zero prediction offset"

# lam= parameter
p_lam = s.predict(X[:5], lam=s.lambdas[10])
p_idx = s.predict(X[:5], lambdidx=10)
assert np.allclose(p_lam, p_idx), "lam= exact match should equal lambdidx="
p_lam_interp = s.predict(X[:5], lam=(s.lambdas[10]+s.lambdas[11])/2)  # prints note
assert p_lam_interp.shape == (5,), "interpolated prediction shape"

empty_x = np.empty((0, d))
for empty_type in ("response", "nonzero"):
    try:
        s.predict(empty_x, type=empty_type)
        assert False, f"predict accepted empty newdata for {empty_type}"
    except ValueError as exc:
        assert "at least one row" in str(exc)

# Support is a property of a fitted path point, not an interpolated numeric
# quantity. Lambda-value support queries therefore use the nearest fitted
# lambda (ties choose the earlier/larger lambda), matching the R interface.
support_solver = object.__new__(pycasso.Solver)
support_solver.family = "gaussian"
support_solver.lambdas = np.array([1.0, 0.5])
support_solver.nlambda = 2
support_solver.num_feature = 1
support_solver.result = {
    'state': 'trained',
    'beta': np.array([[0.0], [1.0]]),
    'intercept': np.zeros(2),
}
support_multinomial = object.__new__(pycasso.Solver)
support_multinomial.family = "multinomial"
support_multinomial.lambdas = support_solver.lambdas.copy()
support_multinomial.nlambda = 2
support_multinomial.num_feature = 1
support_multinomial._K = 3
support_multinomial.result = {
    'state': 'trained',
    'beta': np.array([
        [[0.0], [0.0], [0.0]],
        [[1.0], [0.0], [0.0]],
    ]),
    'intercept': np.zeros((2, 3)),
}
support_stdout = io.StringIO()
with contextlib.redirect_stdout(support_stdout):
    assert support_solver.predict(
        type="nonzero", lam=0.9, lambdidx=99).size == 0, \
        "nearest-lambda support query interpolated a false nonzero"
    assert np.array_equal(
        support_solver.predict(type="nonzero", lam=0.6), np.array([0])), \
        "nearest-lambda support query missed the lower path point"
    assert support_solver.predict(type="nonzero", lam=0.75).size == 0, \
        "equidistant support query did not choose the earlier path point"
    assert support_multinomial.predict(type="nonzero", lam=0.9) == \
        [[], [], []], \
        "multinomial support query interpolated class-specific support"
assert support_stdout.getvalue() == "", \
    "nearest-lambda support query printed an interpolation note"
for invalid_support_lam, expected_message in (
        (-0.1, "nonnegative"), (np.nan, "finite"), (np.inf, "finite")):
    try:
        support_solver.predict(type="nonzero", lam=invalid_support_lam)
        assert False, f"support query accepted lam={invalid_support_lam!r}"
    except ValueError as exc:
        assert expected_message in str(exc)
print("  PASS")

# Python binomial uses the same two-level categorical contract as R while the
# native solver continues to receive compact double 0/1 codes.
print("\n=== Step 2b: binomial class labels ===")
Y_b_string = np.where(Y_b == 0, "no", "yes")
caller_labels = Y_b_string.copy()
sb_string = pycasso.Solver(X, caller_labels, family="binomial")
sb_string_list = pycasso.Solver(
    X, Y_b_string.tolist(), lambdas=np.array([0.3, 0.15]),
    family="binomial")
assert np.array_equal(sb_string_list.y, Y_b) and np.array_equal(
    sb_string_list.result['levels'], np.array(["no", "yes"])), \
    "homogeneous Python string lists do not follow binomial label encoding"
caller_labels[:] = "changed"
assert np.array_equal(sb_string.result['levels'], np.array(["no", "yes"])), \
    "binomial levels were not retained"
assert np.array_equal(sb_string.y, Y_b), \
    "binomial labels were not encoded as 0/1"
assert not np.shares_memory(sb_string.y, caller_labels), \
    "binomial codes alias caller-owned labels"
sb_string.train()
assert np.array_equal(sb_string.lambdas, sb.lambdas), \
    "categorical binomial labels changed the generated lambda path"
assert np.allclose(sb_string.result['beta'], sb.result['beta'],
                   rtol=0, atol=0), \
    "categorical binomial labels changed fitted coefficients"
assert np.allclose(sb_string.result['intercept'], sb.result['intercept'],
                   rtol=0, atol=0), \
    "categorical binomial labels changed fitted intercepts"
string_classes = sb_string.predict(X[:12], type="class")
assert set(string_classes).issubset({0, 1}), \
    "binomial class prediction no longer returns encoded 0/1 values"
restored_classes = sb_string.result['levels'][string_classes]
assert set(restored_classes).issubset({"no", "yes"}), \
    "binomial result levels cannot restore class predictions"
string_assessment = sb_string.assess(X, Y_b_string)
encoded_assessment = sb_string.assess(X, Y_b)
for metric in ('deviance', 'class_error'):
    assert np.array_equal(string_assessment[metric],
                          encoded_assessment[metric]), \
        f"binomial {metric} differs for labels and encoded 0/1"
string_confusion = sb_string.confusion(X, Y_b_string, lambdidx=[0, 10])
encoded_confusion = sb_string.confusion(X, Y_b, lambdidx=[0, 10])
assert all(np.array_equal(left, right) for left, right in
           zip(string_confusion, encoded_confusion)), \
    "binomial confusion differs for labels and encoded 0/1"
try:
    sb_string.assess(X[:3], np.array(["no", "yes", "unknown"]))
    assert False, "binomial assessment accepted an unseen class"
except ValueError as exc:
    assert "unseen binomial" in str(exc)

numeric_labels = np.where(Y_b == 0, -2, 3)
sb_numeric_labels = pycasso.Solver(
    X, numeric_labels, lambdas=np.array([0.3, 0.15]), family="binomial")
assert np.array_equal(sb_numeric_labels.result['levels'], np.array([-2, 3]))
assert np.array_equal(sb_numeric_labels.y, Y_b)
bool_labels = Y_b.astype(bool)
sb_bool_labels = pycasso.Solver(
    X, bool_labels, lambdas=np.array([0.3, 0.15]), family="binomial")
assert np.array_equal(
    sb_bool_labels.result['levels'], np.array([False, True]))
assert np.array_equal(sb_bool_labels.y, Y_b)

for invalid_labels in (
        np.repeat("one", n),
        np.resize(np.array(["a", "b", "c"]), n),
        np.resize(np.array([0, "yes"], dtype=object), n),
        [0, "yes"] * (n // 2),
        ["no", np.nan] * (n // 2),
        ["no", np.inf] * (n // 2),
        ["no", 1j] * (n // 2),
        np.resize(np.array([b"no", b"yes"]), n)):
    try:
        pycasso.Solver(X, invalid_labels, family="binomial")
        assert False, "binomial accepted an invalid class map"
    except ValueError:
        pass

_, overlapping_original = pycasso_core._encode_binomial_labels(
    [1, 1], levels=np.array([1, 2]), name="newy")
_, overlapping_codes = pycasso_core._encode_binomial_labels(
    [0, 1], levels=np.array([1, 2]), name="newy")
assert np.array_equal(overlapping_original, np.array([0.0, 0.0])) and \
       np.array_equal(overlapping_codes, np.array([0.0, 1.0])), \
    "binomial original-label priority or encoded fallback changed"
try:
    pycasso_core._encode_multinomial_labels(["a", np.nan, "c"])
    assert False, "multinomial labels silently stringified a numeric NaN"
except ValueError as exc:
    assert "finite" in str(exc) or "missing" in str(exc)

# Public result metadata is mutable for compatibility, but retraining restores
# it from the private two-element class map.
sb_string.result['levels'][0] = "mutated"
sb_string.train()
assert np.array_equal(sb_string.result['levels'], np.array(["no", "yes"])), \
    "retraining did not restore binomial levels"
label_foldid = np.arange(n) % 5
string_cv = sb_string.cross_validate(
    foldid=label_foldid, type_measure="class")
encoded_cv = sb.cross_validate(
    foldid=label_foldid, type_measure="class")
for field in ('lambda', 'cvm', 'cvsd', 'nzero'):
    assert np.array_equal(string_cv[field], encoded_cv[field]), \
        f"binomial categorical and encoded CV differ in {field}"
assert string_cv['lambda_min'] == encoded_cv['lambda_min'] and \
       string_cv['lambda_1se'] == encoded_cv['lambda_1se'], \
    "binomial categorical and encoded CV selected different lambdas"
print("  PASS")

# Step 3: assess
print("\n=== Step 3: assess / confusion ===")
a = sb.assess(X, Y_b)
assert 'deviance' in a, "assess missing deviance"
assert 'class_error' in a, "assess missing class_error"
assert len(a['deviance']) == sb.nlambda, "assess deviance length"
assess_solver_lambdas = sb.lambdas.copy()
assert a['lambda'].flags.owndata and a['lambda'].base is None and \
    not np.shares_memory(a['lambda'], sb.lambdas), \
    "assessment lambda path is not an owning snapshot"
a['lambda'][0] += 1.0
assert np.array_equal(sb.lambdas, assess_solver_lambdas), \
    "mutating assessment lambda output changed solver state"
conf = sb.confusion(X, Y_b, lambdidx=[0, 5, 10])
assert len(conf) == 3, "confusion list length"
assert conf[0].shape == (2, 2), "confusion matrix shape"
try:
    sb.confusion(X, Y_b, lambdidx=[])
    assert False, "confusion accepted an empty lambda-index selection"
except ValueError as exc:
    assert "at least one" in str(exc)
try:
    s.assess(empty_x, np.empty(0))
    assert False, "assess accepted an empty evaluation set"
except ValueError as exc:
    assert "at least one row" in str(exc)
try:
    sb.confusion(empty_x, np.empty(0))
    assert False, "confusion accepted an empty evaluation set"
except ValueError as exc:
    assert "at least one row" in str(exc)
try:
    sb.assess(X[:3], np.array([0.0, 1.0, 2.0]))
    assert False, "binomial assessment should reject non-binary responses"
except ValueError as exc:
    assert "unseen binomial" in str(exc)
print("  PASS")

# Scalar path evaluation must preserve the former one-model-at-a-time
# formulas while splitting its only n-by-path temporary at the byte cap.
path_x = np.ascontiguousarray(X[:17, :7])
path_beta = np.ascontiguousarray(
    np.linspace(-0.18, 0.21, 5 * 7).reshape(5, 7))
path_intercept = np.linspace(-0.2, 0.25, 5)
path_offset = np.linspace(-0.3, 0.2, path_x.shape[0])
path_inputs_before = tuple(value.copy() for value in (
    path_x, path_beta, path_intercept, path_offset))
path_responses = {
    'gaussian': Y_g[:17],
    'sqrtlasso': Y_g[:17],
    'binomial': Y_b[:17],
    'poisson': Y_p[:17],
}
original_path_block_bytes = pycasso_core._SCALAR_PATH_BLOCK_BYTES
try:
    pycasso_core._SCALAR_PATH_BLOCK_BYTES = path_x.shape[0] * 8 * 2
    predictor_blocks = list(pycasso_core._scalar_linear_predictor_blocks(
        path_x, path_beta, path_intercept, offset=path_offset))
    assert [stop - start for start, stop, _ in predictor_blocks] == [2, 2, 1], \
        "scalar predictor path did not honor its working-memory cap"
    expected_eta = np.column_stack([
        path_x @ path_beta[index] + path_intercept[index] + path_offset
        for index in range(path_beta.shape[0])])
    assert np.allclose(np.vstack(
        [block for _, _, block in predictor_blocks]).T, expected_eta,
        rtol=1e-14, atol=1e-14), \
        "blocked scalar predictors changed the linear predictor"
    strided_x = path_x[:, ::2]
    strided_beta = path_beta[:, ::2]
    strided_eta = np.vstack([
        block for _, _, block in
        pycasso_core._scalar_linear_predictor_blocks(
            strided_x, strided_beta, path_intercept)]).T
    assert np.allclose(strided_eta, np.column_stack([
        strided_x @ strided_beta[index] + path_intercept[index]
        for index in range(strided_beta.shape[0])]),
        rtol=1e-14, atol=1e-14), \
        "blocked scalar predictors changed a non-contiguous design"

    for path_family, path_y in path_responses.items():
        family_offset = (path_offset if path_family in
                         ('binomial', 'poisson') else None)
        expected = {'deviance': [], 'mse': [], 'mae': [],
                    'class_error': []}
        for path_index in range(path_beta.shape[0]):
            eta = (path_x @ path_beta[path_index] +
                   path_intercept[path_index])
            if family_offset is not None:
                eta += family_offset
            if path_family in ('gaussian', 'sqrtlasso'):
                residual = path_y - eta
                expected['deviance'].append(np.mean(residual ** 2) / 2.0)
                expected['mse'].append(np.mean(residual ** 2))
                expected['mae'].append(np.mean(np.abs(residual)))
            elif path_family == 'binomial':
                probability = np.clip(
                    pycasso_core._sigmoid(eta), 1e-15, 1.0 - 1e-15)
                expected['deviance'].append(-np.mean(
                    path_y * np.log(probability) +
                    (1.0 - path_y) * np.log(1.0 - probability)))
                expected['class_error'].append(np.mean(
                    (eta > 0).astype(int) != path_y.astype(int)))
            else:
                mean = pycasso_core._poisson_mean(eta)
                expected['deviance'].append(
                    pycasso_core._poisson_dev(path_y, mean))
                expected['mse'].append(np.mean((path_y - mean) ** 2))

        actual = pycasso_core._scalar_path_metrics(
            path_y, path_x, path_beta, path_intercept, path_family,
            offset=family_offset, include_assessment=True)
        assert np.allclose(
            pycasso_core._fit_deviances(
                path_y, path_x, path_beta, path_intercept, path_family,
                offset=family_offset), expected['deviance'],
            rtol=1e-13, atol=1e-14), \
            f"{path_family} blocked deviance changed its scalar formula"
        for metric_name, metric_value in actual.items():
            assert np.allclose(
                metric_value, expected[metric_name],
                rtol=1e-13, atol=1e-14), \
                f"{path_family} blocked {metric_name} changed its result"

    boundary_metrics = pycasso_core._scalar_path_metrics(
        np.array([0.0, 1.0]), np.zeros((2, 1)), np.zeros((2, 1)),
        np.zeros(2), 'binomial', include_assessment=True)
    assert np.array_equal(boundary_metrics['class_error'], [0.5, 0.5]), \
        "blocked binomial assessment changed the strict eta > 0 boundary"
    extreme_binomial = pycasso_core._fit_deviances(
        np.array([0.0, 1.0]), np.array([[1.0], [-1.0]]),
        np.array([[1000.0], [1000.0]]), np.zeros(2), 'binomial')
    expected_extreme_deviance = np.logaddexp(0.0, 1000.0)
    assert np.all(np.isfinite(extreme_binomial)) and np.allclose(
        extreme_binomial, expected_extreme_deviance, rtol=1e-12), \
        "blocked binomial deviance clipped an extreme link-scale loss"

    extreme_poisson = pycasso_core._fit_deviances(
        np.array([1.0]), np.array([[1.0]]),
        np.array([[-1000.0]]), np.zeros(1), 'poisson')
    assert np.allclose(extreme_poisson, [1998.0], rtol=1e-12), \
        "Poisson deviance clipped a finite extreme link-scale loss"

    extreme_multinomial_logits = np.array([[0.0, -1000.0, -2.0]])
    extreme_multinomial_nll = pycasso_core._multinomial_nll_from_logits(
        np.array([1]), extreme_multinomial_logits)
    expected_multinomial_nll = 1000.0 + np.log1p(np.exp(-2.0))
    assert np.isclose(
        extreme_multinomial_nll, expected_multinomial_nll,
        rtol=1e-13, atol=1e-13), \
        "multinomial NLL clipped an extreme true-class logit"
    common_shift = float(2 ** 53)
    shifted_multinomial_logits = np.array([[
        common_shift, common_shift - 2.0, common_shift - 4.0]])
    assert np.array_equal(
        shifted_multinomial_logits - common_shift,
        np.array([[0.0, -2.0, -4.0]])), \
        "common-shift fixture lost its representable class differences"
    shifted_multinomial_nll = pycasso_core._multinomial_nll_from_logits(
        np.array([0]), shifted_multinomial_logits)
    expected_shifted_multinomial_nll = np.log1p(
        np.exp(-2.0) + np.exp(-4.0))
    assert np.isclose(
        shifted_multinomial_nll, expected_shifted_multinomial_nll,
        rtol=1e-13, atol=1e-13), \
        "multinomial NLL lost accuracy under a large common shift"
    perfect_poisson_deviance = pycasso_core._poisson_deviance_from_eta(
        np.array([1.0, 2.0, 4.0, 8.0]),
        np.log(np.array([1.0, 2.0, 4.0, 8.0])))
    assert perfect_poisson_deviance == 0.0, \
        "perfect Poisson fit retained a negative rounding artifact"
    try:
        pycasso_core._fit_deviances(
            np.array([1.0]), np.array([[1.0]]),
            np.array([[0.0], [1000.0]]), np.zeros(2), 'poisson')
        assert False, "blocked Poisson deviance should reject overflow"
    except ValueError as exc:
        assert "too large" in str(exc)

    for path_input, input_before in zip(
            (path_x, path_beta, path_intercept, path_offset),
            path_inputs_before):
        assert np.array_equal(path_input, input_before), \
            "scalar path evaluation modified one of its inputs"
finally:
    pycasso_core._SCALAR_PATH_BLOCK_BYTES = original_path_block_bytes

# CV path scoring must match the former one-lambda-at-a-time formulas while
# computing only the requested metric. Exercise one-column, multi-column with
# a tail, and single-block layouts, plus a certified shortened path prefix.
assert pycasso_core._SCALAR_CV_PATH_BLOCK_BYTES == 1024 * 1024, \
    "scalar CV predictor blocks must retain their 1 MiB per-worker budget"


def legacy_scalar_cv_fold_losses(y, x, beta, intercept, family, measure,
                                 offset=None, path_size=None):
    if path_size is None:
        path_size = beta.shape[0]
    losses = np.empty(path_size)
    for index in range(path_size):
        eta = x @ beta[index] + intercept[index]
        if offset is not None:
            eta = eta + offset
        if family == 'binomial':
            response_fit = pycasso_core._sigmoid(eta)
        elif family == 'poisson':
            response_fit = pycasso_core._poisson_mean(eta)
        else:
            response_fit = eta
        if measure == 'mse':
            loss = np.mean((y - response_fit) ** 2)
        elif measure == 'mae':
            loss = np.mean(np.abs(y - response_fit))
        elif measure == 'class':
            loss = np.mean((eta > 0).astype(int) != y.astype(int))
        elif family in ('gaussian', 'sqrtlasso'):
            loss = np.mean((y - eta) ** 2) / 2.0
        elif family == 'binomial':
            loss = pycasso_core._binomial_nll_from_eta(y, eta)
        else:
            loss = pycasso_core._poisson_deviance_from_eta(
                y, eta, mean=response_fit)
        losses[index] = loss
    return losses


cv_fold_measures = {
    'gaussian': ('deviance', 'mse', 'mae'),
    'sqrtlasso': ('deviance', 'mse', 'mae'),
    'binomial': ('deviance', 'mse', 'mae', 'class'),
    'poisson': ('deviance', 'mse', 'mae'),
}
cv_fold_inputs_before = tuple(value.copy() for value in (
    path_x, path_beta, path_intercept, path_offset))
for cv_fold_family, cv_fold_y in path_responses.items():
    cv_fold_offset = (path_offset if cv_fold_family in
                      ('binomial', 'poisson') else None)
    for cv_fold_measure in cv_fold_measures[cv_fold_family]:
        expected_full = legacy_scalar_cv_fold_losses(
            cv_fold_y, path_x, path_beta, path_intercept,
            cv_fold_family, cv_fold_measure, offset=cv_fold_offset)
        for cv_fold_block_bytes in (
                path_x.shape[0] * 8,
                path_x.shape[0] * 8 * 2,
                path_x.shape[0] * 8 * path_beta.shape[0]):
            actual_full = pycasso_core._scalar_cv_fold_losses(
                cv_fold_y, path_x, path_beta, path_intercept,
                cv_fold_family, cv_fold_measure, offset=cv_fold_offset,
                block_bytes=cv_fold_block_bytes)
            if cv_fold_measure == 'class':
                assert np.array_equal(actual_full, expected_full), \
                    "blocked binomial CV changed strict class scoring"
            else:
                assert np.allclose(
                    actual_full, expected_full, rtol=1e-12, atol=1e-14), \
                    (f"{cv_fold_family} {cv_fold_measure} CV scoring "
                     "changed the legacy formula")

        shortened = pycasso_core._scalar_cv_fold_losses(
            cv_fold_y, path_x, path_beta, path_intercept,
            cv_fold_family, cv_fold_measure, offset=cv_fold_offset,
            path_size=4, block_bytes=path_x.shape[0] * 8 * 2)
        assert shortened.shape == (4,) and np.allclose(
            shortened, expected_full[:4], rtol=1e-12, atol=1e-14), \
            f"{cv_fold_family} CV scoring ignored a shortened path prefix"

        # Reconstruct a small three-fold aggregate and require the same
        # minimum and one-standard-error model choices as the legacy scorer.
        legacy_rows = np.vstack((expected_full, expected_full + 0.01,
                                 expected_full + 0.02))
        blocked_rows = np.vstack((actual_full, actual_full + 0.01,
                                  actual_full + 0.02))
        legacy_mean = legacy_rows.mean(axis=0)
        blocked_mean = blocked_rows.mean(axis=0)
        legacy_se = legacy_rows.std(axis=0, ddof=1) / np.sqrt(3.0)
        blocked_se = blocked_rows.std(axis=0, ddof=1) / np.sqrt(3.0)
        legacy_minimum = int(np.argmin(legacy_mean))
        blocked_minimum = int(np.argmin(blocked_mean))
        legacy_1se = np.flatnonzero(
            legacy_mean <= legacy_mean[legacy_minimum] +
            legacy_se[legacy_minimum])[0]
        blocked_1se = np.flatnonzero(
            blocked_mean <= blocked_mean[blocked_minimum] +
            blocked_se[blocked_minimum])[0]
        assert legacy_minimum == blocked_minimum and \
            legacy_1se == blocked_1se, \
            f"{cv_fold_family} {cv_fold_measure} changed CV selection"

for cv_fold_input, cv_fold_before in zip(
        (path_x, path_beta, path_intercept, path_offset),
        cv_fold_inputs_before):
    assert np.array_equal(cv_fold_input, cv_fold_before), \
        "scalar CV scoring modified one of its inputs"

# Class and binomial deviance do not need response probabilities. Prohibit the
# sigmoid explicitly so later refactors cannot silently restore wasted work.
original_cv_sigmoid = pycasso_core._sigmoid
try:
    pycasso_core._sigmoid = lambda eta: (_ for _ in ()).throw(
        AssertionError("unused sigmoid evaluated"))
    pycasso_core._scalar_cv_fold_losses(
        path_responses['binomial'], path_x, path_beta, path_intercept,
        'binomial', 'class')
    pycasso_core._scalar_cv_fold_losses(
        path_responses['binomial'], path_x, path_beta, path_intercept,
        'binomial', 'deviance')
finally:
    pycasso_core._sigmoid = original_cv_sigmoid

# A BLAS-3 block can differ from a one-model BLAS-2 reduction in the final
# bit. At the strict binomial eta > 0 boundary that is a discrete model change,
# so class scoring must retain the legacy GEMV arithmetic exactly.
cancellation_x = np.array([
    [-0.7965739661663147, -0.15632014860238724,
     0.16995536585099857],
    [0.05905388012026707, 0.25503824674720255,
     0.6795177039512243],
])
cancellation_beta = np.array([
    [-0.08323198029169392, -1.4015556287646558,
     -0.8670470406306167],
    [-0.23148102042244673, 0.9382690527452191,
     0.015225056436908918],
])
cancellation_intercept = -np.array([
    (cancellation_x @ cancellation_beta[index])[0]
    for index in range(cancellation_beta.shape[0])
])
cancellation_y = np.array([0.0, 1.0])
cancellation_expected = legacy_scalar_cv_fold_losses(
    cancellation_y, cancellation_x, cancellation_beta,
    cancellation_intercept, 'binomial', 'class')
original_predictor_blocks = pycasso_core._scalar_linear_predictor_blocks
try:
    def reject_batched_class_scoring(*args, **kwargs):
        raise AssertionError("binomial class scoring entered a GEMM block")

    pycasso_core._scalar_linear_predictor_blocks = \
        reject_batched_class_scoring
    cancellation_actual = pycasso_core._scalar_cv_fold_losses(
        cancellation_y, cancellation_x, cancellation_beta,
        cancellation_intercept, 'binomial', 'class',
        block_bytes=cancellation_x.shape[0] * 8 * 2)
finally:
    pycasso_core._scalar_linear_predictor_blocks = original_predictor_blocks
assert np.array_equal(cancellation_actual, cancellation_expected), \
    "batched scalar CV changed a binomial class at exact eta == 0"

# Multinomial post-fit rescaling must remain numerically equivalent to the
# former multiply-and-reduce formula while reusing the native output buffers.
# Matrix-vector multiplication can change only last-bit reduction order.
def legacy_multinomial_rescale(beta, intercept, xinvc, xm):
    beta_rescaled = beta * xinvc
    intercept_rescaled = (
        intercept - (beta_rescaled * xm).sum(axis=2))
    return beta_rescaled, intercept_rescaled


mn_rescale_rng = np.random.RandomState(20260719)
mn_rescale_cases = [
    (
        "ordinary path",
        np.linspace(-0.45, 0.55, 5 * 3 * 11).reshape(5, 3, 11),
        np.linspace(-0.2, 0.3, 5 * 3).reshape(5, 3),
        np.linspace(0.4, 1.6, 11),
        np.linspace(-0.7, 0.8, 11),
    ),
    (
        "truncated path",
        mn_rescale_rng.normal(size=(7, 4, 19))[:2].copy(),
        mn_rescale_rng.normal(size=(7, 4))[:2].copy(),
        mn_rescale_rng.uniform(0.25, 1.75, size=19),
        mn_rescale_rng.normal(size=19),
    ),
    (
        "single coefficient",
        np.array([[[3.25]]]),
        np.array([[0.75]]),
        np.array([0.125]),
        np.array([-8.0]),
    ),
    (
        "empty path",
        np.empty((0, 3, 11)),
        np.empty((0, 3)),
        np.linspace(0.4, 1.6, 11),
        np.linspace(-0.7, 0.8, 11),
    ),
    (
        "zero-feature path",
        np.empty((4, 3, 0)),
        mn_rescale_rng.normal(size=(4, 3)),
        np.empty(0),
        np.empty(0),
    ),
    (
        "wide path",
        mn_rescale_rng.normal(size=(3, 4, 257)),
        mn_rescale_rng.normal(size=(3, 4)),
        mn_rescale_rng.uniform(0.25, 1.75, size=257),
        mn_rescale_rng.normal(size=257),
    ),
]
extreme_scales = np.array([
    1e-100, 1e100, 1e-75, 1e75, 1e-50, 1e50, 1e-25, 1e25])
extreme_beta = np.array([
    1e200, -1e-200, 1e150, -1e-150,
    1e100, -1e-100, 1e50, -1e-50])
mn_rescale_cases.append((
    "extreme finite scales",
    np.ascontiguousarray(
        np.tile(extreme_beta, 2 * 2).reshape(2, 2, 8)),
    np.array([[1e100, -1e100], [1e-100, -1e-100]]),
    extreme_scales,
    1.0 / (extreme_beta * extreme_scales),
))
original_mn_rescale_block_bytes = (
    pycasso_core._MULTINOMIAL_RESCALE_BLOCK_BYTES)
try:
    # Force multi-block execution for every nonempty case with features.
    pycasso_core._MULTINOMIAL_RESCALE_BLOCK_BYTES = 2 * 8
    for (case_name, case_beta, case_intercept, case_xinvc,
         case_xm) in mn_rescale_cases:
        beta_owner = np.ascontiguousarray(case_beta, dtype='double')
        intercept_owner = np.ascontiguousarray(
            case_intercept, dtype='double')
        expected_mn_beta, expected_mn_intercept = (
            legacy_multinomial_rescale(
                beta_owner.copy(), intercept_owner.copy(),
                case_xinvc, case_xm))
        actual_mn_beta, actual_mn_intercept = (
            pycasso_core._rescale_multinomial_solution_in_place(
                beta_owner, intercept_owner, case_xinvc, case_xm))
        assert np.array_equal(actual_mn_beta, expected_mn_beta), \
            "multinomial rescaling changed coefficients for %s" % case_name
        if expected_mn_beta.size:
            reduction_scale = np.sum(
                np.abs(expected_mn_beta * case_xm), axis=2)
            rounding_tolerance = (
                64.0 * np.finfo('double').eps *
                (reduction_scale + np.abs(expected_mn_intercept)))
            assert np.all(
                np.abs(actual_mn_intercept - expected_mn_intercept) <=
                rounding_tolerance), \
                "multinomial intercept rescaling drifted for %s" % case_name
        else:
            assert np.array_equal(
                actual_mn_intercept, expected_mn_intercept), \
                "empty multinomial rescaling changed %s" % case_name
        assert actual_mn_beta is beta_owner and \
            actual_mn_intercept is intercept_owner, \
            "multinomial rescaling copied outputs for %s" % case_name
finally:
    pycasso_core._MULTINOMIAL_RESCALE_BLOCK_BYTES = (
        original_mn_rescale_block_bytes)

# Step 4: offset
print("\n=== Step 4: offset ===")
# Y_p was generated as Poisson(_exposure * exp(X@b)), so true model uses log(_exposure) as offset
sp = pycasso.Solver(X, Y_p, family="poisson", offset=np.log(_exposure))
sp.train()
assert 'dev_ratio' in sp.result, "poisson with offset missing dev_ratio"
print(f"  poisson offset dev_ratio range: [{sp.result['dev_ratio'].min():.3f}, {sp.result['dev_ratio'].max():.3f}]")
assert sp.result['dev_ratio'].max() > 0.01, \
    f"Expected dev_ratio > 0.01 when signal is present, got {sp.result['dev_ratio'].max():.4f}"
# compare: without offset, dev_ratio should be lower (offset matters)
sp_no = pycasso.Solver(X, Y_p, family="poisson")
sp_no.train()
print(f"  without offset dev_ratio max: {sp_no.result['dev_ratio'].max():.3f}")
print(f"  with offset dev_ratio max:    {sp.result['dev_ratio'].max():.3f}")

# Null initialization must use log-sum-exp rather than clipping the offset.
extreme_poisson_y = np.array([1.0, 2.0, 3.0, 4.0])
extreme_poisson_offset = np.array([-1000.0, 800.0, 801.0, 802.0])
extreme_poisson_eta = pycasso_core._scalar_null_linear_predictor(
    extreme_poisson_y, "poisson", offset=extreme_poisson_offset,
    include_intercept=True)
extreme_poisson_mu = pycasso_core._poisson_mean(extreme_poisson_eta)
assert np.isclose(extreme_poisson_mu.sum(), extreme_poisson_y.sum(),
                  rtol=1e-13, atol=1e-13), \
    "Poisson extreme-offset null intercept does not satisfy its score equation"
assert np.all(np.isfinite(extreme_poisson_eta)), \
    "Poisson extreme-offset null predictor is not finite"

# Prediction offsets are new-data quantities: an offset-trained model must be
# given one, and it is added before applying the inverse link.
try:
    sp.predict(X[:7], type="link")
    assert False, "offset-trained prediction should require newoffset"
except ValueError as exc:
    assert "newoffset" in str(exc) and "must be provided" in str(exc)

poisson_newoffset = np.linspace(-0.6, 0.7, 7)
poisson_eta = (X[:7] @ sp.result['beta'][-1] +
               sp.result['intercept'][-1] + poisson_newoffset)
assert np.allclose(
    sp.predict(X[:7], type="link", newoffset=poisson_newoffset),
    poisson_eta), "Poisson link prediction ignored newoffset"
assert np.allclose(
    sp.predict(X[:7], type="response", newoffset=poisson_newoffset),
    np.exp(np.clip(poisson_eta, -500, 500))), \
    "Poisson response prediction applied newoffset after the inverse link"
try:
    sp.assess(X[:7], Y_p[:7])
    assert False, "offset-trained assessment should require newoffset"
except ValueError as exc:
    assert "newoffset" in str(exc) and "must be provided" in str(exc)
poisson_assessment = sp.assess(
    X[:7], Y_p[:7], newoffset=poisson_newoffset)
expected_poisson_mean = np.exp(np.clip(poisson_eta, -500, 500))
assert np.isclose(
    poisson_assessment['mse'][-1],
    np.mean((Y_p[:7] - expected_poisson_mean) ** 2)), \
    "Poisson assessment ignored newoffset"

try:
    sp.predict(X[:7], newoffset=np.zeros(6))
    assert False, "short newoffset should be rejected"
except ValueError as exc:
    assert "length" in str(exc)

bad_newoffset = poisson_newoffset.copy()
bad_newoffset[2] = np.inf
try:
    sp.predict(X[:7], newoffset=bad_newoffset)
    assert False, "non-finite newoffset should be rejected"
except ValueError as exc:
    assert "finite" in str(exc)

# The presence flag must not be inferred from numerical nonzeros.
sb_zero_offset = pycasso.Solver(
    X, Y_b, family="binomial", offset=np.zeros(n))
sb_zero_offset.train()
try:
    sb_zero_offset.predict(X[:5], type="link")
    assert False, "an explicitly zero training offset still requires newoffset"
except ValueError as exc:
    assert "newoffset" in str(exc) and "must be provided" in str(exc)
zero_offset_eta = (X[:5] @ sb_zero_offset.result['beta'][-1] +
                   sb_zero_offset.result['intercept'][-1] + prediction_shift)
assert np.allclose(
    sb_zero_offset.predict(
        X[:5], type="response", newoffset=prediction_shift),
    pycasso_core._sigmoid(zero_offset_eta)), \
    "binomial response prediction did not apply newoffset on the link scale"
assert np.array_equal(
    sb_zero_offset.predict(X[:5], type="class", newoffset=prediction_shift),
    (zero_offset_eta > 0).astype(int)), \
    "binomial class prediction did not include newoffset"
try:
    sb_zero_offset.assess(X[:5], Y_b[:5])
    assert False, "offset-trained assessment should require newoffset"
except ValueError as exc:
    assert "newoffset" in str(exc) and "must be provided" in str(exc)
binomial_assessment = sb_zero_offset.assess(
    X[:5], Y_b[:5], newoffset=prediction_shift)
assert np.isclose(
    binomial_assessment['class_error'][-1],
    np.mean((zero_offset_eta > 0).astype(int) != Y_b[:5].astype(int))), \
    "binomial assessment ignored newoffset"
try:
    sb_zero_offset.confusion(X[:5], Y_b[:5], lambdidx=[-1])
    assert False, "offset-trained confusion should require newoffset"
except ValueError as exc:
    assert "newoffset" in str(exc) and "must be provided" in str(exc)
offset_confusion = sb_zero_offset.confusion(
    X[:5], Y_b[:5], lambdidx=[sb_zero_offset.nlambda - 1],
    newoffset=prediction_shift)[0]
expected_confusion = np.zeros((2, 2), dtype=int)
np.add.at(
    expected_confusion,
    ((zero_offset_eta > 0).astype(int), Y_b[:5].astype(int)), 1)
assert np.array_equal(offset_confusion, expected_confusion), \
    "binomial confusion ignored newoffset"

# ValueError if offset used with wrong family
try:
    _ = pycasso.Solver(X, Y_g, family="gaussian", offset=np.ones(n))
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("  PASS")

# Step 4b: scalar adaptive LLA V2 diagnostics and stage control
print("\n=== Step 4b: scalar adaptive LLA ===")
assert pycasso_core._SCALAR_LLA_STATUS_NAMES[10] == \
    'lla_stationarity_limit', "scalar adaptive LLA status map is incomplete"
scalar_rng = np.random.RandomState(7)
n_scalar, d_scalar = 80, 8
X_scalar = scalar_rng.normal(size=(n_scalar, d_scalar))
b_scalar = np.array([0.8, -0.6, 0.4] + [0.0] * (d_scalar - 3))
eta_scalar = X_scalar @ b_scalar
Y_scalar_binomial = (
    scalar_rng.rand(n_scalar) < 1.0 / (1.0 + np.exp(-eta_scalar))
).astype(float)
Y_scalar_poisson = scalar_rng.poisson(
    np.exp(np.clip(0.3 * eta_scalar, -1.0, 1.0))).astype(float)
Y_scalar_sqrt = eta_scalar + scalar_rng.normal(size=n_scalar)
scalar_path = np.array([0.3, 0.15, 0.07, 0.03])
scalar_responses = {
    'binomial': Y_scalar_binomial,
    'poisson': Y_scalar_poisson,
    'sqrtlasso': Y_scalar_sqrt,
}


def assert_scalar_lla_diagnostics(solver, label):
    result = solver.result
    assert result['status_code'] in (0, 1, 10), \
        f"{label} returned hard status {result['status']!r}"
    assert result['status'] == pycasso_core._SCALAR_LLA_STATUS_NAMES[
        result['status_code']], f"{label} status name mismatch"
    assert result['failed_lambda'] == -1 and result['failed_stage'] == -1, \
        f"{label} unexpectedly reported a hard-failure location"
    for field in ('runtime', 'lla_stages', 'stages', 'objective', 'kkt',
                  'stationarity'):
        assert result[field].shape == (solver.nlambda,), \
            f"{label} {field} is not aligned to the returned path"
    assert result['stages'] is result['lla_stages'], \
        f"{label} stages alias allocated a divergent buffer"
    assert result['runtime'] is result['train_time'], \
        f"{label} runtime alias allocated a duplicate buffer"
    assert np.all(np.isfinite(result['objective'])) and \
        np.all(np.isfinite(result['kkt'])) and \
        np.all(np.isfinite(result['stationarity'])), \
        f"{label} diagnostics contain non-finite fitted values"


scalar_statuses = []
for scalar_family, scalar_y in scalar_responses.items():
    for scalar_penalty in ('mcp', 'scad'):
        scalar_solver = pycasso.Solver(
            X_scalar, scalar_y, lambdas=scalar_path,
            family=scalar_family, penalty=scalar_penalty,
            prec=1e-6, max_ite=200)
        with warnings.catch_warnings(record=True) as scalar_warnings:
            warnings.simplefilter("always")
            scalar_solver.train()
        assert not scalar_warnings, \
            f"{scalar_family} {scalar_penalty} usable status emitted warning"
        assert_scalar_lla_diagnostics(
            scalar_solver, f"{scalar_family} {scalar_penalty}")
        assert np.all(scalar_solver.result['lla_stages'] == 3), \
            f"{scalar_family} {scalar_penalty} ignored default stage cap"
        assert scalar_solver.result['state'] == 'trained', \
            f"{scalar_family} {scalar_penalty} usable path is not trained"
        scalar_statuses.append(scalar_solver.result['status_code'])
assert 10 in scalar_statuses, \
    "targeted scalar tests did not exercise usable stationarity-limit status"

# Raising the cap must let a deterministic MCP fit obtain certification.
strict_scalar = pycasso.Solver(
    X_scalar, Y_scalar_binomial, lambdas=scalar_path,
    family='binomial', penalty='mcp', prec=1e-6, max_ite=200,
    lla_max_stages=8)
strict_scalar.train()
assert_scalar_lla_diagnostics(strict_scalar, "binomial MCP cap=8")
assert strict_scalar.result['lla_max_stages'] == 8, \
    "scalar result omitted its requested LLA stage budget"
assert strict_scalar.result['status_code'] == 0 and \
    np.all(strict_scalar.result['stationarity'] <= strict_scalar.prec), \
    "raised scalar LLA cap did not achieve requested stationarity"
assert np.max(strict_scalar.result['lla_stages']) > 3, \
    "raised scalar LLA cap was not propagated to the native solver"

scalar_cv_probe = pycasso.Solver(
    X_scalar, Y_scalar_binomial, lambdas=scalar_path,
    family='binomial', penalty='mcp', prec=1e-6, max_ite=200,
    lla_max_stages=7)
scalar_fold_caps = []
scalar_original_solver_class = pycasso_core.Solver
scalar_global_rng_state = np.random.get_state()


def recording_scalar_fold_solver(*args, **kwargs):
    scalar_fold_caps.append(kwargs.get('lla_max_stages'))
    return scalar_original_solver_class(*args, **kwargs)


try:
    pycasso_core.Solver = recording_scalar_fold_solver
    scalar_cv_probe.cross_validate(nfolds=2, type_measure='class')
finally:
    pycasso_core.Solver = scalar_original_solver_class
    np.random.set_state(scalar_global_rng_state)
assert scalar_fold_caps == [7, 7], \
    f"scalar CV folds lost lla_max_stages: {scalar_fold_caps}"

# GLM MSE/MAE are response-scale losses, not errors against the linear link.
response_foldid = np.empty(n_scalar, dtype=int)
next_response_fold = 0
for response_class in (0, 1):
    response_indices = np.flatnonzero(Y_scalar_binomial == response_class)
    response_foldid[response_indices] = (
        np.arange(response_indices.size) + next_response_fold) % 3
    next_response_fold = (next_response_fold + response_indices.size) % 3
for response_family, response_y, response_measure in (
        ('binomial', Y_scalar_binomial, 'mse'),
        ('poisson', Y_scalar_poisson, 'mae')):
    response_solver = pycasso.Solver(
        X_scalar, response_y, lambdas=np.array([0.3, 0.15]),
        family=response_family, prec=1e-6, max_ite=200)
    response_cv = response_solver.cross_validate(
        foldid=response_foldid, type_measure=response_measure)
    response_losses = np.zeros((3, 2))
    for response_fold in range(3):
        response_train = response_foldid != response_fold
        response_test = ~response_train
        response_fold_solver = pycasso.Solver(
            X_scalar[response_train], response_y[response_train],
            lambdas=np.array([0.3, 0.15]), family=response_family,
            prec=1e-6, max_ite=200)
        response_fold_solver.train()
        for response_lambda in range(2):
            response_prediction = response_fold_solver.predict(
                X_scalar[response_test], lambdidx=response_lambda,
                type='response')
            error = response_y[response_test] - response_prediction
            response_losses[response_fold, response_lambda] = (
                np.mean(error ** 2) if response_measure == 'mse'
                else np.mean(np.abs(error)))
    assert np.allclose(response_cv['cvm'], response_losses.mean(axis=0),
                       rtol=1e-12, atol=1e-14), \
        f"{response_family} CV {response_measure} was not response-scale"

# Automatic binomial folds retain even a two-observation minority class.
rare_binomial_y = np.array([0] * 22 + [1] * 2, dtype='double')
rare_binomial = pycasso.Solver(
    X_scalar[:24], rare_binomial_y, lambdas=np.array([1.0, 0.5]),
    family='binomial')
np.random.seed(20260716)
rare_binomial_cv = rare_binomial.cross_validate(
    nfolds=4, type_measure='class')
for fold in range(4):
    assert np.unique(
        rare_binomial_y[rare_binomial_cv['foldid'] != fold]).size == 2, \
        f"binomial training fold {fold} lost its minority class"
try:
    pycasso.Solver(
        X_scalar[:24], np.array([0] * 23 + [1], dtype='double'),
        lambdas=np.array([1.0, 0.5]), family='binomial').cross_validate(
            nfolds=4, type_measure='class')
    assert False, "singleton binomial CV should fail"
except ValueError as exc:
    assert "at least two" in str(exc)
try:
    rare_binomial.cross_validate(
        foldid=np.array([0] * 22 + [1, 1]), type_measure='class')
    assert False, "binomial training fold without a class should fail"
except ValueError as exc:
    assert "training fold" in str(exc)

# L1 ignores the cap, while Gaussian MCP keeps its direct nonconvex algorithm.
scalar_l1_default = pycasso.Solver(
    X_scalar, Y_scalar_binomial, lambdas=scalar_path, family='binomial')
scalar_l1_raised = pycasso.Solver(
    X_scalar, Y_scalar_binomial, lambdas=scalar_path, family='binomial',
    lla_max_stages=9)
scalar_l1_default.train()
scalar_l1_raised.train()
assert np.all(scalar_l1_default.result['lla_stages'] == 1) and \
    np.all(scalar_l1_raised.result['lla_stages'] == 1) and \
    np.allclose(scalar_l1_default.result['beta'],
                scalar_l1_raised.result['beta']), \
    "lla_max_stages changed an L1 fit"
gaussian_direct = pycasso.Solver(
    X_scalar, Y_scalar_sqrt, lambdas=scalar_path, family='gaussian',
    penalty='mcp', lla_max_stages=9)
gaussian_direct.train()
assert gaussian_direct.result['state'] == 'trained' and \
    'lla_stages' not in gaussian_direct.result, \
    "Gaussian MCP was incorrectly routed through adaptive LLA"
assert gaussian_direct.result['lla_max_stages'] == 9, \
    "Gaussian result omitted its validated but unused LLA stage budget"

# Gaussian intercept semantics are independent of standardization.  At
# lambda=0 both native update modes must recover the corresponding least-
# squares problem on an uncentered design.
gaussian_x = X_scalar + np.linspace(1.0, 2.4, d_scalar)
gaussian_y = 3.2 + gaussian_x @ b_scalar + 0.03 * np.sin(
    np.arange(n_scalar))
with_intercept_oracle = np.linalg.lstsq(
    np.column_stack([np.ones(n_scalar), gaussian_x]),
    gaussian_y, rcond=None)[0]
no_intercept_oracle = np.linalg.lstsq(
    gaussian_x, gaussian_y, rcond=None)[0]
for gaussian_mode in ('naive', 'covariance'):
    unstandardized = pycasso.Solver(
        gaussian_x, gaussian_y, lambdas=np.array([0.0]),
        family='gaussian', penalty='l1', useintercept=True,
        standardize=False, type_gaussian=gaussian_mode,
        prec=1e-12, max_ite=10000)
    unstandardized.train()
    assert np.allclose(
        unstandardized.result['intercept'][0], with_intercept_oracle[0],
        rtol=0, atol=2e-6), \
        f"{gaussian_mode} unstandardized Gaussian intercept is incorrect"
    assert np.allclose(
        unstandardized.result['beta'][0], with_intercept_oracle[1:],
        rtol=0, atol=2e-6), \
        f"{gaussian_mode} unstandardized Gaussian coefficients are incorrect"

    origin_constrained = pycasso.Solver(
        gaussian_x, gaussian_y, lambdas=np.array([0.0]),
        family='gaussian', penalty='l1', useintercept=False,
        standardize=True, type_gaussian=gaussian_mode,
        # This shifted design is deliberately ill-conditioned.  The native
        # stopping rule controls coordinate objective change, so use a tighter
        # solve tolerance than the coefficient-space oracle below.
        prec=1e-14, max_ite=10000)
    origin_constrained.train()
    assert origin_constrained.result['intercept'][0] == 0.0, \
        f"{gaussian_mode} no-intercept Gaussian returned an intercept"
    assert np.allclose(
        origin_constrained.result['beta'][0], no_intercept_oracle,
        rtol=0, atol=2e-6), \
        f"{gaussian_mode} standardized no-intercept Gaussian changed origin"

# Standardized no-intercept scalar fits must retain the original origin.  In
# particular, their first generated lambda is computed at the zero-intercept
# null model, and rescaling must not manufacture an intercept afterwards.
X_scalar_shifted = X_scalar + np.linspace(0.5, 2.0, d_scalar)
for scalar_family, scalar_y in scalar_responses.items():
    no_intercept = pycasso.Solver(
        X_scalar_shifted, scalar_y, lambdas=(4, 0.2),
        family=scalar_family, useintercept=False, standardize=True,
        prec=1e-6, max_ite=300)
    assert np.array_equal(no_intercept._xm,
                          np.zeros(d_scalar)), \
        f"{scalar_family} no-intercept standardization centered the design"
    if scalar_family == 'binomial':
        null_residual = scalar_y - 0.5
        expected_null_deviance = np.log(2.0)
        expected_lambda_max = np.max(
            np.abs(no_intercept.x.T @ null_residual)) / n_scalar
    elif scalar_family == 'poisson':
        null_residual = scalar_y - 1.0
        expected_null_deviance = pycasso_core._poisson_dev(
            scalar_y, np.ones(n_scalar))
        expected_lambda_max = np.max(
            np.abs(no_intercept.x.T @ null_residual)) / n_scalar
    else:
        null_residual = scalar_y
        null_scale = np.sqrt(np.mean(null_residual ** 2))
        expected_null_deviance = np.mean(null_residual ** 2) / 2.0
        expected_lambda_max = np.max(
            np.abs(no_intercept.x.T @ null_residual)
        ) / n_scalar / null_scale
    assert np.isclose(no_intercept.lambdas[0], expected_lambda_max,
                      rtol=1e-12, atol=1e-14), \
        f"{scalar_family} no-intercept lambda_max used the wrong null model"
    no_intercept.train()
    assert np.array_equal(
        no_intercept.result['intercept'],
        np.zeros_like(no_intercept.result['intercept'])), \
        f"{scalar_family} no-intercept fit returned a nonzero intercept"
    assert np.isclose(no_intercept.result['nulldev'],
                      expected_null_deviance, rtol=1e-12, atol=1e-14), \
        f"{scalar_family} no-intercept null deviance is inconsistent"

# A binomial offset changes both the intercept-only null fit and lambda_max.
binomial_offset = np.linspace(-1.2, 0.9, n_scalar)
offset_solver = pycasso.Solver(
    X_scalar, Y_scalar_binomial, lambdas=(4, 0.2), family='binomial',
    offset=binomial_offset, prec=1e-6, max_ite=300)
offset_null_eta = pycasso_core._scalar_null_linear_predictor(
    Y_scalar_binomial, 'binomial', offset=binomial_offset,
    include_intercept=True)
assert abs(np.mean(pycasso_core._sigmoid(offset_null_eta)) -
           np.mean(Y_scalar_binomial)) < 1e-12, \
    "binomial offset null intercept did not satisfy its score equation"
expected_offset_lambda = np.max(np.abs(
    offset_solver.x.T @
    (Y_scalar_binomial - pycasso_core._sigmoid(offset_null_eta)))) / n_scalar
assert np.isclose(offset_solver.lambdas[0], expected_offset_lambda,
                  rtol=1e-12, atol=1e-14), \
    "binomial offset lambda_max ignored the offset null model"
offset_solver.train()
expected_offset_null = np.mean(
    np.logaddexp(0.0, offset_null_eta) -
    Y_scalar_binomial * offset_null_eta)
assert np.isclose(offset_solver.result['nulldev'], expected_offset_null,
                  rtol=1e-12, atol=1e-14), \
    "binomial offset null deviance ignored the offset"
print("  PASS")

# Step 5: cross_validate
print("\n=== Step 5: cross_validate ===")
cv = s.cross_validate(nfolds=5, type_measure="mse")
assert 'lambda_min' in cv, "cv missing lambda_min"
assert 'cvm' in cv, "cv missing cvm"
assert len(cv['cvm']) == s.nlambda, f"cvm length mismatch: {len(cv['cvm'])} != {s.nlambda}"
cv_solver_lambdas = s.lambdas.copy()
assert cv['lambda'].flags.owndata and cv['lambda'].base is None and \
    not np.shares_memory(cv['lambda'], s.lambdas), \
    "cross-validation lambda path is not an owning snapshot"
cv['lambda'][0] += 1.0
assert np.array_equal(s.lambdas, cv_solver_lambdas), \
    "mutating cross-validation lambda output changed solver state"
fresh_gaussian_cv_solver = pycasso.Solver(
    X, Y_g, lambdas=np.array([0.2, 0.1]), family="gaussian")
fresh_gaussian_cv = fresh_gaussian_cv_solver.cross_validate(
    nfolds=3, type_measure="mse")
assert fresh_gaussian_cv_solver.result['state'] == 'trained', \
    "fresh Gaussian CV did not retain its full-data fit"
assert np.array_equal(
    fresh_gaussian_cv['nzero'], fresh_gaussian_cv_solver.result['df']), \
    "fresh Gaussian CV returned placeholder nzero values"
try:
    fresh_gaussian_cv_solver.cross_validate(nfolds=3, type_measure="class")
    assert False, "Gaussian CV should reject class loss"
except ValueError as exc:
    assert "only for binomial or multinomial" in str(exc)

# A shortened Gaussian fold has no non-Gaussian status field. It must produce
# the documented path-coverage error instead of leaking an internal KeyError.
gaussian_solver_class = pycasso_core.Solver


class ShortGaussianFold:
    def __init__(self, x, y, **kwargs):
        self.nlambda = 1
        self.result = {'state': 'trained'}

    def train(self):
        return None


try:
    pycasso_core.Solver = ShortGaussianFold
    fresh_gaussian_cv_solver.cross_validate(
        foldid=np.arange(X.shape[0]) % 3, type_measure="mse")
    assert False, "shortened Gaussian CV fold should fail explicitly"
except PycassoError as exc:
    assert "covered 1/2 lambdas" in str(exc) and \
        "status='completed'" in str(exc), \
        "shortened Gaussian CV fold did not report a stable error"
finally:
    pycasso_core.Solver = gaussian_solver_class

# Fold parallelism is opt-in. Its validation must happen before a fresh
# Solver trains or generated folds consume random state.
assert inspect.signature(
    pycasso.Solver.cross_validate).parameters['n_jobs'].default == 1, \
    "cross_validate n_jobs must default to serial execution"
parallel_validation_solver = pycasso.Solver(
    X[:40, :6], Y_g[:40], lambdas=np.array([0.3, 0.15]))
parallel_validation_lambdas = parallel_validation_solver.lambdas.copy()
parallel_validation_rng = np.random.get_state()
for invalid_n_jobs in (
        0, -1, 1.5, True, None, "2", [2], np.nan, np.inf,
        np.iinfo(np.int32).max + 1):
    try:
        parallel_validation_solver.cross_validate(n_jobs=invalid_n_jobs)
        assert False, f"invalid n_jobs accepted: {invalid_n_jobs!r}"
    except ValueError as exc:
        assert "n_jobs" in str(exc), \
            f"invalid n_jobs error omitted its argument name: {exc}"
assert parallel_validation_solver.result['state'] == 'not trained' and \
    np.array_equal(parallel_validation_solver.lambdas,
                   parallel_validation_lambdas), \
    "invalid n_jobs mutated or trained a fresh Solver"
parallel_validation_rng_after = np.random.get_state()
assert parallel_validation_rng[0] == parallel_validation_rng_after[0] and \
    np.array_equal(parallel_validation_rng[1],
                   parallel_validation_rng_after[1]) and \
    parallel_validation_rng[2:] == parallel_validation_rng_after[2:], \
    "invalid n_jobs consumed random state before validation"


def assert_cv_results_identical(left, right, label):
    """Require bitwise-identical ordered CV outputs."""
    assert left.keys() == right.keys(), f"{label} changed CV fields"
    for key in left:
        if isinstance(left[key], np.ndarray):
            assert np.array_equal(left[key], right[key]), \
                f"{label} changed CV array {key}"
        else:
            assert left[key] == right[key], \
                f"{label} changed CV value {key}"


# Fixed folds make serial/parallel comparisons independent of random fold
# generation. Cover every supported family/measure combination and nonzero
# binomial/Poisson offsets.
parallel_rng = np.random.RandomState(20260720)
parallel_n, parallel_d = 60, 8
parallel_x = parallel_rng.normal(size=(parallel_n, parallel_d))
parallel_signal = parallel_x @ np.array(
    [0.6, -0.4, 0.2] + [0.0] * (parallel_d - 3))
parallel_foldid = np.arange(parallel_n) % 4
parallel_offset = np.linspace(-0.25, 0.25, parallel_n)
parallel_responses = {
    'gaussian': parallel_signal + parallel_rng.normal(
        scale=0.3, size=parallel_n),
    'sqrtlasso': parallel_signal + parallel_rng.normal(
        scale=0.3, size=parallel_n),
    'binomial': (np.arange(parallel_n) % 2).astype(float),
    'poisson': (1 + np.arange(parallel_n) % 4).astype(float),
    'multinomial': (np.arange(parallel_n) % 3).astype(float),
}
parallel_measures = {
    'gaussian': ('deviance', 'mse', 'mae'),
    'sqrtlasso': ('deviance', 'mse', 'mae'),
    'binomial': ('deviance', 'mse', 'mae', 'class'),
    'poisson': ('deviance', 'mse', 'mae'),
    'multinomial': ('deviance', 'class'),
}
for parallel_family, parallel_y in parallel_responses.items():
    parallel_kwargs = {}
    if parallel_family in ('binomial', 'poisson'):
        parallel_kwargs['offset'] = parallel_offset
    parallel_solver = pycasso.Solver(
        parallel_x, parallel_y, lambdas=np.array([0.3, 0.15, 0.08]),
        family=parallel_family, max_ite=300, **parallel_kwargs)
    for parallel_measure in parallel_measures[parallel_family]:
        omitted_jobs = parallel_solver.cross_validate(
            foldid=parallel_foldid, type_measure=parallel_measure)
        explicit_serial = parallel_solver.cross_validate(
            foldid=parallel_foldid, type_measure=parallel_measure,
            n_jobs=1)
        two_threads = parallel_solver.cross_validate(
            foldid=parallel_foldid, type_measure=parallel_measure,
            n_jobs=2)
        four_threads = parallel_solver.cross_validate(
            foldid=parallel_foldid, type_measure=parallel_measure,
            n_jobs=4)
        label = f"{parallel_family} {parallel_measure}"
        assert_cv_results_identical(
            omitted_jobs, explicit_serial, label + " n_jobs=1")
        assert_cv_results_identical(
            omitted_jobs, two_threads, label + " n_jobs=2")
        assert_cv_results_identical(
            omitted_jobs, four_threads, label + " n_jobs=4")


# A blocking fold fixture proves native-independent fold work actually
# overlaps and records the worker cap when n_jobs exceeds the fold count.
overlap_n = 24
overlap_x = np.column_stack((
    np.arange(overlap_n, dtype=float),
    np.linspace(-1.0, 1.0, overlap_n),
))
overlap_y = 0.1 * overlap_x[:, 0] - overlap_x[:, 1]
overlap_foldid = np.arange(overlap_n) % 3
overlap_solver = pycasso.Solver(
    overlap_x, overlap_y, lambdas=np.array([0.3, 0.15]),
    standardize=False)
overlap_solver.train()
parallel_solver_class = pycasso_core.Solver
real_thread_pool = concurrent.futures.ThreadPoolExecutor


class RecordingThreadPool(real_thread_pool):
    requested_workers = []

    def __init__(self, max_workers=None, *args, **kwargs):
        self.requested_workers.append(max_workers)
        super().__init__(max_workers=max_workers, *args, **kwargs)


class BlockingFoldSolver:
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    maximum_active = 0

    def __init__(self, x, y, **kwargs):
        self.nlambda = len(kwargs['lambdas'])
        self.result = {
            'state': 'not trained',
            'status': 'completed',
            'path_early_stopped': False,
            'beta': np.zeros((self.nlambda, x.shape[1])),
            'intercept': np.zeros(self.nlambda),
        }

    def train(self):
        with self.lock:
            type(self).active += 1
            type(self).maximum_active = max(
                type(self).maximum_active, type(self).active)
            if type(self).active >= 2:
                type(self).release.set()
        try:
            if not type(self).release.wait(timeout=2.0):
                raise AssertionError("parallel CV folds did not overlap")
            time.sleep(0.01)
            self.result['state'] = 'trained'
        finally:
            with self.lock:
                type(self).active -= 1


try:
    pycasso_core.Solver = BlockingFoldSolver
    concurrent.futures.ThreadPoolExecutor = RecordingThreadPool
    overlap_result = overlap_solver.cross_validate(
        foldid=overlap_foldid, type_measure='mse', n_jobs=99)
finally:
    concurrent.futures.ThreadPoolExecutor = real_thread_pool
    pycasso_core.Solver = parallel_solver_class
assert BlockingFoldSolver.maximum_active >= 2 and \
    RecordingThreadPool.requested_workers[-1] == 3 and \
    overlap_result['cvm'].shape == (2,), \
    "parallel CV did not overlap folds or cap workers at nfolds"


# Executor.map is consumed in fold order. Even when later folds fail first,
# the public exception must consistently come from the lowest failing fold.
class OrderedFailureFoldSolver:
    def __init__(self, x, y, **kwargs):
        present = set(np.asarray(x[:, 0], dtype=int).tolist())
        missing = sorted(set(range(overlap_n)) - present)
        self.fold = missing[0] % 3
        self.nlambda = len(kwargs['lambdas'])
        self.result = {'state': 'not trained'}

    def train(self):
        if self.fold == 0:
            time.sleep(0.04)
        raise RuntimeError(f"ordered fold {self.fold}")


try:
    pycasso_core.Solver = OrderedFailureFoldSolver
    try:
        overlap_solver.cross_validate(
            foldid=overlap_foldid, type_measure='mse', n_jobs=3)
        assert False, "parallel CV accepted failing folds"
    except RuntimeError as exc:
        assert str(exc) == "ordered fold 0", \
            f"parallel CV exposed a later fold error first: {exc}"
finally:
    pycasso_core.Solver = parallel_solver_class


# A certified shortened fold contributes its valid prefix only. Parallel and
# serial aggregation must choose the same common prefix in fold order.
class EarlyStopFoldSolver:
    def __init__(self, x, y, **kwargs):
        present = set(np.asarray(x[:, 0], dtype=int).tolist())
        missing = sorted(set(range(overlap_n)) - present)
        fold = missing[0] % 3
        requested = len(kwargs['lambdas'])
        self.nlambda = requested - 1 if fold == 0 else requested
        self.result = {
            'state': 'trained',
            'status': 'completed',
            'path_early_stopped': fold == 0,
            'beta': np.zeros((self.nlambda, x.shape[1])),
            'intercept': np.zeros(self.nlambda),
        }

    def train(self):
        return None


early_stop_solver = pycasso.Solver(
    overlap_x, overlap_y, lambdas=np.array([0.3, 0.2, 0.1]),
    standardize=False)
early_stop_solver.train()
try:
    pycasso_core.Solver = EarlyStopFoldSolver
    early_serial = early_stop_solver.cross_validate(
        foldid=overlap_foldid, type_measure='mse', n_jobs=1)
    early_parallel = early_stop_solver.cross_validate(
        foldid=overlap_foldid, type_measure='mse', n_jobs=3)
finally:
    pycasso_core.Solver = parallel_solver_class
assert len(early_parallel['lambda']) == 2, \
    "parallel CV ignored the certified common early-stop prefix"
assert_cv_results_identical(
    early_serial, early_parallel, "parallel early-stop prefix")
print(f"  lambda_min={cv['lambda_min']:.4f}, lambda_1se={cv['lambda_1se']:.4f}")
print("  PASS")

# Step 6: multinomial
print("\n=== Step 6: multinomial ===")
assert pycasso_core._MULTINOMIAL_STATUS_NAMES[10] == \
    'lla_stationarity_limit', "Python status map is missing adaptive LLA limit"
mn_rng = np.random.RandomState(20260715)
n_mn, d_mn, k_mn = 72, 6, 3
Y_mn3 = np.arange(n_mn) % k_mn
X_mn = mn_rng.normal(scale=0.6, size=(n_mn, d_mn))
X_mn[np.arange(n_mn), Y_mn3] += 1.5


def assert_multinomial_fit(solver, expected_nlambda, label):
    assert solver.nlambda == expected_nlambda, \
        f"{label} nlambda: {solver.nlambda} != {expected_nlambda}"
    assert solver.result['beta'].shape == (expected_nlambda, k_mn, d_mn), \
        f"{label} beta shape: {solver.result['beta'].shape}"
    assert solver.result['intercept'].shape == (expected_nlambda, k_mn), \
        f"{label} intercept shape: {solver.result['intercept'].shape}"
    assert solver.result['df'].shape == (expected_nlambda,), \
        f"{label} df shape: {solver.result['df'].shape}"
    assert np.all(np.isfinite(solver.result['beta'])), \
        f"{label} beta contains non-finite values"
    assert np.all(np.isfinite(solver.result['intercept'])), \
        f"{label} intercept contains non-finite values"
    probs = solver.predict(X_mn[:7], type="response")
    assert probs.shape == (7, k_mn), f"{label} probs shape: {probs.shape}"
    assert np.all(np.isfinite(probs)), f"{label} probs are non-finite"
    assert np.all((probs >= 0) & (probs <= 1)), \
        f"{label} probs are outside [0, 1]"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-7), \
        f"{label} probs don't sum to 1"
    return probs


def assert_multinomial_diagnostics(solver, expected_status, label):
    """Check versioned native status and fitted-prefix diagnostics."""
    result = solver.result
    assert result['status_code'] == expected_status, \
        f"{label} status code: {result['status_code']}"
    expected_name = pycasso_core._MULTINOMIAL_STATUS_NAMES[expected_status]
    assert result['status'] == expected_name, \
        f"{label} status: {result['status']}"
    assert result['failed_lambda'] == -1, \
        f"{label} unexpectedly reports failed lambda {result['failed_lambda']}"
    assert result['stage'] == -1, \
        f"{label} unexpectedly reports failed stage {result['stage']}"

    for field in ('train_time', 'runtime', 'outer_ite', 'inner_sweeps',
                  'coordinate_updates', 'objective', 'kkt',
                  'stationarity'):
        values = result[field]
        assert values.shape == (solver.nlambda,), \
            f"{label} {field} shape: {values.shape}"
    assert result['inner_sweeps'].dtype == np.int64, \
        f"{label} inner_sweeps must retain native int64 counts"
    assert result['coordinate_updates'].dtype == np.int64, \
        f"{label} coordinate_updates must retain native int64 counts"
    assert np.all(np.isfinite(result['train_time'])), \
        f"{label} runtime contains non-finite values"
    assert np.all(result['train_time'] >= 0) and \
        np.any(result['train_time'] > 0), \
        f"{label} does not expose real per-lambda runtimes"
    assert np.array_equal(result['runtime'], result['train_time']), \
        f"{label} runtime alias differs from train_time"
    assert np.all(result['outer_ite'] >= 0), \
        f"{label} outer iterations are negative"
    assert np.all(result['inner_sweeps'] >= 0), \
        f"{label} inner sweeps are negative"
    assert np.all(result['coordinate_updates'] >= 0), \
        f"{label} coordinate updates are negative"
    for field in ('objective', 'kkt', 'stationarity'):
        assert np.all(np.isfinite(result[field])), \
            f"{label} {field} contains non-finite fitted diagnostics"


def expect_value_error(callback, label, message_part=None):
    try:
        callback()
    except ValueError as exc:
        if message_part is not None:
            assert message_part.lower() in str(exc).lower(), \
                f"{label} error did not mention {message_part!r}: {exc}"
        return
    raise AssertionError(f"{label} should raise ValueError")


# Default multinomial fit must be repeatable on the same Solver instance.
sm = pycasso.Solver(X_mn, Y_mn3, family="multinomial")
sm_fast_mode = pycasso.Solver(
    X_mn, Y_mn3, lambdas=np.array([0.2, 0.1]),
    family="multinomial", fast_mode=True)
assert sm_fast_mode.fast_mode is True and sm_fast_mode.prec == 1e-4, \
    "multinomial did not inherit fast mode"
assert sm.lla_max_stages == 3, \
    "multinomial default lla_max_stages must be three"
sm.train()
default_nlambda = sm.nlambda
probs_mn = assert_multinomial_fit(sm, default_nlambda, "multinomial L1")
assert_multinomial_diagnostics(sm, 0, "multinomial L1")
first_default = {
    key: sm.result[key].copy()
    for key in ('beta', 'intercept', 'ite_lamb', 'size_act', 'df', 'dev_ratio')
}
first_lambdas = sm.lambdas.copy()
sm.train()
repeat_probs = assert_multinomial_fit(sm, default_nlambda,
                                      "repeated multinomial L1")
for key in ('beta', 'intercept', 'dev_ratio'):
    assert np.allclose(sm.result[key], first_default[key],
                       rtol=1e-9, atol=1e-11), \
        f"repeated multinomial {key} changed"
for key in ('ite_lamb', 'size_act', 'df'):
    assert np.array_equal(sm.result[key], first_default[key]), \
        f"repeated multinomial {key} changed"
assert np.array_equal(sm.lambdas, first_lambdas), \
    "repeated multinomial lambda path changed"
assert np.allclose(repeat_probs, probs_mn, rtol=1e-9, atol=1e-11), \
    "repeated multinomial probabilities changed"

cls_mn = sm.predict(X_mn[:5], type="class")
assert cls_mn.shape == (5,), f"multinomial class shape: {cls_mn.shape}"
assert set(cls_mn).issubset({0, 1, 2}), f"multinomial class out of range: {set(cls_mn)}"
nz_mn = sm.predict(X_mn[:5], type="nonzero")
assert isinstance(nz_mn, list) and len(nz_mn) == 3, "multinomial nonzero"
link_mn = sm.predict(X_mn[:5], type="link")
assert link_mn.shape == (5, 3), f"multinomial link shape: {link_mn.shape}"
conf_mn = sm.confusion(X_mn[:12], Y_mn3[:12], lambdidx=[0, 1])
assert len(conf_mn) == 2 and all(cm.shape == (k_mn, k_mn)
                                 for cm in conf_mn), \
    "multinomial confusion matrix shape"
assert all(cm.sum() == 12 for cm in conf_mn), \
    "multinomial confusion matrices lost observations"

# Use an explicit, short path for the nonconvex LLA penalties.
X_mn_std = X_mn - X_mn.mean(axis=0)
X_mn_std *= 1.0 / np.sqrt(
    np.sum(X_mn_std ** 2, axis=0) / (n_mn - 1))
p0_mn = np.bincount(Y_mn3, minlength=k_mn).astype(float) / n_mn
lambda_max_mn = max(
    np.max(np.abs(X_mn_std.T @ ((Y_mn3 == klass).astype(float)
                                - p0_mn[klass]))) / n_mn
    for klass in range(k_mn)
)

# A successful glmnet-style deviance stop is a trained prefix, not a native
# failure.  The prefix must remain usable by prediction and cross-validation.
explicit_saturated_path = lambda_max_mn * 0.45 * (
    1.0 - 1e-7 * np.arange(12, dtype=float))
sm_explicit = pycasso.Solver(
    X_mn, Y_mn3, lambdas=explicit_saturated_path, family="multinomial",
    prec=5e-7, max_ite=5000)
sm_explicit.train()
assert not sm_explicit.result['path_early_stopped'] and \
    sm_explicit.nlambda == 12, \
    "an explicit multinomial lambda path was truncated"

# Use a separate, strongly separable fixture so the documented glmnet
# threshold is deterministically exercised rather than depending on a noisy
# sample happening to saturate.
early_rng = np.random.RandomState(20260716)
X_mn_early = early_rng.normal(scale=0.15, size=(n_mn, d_mn))
X_mn_early[np.arange(n_mn), Y_mn3] += 2.0
sm_early = pycasso.Solver(
    X_mn_early, Y_mn3, lambdas=(100, 1e-4), family="multinomial",
    prec=5e-7, max_ite=5000)
with warnings.catch_warnings(record=True) as early_warnings:
    warnings.simplefilter("always")
    sm_early.train()
assert not early_warnings, \
    "normal multinomial path early stopping emitted a failure warning"
assert sm_early.result['state'] == 'trained' and \
    sm_early.result['status_code'] == 0 and \
    sm_early.result['path_early_stopped'], \
    "normal multinomial path early stopping was not marked as trained"
assert sm_early.result['requested_nlambda'] == 100 and \
    5 <= sm_early.nlambda < 100 and \
    int(sm_early.result['num_fit'][0]) == sm_early.nlambda, \
    "multinomial early-stop path length metadata is inconsistent"
assert_multinomial_fit(sm_early, sm_early.nlambda,
                       "early-stopped multinomial L1")
cv_early = sm_early.cross_validate(nfolds=3, type_measure="class")
assert len(cv_early['lambda']) == sm_early.nlambda and \
    np.all(np.isfinite(cv_early['cvm'])), \
    "early-stopped multinomial path is not cross-validation usable"

nonconvex_path = lambda_max_mn * np.array([1.2, 0.85, 0.55, 0.35])
default_lla_statuses = []
for penalty in ("mcp", "scad"):
    sm_nc = pycasso.Solver(X_mn, Y_mn3, lambdas=nonconvex_path,
                           family="multinomial", penalty=penalty,
                           prec=1e-6, max_ite=200)
    with warnings.catch_warnings(record=True) as capped_warnings:
        warnings.simplefilter("always")
        sm_nc.train()
    assert not capped_warnings, \
        f"multinomial {penalty.upper()} stage cap was treated as a failure"
    assert_multinomial_fit(sm_nc, len(nonconvex_path),
                           f"multinomial {penalty.upper()}")
    assert sm_nc.result['status_code'] in (0, 10), \
        f"multinomial {penalty.upper()} default stage status"
    default_lla_statuses.append(sm_nc.result['status_code'])
    assert_multinomial_diagnostics(
        sm_nc, sm_nc.result['status_code'],
        f"multinomial {penalty.upper()} default stages")
    assert sm_nc.result['state'] == 'trained', \
        f"multinomial {penalty.upper()} capped model is not usable"

    # A larger cap lets the adaptive rule continue beyond the usual three
    # stages when stricter nonconvex stationarity is required.
    sm_nc_strict = pycasso.Solver(
        X_mn, Y_mn3, lambdas=nonconvex_path, family="multinomial",
        penalty=penalty, prec=1e-6, max_ite=200, lla_max_stages=25)
    sm_nc_strict.train()
    assert_multinomial_fit(
        sm_nc_strict, len(nonconvex_path),
        f"multinomial {penalty.upper()} raised LLA budget")
    assert_multinomial_diagnostics(
        sm_nc_strict, 0,
        f"multinomial {penalty.upper()} raised LLA budget")
    assert np.all(sm_nc_strict.result['stationarity'] <= sm_nc_strict.prec), \
        f"multinomial {penalty.upper()} did not reach target stationarity"
    assert np.max(sm_nc_strict.result['stationarity']) <= \
        np.max(sm_nc.result['stationarity']) + 1e-12, \
        f"multinomial {penalty.upper()} higher LLA budget regressed stationarity"
assert 10 in default_lla_statuses, \
    "the deterministic default-stage test did not exercise usable status 10"

# A V2 failure after a valid prefix must be visible and keep that prefix usable.
sm_partial = pycasso.Solver(
    X_mn, Y_mn3, lambdas=nonconvex_path, family="multinomial",
    penalty="mcp", prec=1e-7, max_ite=1)
with warnings.catch_warnings(record=True) as partial_warnings:
    warnings.simplefilter("always")
    sm_partial.train()
assert len(partial_warnings) == 1 and \
    issubclass(partial_warnings[0].category, RuntimeWarning), \
    "partial V2 failure must emit exactly one RuntimeWarning"
assert sm_partial.result['state'] == 'partially trained', \
    "partial V2 failure state was hidden"
assert sm_partial.result['status_code'] == 4 and \
    sm_partial.result['status'] == 'inner_iteration_limit', \
    f"partial V2 status was lost: {sm_partial.result['status']}"
assert sm_partial.result['failed_lambda'] == 1, \
    f"partial V2 failed lambda: {sm_partial.result['failed_lambda']}"
assert sm_partial.result['stage'] == 0, \
    f"partial MCP failed stage: {sm_partial.result['stage']}"
assert sm_partial.nlambda == 1 and int(sm_partial.result['num_fit'][0]) == 1, \
    "partial V2 fit did not retain exactly its committed prefix"
assert_multinomial_fit(sm_partial, 1, "partial multinomial V2")
for field in ('train_time', 'runtime', 'outer_ite', 'inner_sweeps',
              'coordinate_updates', 'objective', 'kkt', 'stationarity'):
    assert sm_partial.result[field].shape == (1,), \
        f"partial V2 {field} is not aligned to the fitted prefix"
failed_diag = sm_partial.result['failure_diagnostics']
assert failed_diag is not None and set(failed_diag) == {
    'lambda', 'train_time', 'runtime', 'outer_ite', 'inner_sweeps',
    'coordinate_updates', 'objective', 'kkt', 'stationarity'
}, "failed-point V2 diagnostics were not retained"
assert failed_diag['train_time'] > 0 and \
    np.isfinite(failed_diag['objective']), \
    "failed-point runtime/objective diagnostics are unavailable"

# Retrying must restore the immutable requested path, so a truncated failure
# cannot be silently reclassified as a complete one-lambda fit.
with warnings.catch_warnings(record=True) as retry_warnings:
    warnings.simplefilter("always")
    sm_partial.train()
assert len(retry_warnings) == 1 and \
    sm_partial.result['status_code'] == 4 and \
    sm_partial.result['state'] == 'partially trained' and \
    sm_partial.nlambda == 1, \
    "retrying a partial fit hid the original full-path failure"
try:
    sm_partial.cross_validate(nfolds=3)
    raise AssertionError("partial multinomial path should not enter CV")
except PycassoError as exc:
    assert 'partially trained' in str(exc), \
        f"partial-path CV error was unclear: {exc}"

# A failure before the first committed point must raise, not return an empty fit.
no_prefix_path = lambda_max_mn * np.array([0.2, 0.1])
sm_no_prefix = pycasso.Solver(
    X_mn, Y_mn3, lambdas=no_prefix_path, family="multinomial",
    prec=1e-7, max_ite=1)
try:
    sm_no_prefix.train()
    raise AssertionError("zero-prefix V2 failure should raise PycassoError")
except PycassoError as exc:
    assert 'inner_iteration_limit' in str(exc), \
        f"zero-prefix error omitted native status: {exc}"
assert sm_no_prefix.result['state'] == 'not trained', \
    "zero-prefix failure must not look trained"
assert sm_no_prefix.result['status_code'] == 4 and \
    sm_no_prefix.result['failed_lambda'] == 0, \
    "zero-prefix V2 failure diagnostics were lost"
assert sm_no_prefix.result['outer_ite'].shape == no_prefix_path.shape, \
    "zero-prefix diagnostics should remain inspectable after the error"
assert sm_no_prefix.result['failure_diagnostics'] is not None, \
    "zero-prefix failed-point diagnostics are missing"

# Original numeric and string labels must survive the internal 0..K-1 coding.
string_level_lookup = np.array(["zebra", "apple", "mango"])
Y_mn_string = string_level_lookup[Y_mn3]
sm_string = pycasso.Solver(X_mn, Y_mn_string, lambdas=nonconvex_path,
                           family="multinomial", prec=1e-6, max_ite=200)
sm_string.train()
pred_string = sm_string.predict(X_mn[:9], type="class")
assert pred_string.shape == (9,), "string-label class prediction shape"
assert set(pred_string).issubset(set(Y_mn_string)), \
    f"string labels were not restored: {set(pred_string)}"
assert np.array_equal(sm_string.result['levels'], np.unique(Y_mn_string)), \
    "string levels were not retained in the public result"
sm_string.result['levels'][0] = "mutated"
sm_string.train()
assert np.array_equal(sm_string.result['levels'], sm_string._mn_levels), \
    "multinomial retraining did not restore the private class map"
assessed_string = sm_string.assess(X_mn[:12], Y_mn_string[:12])
assert np.all(np.isfinite(assessed_string['deviance'])), \
    "string-label assess deviance is non-finite"


def legacy_multinomial_assessment(y_codes, x, beta, intercept):
    """One-model-at-a-time assessment used as an exact output oracle."""
    deviances = pycasso_core._mn_fit_deviances(
        y_codes, x, beta, intercept)
    predictions = np.array([
        np.argmax(x @ beta[i].T + intercept[i], axis=1)
        for i in range(beta.shape[0])
    ])
    class_errors = np.mean(
        predictions != y_codes[np.newaxis, :], axis=1)
    return deviances, class_errors


# The fused path must preserve the old implementation exactly for a public
# subset assessment while forming logits only once per fitted lambda.  The
# Classification uses the same raw-logit argmax as predict/confusion/CV and
# therefore does not allocate a probability matrix.
_, _, string_subset_codes = pycasso_core._encode_multinomial_labels(
    Y_mn_string[:12], sm_string._mn_levels, name='newy')
legacy_subset_deviance, legacy_subset_error = \
    legacy_multinomial_assessment(
        string_subset_codes, X_mn[:12], sm_string.result['beta'],
        sm_string.result['intercept'])
assert np.array_equal(
    assessed_string['deviance'], legacy_subset_deviance) and \
    np.array_equal(assessed_string['class_error'], legacy_subset_error), \
    "fused multinomial subset assessment changed public metrics"

assessment_counts = {'logits': 0, 'nll': 0, 'softmax': 0}
original_assessment_logits = pycasso_core._mn_assessment_logits
original_multinomial_nll = pycasso_core._multinomial_nll_from_logits
original_softmax = pycasso_core._softmax


def counting_assessment_logits(*args, **kwargs):
    assessment_counts['logits'] += 1
    return original_assessment_logits(*args, **kwargs)


def counting_multinomial_nll(*args, **kwargs):
    assessment_counts['nll'] += 1
    return original_multinomial_nll(*args, **kwargs)


def counting_softmax(*args, **kwargs):
    assessment_counts['softmax'] += 1
    return original_softmax(*args, **kwargs)


try:
    pycasso_core._mn_assessment_logits = counting_assessment_logits
    pycasso_core._multinomial_nll_from_logits = counting_multinomial_nll
    pycasso_core._softmax = counting_softmax
    counted_assessment = sm_string.assess(
        X_mn[:12], Y_mn_string[:12])
finally:
    pycasso_core._mn_assessment_logits = original_assessment_logits
    pycasso_core._multinomial_nll_from_logits = original_multinomial_nll
    pycasso_core._softmax = original_softmax
assert assessment_counts == {
    'logits': sm_string.nlambda,
    'nll': sm_string.nlambda,
    'softmax': 0,
}, f"multinomial assess recomputed its path: {assessment_counts}"
assert np.array_equal(
    counted_assessment['deviance'], assessed_string['deviance']) and \
    np.array_equal(
        counted_assessment['class_error'], assessed_string['class_error']), \
    "assessment instrumentation changed fused metrics"

# Exercise exact ties and logits large enough to underflow naive softmax
# probabilities.  np.argmax uses the first class for a tie in both versions.
assessment_oracle_x = np.array([
    [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0],
    [0.0, 1.0], [0.0, -1.0],
])
assessment_oracle_beta = np.array([
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
])
assessment_oracle_intercept = np.array([
    [0.0, 0.0, 0.0],
    [1000.0, 0.0, -1000.0],
    [0.0, 0.0, 0.0],
])
assessment_oracle_codes = np.array([0, 1, 2, 1, 0], dtype=np.intp)
legacy_oracle_metrics = legacy_multinomial_assessment(
    assessment_oracle_codes, assessment_oracle_x,
    assessment_oracle_beta, assessment_oracle_intercept)
fused_oracle_metrics = pycasso_core._mn_assessment_metrics(
    assessment_oracle_codes, assessment_oracle_x,
    assessment_oracle_beta, assessment_oracle_intercept)
for metric_name, legacy_values, fused_values in zip(
        ('deviance', 'class_error'), legacy_oracle_metrics,
        fused_oracle_metrics):
    assert np.array_equal(fused_values, legacy_values), \
        f"fused multinomial {metric_name} changed on extremes or ties"

# A higher logit can round to the same softmax probability within a fraction
# of one ulp. Classification must retain the strictly larger raw logit.
assessment_ulp = np.spacing(1.0)
near_tie_intercept = np.array([
    [0.0, 0.125 * assessment_ulp, -2.0],
    [0.0, 0.250 * assessment_ulp, -2.0],
    [0.0, 0.500 * assessment_ulp, -2.0],
])
near_tie_x = np.zeros((1, 1))
near_tie_beta = np.zeros((3, 3, 1))
near_tie_codes = np.zeros(1, dtype=np.intp)
near_tie_logits = near_tie_intercept.copy()
near_tie_softmax_classes = np.argmax(
    pycasso_core._softmax(near_tie_logits), axis=1)
near_tie_raw_classes = np.argmax(near_tie_logits, axis=1)
# The sub-ulp offsets sit at or near exp()'s rounding boundary, so which of
# the three rows collapses to a softmax tie is libm-specific. The fixture
# only needs one collapsed row while every raw logit stays strictly larger.
assert np.any(near_tie_softmax_classes == 0) and \
    np.array_equal(near_tie_raw_classes, [1, 1, 1]), \
    "near-tie oracle does not exercise softmax rounding"
legacy_near_tie_metrics = legacy_multinomial_assessment(
    near_tie_codes, near_tie_x, near_tie_beta, near_tie_intercept)
fused_near_tie_metrics = pycasso_core._mn_assessment_metrics(
    near_tie_codes, near_tie_x, near_tie_beta, near_tie_intercept)
assert np.array_equal(fused_near_tie_metrics[1], [1.0, 1.0, 1.0]), \
    "multinomial assessment must classify directly from the larger logit"
for metric_name, legacy_values, fused_values in zip(
        ('deviance', 'class_error'), legacy_near_tie_metrics,
        fused_near_tie_metrics):
    assert np.array_equal(fused_values, legacy_values), \
        f"fused multinomial {metric_name} changed near-tie semantics"


def assessment_error(call):
    try:
        call()
    except ValueError as exc:
        return type(exc), str(exc)
    raise AssertionError("non-finite assessment logits were accepted")


overflow_x = np.array([[np.finfo(float).max]])
overflow_beta = np.array([[[2.0], [0.0], [-2.0]]])
overflow_intercept = np.zeros((1, 3))
overflow_codes = np.array([0], dtype=np.intp)
with np.errstate(over='ignore', invalid='ignore'):
    legacy_assessment_error = assessment_error(
        lambda: legacy_multinomial_assessment(
            overflow_codes, overflow_x, overflow_beta,
            overflow_intercept))
    fused_assessment_error = assessment_error(
        lambda: pycasso_core._mn_assessment_metrics(
            overflow_codes, overflow_x, overflow_beta,
            overflow_intercept))
assert fused_assessment_error == legacy_assessment_error, \
    "fused multinomial assess changed non-finite-logit error semantics"
expect_value_error(lambda: sm_string.assess(
    X_mn[:12], Y_mn_string[:12], newoffset=np.zeros(12)),
    "multinomial assessment offset", "only supported")

numeric_level_lookup = np.array([30, 10, 70])
Y_mn_numeric = numeric_level_lookup[Y_mn3]
sm_numeric = pycasso.Solver(X_mn, Y_mn_numeric, lambdas=nonconvex_path,
                            family="multinomial", prec=1e-6, max_ite=200)
sm_numeric.train()
pred_numeric = sm_numeric.predict(X_mn[:9], type="class")
assert set(pred_numeric).issubset(set(Y_mn_numeric)), \
    f"numeric labels were not restored: {set(pred_numeric)}"

unseen = Y_mn_string[:12].copy()
unseen[0] = "other"
expect_value_error(lambda: sm_string.assess(X_mn[:12], unseen),
                   "unseen assess label", "unseen")

Y_none = Y_mn_string.astype(object)
Y_none[0] = None
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_none, family="multinomial"), "None multinomial label", "missing")
Y_nan = Y_mn_numeric.astype(float)
Y_nan[0] = np.nan
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_nan, family="multinomial"), "NaN multinomial label", "finite")
Y_inf = Y_mn_numeric.astype(float)
Y_inf[0] = np.inf
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_inf, family="multinomial"), "infinite multinomial label", "finite")

# Multinomial CV must use the full-data mapping and stratified folds.
np.random.seed(20260715)
sm_cv = pycasso.Solver(X_mn, Y_mn_string, lambdas=nonconvex_path,
                       family="multinomial", prec=1e-6, max_ite=200,
                       lla_max_stages=7)
fold_lla_budgets = []
original_solver_class = pycasso_core.Solver


def recording_fold_solver(*args, **kwargs):
    fold_lla_budgets.append(kwargs.get('lla_max_stages'))
    return original_solver_class(*args, **kwargs)


try:
    pycasso_core.Solver = recording_fold_solver
    cv_class = sm_cv.cross_validate(nfolds=3, type_measure="class")
finally:
    pycasso_core.Solver = original_solver_class
assert fold_lla_budgets == [7, 7, 7], \
    f"CV folds lost lla_max_stages: {fold_lla_budgets}"
assert cv_class['name'] == 'class', "multinomial class CV name"
assert sm_cv.result['state'] == 'trained', \
    "fresh multinomial CV must establish a full-data fit"
assert np.array_equal(cv_class['nzero'], sm_cv.result['df']), \
    "fresh multinomial CV reported false zero model sizes"
assert np.all(np.isfinite(cv_class['cvm'])), "multinomial class CV is non-finite"
assert np.all((cv_class['cvm'] >= 0) & (cv_class['cvm'] <= 1)), \
    "multinomial class CV must be a misclassification rate"
for fold in range(3):
    held_out = Y_mn_string[cv_class['foldid'] == fold]
    assert set(held_out) == set(Y_mn_string), \
        f"default multinomial fold {fold} is not stratified"

cv_deviance = sm_cv.cross_validate(
    foldid=cv_class['foldid'], type_measure="deviance")
assert cv_deviance['name'] == 'deviance', "multinomial deviance CV name"
assert np.all(np.isfinite(cv_deviance['cvm'])), \
    "multinomial deviance CV is non-finite"
assert np.all(cv_deviance['cvm'] >= 0), "multinomial deviance must be nonnegative"
assert not np.allclose(cv_deviance['cvm'], cv_class['cvm']), \
    "multinomial class and deviance CV must not be the same calculation"

cv_default = sm_cv.cross_validate(foldid=cv_class['foldid'])
assert cv_default['name'] == 'class', \
    "default multinomial CV measure must match the R class-loss default"
assert np.allclose(cv_default['cvm'], cv_class['cvm']), \
    "default multinomial CV did not compute class loss"

rare_labels = np.array(["a"] * 35 + ["b"] * 35 + ["c"] * 2)
rare_cv = pycasso.Solver(
    X_mn, rare_labels, lambdas=nonconvex_path,
    family="multinomial").cross_validate(nfolds=3, type_measure="class")
assert np.all(np.isfinite(rare_cv['cvm'])), \
    "valid rare-class CV was rejected"
singleton_labels = np.array(["a"] * 36 + ["b"] * 35 + ["c"])
expect_value_error(lambda: pycasso.Solver(
    X_mn, singleton_labels, lambdas=nonconvex_path,
    family="multinomial").cross_validate(nfolds=3),
    "singleton-class CV", "at least two")
rare_bad_foldid = np.arange(n_mn) % 3
rare_bad_foldid[-2:] = 0
expect_value_error(lambda: pycasso.Solver(
    X_mn, rare_labels, lambdas=nonconvex_path,
    family="multinomial").cross_validate(foldid=rare_bad_foldid),
    "missing training class CV", "training fold")
expect_value_error(lambda: sm_cv.cross_validate(
    foldid=np.zeros(n_mn - 1, dtype=int)), "short foldid", "length")
bad_foldid = np.arange(n_mn) % 3
bad_foldid[0] = 4
expect_value_error(lambda: sm_cv.cross_validate(foldid=bad_foldid),
                   "noncontiguous foldid", "contiguous")
expect_value_error(lambda: sm_cv.cross_validate(
    foldid=np.linspace(0, 2, n_mn)), "fractional foldid", "integer")
expect_value_error(lambda: sm_cv.cross_validate(
    nfolds=3, type_measure="mse"), "unsupported multinomial CV measure",
    "multinomial")
expect_value_error(lambda: sm.predict(X_mn[:2], lam=np.nan),
                   "NaN prediction lambda", "finite")
expect_value_error(lambda: sm.predict(X_mn[:2], lam=-0.1),
                   "negative prediction lambda", "nonnegative")
expect_value_error(lambda: sm.predict(X_mn[:2], lambdidx=-1),
                   "negative prediction index", "between")
expect_value_error(lambda: sm.predict(X_mn[:2], lambdidx=sm.nlambda),
                   "oversized prediction index", "between")
huge_direction = np.sign(sm.result['beta'][-1, 0])
huge_direction[huge_direction == 0] = 1
with np.errstate(over='ignore', invalid='ignore'):
    expect_value_error(lambda: sm.predict(
        np.asarray([huge_direction * np.finfo(float).max]),
        type="response"), "non-finite multinomial logits", "finite")

# Key validation must happen before entering ctypes/native code.
X_bad = X_mn.copy()
X_bad[0, 0] = np.nan
expect_value_error(lambda: pycasso.Solver(
    X_bad, Y_mn3, family="multinomial"), "non-finite X", "finite")

# Numeric interface arrays must already be real numeric data.  Silently
# parsing strings or discarding complex imaginary parts differs from R and can
# change a model without an error.
for invalid_numeric, invalid_label in (
        (X_mn.astype(str), "numeric-string X"),
        (X_mn.astype(complex) + 1j, "complex X")):
    expect_value_error(lambda values=invalid_numeric: pycasso.Solver(
        values, Y_mn3, family="multinomial"), invalid_label, "real numeric")
for invalid_response, invalid_label in (
        (Y_g.astype(str), "numeric-string response"),
        (Y_g.astype(complex) + 1j, "complex response")):
    expect_value_error(lambda values=invalid_response: pycasso.Solver(
        X, values, family="gaussian"), invalid_label, "real numeric")
for invalid_offset, invalid_label in (
        (np.zeros(n).astype(str), "numeric-string offset"),
        (np.zeros(n, dtype=complex) + 1j, "complex offset")):
    expect_value_error(lambda values=invalid_offset: pycasso.Solver(
        X, Y_b, family="binomial", offset=values),
        invalid_label, "real numeric")
for invalid_lambdas, invalid_label in (
        (np.array(["0.2", "0.1"]), "numeric-string lambdas"),
        (np.array([0.2 + 1j, 0.1 + 1j]), "complex lambdas")):
    expect_value_error(lambda values=invalid_lambdas: pycasso.Solver(
        X, Y_g, lambdas=values), invalid_label, "real numeric")

expect_value_error(lambda: s.predict(X[:2].astype(str)),
                   "numeric-string prediction data", "real numeric")
expect_value_error(lambda: s.assess(
    X[:5], Y_g[:5].astype(str)),
    "numeric-string assessment response", "real numeric")
expect_value_error(lambda: sm.confusion(
    X_mn[:6].astype(complex) + 1j, Y_mn3[:6]),
    "complex confusion data", "real numeric")
expect_value_error(lambda: sm_cv.cross_validate(
    foldid=(np.arange(n_mn) % 3).astype(str)),
    "numeric-string foldid", "nonnegative integers")

for invalid_count in (1.000009, 1000000.1):
    invalid_poisson = Y_p.copy()
    invalid_poisson[0] = invalid_count
    expect_value_error(lambda values=invalid_poisson: pycasso.Solver(
        X, values, family="poisson"),
        f"non-integer Poisson count {invalid_count}", "integers")
masked_x = np.ma.array(X_mn, mask=False)
masked_x.mask[0, 0] = True
masked_y = np.ma.array(Y_mn3, mask=False)
masked_y.mask[0] = True
masked_lambdas = np.ma.array(nonconvex_path, mask=False)
masked_lambdas.mask[0] = True
masked_offset = np.ma.array(np.zeros(n), mask=False)
masked_offset.mask[0] = True
masked_foldid = np.ma.array(np.arange(n_mn) % 3, mask=False)
masked_foldid.mask[0] = True
all_observed_masked_x = np.ma.array(X_mn, mask=False)
for masked_call, masked_label in (
        (lambda: pycasso.Solver(
            all_observed_masked_x, Y_mn3, family="multinomial"),
         "all-observed masked X"),
        (lambda: pycasso.Solver(
            masked_x, Y_mn3, family="multinomial"), "masked X"),
        (lambda: pycasso.Solver(
            X_mn, masked_y, family="multinomial"), "masked y"),
        (lambda: pycasso.Solver(
            X_mn, Y_mn3, lambdas=masked_lambdas,
            family="multinomial"), "masked lambdas"),
        (lambda: pycasso.Solver(
            X, Y_b, family="binomial",
            offset=masked_offset), "masked offset"),
        (lambda: sm.predict(masked_x[:2]), "masked prediction data"),
        (lambda: sm.assess(masked_x, Y_mn3), "masked assessment data"),
        (lambda: sm.confusion(
            masked_x, Y_mn3), "masked confusion data"),
        (lambda: sm_cv.cross_validate(
            foldid=masked_foldid), "masked foldid")):
    expect_value_error(masked_call, masked_label, "masked array")
for invalid_codes, invalid_label in (
        (np.array([0.5]), "fractional multinomial code"),
        (np.array([np.nan]), "NaN multinomial code")):
    expect_value_error(lambda values=invalid_codes:
                       pycasso_core._multinomial_nll_from_logits(
                           values, np.zeros((1, 3))),
                       invalid_label, "finite integers")
native_oversize_view = np.lib.stride_tricks.as_strided(
    np.zeros(1), shape=(46341, 46341), strides=(0, 0))
expect_value_error(lambda: pycasso.Solver(
    native_oversize_view, np.zeros(1), family="multinomial",
    standardize=False), "oversize native design", "native")
oversize_nlambda = np.iinfo(np.int32).max // (d_mn * k_mn) + 1
original_linspace = pycasso_core.np.linspace
linspace_called = [False]


def forbidden_oversize_linspace(*args, **kwargs):
    linspace_called[0] = True
    raise AssertionError("oversize native output reached np.linspace")


try:
    pycasso_core.np.linspace = forbidden_oversize_linspace
    expect_value_error(lambda: pycasso.Solver(
        X_mn, Y_mn3, lambdas=(oversize_nlambda, 0.05),
        family="multinomial"), "oversize generated path", "native")
finally:
    pycasso_core.np.linspace = original_linspace
assert not linspace_called[0], \
    "oversize generated path allocated before native count validation"
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, family="multinomial", gamma=np.nan),
    "NaN gamma", "gamma")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, family="multinomial", penalty="mcp", gamma=1),
    "MCP gamma boundary", "greater than 1")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, family="multinomial", penalty="scad", gamma=2),
    "SCAD gamma boundary", "greater than 2")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, family="multinomial", prec=0),
    "zero precision", "prec")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, family="multinomial", max_ite=1.5),
    "fractional max_ite", "integer")
for invalid_lla_stages in (True, 2, 3.5, np.nan, np.inf):
    expect_value_error(lambda value=invalid_lla_stages: pycasso.Solver(
        X_mn, Y_mn3, family="multinomial", lla_max_stages=value),
        f"invalid lla_max_stages {invalid_lla_stages!r}",
        "lla_max_stages")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, family="multinomial", dfmax=-2),
    "dfmax below sentinel", "dfmax")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, lambdas=[0.2, 0.1, 0.15], family="multinomial"),
    "unordered explicit lambda", "decreasing")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, lambdas=[0.2, 0.1, 0.1], family="multinomial"),
    "duplicate explicit lambda", "decreasing")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, lambdas=(0, 0.05), family="multinomial"),
    "zero generated nlambda", "positive")
expect_value_error(lambda: pycasso.Solver(
    X_mn, Y_mn3, lambdas=(5, 0), family="multinomial"),
    "zero lambda_min_ratio", "positive")

# A zero-gradient design still needs a finite, usable generated path.
X_zero = np.zeros((n_mn, 3))
sm_zero = pycasso.Solver(X_zero, Y_mn_string, lambdas=(4, 0.1),
                         family="multinomial", prec=1e-6, max_ite=200)
assert np.all(np.isfinite(sm_zero.lambdas)), "zero-gradient lambda path is non-finite"
assert np.all(np.diff(sm_zero.lambdas) < 0), \
    "zero-gradient lambda path must remain strictly decreasing"
sm_zero.train()
zero_probs = sm_zero.predict(X_zero[:4], type="response")
assert np.allclose(zero_probs.sum(axis=1), 1.0), \
    "zero-gradient multinomial probabilities do not sum to one"

# No-intercept preprocessing must preserve the origin and use the uniform
# softmax null model for both lambda_max and null deviance.
Y_no_intercept = np.array([0] * 24 + [1] * 4 + [2] * 2)
X_no_intercept = np.ones((Y_no_intercept.size, 1))
sm_no_intercept = pycasso.Solver(
    X_no_intercept, Y_no_intercept, lambdas=(4, 0.1),
    family="multinomial", useintercept=False, standardize=False,
    prec=1e-7, max_ite=500)
uniform_p0 = np.full(3, 1.0 / 3.0)
expected_lambda_max = max(
    np.max(np.abs(X_no_intercept.T @
                  ((Y_no_intercept == klass).astype(float) -
                   uniform_p0[klass]))) / Y_no_intercept.size
    for klass in range(3))
assert np.isclose(sm_no_intercept.lambdas[0], expected_lambda_max), \
    "no-intercept lambda_max did not use uniform null probabilities"
sm_no_intercept.train()
assert sm_no_intercept.result['status'] == 'completed', \
    "corrected no-intercept lambda path did not converge"
assert sm_no_intercept.result['df'][0] == 0, \
    "first no-intercept lambda is not a valid lambda_max"
assert np.array_equal(sm_no_intercept.result['intercept'],
                      np.zeros_like(sm_no_intercept.result['intercept'])), \
    "no-intercept fit returned nonzero intercepts"
assert np.isclose(sm_no_intercept.result['nulldev'], np.log(3.0)), \
    "no-intercept null deviance did not use the uniform softmax model"
assert np.isclose(sm_no_intercept.result['dev_ratio'][0], 0.0), \
    "no-intercept null fit must have zero deviance ratio"

X_shifted = np.column_stack((
    np.linspace(5.0, 34.0, Y_no_intercept.size),
    np.full(Y_no_intercept.size, 7.0)))
sm_no_intercept_std = pycasso.Solver(
    X_shifted, Y_no_intercept, lambdas=(3, 0.2),
    family="multinomial", useintercept=False, standardize=True,
    prec=1e-6, max_ite=500)
assert np.array_equal(sm_no_intercept_std._xm,
                      np.zeros(X_shifted.shape[1])), \
    "no-intercept standardization centered the design"
assert np.all(sm_no_intercept_std._xinvc > 0) and \
    np.all(sm_no_intercept_std.x[:, 1] != 0), \
    "no-intercept scaling discarded a nonzero constant feature"
sm_no_intercept_std.train()
assert np.array_equal(
    sm_no_intercept_std.result['intercept'],
    np.zeros_like(sm_no_intercept_std.result['intercept'])), \
    "standardized no-intercept fit returned nonzero intercepts"

# Standardization must retain its previous numerical contract while using the
# output matrix itself as workspace instead of materializing several n-by-d
# temporaries.
def _legacy_standardize_reference(values):
    nrows = values.shape[0]
    maximum = np.max(np.abs(values), axis=0)
    means = np.zeros(values.shape[1], dtype='double')
    scaled_values = np.zeros_like(values, dtype='double')
    inverse = np.zeros(values.shape[1], dtype='double')
    nonzero = maximum > 0
    scaled = values[:, nonzero] / maximum[nonzero]
    scaled_mean = np.mean(scaled, axis=0)
    means[nonzero] = maximum[nonzero] * scaled_mean
    centered = scaled - scaled_mean
    if nrows > 1:
        norm = np.sqrt(np.sum(centered ** 2, axis=0) / (nrows - 1))
        nonconstant = norm > 0
        columns = np.flatnonzero(nonzero)[nonconstant]
        scaled_values[:, columns] = (
            centered[:, nonconstant] / norm[nonconstant])
        inverse[columns] = (
            (1.0 / maximum[columns]) / norm[nonconstant])
    return scaled_values, means, inverse


def _legacy_scale_no_center_reference(values):
    nrows = values.shape[0]
    maximum = np.max(np.abs(values), axis=0)
    scaled_values = np.zeros_like(values, dtype='double')
    inverse = np.zeros(values.shape[1], dtype='double')
    nonzero = maximum > 0
    scaled = values[:, nonzero] / maximum[nonzero]
    norm = np.sqrt(
        np.sum(scaled ** 2, axis=0) / max(nrows - 1, 1))
    columns = np.flatnonzero(nonzero)
    scaled_values[:, columns] = scaled / norm
    inverse[columns] = (1.0 / maximum[columns]) / norm
    return scaled_values, np.zeros(values.shape[1]), inverse


standardization_rng = np.random.default_rng(20260720)
standardization_input = standardization_rng.normal(size=(137, 43))
standardization_input[:, 0] = 0.0
standardization_input[:, 1] = 7.0
standardization_input[:, 2] *= 1e200
standardization_snapshot = standardization_input.copy()
for new_helper, legacy_helper, helper_name in (
        (pycasso_core._standardize, _legacy_standardize_reference,
         'centered'),
        (pycasso_core._scale_without_centering,
         _legacy_scale_no_center_reference, 'no-center')):
    actual = new_helper(standardization_input)
    expected = legacy_helper(standardization_input)
    assert actual[0].flags.c_contiguous and not np.shares_memory(
        actual[0], standardization_input), \
        f"{helper_name} standardization lost C ownership"
    assert np.allclose(actual[0], expected[0], rtol=5e-14, atol=5e-15), \
        f"{helper_name} standardized design drifted from the legacy result"
    assert np.allclose(actual[1], expected[1], rtol=5e-14, atol=5e-15), \
        f"{helper_name} column means drifted from the legacy result"
    assert np.allclose(actual[2], expected[2], rtol=5e-14, atol=0.0), \
        f"{helper_name} inverse scales drifted from the legacy result"
assert np.array_equal(standardization_input, standardization_snapshot), \
    "standardization modified its owned raw-design input"

single_row = np.array([[0.0, 5.0, -3.0]])
single_centered = pycasso_core._standardize(single_row)
single_origin = pycasso_core._scale_without_centering(single_row)
assert np.array_equal(single_centered[0], np.zeros_like(single_row)) and \
    np.array_equal(single_centered[1], single_row[0]) and \
    np.array_equal(single_centered[2], np.zeros(single_row.shape[1])), \
    "single-row centered standardization changed its constant-column contract"
assert np.array_equal(single_origin[0], np.array([[0.0, 1.0, -1.0]])) and \
    np.array_equal(single_origin[1], np.zeros(single_row.shape[1])), \
    "single-row no-center scaling changed the design origin"

extreme_n = 18
extreme_fraction = np.linspace(0.55, 0.95, extreme_n)
extreme_design = np.column_stack((
    1e308 * extreme_fraction,
    1e308 * np.r_[-1.0, np.ones(extreme_n - 1)],
    np.linspace(-2.0, 2.0, extreme_n),
    np.full(extreme_n, 1e308),
    np.zeros(extreme_n),
))
extreme_snapshot = extreme_design.copy()
extreme_centered, extreme_mean, extreme_inverse = \
    pycasso_core._standardize(extreme_design)
extreme_origin, extreme_zero_mean, extreme_origin_inverse = \
    pycasso_core._scale_without_centering(extreme_design)
assert np.array_equal(extreme_design, extreme_snapshot), \
    "extreme standardization modified its input"
assert np.all(np.isfinite(extreme_centered)) and \
    np.all(np.isfinite(extreme_mean)) and \
    np.all(np.isfinite(extreme_inverse)) and \
    np.all(np.isfinite(extreme_origin)) and \
    np.all(np.isfinite(extreme_origin_inverse)), \
    "extreme finite standardization produced non-finite output"
assert np.allclose(
    np.mean(extreme_centered[:, :3], axis=0), 0.0, atol=2e-15) and \
    np.allclose(
        np.sum(extreme_centered[:, :3] ** 2, axis=0),
        extreme_n - 1, rtol=2e-13, atol=0.0), \
    "extreme centered columns lost their normalization"
assert np.allclose(
    extreme_mean[0], 1e308 * np.mean(extreme_fraction), rtol=2e-15) and \
    np.array_equal(extreme_centered[:, 3:],
                   np.zeros((extreme_n, 2))) and \
    np.array_equal(extreme_inverse[3:], np.zeros(2)), \
    "extreme constant or zero columns changed centered semantics"
assert np.array_equal(extreme_zero_mean, np.zeros(extreme_design.shape[1])) and \
    np.all(extreme_origin[:, 3] != 0.0), \
    "extreme no-center scaling discarded a nonzero constant column"

# Finite large columns must not overflow their squared norms and disappear.
X_large = np.array([[1e200, 1e200], [2e200, -1e200], [3e200, 2e200]])
large_centered, large_mean, large_inverse = pycasso_core._standardize(X_large)
large_origin, zero_mean, large_origin_inverse = \
    pycasso_core._scale_without_centering(X_large)
assert np.all(np.isfinite(large_centered)) and \
       np.all(np.isfinite(large_inverse)) and \
       np.all(large_inverse > 0), \
    "centered scaling overflowed a finite large column"
assert np.all(np.isfinite(large_origin)) and \
       np.all(np.isfinite(large_origin_inverse)) and \
       np.all(large_origin_inverse > 0), \
    "origin-preserving scaling overflowed a finite large column"
assert not np.all(large_centered == 0) and not np.all(large_origin == 0), \
    "finite large columns were silently discarded"
assert np.allclose(large_centered.mean(axis=0), 0.0, atol=1e-15) and \
       np.array_equal(zero_mean, np.zeros(X_large.shape[1])), \
    "large-column scaling changed its centering contract"

# dfmax=0 must retain the first nonzero crossing, then truncate the path.
dfmax_path = lambda_max_mn * np.array([1.5, 1.1, 0.75, 0.4, 0.15])
sm_cut = pycasso.Solver(X_mn, Y_mn3, lambdas=dfmax_path,
                        family="multinomial", dfmax=0,
                        prec=1e-6, max_ite=200)
sm_cut.train()
cut_nlambda = sm_cut.nlambda
assert_multinomial_diagnostics(sm_cut, 1, "dfmax multinomial")
assert 1 < cut_nlambda < len(dfmax_path), \
    f"dfmax path was not truncated at a crossing: {cut_nlambda}"
nonzero_fits = np.flatnonzero(sm_cut.result['df'] > 0)
assert nonzero_fits.size == 1 and nonzero_fits[0] == cut_nlambda - 1, \
    f"dfmax path did not retain exactly the first crossing: {sm_cut.result['df']}"
assert np.array_equal(sm_cut.lambdas, dfmax_path[:cut_nlambda]), \
    "dfmax lambda path was not truncated to the fitted prefix"
assert_multinomial_fit(sm_cut, cut_nlambda, "dfmax multinomial")
first_cut = {
    key: sm_cut.result[key].copy()
    for key in ('beta', 'intercept', 'ite_lamb', 'size_act', 'df', 'dev_ratio')
}
sm_cut.train()
assert_multinomial_fit(sm_cut, cut_nlambda, "repeated dfmax multinomial")
assert_multinomial_diagnostics(
    sm_cut, 1, "repeated dfmax multinomial")
for key in ('beta', 'intercept', 'dev_ratio'):
    assert np.allclose(sm_cut.result[key], first_cut[key],
                       rtol=1e-9, atol=1e-11), \
        f"repeated dfmax multinomial {key} changed"
for key in ('ite_lamb', 'size_act', 'df'):
    assert np.array_equal(sm_cut.result[key], first_cut[key]), \
        f"repeated dfmax multinomial {key} changed"

# Fresh and already-trained CV must use the same full-data path and report the
# same model sizes, including when dfmax truncates that path.
sm_cv_cut = pycasso.Solver(
    X_mn, Y_mn3, lambdas=dfmax_path, family="multinomial", dfmax=0,
    prec=1e-6, max_ite=200)
cv_cut_fresh = sm_cv_cut.cross_validate(
    foldid=cv_class['foldid'], type_measure="class")
assert np.array_equal(cv_cut_fresh['nzero'], sm_cv_cut.result['df']), \
    "fresh dfmax CV reported incorrect nzero"
cv_cut_repeat = sm_cv_cut.cross_validate(
    foldid=cv_class['foldid'], type_measure="class")
assert np.array_equal(cv_cut_repeat['lambda'], cv_cut_fresh['lambda']) and \
    np.array_equal(cv_cut_repeat['nzero'], cv_cut_fresh['nzero']) and \
    np.allclose(cv_cut_repeat['cvm'], cv_cut_fresh['cvm']), \
    "multinomial CV depends on whether train() was called first"

# Old shared libraries remain loadable, but a truncated V1 result must be
# explicitly marked unknown and warned about rather than treated as success.
class LegacyMultinomialLibrary:
    def __init__(self, library):
        self.SolveMultinomialRegression = \
            library.SolveMultinomialRegression


native_library = pycasso_core._PICASSO_LIB


class FakeMultinomialBufferFunction:
    """Versioned multinomial fixture that records native output objects."""
    def __init__(self, version, num_fit, status=0):
        self.version = version
        self.num_fit = num_fit
        self.status = status
        self.calls = []
        self.argtypes = None
        self.restype = 'unset'

    def __call__(self, *args):
        names = ('beta', 'intercept', 'ite_lamb', 'size_act',
                 'train_time', 'num_fit')
        received = dict(zip(names, args[13:19]))
        self.calls.append(received)
        received['beta'].fill(0.0)
        received['intercept'].fill(0.0)
        received['ite_lamb'].fill(2)
        received['size_act'].fill(0)
        received['train_time'].fill(0.01)
        received['num_fit'][0] = self.num_fit

        if self.version >= 2:
            diagnostic_start = 20
            if self.version >= 3:
                diagnostic_start += 1
            if self.version >= 4:
                diagnostic_start += 1
            diagnostic_names = (
                'failed_lambda_buffer', 'failed_stage_buffer', 'outer_ite',
                'inner_sweeps', 'coordinate_updates', 'objective', 'kkt',
                'stationarity')
            diagnostics = dict(zip(
                diagnostic_names,
                args[diagnostic_start:diagnostic_start + 8]))
            received.update(diagnostics)
            diagnostics['failed_lambda_buffer'][0] = -1
            diagnostics['failed_stage_buffer'][0] = -1
            diagnostics['outer_ite'].fill(1)
            diagnostics['inner_sweeps'].fill(2)
            diagnostics['coordinate_updates'].fill(3)
            diagnostics['objective'].fill(1.0)
            diagnostics['kkt'].fill(1e-8)
            diagnostics['stationarity'].fill(1e-8)
            if self.version >= 5:
                received['smooth_nll'] = args[diagnostic_start + 8]
                received['smooth_nll'].fill(np.log(k_mn))
            return self.status

        return None


class FakeMultinomialBufferLibrary:
    def __init__(self, function):
        suffix = '' if function.version == 1 else f'V{function.version}'
        setattr(self, 'SolveMultinomialRegression' + suffix, function)


# Every supported ABI must receive the fresh result buffers directly. A full
# path retains those objects; retraining replaces them rather than mutating a
# caller's references to the previous fit.
buffer_path = np.array([0.3, 0.2, 0.1])
buffer_solvers = {}
buffer_functions = {}
for buffer_version in range(1, 6):
    buffer_function = FakeMultinomialBufferFunction(
        buffer_version, len(buffer_path))
    try:
        pycasso_core._PICASSO_LIB = FakeMultinomialBufferLibrary(
            buffer_function)
        buffer_solver = pycasso.Solver(
            X_mn, Y_mn3, lambdas=buffer_path,
            family='multinomial')
    finally:
        pycasso_core._PICASSO_LIB = native_library
    buffer_solver.train()
    received = buffer_function.calls[-1]
    direct_fields = [
        'beta', 'intercept', 'ite_lamb', 'size_act', 'train_time',
        'num_fit',
    ]
    if buffer_version >= 2:
        direct_fields.extend([
            'outer_ite', 'inner_sweeps', 'coordinate_updates',
            'objective', 'kkt', 'stationarity',
        ])
    if buffer_version >= 5:
        direct_fields.append('smooth_nll')
    for field in direct_fields:
        assert buffer_solver.result[field] is received[field], \
            f"multinomial V{buffer_version} copied full-path {field}"
        assert buffer_solver.result[field].flags.c_contiguous, \
            f"multinomial V{buffer_version} {field} is not C-contiguous"
    expected_status = 0 if buffer_version >= 2 else None
    assert buffer_solver.result['status_code'] == expected_status and \
        buffer_solver.result['state'] == 'trained', \
        f"multinomial V{buffer_version} full-path status changed"
    if buffer_version == 1:
        assert np.array_equal(
            buffer_solver.result['inner_sweeps'],
            buffer_solver.result['ite_lamb'].astype('int64')), \
            "legacy multinomial iteration diagnostics changed"
    buffer_solvers[buffer_version] = buffer_solver
    buffer_functions[buffer_version] = buffer_function

v5_buffer_solver = buffer_solvers[5]
v5_buffer_function = buffer_functions[5]
old_result_dictionary = v5_buffer_solver.result
old_buffers = {
    field: v5_buffer_solver.result[field]
    for field in ('beta', 'intercept', 'ite_lamb', 'size_act',
                  'train_time', 'outer_ite', 'inner_sweeps',
                  'coordinate_updates', 'objective', 'kkt',
                  'stationarity', 'smooth_nll')
}
old_values = {field: values.copy() for field, values in old_buffers.items()}
v5_buffer_solver.train()
assert v5_buffer_solver.result is old_result_dictionary, \
    "multinomial retraining replaced the live result dictionary"
for field, old_buffer in old_buffers.items():
    assert v5_buffer_solver.result[field] is v5_buffer_function.calls[-1][field] \
        and v5_buffer_solver.result[field] is not old_buffer, \
        f"multinomial retraining reused the old {field} buffer"
    assert np.allclose(
        old_buffer, old_values[field], rtol=0.0, atol=0.0,
        equal_nan=True), \
        f"multinomial retraining mutated the previous {field} result"

# A fitted prefix must own compact arrays instead of retaining the requested
# full-path buffers through NumPy views.
compact_function = FakeMultinomialBufferFunction(5, 2, status=1)
try:
    pycasso_core._PICASSO_LIB = FakeMultinomialBufferLibrary(
        compact_function)
    compact_solver = pycasso.Solver(
        X_mn, Y_mn3, lambdas=np.array([0.4, 0.3, 0.2, 0.1]),
        family='multinomial')
finally:
    pycasso_core._PICASSO_LIB = native_library
compact_solver.train()
compact_received = compact_function.calls[-1]
compact_native_fields = (
    'beta', 'intercept', 'ite_lamb', 'size_act', 'train_time',
    'outer_ite', 'inner_sweeps', 'coordinate_updates', 'objective', 'kkt',
    'stationarity', 'smooth_nll')
for field in compact_native_fields:
    compact = compact_solver.result[field]
    assert compact.shape[0] == 2 and compact.flags.owndata and \
        compact.base is None and \
        not np.shares_memory(compact, compact_received[field]), \
        f"truncated multinomial {field} retained its full-path owner"
for field in ('df', 'runtime'):
    compact = compact_solver.result[field]
    assert compact.shape == (2,) and compact.flags.owndata and \
        compact.base is None, \
        f"truncated multinomial {field} is not a compact owner"
assert compact_solver.result['num_fit'] is compact_received['num_fit'] and \
    compact_solver.result['status_code'] == 1 and \
    compact_solver.result['state'] == 'trained' and \
    compact_solver.lambdas.flags.owndata, \
    "truncated multinomial metadata or lambda ownership changed"


class FakeGaussianFunction:
    """Minimal Gaussian V2 fixture for native fit-count validation."""
    def __init__(self, num_fit):
        self.num_fit = num_fit
        self.argtypes = None
        self.restype = 'unset'

    def __call__(self, *args):
        beta, intercept, ite_lamb, size_act, runtime = args[12:17]
        num_fit = args[17]
        smooth_objective = args[19]
        beta[:] = 0.0
        intercept[:] = 0.0
        ite_lamb[:] = 1
        size_act[:] = 0
        runtime[:] = 0.0
        num_fit[0] = self.num_fit
        smooth_objective[:] = 1.0


class FakeGaussianLibrary:
    def __init__(self, function):
        self.SolveLinearRegressionNaiveUpdateV2 = function


invalid_gaussian_fits = []
try:
    for invalid_fit_count in (-1, 0, len(scalar_path) + 1):
        fake_gaussian_function = FakeGaussianFunction(invalid_fit_count)
        pycasso_core._PICASSO_LIB = FakeGaussianLibrary(
            fake_gaussian_function)
        invalid_gaussian_fits.append((
            pycasso.Solver(
                X_scalar, Y_scalar_sqrt, lambdas=scalar_path,
                family='gaussian', type_gaussian='naive'),
            invalid_fit_count))
        assert fake_gaussian_function.restype is None, \
            "Gaussian void C API did not set ctypes restype=None"
finally:
    pycasso_core._PICASSO_LIB = native_library

for invalid_gaussian_solver, invalid_fit_count in invalid_gaussian_fits:
    try:
        invalid_gaussian_solver.train()
        raise AssertionError(
            f"Gaussian native fit count {invalid_fit_count} was accepted")
    except PycassoError as exc:
        assert "invalid fit count" in str(exc) and \
            f"num_fit={invalid_fit_count}" in str(exc), \
            "Gaussian invalid fit-count error omitted native diagnostics"


class FakeScalarV2Function:
    """Minimal ctypes-like scalar V2 fixture with configurable termination."""
    def __init__(self, status, num_fit, failed_lambda=-1, failed_stage=-1):
        self.status = status
        self.num_fit = num_fit
        self.failed_lambda = failed_lambda
        self.failed_stage = failed_stage
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        has_offset = len(args) == 27
        output_start = 13 if has_offset else 12
        beta, intercept, ite_lamb, size_act, runtime, num_fit = \
            args[output_start:output_start + 6]
        diagnostic_start = output_start + 8
        failed_lambda, failed_stage, stages, objective, kkt, stationarity = \
            args[diagnostic_start:diagnostic_start + 6]
        beta[:] = 0.0
        intercept[:] = 0.0
        ite_lamb[:] = 2
        size_act[:] = 0
        runtime[:] = 0.01
        num_fit[0] = self.num_fit
        failed_lambda[0] = self.failed_lambda
        failed_stage[0] = self.failed_stage
        stages[:] = 3
        objective[:] = np.arange(objective.size, dtype=float) + 1.0
        kkt[:] = 1e-8
        stationarity[:] = 1e-4
        return self.status


class FakeScalarV2Library:
    def __init__(self, symbol, function):
        setattr(self, symbol, function)


# A usable status 10 may accompany a normally shortened scalar path. It must
# not warn or be reclassified as a hard failure.
try:
    pycasso_core._PICASSO_LIB = FakeScalarV2Library(
        'SolveLogisticRegressionV2', FakeScalarV2Function(10, 2))
    scalar_capped_prefix = pycasso.Solver(
        X_scalar, Y_scalar_binomial, lambdas=scalar_path,
        family='binomial', penalty='mcp')
finally:
    pycasso_core._PICASSO_LIB = native_library
with warnings.catch_warnings(record=True) as capped_prefix_warnings:
    warnings.simplefilter("always")
    scalar_capped_prefix.train()
assert not capped_prefix_warnings and \
    scalar_capped_prefix.result['status_code'] == 10 and \
    scalar_capped_prefix.result['state'] == 'trained' and \
    scalar_capped_prefix.nlambda == 2, \
    "usable truncated scalar status 10 was treated as a hard failure"
assert scalar_capped_prefix.result['lla_stages'].shape == (2,), \
    "usable scalar prefix diagnostics were not truncated consistently"

# A hard V2 status keeps only committed models and preserves failed-point data.
try:
    pycasso_core._PICASSO_LIB = FakeScalarV2Library(
        'SolveSqrtLinearRegressionV2',
        FakeScalarV2Function(3, 2, failed_lambda=2, failed_stage=1))
    scalar_hard_prefix = pycasso.Solver(
        X_scalar, Y_scalar_sqrt, lambdas=scalar_path,
        family='sqrtlasso', penalty='scad')
finally:
    pycasso_core._PICASSO_LIB = native_library
with warnings.catch_warnings(record=True) as scalar_hard_warnings:
    warnings.simplefilter("always")
    scalar_hard_prefix.train()
assert len(scalar_hard_warnings) == 1 and \
    issubclass(scalar_hard_warnings[0].category, RuntimeWarning), \
    "hard scalar V2 prefix did not emit exactly one RuntimeWarning"
assert scalar_hard_prefix.result['status'] == 'subproblem_failed' and \
    scalar_hard_prefix.result['state'] == 'partially trained' and \
    scalar_hard_prefix.nlambda == 2 and \
    scalar_hard_prefix.result['failed_lambda'] == 2 and \
    scalar_hard_prefix.result['failed_stage'] == 1, \
    "hard scalar V2 status or committed prefix was lost"
assert scalar_hard_prefix.result['failure_diagnostics'] is not None and \
    scalar_hard_prefix.result['failure_diagnostics']['objective'] == 3.0, \
    "hard scalar V2 failed-point diagnostics were not retained"

# A hard failure before the first commit is an error, but its diagnostics stay
# available for debugging after train() raises.
try:
    pycasso_core._PICASSO_LIB = FakeScalarV2Library(
        'SolvePoissonRegressionV2',
        FakeScalarV2Function(7, 0, failed_lambda=0, failed_stage=0))
    scalar_no_prefix = pycasso.Solver(
        X_scalar, Y_scalar_poisson, lambdas=scalar_path,
        family='poisson', penalty='mcp')
finally:
    pycasso_core._PICASSO_LIB = native_library
try:
    scalar_no_prefix.train()
    raise AssertionError("zero-prefix scalar hard failure did not raise")
except PycassoError as exc:
    assert 'numerical_failure' in str(exc) and 'failed_lambda=0' in str(exc), \
        f"zero-prefix scalar error omitted native diagnostics: {exc}"
assert scalar_no_prefix.result['state'] == 'not trained' and \
    scalar_no_prefix.result['status_code'] == 7 and \
    scalar_no_prefix.result['failure_diagnostics'] is not None, \
    "zero-prefix scalar diagnostics were lost after the error"


class LegacyScalarLibrary:
    def __init__(self, library):
        self.SolveLogisticRegression = library.SolveLogisticRegression
        self.SolvePoissonRegression = library.SolvePoissonRegression
        self.SolveSqrtLinearRegression = library.SolveSqrtLinearRegression


# Legacy scalar ABIs may honor the default cap. Nondefault MCP/SCAD must fail
# explicitly, while a nondefault cap remains harmless for L1.
try:
    pycasso_core._PICASSO_LIB = LegacyScalarLibrary(native_library)
    for legacy_family, legacy_y in scalar_responses.items():
        try:
            pycasso.Solver(
                X_scalar, legacy_y, lambdas=scalar_path,
                family=legacy_family, penalty='mcp', lla_max_stages=4)
            raise AssertionError(
                f"{legacy_family} legacy backend ignored nondefault LLA cap")
        except PycassoError as exc:
            assert 'lla_max_stages' in str(exc) and 'V2' in str(exc), \
                f"{legacy_family} legacy compatibility error is unclear: {exc}"
    legacy_l1 = pycasso.Solver(
        X_scalar, Y_scalar_binomial, lambdas=scalar_path,
        family='binomial', penalty='l1', lla_max_stages=9)
finally:
    pycasso_core._PICASSO_LIB = native_library
legacy_l1.train()
assert legacy_l1.result['status_code'] is None and \
    legacy_l1.result['status'] == 'legacy_unknown' and \
    legacy_l1.result['state'] == 'trained', \
    "L1 legacy fallback was affected by lla_max_stages"


class RecordingScalarBufferFunction:
    """Configurable scalar V1/V2/V3 fixture that retains received buffers."""
    def __init__(self, version, num_fits, statuses=None, has_offset=False):
        self.version = version
        self.num_fits = list(num_fits)
        self.statuses = list(statuses or [0] * len(self.num_fits))
        self.has_offset = has_offset
        self.calls = []
        self.argtypes = None
        self.restype = None

    @staticmethod
    def _fill_coefficients(beta, intercept, call_number):
        beta.fill(0.0)
        path_index = np.arange(1, beta.shape[0] + 1, dtype=float)
        beta[:, 0] = 0.01 * (call_number + 1) * path_index
        if beta.shape[1] > 1:
            beta[:, 1] = 1e-300
        if beta.shape[1] > 3:
            beta[1::2, 3] = -0.02 * path_index[1::2]
        intercept[:] = 0.025 * path_index

    def __call__(self, *args):
        call_number = len(self.calls)
        output_start = 13 if self.has_offset else 12
        beta, intercept, ite_lamb, size_act, train_time, num_fit = \
            args[output_start:output_start + 6]
        self._fill_coefficients(beta, intercept, call_number)
        ite_lamb.fill(2)
        size_act.fill(3)
        train_time.fill(0.01)
        sequence_index = min(call_number, len(self.num_fits) - 1)
        num_fit[0] = self.num_fits[sequence_index]
        status = self.statuses[min(call_number, len(self.statuses) - 1)]
        call = {
            'beta': beta,
            'intercept': intercept,
            'ite_lamb': ite_lamb,
            'size_act': size_act,
            'train_time': train_time,
            'num_fit': num_fit,
            'raw_beta': beta.copy(),
            'raw_intercept': intercept.copy(),
        }

        if self.version >= 2:
            diagnostic_start = output_start + 8
            failed_lambda, failed_stage, lla_stages, objective, kkt, \
                stationarity = args[diagnostic_start:diagnostic_start + 6]
            hard_failure = status not in (0, 1, 10)
            failed_lambda[0] = num_fit[0] if hard_failure else -1
            failed_stage[0] = 1 if hard_failure else -1
            lla_stages.fill(3)
            objective[:] = np.arange(objective.size, dtype=float) + 1.0
            kkt.fill(1e-8)
            stationarity.fill(1e-7)
            call.update({
                'lla_stages': lla_stages,
                'objective': objective,
                'kkt': kkt,
                'stationarity': stationarity,
            })
            if self.version >= 3:
                smooth_objective = args[diagnostic_start + 6]
                smooth_objective.fill(0.75)
                call['smooth_objective'] = smooth_objective

        self.calls.append(call)
        return status if self.version >= 2 else None


class RecordingScalarBufferLibrary:
    """Expose exactly one scalar ABI generation for one family."""
    def __init__(self, family, function):
        stem = {
            'binomial': 'SolveLogisticRegression',
            'poisson': 'SolvePoissonRegression',
            'sqrtlasso': 'SolveSqrtLinearRegression',
        }[family]
        suffix = '' if function.version == 1 else f'V{function.version}'
        setattr(self, stem + suffix, function)


class RecordingGaussianBufferFunction:
    """Gaussian V2 fixture with configurable full or truncated paths."""
    def __init__(self, num_fits, version=2):
        self.num_fits = list(num_fits)
        self.version = version
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        call_number = len(self.calls)
        beta, intercept, ite_lamb, size_act, train_time, num_fit = \
            args[12:18]
        RecordingScalarBufferFunction._fill_coefficients(
            beta, intercept, call_number)
        ite_lamb.fill(2)
        size_act.fill(3)
        train_time.fill(0.01)
        sequence_index = min(call_number, len(self.num_fits) - 1)
        num_fit[0] = self.num_fits[sequence_index]
        call = {
            'beta': beta,
            'intercept': intercept,
            'ite_lamb': ite_lamb,
            'size_act': size_act,
            'train_time': train_time,
            'num_fit': num_fit,
            'raw_beta': beta.copy(),
            'raw_intercept': intercept.copy(),
        }
        if self.version >= 2:
            smooth_objective = args[19]
            smooth_objective.fill(0.75)
            call['smooth_objective'] = smooth_objective
        self.calls.append(call)


class RecordingGaussianBufferLibrary:
    def __init__(self, function):
        suffix = '' if function.version == 1 else 'V2'
        setattr(self, 'SolveLinearRegressionNaiveUpdate' + suffix, function)
        setattr(self, 'SolveLinearRegressionCovUpdate' + suffix, function)


scalar_finalize_x = np.array([
    [1.0, -0.2, 0.7, 2.0], [1.2, 0.1, -0.4, 1.7],
    [1.5, 0.4, 0.2, 2.2], [1.7, -0.5, 0.5, 1.4],
    [2.0, 0.7, -0.1, 2.5], [2.2, -0.8, 0.8, 1.1],
    [2.5, 0.9, -0.6, 2.8], [2.7, -1.0, 0.9, 0.8],
    [3.0, 1.1, -0.9, 3.0], [3.2, -1.2, 1.1, 0.5],
    [3.5, 1.3, -1.2, 3.2], [3.7, -1.4, 1.4, 0.2],
])
scalar_finalize_responses = {
    'gaussian': 1.4 + 0.3 * scalar_finalize_x[:, 0],
    'binomial': np.arange(scalar_finalize_x.shape[0]) % 2,
    'poisson': np.arange(scalar_finalize_x.shape[0]) % 4,
    'sqrtlasso': -0.6 + 0.2 * scalar_finalize_x[:, 3],
}
scalar_finalize_path = np.array([0.4, 0.3, 0.2, 0.1])


def make_recording_scalar_solver(family, standardize, use_intercept,
                                 num_fits, version=3, statuses=None,
                                 gaussian_mode='naive'):
    """Construct one solver while its wrapper binds a recording fake ABI."""
    if family == 'gaussian':
        function = RecordingGaussianBufferFunction(num_fits)
        library = RecordingGaussianBufferLibrary(function)
    else:
        function = RecordingScalarBufferFunction(
            version, num_fits, statuses=statuses,
            has_offset=family in ('binomial', 'poisson'))
        library = RecordingScalarBufferLibrary(family, function)
    constructor_args = {}
    if family in ('binomial', 'poisson'):
        constructor_args['offset'] = np.linspace(
            -0.15, 0.15, scalar_finalize_x.shape[0])
    if family == 'gaussian':
        constructor_args['type_gaussian'] = gaussian_mode
    try:
        pycasso_core._PICASSO_LIB = library
        solver = pycasso.Solver(
            scalar_finalize_x, scalar_finalize_responses[family],
            lambdas=scalar_finalize_path, family=family,
            standardize=standardize, useintercept=use_intercept,
            **constructor_args)
    finally:
        pycasso_core._PICASSO_LIB = native_library
    return solver, function


def expected_recording_solution(solver, call, nfit):
    beta = call['raw_beta'][:nfit]
    intercept = call['raw_intercept'][:nfit]
    if solver.standardize:
        beta = beta * solver._xinvc
        if solver.use_intercept:
            intercept = intercept - beta @ solver._xm
    if not solver.use_intercept:
        intercept = np.zeros_like(intercept)
    if solver.family == 'gaussian' and solver._ym != 0.0:
        intercept = intercept + solver._ym
    return beta, intercept


def assert_compact_owner(values, expected_length, label):
    assert values.shape[0] == expected_length and values.flags.owndata and \
        values.base is None and values.flags.c_contiguous, \
        f"{label} is not an owning contiguous fitted prefix"


# Full scalar paths must keep every native output identity, including after
# in-place standardization. Partial paths must instead own compact prefixes.
for finalize_family in ('gaussian', 'binomial', 'poisson', 'sqrtlasso'):
    for finalize_standardize in (False, True):
        for finalize_intercept in (False, True):
            full_solver, full_function = make_recording_scalar_solver(
                finalize_family, finalize_standardize, finalize_intercept,
                [len(scalar_finalize_path)])
            full_solver.train()
            full_call = full_function.calls[-1]
            expected_beta, expected_intercept = expected_recording_solution(
                full_solver, full_call, len(scalar_finalize_path))
            for field in ('beta', 'intercept', 'ite_lamb', 'size_act',
                          'train_time'):
                assert full_solver.result[field] is full_call[field], \
                    f"{finalize_family} full {field} lost native identity"
            assert np.array_equal(full_solver.result['beta'], expected_beta) and \
                np.array_equal(full_solver.result['intercept'],
                               expected_intercept), \
                f"{finalize_family} full scalar rescaling changed values"
            expected_df = np.count_nonzero(expected_beta, axis=1).astype('int32')
            assert np.array_equal(full_solver.result['df'], expected_df), \
                f"{finalize_family} full scalar df changed"
            assert np.all(full_solver.result['df'] >= 2), \
                f"{finalize_family} tiny nonzero coefficient was dropped"
            if finalize_family == 'gaussian':
                assert full_solver.result['smooth_objective'] is \
                    full_call['smooth_objective'], \
                    "Gaussian full smooth objective was copied"
            else:
                assert full_solver.result['runtime'] is \
                    full_solver.result['train_time'] and \
                    full_solver.result['stages'] is \
                    full_solver.result['lla_stages'], \
                    f"{finalize_family} full result aliases changed"
                for field in ('lla_stages', 'objective', 'kkt',
                              'stationarity', 'smooth_objective'):
                    assert full_solver.result[field] is full_call[field], \
                        f"{finalize_family} full {field} lost native identity"

            partial_solver, partial_function = make_recording_scalar_solver(
                finalize_family, finalize_standardize, finalize_intercept,
                [2], statuses=[1])
            partial_solver.train()
            partial_call = partial_function.calls[-1]
            expected_beta, expected_intercept = expected_recording_solution(
                partial_solver, partial_call, 2)
            assert np.array_equal(partial_solver.result['beta'],
                                  expected_beta) and \
                np.array_equal(partial_solver.result['intercept'],
                               expected_intercept), \
                f"{finalize_family} partial scalar rescaling changed values"
            compact_fields = [
                'beta', 'intercept', 'ite_lamb', 'size_act', 'train_time',
                'df', 'smooth_objective', 'dev_ratio',
            ]
            if finalize_family != 'gaussian':
                compact_fields.extend([
                    'lla_stages', 'objective', 'kkt', 'stationarity'])
            for field in compact_fields:
                assert_compact_owner(
                    partial_solver.result[field], 2,
                    f"{finalize_family} partial {field}")
            for field in ('beta', 'intercept', 'ite_lamb', 'size_act',
                          'train_time', 'smooth_objective'):
                assert not np.shares_memory(
                    partial_solver.result[field], partial_call[field]), \
                    f"{finalize_family} partial {field} retained full owner"
            assert partial_solver.lambdas.flags.owndata and \
                partial_solver.lambdas.base is None and \
                partial_solver.lambdas.shape == (2,), \
                f"{finalize_family} lambdas retained requested-path owner"
            if finalize_family != 'gaussian':
                assert partial_solver.result['runtime'] is \
                    partial_solver.result['train_time'] and \
                    partial_solver.result['stages'] is \
                    partial_solver.result['lla_stages'], \
                    f"{finalize_family} partial aliases changed"


# Every scalar LLA ABI generation must preserve full-path identity. V1 partial,
# usable status 10, and hard V3 partial cover each termination branch.
for abi_family in ('binomial', 'poisson', 'sqrtlasso'):
    for abi_version in (1, 2, 3):
        abi_solver, abi_function = make_recording_scalar_solver(
            abi_family, True, True, [len(scalar_finalize_path)],
            version=abi_version, statuses=[0])
        abi_solver.train()
        assert abi_solver.result['beta'] is abi_function.calls[-1]['beta'], \
            f"{abi_family} V{abi_version} full beta identity changed"
        assert abi_solver.result['status_code'] == (
            None if abi_version == 1 else 0), \
            f"{abi_family} V{abi_version} status changed"
        assert ('smooth_objective' in abi_solver.result) == (abi_version == 3), \
            f"{abi_family} V{abi_version} smooth-objective routing changed"

legacy_prefix, legacy_prefix_function = make_recording_scalar_solver(
    'binomial', False, True, [2], version=1)
with warnings.catch_warnings(record=True) as legacy_prefix_warnings:
    warnings.simplefilter('always')
    legacy_prefix.train()
assert len(legacy_prefix_warnings) == 1 and \
    legacy_prefix.result['state'] == 'partially trained', \
    "legacy scalar prefix status changed"
for field in ('beta', 'intercept', 'ite_lamb', 'size_act', 'train_time',
              'df', 'lla_stages', 'objective', 'kkt', 'stationarity'):
    assert_compact_owner(legacy_prefix.result[field], 2,
                         f"legacy partial {field}")

status_ten_prefix, _ = make_recording_scalar_solver(
    'poisson', True, True, [2], version=2, statuses=[10])
with warnings.catch_warnings(record=True) as status_ten_warnings:
    warnings.simplefilter('always')
    status_ten_prefix.train()
assert not status_ten_warnings and \
    status_ten_prefix.result['state'] == 'trained', \
    "usable scalar status 10 changed"

hard_v3_prefix, hard_v3_function = make_recording_scalar_solver(
    'sqrtlasso', True, True, [2], version=3, statuses=[3])
with warnings.catch_warnings(record=True) as hard_v3_warnings:
    warnings.simplefilter('always')
    hard_v3_prefix.train()
assert len(hard_v3_warnings) == 1 and \
    hard_v3_prefix.result['state'] == 'partially trained' and \
    hard_v3_prefix.result['failure_diagnostics']['objective'] == 3.0, \
    "hard scalar V3 partial diagnostics changed"
for field in ('beta', 'intercept', 'train_time', 'lla_stages', 'objective',
              'kkt', 'stationarity', 'smooth_objective'):
    assert_compact_owner(hard_v3_prefix.result[field], 2,
                         f"hard V3 partial {field}")


# Zero-prefix native returns are errors, not empty compact models. Keep the
# requested-size owning diagnostics available without exposing uncommitted
# loss paths or NumPy views that could be mistaken for a fitted prefix.
zero_prefix_cases = []
for zero_gaussian_version in (1, 2):
    zero_gaussian_function = RecordingGaussianBufferFunction(
        [0], version=zero_gaussian_version)
    try:
        pycasso_core._PICASSO_LIB = RecordingGaussianBufferLibrary(
            zero_gaussian_function)
        zero_gaussian_solver = pycasso.Solver(
            scalar_finalize_x, scalar_finalize_responses['gaussian'],
            lambdas=scalar_finalize_path, family='gaussian',
            standardize=True, type_gaussian='naive')
    finally:
        pycasso_core._PICASSO_LIB = native_library
    zero_prefix_cases.append((
        f'Gaussian V{zero_gaussian_version}', zero_gaussian_solver, False))

for zero_scalar_version in (1, 2, 3):
    zero_scalar_solver, _ = make_recording_scalar_solver(
        'sqrtlasso', True, True, [0], version=zero_scalar_version,
        statuses=[7])
    zero_prefix_cases.append((
        f'sqrt-lasso V{zero_scalar_version}', zero_scalar_solver, True))

for zero_label, zero_solver, zero_has_aliases in zero_prefix_cases:
    try:
        zero_solver.train()
        raise AssertionError(f"{zero_label} accepted a zero-prefix fit")
    except PycassoError as exc:
        assert 'did not fit any lambda' in str(exc) or \
            'stopped before fitting any lambda' in str(exc) or \
            'invalid fit count' in str(exc), \
            f"{zero_label} error omitted zero-prefix context: {exc}"
    assert zero_solver.result['state'] == 'not trained' and \
        zero_solver.nlambda == len(scalar_finalize_path) and \
        np.array_equal(zero_solver.lambdas, scalar_finalize_path), \
        f"{zero_label} exposed a compact fitted path"
    for field in ('beta', 'intercept', 'ite_lamb', 'size_act',
                  'train_time', 'df'):
        assert_compact_owner(
            zero_solver.result[field], len(scalar_finalize_path),
            f"{zero_label} requested buffer {field}")
    assert 'smooth_objective' not in zero_solver.result and \
        'dev_ratio' not in zero_solver.result, \
        f"{zero_label} exposed an uncommitted loss path"
    assert zero_solver.lambdas.flags.owndata and \
        zero_solver.lambdas.base is None, \
        f"{zero_label} lambdas became a misleading prefix view"
    if zero_has_aliases:
        for field in ('lla_stages', 'objective', 'kkt', 'stationarity'):
            assert_compact_owner(
                zero_solver.result[field], len(scalar_finalize_path),
                f"{zero_label} requested diagnostic {field}")
        assert zero_solver.result['runtime'] is \
            zero_solver.result['train_time'] and \
            zero_solver.result['stages'] is \
            zero_solver.result['lla_stages'], \
            f"{zero_label} diagnostic aliases changed"


# Retraining restores the requested path and fresh native buffers without
# modifying caller-held arrays from the previous compact fit.
for retrain_family in ('gaussian', 'binomial', 'poisson', 'sqrtlasso'):
    retrain_statuses = None if retrain_family == 'gaussian' else [1, 0]
    retrain_solver, retrain_function = make_recording_scalar_solver(
        retrain_family, True, True, [2, len(scalar_finalize_path)],
        statuses=retrain_statuses)
    retrain_solver.train()
    live_dictionary = retrain_solver.result
    retrain_fields = [
        'beta', 'intercept', 'ite_lamb', 'size_act', 'train_time', 'df',
        'smooth_objective', 'dev_ratio',
    ]
    if retrain_family != 'gaussian':
        retrain_fields.extend([
            'lla_stages', 'objective', 'kkt', 'stationarity'])
    old_arrays = {
        field: retrain_solver.result[field]
        for field in retrain_fields
    }
    old_values = {field: values.copy()
                  for field, values in old_arrays.items()}
    old_lambdas = retrain_solver.lambdas
    old_lambda_values = old_lambdas.copy()
    retrain_solver.train()
    assert retrain_solver.result is live_dictionary and \
        retrain_solver.nlambda == len(scalar_finalize_path) and \
        np.array_equal(retrain_solver.lambdas, scalar_finalize_path), \
        f"{retrain_family} retrain did not restore requested path"
    latest_call = retrain_function.calls[-1]
    for field in ('beta', 'intercept', 'ite_lamb', 'size_act', 'train_time'):
        assert retrain_solver.result[field] is latest_call[field] and \
            retrain_solver.result[field] is not old_arrays[field], \
            f"{retrain_family} retrain reused old {field}"
    for field, old_array in old_arrays.items():
        assert np.array_equal(old_array, old_values[field]), \
            f"{retrain_family} retrain mutated caller-held {field}"
    assert retrain_solver.lambdas is not old_lambdas and \
        np.array_equal(old_lambdas, old_lambda_values), \
        f"{retrain_family} retrain mutated caller-held lambdas"


class V2MultinomialLibrary:
    def __init__(self, library):
        self.SolveMultinomialRegressionV2 = \
            library.SolveMultinomialRegressionV2
        self.SolveMultinomialRegression = \
            library.SolveMultinomialRegression


try:
    pycasso_core._PICASSO_LIB = V2MultinomialLibrary(native_library)
    try:
        pycasso.Solver(
            X_mn, Y_mn3, lambdas=nonconvex_path,
            family="multinomial", penalty="mcp", lla_max_stages=4)
        raise AssertionError(
            "V2 backend silently ignored nondefault lla_max_stages")
    except PycassoError as exc:
        assert 'V3' in str(exc) and 'lla_max_stages' in str(exc), \
            f"V2 compatibility error is unclear: {exc}"
finally:
    pycasso_core._PICASSO_LIB = native_library

try:
    pycasso_core._PICASSO_LIB = LegacyMultinomialLibrary(native_library)
    sm_legacy = pycasso.Solver(
        X_mn, Y_mn3, lambdas=dfmax_path, family="multinomial", dfmax=0,
        prec=1e-6, max_ite=200)
finally:
    pycasso_core._PICASSO_LIB = native_library
with warnings.catch_warnings(record=True) as legacy_warnings:
    warnings.simplefilter("always")
    sm_legacy.train()
assert len(legacy_warnings) == 1 and \
    issubclass(legacy_warnings[0].category, RuntimeWarning), \
    "truncated legacy ABI path must emit a RuntimeWarning"
assert sm_legacy.result['status_code'] is None and \
    sm_legacy.result['status'] == 'legacy_unknown', \
    "legacy ABI termination was incorrectly reported as known"
assert sm_legacy.result['state'] == 'partially trained', \
    "truncated legacy ABI path must be marked partially trained"
assert 1 < sm_legacy.nlambda < len(dfmax_path), \
    "legacy fallback did not exercise truncation"
assert_multinomial_fit(sm_legacy, sm_legacy.nlambda,
                       "legacy fallback multinomial")

print(f"  beta shape: {sm.result['beta'].shape}")
print(f"  probs sample: {probs_mn[0]}")
print(f"  dfmax retained {cut_nlambda}/{len(dfmax_path)} lambdas: "
      f"df={sm_cut.result['df'].tolist()}")
print("  PASS")

# Exact constant-response sqrt-lasso is a solved, nondifferentiable null fit.
print("\n=== Constant-response sqrt-lasso ===")
sqrt_constant_x = np.random.RandomState(731).normal(size=(37, 6))
sqrt_constant_y = np.full(37, 0.125)
sqrt_constant_path = np.array([0.4, 0.1, 0.0])
for sqrt_constant_penalty in ('l1', 'mcp', 'scad'):
    sqrt_constant_solver = pycasso.Solver(
        sqrt_constant_x, sqrt_constant_y,
        lambdas=sqrt_constant_path, family='sqrtlasso',
        penalty=sqrt_constant_penalty, prec=1e-8, max_ite=1000)
    with warnings.catch_warnings(record=True) as sqrt_constant_warnings:
        warnings.simplefilter("always")
        sqrt_constant_solver.train()
    sqrt_constant_result = sqrt_constant_solver.result
    assert not sqrt_constant_warnings, \
        f"{sqrt_constant_penalty} constant response emitted a warning"
    assert sqrt_constant_result['status_code'] == 0 and \
        sqrt_constant_result['status'] == 'completed', \
        f"{sqrt_constant_penalty} constant-response path did not complete"
    assert sqrt_constant_solver.nlambda == len(sqrt_constant_path), \
        f"{sqrt_constant_penalty} constant-response path was truncated"
    assert np.array_equal(
        sqrt_constant_result['beta'],
        np.zeros_like(sqrt_constant_result['beta'])), \
        f"{sqrt_constant_penalty} constant-response slopes are nonzero"
    assert np.array_equal(
        sqrt_constant_result['intercept'],
        np.full(len(sqrt_constant_path), sqrt_constant_y[0])), \
        f"{sqrt_constant_penalty} constant-response intercept is incorrect"
    assert np.array_equal(
        sqrt_constant_result['ite_lamb'],
        np.zeros(len(sqrt_constant_path), dtype='int32')), \
        f"{sqrt_constant_penalty} exact fit performed unnecessary updates"
    for diagnostic in ('objective', 'kkt', 'stationarity'):
        assert np.array_equal(
            sqrt_constant_result[diagnostic],
            np.zeros(len(sqrt_constant_path))), \
            f"{sqrt_constant_penalty} {diagnostic} is not exact zero"
    assert sqrt_constant_result['nulldev'] == 0.0 and np.array_equal(
        sqrt_constant_result['dev_ratio'],
        np.zeros(len(sqrt_constant_path))), \
        f"{sqrt_constant_penalty} zero null deviance was not guarded"
    for sqrt_constant_index in range(len(sqrt_constant_path)):
        assert np.array_equal(
            sqrt_constant_solver.predict(
                sqrt_constant_x[:4], lambdidx=sqrt_constant_index),
            np.full(4, sqrt_constant_y[0])), \
            f"{sqrt_constant_penalty} constant-response predictions are incorrect"
print("  PASS")

# __str__
print("\n=== __str__ ===")
summary = str(s)
print(summary)
assert "\n    0  " in summary, \
    "Solver string summary did not use zero-based Python path indices"

positive_plot_axis, positive_plot_label = \
    pycasso_core._plot_lambda_values(np.array([1.0, 0.1]), True)
zero_plot_axis, zero_plot_label = \
    pycasso_core._plot_lambda_values(np.array([1.0, 0.0]), True)
assert np.allclose(positive_plot_axis, np.log([1.0, 0.1])) and \
    positive_plot_label == 'log(lambda)', \
    "positive lambda path lost its log plotting axis"
assert np.array_equal(zero_plot_axis, [1.0, 0.0]) and \
    zero_plot_label == 'lambda' and np.all(np.isfinite(zero_plot_axis)), \
    "zero-lambda plot axis still evaluates log(0)"


class _PlotAxisProbe:
    def __init__(self):
        self.plotted_shapes = []

    def plot(self, *args, **kwargs):
        self.plotted_shapes.append(np.asarray(args[1]).shape)
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def twiny(self):
        return self

    def get_xlim(self):
        return (0.0, 1.0)

    def set_xlim(self, *args, **kwargs):
        return None

    def set_xticks(self, *args, **kwargs):
        return None

    def set_xticklabels(self, *args, **kwargs):
        return None


plot_probe = _PlotAxisProbe()
untrained_plot = pycasso.Solver(
    X[:20], Y_g[:20], lambdas=np.array([0.2, 0.1]))
try:
    untrained_plot.plot(ax=plot_probe)
    assert False, "plot accepted an untrained model"
except PycassoError:
    pass
for invalid_max_features in (0, -1, 1.5, True, "2", d + 1):
    try:
        s.plot(max_features=invalid_max_features, ax=plot_probe)
        assert False, f"plot accepted max_features={invalid_max_features!r}"
    except ValueError as exc:
        assert "max_features" in str(exc)
try:
    s.plot(log_scale="yes", ax=plot_probe)
    assert False, "plot accepted a non-Boolean log_scale"
except ValueError as exc:
    assert "log_scale" in str(exc)
original_import = builtins.__import__


def _reject_matplotlib_import(name, *args, **kwargs):
    if name.startswith("matplotlib"):
        raise AssertionError(
            "plot(ax=...) imported the optional Matplotlib dependency")
    return original_import(name, *args, **kwargs)


builtins.__import__ = _reject_matplotlib_import
try:
    s.plot(max_features=np.int64(2), ax=plot_probe)
finally:
    builtins.__import__ = original_import
assert plot_probe.plotted_shapes[-1] == (s.nlambda, 2), \
    "plot did not apply the validated max_features selection"


# ctypes function objects are cached per CDLL symbol. Constructors may run in
# parallel, but one shared symbol must be configured only once.
print("\n=== Thread-safe ctypes signature binding ===")


class CountingSignatureFunction:
    """ctypes-like function that records potentially racing assignments."""
    __name__ = 'CountingSignatureFunction'

    def __init__(self):
        self._argtypes = None
        self._restype = 'unset'
        self.argtypes_set_count = 0
        self.restype_set_count = 0

    @property
    def argtypes(self):
        return self._argtypes

    @argtypes.setter
    def argtypes(self, value):
        time.sleep(0.001)
        self.argtypes_set_count += 1
        self._argtypes = value

    @property
    def restype(self):
        return self._restype

    @restype.setter
    def restype(self, value):
        time.sleep(0.001)
        self.restype_set_count += 1
        self._restype = value


counting_function = CountingSignatureFunction()
counting_argtypes = [ctypes.c_double, ctypes.c_int]
with ThreadPoolExecutor(max_workers=32) as signature_pool:
    list(signature_pool.map(
        lambda _: pycasso_core._bind_ctypes_signature(
            counting_function, counting_argtypes, ctypes.c_int),
        range(32)))
assert counting_function.argtypes_set_count == 1 and \
    counting_function.restype_set_count == 1, \
    "concurrent ctypes binding repeated a signature assignment"

pycasso_core._bind_ctypes_signature(
    counting_function, list(counting_argtypes), ctypes.c_int)
assert counting_function.argtypes_set_count == 1 and \
    counting_function.restype_set_count == 1, \
    "repeated equivalent ctypes binding was not a no-op"

original_counting_signature = getattr(
    counting_function, pycasso_core._CTYPES_SIGNATURE_ATTR)
original_counting_argtypes = list(counting_function.argtypes)
original_counting_restype = counting_function.restype
try:
    pycasso_core._bind_ctypes_signature(
        counting_function, [ctypes.c_double], ctypes.c_int)
    raise AssertionError("conflicting ctypes signature was accepted")
except PycassoError as exc:
    assert "Conflicting ctypes signature" in str(exc), \
        f"ctypes conflict error is unclear: {exc}"
assert counting_function.argtypes == original_counting_argtypes and \
    counting_function.restype is original_counting_restype and \
    getattr(counting_function, pycasso_core._CTYPES_SIGNATURE_ATTR) == \
    original_counting_signature and \
    counting_function.argtypes_set_count == 1 and \
    counting_function.restype_set_count == 1, \
    "ctypes conflict mutated the established signature"


def _run_concurrent_native_fit(family):
    stress_x = X[:60, :8]
    stress_path = np.array([0.3, 0.15])
    responses = {
        'gaussian': Y_g[:60],
        'binomial': (np.arange(60) % 2).astype(float),
        'poisson': (1 + np.arange(60) % 4).astype(float),
        'sqrtlasso': Y_g[:60],
        'multinomial': (np.arange(60) % 3).astype(float),
    }
    kwargs = {'type_gaussian': 'naive'} if family == 'gaussian' else {}
    solver = pycasso.Solver(
        stress_x, responses[family], lambdas=stress_path,
        family=family, **kwargs)
    solver.train()
    assert solver.result['state'] == 'trained' and \
        np.all(np.isfinite(solver.result['beta'])) and \
        np.all(np.isfinite(solver.result['intercept'])), \
        f"concurrent {family} native fit returned invalid output"
    return family, solver.result['beta'].copy(), \
        solver.result['intercept'].copy()


fresh_native_library = pycasso_core._load_lib()
concurrent_families = [
    family for family in
    ('gaussian', 'binomial', 'poisson', 'sqrtlasso', 'multinomial')
    for _ in range(4)
]
try:
    pycasso_core._PICASSO_LIB = fresh_native_library
    with ThreadPoolExecutor(max_workers=20) as native_pool:
        concurrent_results = list(native_pool.map(
            _run_concurrent_native_fit, concurrent_families))
finally:
    pycasso_core._PICASSO_LIB = native_library
for family in set(concurrent_families):
    family_results = [result for result in concurrent_results
                      if result[0] == family]
    reference_beta, reference_intercept = family_results[0][1:]
    for _, beta_result, intercept_result in family_results[1:]:
        assert np.array_equal(beta_result, reference_beta) and \
            np.array_equal(intercept_result, reference_intercept), \
            f"concurrent {family} native fits were not deterministic"
print("  PASS")

print("\nAll tests passed.")
