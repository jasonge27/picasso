#!/usr/bin/env python3
"""Verify PICASSO's declared and dynamically exported C ABI."""

import argparse
import ctypes
import math
import os
import re
import sys
from pathlib import Path


EXPECTED_SYMBOLS = (
    "PicassoLlaPathStatusString",
    "PicassoMultinomialPathStatusString",
    "SolveLinearRegressionCovUpdate",
    "SolveLinearRegressionCovUpdateV2",
    "SolveLinearRegressionCovUpdateV3",
    "SolveLinearRegressionNaiveUpdate",
    "SolveLinearRegressionNaiveUpdateV2",
    "SolveLinearRegressionNaiveUpdateV3",
    "SolveLogisticRegression",
    "SolveLogisticRegressionV2",
    "SolveLogisticRegressionV3",
    "SolveMultinomialRegression",
    "SolveMultinomialRegressionV2",
    "SolveMultinomialRegressionV3",
    "SolveMultinomialRegressionV4",
    "SolveMultinomialRegressionV5",
    "SolvePoissonRegression",
    "SolvePoissonRegressionV2",
    "SolvePoissonRegressionV3",
    "SolveSqrtLinearRegression",
    "SolveSqrtLinearRegressionV2",
    "SolveSqrtLinearRegressionV3",
)

_RETURN_TYPE = r"(?:(?:const\s+char\s*\*)|void|int)"
_ANNOTATED_DECLARATION = re.compile(
    r'extern\s+"C"\s+PICASSO_C_API\s+' + _RETURN_TYPE +
    r"\s*([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
_C_DECLARATION = re.compile(
    r'extern\s+"C"\s+(?:PICASSO_C_API\s+)?' + _RETURN_TYPE +
    r"\s*([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


def _set_difference_message(actual):
    expected = set(EXPECTED_SYMBOLS)
    actual = set(actual)
    return "missing={}; unexpected={}".format(
        sorted(expected - actual), sorted(actual - expected))


def _verify_header(header_path):
    header = header_path.read_text(encoding="utf-8")
    annotated = _ANNOTATED_DECLARATION.findall(header)
    declared = _C_DECLARATION.findall(header)

    if len(annotated) != len(set(annotated)):
        raise RuntimeError(
            "duplicate PICASSO_C_API declarations in {}".format(header_path))
    if set(annotated) != set(EXPECTED_SYMBOLS):
        raise RuntimeError(
            "annotated C API differs from the explicit ABI in {}: {}".format(
                header_path, _set_difference_message(annotated)))
    if len(declared) != len(set(declared)):
        raise RuntimeError(
            "duplicate extern C declarations in {}".format(header_path))
    if set(declared) != set(EXPECTED_SYMBOLS):
        raise RuntimeError(
            "extern C declarations differ from the explicit ABI in {}: {}".format(
                header_path, _set_difference_message(declared)))


def _load_library(library_path):
    directory_handle = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        directory_handle = os.add_dll_directory(str(library_path.parent))
    try:
        return ctypes.CDLL(str(library_path))
    finally:
        if directory_handle is not None:
            directory_handle.close()


def _verify_library(library_path):
    library = _load_library(library_path)
    for symbol in EXPECTED_SYMBOLS:
        try:
            getattr(library, symbol)
        except AttributeError as error:
            raise RuntimeError(
                "{} does not export {}".format(library_path, symbol)) from error

    for symbol in (
            "PicassoLlaPathStatusString",
            "PicassoMultinomialPathStatusString"):
        status_string = getattr(library, symbol)
        status_string.argtypes = [ctypes.c_int]
        status_string.restype = ctypes.c_char_p
        if status_string(0) != b"completed":
            raise RuntimeError(
                "{} returned an unexpected value for status 0".format(symbol))

    # Exercise one versioned solver, not just symbol lookup, so the smoke test
    # also catches a Windows calling-convention or bool-ABI mismatch.
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    gaussian = library.SolveLinearRegressionNaiveUpdateV2
    gaussian.argtypes = [
        double_pointer, double_pointer, ctypes.c_int, ctypes.c_int,
        double_pointer, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        double_pointer, double_pointer, int_pointer, int_pointer,
        double_pointer, int_pointer, ctypes.c_bool, double_pointer,
    ]
    gaussian.restype = None

    response = (ctypes.c_double * 4)(-1.5, -0.5, 0.5, 1.5)
    design = (ctypes.c_double * 4)(-1.5, -0.5, 0.5, 1.5)
    regularization = (ctypes.c_double * 1)(10.0)
    beta = (ctypes.c_double * 1)(math.nan)
    intercept = (ctypes.c_double * 1)(math.nan)
    iterations = (ctypes.c_int * 1)(-1)
    active_size = (ctypes.c_int * 1)(-1)
    runtime = (ctypes.c_double * 1)(math.nan)
    number_fit = ctypes.c_int(-1)
    smooth_objective = (ctypes.c_double * 1)(math.nan)
    gaussian(
        response, design, 4, 1, regularization, 1, 3.0, 1000, 1e-9,
        1, True, -1, beta, intercept, iterations, active_size, runtime,
        ctypes.byref(number_fit), False, smooth_objective)
    floating_outputs = (
        beta[0], intercept[0], runtime[0], smooth_objective[0])
    if (number_fit.value != 1 or iterations[0] < 0 or active_size[0] < 0 or
            not all(math.isfinite(value) for value in floating_outputs)):
        raise RuntimeError(
            "SolveLinearRegressionNaiveUpdateV2 runtime smoke failed: "
            "num_fit={}, iterations={}, active_size={}, outputs={}".format(
                number_fit.value, iterations[0], active_size[0],
                floating_outputs))

    # V3 retains the V2 argument order and adds explicit path termination
    # diagnostics. Exercise the new tail argument and integer return value so
    # packaging tests catch ABI drift on every supported platform.
    gaussian_v3 = library.SolveLinearRegressionNaiveUpdateV3
    gaussian_v3.argtypes = gaussian.argtypes + [int_pointer]
    gaussian_v3.restype = ctypes.c_int
    beta_v3 = (ctypes.c_double * 1)(0.0)
    intercept_v3 = (ctypes.c_double * 1)(0.0)
    iterations_v3 = (ctypes.c_int * 1)(-1)
    active_size_v3 = (ctypes.c_int * 1)(-1)
    runtime_v3 = (ctypes.c_double * 1)(math.nan)
    number_fit_v3 = ctypes.c_int(-1)
    smooth_objective_v3 = (ctypes.c_double * 1)(math.nan)
    failed_lambda = ctypes.c_int(-2)
    status = gaussian_v3(
        response, design, 4, 1, regularization, 1, 3.0, 1000, 1e-9,
        1, True, -1, beta_v3, intercept_v3, iterations_v3,
        active_size_v3, runtime_v3, ctypes.byref(number_fit_v3), False,
        smooth_objective_v3, ctypes.byref(failed_lambda))
    floating_outputs_v3 = (
        beta_v3[0], intercept_v3[0], runtime_v3[0],
        smooth_objective_v3[0])
    if (status != 0 or failed_lambda.value != -1 or
            number_fit_v3.value != 1 or iterations_v3[0] < 0 or
            active_size_v3[0] < 0 or
            not all(math.isfinite(value)
                    for value in floating_outputs_v3)):
        raise RuntimeError(
            "SolveLinearRegressionNaiveUpdateV3 runtime smoke failed: "
            "status={}, failed_lambda={}, num_fit={}, iterations={}, "
            "active_size={}, outputs={}".format(
                status, failed_lambda.value, number_fit_v3.value,
                iterations_v3[0], active_size_v3[0],
                floating_outputs_v3))


def _arguments():
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", help="PICASSO .dll, .dylib, or .so to load")
    parser.add_argument(
        "--header",
        default=str(repository_root / "include" / "picasso" / "c_api.hpp"),
        help="C API header whose annotated declarations are checked",
    )
    return parser.parse_args()


def main():
    arguments = _arguments()
    library_path = Path(arguments.library).expanduser().resolve()
    header_path = Path(arguments.header).expanduser().resolve()
    if not library_path.is_file():
        raise RuntimeError("native library is not a file: {}".format(library_path))
    if not header_path.is_file():
        raise RuntimeError("C API header is not a file: {}".format(header_path))

    _verify_header(header_path)
    _verify_library(library_path)
    print("C API export verification passed ({} symbols): {}".format(
        len(EXPECTED_SYMBOLS), library_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
