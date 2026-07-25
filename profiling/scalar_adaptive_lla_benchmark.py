#!/usr/bin/env python3
"""Reproducible scalar adaptive-LLA benchmark for PICASSO.

The benchmark compares the historical scalar fixed-three-stage ABI with the
current versioned ABI at stage budgets 3 and 25.  Every measured call runs in
a fresh subprocess, so libraries with the same Mach-O install name cannot be
coalesced by the dynamic loader and process peak RSS remains interpretable.

Example
-------
python profiling/scalar_adaptive_lla_benchmark.py \
  --old-library /private/tmp/picasso-pre-adaptive-lla-all-20260716/lib/libpicasso.so \
  --new-library lib/libpicasso.so \
  --repeats 5 --output /tmp/scalar-adaptive-lla.json
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as datetime_module
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
WORKER_MARKER = "PICASSO_SCALAR_LLA_BENCHMARK_RESULT="
DEFAULT_SEED = 20260716
FAMILIES = ("binomial", "poisson", "sqrt")
PENALTIES = {"mcp": 2, "scad": 3}
MODES = ("old_fixed3", "new_cap3", "new_cap25")
STATUS_NAMES = {
    0: "completed",
    1: "dfmax_reached",
    2: "invalid_input",
    3: "subproblem_failed",
    4: "inner_iteration_limit",
    5: "line_search_failed",
    6: "no_descent_direction",
    7: "numerical_failure",
    8: "lla_majorization_failed",
    9: "exception",
    10: "lla_stationarity_limit",
}


def _generate_data(seed: int, n: int, d: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create one fixed standardized design and three deterministic responses."""

    rng = np.random.RandomState(seed)
    x = rng.standard_normal((n, d))
    correlation = 0.35
    innovation_scale = math.sqrt(1.0 - correlation * correlation)
    for feature in range(1, d):
        x[:, feature] = (
            correlation * x[:, feature - 1]
            + innovation_scale * x[:, feature]
        )
    x -= np.mean(x, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    x /= np.where(scale > 0.0, scale, 1.0)

    signal_beta = np.zeros(d, dtype=np.float64)
    signal_size = min(12, d)
    magnitudes = np.linspace(1.0, 0.25, signal_size)
    signal_beta[:signal_size] = magnitudes * np.where(
        np.arange(signal_size) % 2 == 0, 1.0, -1.0
    )
    signal = x @ signal_beta

    logistic_eta = -0.15 + 0.60 * signal
    logistic_probability = 1.0 / (1.0 + np.exp(-logistic_eta))
    binomial = (rng.random_sample(n) < logistic_probability).astype(np.float64)

    poisson_mean = np.exp(0.15 + 0.18 * signal)
    poisson = rng.poisson(poisson_mean).astype(np.float64)

    sqrt_response = (
        0.30 + signal + 0.45 * rng.standard_normal(n)
    ).astype(np.float64)
    responses = {
        "binomial": np.ascontiguousarray(binomial),
        "poisson": np.ascontiguousarray(poisson),
        "sqrt": np.ascontiguousarray(sqrt_response),
    }
    return np.ascontiguousarray(x, dtype=np.float64), responses


def _lambda_path(x: np.ndarray, y: np.ndarray, family: str,
                 nlambda: int, lambda_min_ratio: float) -> np.ndarray:
    """Compute lambda_max from the independently evaluated null model."""

    n = x.shape[0]
    if family == "binomial":
        mean_y = float(np.mean(y))
        if not 0.0 < mean_y < 1.0:
            raise RuntimeError("binomial response has a degenerate mean")
        smooth_gradient = x.T @ (mean_y - y) / n
    elif family == "poisson":
        mean_y = float(np.mean(y))
        if not mean_y > 0.0:
            raise RuntimeError("Poisson response has a non-positive mean")
        smooth_gradient = x.T @ (mean_y - y) / n
    else:
        residual = y - np.mean(y)
        loss = math.sqrt(float(np.mean(residual * residual)))
        smooth_gradient = -(x.T @ residual) / (n * loss)

    lambda_max = float(np.max(np.abs(smooth_gradient)))
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        raise RuntimeError("generated data produced an invalid lambda_max")
    return np.ascontiguousarray(
        np.geomspace(lambda_max, lambda_max * lambda_min_ratio, nlambda),
        dtype=np.float64,
    )


def _penalty_derivative(penalty: str, absolute_value: np.ndarray,
                        current_lambda: float, gamma: float) -> np.ndarray:
    if penalty == "mcp":
        return np.maximum(0.0, current_lambda - absolute_value / gamma)
    derivative = np.full_like(absolute_value, current_lambda)
    middle = absolute_value > current_lambda
    derivative[middle] = np.maximum(
        0.0,
        current_lambda
        - (absolute_value[middle] - current_lambda) / (gamma - 1.0),
    )
    return derivative


def _penalty_value(penalty: str, absolute_value: np.ndarray,
                   current_lambda: float, gamma: float) -> np.ndarray:
    if penalty == "mcp":
        boundary = gamma * current_lambda
        return np.where(
            absolute_value < boundary,
            absolute_value * (current_lambda - absolute_value / (2.0 * gamma)),
            0.5 * gamma * current_lambda * current_lambda,
        )

    value = np.empty_like(absolute_value)
    lower = absolute_value <= current_lambda
    upper = absolute_value >= gamma * current_lambda
    middle = ~(lower | upper)
    value[lower] = current_lambda * absolute_value[lower]
    offset = absolute_value[middle] - current_lambda
    value[middle] = (
        current_lambda * current_lambda
        + offset
        * (current_lambda - offset / (2.0 * (gamma - 1.0)))
    )
    value[upper] = 0.5 * (gamma + 1.0) * current_lambda * current_lambda
    return value


def _evaluate_path(x: np.ndarray, y: np.ndarray, family: str, penalty: str,
                   gamma: float, lambdas: np.ndarray, beta: np.ndarray,
                   intercept: np.ndarray) -> Tuple[List[float], List[float]]:
    """Independently compute the true nonconvex objective and stationarity."""

    n = x.shape[0]
    objectives: List[float] = []
    stationarity: List[float] = []
    for index, current_lambda in enumerate(lambdas):
        coefficient = beta[index]
        eta = intercept[index] + x @ coefficient
        if family == "binomial":
            probability = np.empty_like(eta)
            nonnegative = eta >= 0.0
            probability[nonnegative] = 1.0 / (
                1.0 + np.exp(-eta[nonnegative])
            )
            exponential = np.exp(eta[~nonnegative])
            probability[~nonnegative] = exponential / (1.0 + exponential)
            smooth_objective = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
            gradient = x.T @ (probability - y) / n
            intercept_gradient = float(np.mean(probability - y))
        elif family == "poisson":
            mean = np.exp(eta)
            smooth_objective = float(np.mean(mean - y * eta))
            gradient = x.T @ (mean - y) / n
            intercept_gradient = float(np.mean(mean - y))
        else:
            residual = y - eta
            smooth_objective = math.sqrt(float(np.mean(residual * residual)))
            gradient = -(x.T @ residual) / (n * smooth_objective)
            intercept_gradient = -float(np.mean(residual)) / smooth_objective

        absolute_value = np.abs(coefficient)
        derivative = _penalty_derivative(
            penalty, absolute_value, float(current_lambda), gamma
        )
        penalty_sum = float(np.sum(_penalty_value(
            penalty, absolute_value, float(current_lambda), gamma
        )))
        coefficient_residual = np.maximum(np.abs(gradient) - derivative, 0.0)
        positive = coefficient > 1e-8
        negative = coefficient < -1e-8
        coefficient_residual[positive] = np.abs(
            gradient[positive] + derivative[positive]
        )
        coefficient_residual[negative] = np.abs(
            gradient[negative] - derivative[negative]
        )
        objectives.append(smooth_objective + penalty_sum)
        stationarity.append(max(
            float(np.max(coefficient_residual)), abs(intercept_gradient)
        ))
    return objectives, stationarity


P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT = ctypes.POINTER(ctypes.c_int)


def _double_pointer(array: np.ndarray) -> P_DOUBLE:
    return array.ctypes.data_as(P_DOUBLE)


def _int_pointer(array: np.ndarray) -> P_INT:
    return array.ctypes.data_as(P_INT)


def _register_solver(library_path: Path, family: str, versioned: bool):
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    base_name = {
        "binomial": "SolveLogisticRegression",
        "poisson": "SolvePoissonRegression",
        "sqrt": "SolveSqrtLinearRegression",
    }[family]
    symbol = base_name + ("V2" if versioned else "")
    try:
        solve = getattr(library, symbol)
    except AttributeError as exc:
        raise RuntimeError(f"{library_path} does not export {symbol}") from exc

    common = [
        P_DOUBLE, P_DOUBLE, ctypes.c_int, ctypes.c_int, P_DOUBLE,
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_bool, ctypes.c_int,
    ]
    if family != "sqrt":
        common.append(P_DOUBLE)  # offset
    outputs = [
        P_DOUBLE, P_DOUBLE, P_INT, P_INT, P_DOUBLE, P_INT, ctypes.c_bool,
    ]
    if versioned:
        outputs.extend([
            ctypes.c_int, P_INT, P_INT, P_INT,
            P_DOUBLE, P_DOUBLE, P_DOUBLE,
        ])
        solve.restype = ctypes.c_int
    else:
        solve.restype = None
    solve.argtypes = common + outputs
    return library, solve


def _rss_bytes(raw_rss: int) -> int:
    return int(raw_rss) if sys.platform == "darwin" else int(raw_rss) * 1024


def _worker_run(library_path: Path, mode: str, family: str, penalty: str,
                seed: int, n: int, d: int, nlambda: int,
                lambda_min_ratio: float, gamma: float, max_ite: int,
                precision: float) -> dict:
    x, responses = _generate_data(seed, n, d)
    y = responses[family]
    lambdas = _lambda_path(x, y, family, nlambda, lambda_min_ratio)
    x_flat = np.ascontiguousarray(x.reshape(-1), dtype=np.float64)
    beta_flat = np.zeros(nlambda * d, dtype=np.float64)
    intercept = np.zeros(nlambda, dtype=np.float64)
    iterations = np.zeros(nlambda, dtype=np.int32)
    size_active = np.zeros(nlambda, dtype=np.int32)
    solver_time = np.zeros(nlambda, dtype=np.float64)
    num_fit_array = np.zeros(1, dtype=np.int32)
    offset = np.zeros(n, dtype=np.float64)

    versioned = mode != "old_fixed3"
    stage_budget = 3 if mode != "new_cap25" else 25
    stages = np.zeros(nlambda, dtype=np.int32)
    native_objective = np.full(nlambda, np.nan, dtype=np.float64)
    native_kkt = np.full(nlambda, np.nan, dtype=np.float64)
    native_stationarity = np.full(nlambda, np.nan, dtype=np.float64)
    failed_lambda = np.full(1, -1, dtype=np.int32)
    failed_stage = np.full(1, -1, dtype=np.int32)

    library, solve = _register_solver(library_path, family, versioned)
    arguments = [
        _double_pointer(y), _double_pointer(x_flat), n, d,
        _double_pointer(lambdas), nlambda, gamma, max_ite, precision,
        PENALTIES[penalty], True, -1,
    ]
    if family != "sqrt":
        arguments.append(_double_pointer(offset))
    arguments.extend([
        _double_pointer(beta_flat), _double_pointer(intercept),
        _int_pointer(iterations), _int_pointer(size_active),
        _double_pointer(solver_time), _int_pointer(num_fit_array), True,
    ])
    if versioned:
        arguments.extend([
            stage_budget, _int_pointer(failed_lambda),
            _int_pointer(failed_stage), _int_pointer(stages),
            _double_pointer(native_objective), _double_pointer(native_kkt),
            _double_pointer(native_stationarity),
        ])

    rss_before = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    start_ns = time.perf_counter_ns()
    status_code = solve(*arguments)
    wall_time_seconds = (time.perf_counter_ns() - start_ns) / 1e9
    peak_rss_bytes = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    del library

    if not versioned:
        status_code = None
    actual_fit = int(num_fit_array[0])
    if actual_fit < 1 or actual_fit > nlambda:
        raise RuntimeError(
            f"{mode}/{family}/{penalty} returned num_fit={actual_fit}"
        )
    beta = beta_flat[:actual_fit * d].reshape(actual_fit, d)
    independent_objective, independent_stationarity = _evaluate_path(
        x, y, family, penalty, gamma, lambdas[:actual_fit], beta,
        intercept[:actual_fit]
    )

    output_digest = hashlib.sha256()
    for output in (beta_flat, intercept, iterations, size_active, num_fit_array):
        output_digest.update(output.tobytes())

    native_objective_error = None
    native_stationarity_error = None
    if versioned:
        native_objective_error = float(np.max(np.abs(
            native_objective[:actual_fit]
            - np.asarray(independent_objective, dtype=np.float64)
        )))
        native_stationarity_error = float(np.max(np.abs(
            native_stationarity[:actual_fit]
            - np.asarray(independent_stationarity, dtype=np.float64)
        )))

    return {
        "mode": mode,
        "family": family,
        "penalty": penalty,
        "status_code": status_code,
        "status": (
            "legacy_void" if status_code is None
            else STATUS_NAMES.get(int(status_code), "unknown")
        ),
        "failed_lambda": int(failed_lambda[0]) if versioned else None,
        "failed_stage": int(failed_stage[0]) if versioned else None,
        "num_fit": actual_fit,
        "wall_time_seconds": wall_time_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "rss_before_call_bytes": rss_before,
        "peak_rss_increment_bytes": max(0, peak_rss_bytes - rss_before),
        "lambda": lambdas[:actual_fit].tolist(),
        "independent_objective": independent_objective,
        "independent_stationarity": independent_stationarity,
        "native_objective_max_abs_error": native_objective_error,
        "native_stationarity_max_abs_error": native_stationarity_error,
        "stages": (
            stages[:actual_fit].astype(int).tolist()
            if versioned else [3] * actual_fit
        ),
        "iterations": iterations[:actual_fit].astype(int).tolist(),
        "size_active": size_active[:actual_fit].astype(int).tolist(),
        "output_sha256": output_digest.hexdigest(),
        "all_outputs_finite": bool(
            np.all(np.isfinite(beta))
            and np.all(np.isfinite(intercept[:actual_fit]))
            and np.all(np.isfinite(independent_objective))
            and np.all(np.isfinite(independent_stationarity))
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def _summarize_runs(runs: Sequence[dict], precision: float) -> dict:
    num_fit = [run["num_fit"] for run in runs]
    objective_arrays = [np.asarray(run["independent_objective"]) for run in runs]
    stationarity_arrays = [
        np.asarray(run["independent_stationarity"]) for run in runs
    ]
    objective_spread = math.inf
    stationarity_spread = math.inf
    if len({array.size for array in objective_arrays}) == 1:
        objective_stack = np.vstack(objective_arrays)
        stationarity_stack = np.vstack(stationarity_arrays)
        objective_spread = float(np.max(np.ptp(objective_stack, axis=0)))
        stationarity_spread = float(np.max(np.ptp(stationarity_stack, axis=0)))
        objective_median = np.median(objective_stack, axis=0).tolist()
        stationarity_median = np.median(stationarity_stack, axis=0).tolist()
    else:
        objective_median = None
        stationarity_median = None

    rss = [run["peak_rss_bytes"] for run in runs]
    rss_increment = [run["peak_rss_increment_bytes"] for run in runs]
    elapsed = [run["wall_time_seconds"] for run in runs]
    status = sorted({run["status"] for run in runs})
    stages = [run["stages"] for run in runs]
    output_hashes = sorted({run["output_sha256"] for run in runs})
    maximum_stationarity = (
        max(stationarity_median) if stationarity_median else None
    )
    certified_count = (
        sum(value <= precision for value in stationarity_median)
        if stationarity_median else None
    )
    return {
        "wall_time_seconds": {
            "median": _median(elapsed), "minimum": min(elapsed),
            "maximum": max(elapsed),
        },
        "peak_rss_bytes": {"median": _median(rss), "maximum": max(rss)},
        "peak_rss_increment_bytes": {
            "median": _median(rss_increment), "maximum": max(rss_increment),
        },
        "num_fit": {"minimum": min(num_fit), "maximum": max(num_fit)},
        "status": status,
        "objective_median": objective_median,
        "objective_last": objective_median[-1] if objective_median else None,
        "objective_max_repeat_spread": objective_spread,
        "stationarity_median": stationarity_median,
        "stationarity_max": maximum_stationarity,
        "stationarity_last": (
            stationarity_median[-1] if stationarity_median else None
        ),
        "stationarity_max_repeat_spread": stationarity_spread,
        "certified_points": certified_count,
        "maximum_stage": max(max(value) for value in stages),
        "median_total_iterations": _median(
            sum(run["iterations"]) for run in runs
        ),
        "all_outputs_finite": all(run["all_outputs_finite"] for run in runs),
        "outputs_byte_stable": len(output_hashes) == 1,
        "output_sha256": output_hashes,
        "native_objective_max_abs_error": max(
            (run["native_objective_max_abs_error"] or 0.0) for run in runs
        ),
        "native_stationarity_max_abs_error": max(
            (run["native_stationarity_max_abs_error"] or 0.0) for run in runs
        ),
    }


def _invoke_worker(script: Path, library_path: Path, mode: str, family: str,
                   penalty: str, args: argparse.Namespace) -> dict:
    command = [
        sys.executable, str(script), "--_worker",
        "--worker-library", str(library_path), "--worker-mode", mode,
        "--worker-family", family, "--worker-penalty", penalty,
        "--seed", str(args.seed), "--n", str(args.n), "--d", str(args.d),
        "--nlambda", str(args.nlambda),
        "--lambda-min-ratio", repr(args.lambda_min_ratio),
        "--gamma", repr(args.gamma), "--max-ite", str(args.max_ite),
        "--precision", repr(args.precision),
    ]
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {mode}/{family}/{penalty} "
            f"(exit {completed.returncode})\nstdout:\n{completed.stdout}"
            f"\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(WORKER_MARKER):
            return json.loads(line[len(WORKER_MARKER):])
    raise RuntimeError(f"worker produced no result:\n{completed.stdout}")


def _build_comparisons(mode_results: Dict[str, dict], cases: Sequence[str]) -> dict:
    comparisons = {}
    for case in cases:
        old = mode_results["old_fixed3"]["cases"][case]["summary"]
        cap3 = mode_results["new_cap3"]["cases"][case]["summary"]
        cap25 = mode_results["new_cap25"]["cases"][case]["summary"]

        def objective_delta(first: dict, second: dict) -> Optional[float]:
            first_path = first["objective_median"]
            second_path = second["objective_median"]
            if first_path is None or second_path is None:
                return None
            length = min(len(first_path), len(second_path))
            if length == 0:
                return None
            return float(np.max(np.asarray(second_path[:length])
                                - np.asarray(first_path[:length])))

        comparisons[case] = {
            "new_cap3_time_over_old": (
                cap3["wall_time_seconds"]["median"]
                / old["wall_time_seconds"]["median"]
            ),
            "new_cap25_time_over_new_cap3": (
                cap25["wall_time_seconds"]["median"]
                / cap3["wall_time_seconds"]["median"]
            ),
            "new_cap3_peak_rss_over_old": (
                cap3["peak_rss_bytes"]["median"]
                / old["peak_rss_bytes"]["median"]
            ),
            "new_cap25_peak_rss_over_new_cap3": (
                cap25["peak_rss_bytes"]["median"]
                / cap3["peak_rss_bytes"]["median"]
            ),
            "new_cap3_max_objective_increase_over_old": objective_delta(old, cap3),
            "new_cap25_max_objective_increase_over_new_cap3": objective_delta(cap3, cap25),
        }
    return comparisons


def _run_controller(args: argparse.Namespace) -> dict:
    old_library = Path(args.old_library).expanduser().resolve()
    new_library = Path(args.new_library).expanduser().resolve()
    for library in (old_library, new_library):
        if not library.is_file():
            raise FileNotFoundError(f"library does not exist: {library}")

    cases = [f"{family}_{penalty}" for family in FAMILIES for penalty in PENALTIES]
    libraries = {
        "old_fixed3": old_library,
        "new_cap3": new_library,
        "new_cap25": new_library,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime_module.datetime.now(
            datetime_module.timezone.utc
        ).isoformat(),
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "config": {
            "seed": args.seed, "n": args.n, "d": args.d,
            "nlambda": args.nlambda, "lambda_min_ratio": args.lambda_min_ratio,
            "gamma": args.gamma, "max_ite": args.max_ite,
            "precision": args.precision, "intercept": True, "dfmax": -1,
            "repeats": args.repeats, "warmups": args.warmups, "threads": 1,
            "cases": cases,
            "diagnostic_core_bytes_estimate": 32 * args.nlambda + 12,
            "diagnostic_c_output_bytes": 28 * args.nlambda,
        },
        "libraries": {
            "old": {"path": str(old_library), "sha256": _file_sha256(old_library),
                    "size_bytes": old_library.stat().st_size,
                    "mtime_ns": old_library.stat().st_mtime_ns},
            "new": {"path": str(new_library), "sha256": _file_sha256(new_library),
                    "size_bytes": new_library.stat().st_size,
                    "mtime_ns": new_library.stat().st_mtime_ns},
        },
        "modes": {},
    }
    for mode in MODES:
        result["modes"][mode] = {"library": str(libraries[mode]), "cases": {}}

    script = Path(__file__).resolve()
    runs_by_mode = {mode: {case: [] for case in cases} for mode in MODES}
    for case in cases:
        family, penalty = case.rsplit("_", 1)
        for warmup in range(args.warmups):
            order = list(MODES) if warmup % 2 == 0 else list(reversed(MODES))
            for mode in order:
                _invoke_worker(script, libraries[mode], mode, family, penalty, args)
        for repeat in range(args.repeats):
            order = list(MODES) if repeat % 2 == 0 else list(reversed(MODES))
            for mode in order:
                runs_by_mode[mode][case].append(_invoke_worker(
                    script, libraries[mode], mode, family, penalty, args
                ))

    for mode in MODES:
        for case in cases:
            runs = runs_by_mode[mode][case]
            result["modes"][mode]["cases"][case] = {
                "runs": runs,
                "summary": _summarize_runs(runs, args.precision),
            }
    result["comparisons"] = _build_comparisons(result["modes"], cases)
    return result


def _argument_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare old fixed-3 and current adaptive scalar LLA."
    )
    parser.add_argument(
        "--old-library",
        default=("/private/tmp/picasso-pre-adaptive-lla-all-20260716/"
                 "lib/libpicasso.so"),
    )
    parser.add_argument(
        "--new-library", default=str(repository_root / "lib" / "libpicasso.so")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--d", type=int, default=120)
    parser.add_argument("--nlambda", type=int, default=24)
    parser.add_argument("--lambda-min-ratio", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=3.5)
    parser.add_argument("--max-ite", type=int, default=1000)
    parser.add_argument("--precision", type=float, default=1e-7)
    parser.add_argument("--output", default="-", metavar="PATH")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-library", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-family", choices=FAMILIES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-penalty", choices=tuple(PENALTIES),
                        help=argparse.SUPPRESS)
    return parser


def _validate_arguments(parser: argparse.ArgumentParser,
                        args: argparse.Namespace) -> None:
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups non-negative")
    if args.n < 2 or args.d < 1 or args.nlambda < 1:
        parser.error("--n, --d, and --nlambda must define a non-empty problem")
    if not 0.0 < args.lambda_min_ratio <= 1.0:
        parser.error("--lambda-min-ratio must be in (0, 1]")
    if not math.isfinite(args.gamma) or args.gamma <= 2.0:
        parser.error("--gamma must exceed 2 for the shared MCP/SCAD run")
    if args.max_ite < 1 or not math.isfinite(args.precision) or args.precision <= 0.0:
        parser.error("--max-ite and --precision must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    if args._worker:
        if not all((args.worker_library, args.worker_mode,
                    args.worker_family, args.worker_penalty)):
            parser.error("worker mode requires library, mode, family, and penalty")
        result = _worker_run(
            Path(args.worker_library).resolve(), args.worker_mode,
            args.worker_family, args.worker_penalty, args.seed, args.n, args.d,
            args.nlambda, args.lambda_min_ratio, args.gamma, args.max_ite,
            args.precision,
        )
        print(WORKER_MARKER + json.dumps(result, sort_keys=True))
        return 0

    try:
        result = _run_controller(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(serialized)
    else:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
