#!/usr/bin/env python3
"""Reproducible ctypes benchmark for PICASSO multinomial regression.

The runner deliberately avoids importing ``pycasso`` so that two arbitrary
shared libraries can be compared in the same checkout.  Each measured run is
executed in a fresh subprocess; this both isolates the loaded library and makes
``resource.ru_maxrss`` a useful per-run peak-RSS measurement.

Examples
--------
Benchmark the current library::

    python profiling/multinomial_benchmark.py \
        --library legacy=lib/libpicasso.so --output baseline.json

Compare two implementations::

    python profiling/multinomial_benchmark.py \
        --library legacy=/path/to/legacy/libpicasso.so \
        --library new=lib/libpicasso.so --output comparison.json
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _datetime
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
WORKER_MARKER = "PICASSO_MULTINOMIAL_BENCHMARK_RESULT="
DEFAULT_SEED = 1729


@dataclass(frozen=True)
class CaseSpec:
    """Dimensions and signal size for one deterministic workload."""

    name: str
    n: int
    d: int
    k: int
    signal_features: int
    seed_offset: int


CASES: Dict[str, CaseSpec] = {
    "tiny": CaseSpec("tiny", n=60, d=8, k=3, signal_features=3,
                     seed_offset=0),
    "p_gt_n": CaseSpec("p_gt_n", n=96, d=600, k=4,
                       signal_features=12, seed_offset=101),
    "high_k": CaseSpec("high_k", n=240, d=32, k=12,
                       signal_features=8, seed_offset=211),
    "high_n": CaseSpec("high_n", n=4000, d=24, k=4,
                       signal_features=8, seed_offset=307),
}


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def _generate_case(spec: CaseSpec, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a fixed, standardized sparse multinomial problem."""

    rng = np.random.default_rng(seed + spec.seed_offset)
    x = rng.standard_normal((spec.n, spec.d))
    x -= np.mean(x, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    x /= np.where(scale > 0.0, scale, 1.0)

    beta = np.zeros((spec.k, spec.d), dtype=np.float64)
    beta[:, :spec.signal_features] = rng.normal(
        loc=0.0, scale=0.55,
        size=(spec.k, spec.signal_features),
    )
    # A mean-zero class gauge keeps logits well scaled without choosing a
    # reference class.
    beta -= np.mean(beta, axis=0, keepdims=True)
    intercept = np.linspace(-0.25, 0.25, spec.k, dtype=np.float64)
    intercept -= np.mean(intercept)

    probability = _stable_softmax(x @ beta.T + intercept)
    draw = rng.random(spec.n)
    y = np.sum(draw[:, None] > np.cumsum(probability, axis=1), axis=1)
    y = np.minimum(y, spec.k - 1).astype(np.int64)

    # All K classes must be present because K is an explicit C-ABI argument.
    # Enforcing this deterministically also protects very small future cases.
    y[:spec.k] = np.arange(spec.k, dtype=np.int64)
    return (np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.int64))


def _lambda_path(x: np.ndarray, y: np.ndarray, k: int,
                 nlambda: int, lambda_min_ratio: float) -> np.ndarray:
    n = x.shape[0]
    proportions = np.bincount(y, minlength=k).astype(np.float64) / n
    lambda_max = 0.0
    for class_index in range(k):
        residual = (y == class_index).astype(np.float64) - proportions[class_index]
        lambda_max = max(lambda_max,
                         float(np.max(np.abs(x.T @ residual))) / n)
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        raise RuntimeError("generated data produced an invalid lambda_max")
    return np.ascontiguousarray(
        np.geomspace(lambda_max, lambda_max * lambda_min_ratio, nlambda),
        dtype=np.float64,
    )


def _register_solver(library_path: Path):
    library = ctypes.CDLL(str(library_path))
    abi_version = 1
    try:
        solve = library.SolveMultinomialRegressionV4
    except AttributeError as exc:
        try:
            solve = library.SolveMultinomialRegression
        except AttributeError:
            raise RuntimeError(
                f"{library_path} does not export a multinomial solver"
            ) from exc
    else:
        abi_version = 4

    double_array = np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS",)
    )
    int_array = np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=1, flags=("C_CONTIGUOUS",)
    )
    base_argtypes = [
        double_array, double_array, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        double_array, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        double_array, double_array, int_array, int_array, double_array,
        int_array, ctypes.c_bool,
    ]
    if abi_version == 4:
        long_array = np.ctypeslib.ndpointer(
            dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS",)
        )
        solve.argtypes = base_argtypes + [
            ctypes.c_int, ctypes.c_bool, int_array, int_array, int_array,
            long_array, long_array, double_array, double_array, double_array,
        ]
        solve.restype = ctypes.c_int
    else:
        solve.argtypes = base_argtypes
        solve.restype = None
    return library, solve, abi_version


def _rss_bytes(raw_rss: int) -> int:
    # macOS reports bytes; Linux and the BSDs report KiB.
    if sys.platform == "darwin":
        return int(raw_rss)
    return int(raw_rss) * 1024


def _evaluate_path(x: np.ndarray, y: np.ndarray, lambdas: np.ndarray,
                   beta: np.ndarray, intercept: np.ndarray
                   ) -> Tuple[List[float], List[float], dict]:
    objectives: List[float] = []
    kkt_residuals: List[float] = []
    all_finite = True
    minimum_probability = math.inf
    maximum_probability = -math.inf
    maximum_row_sum_error = 0.0

    row_index = np.arange(x.shape[0])
    for lambda_index, current_lambda in enumerate(lambdas):
        logits = x @ beta[lambda_index].T + intercept[lambda_index]
        probability = _stable_softmax(logits)
        finite = bool(np.all(np.isfinite(probability)))
        all_finite = all_finite and finite
        minimum_probability = min(minimum_probability,
                                  float(np.min(probability)))
        maximum_probability = max(maximum_probability,
                                  float(np.max(probability)))
        maximum_row_sum_error = max(
            maximum_row_sum_error,
            float(np.max(np.abs(np.sum(probability, axis=1) - 1.0))),
        )

        chosen_logits = logits[row_index, y]
        max_logits = np.max(logits, axis=1)
        log_exponential_sum = np.log(
            np.sum(np.exp(logits - max_logits[:, None]), axis=1)
        )
        # Avoid subtracting two nearly equal, very large values after forming
        # logsumexp.  This matches the stable native objective exactly.
        nll = float(np.mean(
            log_exponential_sum + (max_logits - chosen_logits)
        ))
        objective = nll + float(current_lambda) * float(
            np.sum(np.abs(beta[lambda_index]))
        )
        objectives.append(objective)

        # KKT residual for the full-K, elementwise-L1 parameterization.  The
        # intercept is unpenalized.  Computing the class correction without a
        # dense one-hot matrix keeps the benchmark memory honest.
        gradient = probability.T @ x
        for class_index in range(beta.shape[1]):
            gradient[class_index] -= np.sum(
                x[y == class_index], axis=0
            )
        gradient /= x.shape[0]
        intercept_gradient = (
            np.mean(probability, axis=0) -
            np.bincount(y, minlength=beta.shape[1]) / x.shape[0]
        )
        current_beta = beta[lambda_index]
        nonzero = np.abs(current_beta) > 1e-8
        coefficient_residual = np.maximum(
            np.abs(gradient) - float(current_lambda), 0.0
        )
        coefficient_residual[nonzero] = np.abs(
            gradient[nonzero] +
            float(current_lambda) * np.sign(current_beta[nonzero])
        )
        kkt_residuals.append(max(
            float(np.max(coefficient_residual)),
            float(np.max(np.abs(intercept_gradient))),
        ))

    within_bounds = (minimum_probability >= -1e-15 and
                     maximum_probability <= 1.0 + 1e-15)
    validity = {
        "all_finite": all_finite,
        "within_bounds": bool(within_bounds),
        "rows_sum_to_one": maximum_row_sum_error <= 1e-12,
        "all_valid": bool(all_finite and within_bounds and
                          maximum_row_sum_error <= 1e-12),
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "maximum_row_sum_error": maximum_row_sum_error,
    }
    return objectives, kkt_residuals, validity


def _worker_run(library_path: Path, case_name: str, seed: int,
                nlambda: int, lambda_min_ratio: float, max_ite: int,
                precision: float, path_early_stop: bool) -> dict:
    spec = CASES[case_name]
    x, y_integer = _generate_case(spec, seed)
    lambdas = _lambda_path(x, y_integer, spec.k, nlambda,
                           lambda_min_ratio)

    # The ABI represents labels as doubles and consumes Python X row-major.
    y = np.ascontiguousarray(y_integer, dtype=np.float64)
    x_flat = np.ascontiguousarray(x.reshape(-1), dtype=np.float64)
    beta_flat = np.zeros(nlambda * spec.k * spec.d, dtype=np.float64)
    intercept_flat = np.zeros(nlambda * spec.k, dtype=np.float64)
    iterations = np.zeros(nlambda, dtype=np.int32)
    size_active = np.zeros(nlambda, dtype=np.int32)
    solver_time = np.zeros(nlambda, dtype=np.float64)
    num_fit = np.zeros(1, dtype=np.int32)
    failed_lambda = np.full(1, -1, dtype=np.int32)
    failed_stage = np.full(1, -1, dtype=np.int32)
    outer_iterations = np.zeros(nlambda, dtype=np.int32)
    inner_sweeps = np.zeros(nlambda, dtype=np.int64)
    coordinate_updates = np.zeros(nlambda, dtype=np.int64)
    native_objective = np.zeros(nlambda, dtype=np.float64)
    native_kkt = np.zeros(nlambda, dtype=np.float64)
    native_stationarity = np.zeros(nlambda, dtype=np.float64)

    library, solve, abi_version = _register_solver(library_path)
    # Keep the CDLL object alive until after the native call.
    rss_before = _rss_bytes(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    start_ns = time.perf_counter_ns()
    base_arguments = (
        y, x_flat, spec.n, spec.d, spec.k, lambdas, nlambda,
        3.0, max_ite, precision, 1, True, -1,
        beta_flat, intercept_flat, iterations, size_active, solver_time,
        num_fit, True,
    )
    if abi_version == 4:
        status = solve(
            *base_arguments, 3, path_early_stop, failed_lambda,
            failed_stage, outer_iterations, inner_sweeps,
            coordinate_updates, native_objective, native_kkt,
            native_stationarity,
        )
        if status not in (0, 10):
            raise RuntimeError(
                "V4 solver failed with status="
                f"{status}, lambda={failed_lambda[0]}, "
                f"stage={failed_stage[0]}"
            )
    else:
        solve(*base_arguments)
    del library
    wall_time_seconds = (time.perf_counter_ns() - start_ns) / 1e9
    peak_rss_bytes = _rss_bytes(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )

    actual_fit = int(num_fit[0])
    if actual_fit < 1 or actual_fit > nlambda:
        raise RuntimeError(
            f"solver returned invalid num_fit={actual_fit}; expected 1..{nlambda}"
        )
    beta = beta_flat[:actual_fit * spec.k * spec.d].reshape(
        actual_fit, spec.k, spec.d
    )
    intercept = intercept_flat[:actual_fit * spec.k].reshape(
        actual_fit, spec.k
    )
    objectives, kkt_residuals, probability_validity = _evaluate_path(
        x, y_integer, lambdas[:actual_fit], beta, intercept
    )

    output_digest = hashlib.sha256()
    for output in (
        beta_flat, intercept_flat, iterations, size_active, num_fit,
    ):
        output_digest.update(output.tobytes())

    output_finite = bool(
        np.all(np.isfinite(beta)) and
        np.all(np.isfinite(intercept)) and
        np.all(np.isfinite(objectives))
    )
    return {
        "case": asdict(spec),
        "lambda": lambdas[:actual_fit].tolist(),
        "objective": objectives,
        "kkt_max": kkt_residuals,
        "probability_validity": probability_validity,
        "output_finite": output_finite,
        "wall_time_seconds": wall_time_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_increment_bytes": max(0, peak_rss_bytes - rss_before),
        "rss_before_call_bytes": rss_before,
        "iterations": iterations[:actual_fit].astype(int).tolist(),
        "size_act": size_active[:actual_fit].astype(int).tolist(),
        "solver_reported_time_seconds": solver_time[:actual_fit].tolist(),
        "num_fit": actual_fit,
        "requested_nlambda": nlambda,
        "path_early_stop_requested": path_early_stop,
        "path_early_stopped": bool(actual_fit < nlambda),
        "native_abi_version": abi_version,
        "output_sha256": output_digest.hexdigest(),
        "coefficient_nonzeros": [
            int(np.count_nonzero(np.abs(beta[index]) > 1e-8))
            for index in range(actual_fit)
        ],
    }


def _parse_library(specification: str) -> Tuple[str, Path]:
    if "=" in specification:
        name, path_text = specification.split("=", 1)
        if not name:
            raise ValueError("library label before '=' may not be empty")
    else:
        path_text = specification
        name = Path(path_text).stem
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"library does not exist: {path}")
    return name, path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def _summarize_runs(runs: Sequence[dict]) -> dict:
    wall = [run["wall_time_seconds"] for run in runs]
    rss = [run["peak_rss_bytes"] for run in runs]
    rss_increment = [run["peak_rss_increment_bytes"] for run in runs]
    num_fit = [run["num_fit"] for run in runs]
    objective_lengths = {len(run["objective"]) for run in runs}
    output_hashes = sorted({run["output_sha256"] for run in runs})

    objective_spread = None
    objective_median = None
    if len(objective_lengths) == 1:
        objective_array = np.asarray(
            [run["objective"] for run in runs], dtype=np.float64
        )
        objective_median = np.median(objective_array, axis=0).tolist()
        objective_spread = float(np.max(
            np.max(objective_array, axis=0) - np.min(objective_array, axis=0)
        ))

    return {
        "wall_time_seconds": {
            "median": _median(wall),
            "minimum": min(wall),
            "maximum": max(wall),
        },
        "peak_rss_bytes": {
            "median": _median(rss),
            "maximum": max(rss),
        },
        "peak_rss_increment_bytes": {
            "median": _median(rss_increment),
            "maximum": max(rss_increment),
        },
        "num_fit": {
            "minimum": min(num_fit),
            "maximum": max(num_fit),
        },
        "all_outputs_finite": all(run["output_finite"] for run in runs),
        "all_probabilities_valid": all(
            run["probability_validity"]["all_valid"] for run in runs
        ),
        "objective_median": objective_median,
        "objective_max_repeat_spread": objective_spread,
        "kkt_max": max(max(run["kkt_max"]) for run in runs),
        "output_sha256": output_hashes,
        "outputs_byte_stable": len(output_hashes) == 1,
    }


def _worker_command(script: Path, library_path: Path, case_name: str,
                    args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable, str(script), "--_worker",
        "--worker-library", str(library_path),
        "--worker-case", case_name,
        "--seed", str(args.seed),
        "--nlambda", str(args.nlambda),
        "--lambda-min-ratio", repr(args.lambda_min_ratio),
        "--max-ite", str(args.max_ite),
        "--precision", repr(args.precision),
    ]
    if args.path_early_stop:
        command.append("--path-early-stop")
    return command


def _invoke_worker(script: Path, library_path: Path, case_name: str,
                   args: argparse.Namespace) -> dict:
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    completed = subprocess.run(
        _worker_command(script, library_path, case_name, args),
        check=False, capture_output=True, text=True, env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {library_path} / {case_name} "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(WORKER_MARKER):
            return json.loads(line[len(WORKER_MARKER):])
    raise RuntimeError(
        f"worker returned no result for {library_path} / {case_name}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _build_comparisons(libraries: Sequence[dict], case_names: Sequence[str]) -> List[dict]:
    if len(libraries) < 2:
        return []
    baseline = libraries[0]
    comparisons: List[dict] = []
    for candidate in libraries[1:]:
        by_case = {}
        for case_name in case_names:
            base_summary = baseline["cases"][case_name]["summary"]
            new_summary = candidate["cases"][case_name]["summary"]
            base_time = base_summary["wall_time_seconds"]["median"]
            new_time = new_summary["wall_time_seconds"]["median"]
            base_rss = base_summary["peak_rss_bytes"]["median"]
            new_rss = new_summary["peak_rss_bytes"]["median"]

            objective_difference = None
            base_objective = base_summary["objective_median"]
            new_objective = new_summary["objective_median"]
            if (base_objective is not None and new_objective is not None and
                    len(base_objective) == len(new_objective)):
                objective_difference = float(np.max(np.abs(
                    np.asarray(base_objective) - np.asarray(new_objective)
                )))

            by_case[case_name] = {
                "wall_time_speedup": (base_time / new_time
                                      if new_time > 0.0 else None),
                "peak_rss_ratio_candidate_over_baseline": (
                    new_rss / base_rss if base_rss > 0.0 else None
                ),
                "objective_max_abs_difference": objective_difference,
            }
        comparisons.append({
            "baseline": baseline["name"],
            "candidate": candidate["name"],
            "cases": by_case,
        })
    return comparisons


def _parse_cases(value: str) -> List[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown cases: " + ", ".join(unknown) +
            "; choices are " + ", ".join(CASES)
        )
    if not names:
        raise argparse.ArgumentTypeError("at least one case is required")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("case names must not be repeated")
    return names


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one or more PICASSO multinomial shared libraries. "
            "The first --library is the comparison baseline."
        )
    )
    parser.add_argument(
        "--library", action="append", default=[], metavar="[LABEL=]PATH",
        help=("shared library to benchmark; repeat for legacy/new comparison "
              "(default: lib/libpicasso.so)"),
    )
    parser.add_argument(
        "--cases", type=_parse_cases, default=list(CASES),
        help="comma-separated cases (default: tiny,p_gt_n,high_k,high_n)",
    )
    parser.add_argument("--repeats", type=int, default=3,
                        help="measured subprocesses per library/case (default: 3)")
    parser.add_argument("--warmups", type=int, default=1,
                        help="discarded subprocesses per library/case (default: 1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nlambda", type=int, default=8)
    parser.add_argument("--lambda-min-ratio", type=float, default=0.1)
    parser.add_argument("--max-ite", type=int, default=1000)
    parser.add_argument("--precision", type=float, default=1e-7)
    parser.add_argument(
        "--path-early-stop", action="store_true",
        help=("enable the V4 glmnet-style tail stop when the library "
              "exports it (default: disabled)"),
    )
    parser.add_argument("--output", default="-", metavar="PATH",
                        help="JSON destination, or '-' for stdout (default: '-')")

    # Private subprocess protocol.  These options intentionally do not appear
    # in normal help output.
    parser.add_argument("--_worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-library", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", choices=list(CASES),
                        help=argparse.SUPPRESS)
    return parser


def _validate_arguments(parser: argparse.ArgumentParser,
                        args: argparse.Namespace) -> None:
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.nlambda < 1:
        parser.error("--nlambda must be at least 1")
    if not 0.0 < args.lambda_min_ratio <= 1.0:
        parser.error("--lambda-min-ratio must be in (0, 1]")
    if args.nlambda > 1 and args.lambda_min_ratio >= 1.0:
        parser.error(
            "--lambda-min-ratio must be smaller than 1 when nlambda > 1"
        )
    if args.max_ite < 1:
        parser.error("--max-ite must be at least 1")
    if not math.isfinite(args.precision) or args.precision <= 0.0:
        parser.error("--precision must be a positive finite number")


def _run_controller(args: argparse.Namespace) -> dict:
    repository_root = Path(__file__).resolve().parents[1]
    specifications = args.library or [
        "current=" + str(repository_root / "lib" / "libpicasso.so")
    ]
    parsed_libraries = [_parse_library(item) for item in specifications]
    names = [name for name, _ in parsed_libraries]
    if len(names) != len(set(names)):
        raise ValueError("--library labels must be unique")

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": (
            _datetime.datetime.now(_datetime.timezone.utc).isoformat()
        ),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "config": {
            "seed": args.seed,
            "cases": args.cases,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "nlambda": args.nlambda,
            "lambda_min_ratio": args.lambda_min_ratio,
            "max_ite": args.max_ite,
            "precision": args.precision,
            "path_early_stop": args.path_early_stop,
            "penalty": "l1",
            "intercept": True,
            "dfmax": -1,
            "threads": 1,
        },
        "libraries": [],
    }

    script = Path(__file__).resolve()
    library_results = []
    for name, library_path in parsed_libraries:
        library_results.append({
            "name": name,
            "path": str(library_path),
            "sha256": _file_sha256(library_path),
            "size_bytes": library_path.stat().st_size,
            "cases": {},
        })

    # Alternate A->B and B->A orders by block.  For two libraries this yields
    # an ABBA sequence across each pair of blocks and limits thermal/order bias.
    for case_name in args.cases:
        runs_by_library: List[List[dict]] = [
            [] for _ in parsed_libraries
        ]
        for warmup_index in range(args.warmups):
            order = list(range(len(parsed_libraries)))
            if warmup_index % 2:
                order.reverse()
            for library_index in order:
                _invoke_worker(
                    script, parsed_libraries[library_index][1], case_name, args
                )
        for repeat_index in range(args.repeats):
            order = list(range(len(parsed_libraries)))
            if repeat_index % 2:
                order.reverse()
            for library_index in order:
                runs_by_library[library_index].append(_invoke_worker(
                    script, parsed_libraries[library_index][1], case_name, args
                ))

        for library_index, runs in enumerate(runs_by_library):
            library_results[library_index]["cases"][case_name] = {
                "case": runs[0]["case"],
                "runs": runs,
                "summary": _summarize_runs(runs),
            }

    result["libraries"] = library_results

    result["comparisons"] = _build_comparisons(
        result["libraries"], args.cases
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    if args._worker:
        if not args.worker_library or not args.worker_case:
            parser.error("worker mode requires a library and case")
        worker_result = _worker_run(
            Path(args.worker_library).resolve(), args.worker_case, args.seed,
            args.nlambda, args.lambda_min_ratio, args.max_ite, args.precision,
            args.path_early_stop,
        )
        print(WORKER_MARKER + json.dumps(worker_result, sort_keys=True))
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
