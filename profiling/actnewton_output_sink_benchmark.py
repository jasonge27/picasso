#!/usr/bin/env python3
"""Validate and benchmark the scalar ActNewton C-API output sink.

The correctness oracle covers logistic, Poisson, and square-root loss across
all penalties, with and without an intercept.  The memory cases use a zero
coefficient logistic path so runtime and peak-RSS differences isolate path
retention rather than optimizer behavior.  Every library runs in a fresh
process; output allocation and deterministic data generation occur before the
peak-RSS baseline is sampled.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MARKER = "PICASSO_ACTNEWTON_OUTPUT_SINK="
MEMORY_CASES = {
    "path_32mb": {"n": 4, "d": 40_000, "nlambda": 100, "seed": 20260721},
    "path_80mb": {"n": 4, "d": 100_000, "nlambda": 100, "seed": 20260722},
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compiler_version() -> str:
    completed = subprocess.run(
        ["c++", "--version"], check=True, capture_output=True, text=True)
    return completed.stdout.splitlines()[0]


def cpu_description() -> str:
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False, capture_output=True, text=True)
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    return platform.processor() or platform.machine()


def checksum_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def configure_function(library: ctypes.CDLL, family: str):
    symbol = {
        "logistic": "SolveLogisticRegressionV3",
        "poisson": "SolvePoissonRegressionV3",
        "sqrtlasso": "SolveSqrtLinearRegressionV3",
    }[family]
    function = getattr(library, symbol)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    common = [
        double_pointer, double_pointer, ctypes.c_int, ctypes.c_int,
        double_pointer, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
    ]
    if family != "sqrtlasso":
        common.append(double_pointer)
    function.argtypes = common + [
        double_pointer, double_pointer, int_pointer, int_pointer,
        double_pointer, int_pointer, ctypes.c_bool, ctypes.c_int,
        int_pointer, int_pointer, int_pointer, double_pointer,
        double_pointer, double_pointer, double_pointer,
    ]
    function.restype = ctypes.c_int
    return function


def solve_case(library: ctypes.CDLL, family: str, x: np.ndarray,
               y: np.ndarray, lambdas: np.ndarray, penalty: int,
               intercept: bool, offset: np.ndarray | None,
               max_stages: int = 5, precision: float = 1.0e-4) -> dict:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    lambdas = np.ascontiguousarray(lambdas, dtype=np.float64)
    if offset is not None:
        offset = np.ascontiguousarray(offset, dtype=np.float64)
    n, d = x.shape
    nlambda = lambdas.size
    beta = np.full((nlambda, d), np.nan, dtype=np.float64)
    intcpt = np.full(nlambda, np.nan, dtype=np.float64)
    iterations = np.full(nlambda, -1, dtype=np.int32)
    active_size = np.full(nlambda, -1, dtype=np.int32)
    runtime = np.full(nlambda, np.nan, dtype=np.float64)
    stages = np.full(nlambda, -1, dtype=np.int32)
    objective = np.full(nlambda, np.nan, dtype=np.float64)
    kkt = np.full(nlambda, np.nan, dtype=np.float64)
    stationarity = np.full(nlambda, np.nan, dtype=np.float64)
    smooth = np.full(nlambda, np.nan, dtype=np.float64)
    num_fit = ctypes.c_int(-1)
    failed_lambda = ctypes.c_int(-2)
    failed_stage = ctypes.c_int(-2)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    arguments = [
        y.ctypes.data_as(double_pointer),
        x.ctypes.data_as(double_pointer), n, d,
        lambdas.ctypes.data_as(double_pointer), nlambda,
        3.7 if penalty == 3 else 3.0, 1000, precision, penalty,
        intercept, -1,
    ]
    if family != "sqrtlasso":
        arguments.append(
            None if offset is None else offset.ctypes.data_as(double_pointer))
    arguments.extend([
        beta.ctypes.data_as(double_pointer),
        intcpt.ctypes.data_as(double_pointer),
        iterations.ctypes.data_as(int_pointer),
        active_size.ctypes.data_as(int_pointer),
        runtime.ctypes.data_as(double_pointer), ctypes.byref(num_fit), True,
        max_stages, ctypes.byref(failed_lambda), ctypes.byref(failed_stage),
        stages.ctypes.data_as(int_pointer),
        objective.ctypes.data_as(double_pointer),
        kkt.ctypes.data_as(double_pointer),
        stationarity.ctypes.data_as(double_pointer),
        smooth.ctypes.data_as(double_pointer),
    ])
    status = int(configure_function(library, family)(*arguments))
    fitted = num_fit.value
    if not 0 <= fitted <= nlambda:
        raise RuntimeError(f"invalid fitted prefix: {fitted}")
    if fitted and not np.all(np.isfinite(beta[:fitted])):
        raise RuntimeError("committed coefficient prefix is not finite")
    return {
        "status": status,
        "num_fit": fitted,
        "failed_lambda": failed_lambda.value,
        "failed_stage": failed_stage.value,
        "iteration_sum": int(iterations[:fitted].sum()),
        "checksum": checksum_arrays(
            beta, intcpt, iterations, active_size, runtime, stages,
            objective, kkt, stationarity, smooth,
        ),
    }


def correctness_oracle(library_path: str) -> dict:
    library = ctypes.CDLL(str(Path(library_path).resolve()))
    rng = np.random.default_rng(20260723)
    x = rng.normal(size=(36, 9))
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    signal = 0.8 * x[:, 1] - 0.55 * x[:, 4] + 0.35 * x[:, 7]
    offset = np.linspace(-0.2, 0.25, x.shape[0])
    responses = {
        "logistic": (signal + offset > np.median(signal + offset)).astype(float),
        "poisson": np.asarray(rng.poisson(np.exp(np.clip(
            0.15 + 0.22 * signal + offset, -1.0, 1.0))), dtype=float),
        "sqrtlasso": 0.25 + signal + rng.normal(scale=0.17, size=x.shape[0]),
    }
    lambdas = np.asarray([0.7, 0.32, 0.14, 0.06], dtype=float)
    results = {}
    for family, response in responses.items():
        for penalty in (1, 2, 3):
            for intercept in (False, True):
                key = f"{family}:penalty{penalty}:intercept{int(intercept)}"
                result = solve_case(
                    library, family, x, response, lambdas, penalty,
                    intercept, offset if family != "sqrtlasso" else None)
                if (result["num_fit"] != lambdas.size
                        or result["status"] not in (0, 10)):
                    raise RuntimeError(
                        f"{key}: oracle path is not fully usable: {result}")
                results[key] = result
    return results


def memory_worker(library_path: str, case_name: str) -> dict:
    config = MEMORY_CASES[case_name]
    rng = np.random.default_rng(config["seed"])
    x = np.empty((config["n"], config["d"]), dtype=np.float64, order="C")
    rng.standard_normal(x.shape, out=x)
    y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    lambdas = np.geomspace(1.0e6, 1.0e3, config["nlambda"])
    library = ctypes.CDLL(str(Path(library_path).resolve()))

    # Allocate and touch all caller-owned outputs before the RSS baseline.
    nlambda, d = lambdas.size, x.shape[1]
    beta = np.full((nlambda, d), np.nan, dtype=np.float64)
    intcpt = np.full(nlambda, np.nan, dtype=np.float64)
    iterations = np.full(nlambda, -1, dtype=np.int32)
    active_size = np.full(nlambda, -1, dtype=np.int32)
    runtime = np.full(nlambda, np.nan, dtype=np.float64)
    stages = np.full(nlambda, -1, dtype=np.int32)
    objective = np.full(nlambda, np.nan, dtype=np.float64)
    kkt = np.full(nlambda, np.nan, dtype=np.float64)
    stationarity = np.full(nlambda, np.nan, dtype=np.float64)
    smooth = np.full(nlambda, np.nan, dtype=np.float64)
    num_fit = ctypes.c_int(-1)
    failed_lambda = ctypes.c_int(-2)
    failed_stage = ctypes.c_int(-2)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    function = configure_function(library, "logistic")

    rss_before = peak_rss_bytes()
    started = time.perf_counter()
    status = int(function(
        y.ctypes.data_as(double_pointer), x.ctypes.data_as(double_pointer),
        x.shape[0], d, lambdas.ctypes.data_as(double_pointer), nlambda,
        3.0, 100, 1.0e-8, 1, True, -1, None,
        beta.ctypes.data_as(double_pointer),
        intcpt.ctypes.data_as(double_pointer),
        iterations.ctypes.data_as(int_pointer),
        active_size.ctypes.data_as(int_pointer),
        runtime.ctypes.data_as(double_pointer), ctypes.byref(num_fit), True, 3,
        ctypes.byref(failed_lambda), ctypes.byref(failed_stage),
        stages.ctypes.data_as(int_pointer),
        objective.ctypes.data_as(double_pointer),
        kkt.ctypes.data_as(double_pointer),
        stationarity.ctypes.data_as(double_pointer),
        smooth.ctypes.data_as(double_pointer)))
    elapsed = time.perf_counter() - started
    rss_after = peak_rss_bytes()
    if status != 0 or num_fit.value != nlambda:
        raise RuntimeError(
            f"zero path failed: status={status}, fitted={num_fit.value}")
    if not np.all(beta == 0.0) or np.any(active_size != 0):
        raise RuntimeError("high-lambda fixture no longer has a zero path")
    return {
        "seconds": elapsed,
        "peak_delta_bytes": rss_after - rss_before,
        "num_fit": num_fit.value,
        "iteration_sum": int(iterations.sum()),
        "checksum": checksum_arrays(
            beta, intcpt, iterations, active_size, runtime, stages,
            objective, kkt, stationarity, smooth),
    }


def run_worker(program: Path, library: Path, mode: str,
               case_name: str | None = None) -> dict:
    command = [sys.executable, str(program), "--worker", mode,
               "--library", str(library)]
    if case_name is not None:
        command.extend(["--case", case_name])
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in THREAD_VARIABLES:
        environment[name] = "1"
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True,
        env=environment)
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError("worker returned no benchmark marker")


def orchestrate(arguments) -> None:
    program = Path(__file__).resolve()
    baseline = Path(arguments.baseline).resolve()
    candidate = Path(arguments.candidate).resolve()
    baseline_oracle = run_worker(program, baseline, "oracle")
    candidate_oracle = run_worker(program, candidate, "oracle")
    if baseline_oracle != candidate_oracle:
        mismatches = sorted(
            key for key in baseline_oracle
            if baseline_oracle.get(key) != candidate_oracle.get(key))
        raise RuntimeError(f"correctness oracle differs: {mismatches}")

    report = {
        "metadata": {
            "script_sha256": sha256_file(program),
            "baseline_library": str(baseline),
            "baseline_sha256": sha256_file(baseline),
            "candidate_library": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "python": sys.version, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": cpu_description(), "compiler": compiler_version(),
            "rss_units": "bytes",
            "thread_environment": {
                name: "1" for name in THREAD_VARIABLES
            },
        },
        "oracle_configurations": len(baseline_oracle),
        "oracle": baseline_oracle,
        "repeats": arguments.repeats,
        "cases": {},
    }
    for case_name, configuration in MEMORY_CASES.items():
        observations = {"baseline": [], "candidate": []}
        for repeat in range(arguments.repeats):
            order = ("baseline", "candidate") if repeat % 2 == 0 else (
                "candidate", "baseline")
            for mode in order:
                library = baseline if mode == "baseline" else candidate
                observations[mode].append(
                    run_worker(program, library, "memory", case_name))
        baseline_checksums = {x["checksum"] for x in observations["baseline"]}
        candidate_checksums = {x["checksum"] for x in observations["candidate"]}
        if baseline_checksums != candidate_checksums or len(baseline_checksums) != 1:
            raise RuntimeError(f"{case_name}: outputs differ")
        baseline_seconds = [x["seconds"] for x in observations["baseline"]]
        candidate_seconds = [x["seconds"] for x in observations["candidate"]]
        baseline_rss = [x["peak_delta_bytes"] for x in observations["baseline"]]
        candidate_rss = [x["peak_delta_bytes"] for x in observations["candidate"]]
        report["cases"][case_name] = {
            "configuration": configuration,
            "coefficient_path_bytes": configuration["d"] * configuration["nlambda"] * 8,
            "checksum": next(iter(baseline_checksums)),
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "baseline_peak_delta_bytes": baseline_rss,
            "candidate_peak_delta_bytes": candidate_rss,
            "baseline_median_seconds": statistics.median(baseline_seconds),
            "candidate_median_seconds": statistics.median(candidate_seconds),
            "speedup": statistics.median(baseline_seconds) / statistics.median(candidate_seconds),
            "baseline_median_peak_delta_bytes": statistics.median(baseline_rss),
            "candidate_median_peak_delta_bytes": statistics.median(candidate_rss),
            "median_peak_delta_reduction_bytes": statistics.median(baseline_rss) - statistics.median(candidate_rss),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if arguments.output:
        Path(arguments.output).write_text(rendered, encoding="utf-8")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output")
    parser.add_argument("--worker", choices=("oracle", "memory"))
    parser.add_argument("--library")
    parser.add_argument("--case", choices=tuple(MEMORY_CASES))
    arguments = parser.parse_args()
    if arguments.worker and not arguments.library:
        parser.error("worker mode requires --library")
    if arguments.worker == "memory" and not arguments.case:
        parser.error("memory worker requires --case")
    if not arguments.worker and (
            not arguments.baseline or not arguments.candidate
            or arguments.repeats < 3):
        parser.error("comparison requires both libraries and at least 3 repeats")
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    if args.worker == "oracle":
        output = correctness_oracle(args.library)
    elif args.worker == "memory":
        output = memory_worker(args.library, args.case)
    else:
        orchestrate(args)
        sys.exit(0)
    print(MARKER + json.dumps(output, sort_keys=True))
