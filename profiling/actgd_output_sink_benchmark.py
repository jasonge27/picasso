#!/usr/bin/env python3
"""Measure the Gaussian C-API output sink against the retained-path ABI.

Each library is loaded in a fresh process.  Solver time excludes deterministic
data generation and output allocation.  Peak RSS is sampled immediately
before and after the native call, so the reported delta is the incremental
process high-water mark reached while fitting the path.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MARKER = "PICASSO_ACTGD_OUTPUT_SINK="
CASES = {
    "path_32mb": {"n": 8, "d": 40_000, "nlambda": 100, "seed": 20260719},
    "path_80mb": {"n": 8, "d": 100_000, "nlambda": 100, "seed": 20260720},
}


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def make_problem(case_name: str):
    config = CASES[case_name]
    rng = np.random.default_rng(config["seed"])
    # Fill the final Fortran-order buffer directly.  Avoiding a same-sized
    # C-order temporary keeps the pre-call process high-water mark meaningful.
    x = np.empty((config["n"], config["d"]), dtype=np.float64, order="F")
    rng.standard_normal(x.shape, out=x)
    y = np.ascontiguousarray(rng.standard_normal(config["n"]), dtype=np.float64)
    # Every coefficient remains zero.  Because the path is strictly decreasing
    # and the fitted df stays zero, all lambdas are committed without deviance
    # early stopping.  This isolates path retention from optimizer differences.
    lambdas = np.ascontiguousarray(
        np.geomspace(1.0e6, 1.0e3, config["nlambda"]), dtype=np.float64
    )
    return x, y, lambdas


def solve(library_path: str, case_name: str) -> dict:
    x, y, lambdas = make_problem(case_name)
    n, d = x.shape
    nlambda = lambdas.size

    beta = np.full((nlambda, d), np.nan, dtype=np.float64)
    intercept = np.full(nlambda, np.nan, dtype=np.float64)
    iterations = np.full(nlambda, -1, dtype=np.int32)
    active_size = np.full(nlambda, -1, dtype=np.int32)
    native_runtime = np.full(nlambda, np.nan, dtype=np.float64)
    smooth_objective = np.full(nlambda, np.nan, dtype=np.float64)
    num_fit = ctypes.c_int(-1)

    library = ctypes.CDLL(str(Path(library_path).resolve()))
    function = library.SolveLinearRegressionNaiveUpdateV2
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    function.argtypes = [
        double_pointer, double_pointer, ctypes.c_int, ctypes.c_int,
        double_pointer, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        double_pointer, double_pointer, int_pointer, int_pointer,
        double_pointer, int_pointer, ctypes.c_bool, double_pointer,
    ]
    function.restype = None

    rss_before = peak_rss_bytes()
    started = time.perf_counter()
    function(
        y.ctypes.data_as(double_pointer),
        x.ctypes.data_as(double_pointer),
        n,
        d,
        lambdas.ctypes.data_as(double_pointer),
        nlambda,
        3.0,
        100,
        1.0e-8,
        1,
        True,
        -1,
        beta.ctypes.data_as(double_pointer),
        intercept.ctypes.data_as(double_pointer),
        iterations.ctypes.data_as(int_pointer),
        active_size.ctypes.data_as(int_pointer),
        native_runtime.ctypes.data_as(double_pointer),
        ctypes.byref(num_fit),
        False,
        smooth_objective.ctypes.data_as(double_pointer),
    )
    elapsed = time.perf_counter() - started
    rss_after = peak_rss_bytes()

    fitted = num_fit.value
    if fitted != nlambda:
        raise RuntimeError(f"expected {nlambda} fitted models, received {fitted}")
    if not np.all(beta == 0.0) or np.any(active_size != 0):
        raise RuntimeError("high-lambda zero-model fixture changed unexpectedly")
    if rss_after < rss_before:
        raise RuntimeError("peak RSS must be monotone within one process")

    return {
        "seconds": elapsed,
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": rss_after,
        "peak_delta_bytes": rss_after - rss_before,
        "num_fit": fitted,
        "iteration_sum": int(iterations.sum()),
        "checksum": checksum_arrays(
            beta, intercept, iterations, active_size, native_runtime,
            smooth_objective,
        ),
    }


def run_worker(program: Path, library: str, case_name: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(program), "--worker", "--library", library,
         "--case", case_name],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError("worker returned no benchmark marker")


def compiler_version() -> str:
    completed = subprocess.run(
        ["c++", "--version"], check=True, capture_output=True, text=True
    )
    return completed.stdout.splitlines()[0]


def orchestrate(arguments) -> None:
    program = Path(__file__).resolve()
    baseline = Path(arguments.baseline).resolve()
    candidate = Path(arguments.candidate).resolve()
    report = {
        "metadata": {
            "script_sha256": sha256_file(program),
            "baseline_library": str(baseline),
            "baseline_sha256": sha256_file(baseline),
            "candidate_library": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "compiler": compiler_version(),
            "rss_units": "bytes",
        },
        "repeats": arguments.repeats,
        "cases": {},
    }

    for case_name, configuration in CASES.items():
        observations = {"baseline": [], "candidate": []}
        for repeat in range(arguments.repeats):
            order = ("baseline", "candidate") if repeat % 2 == 0 else (
                "candidate", "baseline"
            )
            for mode in order:
                library = baseline if mode == "baseline" else candidate
                observations[mode].append(
                    run_worker(program, str(library), case_name)
                )

        baseline_checksums = {item["checksum"] for item in observations["baseline"]}
        candidate_checksums = {item["checksum"] for item in observations["candidate"]}
        if baseline_checksums != candidate_checksums or len(baseline_checksums) != 1:
            raise RuntimeError(f"{case_name}: baseline and candidate outputs differ")
        for field in ("num_fit", "iteration_sum"):
            values = {
                item[field]
                for mode in observations.values()
                for item in mode
            }
            if len(values) != 1:
                raise RuntimeError(f"{case_name}: {field} differs across runs")

        baseline_seconds = [item["seconds"] for item in observations["baseline"]]
        candidate_seconds = [item["seconds"] for item in observations["candidate"]]
        baseline_delta = [item["peak_delta_bytes"] for item in observations["baseline"]]
        candidate_delta = [item["peak_delta_bytes"] for item in observations["candidate"]]
        report["cases"][case_name] = {
            "configuration": configuration,
            "coefficient_path_bytes": configuration["d"] * configuration["nlambda"] * 8,
            "checksum": next(iter(baseline_checksums)),
            "num_fit": observations["candidate"][0]["num_fit"],
            "iteration_sum": observations["candidate"][0]["iteration_sum"],
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "baseline_peak_delta_bytes": baseline_delta,
            "candidate_peak_delta_bytes": candidate_delta,
            "baseline_median_seconds": statistics.median(baseline_seconds),
            "candidate_median_seconds": statistics.median(candidate_seconds),
            "speedup": statistics.median(baseline_seconds)
            / statistics.median(candidate_seconds),
            "baseline_median_peak_delta_bytes": statistics.median(baseline_delta),
            "candidate_median_peak_delta_bytes": statistics.median(candidate_delta),
            "median_peak_delta_reduction_bytes": statistics.median(baseline_delta)
            - statistics.median(candidate_delta),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if arguments.output:
        Path(arguments.output).write_text(rendered, encoding="utf-8")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--library")
    parser.add_argument("--case", choices=tuple(CASES))
    arguments = parser.parse_args()
    if arguments.worker:
        if not arguments.library or not arguments.case:
            parser.error("worker mode requires --library and --case")
    elif (not arguments.baseline or not arguments.candidate
          or arguments.repeats < 3):
        parser.error("comparison requires both libraries and at least 3 repeats")
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    if args.worker:
        print(MARKER + json.dumps(solve(args.library, args.case), sort_keys=True))
    else:
        orchestrate(args)
