#!/usr/bin/env python3
"""Benchmark lazy versus eager scalar-GLM weighted-column caches.

Every timed solve runs in a fresh subprocess.  This prevents two dylibs with
the same exported C symbols from being coalesced and gives an independent peak
RSS measurement for every repetition.  Data generation and dylib loading are
outside the reported solver wall time.

Example:
  python3 profiling/glm_weight_cache_benchmark.py \
    --eager /private/tmp/glm-cache/libpicasso_eager.dylib \
    --lazy /private/tmp/glm-cache/libpicasso_lazy.dylib \
    --repeats 7 --output /private/tmp/glm-cache/results.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


MARKER = "PICASSO_GLM_CACHE_BENCHMARK="
CASES = {
    "binomial_sparse": dict(
        family="binomial", n=900, d=3000, support=12, nlambda=28,
        lambda_ratio=0.24, seed=20260721,
    ),
    "poisson_sparse": dict(
        family="poisson", n=900, d=3000, support=12, nlambda=28,
        lambda_ratio=0.24, seed=20260722,
    ),
    "binomial_wide": dict(
        family="binomial", n=320, d=12000, support=8, nlambda=20,
        lambda_ratio=0.30, seed=20260723,
    ),
    "poisson_wide": dict(
        family="poisson", n=320, d=12000, support=8, nlambda=20,
        lambda_ratio=0.30, seed=20260724,
    ),
}


def _logistic(value):
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -500.0, 500.0)))


def _logistic_null_intercept(y, offset):
    target = float(np.mean(y))
    lower = -float(np.max(offset)) - 40.0
    upper = -float(np.min(offset)) + 40.0
    for _ in range(80):
        midpoint = lower + 0.5 * (upper - lower)
        if float(np.mean(_logistic(offset + midpoint))) < target:
            lower = midpoint
        else:
            upper = midpoint
    return lower + 0.5 * (upper - lower)


def make_problem(case_name):
    config = CASES[case_name]
    rng = np.random.RandomState(config["seed"])
    n, d = config["n"], config["d"]
    x = rng.standard_normal((n, d))
    x -= x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(x * x, axis=0, keepdims=True) / max(n - 1, 1))
    x /= np.where(scale > 0.0, scale, 1.0)

    beta = np.zeros(d)
    if config["family"] == "binomial":
        magnitudes = np.linspace(0.85, 0.30, config["support"])
        offset = 0.20 * np.sin(np.arange(n) * 0.17)
        intercept = -0.15
        beta[:config["support"]] = magnitudes * np.where(
            np.arange(config["support"]) % 2 == 0, 1.0, -1.0
        )
        probability = _logistic(intercept + offset + x @ beta)
        y = rng.binomial(1, probability).astype(np.float64)
        null_intercept = _logistic_null_intercept(y, offset)
        fitted_null = _logistic(offset + null_intercept)
    else:
        magnitudes = np.linspace(0.24, 0.09, config["support"])
        offset = 0.18 * np.cos(np.arange(n) * 0.11)
        intercept = math.log(1.6)
        beta[:config["support"]] = magnitudes * np.where(
            np.arange(config["support"]) % 2 == 0, 1.0, -1.0
        )
        mean = np.exp(intercept + offset + x @ beta)
        y = rng.poisson(mean).astype(np.float64)
        maximum = float(np.max(offset))
        log_mean_exp = maximum + math.log(
            float(np.mean(np.exp(offset - maximum)))
        )
        null_intercept = math.log(float(np.mean(y))) - log_mean_exp
        fitted_null = np.exp(offset + null_intercept)

    lambda_max = float(np.max(np.abs(x.T @ (y - fitted_null))) / n)
    lambdas = np.geomspace(
        lambda_max, lambda_max * config["lambda_ratio"], config["nlambda"]
    )
    return (
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(offset, dtype=np.float64),
        np.ascontiguousarray(lambdas, dtype=np.float64),
    )


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _v2_function(library, family):
    name = ("SolveLogisticRegressionV2" if family == "binomial"
            else "SolvePoissonRegressionV2")
    function = getattr(library, name)
    double_array = ndpointer(np.float64, flags="C_CONTIGUOUS")
    int_array = ndpointer(np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [
        double_array, double_array, ctypes.c_int, ctypes.c_int,
        double_array, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        double_array, double_array, double_array, int_array, int_array,
        double_array, int_array, ctypes.c_bool, ctypes.c_int, int_array,
        int_array, int_array, double_array, double_array, double_array,
    ]
    function.restype = ctypes.c_int
    return function


def solve(library_path, case_name):
    config = CASES[case_name]
    x, y, offset, lambdas = make_problem(case_name)
    n, d = x.shape
    nlambda = len(lambdas)
    library = ctypes.CDLL(str(Path(library_path).resolve()))
    function = _v2_function(library, config["family"])

    beta = np.zeros((nlambda, d), dtype=np.float64)
    intercept = np.zeros(nlambda, dtype=np.float64)
    iterations = np.zeros(nlambda, dtype=np.int32)
    active_size = np.zeros(nlambda, dtype=np.int32)
    native_runtime = np.zeros(nlambda, dtype=np.float64)
    num_fit = np.zeros(1, dtype=np.int32)
    failed_lambda = np.full(1, -1, dtype=np.int32)
    failed_stage = np.full(1, -1, dtype=np.int32)
    lla_stages = np.zeros(nlambda, dtype=np.int32)
    objective = np.full(nlambda, np.nan, dtype=np.float64)
    kkt = np.full(nlambda, np.nan, dtype=np.float64)
    stationarity = np.full(nlambda, np.nan, dtype=np.float64)

    pre_solve_peak = _peak_rss_bytes()
    start = time.perf_counter()
    status = int(function(
        y, x, n, d, lambdas, nlambda, 3.0, 500, 1e-6, 2, True, -1,
        offset, beta, intercept, iterations, active_size, native_runtime,
        num_fit, True, 3, failed_lambda, failed_stage, lla_stages,
        objective, kkt, stationarity,
    ))
    wall_seconds = time.perf_counter() - start
    peak_rss = _peak_rss_bytes()
    fitted = int(num_fit[0])
    if fitted <= 0 or fitted > nlambda:
        raise RuntimeError(
            f"{case_name}: invalid fitted path length {fitted}, status={status}"
        )
    return {
        "wall_seconds": wall_seconds,
        "pre_solve_peak_rss_bytes": pre_solve_peak,
        "peak_rss_bytes": peak_rss,
        "solve_peak_increment_bytes": max(0, peak_rss - pre_solve_peak),
        "status": status,
        "num_fit": fitted,
        "failed_lambda": int(failed_lambda[0]),
        "failed_stage": int(failed_stage[0]),
        "iteration_sum": int(iterations[:fitted].sum()),
        "maximum_active_size": int(active_size[:fitted].max()),
        "beta": beta[:fitted].copy(),
        "intercept": intercept[:fitted].copy(),
        "iterations": iterations[:fitted].copy(),
        "active_size": active_size[:fitted].copy(),
        "lla_stages": lla_stages[:fitted].copy(),
        "objective": objective[:fitted].copy(),
        "kkt": kkt[:fitted].copy(),
        "stationarity": stationarity[:fitted].copy(),
        "lambdas": lambdas[:fitted].copy(),
    }


def worker(args):
    result = solve(args.library, args.case)
    if args.output:
        np.savez(
            args.output,
            beta=result["beta"], intercept=result["intercept"],
            iterations=result["iterations"], active_size=result["active_size"],
            lla_stages=result["lla_stages"], objective=result["objective"],
            kkt=result["kkt"], stationarity=result["stationarity"],
            lambdas=result["lambdas"],
            status=np.asarray([result["status"]], dtype=np.int32),
            num_fit=np.asarray([result["num_fit"]], dtype=np.int32),
            failed_lambda=np.asarray([result["failed_lambda"]], dtype=np.int32),
            failed_stage=np.asarray([result["failed_stage"]], dtype=np.int32),
        )
    public = {
        key: value for key, value in result.items()
        if not isinstance(value, np.ndarray)
    }
    print(MARKER + json.dumps(public, sort_keys=True))


def run_worker(program, library, case_name, output=None):
    command = [
        sys.executable, str(program), "--worker", "--library", library,
        "--case", case_name,
    ]
    if output is not None:
        command.extend(["--output", str(output)])
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment
    )
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError(
        "benchmark worker returned no marker\n" + completed.stdout
        + completed.stderr
    )


def _max_abs_difference(left, right):
    finite = np.isfinite(left) & np.isfinite(right)
    if np.any(np.isfinite(left) != np.isfinite(right)):
        return math.inf
    return float(np.max(np.abs(left[finite] - right[finite]))) \
        if np.any(finite) else 0.0


def compare_outputs(eager_path, lazy_path):
    integer_fields = (
        "status", "num_fit", "failed_lambda", "failed_stage", "iterations",
        "active_size", "lla_stages",
    )
    floating_fields = (
        "beta", "intercept", "objective", "kkt", "stationarity", "lambdas",
    )
    with np.load(eager_path) as eager, np.load(lazy_path) as lazy:
        integer_equal = {
            field: bool(np.array_equal(eager[field], lazy[field]))
            for field in integer_fields
        }
        float_differences = {
            field: _max_abs_difference(eager[field], lazy[field])
            for field in floating_fields
        }
        exact_float_equal = {
            field: bool(np.array_equal(
                eager[field], lazy[field], equal_nan=True
            ))
            for field in floating_fields
        }
    return {
        "all_integer_metadata_equal": all(integer_equal.values()),
        "integer_fields_equal": integer_equal,
        "all_floating_outputs_bitwise_equal": all(exact_float_equal.values()),
        "floating_fields_bitwise_equal": exact_float_equal,
        "maximum_absolute_differences": float_differences,
    }


def _summary(values):
    return {
        "values": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def orchestrate(args):
    program = Path(__file__).resolve()
    report = {
        "repeats": args.repeats,
        "penalty": "MCP",
        "gamma": 3.0,
        "lla_max_stages": 3,
        "precision": 1e-6,
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="picasso-glm-cache-") as directory:
        temporary = Path(directory)
        for case_name in CASES:
            eager_output = temporary / f"{case_name}-eager.npz"
            lazy_output = temporary / f"{case_name}-lazy.npz"
            eager_check = run_worker(
                program, args.eager, case_name, eager_output
            )
            lazy_check = run_worker(
                program, args.lazy, case_name, lazy_output
            )
            correctness = compare_outputs(eager_output, lazy_output)

            measurements = {
                "eager": {"wall": [], "peak_rss": [], "rss_increment": []},
                "lazy": {"wall": [], "peak_rss": [], "rss_increment": []},
            }
            for repeat in range(args.repeats):
                order = ("eager", "lazy") if repeat % 2 == 0 else (
                    "lazy", "eager"
                )
                for implementation in order:
                    library = args.eager if implementation == "eager" else args.lazy
                    result = run_worker(program, library, case_name)
                    measurements[implementation]["wall"].append(
                        result["wall_seconds"]
                    )
                    measurements[implementation]["peak_rss"].append(
                        result["peak_rss_bytes"]
                    )
                    measurements[implementation]["rss_increment"].append(
                        result["solve_peak_increment_bytes"]
                    )

            eager_wall = statistics.median(measurements["eager"]["wall"])
            lazy_wall = statistics.median(measurements["lazy"]["wall"])
            eager_rss = statistics.median(measurements["eager"]["peak_rss"])
            lazy_rss = statistics.median(measurements["lazy"]["peak_rss"])
            report["cases"][case_name] = {
                "configuration": CASES[case_name],
                "status": lazy_check["status"],
                "num_fit": lazy_check["num_fit"],
                "iteration_sum": lazy_check["iteration_sum"],
                "maximum_active_size": lazy_check["maximum_active_size"],
                "correctness": correctness,
                "eager_wall_seconds": _summary(
                    measurements["eager"]["wall"]
                ),
                "lazy_wall_seconds": _summary(
                    measurements["lazy"]["wall"]
                ),
                "eager_peak_rss_bytes": _summary(
                    measurements["eager"]["peak_rss"]
                ),
                "lazy_peak_rss_bytes": _summary(
                    measurements["lazy"]["peak_rss"]
                ),
                "eager_solve_rss_increment_bytes": _summary(
                    measurements["eager"]["rss_increment"]
                ),
                "lazy_solve_rss_increment_bytes": _summary(
                    measurements["lazy"]["rss_increment"]
                ),
                "speedup": eager_wall / lazy_wall,
                "peak_rss_ratio": eager_rss / lazy_rss,
            }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager")
    parser.add_argument("--lazy")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--library")
    parser.add_argument("--case", choices=tuple(CASES))
    args = parser.parse_args()
    if args.worker:
        if not args.library or not args.case:
            parser.error("worker mode requires --library and --case")
    elif not args.eager or not args.lazy or args.repeats < 3:
        parser.error("comparison requires both libraries and at least 3 repeats")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        orchestrate(arguments)
