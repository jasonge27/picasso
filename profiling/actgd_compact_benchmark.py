#!/usr/bin/env python3
"""Compare pre-change and compact ActGD libraries on dense Gaussian paths.

Each timed call runs in a fresh subprocess so two libraries exporting the same
C symbols cannot be coalesced by the dynamic loader. Data generation and
library loading are outside the reported solver wall time.

Example:
  python3 profiling/actgd_compact_benchmark.py \
    --baseline /tmp/libpicasso_actgd_before.dylib \
    --candidate /tmp/libpicasso_actgd_compact.dylib --repeats 7
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


MARKER = "PICASSO_ACTGD_BENCHMARK="
CASES = {
    "wide_sparse": dict(n=220, d=8000, support=10, nlambda=45,
                        lambda_ratio=0.18, seed=20260716),
    "moderate_sparse": dict(n=600, d=3500, support=24, nlambda=50,
                            lambda_ratio=0.12, seed=20260717),
    "correlated_screen": dict(n=300, d=6000, support=8, nlambda=55,
                              lambda_ratio=0.25, seed=20260718,
                              groups=40, correlation=0.97),
}


def make_problem(case_name: str):
    config = CASES[case_name]
    rng = np.random.RandomState(config["seed"])
    if "groups" in config:
        latent = rng.standard_normal((config["n"], config["groups"]))
        group_index = np.arange(config["d"]) % config["groups"]
        rho = config["correlation"]
        x = (
            rho * latent[:, group_index]
            + math.sqrt(1.0 - rho * rho)
            * rng.standard_normal((config["n"], config["d"]))
        )
    else:
        x = rng.standard_normal((config["n"], config["d"]))
        rho = 0.18
        x[:, 1:] = (
            rho * x[:, :-1]
            + math.sqrt(1.0 - rho * rho) * x[:, 1:]
        )
    x -= x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    x /= np.where(scale > 0.0, scale, 1.0)

    beta = np.zeros(config["d"])
    magnitude = np.linspace(1.2, 0.35, config["support"])
    beta[:config["support"]] = magnitude * np.where(
        np.arange(config["support"]) % 2 == 0, 1.0, -1.0
    )
    y = 1.4 + x @ beta + rng.standard_normal(config["n"])
    residual = y - y.mean()
    lambda_max = float(np.max(np.abs(x.T @ residual)) / config["n"])
    lambdas = np.geomspace(
        lambda_max, lambda_max * config["lambda_ratio"], config["nlambda"]
    )
    return (
        np.asfortranarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(lambdas, dtype=np.float64),
    )


def solve(library_path: str, case_name: str):
    x, y, lambdas = make_problem(case_name)
    n, d = x.shape
    nlambda = len(lambdas)
    library = ctypes.CDLL(str(Path(library_path).resolve()))
    function = library.SolveLinearRegressionNaiveUpdate
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    function.argtypes = [
        double_pointer, double_pointer, ctypes.c_int, ctypes.c_int,
        double_pointer, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
        double_pointer, double_pointer, int_pointer, int_pointer,
        double_pointer, int_pointer, ctypes.c_bool,
    ]
    function.restype = None

    beta = np.zeros((nlambda, d), dtype=np.float64)
    intercept = np.zeros(nlambda, dtype=np.float64)
    iterations = np.zeros(nlambda, dtype=np.int32)
    size_active = np.zeros(nlambda, dtype=np.int32)
    native_runtime = np.zeros(nlambda, dtype=np.float64)
    num_fit = ctypes.c_int(0)

    start = time.perf_counter()
    function(
        y.ctypes.data_as(double_pointer),
        x.ctypes.data_as(double_pointer),
        n, d,
        lambdas.ctypes.data_as(double_pointer), nlambda,
        3.0, 5000, 1e-10, 1, True, -1,
        beta.ctypes.data_as(double_pointer),
        intercept.ctypes.data_as(double_pointer),
        iterations.ctypes.data_as(int_pointer),
        size_active.ctypes.data_as(int_pointer),
        native_runtime.ctypes.data_as(double_pointer),
        ctypes.byref(num_fit), False,
    )
    wall_seconds = time.perf_counter() - start
    fitted = num_fit.value
    if fitted <= 0 or fitted > nlambda:
        raise RuntimeError(f"invalid fitted path length: {fitted}")
    return {
        "wall_seconds": wall_seconds,
        "num_fit": fitted,
        "iteration_sum": int(iterations[:fitted].sum()),
        "maximum_df": int(size_active[:fitted].max()),
        "beta": beta[:fitted].copy(),
        "intercept": intercept[:fitted].copy(),
        "iterations": iterations[:fitted].copy(),
        "size_active": size_active[:fitted].copy(),
        "lambdas": lambdas[:fitted].copy(),
    }


def worker(args):
    result = solve(args.library, args.case)
    if args.output:
        np.savez(
            args.output,
            beta=result["beta"], intercept=result["intercept"],
            iterations=result["iterations"], size_active=result["size_active"],
            lambdas=result["lambdas"],
            num_fit=np.asarray([result["num_fit"]], dtype=np.int32),
        )
    public = {key: value for key, value in result.items()
              if not isinstance(value, np.ndarray)}
    print(MARKER + json.dumps(public, sort_keys=True))


def run_worker(program: Path, library: str, case_name: str,
               output: Path | None = None):
    command = [
        sys.executable, str(program), "--worker", "--library", library,
        "--case", case_name,
    ]
    if output is not None:
        command.extend(["--output", str(output)])
    completed = subprocess.run(command, check=True, text=True,
                               capture_output=True)
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError("benchmark worker returned no result marker")


def maximum_l1_kkt(x, y, lambdas, beta, intercept):
    residual = y[:, None] - x @ beta.T - intercept[None, :]
    correlation = x.T @ residual / x.shape[0]
    active = np.abs(beta.T) > 1e-8
    stationarity = np.where(
        active,
        np.abs(correlation - lambdas[None, :] * np.sign(beta.T)),
        np.maximum(0.0, np.abs(correlation) - lambdas[None, :]),
    )
    return float(max(np.max(stationarity), np.max(np.abs(residual.mean(axis=0)))))


def compare_paths(baseline_path: Path, candidate_path: Path, case_name: str):
    x, y, _ = make_problem(case_name)
    with np.load(baseline_path) as baseline, np.load(candidate_path) as candidate:
        baseline_fit = int(baseline["num_fit"][0])
        candidate_fit = int(candidate["num_fit"][0])
        if baseline_fit != candidate_fit:
            raise RuntimeError(
                f"path length differs: {baseline_fit} versus {candidate_fit}"
            )
        baseline_prediction = (
            x @ baseline["beta"].T + baseline["intercept"][None, :]
        )
        candidate_prediction = (
            x @ candidate["beta"].T + candidate["intercept"][None, :]
        )
        baseline_objective = np.mean(
            (y[:, None] - baseline_prediction) ** 2, axis=0
        ) / 2
        candidate_objective = np.mean(
            (y[:, None] - candidate_prediction) ** 2, axis=0
        ) / 2
        return {
            "num_fit": baseline_fit,
            "max_abs_beta_difference": float(np.max(np.abs(
                baseline["beta"] - candidate["beta"]
            ))),
            "max_abs_intercept_difference": float(np.max(np.abs(
                baseline["intercept"] - candidate["intercept"]
            ))),
            "active_size_identical": bool(np.array_equal(
                baseline["size_active"], candidate["size_active"]
            )),
            "max_abs_active_size_difference": int(np.max(np.abs(
                baseline["size_active"] - candidate["size_active"]
            ))),
            "max_abs_prediction_difference": float(np.max(np.abs(
                baseline_prediction - candidate_prediction
            ))),
            "max_abs_objective_difference": float(np.max(np.abs(
                baseline_objective - candidate_objective
            ))),
            "baseline_max_l1_kkt": maximum_l1_kkt(
                x, y, baseline["lambdas"], baseline["beta"],
                baseline["intercept"]
            ),
            "candidate_max_l1_kkt": maximum_l1_kkt(
                x, y, candidate["lambdas"], candidate["beta"],
                candidate["intercept"]
            ),
        }


def orchestrate(args):
    program = Path(__file__).resolve()
    report = {"repeats": args.repeats, "cases": {}}
    with tempfile.TemporaryDirectory(prefix="picasso-actgd-") as directory:
        temporary = Path(directory)
        for case_name in CASES:
            baseline_output = temporary / f"{case_name}-baseline.npz"
            candidate_output = temporary / f"{case_name}-candidate.npz"
            baseline_check = run_worker(
                program, args.baseline, case_name, baseline_output
            )
            candidate_check = run_worker(
                program, args.candidate, case_name, candidate_output
            )
            correctness = compare_paths(
                baseline_output, candidate_output, case_name
            )

            timings = {"baseline": [], "candidate": []}
            for repeat in range(args.repeats):
                order = ("baseline", "candidate") if repeat % 2 == 0 else (
                    "candidate", "baseline"
                )
                for mode in order:
                    library = args.baseline if mode == "baseline" else args.candidate
                    result = run_worker(program, library, case_name)
                    timings[mode].append(result["wall_seconds"])

            baseline_median = statistics.median(timings["baseline"])
            candidate_median = statistics.median(timings["candidate"])
            report["cases"][case_name] = {
                "configuration": CASES[case_name],
                "correctness": correctness,
                "candidate_iteration_sum": candidate_check["iteration_sum"],
                "maximum_df": candidate_check["maximum_df"],
                "baseline_seconds": timings["baseline"],
                "candidate_seconds": timings["candidate"],
                "baseline_median_seconds": baseline_median,
                "candidate_median_seconds": candidate_median,
                "speedup": baseline_median / candidate_median,
            }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--library")
    parser.add_argument("--case", choices=tuple(CASES))
    args = parser.parse_args()
    if args.worker:
        if not args.library or not args.case:
            parser.error("worker mode requires --library and --case")
    elif not args.baseline or not args.candidate or args.repeats < 3:
        parser.error("comparison requires both libraries and at least 3 repeats")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        orchestrate(arguments)
