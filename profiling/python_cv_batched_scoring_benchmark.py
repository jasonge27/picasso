"""Benchmark one scalar Python CV scoring configuration.

Use ``PYTHONPATH`` to select the baseline or candidate source tree and keep
``PICASSO_NATIVE_LIBRARY`` fixed. Limit BLAS and OMP to one thread.
"""

import json
import os
import sys
import time

import numpy as np

import pycasso


def make_problem(family, setting):
    if setting == "tall":
        n, d = 20000, 20
    elif setting == "wide":
        n, d = 240, 1000
    else:
        raise ValueError(setting)
    rng = np.random.default_rng(20260720)
    x = rng.normal(size=(n, d))
    coefficient = np.zeros(d)
    coefficient[:min(d, 12)] = np.linspace(0.8, -0.4, min(d, 12))
    signal = x @ coefficient
    offset = np.linspace(-0.2, 0.2, n)
    kwargs = {}
    if family in ("gaussian", "sqrtlasso"):
        y = signal + rng.normal(scale=0.7, size=n)
    elif family == "binomial":
        probability = 1.0 / (1.0 + np.exp(-np.clip(signal + offset,
                                                   -30.0, 30.0)))
        y = rng.binomial(1, probability, size=n).astype(float)
        kwargs["offset"] = offset
    elif family == "poisson":
        mean = np.exp(np.clip(0.25 * signal + offset, -2.0, 2.0))
        y = rng.poisson(mean, size=n).astype(float)
        kwargs["offset"] = offset
    else:
        raise ValueError(family)
    return x, y, np.arange(n, dtype=int) % 5, kwargs


def main():
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: benchmark.py SETTING FAMILY MEASURE N_JOBS "
            "REPETITIONS OUTPUT.json")
    setting, family, measure = sys.argv[1:4]
    n_jobs, repetitions = map(int, sys.argv[4:6])
    output = sys.argv[6]
    x, y, foldid, kwargs = make_problem(family, setting)
    ratio = 0.2 if family == "sqrtlasso" and setting == "wide" else 0.05

    elapsed = []
    result = None
    for _ in range(repetitions):
        solver = pycasso.Solver(
            x, y, family=family, lambdas=(100, ratio), fast_mode=True,
            max_ite=1000, **kwargs)
        start = time.perf_counter()
        result = solver.cross_validate(
            foldid=foldid, type_measure=measure, n_jobs=n_jobs)
        elapsed.append(time.perf_counter() - start)
    summary = {
        "setting": setting,
        "family": family,
        "measure": measure,
        "n_jobs": n_jobs,
        "repetitions": repetitions,
        "times": elapsed,
        "median": float(np.median(elapsed)),
        "lambda_min": float(result["lambda_min"]),
        "lambda_1se": float(result["lambda_1se"]),
        "cvm_checksum": float(np.sum(result["cvm"])),
        "pid": os.getpid(),
    }
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
