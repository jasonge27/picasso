"""Benchmark serial versus opt-in threaded Python cross-validation.

Set ``PYTHONPATH`` to the Python source tree under test and
``PICASSO_NATIVE_LIBRARY`` to one fixed native library. Also set the vendor
BLAS, OpenBLAS, and OMP thread limits to one for comparable measurements.
"""

import json
import os
import sys
import time

import numpy as np

import pycasso


def make_problem(setting, family):
    rng = np.random.RandomState(20260720)
    if setting == "tall":
        n, d = 20000, 120
    elif setting == "wide":
        n, d = 300, 2500
    else:
        raise ValueError(setting)

    x = rng.normal(size=(n, d))
    coefficients = np.zeros(d)
    coefficients[:8] = np.linspace(0.45, -0.2, 8)
    signal = x @ coefficients
    offset = np.linspace(-0.25, 0.25, n)
    kwargs = {}
    if family in ("gaussian", "sqrtlasso"):
        y = signal + rng.normal(scale=0.6, size=n)
    elif family == "binomial":
        y = (signal + offset + rng.logistic(size=n) > 0).astype(float)
        kwargs["offset"] = offset
    elif family == "poisson":
        mean = np.exp(np.clip(0.15 + 0.2 * signal + offset, -2.0, 2.0))
        y = rng.poisson(mean).astype(float)
        kwargs["offset"] = offset
    elif family == "multinomial":
        y = (np.arange(n) % 3).astype(float)
        x[np.arange(n), y.astype(int)] += 1.25
    else:
        raise ValueError(family)
    return x, y, kwargs


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: benchmark.py SETTING FAMILY N_JOBS REPETITIONS")
    setting, family = sys.argv[1:3]
    n_jobs, repetitions = map(int, sys.argv[3:5])
    x, y, kwargs = make_problem(setting, family)
    foldid = np.arange(x.shape[0]) % 5
    if family == "sqrtlasso" and setting == "wide":
        lambdas = np.geomspace(0.5, 0.2, 12)
    else:
        lambdas = np.geomspace(0.25, 0.015, 20)
    solver = pycasso.Solver(
        x, y, lambdas=lambdas, family=family,
        fast_mode=True, max_ite=300, **kwargs)
    solver.train()

    result = solver.cross_validate(foldid=foldid, n_jobs=n_jobs)
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = solver.cross_validate(foldid=foldid, n_jobs=n_jobs)
        elapsed.append(time.perf_counter() - start)
    print(json.dumps({
        "setting": setting,
        "family": family,
        "n_jobs": n_jobs,
        "repetitions": repetitions,
        "times": elapsed,
        "median": float(np.median(elapsed)),
        "checksum": float(np.sum(result["cvm"])),
        "nlambda": int(len(result["lambda"])),
        "pid": os.getpid(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
