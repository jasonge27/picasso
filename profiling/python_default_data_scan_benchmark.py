"""Compare omitted versus explicit training data in prediction/assessment."""

import json
import sys
import time

import numpy as np

import pycasso


def median_elapsed(call, repetitions):
    elapsed = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = call()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed)), result


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: benchmark.py N D REPETITIONS OUTPUT.json")
    n, d, repetitions = map(int, sys.argv[1:4])
    output = sys.argv[4]
    rng = np.random.default_rng(20260720)
    x = rng.normal(size=(n, d))
    y = np.linspace(-1.0, 1.0, n)
    solver = pycasso.Solver(
        x, y, family="gaussian", lambdas=np.array([0.3, 0.2, 0.1]),
        standardize=False)
    solver.result.update({
        "state": "trained",
        "beta": np.zeros((3, d)),
        "intercept": np.zeros(3),
    })

    predict_default, default_prediction = median_elapsed(
        solver.predict, repetitions)
    predict_explicit, explicit_prediction = median_elapsed(
        lambda: solver.predict(solver._x_orig), repetitions)
    assess_default, default_assessment = median_elapsed(
        solver.assess, repetitions)
    assess_explicit, explicit_assessment = median_elapsed(
        lambda: solver.assess(solver._x_orig, solver.y), repetitions)
    if not np.array_equal(default_prediction, explicit_prediction):
        raise AssertionError("prediction output changed")
    if any(not np.array_equal(default_assessment[key],
                              explicit_assessment[key])
           for key in default_assessment):
        raise AssertionError("assessment output changed")
    summary = {
        "n": n,
        "d": d,
        "repetitions": repetitions,
        "predict_default": predict_default,
        "predict_explicit": predict_explicit,
        "assess_default": assess_default,
        "assess_explicit": assess_explicit,
    }
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
