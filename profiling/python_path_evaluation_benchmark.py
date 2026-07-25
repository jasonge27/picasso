#!/usr/bin/env python3
"""Benchmark scalar path evaluation without including native model fitting."""

import argparse
import resource
import time

import numpy as np

from pycasso import core


def response_for_family(rng, x, family):
    signal = x[:, :min(5, x.shape[1])] @ np.linspace(
        0.25, -0.15, min(5, x.shape[1]))
    if family in ("gaussian", "sqrtlasso"):
        return signal + rng.normal(scale=0.5, size=x.shape[0])
    if family == "binomial":
        probability = core._sigmoid(signal)
        return (rng.random(x.shape[0]) < probability).astype("double")
    return rng.poisson(np.exp(np.clip(signal, -2.0, 2.0))).astype("double")


def make_solver(x, y, beta, intercept, family, offset):
    solver = core.Solver.__new__(core.Solver)
    solver.family = family
    solver._x_orig = x
    solver.y = y
    solver.num_feature = x.shape[1]
    solver.nlambda = beta.shape[0]
    solver.lambdas = np.geomspace(1.0, 0.01, beta.shape[0])
    solver._offset_supplied = offset is not None
    solver._offset = (offset if offset is not None else
                      np.zeros(x.shape[0], dtype="double"))
    solver.result = {
        "state": "trained",
        "beta": beta,
        "intercept": intercept,
    }
    return solver


def checksum(metrics):
    return float(sum(np.sum(value) for key, value in metrics.items()
                     if key != "lambda"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=(
        "gaussian", "sqrtlasso", "binomial", "poisson"),
        default="gaussian")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--features", type=int, default=200)
    parser.add_argument("--lambdas", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--operation", choices=("deviance", "assess"),
                        default="assess")
    parser.add_argument("--layout", choices=("contiguous", "fortran", "strided"),
                        default="contiguous")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--block-mib", type=float)
    args = parser.parse_args()

    if args.block_mib is not None:
        core._SCALAR_PATH_BLOCK_BYTES = max(
            1, int(args.block_mib * 1024 * 1024))

    rng = np.random.default_rng(args.seed)
    if args.layout == "strided":
        x = rng.normal(size=(args.samples, 2 * args.features))[:, ::2]
    elif args.layout == "fortran":
        x = np.asfortranarray(
            rng.normal(size=(args.samples, args.features)), dtype="double")
    else:
        x = np.ascontiguousarray(
            rng.normal(size=(args.samples, args.features)), dtype="double")
    beta = np.ascontiguousarray(
        rng.normal(scale=0.05, size=(args.lambdas, args.features)),
        dtype="double")
    intercept = rng.normal(scale=0.05, size=args.lambdas)
    y = response_for_family(rng, x, args.family)
    offset = (rng.normal(scale=0.1, size=args.samples)
              if args.family in ("binomial", "poisson") else None)
    solver = make_solver(x, y, beta, intercept, args.family, offset)

    def evaluate():
        if args.operation == "deviance":
            return {"deviance": core._fit_deviances(
                y, x, beta, intercept, args.family, offset=offset)}
        return solver.assess(x, y, newoffset=offset)

    evaluate()
    timings = []
    result = None
    for _ in range(args.repetitions):
        start = time.perf_counter()
        result = evaluate()
        timings.append(time.perf_counter() - start)

    print(f"family={args.family}")
    print(f"operation={args.operation}")
    print(f"samples={args.samples}")
    print(f"features={args.features}")
    print(f"lambdas={args.lambdas}")
    print(f"layout={args.layout}")
    print("block_bytes=" + str(getattr(
        core, "_SCALAR_PATH_BLOCK_BYTES", "one-model-at-a-time")))
    print(f"median_seconds={np.median(timings):.9f}")
    print(f"minimum_seconds={np.min(timings):.9f}")
    print(f"checksum={checksum(result):.17g}")
    print(f"peak_rss_bytes={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")


if __name__ == "__main__":
    main()
