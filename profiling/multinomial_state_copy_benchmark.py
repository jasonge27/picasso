#!/usr/bin/env python3
"""A/B benchmark for multinomial path-state copy elimination.

Each measurement runs one shared library in a fresh process.  Inputs and
caller-owned outputs are allocated before the RSS baseline, and all native
diagnostics except runtime participate in a byte-exact checksum.
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


MARKER = "PICASSO_MULTINOMIAL_STATE_COPY="
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def cpu_description() -> str:
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=False,
            capture_output=True, text=True)
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    return platform.processor() or platform.machine()


def configure_solver(library: ctypes.CDLL):
    function = library.SolveMultinomialRegressionV5
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    long_pointer = ctypes.POINTER(ctypes.c_longlong)
    function.argtypes = [
        double_pointer, double_pointer, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, double_pointer, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_bool,
        ctypes.c_int, double_pointer, double_pointer, int_pointer,
        int_pointer, double_pointer, int_pointer, ctypes.c_bool,
        ctypes.c_int, ctypes.c_bool, int_pointer, int_pointer, int_pointer,
        long_pointer, long_pointer, double_pointer, double_pointer,
        double_pointer, double_pointer,
    ]
    function.restype = ctypes.c_int
    return function


def checksum_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def worker(args: argparse.Namespace) -> dict:
    library_path = Path(args.worker_library).expanduser().resolve()
    library_digest = sha256_file(library_path)
    rng = np.random.default_rng(args.seed)
    x = np.ascontiguousarray(rng.standard_normal((args.n, args.d)))
    y_codes = np.arange(args.n, dtype=np.int64) % args.classes
    y = np.ascontiguousarray(y_codes, dtype=np.float64)

    proportions = np.bincount(
        y_codes, minlength=args.classes).astype(np.float64) / args.n
    lambda_max = max(
        float(np.max(np.abs(
            x.T @ ((y_codes == klass).astype(np.float64) -
                   proportions[klass])))) / args.n
        for klass in range(args.classes)
    )
    lambdas = np.ascontiguousarray(np.linspace(
        args.lambda_multiplier, args.lambda_end_multiplier, args.nlambda
    ) * lambda_max)

    coefficient_count = args.nlambda * args.classes * args.d
    beta = np.full(coefficient_count, np.nan, dtype=np.float64)
    intercept = np.full(
        args.nlambda * args.classes, np.nan, dtype=np.float64)
    iterations = np.full(args.nlambda, -1, dtype=np.int32)
    active_size = np.full(args.nlambda, -1, dtype=np.int32)
    runtime = np.full(args.nlambda, np.nan, dtype=np.float64)
    outer = np.full(args.nlambda, -1, dtype=np.int32)
    inner = np.full(args.nlambda, -1, dtype=np.int64)
    updates = np.full(args.nlambda, -1, dtype=np.int64)
    objective = np.full(args.nlambda, np.nan, dtype=np.float64)
    kkt = np.full(args.nlambda, np.nan, dtype=np.float64)
    stationarity = np.full(args.nlambda, np.nan, dtype=np.float64)
    smooth = np.full(args.nlambda, np.nan, dtype=np.float64)
    num_fit = ctypes.c_int(-1)
    failed_lambda = ctypes.c_int(-2)
    failed_stage = ctypes.c_int(-2)

    library = ctypes.CDLL(str(library_path))
    solve = configure_solver(library)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    long_pointer = ctypes.POINTER(ctypes.c_longlong)
    rss_before = peak_rss_bytes()
    started = time.perf_counter()
    status = int(solve(
        y.ctypes.data_as(double_pointer),
        x.ctypes.data_as(double_pointer), args.n, args.d, args.classes,
        lambdas.ctypes.data_as(double_pointer), args.nlambda, args.gamma,
        args.max_iterations, args.precision, args.penalty, True, -1,
        beta.ctypes.data_as(double_pointer),
        intercept.ctypes.data_as(double_pointer),
        iterations.ctypes.data_as(int_pointer),
        active_size.ctypes.data_as(int_pointer),
        runtime.ctypes.data_as(double_pointer), ctypes.byref(num_fit), True,
        args.max_stages, False, ctypes.byref(failed_lambda),
        ctypes.byref(failed_stage), outer.ctypes.data_as(int_pointer),
        inner.ctypes.data_as(long_pointer),
        updates.ctypes.data_as(long_pointer),
        objective.ctypes.data_as(double_pointer),
        kkt.ctypes.data_as(double_pointer),
        stationarity.ctypes.data_as(double_pointer),
        smooth.ctypes.data_as(double_pointer),
    ))
    elapsed = time.perf_counter() - started
    rss_after = peak_rss_bytes()

    if status not in (0, 10) or num_fit.value != args.nlambda:
        raise RuntimeError(
            "multinomial path failed: "
            f"status={status}, num_fit={num_fit.value}, "
            f"failed_lambda={failed_lambda.value}, "
            f"failed_stage={failed_stage.value}")
    if (not np.all(np.isfinite(beta)) or
            not np.all(np.isfinite(intercept)) or
            not np.all(np.isfinite(objective)) or
            not np.all(np.isfinite(kkt)) or
            not np.all(np.isfinite(stationarity)) or
            not np.all(np.isfinite(smooth))):
        raise RuntimeError("committed multinomial outputs are not finite")

    return {
        "checksum": checksum_arrays(
            beta, intercept, iterations, active_size, outer, inner, updates,
            objective, kkt, stationarity, smooth,
        ),
        "elapsed_seconds": elapsed,
        "failed_lambda": failed_lambda.value,
        "failed_stage": failed_stage.value,
        "library": str(library_path),
        "library_sha256": library_digest,
        "num_fit": num_fit.value,
        "peak_rss_delta_bytes": max(0, rss_after - rss_before),
        "status": status,
        "total_coordinate_updates": int(updates.sum()),
        "total_inner_sweeps": int(inner.sum()),
    }


def worker_command(args: argparse.Namespace, label: str, library: Path,
                   seed: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--label", label, "--worker-library", str(library),
        "--seed", str(seed), "--n", str(args.n), "--d", str(args.d),
        "--classes", str(args.classes), "--nlambda", str(args.nlambda),
        "--penalty", str(args.penalty), "--gamma", repr(args.gamma),
        "--precision", repr(args.precision), "--max-iterations",
        str(args.max_iterations), "--max-stages", str(args.max_stages),
        "--lambda-multiplier", repr(args.lambda_multiplier),
        "--lambda-end-multiplier", repr(args.lambda_end_multiplier),
    ]


def invoke_worker(args: argparse.Namespace, label: str, library: Path,
                  seed: int) -> dict:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in THREAD_VARIABLES:
        environment[name] = "1"
    completed = subprocess.run(
        worker_command(args, label, library, seed), check=False,
        capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} worker failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError(f"{label} worker returned no benchmark record")


def summarize(records: list[dict]) -> dict:
    checksums = {record["checksum"] for record in records}
    control_signatures = {
        (record["status"], record["num_fit"], record["failed_lambda"],
         record["failed_stage"])
        for record in records
    }
    return {
        "checksum": records[0]["checksum"],
        "checksum_stable": len(checksums) == 1,
        "control_signature": list(control_signatures)[0],
        "control_signature_stable": len(control_signatures) == 1,
        "median_elapsed_seconds": statistics.median(
            record["elapsed_seconds"] for record in records),
        "median_peak_rss_delta_bytes": statistics.median(
            record["peak_rss_delta_bytes"] for record in records),
        "runs": records,
    }


def controller(args: argparse.Namespace) -> dict:
    baseline = Path(args.baseline_library).expanduser().resolve()
    candidate = Path(args.candidate_library).expanduser().resolve()
    for path in (baseline, candidate):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Discard one fresh-process warmup for each implementation.
    for label, library in (("baseline", baseline), ("candidate", candidate)):
        invoke_worker(args, label, library, args.seed - 1)

    records = {"baseline": [], "candidate": []}
    labels = ("baseline", "candidate")
    libraries = {"baseline": baseline, "candidate": candidate}
    for repeat in range(args.repeats):
        order = labels if repeat % 2 == 0 else tuple(reversed(labels))
        for label in order:
            records[label].append(invoke_worker(
                args, label, libraries[label], args.seed))

    summaries = {label: summarize(values)
                 for label, values in records.items()}
    if (not summaries["baseline"]["checksum_stable"] or
            not summaries["candidate"]["checksum_stable"] or
            not summaries["baseline"]["control_signature_stable"] or
            not summaries["candidate"]["control_signature_stable"] or
            summaries["baseline"]["checksum"] !=
            summaries["candidate"]["checksum"] or
            summaries["baseline"]["control_signature"] !=
            summaries["candidate"]["control_signature"]):
        raise RuntimeError(
            "baseline and candidate outputs/statuses are not equivalent")
    old_time = summaries["baseline"]["median_elapsed_seconds"]
    new_time = summaries["candidate"]["median_elapsed_seconds"]
    old_rss = summaries["baseline"]["median_peak_rss_delta_bytes"]
    new_rss = summaries["candidate"]["median_peak_rss_delta_bytes"]
    return {
        "comparison": {
            "checksum_equal": True,
            "runtime_speedup_baseline_over_candidate": old_time / new_time,
            "peak_rss_delta_reduction_bytes": old_rss - new_rss,
        },
        "config": {
            name: getattr(args, name) for name in (
                "n", "d", "classes", "nlambda", "penalty", "gamma",
                "precision", "max_iterations", "max_stages",
                "lambda_multiplier", "lambda_end_multiplier", "repeats",
                "seed")
        },
        "environment": {
            "cpu": cpu_description(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "thread_controls": {name: "1" for name in THREAD_VARIABLES},
        },
        "libraries": {
            "baseline": {"path": str(baseline),
                         "sha256": sha256_file(baseline)},
            "candidate": {"path": str(candidate),
                          "sha256": sha256_file(candidate)},
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "summaries": summaries,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-library")
    result.add_argument("--candidate-library")
    result.add_argument("--output", default="-")
    result.add_argument("--repeats", type=int, default=7)
    result.add_argument("--seed", type=int, default=20260719)
    result.add_argument("--n", type=int, default=8)
    result.add_argument("--d", type=int, default=100000)
    result.add_argument("--classes", type=int, default=4)
    result.add_argument("--nlambda", type=int, default=30)
    result.add_argument("--penalty", type=int, choices=(1, 2, 3), default=2)
    result.add_argument("--gamma", type=float, default=3.0)
    result.add_argument("--precision", type=float, default=1e-4)
    result.add_argument("--max-iterations", type=int, default=1000)
    result.add_argument("--max-stages", type=int, default=3)
    result.add_argument("--lambda-multiplier", type=float, default=1.5)
    result.add_argument("--lambda-end-multiplier", type=float, default=1.05)
    result.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    result.add_argument("--worker-library", help=argparse.SUPPRESS)
    result.add_argument("--label", help=argparse.SUPPRESS)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        print(MARKER + json.dumps(worker(args), sort_keys=True))
        return
    if not args.baseline_library or not args.candidate_library:
        raise SystemExit(
            "--baseline-library and --candidate-library are required")
    output = controller(args)
    serialized = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(serialized)
    else:
        Path(args.output).write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
