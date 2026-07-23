#!/usr/bin/env python3
"""A/B benchmark one-pass multinomial assessment across Python sources.

Each timing or memory observation runs in a fresh process.  The controller
interleaves baseline and candidate workers, stages Python sources without
bundled binaries or bytecode, and forces every worker to load one explicitly
hashed native library.  Model fitting is intentionally excluded.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np


SOURCE_FILES = (
    'python-package/pycasso/core.py',
    'python-package/pycasso/libpath.py',
    'python-package/pycasso/__init__.py',
    'python-package/pycasso/VERSION',
)
THREAD_ENVIRONMENT = {
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance(source_root):
    source_root = Path(source_root).expanduser().resolve()
    files = {}
    for relative_name in SOURCE_FILES:
        source_file = source_root / relative_name
        if not source_file.is_file():
            raise ValueError(
                f'Python source root is missing {relative_name}: '
                f'{source_root}')
        files[relative_name] = {
            'path': str(source_file),
            'sha256': _sha256_file(source_file),
        }
    return files


def _stage_source(source_root, staging_root, label):
    source_package = (
        Path(source_root).expanduser().resolve() /
        'python-package' / 'pycasso')
    staged_root = Path(staging_root) / label
    staged_package = staged_root / 'python-package' / 'pycasso'
    staged_package.parent.mkdir(parents=True)
    shutil.copytree(
        source_package, staged_package,
        ignore=shutil.ignore_patterns(
            'lib', '__pycache__', 'src', '*.pyc'))
    return staged_root


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def _metric_checksum(metrics):
    digest = hashlib.sha256()
    for name in ('lambda', 'deviance', 'class_error'):
        values = np.ascontiguousarray(metrics[name])
        digest.update(name.encode('ascii'))
        digest.update(values.dtype.str.encode('ascii'))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def _make_solver(core, args):
    rng = np.random.default_rng(args.seed)
    x = np.ascontiguousarray(
        rng.normal(size=(args.n, args.d)), dtype='double')
    beta = np.ascontiguousarray(
        rng.normal(
            scale=0.05,
            size=(args.nlambda, args.classes, args.d)),
        dtype='double')
    intercept = np.ascontiguousarray(
        rng.normal(scale=0.05, size=(args.nlambda, args.classes)),
        dtype='double')
    y = np.arange(args.n, dtype=np.intp) % args.classes
    rng.shuffle(y)

    solver = core.Solver.__new__(core.Solver)
    solver.family = 'multinomial'
    solver._x_orig = x
    solver.y = y.copy()
    solver.num_feature = args.d
    solver.nlambda = args.nlambda
    solver.lambdas = np.geomspace(1.0, 0.01, args.nlambda)
    solver._mn_levels = np.arange(args.classes)
    solver._offset_supplied = False
    solver.result = {
        'state': 'trained',
        'beta': beta,
        'intercept': intercept,
    }
    return solver, x, y


def _verify_loaded_library(core, expected_path, expected_digest):
    loaded_name = getattr(core._PICASSO_LIB, '_name', None)
    if not loaded_name:
        raise RuntimeError('Could not determine loaded native-library path.')
    loaded_path = Path(loaded_name).expanduser().resolve()
    if loaded_path != expected_path:
        raise RuntimeError(
            f'Loaded native library {loaded_path}, expected {expected_path}.')
    loaded_digest = _sha256_file(loaded_path)
    if loaded_digest != expected_digest:
        raise RuntimeError(
            f'Loaded native SHA-256 {loaded_digest}, expected '
            f'{expected_digest}.')
    return loaded_path, loaded_digest


def _worker(args):
    sys.dont_write_bytecode = True
    source_root = Path(args.source_root).expanduser().resolve()
    native_path = Path(args.native_library).expanduser().resolve()
    os.environ['PICASSO_NATIVE_LIBRARY'] = str(native_path)
    sys.path.insert(0, str(source_root / 'python-package'))
    from pycasso import core

    loaded_path, loaded_digest = _verify_loaded_library(
        core, native_path, args.native_sha256)
    solver, x, y = _make_solver(core, args)

    if args.mode == 'runtime':
        warmup = solver.assess(x, y)
        timings = []
        metrics = warmup
        for _ in range(args.inner_repeats):
            start = time.perf_counter()
            metrics = solver.assess(x, y)
            timings.append(time.perf_counter() - start)
        elapsed = statistics.median(timings)
        trace_peak = None
        rss_before = None
        rss_after = _peak_rss_bytes()
        rss_delta = None
    else:
        gc.collect()
        rss_before = _peak_rss_bytes()
        tracemalloc.start()
        tracemalloc.reset_peak()
        metrics = solver.assess(x, y)
        _, trace_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _peak_rss_bytes()
        rss_delta = max(0, rss_after - rss_before)
        elapsed = None
        timings = None

    if (metrics['deviance'].shape != (args.nlambda,) or
            metrics['class_error'].shape != (args.nlambda,) or
            not np.all(np.isfinite(metrics['deviance'])) or
            not np.all(np.isfinite(metrics['class_error']))):
        raise RuntimeError('Assessment returned invalid metric arrays.')
    record = {
        'checksum': _metric_checksum(metrics),
        'elapsed_seconds': elapsed,
        'inner_timings_seconds': timings,
        'label': args.label,
        'loaded_library': str(loaded_path),
        'loaded_library_sha256': loaded_digest,
        'mode': args.mode,
        'peak_rss_after_bytes': rss_after,
        'peak_rss_before_bytes': rss_before,
        'peak_rss_delta_bytes': rss_delta,
        'tracemalloc_peak_bytes': trace_peak,
    }
    print(json.dumps(record, sort_keys=True))


def _worker_command(args, label, source_root, native_digest, mode):
    return [
        sys.executable, str(Path(__file__).resolve()), '--worker',
        '--label', label, '--source-root', str(source_root),
        '--native-library', str(args.native_library),
        '--native-sha256', native_digest, '--mode', mode,
        '--repeats', str(args.repeats),
        '--inner-repeats', str(args.inner_repeats),
        '--n', str(args.n), '--d', str(args.d),
        '--classes', str(args.classes), '--nlambda', str(args.nlambda),
        '--seed', str(args.seed),
    ]


def _run_worker(args, label, source_root, native_digest, mode):
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(
        _worker_command(
            args, label, source_root, native_digest, mode),
        check=False, text=True, capture_output=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Worker failed for {label}/{mode} '
            f'(exit {completed.returncode}).\n'
            f'stdout:\n{completed.stdout}\n'
            f'stderr:\n{completed.stderr}')
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f'Worker returned no JSON for {label}/{mode}.')
    try:
        record = json.loads(lines[-1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'Worker returned invalid JSON for {label}/{mode}:\n'
            f'{completed.stdout}') from exc
    expected_path = str(Path(args.native_library).expanduser().resolve())
    if (record.get('loaded_library') != expected_path or
            record.get('loaded_library_sha256') != native_digest):
        raise RuntimeError(
            f'Worker native provenance mismatch for {label}/{mode}: '
            f'{record}.')
    return record


def _median(records, field):
    values = [record[field] for record in records]
    if any(value is None for value in values):
        return None
    return statistics.median(values)


def _summarize(records):
    return {
        'checksum': records[0]['checksum'],
        'median_elapsed_seconds': _median(records, 'elapsed_seconds'),
        'median_peak_rss_delta_bytes': _median(
            records, 'peak_rss_delta_bytes'),
        'median_tracemalloc_peak_bytes': _median(
            records, 'tracemalloc_peak_bytes'),
        'runs': records,
    }


def _controller(args):
    sources = (
        ('baseline', Path(args.baseline_root).expanduser().resolve()),
        ('candidate', Path(args.candidate_root).expanduser().resolve()),
    )
    native_path = Path(args.native_library).expanduser().resolve()
    native_digest = _sha256_file(native_path)
    source_files = {
        label: _source_provenance(source_root)
        for label, source_root in sources
    }
    records = {
        label: {'runtime': [], 'memory': []} for label, _ in sources
    }

    with tempfile.TemporaryDirectory(
            prefix='picasso-mn-assess-benchmark-') as temporary_directory:
        staged = {
            label: _stage_source(
                source_root, temporary_directory, label)
            for label, source_root in sources
        }
        for repetition in range(args.repeats):
            ordered_sources = (sources if repetition % 2 == 0
                               else tuple(reversed(sources)))
            for mode in ('runtime', 'memory'):
                for label, _ in ordered_sources:
                    records[label][mode].append(_run_worker(
                        args, label, staged[label], native_digest, mode))

    summaries = {
        label: {
            mode: _summarize(mode_records)
            for mode, mode_records in source_records.items()
        }
        for label, source_records in records.items()
    }
    checksums = {
        summary['checksum']
        for source_summary in summaries.values()
        for summary in source_summary.values()
    }
    if len(checksums) != 1:
        raise RuntimeError(
            'Baseline/candidate metric checksums differ: '
            f'{sorted(checksums)}')

    baseline_seconds = summaries['baseline']['runtime'][
        'median_elapsed_seconds']
    candidate_seconds = summaries['candidate']['runtime'][
        'median_elapsed_seconds']
    payload = {
        'config': {
            'classes': args.classes,
            'd': args.d,
            'inner_repeats': args.inner_repeats,
            'n': args.n,
            'nlambda': args.nlambda,
            'repeats': args.repeats,
            'seed': args.seed,
        },
        'environment': {
            'machine': platform.machine(),
            'numpy': np.__version__,
            'platform': platform.platform(),
            'python': sys.version,
            'thread_controls': THREAD_ENVIRONMENT,
        },
        'native_library': str(native_path),
        'native_library_sha256': native_digest,
        'source_files': source_files,
        'sources': {label: str(root) for label, root in sources},
        'summaries': summaries,
        'comparison': {
            'metric_checksum_equal': True,
            'runtime_speedup_baseline_over_candidate': (
                baseline_seconds / candidate_seconds),
            'tracemalloc_reduction_bytes': (
                summaries['baseline']['memory'][
                    'median_tracemalloc_peak_bytes'] -
                summaries['candidate']['memory'][
                    'median_tracemalloc_peak_bytes']),
            'peak_rss_delta_reduction_bytes': (
                summaries['baseline']['memory'][
                    'median_peak_rss_delta_bytes'] -
                summaries['candidate']['memory'][
                    'median_peak_rss_delta_bytes']),
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        Path(args.output).expanduser().resolve().write_text(
            serialized, encoding='utf-8')
    sys.stdout.write(serialized)


def _arguments():
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root')
    parser.add_argument('--candidate-root', default=str(repository_root))
    parser.add_argument('--native-library', required=True)
    parser.add_argument('--output')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--inner-repeats', type=int, default=3)
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--nlambda', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20260719)
    parser.add_argument('--worker', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--label', help=argparse.SUPPRESS)
    parser.add_argument('--source-root', help=argparse.SUPPRESS)
    parser.add_argument('--native-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('runtime', 'memory'),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    for name in ('repeats', 'inner_repeats', 'n', 'd', 'nlambda'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive.')
    if args.classes < 3 or args.n < args.classes:
        parser.error('--classes must be at least 3 and no greater than --n.')
    if not Path(args.native_library).expanduser().resolve().is_file():
        parser.error('--native-library must name an existing file.')
    if args.worker:
        for name in ('label', 'source_root', 'native_sha256', 'mode'):
            if getattr(args, name) is None:
                parser.error(f'--{name.replace("_", "-")} is required.')
        _source_provenance(args.source_root)
    else:
        if args.baseline_root is None:
            parser.error('--baseline-root is required.')
        _source_provenance(args.baseline_root)
        _source_provenance(args.candidate_root)
        if args.output is not None and not Path(
                args.output).expanduser().resolve().parent.is_dir():
            parser.error('--output parent directory must exist.')
    return args


if __name__ == '__main__':
    arguments = _arguments()
    if arguments.worker:
        _worker(arguments)
    else:
        _controller(arguments)
