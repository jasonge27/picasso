#!/usr/bin/env python3
"""Fresh-process A/B benchmark for Python multinomial rescaling.

The controller stages baseline and candidate Python sources independently and
injects byte-identical copies of one explicitly selected native library.  The
worker times only ``_rescale_multinomial_solution_in_place``; native fitting is
not part of this benchmark.
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
                'Python source root is missing %s: %s' %
                (relative_name, source_root))
        files[relative_name] = {
            'path': str(source_file),
            'sha256': _sha256_file(source_file),
        }
    return files


def _native_filename():
    if sys.platform == 'win32':
        return 'picasso.dll'
    if sys.platform == 'darwin':
        return 'libpicasso.dylib'
    if sys.platform.startswith('linux'):
        return 'libpicasso.so'
    raise RuntimeError('Unsupported benchmark platform: %s' % sys.platform)


def _stage_source(source_root, staging_root, label, native_library,
                  native_digest):
    source_package = (
        Path(source_root).expanduser().resolve() /
        'python-package' / 'pycasso')
    staged_root = Path(staging_root) / label
    staged_package = staged_root / 'python-package' / 'pycasso'
    staged_package.parent.mkdir(parents=True)
    shutil.copytree(
        source_package, staged_package,
        ignore=shutil.ignore_patterns(
            'lib', 'src', '__pycache__', '*.pyc'))
    library_directory = staged_package / 'lib'
    library_directory.mkdir()
    staged_library = library_directory / _native_filename()
    shutil.copy2(native_library, staged_library)
    if _sha256_file(staged_library) != native_digest:
        raise RuntimeError('Staged native library hash mismatch for %s.' % label)
    return staged_root, staged_library


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def _array_checksum(*arrays):
    """Hash complete array values together with dtype and shape metadata."""
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode('ascii'))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(contiguous).cast('B'))
    return digest.hexdigest()


def _maximum_ulp_distance(actual, expected):
    """Return the exact largest representable-double distance."""
    sign_bit = np.uint64(1 << 63)

    def ordered(values):
        bits = np.ascontiguousarray(values, dtype='double').view(np.uint64)
        return np.where(bits & sign_bit, ~bits, bits ^ sign_bit)

    actual_ordered = ordered(actual)
    expected_ordered = ordered(expected)
    distance = (
        np.maximum(actual_ordered, expected_ordered) -
        np.minimum(actual_ordered, expected_ordered))
    return int(distance.max(initial=np.uint64(0)))


def _legacy_rescale(beta, intercept, xinvc, xm, block_bytes):
    """Apply the pre-refactor multiply-and-row-sum implementation once."""
    np.multiply(beta, xinvc, out=beta)
    if beta.size == 0:
        return beta, intercept
    feature_count = beta.shape[-1]
    beta_rows = beta.reshape(-1, feature_count)
    intercept_rows = intercept.reshape(-1)
    bytes_per_row = feature_count * np.dtype('double').itemsize
    rows_per_block = max(
        1, block_bytes // max(bytes_per_row, 1))
    rows_per_block = min(rows_per_block, beta_rows.shape[0])
    for start in range(0, beta_rows.shape[0], rows_per_block):
        stop = min(start + rows_per_block, beta_rows.shape[0])
        adjustment = np.sum(beta_rows[start:stop] * xm, axis=1)
        np.subtract(intercept_rows[start:stop], adjustment,
                    out=intercept_rows[start:stop])
    return beta, intercept


def _verify_loaded_library(core, expected_path, expected_digest):
    loaded_name = getattr(core._PICASSO_LIB, '_name', None)
    if not loaded_name:
        raise RuntimeError('Could not determine loaded native-library path.')
    loaded_path = Path(loaded_name).expanduser().resolve()
    if loaded_path != expected_path:
        raise RuntimeError(
            'Loaded native library %s, expected %s.' %
            (loaded_path, expected_path))
    loaded_digest = _sha256_file(loaded_path)
    if loaded_digest != expected_digest:
        raise RuntimeError(
            'Loaded native SHA-256 %s, expected %s.' %
            (loaded_digest, expected_digest))
    return loaded_path


def _make_inputs(args):
    rng = np.random.RandomState(args.seed)
    beta = np.ascontiguousarray(
        rng.normal(scale=0.025,
                   size=(args.nlambda, args.classes, args.d)),
        dtype='double')
    intercept = np.ascontiguousarray(
        rng.normal(scale=0.05, size=(args.nlambda, args.classes)),
        dtype='double')
    # Multiplication by one retains the production scaling pass without
    # changing coefficients over repeated runtime measurements.
    xinvc = np.ones(args.d, dtype='double')
    xm = np.ascontiguousarray(
        rng.normal(scale=0.1, size=args.d), dtype='double')
    return beta, intercept, xinvc, xm


def _worker(args):
    sys.dont_write_bytecode = True
    source_root = Path(args.source_root).expanduser().resolve()
    native_library = Path(args.native_library).expanduser().resolve()
    os.environ['PICASSO_NATIVE_LIBRARY'] = str(native_library)
    sys.path.insert(0, str(source_root / 'python-package'))
    from pycasso import core

    loaded_library = _verify_loaded_library(
        core, native_library, args.native_sha256)
    beta, intercept, xinvc, xm = _make_inputs(args)
    initial_intercept = intercept.copy()

    # Initialize the host BLAS in both arms before observing memory.  This
    # keeps one-time library setup out of the candidate's matrix-vector call.
    np.ones((2, 2), dtype='double') @ np.ones(2, dtype='double')

    timings = None
    trace_peak = None
    rss_before = None
    if args.mode == 'runtime':
        core._rescale_multinomial_solution_in_place(
            beta, intercept, xinvc, xm)
        timings = []
        for _ in range(args.inner_repeats):
            # Reset outside the timed region so every observation sees the
            # same owner values without charging either implementation for a
            # benchmark-only copy.
            np.copyto(intercept, initial_intercept)
            start = time.perf_counter()
            core._rescale_multinomial_solution_in_place(
                beta, intercept, xinvc, xm)
            timings.append(time.perf_counter() - start)
        elapsed = statistics.median(timings)
        rss_after = _peak_rss_bytes()
        rss_delta = None
    else:
        gc.collect()
        rss_before = _peak_rss_bytes()
        tracemalloc.start()
        tracemalloc.reset_peak()
        core._rescale_multinomial_solution_in_place(
            beta, intercept, xinvc, xm)
        _, trace_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _peak_rss_bytes()
        rss_delta = max(0, rss_after - rss_before)
        elapsed = None

    # Build a complete old-formula oracle only after timing and memory have
    # been captured.  Both modes leave one rescaling result in the owners.
    oracle_beta, oracle_intercept, oracle_xinvc, oracle_xm = _make_inputs(args)
    _legacy_rescale(
        oracle_beta, oracle_intercept, oracle_xinvc, oracle_xm,
        core._MULTINOMIAL_RESCALE_BLOCK_BYTES)
    if not np.array_equal(beta, oracle_beta):
        raise RuntimeError('Rescaling changed a coefficient value.')
    absolute_drift = np.abs(intercept - oracle_intercept)
    reduction_scale = np.sum(np.abs(oracle_beta * oracle_xm), axis=2)
    error_bound = (
        64.0 * np.finfo('double').eps *
        (reduction_scale + np.abs(oracle_intercept)))
    if not np.all(absolute_drift <= error_bound):
        raise RuntimeError(
            'Matrix-vector intercept adjustment exceeded its rounding '
            'error bound: max drift=%r, max bound=%r.' %
            (float(absolute_drift.max(initial=0.0)),
             float(error_bound.max(initial=0.0))))

    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(intercept)):
        raise RuntimeError('Rescaling produced non-finite output.')
    record = {
        'actual_full_checksum': _array_checksum(beta, intercept),
        'beta_full_checksum': _array_checksum(beta),
        'elapsed_seconds': elapsed,
        'inner_timings_seconds': timings,
        'label': args.label,
        'loaded_library': str(loaded_library),
        'loaded_library_sha256': _sha256_file(loaded_library),
        'max_abs_intercept_drift': float(
            absolute_drift.max(initial=0.0)),
        'max_intercept_error_bound': float(
            error_bound.max(initial=0.0)),
        'max_ulp_intercept_drift': _maximum_ulp_distance(
            intercept, oracle_intercept),
        'mode': args.mode,
        'oracle_full_checksum': _array_checksum(
            oracle_beta, oracle_intercept),
        'peak_rss_after_bytes': rss_after,
        'peak_rss_before_bytes': rss_before,
        'peak_rss_delta_bytes': rss_delta,
        'tracemalloc_peak_bytes': trace_peak,
    }
    print(json.dumps(record, sort_keys=True))


def _worker_command(args, label, source_root, native_library, native_digest,
                    mode):
    return [
        sys.executable, str(Path(__file__).resolve()), '--worker',
        '--label', label, '--source-root', str(source_root),
        '--native-library', str(native_library),
        '--native-sha256', native_digest, '--mode', mode,
        '--nlambda', str(args.nlambda), '--classes', str(args.classes),
        '--d', str(args.d), '--seed', str(args.seed),
        '--inner-repeats', str(args.inner_repeats),
    ]


def _run_worker(args, label, source_root, native_library, native_digest,
                mode):
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(
        _worker_command(
            args, label, source_root, native_library, native_digest, mode),
        check=True, capture_output=True, text=True, env=environment)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            'Unexpected %s/%s worker output:\n%s\n%s' %
            (label, mode, completed.stdout, completed.stderr))
    return json.loads(lines[0])


def _summarize(records, mode):
    summary = {'runs': records}
    if mode == 'runtime':
        summary['median_elapsed_seconds'] = statistics.median(
            record['elapsed_seconds'] for record in records)
    else:
        summary['median_peak_rss_delta_bytes'] = statistics.median(
            record['peak_rss_delta_bytes'] for record in records)
        summary['median_tracemalloc_peak_bytes'] = statistics.median(
            record['tracemalloc_peak_bytes'] for record in records)
    return summary


def _validate_outputs(records):
    by_mode = {}
    for record in records:
        by_mode.setdefault(record['mode'], {}).setdefault(
            record['label'], []).append(record)
    for mode, labels in by_mode.items():
        oracle_checksums = {
            record['oracle_full_checksum']
            for label_records in labels.values() for record in label_records
        }
        beta_checksums = {
            record['beta_full_checksum']
            for label_records in labels.values() for record in label_records
        }
        if len(oracle_checksums) != 1 or len(beta_checksums) != 1:
            raise RuntimeError(
                'Full fixture/oracle checksums changed in %s mode.' % mode)
        for label, label_records in labels.items():
            actual_checksums = {
                record['actual_full_checksum'] for record in label_records
            }
            if len(actual_checksums) != 1:
                raise RuntimeError(
                    'Full output checksum is nondeterministic for %s/%s.' %
                    (mode, label))
            for record in label_records:
                if (record['max_abs_intercept_drift'] >
                        record['max_intercept_error_bound']):
                    raise RuntimeError(
                        'Full output exceeded the numerical oracle for '
                        '%s/%s.' % (mode, label))


def _blas_metadata():
    """Return JSON-safe NumPy BLAS metadata across old and new NumPy."""
    try:
        configuration = np.show_config(mode='dicts')
        return configuration.get('Build Dependencies', {}).get('blas', {})
    except (TypeError, AttributeError):
        try:
            return np.__config__.get_info('blas_opt_info')
        except AttributeError:
            return {'description': 'NumPy BLAS metadata unavailable'}


def _controller(args):
    baseline_root = Path(args.baseline_root).expanduser().resolve()
    candidate_root = Path(args.candidate_root).expanduser().resolve()
    native_library = Path(args.native_library).expanduser().resolve()
    native_digest = _sha256_file(native_library)
    provenance = {
        'baseline': _source_provenance(baseline_root),
        'candidate': _source_provenance(candidate_root),
    }

    all_records = []
    with tempfile.TemporaryDirectory(
            prefix='picasso-mn-rescale-benchmark-') as staging_directory:
        staged = {}
        for label, source_root in (
                ('baseline', baseline_root), ('candidate', candidate_root)):
            staged[label] = _stage_source(
                source_root, staging_directory, label,
                native_library, native_digest)
        for mode in ('runtime', 'memory'):
            for repetition in range(args.repeats):
                labels = (('baseline', 'candidate') if repetition % 2 == 0
                          else ('candidate', 'baseline'))
                for label in labels:
                    source_root, staged_library = staged[label]
                    all_records.append(_run_worker(
                        args, label, source_root, staged_library,
                        native_digest, mode))

    _validate_outputs(all_records)
    grouped = {}
    for mode in ('runtime', 'memory'):
        grouped[mode] = {}
        for label in ('baseline', 'candidate'):
            grouped[mode][label] = _summarize([
                record for record in all_records
                if record['mode'] == mode and record['label'] == label
            ], mode)

    old_time = grouped['runtime']['baseline']['median_elapsed_seconds']
    new_time = grouped['runtime']['candidate']['median_elapsed_seconds']
    old_trace = grouped['memory']['baseline']['median_tracemalloc_peak_bytes']
    new_trace = grouped['memory']['candidate']['median_tracemalloc_peak_bytes']
    result = {
        'configuration': {
            'classes': args.classes,
            'd': args.d,
            'inner_repeats': args.inner_repeats,
            'nlambda': args.nlambda,
            'repeats': args.repeats,
            'seed': args.seed,
            'thread_environment': THREAD_ENVIRONMENT,
        },
        'environment': {
            'blas': _blas_metadata(),
            'cpu_count': os.cpu_count(),
            'machine': platform.machine(),
            'numpy': np.__version__,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python': sys.version,
        },
        'native_library': {
            'path': str(native_library),
            'sha256': native_digest,
        },
        'benchmark_script': {
            'path': str(Path(__file__).resolve()),
            'sha256': _sha256_file(Path(__file__).resolve()),
        },
        'provenance': provenance,
        'results': grouped,
        'summary': {
            'runtime_speedup': old_time / new_time,
            'tracemalloc_reduction_bytes': old_trace - new_trace,
            'tracemalloc_reduction_ratio': (
                1.0 - new_trace / old_trace if old_trace else 0.0),
        },
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(result['summary'], indent=2, sort_keys=True))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--source-root')
    parser.add_argument('--native-library', required=True)
    parser.add_argument('--native-sha256')
    parser.add_argument('--mode', choices=('runtime', 'memory'))
    parser.add_argument('--baseline-root')
    parser.add_argument('--candidate-root', default='.')
    parser.add_argument('--output', default=(
        'profiling/python_multinomial_rescale_results.json'))
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--inner-repeats', type=int, default=15)
    parser.add_argument('--nlambda', type=int, default=100)
    parser.add_argument('--classes', type=int, default=8)
    parser.add_argument('--d', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260719)
    return parser


def main():
    args = _parser().parse_args()
    if args.worker:
        _worker(args)
    else:
        if not args.baseline_root:
            raise ValueError('--baseline-root is required in controller mode.')
        _controller(args)


if __name__ == '__main__':
    main()
