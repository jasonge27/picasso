#!/usr/bin/env python3
"""Measure Python multinomial output-buffer runtime and peak RSS.

The controller launches a fresh process for every measurement.  Pass the same
native library to a baseline and candidate source tree so the comparison
isolates Python wrapper changes rather than C++ build differences.
"""

import argparse
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


OUTPUT_FIELDS = (
    'beta', 'intercept', 'ite_lamb', 'size_act', 'train_time', 'num_fit',
    'outer_ite', 'inner_sweeps', 'coordinate_updates', 'objective', 'kkt',
    'stationarity', 'smooth_nll',
)
CHECKSUM_FIELDS = (
    'beta', 'intercept', 'ite_lamb', 'size_act', 'df', 'outer_ite',
    'inner_sweeps', 'coordinate_updates', 'objective', 'kkt',
    'stationarity', 'smooth_nll', 'dev_ratio',
)
SOURCE_FILES = (
    'python-package/pycasso/core.py',
    'python-package/pycasso/libpath.py',
    'python-package/pycasso/__init__.py',
    'python-package/pycasso/VERSION',
)


def _sha256_file(path):
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _compatible_native_name():
    """Return the single package-local filename used by staged workers."""
    if sys.platform == 'win32':
        return 'picasso.dll'
    if sys.platform == 'darwin' or sys.platform.startswith('linux'):
        return 'libpicasso.so'
    raise RuntimeError(
        'The output-buffer benchmark does not support native libraries on '
        f'platform {sys.platform!r}.')


def _source_provenance(source_root):
    """Validate and hash the original Python files used by one source tree."""
    source_root = Path(source_root).expanduser().resolve()
    files = {}
    missing = []
    for relative_name in SOURCE_FILES:
        source_file = source_root / relative_name
        if not source_file.is_file():
            missing.append(str(source_file))
            continue
        files[relative_name] = {
            'path': str(source_file),
            'sha256': _sha256_file(source_file),
        }
    if missing:
        raise ValueError(
            'Python source root is missing required benchmark file(s):\n' +
            '\n'.join(missing))
    return files


def _stage_source(source_root, staging_root, label, native_library):
    """Create an isolated runtime tree containing exactly one native library."""
    source_root = Path(source_root).expanduser().resolve()
    staging_root = Path(staging_root).expanduser().resolve()
    staged_root = staging_root / label
    source_package = source_root / 'python-package' / 'pycasso'
    staged_package = staged_root / 'python-package' / 'pycasso'
    staged_package.parent.mkdir(parents=True)
    shutil.copytree(
        source_package, staged_package,
        ignore=shutil.ignore_patterns(
            'lib', '__pycache__', 'src', '*.pyc'))

    staged_library_dir = staged_package / 'lib'
    staged_library_dir.mkdir()
    staged_library = staged_library_dir / _compatible_native_name()
    shutil.copy2(native_library, staged_library)

    packaged_libraries = [
        path for path in staged_library_dir.iterdir() if path.is_file()
    ]
    if packaged_libraries != [staged_library]:
        raise RuntimeError(
            f'Staging source {label!r} did not produce exactly one native '
            f'library: {[str(path) for path in packaged_libraries]}')
    source_digest = _sha256_file(native_library)
    staged_digest = _sha256_file(staged_library)
    if staged_digest != source_digest:
        raise RuntimeError(
            f'Staged native library hash mismatch for source {label!r}: '
            f'expected {source_digest}, got {staged_digest}.')
    return staged_root, staged_library


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def _touch_reset_buffers(solver):
    """Commit one byte per page without changing reset buffer contents."""
    reset = solver._reset_result_for_training

    def touched_reset():
        reset()
        for field in OUTPUT_FIELDS:
            values = solver.result.get(field)
            if values is None or values.nbytes == 0:
                continue
            byte_view = values.view(np.uint8).reshape(-1)
            page_bytes = byte_view[::4096]
            np.bitwise_xor(page_bytes, np.uint8(1), out=page_bytes)
            np.bitwise_xor(page_bytes, np.uint8(1), out=page_bytes)

    solver._reset_result_for_training = touched_reset


def _result_checksum(result):
    digest = hashlib.sha256()
    for field in CHECKSUM_FIELDS:
        values = np.ascontiguousarray(result[field])
        digest.update(field.encode('ascii'))
        digest.update(values.dtype.str.encode('ascii'))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def _require_result_fields(result):
    """Fail clearly when a wrapper/native ABI omits benchmark outputs."""
    required = set(OUTPUT_FIELDS) | set(CHECKSUM_FIELDS) | {'status_code'}
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(
            'Benchmark result is missing required output field(s): ' +
            ', '.join(missing))


def _worker(args):
    sys.dont_write_bytecode = True
    source_root = Path(args.source_root).expanduser().resolve()
    sys.path.insert(0, str(source_root / 'python-package'))
    expected_library = Path(args.native_library).expanduser().resolve()
    os.environ['PICASSO_NATIVE_LIBRARY'] = str(expected_library)
    import pycasso
    from pycasso import core as pycasso_core

    loaded_name = getattr(pycasso_core._PICASSO_LIB, '_name', None)
    if not loaded_name:
        raise RuntimeError(
            f'Worker source={args.label!r}, mode={args.mode!r} could not '
            'determine the loaded native-library path.')
    loaded_library = Path(loaded_name).expanduser().resolve()
    if loaded_library != expected_library:
        raise RuntimeError(
            f'Worker source={args.label!r}, mode={args.mode!r} loaded '
            f'{loaded_library}, expected {expected_library}.')
    loaded_digest = _sha256_file(loaded_library)
    if loaded_digest != args.native_sha256:
        raise RuntimeError(
            f'Worker source={args.label!r}, mode={args.mode!r} loaded native '
            f'SHA-256 {loaded_digest}, expected {args.native_sha256}.')

    rng = np.random.RandomState(args.seed)
    x = rng.normal(size=(args.n, args.d))
    y = np.arange(args.n) % args.classes
    class_probability = (
        np.bincount(y, minlength=args.classes).astype(float) / args.n)
    lambda_max = max(
        np.max(np.abs(
            x.T @ ((y == klass).astype(float) - class_probability[klass])
        )) / args.n
        for klass in range(args.classes))
    lambdas = np.linspace(
        1.5 * lambda_max, 1.05 * lambda_max, args.nlambda)
    solver = pycasso.Solver(
        x, y, lambdas=lambdas, family='multinomial', standardize=False,
        prec=1e-7, max_ite=1000)
    if args.mode == 'touch':
        _touch_reset_buffers(solver)

    start = time.perf_counter()
    solver.train()
    elapsed = time.perf_counter() - start
    # Capture peak before checksum construction so measurement code cannot
    # obscure the fit peak. The checksum itself uses the buffer protocol and
    # does not copy the coefficient path.
    peak_rss = _peak_rss_bytes()
    _require_result_fields(solver.result)
    output_bytes = sum(solver.result[field].nbytes for field in OUTPUT_FIELDS)
    record = {
        'checksum': _result_checksum(solver.result),
        'elapsed_seconds': elapsed,
        'loaded_library': str(loaded_library),
        'loaded_library_sha256': loaded_digest,
        'mode': args.mode,
        'nlambda': solver.nlambda,
        'output_bytes': output_bytes,
        'peak_rss_bytes': peak_rss,
        'status_code': solver.result['status_code'],
    }
    print(json.dumps(record, sort_keys=True))


def _worker_command(args, label, source_root, native_library, native_digest,
                    mode):
    return [
        sys.executable, str(Path(__file__).resolve()), '--worker',
        '--label', label,
        '--source-root', str(source_root),
        '--native-library', str(native_library),
        '--native-sha256', native_digest,
        '--mode', mode, '--n', str(args.n), '--d', str(args.d),
        '--classes', str(args.classes), '--nlambda', str(args.nlambda),
        '--seed', str(args.seed),
    ]


def _summarize(records):
    return {
        'checksum': records[0]['checksum'],
        'median_elapsed_seconds': statistics.median(
            record['elapsed_seconds'] for record in records),
        'median_peak_rss_bytes': statistics.median(
            record['peak_rss_bytes'] for record in records),
        'runs': records,
    }


def _run_worker(args, label, source_root, native_library, native_digest,
                mode):
    """Run and validate one isolated worker with actionable diagnostics."""
    command = _worker_command(
        args, label, source_root, native_library, native_digest, mode)
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Benchmark worker failed for source={label!r}, mode={mode!r}, '
            f'returncode={completed.returncode}.\n'
            f'stdout:\n{completed.stdout}\n'
            f'stderr:\n{completed.stderr}')
    output_lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if not output_lines:
        raise RuntimeError(
            f'Benchmark worker returned no JSON for source={label!r}, '
            f'mode={mode!r}.\nstderr:\n{completed.stderr}')
    try:
        record = json.loads(output_lines[-1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'Benchmark worker returned invalid JSON for source={label!r}, '
            f'mode={mode!r}.\nstdout:\n{completed.stdout}\n'
            f'stderr:\n{completed.stderr}') from exc

    expected_library = str(Path(native_library).expanduser().resolve())
    actual_library = record.get('loaded_library')
    actual_digest = record.get('loaded_library_sha256')
    if actual_library != expected_library or actual_digest != native_digest:
        raise RuntimeError(
            f'Benchmark worker native-library mismatch for source={label!r}, '
            f'mode={mode!r}: path={actual_library!r}, '
            f'sha256={actual_digest!r}; expected path={expected_library!r}, '
            f'sha256={native_digest!r}.\nstderr:\n{completed.stderr}')
    return record


def _controller(args):
    sources = []
    if args.baseline_root is not None:
        sources.append((
            'baseline', Path(args.baseline_root).expanduser().resolve()))
    sources.append((
        'candidate', Path(args.candidate_root).expanduser().resolve()))
    native_library = Path(args.native_library).expanduser().resolve()
    native_digest = _sha256_file(native_library)
    source_files = {
        label: _source_provenance(source_root)
        for label, source_root in sources
    }
    records = {
        label: {'normal': [], 'touch': []} for label, _ in sources
    }
    with tempfile.TemporaryDirectory(
            prefix='picasso-buffer-benchmark-') as temporary_directory:
        staged_sources = {
            label: _stage_source(
                source_root, temporary_directory, label, native_library)
            for label, source_root in sources
        }
        # Interleave source trees within each repetition to reduce timing
        # drift while reusing only immutable staged source trees.
        for _ in range(args.repeats):
            for mode in ('normal', 'touch'):
                for label, _ in sources:
                    staged_root, staged_library = staged_sources[label]
                    records[label][mode].append(_run_worker(
                        args, label, staged_root, staged_library,
                        native_digest, mode))

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
        checksum_details = {
            label: {
                mode: summary['checksum']
                for mode, summary in source_summary.items()
            }
            for label, source_summary in summaries.items()
        }
        raise RuntimeError(
            'Baseline/candidate or normal/touch output checksums differ: '
            f'{checksum_details}')
    payload = {
        'config': {
            'classes': args.classes,
            'd': args.d,
            'n': args.n,
            'nlambda': args.nlambda,
            'repeats': args.repeats,
            'seed': args.seed,
        },
        'native_library': str(native_library),
        'native_library_sha256': native_digest,
        'source_files': source_files,
        'sources': {label: str(root) for label, root in sources},
        'summaries': summaries,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        Path(args.output).expanduser().resolve().write_text(
            serialized, encoding='utf-8')
    sys.stdout.write(serialized)


def _validate_arguments(args):
    """Reject invalid benchmark dimensions before launching workers."""
    for name in ('repeats', 'n', 'd', 'nlambda'):
        if getattr(args, name) <= 0:
            raise ValueError(f'"--{name}" must be a positive integer.')
    if args.classes < 3:
        raise ValueError('"--classes" must be an integer of at least 3.')
    if args.n < args.classes:
        raise ValueError('"--n" must be greater than or equal to "--classes".')
    if args.worker and args.source_root is None:
        raise ValueError('"--source-root" is required in worker mode.')
    native_library = Path(args.native_library).expanduser().resolve()
    if not native_library.is_file():
        raise ValueError(
            '"--native-library" must name an existing file: '
            f'{native_library}')
    if args.worker and not getattr(args, 'label', None):
        raise ValueError('"--label" is required in worker mode.')
    if args.worker and not getattr(args, 'native_sha256', None):
        raise ValueError('"--native-sha256" is required in worker mode.')
    source_roots = ([args.source_root] if args.worker else
                    [args.candidate_root, args.baseline_root])
    for source_root in source_roots:
        if source_root is None:
            continue
        resolved_root = Path(source_root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise ValueError(
                f'Python source root is not a directory: {source_root}')
        _source_provenance(resolved_root)
    if args.output is not None:
        output_parent = Path(args.output).expanduser().resolve().parent
        if not output_parent.is_dir():
            raise ValueError(
                f'Output directory does not exist: {output_parent}')


def _arguments():
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root')
    parser.add_argument('--candidate-root', default=str(repository_root))
    parser.add_argument('--native-library', required=True)
    parser.add_argument(
        '--output', help='Write the same JSON emitted on stdout to this file.')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--d', type=int, default=30000)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--nlambda', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20260719)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--label', help=argparse.SUPPRESS)
    parser.add_argument('--source-root', help=argparse.SUPPRESS)
    parser.add_argument('--native-sha256', help=argparse.SUPPRESS)
    parser.add_argument(
        '--mode', choices=('normal', 'touch'), default='normal',
        help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    try:
        _validate_arguments(arguments)
    except ValueError as exc:
        parser.error(str(exc))
    return arguments


if __name__ == '__main__':
    arguments = _arguments()
    if arguments.worker:
        _worker(arguments)
    else:
        _controller(arguments)
