#!/usr/bin/env python3
"""Benchmark Python scalar-path finalization with an isolated fake solver.

The controller stages the baseline and candidate Python packages separately,
injects byte-identical copies of one native library, and launches a fresh
worker for every measurement.  Each worker verifies the library actually
loaded by ``ctypes`` and then replaces the Gaussian V2 entry point with a
Python fake.  The fake page-touches the complete native coefficient output but
does no optimization, so train time and memory isolate Python result reset,
finalization, and Gaussian post-processing.
"""

from __future__ import print_function

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - Windows-only fallback
    resource = None

import numpy as np


MODES = ('full_standardized', 'partial_unstandardized')
SOURCE_FILES = (
    'python-package/pycasso/core.py',
    'python-package/pycasso/libpath.py',
    'python-package/pycasso/__init__.py',
    'python-package/pycasso/VERSION',
)
CHECKSUM_FIELDS = (
    'beta', 'intercept', 'ite_lamb', 'size_act', 'df',
    'smooth_objective', 'dev_ratio',
)


def _sha256_file(path):
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _compatible_native_name():
    """Return the one package-local native filename used in a stage."""
    if sys.platform == 'win32':
        return 'picasso.dll'
    if sys.platform == 'darwin' or sys.platform.startswith('linux'):
        # Both current and historical package loaders accept this name.  The
        # dynamic loader detects the binary format rather than its suffix.
        return 'libpicasso.so'
    raise RuntimeError(
        'The scalar-finalization benchmark does not support platform %r.' %
        sys.platform)


def _source_provenance(source_root):
    """Validate and hash the original files that define one Python wrapper."""
    source_root = Path(source_root).expanduser().resolve()
    files = {}
    missing = []
    for relative_name in SOURCE_FILES:
        source_file = source_root / relative_name
        if not source_file.is_file():
            missing.append(str(source_file))
        else:
            files[relative_name] = {
                'path': str(source_file),
                'sha256': _sha256_file(source_file),
            }
    if missing:
        raise ValueError(
            'Python source root is missing required benchmark file(s):\n' +
            '\n'.join(missing))
    return files


def _stage_source(source_root, staging_root, label, native_library,
                  native_digest):
    """Stage Python sources without stale artifacts and inject one library."""
    source_root = Path(source_root).expanduser().resolve()
    staging_root = Path(staging_root).expanduser().resolve()
    source_package = source_root / 'python-package' / 'pycasso'
    staged_root = staging_root / label
    staged_package = staged_root / 'python-package' / 'pycasso'
    staged_package.parent.mkdir(parents=True)
    shutil.copytree(
        source_package, staged_package,
        ignore=shutil.ignore_patterns(
            'lib', 'src', '__pycache__', '*.pyc'))

    staged_library_dir = staged_package / 'lib'
    staged_library_dir.mkdir()
    staged_library = staged_library_dir / _compatible_native_name()
    shutil.copy2(str(native_library), str(staged_library))

    staged_files = sorted(
        path for path in staged_library_dir.iterdir() if path.is_file())
    if staged_files != [staged_library]:
        raise RuntimeError(
            'Staging source %r produced unexpected native files: %r.' %
            (label, [str(path) for path in staged_files]))
    staged_digest = _sha256_file(staged_library)
    if staged_digest != native_digest:
        raise RuntimeError(
            'Staged native SHA-256 mismatch for source %r: expected %s, '
            'got %s.' % (label, native_digest, staged_digest))

    forbidden = []
    for root, directories, filenames in os.walk(str(staged_package)):
        for directory in directories:
            if directory in ('src', '__pycache__'):
                forbidden.append(os.path.join(root, directory))
        for filename in filenames:
            if filename.endswith('.pyc'):
                forbidden.append(os.path.join(root, filename))
    if forbidden:
        raise RuntimeError(
            'Staging source %r retained excluded artifact(s): %r.' %
            (label, forbidden))
    return staged_root, staged_library


def _windows_memory_bytes():
    """Return (current, peak) working-set bytes on Windows, if available."""
    if sys.platform != 'win32':
        return None, None
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb)
        if not succeeded:
            return None, None
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError):
        return None, None


def _darwin_current_rss_bytes():
    """Return current resident bytes through Mach task_info on macOS."""
    if sys.platform != 'darwin':
        return None
    try:
        class TimeValue(ctypes.Structure):
            _fields_ = [
                ('seconds', ctypes.c_int),
                ('microseconds', ctypes.c_int),
            ]

        class MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ('virtual_size', ctypes.c_uint64),
                ('resident_size', ctypes.c_uint64),
                ('resident_size_max', ctypes.c_uint64),
                ('user_time', TimeValue),
                ('system_time', TimeValue),
                ('policy', ctypes.c_int),
                ('suspend_count', ctypes.c_int),
            ]

        system = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        system.mach_task_self.restype = ctypes.c_uint
        system.task_info.argtypes = [
            ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        info = MachTaskBasicInfo()
        count = ctypes.c_uint(
            ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
        # MACH_TASK_BASIC_INFO is flavor 20 on supported macOS releases.
        status = system.task_info(
            system.mach_task_self(), 20, ctypes.byref(info),
            ctypes.byref(count))
        if status == 0:
            return int(info.resident_size)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _peak_rss_bytes():
    """Return process peak RSS bytes when the host exposes it."""
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == 'darwin' else peak * 1024)
    _, peak = _windows_memory_bytes()
    return peak


def _current_rss_bytes():
    """Return current RSS bytes through a portable host-specific fallback."""
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/self/statm', 'r') as status:
                resident_pages = int(status.read().split()[1])
            return resident_pages * int(os.sysconf('SC_PAGE_SIZE'))
        except (IOError, OSError, ValueError, IndexError):
            pass
    if sys.platform == 'win32':
        current, _ = _windows_memory_bytes()
        return current
    if sys.platform == 'darwin':
        current = _darwin_current_rss_bytes()
        if current is not None:
            return current
    if os.name == 'posix':
        try:
            output = subprocess.check_output(
                ['ps', '-o', 'rss=', '-p', str(os.getpid())],
                universal_newlines=True)
            return int(output.strip()) * 1024
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return None


def _page_size_bytes():
    try:
        return int(os.sysconf('SC_PAGE_SIZE'))
    except (AttributeError, OSError, ValueError):
        return 4096


class _FakeGaussianV2(object):
    """Callable Gaussian V2 stand-in that retains no output-array reference."""

    def __init__(self, fitted_lambdas):
        self.fitted_lambdas = int(fitted_lambdas)
        self.call_count = 0
        # ctypes decorators assign these fields on native functions.
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        self.call_count += 1
        beta = arguments[12]
        intercept = arguments[13]
        iterations = arguments[14]
        active_size = arguments[15]
        runtime = arguments[16]
        num_fit = arguments[17]
        smooth_objective = arguments[19]

        # Write at least one double on every memory page in the complete
        # requested coefficient path, including the portion beyond nfit.
        flat_beta = beta.reshape(-1)
        doubles_per_page = max(
            1, _page_size_bytes() // beta.dtype.itemsize)
        flat_beta[::doubles_per_page] = 0.25
        if flat_beta.size:
            flat_beta[-1] = 0.5

        intercept[:] = np.arange(intercept.size, dtype='double') * 1e-3
        iterations[:] = 1
        active_size[:] = 1
        runtime[:] = np.arange(runtime.size, dtype='double') * 1e-6
        smooth_objective[:] = 2.0
        num_fit[0] = self.fitted_lambdas
        # All ndarray references above are local and disappear on return.


def _result_checksum(solver):
    """Hash deterministic public fit outputs without copying beta."""
    missing = [field for field in CHECKSUM_FIELDS
               if field not in solver.result]
    if missing:
        raise RuntimeError(
            'Fake Gaussian result is missing required field(s): %s.' %
            ', '.join(missing))

    digest = hashlib.sha256()
    for field in CHECKSUM_FIELDS:
        values = np.ascontiguousarray(solver.result[field])
        digest.update(field.encode('ascii'))
        digest.update(values.dtype.str.encode('ascii'))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(values).cast('B'))
    lambdas = np.ascontiguousarray(solver.lambdas)
    digest.update(b'lambdas')
    digest.update(lambdas.dtype.str.encode('ascii'))
    digest.update(np.asarray(lambdas.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(lambdas).cast('B'))
    return digest.hexdigest()


def _beta_owner_record(beta):
    """Describe the logical array and its ultimate ndarray backing owner."""
    owner = beta
    seen = set()
    base_depth = 0
    while isinstance(getattr(owner, 'base', None), np.ndarray):
        if id(owner) in seen:
            raise RuntimeError('Detected a cycle in the beta ndarray base chain.')
        seen.add(id(owner))
        owner = owner.base
        base_depth += 1
    backing_bytes = int(owner.nbytes) if isinstance(owner, np.ndarray) else None
    return {
        'beta_backing_owner_bytes': backing_bytes,
        'beta_base_depth': base_depth,
        'beta_c_contiguous': bool(beta.flags.c_contiguous),
        'beta_has_base': beta.base is not None,
        'beta_logical_bytes': int(beta.nbytes),
        'beta_owner_is_result': owner is beta,
        'beta_owner_owns_data': bool(owner.flags.owndata),
        'beta_owns_data': bool(beta.flags.owndata),
    }


def _worker(args):
    """Run one fake-native fit in a fresh process and emit one JSON record."""
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
            'Worker source=%r, mode=%r could not determine the loaded native '
            'library path.' % (args.label, args.mode))
    loaded_library = Path(loaded_name).expanduser().resolve()
    if loaded_library != expected_library:
        raise RuntimeError(
            'Worker source=%r, mode=%r loaded %s; expected %s.' %
            (args.label, args.mode, loaded_library, expected_library))
    loaded_digest = _sha256_file(loaded_library)
    if loaded_digest != args.native_sha256:
        raise RuntimeError(
            'Worker source=%r, mode=%r loaded native SHA-256 %s; expected '
            '%s.' % (args.label, args.mode, loaded_digest,
                     args.native_sha256))

    fitted_lambdas = (args.nlambda if args.mode == 'full_standardized'
                      else args.partial_nfit)
    fake = _FakeGaussianV2(fitted_lambdas)
    setattr(pycasso_core._PICASSO_LIB,
            'SolveLinearRegressionNaiveUpdateV2', fake)

    rng = np.random.RandomState(args.seed)
    x = rng.normal(size=(args.n, args.d))
    y = np.linspace(-1.0, 1.0, args.n, dtype='double')
    lambdas = np.linspace(1.0, 0.1, args.nlambda, dtype='double')
    standardize = args.mode == 'full_standardized'
    solver = pycasso.Solver(
        x, y, lambdas=lambdas, family='gaussian', penalty='l1',
        standardize=standardize, useintercept=True,
        type_gaussian='naive')

    start = time.perf_counter()
    solver.train()
    train_elapsed = time.perf_counter() - start
    current_rss = _current_rss_bytes()
    peak_rss = _peak_rss_bytes()
    if (current_rss is not None and peak_rss is not None and
            peak_rss < current_rss):
        raise RuntimeError(
            'Worker source=%r, mode=%r reported peak RSS %d below current '
            'RSS %d.' %
            (args.label, args.mode, peak_rss, current_rss))

    if fake.call_count != 1:
        raise RuntimeError(
            'Worker source=%r, mode=%r expected one fake Gaussian call; got '
            '%d.' % (args.label, args.mode, fake.call_count))
    retained_arrays = [
        name for name, value in fake.__dict__.items()
        if isinstance(value, np.ndarray)
    ]
    if retained_arrays:
        raise RuntimeError(
            'Fake Gaussian function retained output array field(s): %s.' %
            ', '.join(retained_arrays))
    if solver.nlambda != fitted_lambdas:
        raise RuntimeError(
            'Worker source=%r, mode=%r retained %d lambdas; expected %d.' %
            (args.label, args.mode, solver.nlambda, fitted_lambdas))

    record = {
        'checksum': _result_checksum(solver),
        'current_rss_bytes': current_rss,
        'fake_call_count': fake.call_count,
        'fitted_lambdas': fitted_lambdas,
        'loaded_library': str(loaded_library),
        'loaded_library_sha256': loaded_digest,
        'mode': args.mode,
        'peak_rss_bytes': peak_rss,
        'requested_lambdas': args.nlambda,
        'train_elapsed_seconds': train_elapsed,
    }
    record.update(_beta_owner_record(solver.result['beta']))
    print(json.dumps(record, sort_keys=True))


def _worker_command(args, label, staged_root, staged_library,
                    native_digest, mode):
    return [
        sys.executable, str(Path(__file__).resolve()), '--worker',
        '--label', label,
        '--source-root', str(staged_root),
        '--native-library', str(staged_library),
        '--native-sha256', native_digest,
        '--mode', mode,
        '--n', str(args.n),
        '--d', str(args.d),
        '--nlambda', str(args.nlambda),
        '--partial-nfit', str(args.partial_nfit),
        '--seed', str(args.seed),
    ]


def _run_worker(args, label, staged_root, staged_library, native_digest,
                mode):
    """Launch and validate one worker with label/mode-rich diagnostics."""
    command = _worker_command(
        args, label, staged_root, staged_library, native_digest, mode)
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    for thread_variable in (
            'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[thread_variable] = '1'
    completed = subprocess.run(
        command, check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(
            'Scalar-finalization worker failed for source=%r, mode=%r, '
            'returncode=%d.\nstdout:\n%s\nstderr:\n%s' %
            (label, mode, completed.returncode, completed.stdout,
             completed.stderr))
    output_lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if not output_lines:
        raise RuntimeError(
            'Scalar-finalization worker returned no JSON for source=%r, '
            'mode=%r.\nstderr:\n%s' %
            (label, mode, completed.stderr))
    try:
        record = json.loads(output_lines[-1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            'Scalar-finalization worker returned invalid JSON for source=%r, '
            'mode=%r.\nstdout:\n%s\nstderr:\n%s' %
            (label, mode, completed.stdout, completed.stderr)) from exc

    expected_library = str(Path(staged_library).expanduser().resolve())
    if (record.get('loaded_library') != expected_library or
            record.get('loaded_library_sha256') != native_digest):
        raise RuntimeError(
            'Scalar-finalization worker native mismatch for source=%r, '
            'mode=%r: path=%r, sha256=%r; expected path=%r, sha256=%r.\n'
            'stderr:\n%s' %
            (label, mode, record.get('loaded_library'),
             record.get('loaded_library_sha256'), expected_library,
             native_digest, completed.stderr))
    if record.get('mode') != mode:
        raise RuntimeError(
            'Scalar-finalization worker source=%r reported mode=%r; '
            'expected %r.' % (label, record.get('mode'), mode))
    return record


def _optional_median(records, field):
    values = [record[field] for record in records
              if record.get(field) is not None]
    return statistics.median(values) if values else None


def _summarize(records):
    """Summarize stable metadata and median process measurements."""
    return {
        'beta_backing_owner_bytes': records[0]['beta_backing_owner_bytes'],
        'beta_logical_bytes': records[0]['beta_logical_bytes'],
        'beta_owner_flags': {
            'base_depth': records[0]['beta_base_depth'],
            'c_contiguous': records[0]['beta_c_contiguous'],
            'has_base': records[0]['beta_has_base'],
            'owner_is_result': records[0]['beta_owner_is_result'],
            'owner_owns_data': records[0]['beta_owner_owns_data'],
            'owns_data': records[0]['beta_owns_data'],
        },
        'checksum': records[0]['checksum'],
        'median_current_rss_bytes': _optional_median(
            records, 'current_rss_bytes'),
        'median_peak_rss_bytes': _optional_median(records, 'peak_rss_bytes'),
        'median_train_elapsed_seconds': statistics.median(
            record['train_elapsed_seconds'] for record in records),
        'runs': records,
    }


def _validate_mode_records(records, label, mode):
    """Require deterministic checksums and invariant owner metadata."""
    checksum_values = set(record.get('checksum') for record in records)
    if len(checksum_values) != 1:
        raise RuntimeError(
            'Repeated checksums differ for source=%r, mode=%r: %r.' %
            (label, mode, sorted(checksum_values)))
    invariant_fields = (
        'beta_backing_owner_bytes', 'beta_base_depth', 'beta_c_contiguous',
        'beta_has_base', 'beta_logical_bytes', 'beta_owner_is_result',
        'beta_owner_owns_data', 'beta_owns_data', 'fitted_lambdas',
        'requested_lambdas',
    )
    for field in invariant_fields:
        values = set(record.get(field) for record in records)
        if len(values) != 1:
            raise RuntimeError(
                'Repeated field %r differs for source=%r, mode=%r: %r.' %
                (field, label, mode, sorted(values)))


def _controller(args):
    """Stage both wrappers, interleave fresh workers, and emit one report."""
    benchmark_script = Path(__file__).resolve()
    sources = (
        ('baseline', Path(args.baseline_root).expanduser().resolve()),
        ('candidate', Path(args.candidate_root).expanduser().resolve()),
    )
    native_library = Path(args.native_library).expanduser().resolve()
    native_digest = _sha256_file(native_library)
    source_files = {
        label: _source_provenance(source_root)
        for label, source_root in sources
    }
    records = {
        label: {mode: [] for mode in MODES}
        for label, _ in sources
    }

    with tempfile.TemporaryDirectory(
            prefix='picasso-scalar-finalize-benchmark-') as temporary:
        staged_sources = {
            label: _stage_source(
                source_root, temporary, label, native_library, native_digest)
            for label, source_root in sources
        }
        # Alternate source order between repetitions while keeping each mode
        # adjacent, reducing systematic timing drift without sharing a process.
        for repetition in range(args.repeats):
            source_order = (sources if repetition % 2 == 0
                            else tuple(reversed(sources)))
            for mode in MODES:
                for label, _ in source_order:
                    staged_root, staged_library = staged_sources[label]
                    records[label][mode].append(_run_worker(
                        args, label, staged_root, staged_library,
                        native_digest, mode))

    for label, source_records in records.items():
        for mode, mode_records in source_records.items():
            _validate_mode_records(mode_records, label, mode)
    for mode in MODES:
        checksums = {
            records[label][mode][0]['checksum'] for label, _ in sources
        }
        if len(checksums) != 1:
            details = {
                label: records[label][mode][0]['checksum']
                for label, _ in sources
            }
            raise RuntimeError(
                'Baseline/candidate checksums differ for mode=%r: %r.' %
                (mode, details))

    summaries = {
        label: {
            mode: _summarize(mode_records)
            for mode, mode_records in source_records.items()
        }
        for label, source_records in records.items()
    }
    payload = {
        'benchmark_script': {
            'path': str(benchmark_script),
            'sha256': _sha256_file(benchmark_script),
        },
        'config': {
            'coefficient_path_bytes': (
                args.d * args.nlambda * np.dtype('double').itemsize),
            'd': args.d,
            'modes': list(MODES),
            'n': args.n,
            'nlambda': args.nlambda,
            'partial_nfit': args.partial_nfit,
            'repeats': args.repeats,
            'seed': args.seed,
        },
        'environment': {
            'numpy_version': np.__version__,
            'platform': platform.platform(),
            'python_version': sys.version,
            'sys_platform': sys.platform,
        },
        'native_library': str(native_library),
        'native_library_sha256': native_digest,
        'source_files': source_files,
        'sources': {label: str(root) for label, root in sources},
        'summaries': summaries,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        with Path(args.output).expanduser().open('w', encoding='utf-8') as out:
            out.write(serialized)
    sys.stdout.write(serialized)


def _validate_arguments(args):
    """Reject invalid dimensions and paths before allocating fit buffers."""
    for name in ('repeats', 'n', 'd', 'nlambda', 'partial_nfit'):
        if getattr(args, name) <= 0:
            raise ValueError('"--%s" must be a positive integer.' %
                             name.replace('_', '-'))
    if args.n < 2:
        raise ValueError('"--n" must be at least 2 for standardization.')
    if args.partial_nfit >= args.nlambda:
        raise ValueError(
            '"--partial-nfit" must be smaller than "--nlambda".')
    native_limit = np.iinfo(np.int32).max
    for name in ('n', 'd', 'nlambda'):
        if getattr(args, name) > native_limit:
            raise ValueError(
                '"--%s" exceeds the native int32 limit.' % name)
    coefficient_elements = args.d * args.nlambda
    if coefficient_elements > (
            np.iinfo(np.intp).max // np.dtype('double').itemsize):
        raise ValueError(
            '"--d" times "--nlambda" is too large for a NumPy array.')

    native_library = Path(args.native_library).expanduser().resolve()
    if not native_library.is_file():
        raise ValueError(
            '"--native-library" must name an existing file: %s.' %
            native_library)
    if not args.worker and not args.baseline_root:
        raise ValueError('"--baseline-root" is required in controller mode.')
    roots = ([args.source_root] if args.worker else
             [args.baseline_root, args.candidate_root])
    for source_root in roots:
        if source_root is None:
            continue
        resolved_root = Path(source_root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise ValueError(
                'Python source root is not a directory: %s.' % source_root)
        _source_provenance(resolved_root)
    if args.worker:
        if not args.label:
            raise ValueError('"--label" is required in worker mode.')
        if not args.source_root:
            raise ValueError('"--source-root" is required in worker mode.')
        if not args.native_sha256:
            raise ValueError('"--native-sha256" is required in worker mode.')
    if args.output is not None:
        output_path = Path(args.output).expanduser()
        output_parent = output_path.parent.resolve()
        if not output_parent.is_dir():
            raise ValueError(
                'Output directory does not exist: %s.' % output_parent)


def _arguments():
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root')
    parser.add_argument('--candidate-root', default=str(repository_root))
    parser.add_argument('--native-library', required=True)
    parser.add_argument(
        '--output', help='Write the same JSON emitted on stdout to this file.')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--d', type=int, default=100000)
    parser.add_argument('--nlambda', type=int, default=100)
    parser.add_argument('--partial-nfit', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260719)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--label', help=argparse.SUPPRESS)
    parser.add_argument('--source-root', help=argparse.SUPPRESS)
    parser.add_argument('--native-sha256', help=argparse.SUPPRESS)
    parser.add_argument(
        '--mode', choices=MODES, default=MODES[0], help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    try:
        _validate_arguments(arguments)
    except ValueError as exc:
        parser.error(str(exc))
    return arguments


if __name__ == '__main__':
    parsed_arguments = _arguments()
    if parsed_arguments.worker:
        _worker(parsed_arguments)
    else:
        _controller(parsed_arguments)
