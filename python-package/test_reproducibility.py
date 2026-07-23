"""Focused tests for native-library selection and benchmark isolation."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
LIBPATH_FILE = Path(__file__).resolve().parent / 'pycasso' / 'libpath.py'
BENCHMARK_FILE = (
    REPOSITORY_ROOT / 'profiling' /
    'python_multinomial_output_buffer_benchmark.py')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


libpath = _load_module('picasso_test_libpath', LIBPATH_FILE)
benchmark = _load_module('picasso_test_buffer_benchmark', BENCHMARK_FILE)
native_library = os.environ.get('PICASSO_NATIVE_LIBRARY')
if native_library is None or not Path(native_library).expanduser().is_file():
    raise RuntimeError(
        'Set PICASSO_NATIVE_LIBRARY to a fresh native build before running '
        'test_reproducibility.py.')
native_library = str(Path(native_library).expanduser().resolve())


def _setup_keyword(path, keyword):
    """Read one literal setup() keyword without executing packaging code."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Name) and node.func.id == 'setup'
    ]
    assert len(calls) == 1, f'{path} must contain exactly one setup() call'
    values = [item.value for item in calls[0].keywords if item.arg == keyword]
    assert len(values) == 1, f'{path} must define setup({keyword}=...)'
    return ast.literal_eval(values[0])


# Keep both setup entry points and the canonical README aligned. Plotting is
# optional; the numerical wrapper itself requires only NumPy.
for setup_name in ('setup.py', 'setup-pip.py'):
    setup_path = Path(__file__).resolve().parent / setup_name
    assert _setup_keyword(setup_path, 'install_requires') == ['numpy'], \
        f'{setup_name} retained an unused runtime dependency'
    assert _setup_keyword(setup_path, 'extras_require') == {
        'plot': ['matplotlib']
    }, f'{setup_name} omitted the plotting extra'
assert not (Path(__file__).resolve().parent / 'data' / 'eyedata.npy').exists(), \
    'orphan pickle-backed eyedata.npy remains in the Python source package'
manifest = (Path(__file__).resolve().parent / 'MANIFEST.in').read_text(
    encoding='utf-8')
assert 'recursive-include data ' not in manifest, \
    'MANIFEST.in still packages the orphan top-level data directory'


# An explicit runtime library wins even when it has a nonstandard filename;
# missing paths and directories fail before ctypes attempts to load them.
saved_override = os.environ.get('PICASSO_NATIVE_LIBRARY')
try:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        explicit_library = temporary_path / 'custom-native-build.bin'
        explicit_library.write_bytes(b'test fixture')
        relative_library = os.path.relpath(explicit_library, Path.cwd())
        os.environ['PICASSO_NATIVE_LIBRARY'] = relative_library
        expected_library = os.path.abspath(os.path.expanduser(
            relative_library))
        assert libpath.find_lib_path() == [expected_library]

        missing_library = temporary_path / 'missing-library.dylib'
        os.environ['PICASSO_NATIVE_LIBRARY'] = str(missing_library)
        try:
            libpath.find_lib_path()
            raise AssertionError('missing native-library override was accepted')
        except libpath.PicassoLibraryNotFound as exc:
            assert 'PICASSO_NATIVE_LIBRARY' in str(exc) and \
                'does not exist' in str(exc)

        os.environ['PICASSO_NATIVE_LIBRARY'] = str(temporary_path)
        try:
            libpath.find_lib_path()
            raise AssertionError('native-library directory was accepted')
        except libpath.PicassoLibraryNotFound as exc:
            assert 'PICASSO_NATIVE_LIBRARY' in str(exc) and \
                'must name a file' in str(exc)
finally:
    if saved_override is None:
        os.environ.pop('PICASSO_NATIVE_LIBRARY', None)
    else:
        os.environ['PICASSO_NATIVE_LIBRARY'] = saved_override

# Without an override, exercise zero, one, and two synthetic bundled
# libraries. This test deliberately does not depend on repository artifacts.
saved_libpath_file = libpath.__file__
saved_native_filenames = libpath.native_library_filenames
saved_doc_mode = os.environ.get('PICASSO_BUILD_DOC')
try:
    os.environ.pop('PICASSO_NATIVE_LIBRARY', None)
    os.environ.pop('PICASSO_BUILD_DOC', None)
    with tempfile.TemporaryDirectory() as temporary_directory:
        synthetic_package = Path(temporary_directory) / 'pycasso'
        synthetic_package.mkdir()
        libpath.__file__ = str(synthetic_package / 'libpath.py')
        libpath.native_library_filenames = (
            lambda platform_name=None: ('first.so', 'second.so'))
        first_library = synthetic_package / 'lib' / 'first.so'
        second_library = synthetic_package / 'lib' / 'second.so'

        try:
            libpath.find_lib_path()
            raise AssertionError('zero bundled libraries were accepted')
        except libpath.PicassoLibraryNotFound as exc:
            assert str(first_library) in str(exc)
            assert str(second_library) in str(exc)

        os.environ['PICASSO_BUILD_DOC'] = '1'
        assert libpath.find_lib_path() == []
        os.environ.pop('PICASSO_BUILD_DOC', None)

        first_library.parent.mkdir()
        first_library.write_bytes(b'first synthetic library')
        assert libpath.find_lib_path() == [str(first_library)]

        second_library.write_bytes(b'second synthetic library')
        try:
            libpath.find_lib_path()
            raise AssertionError('two bundled libraries were accepted')
        except libpath.PicassoLibraryNotFound as exc:
            assert 'Multiple Picasso native libraries' in str(exc)
            assert str(first_library) in str(exc)
            assert str(second_library) in str(exc)
finally:
    libpath.__file__ = saved_libpath_file
    libpath.native_library_filenames = saved_native_filenames
    if saved_override is not None:
        os.environ['PICASSO_NATIVE_LIBRARY'] = saved_override
    if saved_doc_mode is None:
        os.environ.pop('PICASSO_BUILD_DOC', None)
    else:
        os.environ['PICASSO_BUILD_DOC'] = saved_doc_mode


# Validate the controller dimensions before it launches any subprocess.
valid_arguments = argparse.Namespace(
    repeats=1, n=3, d=1, nlambda=1, classes=3, worker=False,
    source_root=None, candidate_root=str(REPOSITORY_ROOT),
    baseline_root=None, native_library=native_library, output=None)
benchmark._validate_arguments(valid_arguments)
for field, invalid_value, expected_message in (
        ('repeats', 0, '--repeats'),
        ('n', 0, '--n'),
        ('d', 0, '--d'),
        ('nlambda', 0, '--nlambda'),
        ('classes', 2, '--classes')):
    invalid_arguments = argparse.Namespace(**vars(valid_arguments))
    setattr(invalid_arguments, field, invalid_value)
    try:
        benchmark._validate_arguments(invalid_arguments)
        raise AssertionError(f'invalid benchmark {field} was accepted')
    except ValueError as exc:
        assert expected_message in str(exc)
narrow_arguments = argparse.Namespace(**vars(valid_arguments))
narrow_arguments.n = 2
narrow_arguments.classes = 3
try:
    benchmark._validate_arguments(narrow_arguments)
    raise AssertionError('benchmark accepted n < classes')
except ValueError as exc:
    assert '--n' in str(exc) and '--classes' in str(exc)

# Missing native outputs must produce an ABI-oriented error instead of an
# incidental KeyError from checksum or byte-count construction.
try:
    benchmark._require_result_fields({})
    raise AssertionError('missing benchmark outputs were accepted')
except RuntimeError as exc:
    assert 'missing required output field(s)' in str(exc)
    assert 'beta' in str(exc) and 'status_code' in str(exc)


# Simulate a historical loader that ignores PICASSO_NATIVE_LIBRARY and ships a
# stale package-local binary. The controller must discard that binary, retain
# the old loader source, and inject only the exact requested native build.
with tempfile.TemporaryDirectory() as temporary_directory:
    temporary_path = Path(temporary_directory)
    old_source_root = temporary_path / 'old-source'
    old_package = old_source_root / 'python-package' / 'pycasso'
    old_package.mkdir(parents=True)
    package_source = Path(__file__).resolve().parent / 'pycasso'
    for filename in ('__init__.py', 'core.py', 'VERSION'):
        shutil.copy2(package_source / filename, old_package / filename)
    compatible_name = benchmark._compatible_native_name()
    old_loader = old_package / 'libpath.py'
    old_loader.write_text(
        "import os\n\n"
        "def find_lib_path():\n"
        "    return [os.path.join(os.path.dirname(__file__), 'lib', "
        f"'{compatible_name}')]\n",
        encoding='utf-8')
    stale_library = old_package / 'lib' / compatible_name
    stale_library.parent.mkdir()
    stale_library.write_bytes(b'deliberately stale native fixture')
    (old_package / 'src').mkdir()
    (old_package / 'src' / 'stale.cpp').write_text(
        'not part of a runtime stage\n', encoding='utf-8')
    (old_package / '__pycache__').mkdir()
    (old_package / '__pycache__' / 'stale.pyc').write_bytes(b'stale bytecode')
    (old_package / 'orphan.pyc').write_bytes(b'orphan bytecode')

    inspection_root = temporary_path / 'inspection'
    staged_root, staged_library = benchmark._stage_source(
        old_source_root, inspection_root, 'candidate', native_library)
    staged_package = staged_root / 'python-package' / 'pycasso'
    assert not (staged_package / 'src').exists()
    assert not (staged_package / '__pycache__').exists()
    assert not (staged_package / 'orphan.pyc').exists()
    assert [path.name for path in (staged_package / 'lib').iterdir()] == [
        compatible_name]
    assert benchmark._sha256_file(staged_library) == \
        benchmark._sha256_file(native_library)
    assert benchmark._sha256_file(staged_library) != \
        hashlib.sha256(stale_library.read_bytes()).hexdigest()

    output_path = temporary_path / 'benchmark.json'
    completed = subprocess.run([
        sys.executable, str(BENCHMARK_FILE),
        '--candidate-root', str(old_source_root),
        '--native-library', native_library,
        '--repeats', '1', '--n', '9', '--d', '4', '--classes', '3',
        '--nlambda', '2', '--output', str(output_path),
    ], check=True, text=True, capture_output=True)
    assert completed.stdout == output_path.read_text(encoding='utf-8')
    payload = json.loads(completed.stdout)
    assert set(payload['summaries']['candidate']) == {'normal', 'touch'}
    checksums = {
        payload['summaries']['candidate'][mode]['checksum']
        for mode in ('normal', 'touch')
    }
    assert len(checksums) == 1
    assert payload['native_library'] == native_library
    native_digest = benchmark._sha256_file(native_library)
    for mode in ('normal', 'touch'):
        run = payload['summaries']['candidate'][mode]['runs'][0]
        assert Path(run['loaded_library']).name == compatible_name
        assert run['loaded_library_sha256'] == native_digest
        assert str(old_source_root) not in run['loaded_library']
    old_loader_provenance = payload['source_files']['candidate'][
        'python-package/pycasso/libpath.py']
    assert old_loader_provenance['path'] == str(old_loader.resolve())
    assert old_loader_provenance['sha256'] == \
        benchmark._sha256_file(old_loader)

print('Native-library and benchmark reproducibility tests passed.')
