# coding: utf-8
"""Find the path to picasso dynamic library files."""

import os
import sys


class PicassoLibraryNotFound(Exception):
    """Error thrown by when picasso is not found"""
    pass


ALL_NATIVE_LIBRARY_SUFFIXES = ('.so', '.dylib', '.dll')


def native_library_filenames(platform_name=None):
    """Return supported native-library names for one host platform."""
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name == 'win32':
        return ('picasso.dll', 'libpicasso.dll', 'libpicasso.so')
    if platform_name.startswith('linux'):
        return ('libpicasso.so',)
    if platform_name == 'darwin':
        # The root Makefile historically emits a Mach-O library with a .so
        # suffix, while CMake uses the conventional .dylib suffix.
        return ('libpicasso.dylib', 'libpicasso.so')
    raise PicassoLibraryNotFound(
        'Unsupported platform for the Picasso native library: %s' %
        platform_name)


def find_lib_path():
    """Find the path to picasso dynamic library files.

    ``PICASSO_NATIVE_LIBRARY`` takes precedence over package-local discovery
    and may name a library built outside the Python package. The override is
    expanded to an absolute path and must identify an existing regular file.
    Without the override, exactly one platform-compatible bundled library must
    still be present; refusing ambiguity prevents an older .so from silently
    taking precedence over a newly built .dylib (or vice versa).

    :return: A one-element list containing the native library path.
    :rtype: list(string)
    """
    if 'PICASSO_NATIVE_LIBRARY' in os.environ:
        explicit = os.path.abspath(os.path.expanduser(
            os.environ['PICASSO_NATIVE_LIBRARY']))
        if not os.path.exists(explicit):
            raise PicassoLibraryNotFound(
                'PICASSO_NATIVE_LIBRARY does not exist: %s' % explicit)
        if not os.path.isfile(explicit):
            raise PicassoLibraryNotFound(
                'PICASSO_NATIVE_LIBRARY must name a file: %s' % explicit)
        return [explicit]

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    library_dir = os.path.join(curr_path, 'lib')
    candidates = [
        os.path.join(library_dir, filename)
        for filename in native_library_filenames()
    ]
    lib_path = [
        path for path in candidates
        if os.path.isfile(path)
    ]

    if not lib_path:
        if os.environ.get('PICASSO_BUILD_DOC', False):
            return []
        raise PicassoLibraryNotFound(
            'Cannot find the Picasso native library. Expected exactly one of:\n' +
            '\n'.join(candidates))
    if len(lib_path) != 1:
        raise PicassoLibraryNotFound(
            'Multiple Picasso native libraries were found. Remove stale '
            'artifacts so exactly one platform library remains:\n' +
            '\n'.join(lib_path))

    return lib_path
