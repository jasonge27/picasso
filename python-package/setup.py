# pylint: disable=invalid-name, exec-used
"""Setup picasso package."""
from __future__ import absolute_import
import sys
import os
import shutil
import tempfile
from setuptools import setup, find_packages
from setuptools.dist import Distribution
sys.path.insert(0, '.')

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# We cannot import pycasso.libpath here because importing the package also loads
# the native library. Execute this dependency-free helper module directly.
libpath_py = os.path.join(CURRENT_DIR, 'pycasso/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)


def _clean_native_libraries(directory):
    """Remove every stale platform library from the package destination."""
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if (filename.endswith(libpath['ALL_NATIVE_LIBRARY_SUFFIXES']) and
                (os.path.isfile(path) or os.path.islink(path))):
            os.unlink(path)


def _select_native_library():
    """Select one explicit or unambiguous library produced outside the package."""
    supported_names = libpath['native_library_filenames']()
    explicit = os.environ.get('PICASSO_NATIVE_LIBRARY')
    if explicit:
        candidate = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(candidate):
            raise RuntimeError(
                'PICASSO_NATIVE_LIBRARY is not a file: %s' % candidate)
        if os.path.basename(candidate) not in supported_names:
            raise RuntimeError(
                'PICASSO_NATIVE_LIBRARY has an unsupported platform name: %s; '
                'expected one of %s' %
                (os.path.basename(candidate), ', '.join(supported_names)))
        return candidate

    root_lib = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'lib'))
    candidates = [
        os.path.join(root_lib, filename) for filename in supported_names
        if os.path.isfile(os.path.join(root_lib, filename))
    ]
    if len(candidates) != 1:
        detail = '\n'.join(candidates) if candidates else '(none found)'
        raise RuntimeError(
            'Expected exactly one platform-native Picasso library in %s, '
            'but found:\n%s\nSet PICASSO_NATIVE_LIBRARY to the library '
            'staged by the build you intend to package.' % (root_lib, detail))
    return candidates[0]


def _packaged_native_library(directory):
    """Return the one platform library actually present in the package."""
    candidates = [
        os.path.join(directory, filename)
        for filename in libpath['native_library_filenames']()
        if os.path.isfile(os.path.join(directory, filename))
    ]
    if len(candidates) != 1:
        detail = '\n'.join(candidates) if candidates else '(none found)'
        raise RuntimeError(
            'Expected exactly one copied platform-native Picasso library in '
            '%s, but found:\n%s' % (directory, detail))
    return candidates[0]


native_source = _select_native_library()
native_destination = os.path.join(CURRENT_DIR, 'pycasso', 'lib')
temporary_source = None
source_to_copy = native_source
if os.path.realpath(os.path.dirname(native_source)) == os.path.realpath(
        native_destination):
    # Preserve the explicitly selected file while the destination is cleared.
    handle, temporary_source = tempfile.mkstemp(
        suffix=os.path.splitext(native_source)[1])
    os.close(handle)
    shutil.copy2(native_source, temporary_source)
    source_to_copy = temporary_source
try:
    _clean_native_libraries(native_destination)
    shutil.copy2(
        source_to_copy,
        os.path.join(native_destination, os.path.basename(native_source)))
finally:
    if temporary_source is not None and os.path.exists(temporary_source):
        os.unlink(temporary_source)

LIB_PATH = [os.path.relpath(
    _packaged_native_library(native_destination), CURRENT_DIR)]
print("Using Picasso native library: %s" % LIB_PATH[0])


class BinaryDistribution(Distribution):
    """Mark wheels as platform-specific because they contain a ctypes library."""

    def has_ext_modules(self):
        return True

VERSION_PATH = os.path.join(CURRENT_DIR, 'pycasso/VERSION')

setup(name='pycasso',
      version=open(VERSION_PATH).read().strip(),
      description="Picasso Python Package",
      long_description=open(os.path.join(CURRENT_DIR, 'README.rst')).read(),
      long_description_content_type='text/x-rst',
      install_requires=[
          'numpy',
      ],
      extras_require={
          'plot': ['matplotlib'],
      },
      python_requires='>=3.6',
      maintainer='Tuo Zhao',
      maintainer_email='tourzhao@gatech.edu',
      zip_safe=False,
      packages=find_packages(),
      package_data={
          'pycasso': [
              'VERSION', 'lib/*.so', 'lib/*.dylib', 'lib/*.dll',
          ],
      },
      # Keep the wheel small and explicit: C++ sources belong only to the
      # standalone sdist assembled by setup-pip.py.
      include_package_data=False,
      distclass=BinaryDistribution,
      license='GPL-3.0',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Programming Language :: Python :: 3 :: Only',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],
      url='https://github.com/tourzhao/picasso')
