# pylint: disable=invalid-name, exec-used
"""Setup picasso package."""
from __future__ import absolute_import
import sys
import os
import shlex
import shutil
import subprocess
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
    """Remove stale binaries before attempting a fresh native build."""
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if (filename.endswith(libpath['ALL_NATIVE_LIBRARY_SUFFIXES']) and
                (os.path.isfile(path) or os.path.islink(path))):
            os.unlink(path)


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


native_destination = os.path.join(CURRENT_DIR, 'pycasso', 'lib')
_clean_native_libraries(native_destination)

if os.name == 'nt':
    raise RuntimeError(
        'The standalone sdist native build supports Unix-like platforms. '
        'On Windows, build picasso.dll with CMake and install the repository '
        'checkout with setup.py and PICASSO_NATIVE_LIBRARY.')

native_source_dir = os.path.join(CURRENT_DIR, 'pycasso', 'src')
if not os.path.isfile(os.path.join(native_source_dir, 'Makefile')):
    raise RuntimeError(
        'The standalone package is missing pycasso/src/Makefile; run '
        '`make pippack` from the repository root first.')

make_command = shlex.split(os.environ.get('MAKE', 'make'))
if not make_command:
    raise RuntimeError('MAKE must name a usable make command.')

# check_call propagates compiler and linker failures to pip. The destination
# was already cleared, so a failed build cannot fall back to a bundled binary.
subprocess.check_call(make_command + ['clean_all'], cwd=native_source_dir)
subprocess.check_call(make_command + ['dylib'], cwd=native_source_dir)

native_build_dir = os.path.join(native_source_dir, 'lib')
supported_names = libpath['native_library_filenames']()
native_candidates = [
    os.path.join(native_build_dir, filename) for filename in supported_names
    if os.path.isfile(os.path.join(native_build_dir, filename))
]
if len(native_candidates) != 1:
    detail = '\n'.join(native_candidates) if native_candidates else '(none found)'
    raise RuntimeError(
        'Native build produced %d platform libraries; expected exactly one in '
        '%s:\n%s' %
        (len(native_candidates), native_build_dir, detail))
shutil.copy2(native_candidates[0], native_destination)

LIB_PATH = [os.path.relpath(
    _packaged_native_library(native_destination), CURRENT_DIR)]
print("Built Picasso native library: %s" % LIB_PATH[0])


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
      python_requires='>=3.9',
      maintainer='Tuo Zhao',
      maintainer_email='tourzhao@gatech.edu',
      zip_safe=False,
      packages=find_packages(),
      package_data={
          'pycasso': [
              'VERSION', 'lib/*.so', 'lib/*.dylib', 'lib/*.dll',
          ],
      },
      # MANIFEST.in carries native sources in the sdist; the wheel needs only
      # Python modules, VERSION, and the freshly compiled platform library.
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
