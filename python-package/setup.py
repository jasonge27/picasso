# pylint: disable=invalid-name, exec-used
"""Setup picasso package."""
from __future__ import absolute_import
import sys
import os
import shutil
from setuptools import setup, find_packages
sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

#try to copy the complied lib files
libdir_candidate = [os.path.join(CURRENT_DIR, '../lib/')]

if sys.platform == 'win32':
    libcand_path = [os.path.join(p, 'picasso.dll') for p in libdir_candidate]
    libcand_path = libcand_path + [os.path.join(p, 'libpicasso.so') for p in libdir_candidate]
elif sys.platform.startswith('linux'):
    libcand_path = [os.path.join(p, 'libpicasso.so') for p in libdir_candidate]
elif sys.platform == 'darwin':
    libcand_path = [os.path.join(p, 'libpicasso.so') for p in libdir_candidate]
    libcand_path = libcand_path + [os.path.join(p, 'libpicasso.dylib') for p in libdir_candidate]

lib_path = [p for p in libcand_path if os.path.exists(p) and os.path.isfile(p)]
for lib_file in lib_path:
    shutil.copy(lib_file,os.path.join(CURRENT_DIR, './pycasso/lib/'))


# We can not import `picasso.libpath` in setup.py directly, since it will automatically import other package
# and case conflict to `install_requires`
libpath_py = os.path.join(CURRENT_DIR, 'pycasso/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)
LIB_PATH = [os.path.relpath(libfile, CURRENT_DIR) for libfile in libpath['find_lib_path']()]
if not LIB_PATH:
    raise RuntimeError("libpicasso does not exists")
else:
    print("libpicasso already exists: %s" % LIB_PATH)

VERSION_PATH = os.path.join(CURRENT_DIR, 'pycasso/VERSION')

setup(name='pycasso',
      version=open(VERSION_PATH).read().strip(),
      description="Picasso Python Package",
      long_description=open(os.path.join(CURRENT_DIR, 'README.rst')).read(),
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Tuo Zhao',
      maintainer_email='tourzhao@gatech.edu',
      zip_safe=False,
      packages=find_packages(),
      include_package_data=True,
      license='GPL-3.0',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Programming Language :: Python :: 3 :: Only',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],
      url='https://github.com/jasonge27/picasso')
