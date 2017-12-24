# pylint: disable=invalid-name, exec-used
"""Setup picasso package."""
from __future__ import absolute_import
import sys
import os
import shutil
from setuptools import setup, find_packages
# import subprocess
sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

# complie c code
if os.name != 'nt':
    os.system('cd pycasso/src/; make clean; make dylib; cd ../../;')
else:
    raise RuntimeError('Windows users please install from source')


#try to copy the complied lib files
libdir_candidate = [os.path.join(CURRENT_DIR, './pycasso/src/lib/')]

if sys.platform == 'win32':
    libcand_path = [os.path.join(p, 'picasso.dll') for p in libdir_candidate]
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
      maintainer='Haoming Jiang',
      maintainer_email='jianghm.ustc@gmail.com',
      zip_safe=False,
      packages=find_packages(),
      # this will use MANIFEST.in during install where we specify additional files,
      # this is the golden line
      include_package_data=True,
      # data_files=[('pycasso',LIB_PATH)],
      license='MIT',
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Programming Language :: Python :: 3 :: Only',
                   'License :: OSI Approved :: MIT License'],
      url='https://hmjianggatech.github.io/picasso/')
