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

# We can not import `picasso.libpath` in setup.py directly, since it will automatically import other package
# and case conflict to `install_requires`
libpath_py = os.path.join(CURRENT_DIR, 'pycasso/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)
LIB_PATH = [os.path.relpath(libfile, CURRENT_DIR) for libfile in libpath['find_lib_path'](False)]
if not LIB_PATH:
    LIB_PATH = [os.path.relpath(libfile, CURRENT_DIR) for libfile in libpath['find_lib_path'](True)]
    print("Install libpicasso from: %s" % LIB_PATH)
    # copy LIB_PATH to '.\\lib'
    for oldlibpath in LIB_PATH:
        newlibpath = os.path.join('.\\lib', os.path.dirname(oldlibpath))
        os.makedirs(newlibpath, exist_ok=True)
        shutil.copy(oldlibpath, newlibpath)
else:
    print("libpicasso already exists: " % LIB_PATH)

setup(name='pycasso',
      version=open(os.path.join(CURRENT_DIR, 'pycasso/VERSION')).read().strip(),
      # version='0.4a23',
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
      license='MIT',
      classifiers=['Development Status :: 3 - Alpha',
                   'Programming Language :: Python :: 3 :: Only',
                   'License :: OSI Approved :: MIT License'],
      url='https://github.com/jasonge27/picasso')
