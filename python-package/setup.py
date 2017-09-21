# pylint: disable=invalid-name, exec-used
"""Setup picasso package."""
from __future__ import absolute_import
import sys
import os
from setuptools import setup, find_packages
# import subprocess
sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

libpath_py = os.path.join(CURRENT_DIR, 'picasso/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

LIB_PATH = [os.path.relpath(libfile, CURRENT_DIR) for libfile in libpath['find_lib_path']()]

LIB_PATH = [os.path.relpath(libfile, CURRENT_DIR) for libfile in libpath['find_lib_path']()]
print("Install libpicasso from: %s" % LIB_PATH)
# Please use setup_pip.py for generating and deploying pip installation
# detailed instruction in setup_pip.py

setup(name='picasso',
      version=open(os.path.join(CURRENT_DIR, 'picasso/VERSION')).read().strip(),
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
      data_files=[('picasso', LIB_PATH)],
      license='MIT',
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: MIT Software License'],
      url='https://github.com/jasonge27/picasso')
