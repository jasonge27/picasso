Picasso Python Package
======================
PICASSO: Penalized Generalized Linear Model Solver - Unleash the Power of Non-convex Penalty

Installation
------------

Install from source file (Github):

- Clone ``picasso.git`` via ``git clone https://github.com/jasonge27/picasso.git``
- Build the source file first via the ``cmake`` with ``CMakeLists.txt`` in the root directory.
  (You will see a lib file under ``(root)/lib/`` )
-  Make sure you have
   `setuptools <https://pypi.python.org/pypi/setuptools>`__
-  Install with ``cd python-package; python setup.py install`` command from this directory.
-  **Note**: If you are installing in this way, make sure `python-package/lib` is deleted before installing.

Install from PyPI:

- ``pip install pycasso``

You can test if the package has been successfully installed by:

.. code-block:: python

        import pycasso
        picasso.test()

..

Usage
-----

.. code-block:: python

        import pycasso
        picasso.test()

..

