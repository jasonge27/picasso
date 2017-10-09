Picasso Python Package
======================
PICASSO: Penalized Generalized Linear Model Solver - Unleash the Power of Non-convex Penalty

Unleash the power of nonconvex penalty
--------------------------------------
L1 penalized regression (LASSO) is great for feature selection. However when you use LASSO in
very noisy setting, especially when some columns in your data have strong colinearity, LASSO
tends to give biased estimator due to the penalty term. As demonstrated in the example below,
the lowest estimation error among all the lambdas computed is as high as **16.41%**.



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
- **Note**: Owing to the setting on different OS, our binary distribution might not be working in your environment. Thus please build from source.

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

For Developer
-------------
Please follow the `sphinx syntax style
<https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html>`__

To update the document: ``cd doc; make html``

Copy Right
----------

:Author: Jason(Jian) Ge, Haoming Jiang
:Maintainer: Haoming Jiang <jianghm@gatech.edu>
