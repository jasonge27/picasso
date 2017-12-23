Picasso Python Package
======================
PICASSO: Penalized Generalized Linear Model Solver - Unleash the Power of Non-convex Penalty

Unleash the power of nonconvex penalty
--------------------------------------
L1 penalized regression (LASSO) is great for feature selection. However when you use LASSO in
very noisy setting, especially when some columns in your data have strong colinearity, LASSO
tends to give biased estimator due to the penalty term. As demonstrated in the example below,
the lowest estimation error among all the lambdas computed is as high as **16.41%**.

Requirements
------------
- Python3
- Linux or MacOS


Installation
------------

Install from source file (Github):

- Clone ``picasso.git`` via ``git clone https://github.com/jasonge27/picasso.git``
- Make sure you have `setuptools <https://pypi.python.org/pypi/setuptools>`__

  Using **Makefile**
- Run ``sudo make Pyinstall`` command.

  Using **CMAKE**
- Build the source file first via the ``cmake`` with ``CMakeLists.txt`` in the root directory.
  (You will see a ``.so`` or ``.lib`` file under ``(root)/lib/`` )
- Run ``cd python-package; sudo python setup.py install`` command.


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

        from pycasso import *
        x = [[1,2,3,4,5,0],[3,4,1,7,0,1],[5,6,2,1,4,0]]
        y = [3.1,6.9,11.3]
        s = core.Solver(x,y)
        s.train()
        s.predict()

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
