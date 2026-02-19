Pycasso Python Package
======================

``pycasso`` is the Python wrapper of PICASSO for sparse learning with
L1/MCP/SCAD penalties. It supports four model families:

- ``gaussian`` (linear regression)
- ``binomial`` (logistic regression)
- ``poisson`` (count regression)
- ``sqrtlasso`` (square-root lasso)

Requirements
------------

- Linux or macOS is recommended.
- Python with ``numpy`` and ``scipy``.

Windows users can still build from source, but toolchain setup may require
extra work (for example, MinGW/MinGW-w64).

Installation
------------

Install from PyPI (recommended):

.. code-block:: bash

   pip install pycasso

Install from source (repository root):

.. code-block:: bash

   git clone --recurse-submodules https://github.com/jasonge27/picasso.git
   cd picasso
   make Pyinstall

Alternative source install via ``setup.py``:

.. code-block:: bash

   cd python-package
   python setup.py install --user

Verify installation:

.. code-block:: python

   import pycasso
   pycasso.test()

Quick Start
-----------

.. code-block:: python

   import numpy as np
   import pycasso

   n, d, s = 200, 100, 10
   X = np.random.randn(n, d)
   beta_true = np.r_[np.random.randn(s), np.zeros(d - s)]
   y = X @ beta_true + np.random.randn(n)

   solver = pycasso.Solver(
       X,
       y,
       family="gaussian",
       penalty="l1",
       lambdas=(100, 0.05),  # (nlambda, lambda_min_ratio)
       useintercept=True
   )
   solver.train()

   result = solver.coef()
   beta_path = result["beta"]      # shape: (nlambda, d)
   intercept_path = result["intercept"]
   y_pred = solver.predict(X[:5], lambdidx=20)

API Notes
---------

``Solver`` inputs:

- ``x``: numeric array of shape ``(n_samples, n_features)``
- ``y``:
  - ``gaussian``/``sqrtlasso``: numeric values
  - ``binomial``: binary labels in ``{0, 1}``
  - ``poisson``: non-negative integers
- ``lambdas``:
  - tuple ``(nlambda, lambda_min_ratio)`` to auto-generate the path, or
  - explicit decreasing sequence of positive values
- ``family``: one of ``"gaussian"``, ``"binomial"``, ``"poisson"``,
  ``"sqrtlasso"``
- ``penalty``: one of ``"l1"``, ``"mcp"``, ``"scad"``

For nonconvex logistic/poisson fits (``penalty="mcp"`` or ``"scad"``),
keep ``lambda_min_ratio >= 0.05`` to avoid unstable optimization.

Typical workflow:

1. Build a ``Solver``.
2. Call ``train()``.
3. Inspect coefficients from ``coef()``.
4. Use ``predict(newdata, lambdidx=...)`` for prediction.

For developers
--------------

Build Sphinx docs locally:

.. code-block:: bash

   cd doc
   make html
