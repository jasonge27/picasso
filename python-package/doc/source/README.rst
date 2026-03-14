Pycasso - Python Interface for PICASSO
=======================================

**PICASSO**: Penalized Generalized Linear Model Solver - Unleash the Power of Non-convex Penalty

Pycasso provides a Python interface to the PICASSO C++ solver for fitting sparse
generalized linear models with L1 (Lasso), MCP, and SCAD penalties via pathwise
coordinate optimization.

Features
--------

- **Families**: Gaussian (linear), Binomial (logistic), Poisson, Sqrt-Lasso
- **Penalties**: L1 (Lasso), MCP, SCAD
- **Standardization**: Automatic design matrix standardization with proper coefficient rescaling
- **Gaussian solvers**: Naive update and covariance update
- **Early stopping**: ``dfmax`` parameter to stop when too many coefficients become nonzero
- **Intercept**: Optional intercept term with correct initialization for all families

Requirements
------------

- Linux or macOS
- Python 3
- NumPy, SciPy

Installation
------------

Install from PyPI::

    pip install pycasso

Install from source::

    git clone https://github.com/jasonge27/picasso.git
    cd picasso
    mkdir build && cd build && cmake .. && make
    cd ../python-package
    pip install .

Usage
-----

.. code-block:: python

    import numpy as np
    import pycasso

    # Generate example data
    n, d = 200, 50
    X = np.random.randn(n, d)
    beta_true = np.zeros(d)
    beta_true[:3] = [1, -0.5, 0.3]
    y = X @ beta_true + np.random.randn(n) * 0.5

    # Fit sparse linear regression with Lasso
    s = pycasso.Solver(X, y, family="gaussian", penalty="l1")
    s.train()
    result = s.coef()
    print(result['beta'])     # coefficient matrix (nlambda x d)
    print(result['intercept'])  # intercept for each lambda

    # Predict
    y_pred = s.predict(X[:5, :])

    # Logistic regression with MCP penalty
    y_bin = (np.random.rand(n) < 0.5).astype(float)
    s2 = pycasso.Solver(X, y_bin, family="binomial", penalty="mcp")
    s2.train()

    # Early stopping with dfmax
    s3 = pycasso.Solver(X, y, family="gaussian", dfmax=10)
    s3.train()

..

Reference
---------

Jason Ge, Xingguo Li, Haoming Jiang, Han Liu, Tong Zhang, Mengdi Wang, and Tuo Zhao.
"Picasso: A Sparse Learning Library for High Dimensional Data Analysis in R and Python."
*Journal of Machine Learning Research*, 20(44):1-5, 2019.

License
-------

GPL-3.0
