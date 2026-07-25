Pycasso: Python interface for PICASSO
=====================================

``pycasso`` fits sparse regularization paths with a C++ active-set solver.
It supports lasso, MCP, and SCAD penalties for Gaussian, binomial, Poisson,
square-root-lasso, and multinomial models.

Highlights
----------

* Gaussian models use active-set coordinate descent. The default
  ``type_gaussian="auto"`` selects residual or lazy-covariance updates from
  the design shape, path length, and an 8 MiB covariance-cache guard.
* Binomial and Poisson models use active-set Proximal Newton/IRLS.
  Square-root-lasso uses a global quadratic majorizer with active-set
  coordinate updates.
* Multinomial models use a class-coupled active-set Proximal Newton/IRLS
  solver with strong screening and full KKT checks.
* MCP and SCAD use adaptive local linear approximation (LLA) for every
  non-Gaussian family. Gaussian MCP/SCAD uses its direct coordinate solver.
* Prediction, path assessment, confusion matrices, cross-validation, offsets,
  early stopping, and per-lambda diagnostics are available from ``Solver``.
* Standardization reuses its final C-contiguous design as workspace, avoiding
  additional full-size scaled, centered, and squared temporary matrices.

Installation
------------

Install the latest published release with::

    python -m pip install pycasso

Published releases can lag the development interfaces described in this
checkout. To use exactly the code documented here, build the native library
and install this checkout::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target stage_picasso
    cd python-package
    PICASSO_NATIVE_LIBRARY=../build/stage/libpicasso.so python -m pip install .

Use ``libpicasso.dylib`` on macOS or ``picasso.dll`` on Windows. The
``PICASSO_NATIVE_LIBRARY`` path must identify exactly the library produced by
the intended build. The same variable also overrides package-local library
discovery at runtime, which is useful for testing a source checkout against a
fresh native build. Its expanded absolute path must name an existing file.
NumPy is the only runtime dependency. Install the optional plotting dependency
with ``pip install "pycasso[plot]"``.

Quick start
-----------

.. code-block:: python

    import numpy as np
    import pycasso

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 40))
    y = 0.5 + X[:, 0] - 0.7 * X[:, 1] + rng.normal(size=200)

    model = pycasso.Solver(
        X, y, family="gaussian", penalty="l1", lambdas=(30, 0.05))
    model.train()

    coef = model.coef()
    print(coef["beta"].shape)       # (fitted_nlambda, n_features)
    print(coef["intercept"].shape)  # (fitted_nlambda,)
    prediction = model.predict(X[:5], lambdidx=model.nlambda - 1)
    cv = model.cross_validate(nfolds=5)
    cv_parallel = model.cross_validate(nfolds=5, n_jobs=4)

``lambdas=(30, 0.05)`` requests a 30-point log-spaced path ending nominally
at ``0.05 * lambda_max``. A two-element non-NumPy sequence has this
``(count, ratio)`` meaning. A NumPy array is always an explicit path, even
when it has one or two elements::

    explicit = np.array([0.8, 0.4, 0.2])
    model = pycasso.Solver(X, y, lambdas=explicit)

Explicit values must be finite, nonnegative, and strictly decreasing.
``dfmax``, normal path stopping, or a solver failure can make the fitted path
shorter than the requested path.

All ``numpy.ma.MaskedArray`` inputs are rejected, even when their current mask
is entirely ``False``. This is an intentional API contract: silently dropping
the mask would make an accepted object ambiguous if its mask were modified
later. Pass an ordinary NumPy array containing only the deliberately observed
values instead.

Explicit prediction, assessment, and confusion design matrices must contain
at least one row; empty evaluation sets are rejected rather than reported as
NaN metrics or all-zero confusion matrices.

Families and responses
----------------------

``family="gaussian"``
    Finite numeric response. ``predict(..., type="response")`` returns the
    fitted mean.

``family="binomial"``
    Exactly two observed numeric, string, or Boolean response levels. Response
    prediction returns the probability of ``result["levels"][1]``;
    ``type="link"`` returns log-odds and ``type="class"`` returns encoded
    zero/one values. Use ``result["levels"][codes]`` to restore labels.

``family="poisson"``
    Nonnegative integer response with at least one positive value. Response
    prediction returns the fitted mean and link prediction returns log-mean.

``family="sqrtlasso"``
    Finite numeric response. Prediction has the same shape as Gaussian.

``family="multinomial"``
    At least three numeric or string classes. For one lambda, response and
    link predictions have shape ``(n_new, n_classes)``; class prediction
    returns the original labels. Coefficients have shape
    ``(fitted_nlambda, n_classes, n_features)`` and intercepts have shape
    ``(fitted_nlambda, n_classes)``.

New categorical class maps use NumPy's sorted unique order; always treat
``result["levels"]`` as authoritative. This deterministic order can differ
from R's locale-dependent factor ordering for character labels. Binomial
assessment and confusion match original fitted labels first; only when an
input contains an unmatched label is a wholly numeric 0/1 vector interpreted
as already encoded codes.

All Python lambda indices are zero-based. ``predict`` uses the last fitted
lambda by default. Supply ``lam=<value>`` to interpolate coefficients between
path points; values outside the fitted range are clamped to an endpoint.
``type="nonzero"`` returns selected feature indices (one list per class for
multinomial models). Since support sets cannot be interpolated, a nonzero
query with ``lam`` uses the nearest fitted lambda; an exact distance tie uses
the earlier, larger lambda.

Offsets
-------

Binomial and Poisson models accept one finite link-scale offset per training
row. Prediction-derived operations on an offset-fitted model require a
matching new offset, except ``predict(type="nonzero")`` because support
extraction does not evaluate new rows:

.. code-block:: python

    exposure = rng.uniform(0.5, 2.0, size=X.shape[0])
    offset = np.log(exposure)
    count = rng.poisson(np.exp(offset + 0.2 + 0.25 * X[:, 0]))

    poisson = pycasso.Solver(
        X, count, family="poisson", offset=offset, lambdas=(20, 0.1))
    poisson.train()
    mean = poisson.predict(
        X[:5], lambdidx=poisson.nlambda - 1, newoffset=offset[:5])

Assessment and cross-validation
-------------------------------

``assess()`` evaluates every fitted lambda. It always returns ``lambda`` and
``deviance``. Deviance means half MSE for Gaussian and
square-root-lasso, negative log-likelihood for categorical models, and
conventional mean Poisson deviance. Gaussian and square-root-lasso add
``mse`` and ``mae``;
Poisson adds ``mse``; binomial and multinomial add ``class_error``.
``confusion()`` returns one predicted-by-observed count matrix per requested
lambda for binomial or multinomial models. All fitted class levels remain on
both axes, even when a supplied subset omits one. For binomial models, axis
positions 0 and 1 correspond to ``result["levels"]``.

``cross_validate()`` returns ``cvm``, the fold-based standard error
``cvsd``, error bounds, ``nzero``,
``lambda_min``, and ``lambda_1se``. Fold IDs are zero-based, contiguous, and
start at zero. Automatically generated categorical folds are stratified and
every categorical training fold must retain every fitted class. The initial
full-data generated multinomial path may stop normally after saturation; its
retained prefix becomes the fixed path for every fold. Any shortened,
partially trained, or otherwise unusable fold is an error. Multinomial
``nzero`` counts nonzero class-feature entries.

Fold fitting is serial by default (``n_jobs=1``). Set ``n_jobs`` to a
positive integer above one to fit independent folds concurrently with
threads; the worker count is capped at the number of folds, and results and
errors are still aggregated in fold order. Native PICASSO calls release the
Python GIL, so this can accelerate sufficiently expensive folds. Each active
fold owns its training design and solver outputs, however, so peak memory
increases with the number of concurrent workers. Also cap the BLAS library to
one thread (for example with ``OPENBLAS_NUM_THREADS=1``,
``OMP_NUM_THREADS=1``, or the vendor-equivalent setting) to avoid multiplying
fold threads by BLAS threads.

An unusually deep scalar path can make a training fold hit normal saturation
before it covers the fixed full-data sequence. This is reported as a
truncated-fold error instead of averaging different paths. Request fewer
lambdas or a less aggressive endpoint ratio when that occurs.

``train()`` and ``cross_validate()`` calls on the same ``Solver`` instance are
serialized because both can replace native output buffers. Separate
``Solver`` instances remain independent and may run concurrently. Since
``coef()`` returns the live result dictionary, do not read it while another
thread is retraining that same instance.

Accuracy and adaptive LLA
-------------------------

High precision is the default: ``fast_mode=False`` defaults to
``prec=1e-7`` and allows a custom positive precision.
``fast_mode=True`` uses calibrated
stopping/KKT tolerances of ``4e-4`` for Poisson and ``1e-4`` for binomial,
square-root-lasso, and multinomial. Gaussian remains at ``1e-7`` because its
scaled objective-change test already follows the glmnet convention. These
are achieved-accuracy presets, not glmnet's literal ``thresh`` value.

For MCP and SCAD non-Gaussian fits, ``lla_max_stages=3`` is both the minimum
and default maximum: one lasso master plus two weighted-lasso updates. A
larger value permits adaptive continuation until target stationarity is at
most ``prec``. ``max_ite`` independently limits work inside each weighted-L1
subproblem. Every fit validates ``lla_max_stages`` as an integer of at
least three; only non-Gaussian MCP/SCAD optimization consumes the budget.

Termination and diagnostics
---------------------------

Every result dictionary exposes ``status``, ``status_code``, and failure
metadata. Gaussian paths report an exhausted coordinate-descent budget as
``inner_iteration_limit`` and retain only the previously converged lambda
prefix. Non-Gaussian results additionally expose ``runtime``, ``objective``,
``kkt``, and ``stationarity``. Scalar non-Gaussian families also expose
``lla_stages`` (and the alias ``stages``);
multinomial fits expose ``outer_ite``, ``inner_sweeps``, and
``coordinate_updates``.

Status codes 0 (completed) and 1 (``dfmax_reached``) retain usable models for
every family. For non-Gaussian MCP/SCAD fits, code 10
(``lla_stationarity_limit``) also retains usable models: the requested LLA
budget ended before stationarity certification, rather than at a hard failure.
A later hard failure retains the committed prefix, sets
``result["state"] == "partially trained"``, and emits ``RuntimeWarning``.
Failure before the first committed lambda raises ``PycassoError``.

Generated multinomial paths may omit a saturated tail after at least five
lambdas. Explicit multinomial paths disable that saturation rule, but
``dfmax`` or a hard failure can still truncate them. Inspect
``path_early_stopped`` and ``requested_nlambda`` when those keys are present.

Reference
---------

Jason Ge, Xingguo Li, Haoming Jiang, Han Liu, Tong Zhang, Mengdi Wang, and
Tuo Zhao. "Picasso: A Sparse Learning Library for High Dimensional Data
Analysis in R and Python." *Journal of Machine Learning Research*,
20(44):1-5, 2019.
https://www.jmlr.org/papers/v20/17-722.html

License
-------

GPL-3.0
