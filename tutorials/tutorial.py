"""Current pycasso interface examples for every supported family.

Run after installing the development checkout described in
``python-package/README.rst``.
"""

import numpy as np

import pycasso


rng = np.random.default_rng(7)
n, d = 180, 30
X = rng.normal(size=(n, d))


# Gaussian: the public default resolves the residual/covariance backend.
y_gaussian = 0.4 + X[:, 0] - 0.6 * X[:, 1] + rng.normal(scale=0.7, size=n)
gaussian = pycasso.Solver(
    X, y_gaussian, family="gaussian", penalty="l1",
    lambdas=(20, 0.05), type_gaussian="auto")
gaussian.train()
print("Gaussian backend:", gaussian.type_gaussian)
print("Gaussian beta shape:", gaussian.coef()["beta"].shape)
print("Gaussian prediction:",
      gaussian.predict(X[:3], lambdidx=gaussian.nlambda - 1))


# Binomial MCP: adaptive LLA may continue beyond its default three stages.
logit = -0.2 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
y_binomial = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
binomial = pycasso.Solver(
    X, y_binomial, family="binomial", penalty="mcp",
    lambdas=(16, 0.1), lla_max_stages=5, fast_mode=True)
binomial.train()
print("Binomial status:", binomial.coef()["status"])
print("Binomial probabilities:",
      binomial.predict(X[:3], lambdidx=binomial.nlambda - 1))
print("Binomial confusion:",
      binomial.confusion(
          X, y_binomial, lambdidx=binomial.nlambda - 1)[0])


# Poisson: offsets are added on the link scale and required for new rows.
exposure = rng.uniform(0.5, 2.0, size=n)
offset = np.log(exposure)
poisson_mean = np.exp(offset + 0.1 + 0.25 * X[:, 0])
y_poisson = rng.poisson(poisson_mean)
poisson = pycasso.Solver(
    X, y_poisson, family="poisson", offset=offset,
    lambdas=(16, 0.1), fast_mode=True)
poisson.train()
print("Poisson means:",
      poisson.predict(
          X[:3], lambdidx=poisson.nlambda - 1,
          newoffset=offset[:3]))


# Square-root-lasso.
y_sqrt = X[:, 0] - X[:, 2] + rng.normal(size=n)
sqrt_model = pycasso.Solver(
    X, y_sqrt, family="sqrtlasso", lambdas=(16, 0.1))
sqrt_model.train()
sqrt_metrics = sqrt_model.assess()
print("Square-root-lasso final MSE:", sqrt_metrics["mse"][-1])


# Multinomial: original string labels are retained in class prediction.
scores = np.column_stack((
    0.8 * X[:, 0],
    -0.5 * X[:, 0] + 0.7 * X[:, 1],
    -0.6 * X[:, 1],
))
labels = np.array(["red", "green", "blue"])
y_multinomial = labels[np.argmax(
    scores + rng.normal(scale=0.5, size=scores.shape), axis=1)]
if np.unique(y_multinomial).size != 3:
    y_multinomial[:3] = labels

multinomial = pycasso.Solver(
    X, y_multinomial, family="multinomial", penalty="l1",
    lambdas=(16, 0.1), fast_mode=True)
multinomial.train()
print("Multinomial beta shape:", multinomial.coef()["beta"].shape)
print("Multinomial classes:",
      multinomial.predict(
          X[:5], lambdidx=multinomial.nlambda - 1, type="class"))


# Assessment and CV evaluate the retained path. Python indices are zero-based.
print("Multinomial class error:",
      multinomial.assess()["class_error"][-1])
cv = gaussian.cross_validate(nfolds=3, type_measure="deviance")
print("CV lambda.min:", cv["lambda_min"])
print("CV lambda.1se:", cv["lambda_1se"])

# Plotting is optional and requires Matplotlib:
# gaussian.plot(max_features=8)
