import numpy as np
import pycasso

np.random.seed(2016)


def standardize_columns(x):
    """Standardize each feature column to zero mean and unit variance."""
    x = np.asarray(x, dtype="double")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1, keepdims=True)
    std[std == 0.0] = 1.0
    return (x - mean) / std


def generate_design(n, d, corr=0.5):
    """Generate a correlated design matrix."""
    return standardize_columns(np.random.randn(n, d) + corr * np.random.randn(n, 1))


def summarize_path(solver, label, idx=30):
    """Print a compact summary for one fitted solver."""
    result = solver.coef()
    i = min(idx, solver.nlambda - 1)
    print("\n==== {} ====".format(label))
    print("First 5 lambdas:", solver.lambdas[:5])
    print("First 5 df values:", result["df"][:5])
    print("lambda[{}]: {}".format(i, solver.lambdas[i]))
    print("intercept[{}]: {}".format(i, result["intercept"][i]))
    print("total_train_time:", result["total_train_time"])
    return result


################################################################
## Sparse linear regression
n, d, s = 100, 80, 20
X = generate_design(n, d, corr=0.5)
beta = np.r_[np.random.rand(s), np.zeros(d - s)]
Y = X.dot(beta) + np.random.randn(n)

solver_g_l1 = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="gaussian", penalty="l1")
solver_g_l1.train()
solver_g_mcp = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="gaussian", penalty="mcp")
solver_g_mcp.train()
solver_g_scad = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="gaussian", penalty="scad")
solver_g_scad.train()

summarize_path(solver_g_l1, "Gaussian / L1")

## Optional: visualize solution paths
solver_g_l1.plot()
solver_g_mcp.plot()
solver_g_scad.plot()


################################################################
## Sparse logistic regression
X = generate_design(n, d, corr=0.5)
beta = np.r_[np.random.rand(s), np.zeros(d - s)]
p = 1.0 / (1.0 + np.exp(-X.dot(beta)))
Y = np.random.binomial(1, p).astype("int64")

solver_b_l1 = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="binomial", penalty="l1")
solver_b_l1.train()
solver_b_mcp = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="binomial", penalty="mcp")
solver_b_mcp.train()
solver_b_scad = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="binomial", penalty="scad")
solver_b_scad.train()

result_b = summarize_path(solver_b_l1, "Binomial / L1")

## Fitted Bernoulli probabilities on training data at each lambda
y_prob_train = solver_b_l1.predict()
print("Prediction shape:", y_prob_train.shape)

## Optional: visualize solution paths
solver_b_l1.plot()
solver_b_mcp.plot()
solver_b_scad.plot()


################################################################
## Sparse poisson regression
X = generate_design(n, d, corr=0.5)
beta = np.r_[np.random.rand(s), np.zeros(d - s)] / np.sqrt(s)
rate = np.exp(X.dot(beta) + np.random.randn(n))
Y = np.random.poisson(rate).astype("int64")

solver_p_l1 = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="poisson", penalty="l1")
solver_p_l1.train()
solver_p_mcp = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="poisson", penalty="mcp")
solver_p_mcp.train()
solver_p_scad = pycasso.Solver(X, Y, lambdas=(100, 0.05), family="poisson", penalty="scad")
solver_p_scad.train()

summarize_path(solver_p_l1, "Poisson / L1")

## Optional: visualize solution paths
solver_p_l1.plot()
solver_p_mcp.plot()
solver_p_scad.plot()
