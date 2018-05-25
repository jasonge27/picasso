import pycasso
import numpy as np
from sklearn.preprocessing import scale

## Sparse linear regression
## Generate the design matrix and regression coefficient vector
n = 100 # sample number
d = 80 # sample dimension
c = 0.5 # correlation parameter
s = 20  # support size of coefficient

X = scale(np.random.randn(n,d)+c* np.tile(np.random.randn(n),[d,1]).T )/ (n*(n-1))**0.5
beta = np.append(np.random.rand(s), np.zeros(d-s))

## Generate response using Gaussian noise, and fit sparse linear models
noise = np.random.randn(n)
Y = np.matmul(X,beta) + noise


## l1 regularization solved with naive update
solver_l1_naive = pycasso.Solver(X,Y, nlambda=100, family="gaussian", type_gaussian="naive")
solver_l1_naive.train()

## l1 regularization solved with covariance update
solver_l1_cov = pycasso.Solver(X,Y, nlambda=100, family="gaussian", type_gaussian="covariance")
solver_l1_cov.train()

## mcp regularization
solver_mcp = pycasso.Solver(X,Y, nlambda=100, penalty="mcp")
solver_mcp.train()

## scad regularization
solver_scad = pycasso.Solver(X,Y, nlambda=100, penalty="scad")
solver_scad.train()

## Obtain the result
result = solver_l1_naive.coef()

## print out training time
print(result['total_train_time'])

## lambdas used
print(solver_l1_naive.lambdas)

## number of nonzero coefficients for each lambda
print(result['df'])

## coefficients and intercept for the i-th lambda
i = 30
print(solver_l1_naive.lambdas[i])
print(result['beta'][i])
print(result['intercept'][i])

## Visualize the solution path
solver_l1_naive.plot()
solver_l1_cov.plot()
solver_mcp.plot()
solver_scad.plot()


################################################################
## Sparse logistic regression
## Generate the design matrix and regression coefficient vector
n = 100 # sample number
d = 80 # sample dimension
c = 0.5 # correlation parameter
s = 20  # support size of coefficient

X = scale(np.random.randn(n,d)+c* np.tile(np.random.randn(n),[d,1]).T )/ (n*(n-1))**0.5
beta = np.append(np.random.rand(s), np.zeros(d-s))

## Generate response and fit sparse logistic models
noise = np.random.randn(n)
p = 1/(1+np.exp(-np.matmul(X,beta) - noise))
Y = np.random.binomial(np.ones(n,dtype='int64'),p)

## l1 regularization
solver_l1 = pycasso.Solver(X,Y, nlambda=100, family="binomial", penalty="l1")
solver_l1.train()

## mcp regularization
solver_mcp = pycasso.Solver(X,Y, nlambda=100, family="binomial", penalty="mcp")
solver_mcp.train()

## scad regularization
solver_scad = pycasso.Solver(X,Y, nlambda=100, family="binomial", penalty="scad")
solver_scad.train()

## Obtain the result
result = solver_l1.coef()

## print out training time
print(result['total_train_time'])

## lambdas used
print(solver_l1.lambdas)

## number of nonzero coefficients for each lambda
print(result['df'])

## coefficients and intercept for the i-th lambda
i = 30
print(solver_l1.lambdas[i])
print(result['beta'][i])
print(result['intercept'][i])

## Visualize the solution path
solver_l1.plot()


################################################################
## Sparse logistic regression
## Generate the design matrix and regression coefficient vector
n = 100 # sample number
d = 80 # sample dimension
c = 0.5 # correlation parameter
s = 20  # support size of coefficient

X = scale(np.random.randn(n,d)+c* np.tile(np.random.randn(n),[d,1]).T )/ (n*(n-1))**0.5
beta = np.append(np.random.rand(s), np.zeros(d-s))/(s**0.5)

## Generate response and fit sparse logistic models
noise = np.random.randn(n)
p = np.exp(-np.matmul(X,beta) - noise)
Y = np.random.poisson(p, n)

## l1 regularization
solver_l1 = pycasso.Solver(X,Y, nlambda=100, family="poisson", penalty="l1")
solver_l1.train()

## mcp regularization
solver_mcp = pycasso.Solver(X,Y, nlambda=100, family="poisson", penalty="mcp")
solver_mcp.train()

## scad regularization
solver_scad = pycasso.Solver(X,Y, nlambda=100, family="poisson", penalty="scad")
solver_scad.train()

## Obtain the result
result = solver_l1.coef()

## print out training time
print(result['total_train_time'])

## lambdas used
print(solver_l1.lambdas)

## number of nonzero coefficients for each lambda
print(result['df'])

## coefficients and intercept for the i-th lambda
i = 30
print(solver_l1.lambdas[i])
print(result['beta'][i])
print(result['intercept'][i])

## Visualize the solution path
solver_l1.plot()
