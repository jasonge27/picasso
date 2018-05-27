import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pycasso
from sklearn.linear_model import lasso_path
from sklearn.linear_model import lars_path
from sklearn import linear_model
import time

import pdb


def generate_sim_lognet(n, d, c, seed=1024):
  np.random.seed(seed)
  cor_X = c
  S = cor_X * np.ones((d, d)) + (1 - cor_X) * np.diag(np.ones(d))
  R = np.linalg.cholesky(S)

  X = np.dot(np.random.normal(size=n * d).reshape(n, d), R)
  X = preprocessing.scale(X) * np.sqrt(float(n - 1) / n)

  s = 20
  true_beta = np.zeros(d)
  true_beta[0:s] = np.random.uniform(low=0, high=1.0, size=s)

  # strictly seperable
  Y = np.random.binomial(1, 1 / (1 + np.exp(-np.dot(X, true_beta))))

  return (X, Y, true_beta)


def generate_sim_elnet(n, d, c, seed=1024):
  np.random.seed(seed)
  cor_X = c
  S = cor_X * np.ones((d, d)) + (1 - cor_X) * np.diag(np.ones(d))
  R = np.linalg.cholesky(S)

  X = np.dot(np.random.normal(size=n * d).reshape(n, d), R)
  X = preprocessing.scale(X) * np.sqrt(float(n - 1) / n)

  s = 20
  true_beta = np.zeros(d)
  true_beta[0:s] = np.random.uniform(low=0, high=1.0, size=s)

  Y = np.dot(X, true_beta) + np.random.normal(size=n) * 5

  return (X, Y, true_beta)


def elnet_obj(X, Y, beta, intcpt, lamb):
  n, d = X.shape
  return np.sum((Y - np.dot(X, beta.reshape(-1, 1)) - intcpt)**2) / (
      2 * n) + lamb * np.sum(np.abs(beta))


def lognet_obj(X, Y, beta, intcpt, lamb):
  n, d = X.shape
  rp = np.dot(X, beta) + intcpt
  return np.sum(np.log(1 + np.exp(rp)) - Y * rp) / n + lamb * np.sum(
      np.abs(beta))


def test_elnet(n, p, c, nlambda=100):
  X, Y, true_beta = generate_sim_elnet(n, p, c)
  time0 = time.time()
  picasso = pycasso.Solver(
      X,
      Y,
      lambdas=(nlambda, 0.01),
      family='gaussian',
      penalty='l1')
  picasso.train()
  time1 = time.time()
  picasso_time = time1 - time0

  idx = 50
  picasso_obj = elnet_obj(X, Y, picasso.result['beta'][idx, :],
                          picasso.result['intercept'][idx],
                          picasso.lambdas[idx])

  time0 = time.time()

  X_intcpt = np.concatenate((X, np.ones(n).reshape(-1, 1)), axis=1)

  alphas_lasso, coefs_lasso, _ = lasso_path(
      X_intcpt, Y, alphas=picasso.lambdas, eps=1e-3)
  time1 = time.time()

  sklearn_obj = elnet_obj(X, Y, coefs_lasso[0:p, idx], coefs_lasso[p, idx],
                          alphas_lasso[idx] * 2)
  sklearn_time = time1 - time0

  print(
      "Testing L1 penalized linear regression, number of samples:%d, sample dimension:%d, correlation:%f"
      % (n, p, c))
  print("Picasso time:%f, Obj function value:%f" % (picasso_time, picasso_obj))
  print("Sklearn time:%f, Obj function value:%f" % (sklearn_time, sklearn_obj))
  return picasso_time, picasso_obj, sklearn_time, sklearn_obj


def test_lognet(n, p, c, nlambda=100):
  X, Y, true_beta = generate_sim_lognet(n, p, c)
  time0 = time.time()
  picasso = pycasso.Solver(
      X,
      Y,
      lambdas=(nlambda,0.01),
      family='binomial',
      penalty='l1')
  picasso.train()
  time1 = time.time()
  picasso_time = time1 - time0

  idx = 50
  picasso_obj = lognet_obj(X, Y, picasso.result['beta'][idx, :],
                           picasso.result['intercept'][idx],
                           picasso.lambdas[idx])

  time0 = time.time()
  clf = linear_model.LogisticRegression(penalty='l1', tol=1e-6, warm_start=True)
  coefs_ = []
  intcpt_ = []
  for lamb in picasso.lambdas:
    clf.set_params(C=1.0 / (n * lamb))
    clf.fit(X, Y)
    coefs_.append(clf.coef_.ravel().copy())
    intcpt_.append(clf.intercept_.ravel().copy())

  time1 = time.time()

  sklearn_obj = lognet_obj(X, Y, coefs_[idx], intcpt_[idx],
                           picasso.lambdas[idx])
  sklearn_time = time1 - time0

  print(
      "Testing L1 penalized linear regression, number of samples:%d, sample dimension:%d, correlation:%f"
      % (n, p, c))
  print("Picasso time:%f, Obj function value:%f" % (picasso_time, picasso_obj))
  print("Sklearn time:%f, Obj function value:%f" % (sklearn_time, sklearn_obj))
  return picasso_time, picasso_obj, sklearn_time, sklearn_obj


def plot_elnet():
  picasso_time = []
  sklearn_time = []

  d_series = [200, 400, 600, 800, 1000]
  n = 500
  for d in d_series:
    ptime, pobj, sktime, skobj = test_elnet(n, d, 0.1)
    picasso_time.extend([ptime])
    sklearn_time.extend([sktime])

  ptime_handle = plt.plot(d_series, picasso_time, label="picasso")
  sktime_handle = plt.plot(d_series, sklearn_time, label="sklearn")
  plt.legend(loc=2)
  plt.title("L1 Penalized Linear Regression")
  plt.xlabel("Dimension")
  plt.ylabel("CPU Time(s)")
  plt.show()


def plot_lognet():
  picasso_time = []
  sklearn_time = []

  d_series = [200, 400, 600, 800, 1000]
  n = 500
  for d in d_series:
    ptime, pobj, sktime, skobj = test_lognet(n, d, 0.1)
    picasso_time.extend([ptime])
    sklearn_time.extend([sktime])

  ptime_handle = plt.plot(d_series, picasso_time, label="picasso")
  sktime_handle = plt.plot(d_series, sklearn_time, label="sklearn")
  plt.legend(loc=2)
  plt.title("L1 Penalized Logistic Regression")
  plt.xlabel("Dimension")
  plt.ylabel("CPU Time(s)")
  plt.show()


if __name__ == "__main__":
  plot_elnet()
  plot_lognet()
