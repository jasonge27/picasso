[![Build Status](http://circleci-badges-max.herokuapp.com/img/jasonge27/picasso/1?token=65317b48c13e9567b12e5a8c52661d709d3f605e)](https://circleci.com/gh/jasonge27/picasso/1)
<h1 align="center">PICASSO</h1>
<h4 align="center">High Performance R and Python Library for Sparse Learning</h4>

___PICASSO___ (PathwIse
CalibrAted Sparse Shooting algOrithm) implements a unified framework of pathwise coordinate optimization for a variety of sparse learning problems (e.g., sparse linear regression, sparse logistic regression, sparse Poisson regression and scaled sparse linear regression) combined with efficient active set selection strategies. The core algorithm is implemented in C++ with Eigen3 support for portable high performance linear algebra. Runtime profiling is documented in the [__Performance__](#performance) section.

You can cite this work by 
```
@article{ge2019picasso,
  title={Picasso: A Sparse Learning Library for High Dimensional Data Analysis in R and Python.},
  author={Ge, Jason and Li, Xingguo and Jiang, Haoming and Liu, Han and Zhang, Tong and Wang, Mengdi and Zhao, Tuo},
  journal={J. Mach. Learn. Res.},
  volume={20},
  pages={44--1},
  year={2019}
}
```

## Table of contents

- [Table of contents](#table-of-contents)
- [Directory structure](#directory-structure)
- [Introduction](#introduction)
- [Background](#background)
- [Power of Nonconvex Penalties](#power-of-nonconvex-penalties)
- [Performance](#performance)
    - [R package](#r-package)
    - [Python package](#python-package)
- [Installation](#installation)
    - [Installing R package](#installing-r-package)
    - [Installing Python package](#installing-python-package)
- [Tutorials](#tutorials)
- [References](#references)

## Directory structure
The directory is organized as follows:
* [__src__](src): C++ implementation of the PICASSO algorithm.
   * [__c_api__](src/c_api): C API used by the R and Python wrappers.
   * [__objective__](src/objective): Objective definitions for linear, logistic, poisson, and square-root lasso models.
   * [__solver__](src/solver): Pathwise active-set solvers (gradient-based and Newton-style variants).
* [__include__](include): Public headers.
   * [__picasso__](include/picasso): Core C++ declarations.
   * [__eigen3__](include/eigen3): Eigen3 headers bundled as a submodule.
* [__amalgamation__](amalgamation): Single-file C++ amalgamation used for wrapper builds.
* [__cmake__](cmake): CMake helper modules/configuration.
* [__make__](make): Makefile-based build configuration.
* [__R-package__](R-package): R wrapper for the source code.
* [__python-package__](python-package): Python wrapper for the source code.
* [__tutorials__](tutorials): End-to-end examples in R and Python.
* [__profiling__](profiling): Benchmark scripts and performance comparisons.

## Introduction
Pathwise coordinate optimization is one of the most widely used approaches for sparse learning. By exploiting solution sparsity through a simple iterative structure, it delivers strong practical performance (Friedman et al., 2007). Recent theory (Zhao et al., 2017; Li et al., 2017) further supports its computational and statistical advantages for both convex and nonconvex sparse estimation.

PICASSO implements a unified toolkit for convex and nonconvex regularized sparse models. It combines pathwise optimization with efficient active-set selection to improve both speed and statistical performance.


The pathwise framework has three nested stages: (1) warm-start initialization, (2) active-set selection with strong-rule pre-screening, and (3) active-coordinate minimization. See [tutorials/PICASSO.pdf](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/PICASSO.pdf) for algorithmic details.

![The pathwise coordinate optimization framework](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/images/picasso_flow.png)

## Background
Several R packages (for example, `glmnet` and `ncvreg`) implement effective sparse-learning solvers. However, support for nonconvex penalties can be limited, and some methods become unstable under strong feature collinearity. PICASSO combines pathwise optimization with multi-stage convex relaxation to compute a statistically well-behaved local optimum with strong practical stability.

## Power of Nonconvex Penalties

L1 penalized regression (LASSO) is a useful tool for feature selection but it tends to give very biased estimator due to the penalty term. As demonstrated in the example below, the lowest estimation error among all the lambdas computed is as high as **16.41%**.

```R
> set.seed(2016)
> library(glmnet)
> n <- 1000; p <- 1000; c <- 0.1
> # n sample number, p dimension, c correlation parameter
> X <- scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(n-1)*sqrt(n) # n is sample number,
> s <- 20  # sparsity level
> true_beta <- c(runif(s), rep(0, p-s))
> Y <- X%*%true_beta + rnorm(n)
> fitg<-glmnet(X,Y,family="gaussian")
> # the minimal estimation error |\hat{beta}-beta| / |beta|
> min(apply(abs(fitg$beta - true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
[1] 0.1641195
```

Nonconvex penalties such as SCAD [1] and MCP [2] are statistically better but computationally harder. The solution for SCAD/MCP penalized linear model has much less estimation error than lasso but calculating the estimator involves non-convex optimization. With limited computation resource, we can only get a local optimum which probably lacks the good property of the global optimum.

PICASSO [3, 4] solves nonconvex problems through multi-stage convex relaxation. Although the algorithm returns a local minimum, theory shows this solution can retain the key statistical advantages of the global optimum. The multi-stage strategy is also empirically more stable in difficult settings.

Let's see PICASSO in action — the estimation error drops to **6.06%** using SCAD penalty from **16.41%** error produced by LASSO.

```R
> library(picasso)
> fitp <- picasso(X, Y, family="gaussian", method="scad")
> min(apply(abs(fitp$beta-true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
[1] 0.06064173
```



## Performance
```bash
cd profiling
Rscript benchmark.R
python benchmark.py
```

### R package
 - Sparse linear regression: `picasso` achieves timing and objective values comparable to `glmnet` and `ncvreg`.
 - Sparse logistic regression: with L1 regularization, `picasso`, `glmnet`, and `ncvreg` perform similarly; with nonconvex regularization, `picasso` is notably better than `ncvreg`, especially in ill-conditioned regimes.
 - Scaled sparse linear regression: PICASSO significantly outperforms `scalreg` and `flare`; see Table 5.3 in [tutorials/PICASSO.pdf](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/PICASSO.pdf).

Benchmark setup details are documented in [tutorials/PICASSO.pdf](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/PICASSO.pdf).

![Performance_R](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/images/performance_R.jpeg)

### Python package
For L1-regularized linear and logistic regression, we compare against scikit-learn (for example, `sklearn.linear_model.lasso_path` and `sklearn.linear_model.LogisticRegression` with liblinear). Benchmark code is available at [profiling/benchmark.py](https://raw.githubusercontent.com/jasonge27/picasso/master/profiling/benchmark.py). With fixed sample size and increasing feature dimension, PICASSO runtime remains relatively stable due to active-set updates, while reaching comparable objective values under matched optimization precision.

![Performance_Python](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/images/performance_python.jpeg)

## Installation
PICASSO relies on Eigen3 headers, included in this repository as a submodule.
Because Eigen is header-only, existing system installations typically do not conflict.

### Installing R package
There are two ways to install the `picasso` R package.

- Install from CRAN (recommended):
```R
install.packages("picasso")
```

- Install from source:
```bash
git clone --recurse-submodules https://github.com/jasonge27/picasso.git
cd picasso
make Rinstall
```

### Installing Python package
There are two ways to install the `pycasso` Python package.

- Install from PyPI (recommended):
```bash
pip install pycasso
```

- Install from source:
```bash
git clone --recurse-submodules https://github.com/jasonge27/picasso.git
cd picasso
make Pyinstall
```

You can verify installation with:
```bash
python -c "import pycasso; pycasso.test()"
```

More wrapper-specific details are available in [python-package/README.rst](python-package/README.rst) and [R-package](R-package).

## Tutorials
See [tutorials/tutorial.R](tutorials/tutorial.R) and
[tutorials/tutorial.py](tutorials/tutorial.py) for runnable examples.

## References

[1] Jianqing Fan and Runze Li, Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties, 2001

[2] Cun-Hui Zhang, Nearly Unbiased Variable Selection Under Minimax Concave Penalty, 2010

[3] Xingguo Li, Jason Ge, Haoming Jiang, Mingyi Hong, Mengdi Wang, and Tuo Zhao, Boosting Pathwise Coordinate Optimization in High Dimensions: Sequential Screening and Proximal Sub-sampled Newton Algorithm, 2016

[4] Tuo Zhao, Han Liu, and Tong Zhang, Pathwise Coordinate Optimization for Nonconvex Sparse Learning: Algorithm and Theory, 2014

[5] Xingguo Li,Lin F. Yang, Jason Ge, Jarvis Haupt, Tong Zhang and Tuo Zhao, On Quadratic Convergence of DC Proximal Newton Algorithm in Nonconvex Sparse Learning, 2017
