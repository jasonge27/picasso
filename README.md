<h1 align="center">PICASSO</h1>
<h4 align="center">R and Python Library for Pathwise Coordinate Optimization for Sparse Learning</h4>

___PICASSO___ (PathwIse
CalibrAted Sparse Shooting algOrithm) implements a unified framework of pathwise coordinate optimization for a variety of sparse learning problems (e.g., sparse linear regression, sparse logistic regression, sparse Poisson regression and scaled sparse linear regression) combined with efficient active set selection strategies. The core algorithm is implemented in C++ with Eigen3 support for portable high performance linear algebra. Runtime profiling is documented in the [__Performance__](#performance) section.

## Table of contents

- [Directory structure](#directory-structure)
- [Introduction](#introduction)
- [Background](#background)
- [Solvers](#solvers)
  - [GLM Linear Regression](#glmlinear)
  - [GLM Logistic Regression](#glmlogistic)
  - [GLM Poisson Regression](#glmpoisson)
  - [Scaled Linear Regression](#glmscaled)
- [Penalties](#penalties)
  - [L1 penalty](#l1)
  - [SCAD penalty](#scad)
  - [MCP penalty](#mcp)
- [Power of Nonconvex Penalties](#ncvpenalty)
- [Performance](#performance)
- [Installation](#installation)
- [Tutorials](#tutorials)
- [References](#references)

## Directory structure
The directory is organized as follows:
* [__src__](src): C++ implementation of the PICASSO algorithm.
   * [__c_api__](c_api): C API as an interface for R and Python package.
   * [__objective__](objective): Objective functions, which includes linear regression, logsitic regression, poisson regression and scaled linear regression.
   * [__solver__](solver): Two types of pathwise active set algorithms. Actgd.cpp implements pathwise active set + gradient descent. Actnewton.cpp implements pathwise active set + newton algorithm.
* [__include__](include) 
   * [__picasso__](picasso): declarations of the C++ implementation
   * [__Eigen__](Eigen): Eigen3 header files for high performance linear algebra.
* [__amalgamation__](amalgamation):flag all the c++ implementation for compiling.
* [__cmake__](cmake):Makefile local configurations.
* [__make__](make):Makefile local configurations.
* [__R-package__](R-package): R wrapper for the source code.
* [__python-package__](python-package): Python wrapper for the source code.
* [__tutorials__](tutorials): tutorials for using the code in R and Python.
* [__profiling__](profiling): profiling the performance from R package.

## Introduction
The pathwise coordinate optimization is undoubtedly one the of the most popular solvers for a large variety of sparse learning problems. By leveraging the solution sparsity through a simple but elegant algorithmic structure, it significantly boosts the computational performance in practice (Friedman et al., 2007). Some recent progresses in (Zhao et al., 2017; Li et al., 2017) establish theoretical guarantees to further justify its computational and statistical superiority for both convex and nonvoncex sparse learning, which makes it even more attractive to practitioners.

We recently developed a new library named PICASSO, which implements a unified toolkit of pathwise coordinate optimization for solving a large class of convex and nonconvex regularized sparse learning problems. Efficient active set selection strategies are provided to guarantee superior statistical and computational preference.


The pathwise coordinate optimization framework with 3 nested loops : (1) Warm start initialization; (2) Active set selection, and strong rule for coordinate preselection; (3) Active coordinate minimization. Please refer to tutorials/PICASSO.pdf((https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/PICASSO.pdf) for details of the algorithm design.

![The pathwise coordinate optimization framework](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/images/picasso_flow.png)

## Background

## Solvers

## Penalties



## Power of Nonconvex Penalties 

L1 penalized regression (LASSO) is great for feature selection. However when you use LASSO in very noisy setting, especially when some columns in your data have strong colinearity, LASSO tends to give biased estimator due to the penalty term. As demonstrated in the example below, the lowest estimation error among all the lambdas computed is as high as **16.41%**.

```R
> set.seed(2016)
> library(glmnet)
> n <- 1000; p <- 1000; c <- 0.1
> # n sample number, p dimension, c correlation parameter
> X <- scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(n-1)*sqrt(n) # n is smaple number,
> s <- 20  # sparsity level
> true_beta <- c(runif(s), rep(0, p-s))
> Y <- X%*%true_beta + rnorm(n)
> fitg<-glmnet(X,Y,family="gaussian")
> # the minimal estimation error |\hat{beta}-beta| / |beta|
> min(apply(abs(fitg$beta - true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
[1] 0.1641195
```



Nonconvex penalties such as SCAD [1] and MCP [2] are statistically better but computationally harder. The solution for SCAD/MCP penalized linear model has much less estimation error than lasso but calculating the estimator involves non-convex optimization. With limited computation resource, we can only get a local optimum which probably lacks the good property of the global optimum.

The PICASSO package [3, 4]  solves non-convex optimization through multi-stage convex relaxation. Although we only find a local minimum, it can be proved that this local minimum does not lose the superior statistcal property of the global minimum. Multi-stage convex relaxation is also much more stable than other packages (see benchmark below).

Let's see PICASSO in action — the estimation error drops to **6.06%** using SCAD penalty from **16.41%** error produced by LASSO.

```R
> library(picasso)
> fitp <- picasso(X, Y, family="gaussian", method="scad")
> min(apply(abs(fitp$beta-true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
[1] 0.06064173
```




## Performance 
```bash
$cd tests
$Rscript benchmark.R
```
As a traditional LASSO solver, our package achieves state-of-the-art performance. For SCAD regularzied linear/logistic regression, it can be shown that our package is much faster and more stable than the alternative ncvreg. (The experiments can also be done with MCP penalty.)  More experiments can be found in vignettes/PICASSO.pdf

For the experiments, sample number is denoted by n and sample dimension is denoted by p. The parameter c is used to control the column-wise correlation of the desgin matrix X as before.

### State-of-the-art LASSO Solver

We're as fast as glmnet for L1 regularized linear/logistic regression. Here we benchmark logistic regression as an example. Parameter c is used to add correlation between columns of X to mimic multi-colinearity. PICASSO is the best solver when we have multi-colinearity in the data.

```R
source("tests/test_picasso.R")
test_lognet(n=2000, p=1000, c=0.1)
test_lognet(n=2000, p=1000, c=0.5)
test_lognet(n=2000, p=1000, c=1.0)
```

|         |     c=0.1     |    c=0.5     |    c= 1.0    |
| :-----: | :-----------: | :----------: | :----------: |
| PICASSO |   1.526(s)    |   0.721(s)   |   0.718(s)   |
| glmnet  |   1.494(s)    |   0.845(s)   |   1.743(s)   |
| ncvreg  | 10.564(s) [*] | 7.825(s) [#] | 5.458(s) [#] |

'[*]': Package exited with warning: Algorithm failed to converge for some values of lambda.

'[#]': Package exited with error messages.

### Nonconvex Penalty

As glmnet does not provide nonconvex penalty solver, we will compare with ncvreg for run-time and best estimation error along the regularization path.

For well-conditioned cases when there's no multi-colinearity, LASSO tends to have lower estimation error. However, as c becomes larger, LASSO's estimation error quickly increases. Nonconvex penalty can be very helpful when some columns of the data are highly correlated.

```R
source('tests/test_picasso.R')
test_lognet_nonlinear(n=3000, p =3000, c=0.1)
test_lognet_nonlinear(n=3000, p =3000, c=0.5)
test_lognet_nonlinear(n=3000, p =3000, c=1.0)
```

Timing for SCAD regularized logistic regression. Estimation error is calculated by finding the best approximation to the true regression coefficient across all regularization parameter.

```R
min(apply(abs(fitted.model$beta - true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
```



|                                        |     c = 0.1     |    c = 0.5     |       c = 1.0        |
| :------------------------------------: | :-------------: | :------------: | :------------------: |
|          PICASSO (time/error)          | 10.98(s) / 5.0% | 5.16(s) / 1.5% |    7.17(s) / 8.6%    |
|          ncvreg (time/error)           | 8.05(s) / 5.6%  | 7.08(s) / 6.0% | 56.16(s) / 35.3% [*] |
| Estimation Error using LASSO in glmnet |      1.3%       |     14.0%      |        36.6%         |

'[*]': Package exited with warning: Algorithm failed to converge for some values of lambda.

The experiments are conducted on a MacBook Pro with 2.4GHz Intel Core i5 and 8GB RAM. R version is 3.3.0. The ncvreg version is 3.6-0. The glmnet version is 2.0-5.



### Square Root Lasso Solver

We also implemented sqaure root loss function with L1/SCAD/MCP penalty using the same active set based second order algorithms. With fixed sample size, we change the sample dimension and report the CPU time for pathwise square root Lasso. For fair comparisons, all three solvers follows the same solution path and their precisions are adjusted to achieve the same level of accuracy.

```R
source('tests/test_picasso.R')
test_sqrt_mse(n=500, p =400, c=0.5)
test_sqrt_mse(n=500, p =800, c=0.5)
test_sqrt_mse(n=500, p =1600, c=0.5)
```

|         |     d=400     |     d=800      |     d=1600      |
| :-----: | :-----------: | :------------: | :-------------: |
| PICASSO | 0.51 (0.02) s | 1.41 (0.08) s  |  2.32 (0.10) s  |
| scalreg | 3.46 (0.27) s | 22.08 (0.89) s | 49.13 (1.32) s  |
|  flare  | 5.50 (0.25) s | 28.90 (0.26)s  | 178.65 (3.32) s |

 The experiments are run in Microsoft R Open 3.3.2 on Mac OS 10.12.3 with 2.4GHz Intel Core i5 and 8GB RAM. R package flare is in version 1.5.0. R package scalreg is in version 1.0.  For each method and dataset, the experiment is repeated 10 times and we report the mean and standard deviations of the CPU time in the table.

## Installation
### Installing R package
The R package is hosted on CRAN. The easiest way to install R package is by running the following command in R
```R
install.packages("picasso")
```

### Installing Python package
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
- **Note**: Owing to the setting on different OS, our distribution might not be working in your environment (especially in **Windows**). Thus please build from source.

You can test if the package has been successfully installed by:

.. code-block:: python

        import pycasso
        pycasso.test()

..

Details can also be found in [document](https://hmjianggatech.github.io/picasso/) or [github](https://github.com/jasonge27/picasso/tree/master/python-package)

## Tutorials
Check the R tutorial in tutorials/tutorial.R and Python tutorial in tutorials/tutorial.py. Let us know if anything is hard to use or if you want any other features. 

## References

[1] Jianqing Fan and Runze Li, Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties, 2001

[2] Cun-Hui Zhang, Nearly Unbiased Variable Selection Under Minimax Concave Penalty, 2010

[3] Jason Ge, Mingyi Hong, Mengdi Wang, Han Liu, and Tuo Zhao, Homotopy Active Set Proximal Newton Algorithm for Sparse Learning, 2016

[4] Tuo Zhao, Han Liu, and Tong Zhang, Pathwise Coordinate Optimization for Nonconvex Sparse Learning: Algorithm and Theory, 2014
