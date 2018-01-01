<h1 align="center">PICASSO</h1>
<h4 align="center">R and Python Library for Pathwise Coordinate Optimization for Sparse Learning</h4>

___PICASSO___ (PathwIse
CalibrAted Sparse Shooting algOrithm) implements a unified framework of pathwise coordinate optimization for a variety of sparse learning problems (e.g., sparse linear regression, sparse logistic regression, sparse Poisson regression and scaled sparse linear regression) combined with efficient active set selection strategies. The core algorithm is implemented in C++ with Eigen3 support for portable high performance linear algebra. Runtime profiling is documented in the [__Performance__](#performance) section.

## Table of contents

- [Directory structure](#directory-structure)
- [Introduction](#introduction)
- [Background](#background)
- [Objetives and Penalties](#objectives-and-penalties)
- [Power of Nonconvex Penalties](#power-of-nonconvex-penalties)
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


The pathwise coordinate optimization framework with 3 nested loops : (1) Warm start initialization; (2) Active set selection, and strong rule for coordinate preselection; (3) Active coordinate minimization. Please refer to [tutorials/PICASSO.pdf](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/PICASSO.pdf) for details of the algorithm design.

![The pathwise coordinate optimization framework](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/images/picasso_flow.png)

## Background
There exists several R pakcages (such as ncvreg and glmnet) which implement state-of-the-art heuristic optimization algorithms for sparse learning. However they either lack support for nonconvex penalties or becomes very unstable when there are multi-colinear features. PICASSO combines pathwise coordinate optimization and multi-stage convex relaxation for nonconvex optimization and finds a 'good' local minimal which has provable statistical property.

## Objectives and Penalties 
PICASSO supports four types of objective functions: linear regression, logistic regression, poission regression and scaled linear regression. Three types of penalties are supported: L1, MCP (minimax concave penalty) and SCAD (smoothly clipped absolute deviation) penalty.

## Power of Nonconvex Penalties 

L1 penalized regression (LASSO) is a useful tool for feature selection. However when you use LASSO in very noisy setting, especially when some columns in your data have strong colinearity, LASSO tends to give biased estimator due to the penalty term. As demonstrated in the example below, the lowest estimation error among all the lambdas computed is as high as **16.41%**.

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

The PICASSO package [3, 4] solves non-convex optimization through multi-stage convex relaxation. Although we only find a local minimum, it can be proved that this local minimum does not lose the superior statistcal property of the global minimum. Multi-stage convex relaxation is also much more stable than other packages (see benchmark below).

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
 - Sparse linear regression. picasso achieves similar timing and optimization performance to glmnet and ncvreg.
 - Sparse logistic regression. When using the l1 regularizer, picasso, glmnet and ncvreg achieves similar optimization performance. When using the nonconvex regularizers, picasso achieves significantly better optimization performance than ncvreg, especially in ill-conditioned cases.
 - Scaled sparse linear regression. Picasso significantly outperforms scalreg and flare in timing performance. In Table 5.3 in [tutorials/PICASSO.pdf](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/PICASSO.pdf), picasso is 20 − 100 times faster and achieves smaller objective function values.

Details of our benmarking process is documented in [tutorials/PICASSO.pdf](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/PICASSO.pdf).

![Performance](https://raw.githubusercontent.com/jasonge27/picasso/master/tutorials/images/performance.jpeg)


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

You can test if the package has been successfully installed by ``python -c "import pycasso; pycasso.test() ``

Details for installing python package can also be found in [document](https://hmjianggatech.github.io/picasso/) or [github](https://github.com/jasonge27/picasso/tree/master/python-package)

## Tutorials
Check the R tutorial in tutorials/tutorial.R and Python tutorial in tutorials/tutorial.py. Let us know if anything is hard to use or if you want any other features. 

## References

[1] Jianqing Fan and Runze Li, Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties, 2001

[2] Cun-Hui Zhang, Nearly Unbiased Variable Selection Under Minimax Concave Penalty, 2010

[3] Jason Ge, Mingyi Hong, Mengdi Wang, Han Liu, and Tuo Zhao, Homotopy Active Set Proximal Newton Algorithm for Sparse Learning, 2016

[4] Tuo Zhao, Han Liu, and Tong Zhang, Pathwise Coordinate Optimization for Nonconvex Sparse Learning: Algorithm and Theory, 2014
