# PICASSO
## Installation in R

```R
> library(devtools)
> devtools::install_github("jasonge27/picasso")
```



## Unleash the power of nonconvex penalty

L1 penalized linear regression (LASSO) is great for feature selection in linear regression. However when you use LASSO in very noisy setting, especially when some columns in your data has strong colinearity, it's easy to see that LASSO tends to give biased estimator due to the penalty term. As demonstrated in the example below, the lowest estimation error among all the lambdas computed is as high as **10.589%**.

```R
> set.seed(2016)
> library(glmnet)
> n <- 2000; p <- 1000; c <- 0.1
> # n sample number, p dimension, c correlation parameter
> X <- scale(matrix(rnorm(n*p),n,p)+c*rnorm(n)) # n is smaple number, 
> s <- 20  # sparsity level
> true_beta <- c(runif(s), rep(0, p-s))
> Y <- X%*%true_beta + rnorm(n)
> fitg<-glmnet(X,Y,family="gaussian")
> # the minimal estimation error |\hat{beta}-beta| / |beta|
> min(apply(abs(fitg$beta - true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
[1] 0.10589
```



Nonconvex penalties such as SCAD [1] and MCP [2] are statistically better but computationally harder. The solution for SCAD/MCP penalized linear model has much less estimation error than lasso but calculating the estimator involves non-convex optimization. With limited computation resource, we can only get a local optimum which probably lacks the good property of the global optimum. 

The PICASSO package [3, 4]  solves non-convex optimization through multi-stage convex relaxation. Although we only find a local minimum, it can be proved that this local minimum does not lose the superior statistcal property of the global minimum. Multi-stage convex relaxation is also much more stable than other packages (see benchmark below). 

Let's see PICASSO in action — the estimation error drops to **3.4%** using SCAD penalty from **10.57%** error produced by LASSO.

```R
> library(picasso)
> fitp <- picasso(X, Y, family="gaussian", method="scad")
> min(apply(abs(fitp$beta-true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
[1] 0.03392717
```



## Fast and Stable

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



|                                        |     c = 0.1     |     c = 0.5     |       c = 1.0        |
| :------------------------------------: | :-------------: | :-------------: | :------------------: |
|          PICASSO (time/error)          | 8.727(s) / 4.6% | 5.247(s) / 2.6% |   6.062(s) / 11.2%   |
|          ncvreg (time/error)           | 7.461(s) / 5.6% | 7.056(s) / 6.0% | 51.85(s) / 35.3% [*] |
| Estimation Error using LASSO in glmnet |      2.0%       |      14.1%      |        28.7%         |

'[*]': Package exited with warning: Algorithm failed to converge for some values of lambda.

The experiments are conducted on a MacBook Pro with 2.4GHz Intel Core i5 and 8GB RAM. R version is 3.3.0. The ncvreg version is 3.5-2. The glmnet version is 2.0-5.



References

[1] Jianqing Fan and Runze Li, Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties, 2001

[2] Cun-Hui Zhang, Nearly Unbiased Variable Selection Under Minimax Concave Penalty, 2010

[3] Jason Ge, Mingyi Hong, Mengdi Wang, Han Liu, and Tuo Zhao, Homotopy Active Set Proximal Newton Algorithm for Sparse Learning, 2016

[4] Tuo Zhao, Han Liu, and Tong Zhang, Pathwise Coordinate Optimization for Nonconvex Sparse Learning: Algorithm and Theory, 2014
