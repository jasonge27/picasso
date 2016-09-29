# PICASSO
## Installation in R

```R
> library(devtools)
> devtools::install_github("jasonge27/picasso")
```



## Unleash the power of nonconvex penalty

L1 penalized linear regression (LASSO) is great for feature selection in linear regression. However when you use LASSO in very noisy setting, when some columns in your data has strong colinearity or they are just useless noise, it's easy to see that LASSO actually gives biased estimator due to the penalty term. As demonstrated in the example below, the lowest estimation error among all the lambdas computed is as high as **10.589%**.

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



Nonconvex penalties such as SCAD and MCP are statistically better but computationally harder. The solution for SCAD/MCP penalized linear model has much less estimation error than lasso but calculating the estimator involves non-convex optimization. With limited computation resource, we can only get a local optimum which probably lacks the good property of the global optimum. 

The PICASSO package solves non-convex optimization through multi-stage convex relaxation. Although we only find a local minimum, it can be proved that this local minimum does not lose the superior statistcal property of the global minimum. Multi-stage convex relaxation is also much more stable than other packages (see benchmark below). 

Let's see PICASSO in action — the estimation error drops to **3.4%** using SCAD penalty from **10.57%** error produced by LASSO.

```R
> library(picasso)
> fitp <- picasso(X, Y, family="gaussian", method="scad")
> min(apply(abs(fitp$beta-true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))
[1] 0.03392717
```



## Fast and Stable

As a traditional LASSO solver, our package is as fast as the state-of-the-art glmnet solver. For SCAD regularzied linear/logistic regression, it can be shown that our package is much faster and more stable than the alternative ncvreg. (The experiments can also be done with MCP penalty.)

For the experiments, we use sample number n = 2000 and sample dimension p = 1000. The parameter c is used to control the column-wise correlation of the desgin matrix X as before.

Timing for L1 regularized linear regression.

|         |  c=0.1   |  c=0.5   |    c= 1.0    |
| :-----: | :------: | :------: | :----------: |
| PICASSO | 0.296(s) | 0.365(s) |   0.575(s)   |
| glmnet  | 0.276(s) | 0.297(s) |   0.445(s)   |
| ncvreg  | 0.318(s) | 1.818(s) | 9.429(s) [*] |

'[*]': Package exited with warning: Algorithm failed to converge for some values of lambda.

Timing for SCAD regularized linear regression.

|         | c = 0.1  | c = 0.5  |   c = 1.0    |
| :-----: | :------: | :------: | :----------: |
| PICASSO | 0.446(s) | 0.445(s) |   0.712(s)   |
| ncvreg  | 0.374(s) | 0.405(s) | 2.177(s) [*] |

'[*]': Package exited with warning: Algorithm failed to converge for some values of lambda.

Timing for SCAD regularized logistic regression

|         |   c = 0.1    | c = 0.5  | c = 1.0 |
| :-----: | :----------: | :------: | :-----: |
| PICASSO |   1.016(s)   | 0.528(s) |  0.603  |
| ncvreg  | 3.104(s) [*] | 2.778(s) |  5.862  |

'[*]': Package exited with warning: Algorithm failed to converge for some values of lambda.

