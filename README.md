# PICASSO
## Unleash the power of nonconvex penalty

L1 penalized linear regression (LASSO) has been one of the most popular tools for feature selection in linear regression. What's the caveats for using LASSO? Unstableness in multi-colinearity and estimation bais due to the penalty term. 

Nonconvex penalties such as SCAD and MCP are statistically better but computationally harder. The solution for SCAD/MCP penalized linear model has much less estimation error than lasso but unfortunately there hasn't been any reliable solver for people to use so far.

The PICASSO package solves non-convex optimization through multi-stage convex relaxation. Although we only find a local minimum, it can be proved that this local minimum does not lose the superior statistcal property of the global minimu.

## Fast and stable

As a traditional LASSO solver, our package is as fast as the state-of-the-art glmnet solver. For SCAD regularzied linear/logistic regression, it can be shown that our package is much faster and more stable than the alternative ncvreg.

For the experiments, we use sample number n = 2000 and sample dimension d = 1000. The parameter c is used to control the column-wise correlation of the desgin matrix X.

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
| PICASSO |   1.016(s)   | 0.528(s) |  0.503  |
| ncvreg  | 3.104(s) [*] | 2.778(s) |  5.862  |

'[*]': Package exited with warning: Algorithm failed to converge for some values of lambda.

