\name{picasso}
\alias{picasso}

\title{
Fit Sparse Regression Models with Convex and Non-Convex Penalties
}

\description{
\code{picasso} fits regularization paths for sparse Gaussian, logistic
(\code{family = "binomial"}), Poisson, sqrt-lasso, and multinomial models
using lasso, MCP, or SCAD penalties.
}

\usage{
picasso(X, Y, lambda = NULL, nlambda = 100, lambda.min.ratio =
                 0.05, family = "gaussian", method = "l1",
                 type.gaussian = "auto", gamma = 3, df = NULL,
                 dfmax = NULL, standardize = TRUE, intercept = TRUE,
                 prec = 1e-07, max.ite = 1000, verbose = FALSE,
                 offset = NULL, lla.max.stages = 3L,
                 fast.mode = FALSE)
}

\arguments{
  \item{X}{
  Numeric design matrix with \eqn{n} rows (samples) and \eqn{d} columns (features).
  }

  \item{Y}{
  Response vector of length \eqn{n}. Use finite numeric responses for
  \code{family = "gaussian"} and \code{family = "sqrtlasso"}, exactly two
  levels for \code{family = "binomial"}, non-negative integer counts with at
  least one positive count for \code{family = "poisson"}, and at least three
  observed class labels for \code{family = "multinomial"}. Binomial and
  multinomial labels may be numeric, character, or factor values.
  Multinomial fits drop unused factor levels; binomial factor inputs must
  already contain exactly two levels.
  }

  \item{lambda}{
  Optional finite, non-negative, strictly decreasing sequence of
  regularization values. If \code{NULL}, the sequence is generated
  automatically from \code{nlambda} and \code{lambda.min.ratio}.
  }

  \item{nlambda}{
  Positive integer number of regularization values to generate when
  \code{lambda = NULL}. The default is \code{100}.
  }

  \item{lambda.min.ratio}{
  Smallest generated \code{lambda} value as a fraction of \eqn{\lambda_{\max}}
  when \code{lambda = NULL}. It must lie strictly between zero and one; the
  default is \code{0.05}. For non-convex
  penalties in logistic and Poisson models, values much smaller than
  \code{0.05} may lead to ill-conditioning.
  }

  \item{family}{
  Model family. Supported values are \code{"gaussian"}, \code{"binomial"},
  \code{"poisson"}, \code{"sqrtlasso"}, and \code{"multinomial"}.
  The default is \code{"gaussian"}.
  }

  \item{method}{
  Penalty type. Supported values are \code{"l1"} (lasso), \code{"mcp"}, and
  \code{"scad"}. The default is \code{"l1"}.
  }

  \item{type.gaussian}{
  Gaussian solver mode. \code{"naive"} maintains residuals and performs
  O(n) work per coordinate. \code{"covariance"} lazily caches Gram-matrix
  columns and then updates gradients in O(d) per coordinate. The default,
  \code{"auto"}, uses covariance updates only when an at-least-eight-point
  path can amortize the cache and the problem is small, path-sparse, or
  sufficiently tall; wide designs and designs above 1024 features use naive
  updates. Within those guards it selects covariance for \eqn{d <= 160},
  \eqn{\lambda_{min}/\lambda_{max} >= 0.1}, a path ratio of at least 0.05
  with \eqn{n >= 4d}, or any problem with \eqn{n >= 16d}. The
  1024-feature guard
  caps a fully populated numeric cache at 8 MiB. Explicit \code{"naive"} and
  \code{"covariance"} requests bypass automatic selection.
  }

  \item{gamma}{
  Concavity parameter for MCP and SCAD penalties. MCP requires
  \code{gamma > 1} and SCAD requires \code{gamma > 2}. The default is
  \code{3}.
  }

  \item{df}{
  Reserved for backward compatibility with older Gaussian covariance-update
  implementations. It is currently not used by the active solver backend.
  }

  \item{dfmax}{
  Nonzero-count threshold for early stopping on the lambda path. The crossing
  model is committed before stopping, so this is not a hard upper bound. If
  \code{NULL} (default), no count limit is imposed. Scalar-family solvers
  check the threshold after the minimum five-point path prefix; multinomial
  checks it after each committed lambda and counts nonzero class-feature
  entries. Gaussian, binomial, Poisson, and sqrt-lasso paths also apply their
  family-specific deviance saturation checks after at least 5 lambda values.
  Automatically generated multinomial paths stop when explained deviance
  exceeds 0.999 or its gain from the preceding lambda is below 1e-5. Explicit
  multinomial lambda sequences disable only that saturation rule; they may
  still return a prefix after \code{dfmax} stopping or a hard failure.
  }

  \item{standardize}{
  If \code{TRUE}, scale columns of \code{X} before fitting. Columns are
  centered only when \code{intercept = TRUE}; a no-intercept model preserves
  the original zero point. The default is \code{TRUE}.
  }

  \item{intercept}{
  If \code{TRUE}, include an intercept term. The default is \code{TRUE}.
  }

  \item{prec}{
  Convergence tolerance used when \code{fast.mode = FALSE}. The default is
  \code{1e-7}. A custom positive value may be supplied in high-precision mode.
  }

  \item{max.ite}{
  Maximum number of native solver iterations. For Gaussian models it bounds
  coordinate-descent sweeps at each lambda. For binomial, Poisson,
  sqrt-lasso, and multinomial models, it bounds active-set subproblem work
  within each weighted-L1 problem and is distinct from
  \code{lla.max.stages}. The default is \code{1000}.
  }

  \item{lla.max.stages}{
  Maximum total number of local-linear-approximation (LLA) stages for MCP
  and SCAD binomial, Poisson, sqrt-lasso, and multinomial fits. The count
  includes the initial L1 master, so the default \code{3L} means one L1
  master plus two weighted-L1 updates. Three stages are generally enough to
  obtain a useful model. Set a higher value when stricter stationarity is
  needed; after the minimum three stages, adaptive LLA stops once
  \code{prec} is met or the requested maximum is exhausted. The input is
  still validated for L1 fits but does not change their optimization. It is
  ignored by the direct Gaussian MCP/SCAD coordinate-descent solver.
  }

  \item{verbose}{
  If \code{TRUE}, print tracing information during fitting.
  The default is \code{FALSE}.
  }

  \item{offset}{
  Optional numeric vector of length \eqn{n} added to the linear predictor
  before the link function. Supported for \code{family = "binomial"} and
  \code{family = "poisson"}. Pass \code{NULL} (default) for no offset.
  Response/link predictions, assessment, and binomial class prediction from
  an offset-fitted model require a corresponding \code{newoffset} vector.
  Support extraction with \code{type = "nonzero"} does not.
  }

  \item{fast.mode}{
  Logical precision preset. The default \code{FALSE} preserves PICASSO's
  high-precision tolerance (\code{prec = 1e-7}). Set to \code{TRUE} to use
  benchmark-calibrated glmnet-like stopping/KKT accuracy: \code{4e-4} for
  Poisson and \code{1e-4} for binomial, sqrt-lasso, and multinomial models.
  Gaussian retains \code{1e-7} because its scaled objective-change test already
  follows glmnet's convention; loosening it to \code{1e-4} was not
  accuracy-equivalent.
  Because stopping criteria differ by solver, these are achieved-accuracy
  presets, not a literal copy of glmnet's \code{thresh}. Supply custom
  \code{prec} values only when this option is \code{FALSE}.
  }
}
\details{
When \code{lambda} is not supplied, \code{picasso} constructs a logarithmically
spaced path between \eqn{\lambda_{\max}} and
\eqn{\lambda_{\min} = \mathrm{lambda.min.ratio} \cdot \lambda_{\max}}.

The method solves a penalized optimization problem:
\deqn{
\min_{\beta_0, \beta}\; \mathcal{L}(\beta_0,\beta) + \sum_{j=1}^d p_{\lambda,\gamma}(|\beta_j|),
}
where \eqn{p_{\lambda,\gamma}} is lasso, MCP, or SCAD.
For multinomial models, the penalty term is instead
\eqn{\sum_{k=1}^K\sum_{j=1}^d
p_{\lambda,\gamma}(|B_{jk}|)}; intercepts are never penalized.

Loss functions by family are:
\itemize{
  \item \code{"gaussian"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{2n}\lVert Y-\beta_0\mathbf{1}-X\beta \rVert_2^2}.
  \item \code{"binomial"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{n}\sum_{i=1}^n[\log(1+\exp(\eta_i))-y_i\eta_i]}.
  \item \code{"poisson"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{n}\sum_{i=1}^n[\exp(\eta_i)-y_i\eta_i]}.
  \item \code{"sqrtlasso"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{\sqrt{n}}\lVert Y-\beta_0\mathbf{1}-X\beta \rVert_2}.
  \item \code{"multinomial"}:
  \eqn{\mathcal{L}(b,B)=-\frac{1}{n}\sum_{i=1}^n\log\left(\frac{\exp(b_{y_i}+x_i^\top B_{y_i})}{\sum_{k=1}^K\exp(b_k+x_i^\top B_k)}\right)}.
}
For binomial and Poisson models, \eqn{\eta_i=\beta_0+x_i^\top\beta+o_i},
where \eqn{o_i} is the supplied offset or zero.

Gaussian paths use active-set coordinate descent with residual or lazy
covariance updates. Binomial and Poisson weighted-L1 subproblems use
active-set Proximal Newton/IRLS, square-root-lasso uses a global quadratic
majorizer with active-set coordinate updates, and multinomial uses
class-coupled active-set Proximal Newton/IRLS.
MCP and SCAD use adaptive LLA around those weighted-L1 solvers except for the
direct Gaussian nonconvex coordinate-descent implementation.
}

\value{
A fitted object of class \code{"gaussian"}, \code{"logit"},
\code{"poisson"}, \code{"sqrtlasso"}, or \code{"multinomial"} depending on
\code{family}.

Common components include:
\describe{
  \item{beta}{A \pkg{Matrix} object of dimension
  \eqn{d \times \mathrm{nlambda}} containing fitted coefficients. Its concrete
  representation may be sparse or dense. Multinomial fits instead return a
  list of \eqn{K} such objects, one per class.}
  \item{intercept}{Numeric vector of intercepts. Multinomial fits return a
  list of \eqn{K} numeric vectors, one per class.}
  \item{lambda}{Regularization sequence used for fitting.}
  \item{nlambda}{Number of regularization values.}
  \item{df}{Degrees of freedom (number of nonzero coefficients) along the path.}
  \item{nulldev}{Null-model loss or deviance under the fitted intercept
  convention and family-specific scale.}
  \item{dev.ratio}{Fraction of null deviance explained at each lambda,
  analogous to \eqn{R^2} for Gaussian fits. Values are clipped to zero through
  one.}
  \item{method}{Penalty type used for fitting.}
  \item{alg}{Identifier for the resolved native algorithm.}
  \item{ite}{Iteration information returned by the solver.}
  \item{verbose}{Input \code{verbose} value.}
  \item{runtime}{Elapsed fitting time.}
  \item{family}{Input \code{family} value.}
  \item{fast.mode}{Logical precision preset used for fitting.}
  \item{prec}{Effective convergence tolerance used by the native solver.}
  \item{type.gaussian.requested}{For Gaussian fits, the requested solver mode.}
  \item{type.gaussian}{For Gaussian fits, the resolved native solver mode.}
  \item{status}{Native termination label. Gaussian iteration-limit failures
  retain only the previously converged lambda prefix.}
  \item{status.code}{Integer form of \code{status}.}
  \item{failure}{Details for the first uncommitted hard failure, otherwise
  \code{NULL}.}
}

Binomial, Poisson, sqrt-lasso, and multinomial fits additionally include:
\describe{
  \item{runt}{Native per-lambda runtime.}
  \item{lla.max.stages}{Requested adaptive-LLA stage budget.}
  \item{diagnostics}{Per-lambda runtime, penalized objective, weighted-L1 KKT
  residual, and target stationarity. Scalar-family diagnostics also report
  solver iterations, active-set size, and completed LLA stages; multinomial
  diagnostics instead report outer iterations, inner sweeps, coordinate
  updates, and active-set size.}
}

For non-Gaussian MCP/SCAD fits, status \code{lla_stationarity_limit} means
that every returned model is usable, but at least one point reached its
LLA-stage budget before meeting \code{prec}.

Multinomial fits also include \code{path.early.stopped} and
\code{requested.nlambda}. A successful automatically generated path may
contain fewer lambdas than requested after its deviance has saturated.
Explicit paths do not apply saturation stopping, but may still be truncated
by \code{dfmax} or a hard failure. Every retained weighted-L1
subproblem meets its KKT tolerance; for MCP/SCAD, inspect target stationarity
when \code{status.code == 10}.

Binomial and Poisson fits include \code{offset.used}, a logical flag recording
whether \code{offset} was explicitly supplied. Binomial fits include the two
response \code{levels} used for zero/one encoding; multinomial fits include
\code{K} and the retained class \code{levels}.
}

\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\references{
1. J. Friedman, T. Hastie, H. Hofling, and R. Tibshirani. Pathwise coordinate optimization. \emph{The Annals of Applied Statistics}, 2007.\cr
2. C.-H. Zhang. Nearly unbiased variable selection under minimax concave penalty. \emph{The Annals of Statistics}, 2010.\cr
3. J. Fan and R. Li. Variable selection via nonconcave penalized likelihood and its oracle properties. \emph{Journal of the American Statistical Association}, 2001.\cr
4. R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor, and R. Tibshirani. Strong rules for discarding predictors in lasso-type problems. \emph{Journal of the Royal Statistical Society: Series B}, 2012.\cr
5. J. Ge, X. Li, H. Jiang, H. Liu, T. Zhang, M. Wang, and T. Zhao. PICASSO: A sparse learning library for high-dimensional data analysis in R and Python. \emph{Journal of Machine Learning Research}, 20(44):1--5, 2019. \url{https://www.jmlr.org/papers/v20/17-722.html}.
}

\seealso{
\code{\link{picasso-package}}.
}

\examples{
################################################################
## Sparse linear regression
## Generate the design matrix and regression coefficient vector
n = 100 # sample number 
d = 80 # sample dimension
c = 0.5 # correlation parameter
s = 20  # support size of coefficient
set.seed(2016)
X = scale(matrix(rnorm(n*d),n,d)+c*rnorm(n))/sqrt(n-1)*sqrt(n)
beta = c(runif(s), rep(0, d-s))

## Generate response using Gaussian noise, and fit sparse linear models
noise = rnorm(n)
Y = X\%*\%beta + noise

## automatic Gaussian backend selection (the default)
fitted.l1.auto = picasso(X, Y, nlambda=100)

## l1 regularization solved with naive update
fitted.l1.naive = picasso(X, Y, nlambda=100, type.gaussian="naive")

## covariance updates can be faster when n >> d
fitted.l1.covariance  = picasso(X, Y, nlambda=100, type.gaussian="covariance")

## early stopping: stop when more than 10 nonzero coefficients
fitted.l1.dfmax = picasso(X, Y, nlambda=100, dfmax=10)

## mcp regularization
fitted.mcp = picasso(X, Y, nlambda=100, method="mcp")

## scad regularization
fitted.scad = picasso(X, Y, nlambda=100, method="scad")

## lambdas used 
print(fitted.l1.naive$lambda)

## number of nonzero coefficients for each lambda
print(fitted.l1.naive$df)

## coefficients and intercept for the i-th lambda
i = 30
print(fitted.l1.naive$lambda[i])
print(fitted.l1.naive$beta[,i])
print(fitted.l1.naive$intercept[i])

## Visualize the solution path
plot(fitted.l1.naive)
plot(fitted.l1.covariance)
plot(fitted.mcp)
plot(fitted.scad)


################################################################
## Sparse logistic regression
## Generate the design matrix and regression coefficient vector
n <- 100  # sample number 
d <- 80   # sample dimension
c <- 0.5   # parameter controlling the correlation between columns of X
s <- 20    # support size of coefficient
set.seed(2016)
X <- scale(matrix(rnorm(n*d),n,d)+c*rnorm(n))/sqrt(n-1)*sqrt(n)
beta <- c(runif(s), rep(0, d-s))

## Generate response and fit sparse logistic models
p = 1/(1+exp(-X\%*\%beta))
Y = rbinom(n, rep(1,n), p)

## l1 regularization
fitted.l1 = picasso(X, Y, nlambda=100, family="binomial", method="l1")

## mcp regularization
fitted.mcp = picasso(X, Y, nlambda=100, family="binomial", method="mcp")

## scad regularization
fitted.scad = picasso(X, Y, nlambda=100, family="binomial", method="scad")

## lambdas used 
print(fitted.l1$lambda)

## number of nonzero coefficients for each lambda
print(fitted.l1$df)

## coefficients and intercept for the i-th lambda
i = 30
print(fitted.l1$lambda[i])
print(fitted.l1$beta[,i])
print(fitted.l1$intercept[i])

## Visualize the solution path
plot(fitted.l1)

## Predicted probabilities for the first 5 observations at lambda[30]
param.l1 = predict(fitted.l1, X[1:5, ], lambda.idx = 30,
                   type = "response")


################################################################
## Sparse poisson regression
## Generate the design matrix and regression coefficient vector
n <- 100  # sample number 
d <- 80   # sample dimension
c <- 0.5   # parameter controlling the correlation between columns of X
s <- 20    # support size of coefficient
set.seed(2016)
X <- scale(matrix(rnorm(n*d),n,d)+c*rnorm(n))/sqrt(n-1)*sqrt(n)
beta <- c(runif(s), rep(0, d-s))/sqrt(s)

## Generate response and fit sparse poisson models
p = X\%*\%beta+rnorm(n)
Y = rpois(n, exp(p))

## l1 regularization
fitted.l1 = picasso(X, Y, nlambda=100, family="poisson", method="l1")

## mcp regularization
fitted.mcp = picasso(X, Y, nlambda=100, family="poisson", method="mcp")

## scad regularization
fitted.scad = picasso(X, Y, nlambda=100, family="poisson", method="scad")

## lambdas used 
print(fitted.l1$lambda)

## number of nonzero coefficients for each lambda
print(fitted.l1$df)

## coefficients and intercept for the i-th lambda
i = 30
print(fitted.l1$lambda[i])
print(fitted.l1$beta[,i])
print(fitted.l1$intercept[i])

## Visualize the solution path
plot(fitted.l1)


################################################################
## Square-root-lasso
set.seed(2)
X <- matrix(rnorm(80 * 12), 80, 12)
Y <- X[, 1] - X[, 2] + rnorm(80)
fitted.sqrt <- picasso(
  X, Y, family = "sqrtlasso", nlambda = 12, method = "l1"
)
predict(fitted.sqrt, X[1:3, ], lambda.idx = fitted.sqrt$nlambda)


################################################################
## Multinomial regression with retained class labels
score <- cbind(X[, 1], -X[, 1] + X[, 2], -X[, 2])
Y <- factor(c("red", "green", "blue")[max.col(score)])
fitted.multi <- picasso(
  X, Y, family = "multinomial", nlambda = 12, method = "l1"
)
predict(fitted.multi, X[1:3, ], lambda.idx = fitted.multi$nlambda,
        type = "class")
}
