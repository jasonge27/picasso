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
                 type.gaussian = "naive", gamma = 3, df = NULL,
                 dfmax = NULL, standardize = TRUE, intercept = TRUE,
                 prec = 1e-07, max.ite = 1000, verbose = FALSE,
                 offset = NULL)
}

\arguments{
  \item{X}{
  Numeric design matrix with \eqn{n} rows (samples) and \eqn{d} columns (features).
  }

  \item{Y}{
  Response vector of length \eqn{n}. Use numeric responses for
  \code{family = "gaussian"} and \code{family = "sqrtlasso"}, a two-level
  response for \code{family = "binomial"}, and non-negative integer counts
  for \code{family = "poisson"}.
  }

  \item{lambda}{
  Optional sequence of decreasing regularization values. If \code{NULL}, the
  sequence is generated automatically from \code{nlambda} and
  \code{lambda.min.ratio}.
  }

  \item{nlambda}{
  Number of regularization values to generate when \code{lambda = NULL}.
  The default is \code{100}.
  }

  \item{lambda.min.ratio}{
  Smallest generated \code{lambda} value as a fraction of \eqn{\lambda_{\max}}
  when \code{lambda = NULL}. The default is \code{0.05}. For non-convex
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
  Gaussian solver mode. \code{"naive"} maintains residuals and updates them
  in O(n) per coordinate; \code{"covariance"} pre-computes the Gram matrix
  and updates gradients in O(d) per coordinate, which is faster when
  \eqn{n \gg d}. The default is \code{"naive"}.
  }

  \item{gamma}{
  Concavity parameter for MCP and SCAD penalties. The default is \code{3}.
  }

  \item{df}{
  Reserved for backward compatibility with older Gaussian covariance-update
  implementations. It is currently not used by the active solver backend.
  }

  \item{dfmax}{
  Maximum number of nonzero coefficients for early stopping on the lambda
  path. When the number of nonzero coefficients exceeds \code{dfmax}, the
  solver stops and returns only the solutions computed so far. If
  \code{NULL} (default), no limit is imposed. In addition to this
  user-controlled limit, the solver automatically stops when the deviance
  ratio exceeds 0.999 or the relative deviance change falls below 1e-5
  (checked after at least 5 lambda values).
  }

  \item{standardize}{
  If \code{TRUE}, standardize columns of \code{X} before fitting.
  The default is \code{TRUE}.
  }

  \item{intercept}{
  If \code{TRUE}, include an intercept term. The default is \code{TRUE}.
  }

  \item{prec}{
  Convergence tolerance. The default is \code{1e-7}.
  }

  \item{max.ite}{
  Maximum number of iterations. The default is \code{1000}.
  }

  \item{verbose}{
  If \code{TRUE}, print tracing information during fitting.
  The default is \code{FALSE}.
  }

  \item{offset}{
  Optional numeric vector of length \eqn{n} added to the linear predictor
  before the link function. Supported for \code{family = "binomial"} and
  \code{family = "poisson"}. Pass \code{NULL} (default) for no offset.
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

Loss functions by family are:
\itemize{
  \item \code{"gaussian"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{2n}\lVert Y-\beta_0\mathbf{1}-X\beta \rVert_2^2}.
  \item \code{"binomial"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{n}\sum_{i=1}^n[\log(1+\exp(\beta_0+x_i^\top\beta))-y_i(\beta_0+x_i^\top\beta)]}.
  \item \code{"poisson"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{n}\sum_{i=1}^n[\exp(\beta_0+x_i^\top\beta)-y_i(\beta_0+x_i^\top\beta)]}.
  \item \code{"sqrtlasso"}:
  \eqn{\mathcal{L}(\beta_0,\beta)=\frac{1}{\sqrt{n}}\lVert Y-\beta_0\mathbf{1}-X\beta \rVert_2}.
}
}

\value{
A fitted object of class \code{"gaussian"}, \code{"logit"},
\code{"poisson"}, \code{"sqrtlasso"}, or \code{"multinomial"} depending on
\code{family}.

Common components include:
\describe{
  \item{beta}{Sparse matrix (or list of sparse matrices for \code{"multinomial"})
  of fitted coefficients. Columns correspond to regularization parameters.}
  \item{intercept}{Intercept values along the solution path.}
  \item{lambda}{Regularization sequence used for fitting.}
  \item{nlambda}{Number of regularization values.}
  \item{df}{Degrees of freedom (number of nonzero coefficients) along the path.}
  \item{nulldev}{Null-model deviance (intercept-only fit).}
  \item{dev.ratio}{Fraction of null deviance explained at each lambda,
  analogous to \eqn{R^2}. Values range from 0 to 1.}
  \item{method}{Penalty type used for fitting.}
  \item{ite}{Iteration information returned by the solver.}
  \item{verbose}{Input \code{verbose} value.}
  \item{runtime}{Elapsed fitting time.}
  \item{family}{Input \code{family} value.}
}
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
5. J. Ge, X. Li, H. Jiang, M. Wang, T. Zhang, H. Liu, and T. Zhao. PICASSO: A sparse learning library for high-dimensional data analysis in R and Python. \emph{arXiv preprint} \url{https://arxiv.org/abs/2006.15261}.
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

## l1 regularization solved with naive update
fitted.l1.naive = picasso(X, Y, nlambda=100, type.gaussian="naive")

## covariance update (faster when n >> d)
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
param.l1 = predict(fitted.l1, X, lambda.idx = 30, type = "response")


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
}
