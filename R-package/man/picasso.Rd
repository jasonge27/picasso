\name{picasso}
\alias{picasso}

\title{
Pathwise Calibrated Sparse Shooting Algorithm (PICASSO)
}

\description{
User-facing interface for fitting sparse generalized linear models along a
regularization path.
}

\usage{
picasso(X, Y, lambda = NULL, nlambda = 100, lambda.min.ratio = 0.05,
        family = "gaussian", method = "l1", type.gaussian = "naive",
        gamma = 3, df = NULL, standardize = TRUE, intercept = TRUE,
        prec = 1e-07, max.ite = 1000, verbose = FALSE)
}

\arguments{
  \item{X}{
    Numeric design matrix with \eqn{n} rows (samples) and \eqn{d} columns
    (features).
  }

  \item{Y}{
    Response of length \eqn{n}. Use:
    numeric values for \code{family = "gaussian"} and
    \code{family = "sqrtlasso"};
    binary values (0/1 or a two-level factor) for
    \code{family = "binomial"};
    non-negative integers for \code{family = "poisson"}.
  }

  \item{lambda}{
    Optional decreasing sequence of positive regularization values.
    If \code{NULL}, the path is generated automatically from
    \eqn{\lambda_{\max}} to
    \code{lambda.min.ratio * lambda.max} with \code{nlambda} points.
  }

  \item{nlambda}{
    Number of regularization values in the automatically generated path.
    Ignored when \code{lambda} is supplied.
  }

  \item{lambda.min.ratio}{
    Smallest \code{lambda} as a fraction of \eqn{\lambda_{\max}} when
    \code{lambda} is generated automatically. For nonconvex
    logistic/poisson fits, values below \code{0.05} may lead to unstable or
    ill-conditioned optimization.
  }

  \item{family}{
    Model family. One of \code{"gaussian"}, \code{"binomial"},
    \code{"poisson"}, or \code{"sqrtlasso"}.
  }

  \item{method}{
    Penalty type. One of \code{"l1"}, \code{"mcp"}, or \code{"scad"}.
  }

  \item{type.gaussian}{
    Gaussian-only solver choice: \code{"naive"} or \code{"covariance"}.
  }

  \item{gamma}{
    Concavity parameter for MCP/SCAD penalties.
  }

  \item{df}{
    Gaussian-only maximum active-set size used by the covariance update
    solver. If \code{NULL}, a default is chosen internally.
  }

  \item{standardize}{
    If \code{TRUE}, standardize \code{X} before fitting and map coefficients
    back to the original scale.
  }

  \item{intercept}{
    If \code{TRUE}, include an intercept term.
  }

  \item{prec}{
    Stopping tolerance for inner optimization.
  }

  \item{max.ite}{
    Maximum number of iterations for each \code{lambda}.
  }

  \item{verbose}{
    If \code{TRUE}, print training progress.
  }
}

\details{
For each \code{lambda}, \code{picasso} solves a penalized objective.
\eqn{R(\beta)} denotes \eqn{\ell_1}, MCP, or SCAD depending on
\code{method}.

Gaussian:
\deqn{
\min_{\beta,\beta_0}
\frac{1}{2n}\|Y - X\beta - \beta_0\mathbf{1}\|_2^2 + \lambda R(\beta)
}

Binomial:
\deqn{
\min_{\beta,\beta_0}
\frac{1}{n}\sum_{i=1}^n
\left[\log\left(1 + e^{x_i^\top\beta + \beta_0}\right)
- y_i(x_i^\top\beta + \beta_0)\right]
+ \lambda R(\beta)
}

Poisson:
\deqn{
\min_{\beta,\beta_0}
\frac{1}{n}\sum_{i=1}^n
\left[e^{x_i^\top\beta + \beta_0}
- y_i(x_i^\top\beta + \beta_0)\right]
+ \lambda R(\beta)
}

Sqrt-Lasso:
\deqn{
\min_{\beta,\beta_0}
\frac{1}{\sqrt{n}}\|Y - X\beta - \beta_0\mathbf{1}\|_2 + \lambda R(\beta)
}
}

\value{
An S3 object of class \code{"gaussian"}, \code{"logit"},
\code{"poisson"}, or \code{"sqrtlasso"} (depending on \code{family}),
with fields including:
  \item{beta}{Sparse coefficient matrix of dimension \code{d x nlambda}.}
  \item{intercept}{Intercept values along the \code{lambda} path.}
  \item{lambda}{Regularization sequence used for fitting.}
  \item{nlambda}{Number of regularization values.}
  \item{df}{Degrees of freedom (number of nonzero coefficients) per \code{lambda}.}
  \item{ite}{Iteration diagnostics returned by the solver.}
  \item{method}{Penalty method used for fitting.}
  \item{runtime}{Overall runtime.}

Some families include extra fields, e.g. \code{p} (fitted Bernoulli
probabilities for binomial models) and \code{alg} (algorithm tag).
}

\author{
Jason Ge, Xingguo Li, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\references{
1. J. Friedman, T. Hastie, H. Hofling, and R. Tibshirani. Pathwise coordinate optimization. \emph{The Annals of Applied Statistics}, 2007.\cr
2. C. H. Zhang. Nearly unbiased variable selection under minimax concave penalty. \emph{Annals of Statistics}, 2010.\cr
3. J. Fan and R. Li. Variable selection via nonconcave penalized likelihood and its oracle properties. \emph{Journal of the American Statistical Association}, 2001.\cr
4. R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor, and R. Tibshirani. Strong rules for discarding predictors in lasso-type problems. \emph{Journal of the Royal Statistical Society: Series B}, 2012.\cr
5. T. Zhao, H. Liu, and T. Zhang. A General Theory of Pathwise Coordinate Optimization.
}

\seealso{
\code{\link{picasso-package}}.
}

\examples{
set.seed(2016)
n <- 100
d <- 80
s <- 20
X <- scale(matrix(rnorm(n * d), n, d))
beta <- c(runif(s), rep(0, d - s))

## Gaussian model
Y <- X \%*\% beta + rnorm(n)
fit.g <- picasso(X, Y, family = "gaussian", method = "mcp")
print(fit.g)
plot(fit.g)

## Binomial model
p <- 1 / (1 + exp(-X \%*\% beta))
Yb <- rbinom(n, 1, p)
fit.b <- picasso(X, Yb, family = "binomial", method = "l1")
head(fit.b$df)
}
