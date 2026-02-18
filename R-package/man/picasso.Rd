\name{picasso}
\alias{picasso}

\title{
PathwIse CAlibrated Sparse Shooting algOrithm (PICASSO)
}

\description{
The function "picasso" implements the user interface.
}

\usage{
picasso(X, Y, lambda = NULL, nlambda = 100, lambda.min.ratio =
                 0.05, family = "gaussian", method = "l1",
                 type.gaussian = "naive", gamma = 3, df = NULL,
                 standardize = TRUE, intercept = TRUE, prec = 1e-07,
                 max.ite = 1000, verbose = FALSE)
}

\arguments{
  \item{X}{ \code{X} is an \eqn{n} by \eqn{d} design matrix where \code{n} is the sample size and \code{d} is the data dimension.}
  
  \item{Y}{\code{Y} is the \eqn{n} dimensional response vector. Y is numeric vector for \code{family=``gaussian''} and \code{family=``sqrtlasso''}, or a two-level factor for \code{family=``binomial''}, or a non-negative integer vector representing counts for \code{family = ``gaussian''}. }
  
  \item{lambda}{A sequence of decresing positive values to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda} and \code{lambda.min.ratio}. Users can also specify a sequence to override this. Default value is from \eqn{lambda.max} to \code{lambda.min.ratio*lambda.max}. The default value of \eqn{lambda.max} is the minimum regularization parameter which yields an all-zero estimates.}
  
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 100.}

  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.05}.} Caution: logistic and poisson regression can be ill-conditioned if lambda is too small for nonconvex penalty. We suggest the user to avoid using any lambda.min.raito smaller than 0.05 for logistic/poisson regression under nonconvex penalty.
  
  
  \item{family}{Options for model. Sparse linear regression and sparse multivariate regression is applied if \code{family = "gaussian"}, sqrt lasso is applied if \code{family = "sqrtlasso"}, sparse logistic regression is applied if \code{family = "binomial"} and sparse poisson regression is applied if \code{family = "poisson"}. The default value is \code{"gaussian"}.}
  
  \item{method}{Options for regularization. Lasso is applied if \code{method = "l1"}, MCP is applied if \code{method = "mcp"} and SCAD Lasso is applied if \code{method = "scad"}. The default value is \code{"l1"}.}
  
  \item{type.gaussian}{Options for updating residuals in sparse linear regression. The naive update rule is applied if \code{opt = "naive"}, and the covariance update rule is applied if \code{opt = "covariance"}. The default value is \code{"naive"}.}
  
  \item{gamma}{The concavity parameter for MCP and SCAD. The default value is \code{3}.}
  
  \item{df}{Maximum degree of freedom for the covariance update. The default value is \code{2*n}.}
  
  \item{standardize}{Design matrix X will be standardized to have mean zero and unit standard deviation if \code{standardize = TRUE}. The default value is \code{TRUE}.}
      
  
  \item{intercept}{Does the model has intercept term or not. Default value is \code{TRUE}.}

  \item{prec}{Stopping precision. The default value is 1e-7.}
  
  \item{max.ite}{Max number of iterations for the algorithm. The default value is 1000.}

  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{FALSE}.}
  }
\details{
For sparse linear regression,

\deqn{
    \min_{\beta} {\frac{1}{2n}}|| Y - X \beta - \beta_0||_2^2 + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, SCAD regularizers.
   
For sparse logistic regression,

\deqn{
    \min_{\beta} {\frac{1}{n}}\sum_{i=1}^n (\log(1+e^{x_i^T \beta+ \beta_0}) - y_i x_i^T \beta) + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, and SCAD regularizers. 
    
For sparse poisson regression,
\deqn{
    \min_{\beta} {\frac{1}{n}}\sum_{i=1}^n (e^{x_i^T \beta + \beta_0} - y_i (x_i^T \beta+\beta_0) + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP or SCAD  regularizers.

}

\value{
An object with S3 classes \code{"gaussian"}, \code{"binomial"}, and \code{"poisson"} corresponding to  sparse linear regression, sparse logistic regression, and sparse poisson regression respectively is returned: 
  \item{beta}{
  A matrix of regression estimates whose columns correspond to regularization parameters for sparse linear regression and sparse logistic regression. A list of matrices of regression estimation corresponding to regularization parameters for sparse column inverse operator.
  }
  \item{intercept}{The value of intercepts corresponding to regularization parameters for sparse linear regression, and sparse logistic regression. }
  \item{Y}{The value of \code{Y} used in the program. }
  \item{X}{The value of \code{X} used in the program. }
  \item{lambda}{
  The sequence of regularization parameters \code{lambda} used in the program.
  }
  \item{nlambda}{
  The number of values used in \code{lambda}.
  }
  \item{family}{
  The \code{family} from the input. 
  }
  \item{method}{
  The \code{method} from the input. 
  }
  \item{path}{
  A list of \code{d} by \code{d} adjacency matrices of estimated graphs as a graph path corresponding to \code{lambda}.
  }
  \item{sparsity}{
  The sparsity levels of the graph path for sparse inverse column operator.
  }
  \item{standardize}{The \code{standardize} from the input.}
  \item{df}{
  The degree of freecom (number of nonzero coefficients) along the solution path for sparse linear regression, nd sparse logistic regression. 
  }
  \item{ite}{
  A list of vectors where the i-th entries of ite[[1]] and ite[[2]] correspond to the outer iteration and inner iteration of i-th regularization parameter respectively.
  }
  \item{verbose}{The \code{verbose} from the input.}
}

\author{
Jason Ge, Xingguo Li, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Jason Ge <jiange@princeton.edu>
}

\references{
1. J. Friedman, T. Hastie and H. Hofling and R. Tibshirani. Pathwise coordinate optimization. \emph{The Annals of Applied Statistics}, 2007.\cr
2. C.H. Zhang. Nearly unbiased variable selection under minimax concave penalty. \emph{Annals of Statistics}, 2010.\cr
3. J. Fan and R. Li. Variable selection via nonconcave penalized likelihood and its oracle
properties. \emph{Journal of the American Statistical Association}, 2001.\cr
4. R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor and R. Tibshirani. Strong rules for discarding predictors in lasso-type problems. \emph{Journal of the Royal Statistical Society: Series B}, 2012.\cr
5. T. Zhao, H. Liu, and T. Zhang. A General Theory of Pathwise Coordinate Optimization. Techinical Report, Princeton Univeristy.
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

## l1 regularization solved with covariance update
fitted.l1.covariance  = picasso(X, Y, nlambda=100, type.gaussian="covariance")

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

## Estimate of Bernoulli parameters
param.l1 = fitted.l1$p


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