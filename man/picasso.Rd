\name{picasso}
\alias{picasso}

\title{
Pathwise Calibrated Sparse Shooting Algorithm
}

\description{
The function "picasso" implements the user interface.
}

\usage{
picasso(X, Y, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL,
        lambda.min = NULL,family = "gaussian", method="l1", alg = "greedy", 
        opt = "naive",gamma = 3, df = NULL, sym = "or", standardize = TRUE, 
        perturb = TRUE, max.act.in = 3, truncation = 0.01, prec = 1e-4, 
        max.ite = 1e3, verbose = TRUE)
}

\arguments{
  \item{X}{For sparse linear regression and sparse logistic regression, \code{X} is an \eqn{n} by \eqn{d} design matrix. For sparse inverse column operator, there are 2 options when \code{family = "graph"}: (1) \code{data} is an \code{n} by \code{d} data matrix (2) a \code{d} by \code{d} sample covariance matrix. The program automatically identifies the input matrix by checking the symmetry. (\code{n} is the sample size and \code{d} is the dimension). }
  \item{Y}{For sparse linear regression and sparse logistic regression, \code{Y} is an \eqn{n} dimensional response vector. For sparse inverse column operation, no input for \code{Y} is needed.}
  \item{lambda}{A sequence of decresing positive values to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda} and \code{lambda.min.ratio}. Users can also specify a sequence to override this. Default value is from \eqn{lambda.max} to \code{lambda.min.ratio*lambda.max}. The default value of \eqn{lambda.max} is the minimum regularization parameter which yields an all-zero estiamtes.}
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 5.}
  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.1}.}
  \item{lambda.min}{The smallest value for \code{lambda}. If $\code{lambda.min.ratio} is provided, then it is set to \code{lambda.min.ratio*MAX}, where \code{MAX} is the uppperbound of the regularization parameter. The default value is \code{0.1*MAX}.}
  \item{family}{Options for model. Sparse linear regression and sparse multivariate regression is applied if \code{family = "gaussian"}, sparse logistic regression is applied if \code{family = "binomial"} and sparse column inverse operation is applied if \code{family = "graph"}. The default value is \code{"gaussian"}.}
  \item{method}{Options for regularization. Lasso is applied if \code{method = "l1"}, MCP is applied if \code{method = "mcp"} and SCAD Lasso is applied if \code{method = "scad"}. The default value is \code{"l1"}.}
  \item{alg}{Options for updating active sets: The cyclic selection rule is applied if \code{alg = "cyclic"}; The greedy selection rule is applied if \code{alg = "greedy"}, the proximal gradient selection rule is applied if \code{alg = "proximal"}; The randomized selection rule is applied if \code{alg = "random"}; The hybrid of the cyclic and greedy selection rules is applied if \code{alg = "hybrid"}. The default value is \code{"greedy"}.}
  \item{opt}{Options for updating residuals. The naive update rule is applied if \code{opt = "naive"}, and the covariance update rule is applied if \code{alg = "cov"}. The default value is \code{"naive"}.}
  \item{gamma}{The concavity parameter for MCP and SCAD. The default value is \code{3}.}
  \item{df}{Maximum degree of freedom for the covariance update. The default value is \code{2*n}.}
  \item{sym}{Symmetrization of output graphs. If \code{sym = "and"}, the edge between node \code{i} and node \code{j} is selected ONLY when both node \code{i} and node \code{j} are selected as neighbors for each other. If \code{sym = "or"}, the edge is selected when either node \code{i} or node \code{j} is selected as the neighbor for each other. The default value is \code{"or"}. ONLY applicable to the sparse column inverse operator.}
  \item{standardize}{Variables are standardized to have mean zero and unit standard deviation if \code{standardize = TRUE}. The default value is \code{TRUE}.}
  \item{perturb}{The diagonal of \code{Sigma} is added by a positive value to guarantee that \code{Sigma} is positive definite if \code{perturb = TRUE}. User can specify a numeric value for \code{perturbe}. The default value is \code{perturb = TRUE}.}
  \item{max.act.in}{The maximum number of active variables to add into the active set when \code{alg = "greedy"}. The default value is 3.}
  \item{truncation}{The critical value for updating active sets when \code{alg = "cyclic"}. The default value is 1e-2.}
  \item{prec}{Stopping precision. The default value is 1e-4.}
  \item{max.ite}{The iteration limit. The default value is 1e4.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{
For sparse linear regression,

\deqn{
    \min_{\beta} {\frac{1}{2n}}|| Y - X \beta ||_2^2 + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, SCAD regularizers.
   
For sparse logistic regression,

\deqn{
    \min_{\beta} {\frac{1}{n}}\sum_{i=1}^n (\log(1+e^{x_i^T \beta}) - y_i x_i^T \beta) + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, and SCAD regularizers.
    
For sparse column inverse operation,
\deqn{
    \min_{\beta} {\frac{1}{2}} \beta^T S \beta - e^T \beta + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP or SCAD  regularizers.

}

\value{
An object with S3 classes \code{"lasso"}, \code{"binomial"}, and \code{"scio"} corresponding to  sparse linear regression, sparse logistic regression, and sparse column inverse operator respectively is returned: 
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
  \item{alg}{
  The \code{alg} from the input. 
  }
  \item{sym}{
  The \code{sym} from the input.
  }
  \item{path}{
  A list of \code{d} by \code{d} adjacency matrices of estimated graphs as a graph path corresponding to \code{lambda}.
  }
  \item{sparsity}{
  The sparsity levels of the graph path for sparse inverse column operator.
  }
  \item{standardize}{The \code{standardize} from the input.}
  \item{perturb}{The \code{perturb} from the input.}
  \item{df}{
  The degree of freecom (number of nonzero coefficients) along the solution path for sparse linear regression, nd sparse logistic regression. 
  }
  \item{ite}{
  A list of vectors where the i-th entries of ite[[1]] and ite[[2]] correspond to the outer iteration and inner iteration of i-th regularization parameter respectively.
  }
  \item{verbose}{The \code{verbose} from the input.}
}

\author{
Xingguo Li, Tuo Zhao, Tong Zhang and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
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
n = 100
d = 400
X = matrix(rnorm(n*d), n, d)
beta = c(3,2,0,1.5,rep(0,d-4))

## Generate response using Gaussian noise, and fit sparse linear models
noise = rnorm(n)
Y = X\%*\%beta + noise
out.l1.cyclic = picasso(X, Y, nlambda=10)
out.l1.greedy = picasso(X, Y, nlambda=10, alg="greedy")
out.mcp.greedy = picasso(X, Y, nlambda=10, method="mcp")

## Visualize the solution path
plot(out.l1.cyclic)
plot(out.l1.greedy)
plot(out.mcp.greedy)


################################################################
## Sparse logistic regression
## Generate the design matrix and regression coefficient vector
n = 100
d = 400
X = matrix(rnorm(n*d), n, d)
beta = c(3,2,0,1.5,rep(0,d-4))

## Generate response and fit sparse logistic models
p = exp(X\%*\%beta)/(1+exp(X\%*\%beta))
Y = rbinom(n,rep(1,n),p)
out.l1.cyclic = picasso(X, Y, nlambda=10, family="binomial")
out.l1.greedy = picasso(X, Y, nlambda=10, family="binomial", alg="greedy")
out.mcp.greedy = picasso(X, Y, nlambda=10, family="binomial", method="mcp")

## Visualize the solution path
plot(out.l1.cyclic)
plot(out.l1.greedy)
plot(out.mcp.greedy)

## Estimate of Bernoulli parameters
p.l1 = out.l1.cyclic$p


################################################################
## Sparse column inverse operator
## generating data
n = 100
d = 200
D = scio.generator(n=n,d=d,graph="band",g=1)
plot(D)

## sparse precision matrix estimation
out1 = picasso(D$data, nlambda=10, family="graph")
plot(out1)
scio.plot(out1$path[[4]])

}