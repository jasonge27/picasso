\name{assess.picasso}
\alias{assess.picasso}
\alias{print.assess.picasso}
\alias{confusion.picasso}
\title{Model Assessment for picasso Fits}
\description{
Compute prediction-quality metrics across the regularization path.
\code{assess.picasso} evaluates deviance, MSE, MAE, or classification error
on a test set; \code{confusion.picasso} returns confusion matrices for
binary classification.
}
\usage{
assess.picasso(object, newx, newy, \dots)

\method{print}{assess.picasso}(x, \dots)

confusion.picasso(object, newx, newy, lambda.idx = NULL, \dots)
}
\arguments{
  \item{object}{A fitted \code{picasso} object with a \code{$family} field.}
  \item{newx}{Numeric design matrix for the test set, \eqn{n \times d}.}
  \item{newy}{Test-set response vector.}
  \item{lambda.idx}{Integer vector of lambda indices for which to compute
    confusion matrices. Defaults to all lambdas.}
  \item{x}{An \code{"assess.picasso"} object.}
  \item{\dots}{Currently unused.}
}
\value{
\code{assess.picasso} returns an object of class \code{"assess.picasso"}
containing named numeric vectors (one value per lambda):
\describe{
  \item{lambda}{Lambda values.}
  \item{deviance}{Test-set negative log-likelihood.}
  \item{mse}{Mean squared error (Gaussian/Poisson).}
  \item{mae}{Mean absolute error (Gaussian).}
  \item{class}{Misclassification rate (binomial).}
}

\code{confusion.picasso} returns a list of \code{table} objects, one per
selected lambda value.
}
\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}
\seealso{\code{\link{picasso}}, \code{\link{cv.picasso}}}
\examples{
set.seed(1)
n <- 100; d <- 30
X <- matrix(rnorm(n * d), n, d)
Y <- rbinom(n, 1, 0.5)
fit <- picasso(X, Y, family = "binomial", nlambda = 20)
a <- assess.picasso(fit, X, Y)
print(a)
}
