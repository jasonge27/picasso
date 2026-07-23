\name{assess.picasso}
\alias{assess.picasso}
\alias{print.assess.picasso}
\alias{confusion.picasso}
\title{Model Assessment for picasso Fits}
\description{
Compute prediction-quality metrics across the regularization path.
\code{assess.picasso} evaluates deviance, MSE, MAE, or classification error
on a test set; \code{confusion.picasso} returns confusion matrices for
binary and multinomial classification.
For scalar families, large lambda paths are evaluated in bounded link-predictor
blocks; small paths still use one matrix multiplication.
}
\usage{
assess.picasso(object, newx, newy, newoffset = NULL, \dots)

\method{print}{assess.picasso}(x, \dots)

confusion.picasso(object, newx, newy, lambda.idx = NULL,
  newoffset = NULL, \dots)
}
\arguments{
  \item{object}{A fitted \code{picasso} object with a \code{$family} field.}
  \item{newx}{Numeric design matrix for the test set, \eqn{n \times d}.}
  \item{newy}{Test-set response vector. Binomial and multinomial values must
    match class labels seen during fitting. Numeric zero/one remains accepted
    as an encoded binomial response; legacy fits without a stored class map
    require that encoding.}
  \item{lambda.idx}{Integer vector of lambda indices for which to compute
    confusion matrices. Defaults to all lambdas.}
  \item{newoffset}{Optional finite numeric vector with one offset per row of
    \code{newx}. It is required for binomial or Poisson assessment, and for
    binomial confusion matrices, when the model was fitted with an offset.}
  \item{x}{An \code{"assess.picasso"} object.}
  \item{\dots}{Currently unused.}
}
\value{
\code{assess.picasso} returns an object of class \code{"assess.picasso"}
containing named numeric vectors (one value per lambda):
\describe{
  \item{lambda}{Lambda values.}
  \item{deviance}{Family-specific test loss: half mean squared error for
    Gaussian and square-root-lasso, Bernoulli or multinomial negative
    log-likelihood for categorical models, and conventional mean Poisson
    deviance for Poisson.}
  \item{mse}{Mean squared error (Gaussian, square-root-lasso, or Poisson).}
  \item{mae}{Mean absolute error (Gaussian or square-root-lasso).}
  \item{class}{Misclassification rate (binomial or multinomial).}
}

\code{confusion.picasso} returns a list of \code{table} objects, one per
selected lambda value, with predicted classes in rows and observed classes in
columns. Multinomial tables retain all fitted class levels on both axes,
including levels absent from the supplied test subset; binomial tables always
retain levels 0 and 1.

\code{print.assess.picasso} prints the range of each metric across the fitted
path and returns its input invisibly.

Scalar-family assessment targets about 8 MiB for each link-predictor and
coefficient block, subject to a minimum of one lambda column. Metric
calculation may use additional arrays of the same bounded block shape. This
changes only working-memory use, not the returned metrics or their ordering.
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
