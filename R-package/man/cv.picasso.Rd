\name{cv.picasso}
\alias{cv.picasso}
\alias{print.cv.picasso}
\alias{plot.cv.picasso}
\alias{coef.cv.picasso}
\alias{predict.cv.picasso}
\title{Cross-Validation for picasso}
\description{
Performs K-fold cross-validation to select the regularization parameter for
\code{picasso} models.
}
\usage{
cv.picasso(X, Y, \dots, nfolds = 10, foldid = NULL, type.measure = "default")

\method{print}{cv.picasso}(x, \dots)

\method{plot}{cv.picasso}(x, sign.lambda = 1, \dots)

\method{coef}{cv.picasso}(object, s = c("lambda.min", "lambda.1se"), \dots)

\method{predict}{cv.picasso}(object, newdata, s = c("lambda.min", "lambda.1se"),
  type = "response", \dots)
}
\arguments{
  \item{X}{Numeric design matrix, \eqn{n \times d}.}
  \item{Y}{Response vector or matrix.}
  \item{\dots}{Additional arguments passed to \code{\link{picasso}}.}
  \item{nfolds}{Number of cross-validation folds. Default is \code{10}.}
  \item{foldid}{Optional integer vector of fold assignments (length \eqn{n}).
    If \code{NULL}, fold assignments are generated randomly.}
  \item{type.measure}{Loss measure for cross-validation. \code{"default"}
    uses \code{"class"} for binomial and \code{"deviance"} otherwise.
    Additional options: \code{"mse"}, \code{"mae"}, \code{"deviance"},
    \code{"class"}.}
  \item{x}{A \code{"cv.picasso"} object.}
  \item{object}{A \code{"cv.picasso"} object.}
  \item{s}{Which lambda to use: \code{"lambda.min"} (default) selects the
    lambda that minimises CV error; \code{"lambda.1se"} uses the largest
    lambda within one standard error of the minimum.}
  \item{sign.lambda}{Multiplier for the x-axis of the CV plot (\code{1} for
    \eqn{\log(\lambda)}, \code{-1} for \eqn{-\log(\lambda)}).}
  \item{newdata}{Matrix of new observations for prediction.}
  \item{type}{Prediction type; passed to the underlying \code{predict}
    method.}
}
\value{
\code{cv.picasso} returns an object of class \code{"cv.picasso"} with components:
\describe{
  \item{lambda}{Lambda sequence from the full-data fit.}
  \item{cvm}{Mean cross-validated error for each lambda.}
  \item{cvsd}{Standard deviation of cross-validated errors.}
  \item{cvup, cvlo}{Upper and lower error-bar bounds.}
  \item{nzero}{Number of nonzero coefficients at each lambda.}
  \item{lambda.min}{Lambda that minimises \code{cvm}.}
  \item{lambda.1se}{Largest lambda within one standard error of the minimum.}
  \item{name}{Name of the loss measure used.}
  \item{picasso.fit}{Full-data \code{picasso} fit.}
}
}
\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}
\seealso{\code{\link{picasso}}}
\examples{
set.seed(1)
n <- 100; d <- 50
X <- matrix(rnorm(n * d), n, d)
Y <- X[, 1:5] \%*\% rnorm(5) + rnorm(n)
cv <- cv.picasso(X, Y, family = "gaussian", nlambda = 20)
print(cv$lambda.min)
}
