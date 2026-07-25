\name{cv.picasso}
\alias{cv.picasso}
\alias{print.cv.picasso}
\alias{plot.cv.picasso}
\alias{coef.cv.picasso}
\alias{predict.cv.picasso}
\title{Cross-Validation for picasso}
\description{
Performs K-fold cross-validation to select the regularization parameter for
\code{picasso} models. Multinomial models use a fixed full-data class map and
softmax negative log-likelihood or misclassification loss.
}
\usage{
cv.picasso(X, Y, \dots, nfolds = 10, foldid = NULL,
  type.measure = "default", fast.mode = FALSE)

\method{print}{cv.picasso}(x, \dots)

\method{plot}{cv.picasso}(x, sign.lambda = 1, \dots)

\method{coef}{cv.picasso}(object, s = c("lambda.min", "lambda.1se"), \dots)

\method{predict}{cv.picasso}(object, newdata, s = c("lambda.min", "lambda.1se"),
  type = "response", \dots)
}
\arguments{
  \item{X}{Numeric design matrix, \eqn{n \times d}.}
  \item{Y}{Response vector. A one-column Gaussian response matrix is accepted
    for compatibility.}
  \item{\dots}{Additional arguments passed to \code{\link{picasso}}.}
  \item{nfolds}{Number of cross-validation folds. Default is \code{10}.}
  \item{foldid}{Optional integer vector of fold assignments (length \eqn{n}).
    Labels must be consecutive positive integers starting at one. If
    \code{NULL}, assignments are generated randomly; binomial and multinomial
    assignments are stratified by class. (The Python
    \code{cross_validate} counterpart instead expects zero-based labels,
    matching its zero-based lambda indices.)}
  \item{type.measure}{Loss measure for cross-validation. \code{"default"}
    uses \code{"class"} for binomial and multinomial models and
    \code{"deviance"} otherwise. Multinomial models support only
    \code{"class"} and \code{"deviance"}.
    Additional options: \code{"mse"}, \code{"mae"}, \code{"deviance"},
    \code{"class"}. MSE and MAE are evaluated on the response scale: fitted
    probabilities for binomial models and fitted means for Poisson models.}
  \item{fast.mode}{Logical precision preset passed consistently to the
    full-data and fold fits. With the default \code{FALSE}, \code{prec}
    defaults to \code{1e-7} but may be supplied through \code{\dots};
    \code{TRUE} uses the benchmark-calibrated fast tolerance \code{4e-4} for
    Poisson and \code{1e-4} for other non-Gaussian families. Gaussian retains
    its glmnet-aligned \code{1e-7} objective-change tolerance.}
  \item{x}{A \code{"cv.picasso"} object.}
  \item{object}{A \code{"cv.picasso"} object.}
  \item{s}{Which lambda to use: \code{"lambda.min"} (default) selects the
    lambda that minimises CV error; \code{"lambda.1se"} uses the largest
    lambda within one standard error of the minimum.}
  \item{sign.lambda}{Multiplier for the x-axis of the CV plot (\code{1} for
    \eqn{\log(\lambda)}, \code{-1} for \eqn{-\log(\lambda)}). Zero lambda
    values are displayed at a finite position below the positive path.}
  \item{newdata}{Matrix of new observations for prediction.}
  \item{type}{Prediction type; passed to the underlying \code{predict}
    method.}
}
\value{
\code{cv.picasso} returns an object of class \code{"cv.picasso"} with components:
\describe{
  \item{lambda}{Ordered lambda sequence evaluated by every fold. A generated
    full-data fit may first stop normally and thereby establish a shorter
    path; every fold must then cover that fixed sequence completely.}
  \item{cvm}{Mean cross-validated error for each lambda.}
  \item{cvsd}{Standard error of the mean cross-validated error.}
  \item{cvup, cvlo}{Upper and lower error-bar bounds.}
  \item{nzero}{Number of nonzero coefficients at each lambda. For
    multinomial models this is the total across all classes.}
  \item{lambda.min}{Lambda that minimises \code{cvm}.}
  \item{lambda.1se}{Largest lambda within one standard error of the minimum.}
  \item{name}{Name of the loss measure used.}
  \item{foldid}{Fold assignment used for each observation.}
  \item{family}{Model family.}
  \item{fast.mode}{Logical precision preset used for all fits.}
  \item{prec}{Effective convergence tolerance used for all fits.}
  \item{picasso.fit}{Full-data \code{picasso} fit.}
}
\code{print.cv.picasso} returns its input invisibly, and
\code{plot.cv.picasso} draws the CV curve and returns \code{NULL} invisibly.
\code{coef.cv.picasso} and \code{predict.cv.picasso} return the corresponding
underlying fit-method result at \code{lambda.min} or \code{lambda.1se}.
}
\details{
Every binomial or multinomial training fold must contain every class observed
in the full data; automatic stratification therefore requires at least two
observations per class. A custom \code{foldid} that violates this rule or a
failed, truncated, or otherwise unusable fold fit produces an explicit error
rather than silently omitting the fold. Normal deviance-tail stopping may
shorten the initial full-data generated path; that retained path is then
passed explicitly to every fold and must be covered in full.
An unusually deep scalar path can make a training fold reach normal
saturation before completing that fixed sequence. This is reported as a
truncated-fold error; request fewer lambdas or a larger
\code{lambda.min.ratio} when it occurs.

Binomial factor responses use the full-data two-level map in every fold and
are scored as zero/one outcomes. When predicting from an offset-fitted
binomial or Poisson \code{cv.picasso} object, pass \code{newoffset} through
\code{\dots} to the underlying prediction method.

For Gaussian models, the full-data \code{type.gaussian = "auto"} decision is
also reused in every fold, so changing training-fold sample sizes cannot switch
the native backend.
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
