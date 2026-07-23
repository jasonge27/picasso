\name{multinomial-methods}
\alias{print.multinomial}
\alias{plot.multinomial}
\alias{coef.multinomial}
\alias{predict.multinomial}
\title{Methods for Multinomial picasso Fits}
\description{
Fit multinomial models with \code{\link{picasso}} using
\code{family = "multinomial"}, then print, plot, extract coefficients, or
predict from the fitted class-coupled regularization path.
}
\usage{
\method{print}{multinomial}(x, \dots)

\method{plot}{multinomial}(x, which.class = 1, \dots)

\method{coef}{multinomial}(object, lambda.idx = NULL, beta.idx = NULL, \dots)

\method{predict}{multinomial}(object, newdata, lambda.idx = NULL,
  type = "response", s = NULL, \dots)
}
\arguments{
  \item{x}{A fitted object with S3 class \code{"multinomial"}.}
  \item{object}{A fitted object with S3 class \code{"multinomial"}.}
  \item{which.class}{One-based class index whose coefficient path is plotted.
    The default is \code{1}.}
  \item{lambda.idx}{One-based lambda indices for extraction or prediction.
    If \code{NULL}, the first three available lambdas (or the entire path when
    shorter) are used. Supply only one of \code{lambda.idx} and \code{s}.}
  \item{beta.idx}{One-based feature indices to extract. If \code{NULL}, the
    first three available features are used.}
  \item{newdata}{Finite numeric matrix of new observations with the same
    number of columns as the fitted design.}
  \item{type}{Prediction type. \code{"response"} returns softmax
    probabilities, \code{"link"} returns class-specific linear predictors,
    \code{"class"} returns the fitted label with the largest finite link-scale
    score (first fitted class on an exact tie), and
    \code{"nonzero"} returns nonzero feature indices separately by class.}
  \item{s}{Optional finite non-negative lambda \emph{values}. Coefficients are
    linearly interpolated between fitted lambdas for response, link, and class
    predictions; values outside the path are clamped to an endpoint.
    \code{type = "nonzero"} uses the nearest fitted lambda instead.}
  \item{\dots}{Additional arguments (currently unused).}
}
\details{
The public fitting call is
\code{picasso(X, Y, family = "multinomial", ...)}. Responses must contain at
least three observed classes; numeric, character, and factor labels are
accepted, and unused factor levels are dropped.

The solver first identifies a strong working set and then solves the
class-coupled restricted problem with Proximal Newton/IRLS iterations until
the weighted-L1 subproblem converges. Full KKT checks update the active set and
certify every retained subproblem. Pathwise warm starts, sequential strong
screening, adaptive inexact Newton tolerances, vectorized coordinate kernels,
and accepted-probability reuse reduce repeated work.

MCP and SCAD use adaptive local linear approximation. Every fit performs the
minimum three total stages (one L1 master and two weighted-L1 updates). Raising
\code{lla.max.stages} permits further stages, which stop once target
stationarity is at most \code{prec}; exhausting the budget yields the usable
status \code{lla_stationarity_limit}.

After at least five lambdas, an automatically generated path stops normally
when explained deviance exceeds \code{0.999} or its improvement is below
\code{1e-5}. Explicit lambda sequences disable only this saturation rule;
\code{dfmax} or a hard failure can still truncate them. A hard failure after
one or more completed lambdas returns the committed prefix with a warning and
failure metadata; failure before the first completed lambda is an error.

With no intercept, standardized predictors are scaled about the origin rather
than centered. The zero model then uses uniform class probabilities for the
automatic lambda path and null deviance.
}
\value{
\code{print.multinomial} returns \code{x} invisibly.
\code{plot.multinomial} draws one class's coefficient paths and returns
\code{NULL} invisibly.

\code{coef.multinomial} returns a list of \eqn{K} numeric matrices, one per
class. Rows contain the intercept followed by selected coefficients, and
columns correspond to selected lambdas.

For one selected lambda, \code{predict.multinomial} returns an
\eqn{n_{\mathrm{new}} \times K} matrix for \code{"response"} or
\code{"link"}, a factor of length \eqn{n_{\mathrm{new}}} for \code{"class"},
or a list of \eqn{K} integer vectors for \code{"nonzero"}. Multiple
\code{lambda.idx} or \code{s} values return a list of those objects.

The fitted object returned by \code{picasso} contains class-specific
\code{beta} and \code{intercept} lists, \code{lambda}, \code{df},
\code{levels}, \code{K}, \code{dev.ratio}, status/failure metadata,
per-lambda diagnostics, \code{path.early.stopped}, and
\code{requested.nlambda}; see \code{\link{picasso}}.
}
\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}
\seealso{
\code{\link{picasso}}, \code{\link{cv.picasso}}, and
\code{\link{assess.picasso}}.
}
\examples{
set.seed(1)
n <- 120; d <- 12
X <- matrix(rnorm(n * d), n, d)
score <- cbind(X[, 1], -X[, 1] + X[, 2], -X[, 2])
Y <- factor(c("red", "green", "blue")[max.col(score)])
fit <- picasso(X, Y, family = "multinomial", nlambda = 12)
prob <- predict(fit, X[1:5, ], lambda.idx = fit$nlambda,
                type = "response")
pred <- predict(fit, X[1:5, ], lambda.idx = fit$nlambda,
                type = "class")
}
