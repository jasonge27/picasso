\name{picasso.multinomial}
\alias{picasso.multinomial}
\alias{print.multinomial}
\alias{plot.multinomial}
\alias{coef.multinomial}
\alias{predict.multinomial}
\title{Sparse Multinomial Logistic Regression}
\description{
Fit a sparse multinomial (softmax) logistic regression model along a
regularization path using L1, MCP, or SCAD penalties.
}
\usage{
\method{print}{multinomial}(x, \dots)

\method{plot}{multinomial}(x, which.class = 1, \dots)

\method{coef}{multinomial}(object, lambda.idx = 1:3, beta.idx = 1:3, \dots)

\method{predict}{multinomial}(object, newdata, lambda.idx = 1:3,
  type = "response", s = NULL, \dots)
}
\arguments{
  \item{x}{An object with S3 class \code{"multinomial"}.}
  \item{object}{An object with S3 class \code{"multinomial"}.}
  \item{which.class}{Integer index of the class whose coefficient path is
    plotted (default \code{1}).}
  \item{lambda.idx}{Integer vector of lambda indices for extraction or
    prediction.}
  \item{beta.idx}{Integer vector of coefficient (feature) indices to extract.}
  \item{newdata}{Numeric matrix of new observations, \eqn{n \times d}.}
  \item{type}{Prediction type: \code{"response"} (default) returns softmax
    probabilities; \code{"class"} returns the predicted class label;
    \code{"link"} returns raw linear predictors; \code{"nonzero"} returns
    a list of nonzero variable indices.}
  \item{s}{Optional numeric vector of lambda \emph{values} (not indices).
    When supplied, \code{lambda.idx} is ignored.  Values between path lambdas
    are linearly interpolated with a message; values outside the path are
    clamped to the nearest endpoint.}
  \item{\dots}{Additional arguments (currently unused).}
}
\details{
\code{picasso} with \code{family = "multinomial"} calls
\code{picasso.multinomial} internally.  The model solves
\deqn{
  \min_{\{B_k, b_k\}} -\frac{1}{n}\sum_{i=1}^n \log P(Y_i | x_i)
  + \sum_{k=1}^K \sum_{j=1}^d p_{\lambda,\gamma}(|\beta_{kj}|),
}
where \eqn{P(Y_i = k | x_i) = \text{softmax}_k(b_k + x_i^\top B_k)}.
}
\value{
An object of class \code{"multinomial"} with components:
\describe{
  \item{beta}{List of K sparse matrices (each \eqn{d \times \mathrm{nlambda}}),
    one per class.}
  \item{intercept}{List of K numeric vectors (length nlambda).}
  \item{lambda}{Lambda sequence.}
  \item{nlambda}{Number of lambda values.}
  \item{df}{Total nonzero coefficients across all classes at each lambda.}
  \item{K}{Number of classes.}
  \item{levels}{Class labels from \code{as.factor(Y)}.}
  \item{method}{Penalty type used for fitting.}
  \item{alg}{Algorithm identifier (\code{"multinomial-cd"}).}
  \item{ite}{Iteration counts per lambda.}
  \item{verbose}{Input \code{verbose} value.}
  \item{runtime}{Elapsed fitting time.}
}
}
\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}
\seealso{\code{\link{picasso}}}
\examples{
set.seed(1)
n <- 150; d <- 20
X <- matrix(rnorm(n * d), n, d)
Y <- sample(1:3, n, replace = TRUE)
fit <- picasso(X, Y, family = "multinomial", nlambda = 20)
pred_class <- predict(fit, X[1:10, ], lambda.idx = 1, type = "class")
}
