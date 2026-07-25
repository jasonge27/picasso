\name{predict.logit}
\alias{predict.logit}
\title{
Prediction Method for an Object with S3 Class \code{"logit"}
}
\description{
Predict responses for new data using fitted models.
}
\usage{
\method{predict}{logit}(object, newdata, lambda.idx = NULL, p.pred.idx = NULL,
  type = "response", s = NULL, newoffset = NULL, \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"logit"}. 
  }
  \item{newdata}{
  Nonempty finite numeric matrix of new observations for prediction
  (\eqn{n_{new} \times d}) with the same number of columns as the fitted design.
  }
  \item{lambda.idx}{
  Positive integer indices of regularization parameters along the solution
  path used for prediction.
  By default, at most the first three fitted path points are used.
  }
  \item{p.pred.idx}{
  Optional row indices to subset returned predictions. \code{NULL} returns
  every prediction row.
  }
  \item{type}{
  Type of prediction. \code{"response"} (default) returns probabilities;
  \code{"link"} returns the log-odds; \code{"class"} returns 1 when the
  link-scale score is positive and 0 otherwise; \code{"nonzero"} returns
  nonzero variable indices.
  }
  \item{s}{
  Optional nonempty vector of finite non-negative lambda \emph{values}; see
  \code{\link{predict.gaussian}}.
  }
  \item{newoffset}{
  Optional finite numeric vector with one offset per row of \code{newdata}.
  It is required for response, link, or class prediction when \code{object}
  was fitted with \code{offset}; \code{type = "nonzero"} does not use it.
  For a model fitted without offset, the default is a vector of zeros.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
\code{predict.logit} returns predicted Bernoulli probabilities for \code{newdata} using fitted coefficients from \code{object}:
\deqn{
    \hat{p} = \frac{e^{o_{new} + \hat{\beta}_0 + X_{new} \hat{\beta}}}{1+e^{o_{new} + \hat{\beta}_0 + X_{new} \hat{\beta}}}.
    }\cr
}
\value{
Return type depends on \code{type}:
\itemize{
  \item \code{"response"} (default): numeric matrix of predicted probabilities
    \eqn{\hat{p} = \sigma(o_{new} + \hat{\beta}_0 + X \hat{\beta})}.
  \item \code{"link"}: numeric matrix of log-odds
    \eqn{o_{new} + \hat{\beta}_0 + X \hat{\beta}}.
  \item \code{"class"}: integer matrix of predicted class labels (0 or 1).
  \item \code{"nonzero"}: list of integer vectors of nonzero coefficient indices,
    one element per selected lambda.
}
Rows correspond to observations; columns correspond to \code{lambda.idx} (or
\code{s} values when \code{s} is specified).
}
\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}
