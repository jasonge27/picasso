\name{predict.poisson}
\alias{predict.poisson}
\title{
Prediction Method for an Object with S3 Class \code{"poisson"}
}
\description{
Predict responses for new data using fitted models.
}
\usage{
\method{predict}{poisson}(object, newdata, lambda.idx = NULL, p.pred.idx = NULL,
  type = "response", s = NULL, newoffset = NULL, \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"poisson"}. 
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
  Type of prediction. \code{"response"} (default) returns predicted Poisson
  means; \code{"link"} returns the log-mean; \code{"nonzero"} returns
  nonzero variable indices.
  }
  \item{s}{
  Optional nonempty vector of finite non-negative lambda \emph{values}; see
  \code{\link{predict.gaussian}}.
  }
  \item{newoffset}{
  Optional finite numeric vector with one offset per row of \code{newdata}.
  It is required for response or link prediction when \code{object} was
  fitted with \code{offset}; \code{type = "nonzero"} does not use it.
  For a model fitted without offset, the default is a vector of zeros.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
\code{predict.poisson} returns predicted Poisson means for \code{newdata} using fitted coefficients from \code{object}:
\deqn{
    \hat{\mu} = e^{o_{new} + \hat{\beta}_0 + X_{new} \hat{\beta}}.
    }\cr
}
\value{
Return type depends on \code{type}:
\itemize{
  \item \code{"response"} (default): numeric matrix of predicted Poisson means
    \eqn{\hat{\mu} = \exp(o_{new} + \hat{\beta}_0 + X \hat{\beta})}.
  \item \code{"link"}: numeric matrix of log-means
    \eqn{o_{new} + \hat{\beta}_0 + X \hat{\beta}}.
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
