\name{predict.poisson}
\alias{predict.poisson}
\title{
Prediction Method for an Object with S3 Class \code{"poisson"}
}
\description{
Predict responses for new data using fitted models.
}
\usage{
\method{predict}{poisson}(object, newdata, lambda.idx = c(1:3), p.pred.idx = c(1:5),
  type = "response", s = NULL, \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"poisson"}. 
  }
  \item{newdata}{
  Numeric matrix of new observations for prediction (\eqn{n_{new} \times d}).
  }
  \item{lambda.idx}{
  Indices of regularization parameters along the solution path used for prediction.
  The default is \code{c(1:3)}.
  }
  \item{p.pred.idx}{
  Optional row indices to subset returned predictions when explicitly set to a
  value different from the default \code{c(1:5)}.
  }
  \item{type}{
  Type of prediction. \code{"response"} (default) returns predicted Poisson
  means; \code{"link"} returns the log-mean; \code{"nonzero"} returns
  nonzero variable indices.
  }
  \item{s}{
  Optional numeric vector of lambda \emph{values}; see \code{\link{predict.gaussian}}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
\code{predict.poisson} returns predicted Poisson means for \code{newdata} using fitted coefficients from \code{object}:
\deqn{
    \hat{\mu} = e^{\hat{\beta}_0 + X_{new} \hat{\beta}}.
    }\cr
}
\value{
Return type depends on \code{type}:
\itemize{
  \item \code{"response"} (default): numeric matrix of predicted Poisson means
    \eqn{\hat{\mu} = \exp(\hat{\beta}_0 + X \hat{\beta})}.
  \item \code{"link"}: numeric matrix of log-means
    \eqn{\hat{\beta}_0 + X \hat{\beta}}.
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
