\name{predict.gaussian}
\alias{predict.gaussian}
\title{
Prediction Method for an Object with S3 Class \code{"gaussian"}
}
\description{
Predict responses for new data using fitted models.
}
\usage{
\method{predict}{gaussian}(object, newdata, lambda.idx = c(1:3), Y.pred.idx = c(1:5),
  type = "response", s = NULL, \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"gaussian"}. 
  }
  \item{newdata}{
  Numeric matrix of new observations for prediction (\eqn{n_{new} \times d}).
  }
  \item{lambda.idx}{
  Indices of regularization parameters along the solution path used for prediction.
  The default is \code{c(1:3)}.
  }
  \item{Y.pred.idx}{
  Optional row indices to subset returned predictions when explicitly set to a
  value different from the default \code{c(1:5)}.
  }
  \item{type}{
  Type of prediction. \code{"response"} (default) returns predicted values;
  \code{"link"} returns the linear predictor; \code{"nonzero"} returns a list
  of nonzero variable indices per selected lambda.
  }
  \item{s}{
  Optional numeric vector of lambda \emph{values} (not indices).  When
  supplied, \code{lambda.idx} is ignored.  If a requested value falls between
  two path lambdas the coefficients are linearly interpolated and a message is
  issued.  Values outside the path range are clamped to the nearest endpoint.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
\code{predict.gaussian} returns predicted responses for \code{newdata} using fitted coefficients from \code{object}:
\deqn{
    \hat{Y} = \hat{\beta}_0 + X_{new} \hat{\beta}.
    }\cr
}
\value{
Return type depends on \code{type}:
\itemize{
  \item \code{"response"} or \code{"link"} (default): numeric matrix of
    predicted values \eqn{\hat{\beta}_0 + X \hat{\beta}}.
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
