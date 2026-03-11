\name{predict.logit}
\alias{predict.logit}
\title{
Prediction Method for an Object with S3 Class \code{"logit"}
}
\description{
Predict responses for new data using fitted models.
}
\usage{
\method{predict}{logit}(object, newdata, lambda.idx = c(1:3), p.pred.idx = c(1:5), \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"logit"}. 
  }
  \item{newdata}{
  Optional matrix/data frame used for prediction. If omitted, the training design matrix from fitting is used.
  }
  \item{lambda.idx}{
  Indices of regularization parameters along the solution path used for prediction.
  The default is \code{c(1:3)}.
  }
  \item{p.pred.idx}{
  Optional row indices to subset returned predictions when explicitly set to a
  value different from the default \code{c(1:5)}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
\code{predict.logit} returns predicted Bernoulli probabilities for \code{newdata} using fitted coefficients from \code{object}:
\deqn{
    \hat{p} = \frac{e^{\hat{\beta}_0 + X_{new} \hat{\beta}}}{1+e^{\hat{\beta}_0 + X_{new} \hat{\beta}}}.
    }\cr
}
\value{
Numeric matrix of predicted Bernoulli probabilities. Rows correspond to
observations in \code{newdata}; columns correspond to \code{lambda.idx}.
}
\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}
