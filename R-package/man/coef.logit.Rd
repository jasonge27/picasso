\name{coef.logit}
\alias{coef.logit}
\title{
Extract model coefficients for an object with S3 class \code{"logit"}
}
\description{
Extract estimated regression coefficient vectors from the solution path.
}
\usage{
\method{coef}{logit}(object, lambda.idx = c(1:3), beta.idx = c(1:3), \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"logit"} 
  }
  \item{lambda.idx}{
  The indices of the regularization parameters in the solution path to be displayed. The default values are \code{c(1:3)}.
  }
  \item{beta.idx}{
  The indices of the estimated regression coefficient vectors in the solution path to be displayed. The default values are \code{c(1:3)}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}

\author{
Jason Ge, Xingguo Li, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}
