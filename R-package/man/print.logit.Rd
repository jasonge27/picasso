\name{print.logit}
\alias{print.logit}
\title{
Print Method for an Object with S3 Class \code{"logit"}
}
\description{
Print a concise summary for an object with S3 class \code{"logit"}.
}
\usage{
\method{print}{logit}(x, ...)
}
\arguments{
  \item{x}{
  An object with S3 class \code{"logit"}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
This method prints tuning parameters and fit statistics for a fitted logistic model object.
}
\value{
The input object \code{x}, returned invisibly.
}
\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}
\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}
