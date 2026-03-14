\name{print.gaussian}
\alias{print.gaussian}
\title{
Print Method for an Object with S3 Class \code{"gaussian"}
}
\description{
Print a concise summary for an object with S3 class \code{"gaussian"}.
}
\usage{
\method{print}{gaussian}(x, ...)
}
\arguments{
  \item{x}{
  An object with S3 class \code{"gaussian"}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
This method prints tuning parameters and fit statistics for a fitted model object.
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
