\name{coef.logit}
\alias{coef.logit}
\title{
Extract Coefficients from an Object with S3 Class \code{"logit"}
}
\description{
Extract estimated coefficient vectors from selected points on the regularization path.
}
\usage{
\method{coef}{logit}(object, lambda.idx = c(1:3), beta.idx = c(1:3), \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"logit"}. 
  }
  \item{lambda.idx}{
  Indices of regularization parameters in the solution path to extract.
  The default is \code{c(1:3)}.
  }
  \item{beta.idx}{
  Indices of coefficient entries to extract. The default is \code{c(1:3)}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\value{
A numeric matrix of extracted coefficients.
Columns correspond to \code{lambda.idx}. Rows contain the intercept
(\code{"(Intercept)"}) and the selected coefficients
(\code{"beta[i]"} for \code{i} in \code{beta.idx}).
}

\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}
