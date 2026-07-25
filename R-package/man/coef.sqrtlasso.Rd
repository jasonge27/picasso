\name{coef.sqrtlasso}
\alias{coef.sqrtlasso}
\title{
Extract Coefficients from an Object with S3 Class \code{"sqrtlasso"}
}
\description{
Extract estimated coefficient vectors from selected points on the regularization path.
}
\usage{
\method{coef}{sqrtlasso}(object, lambda.idx = NULL, beta.idx = NULL, \dots)
}
\arguments{
  \item{object}{
  An object with S3 class \code{"sqrtlasso"}. 
  }
  \item{lambda.idx}{
  Indices of regularization parameters in the solution path to extract.
  By default, at most the first three fitted path points are used.
  }
  \item{beta.idx}{
  Indices of coefficient entries to extract. By default, at most the first
  three coefficients are used.
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
