\name{eyedata}
\alias{eyedata}

\title{
The Bardet-Biedl syndrome Gene expression data from Scheetz et al. (2006)
}

\description{
Gene expression data (200 genes for 120 samples) from the microarray experiments of mammalian eye tissue samples of Scheetz et al. (2006).
}

\usage{data(eyedata)}

\details{
	This data set contains 120 samples with 200 predictors
}

\format{
  A list with components:
  \describe{
    \item{x}{A 120 by 200 matrix of gene probes.}
    \item{y}{A numeric response vector of length 120 representing the expression level of TRIM32 gene.}
  }
}

\author{
Xingguo Li, Tuo Zhao, Tong Zhang and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\references{
1. T. Scheetz, k. Kim, R. Swiderski, A. Philp, T. Braun, K. Knudtson, A. Dorrance, G. DiBona, J. Huang, T. Casavant, V. Sheffield, E. Stone .Regulation of gene expression in the mammalian eye and its relevance to eye disease. \emph{Proceedings of the National Academy of Sciences of the United States of America}, 2006.
}

\seealso{
\code{\link{picasso-package}}.
}

\examples{
data(eyedata)
image(eyedata$x)
}
