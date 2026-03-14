\name{eyedata}
\alias{eyedata}

\title{
Bardet-Biedl Syndrome Gene Expression Data from Scheetz et al. (2006)
}

\description{
Gene expression data (200 genes, 120 samples) from microarray experiments on mammalian eye tissue reported by Scheetz et al. (2006).
}

\usage{data(eyedata)}

\details{
This data set contains 120 samples and 200 predictors.
}

\format{
  A list with components:
  \describe{
    \item{x}{A 120 by 200 matrix of gene probes.}
    \item{y}{A numeric response vector of length 120 representing the expression level of the TRIM32 gene.}
  }
}

\author{
Xingguo Li, Tuo Zhao, Tong Zhang and Han Liu \cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\references{
1. T. Scheetz, K. Kim, R. Swiderski, A. Philp, T. Braun, K. Knudtson, A. Dorrance, G. DiBona, J. Huang, T. Casavant, V. Sheffield, and E. Stone. Regulation of gene expression in the mammalian eye and its relevance to eye disease. \emph{Proceedings of the National Academy of Sciences of the United States of America}, 2006.
}

\seealso{
\code{\link{picasso-package}}.
}

\examples{
data(eyedata)
image(eyedata$x)
}
