\name{eyedata}
\alias{eyedata}

\title{
The Bardet-Biedl syndrome Gene expression data from Scheetz et al. (2006)
}

\description{
Gene expression data (20 genes for 120 samples) from the microarray experiments of mammalianeye tissue samples of Scheetz et al. (2006).
}

\usage{data(eyedata)}

\details{
	This data set contains 120 samples with 200 predictors
}

\format{
  The format is a list containing conatins a matrix and a vector.
  1. x - an 120 by 200 matrix, which represents the data of 120 rats with 200 gene probes.
  2. y - a 120-dimensional vector of, which represents the expression level of TRIM32 gene.
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
image(x)
}
