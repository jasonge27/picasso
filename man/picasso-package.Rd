\name{picasso-package}
\alias{picasso-package}
\docType{package}
\title{
picasso: Pathwise Calibrated Sparse Shooting Algorithm
}
\description{
This package implements the pathwise calibrated sparse shooting algorithm (PICASSO) for high dimensional regularized sparse linear regression, sparse logistic regression, and sparse undirected graphical model estimation. As a pathwise coordinate optimization algorithm, PICASSO exploits warm start initialization, active set updating, and strong rule for coordinate preselection to boost computation, and attains a linear convergence to a unique sparse local optimum with optimal statistical properties. Moreover, users can choose different regularizers, including L1, MCP, and SCAD regularizers. The computation is memory-optimized using the sparse matrix output.
}
\details{
\tabular{ll}{
Package: \tab picasso\cr
Type: \tab Package\cr
Version: \tab 0.5.0\cr
Date: \tab 2015-12-15\cr
License: \tab GPL-2\cr
}
}

\author{
Xingguo Li, Tuo Zhao, Tong Zhang and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\seealso{
\code{\link{picasso}} and \code{\link{scio.generator}}.
}
