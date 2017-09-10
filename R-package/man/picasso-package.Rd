\name{picasso-package}
\alias{picasso-package}
\docType{package}
\title{
PICASSO: PathwIse CAlibrated Sparse Shooting algOrithm
}
\description{
This package provides computationally efficient tools for fitting generalized linear model with convex and non-convex penalty. Users can enjoy the superior statistical property of non-convex penalty such as SCAD and MCP which has significantly less estimation error and overfitting compared to convex penalty such as l1 and ridge. Computation is handled by multi-stage convex relaxation and the PathwIse CAlibrated Sparse Shooting algOrithm (PICASSO) which exploits warm start initialization, active set updating, and strong rule for coordinate preselection to boost computation, and attains a linear convergence to a unique sparse local optimum with optimal statistical properties. The computation is memory-optimized using the sparse matrix output.
}
\details{
\tabular{ll}{
Package: \tab picasso\cr
Type: \tab Package\cr
Version: \tab 0.5.4\cr
Date: \tab 2016-09-20\cr
License: \tab GPL-2\cr
}
}

\author{
Jason Ge, Xingguo Li, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Jason Ge <jiange@princeton.edu>
}

\seealso{
\code{\link{picasso}}.
}
