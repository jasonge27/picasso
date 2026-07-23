\name{picasso-package}
\alias{picasso-package}
\docType{package}
\title{
picasso: Sparse Learning with Convex and Non-Convex Penalties
}
\description{
Computationally efficient tools for fitting sparse generalized linear models with
convex or non-convex penalties. Supported penalties include lasso, smoothly
clipped absolute deviation (SCAD), and minimax concave penalty (MCP).
Supported families are Gaussian, binomial, Poisson, square-root-lasso, and
multinomial. Computation combines pathwise coordinate optimization with warm
starts, active-set updates, screening rules, Proximal Newton iterations, and
adaptive local linear approximation for applicable nonconvex models.
}
\details{
Core optimization routines are implemented in C++. Gaussian models use
active-set coordinate descent with an automatic residual/covariance backend;
binomial and Poisson models use active-set Proximal Newton/IRLS,
square-root-lasso uses a global quadratic majorizer with active-set coordinate
updates, and multinomial uses class-coupled Proximal Newton/IRLS. The R
interface returns coefficient paths as \pkg{Matrix} objects and provides
prediction, assessment, confusion-matrix, and cross-validation helpers.
}

\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\seealso{
\code{\link{picasso}}.
}
