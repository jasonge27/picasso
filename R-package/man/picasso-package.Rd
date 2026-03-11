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
Computation is based on multi-stage convex relaxation and pathwise coordinate
optimization with warm starts, active-set updates, and screening rules.
}
\details{
Core optimization routines are implemented in C++ for speed, and coefficient
paths are stored as sparse matrices for memory efficiency.
}

\author{
Jason Ge, Xingguo Li, Haoming Jiang, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}

\seealso{
\code{\link{picasso}}.
}
