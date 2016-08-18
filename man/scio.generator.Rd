\name{scio.generator}
\alias{scio.generator}

\title{
Data generator for sparse undirected graph estimation.
}

\description{
Implements the data generation from multivariate normal distributions with different graph structures, including \code{"random"}, \code{"hub"}, \code{"cluster"}, \code{"band"}, and \code{"scale-free"}.
}

\usage{
scio.generator(n = 200, d = 50, graph = "random", v = NULL, u = NULL,
      g = NULL, prob = NULL, seed = NULL, vis = FALSE, verbose = TRUE)
}

\arguments{
  \item{n}{
The number of observations (sample size). The default value is \eqn{200}.
}
  \item{d}{
The number of variables (dimension). For \code{"hub"} and \code{"cluster"}, \code{d}\eqn{ \ge }4 is required. For \code{"random"}, \code{"band"} and \code{"scale-free"}, \code{d}\eqn{ \ge }3 is required. The default value is \eqn{50}.
}
  \item{graph}{
The graph structure with 5 options: \code{"random"}, \code{"hub"}, \code{"cluster"}, \code{"band"}, and \code{"scale-free"}.
}
  \item{v}{
The off-diagonal elements of the precision matrix, controlling the magnitude of partial correlations with \code{u}. The default value is 0.3.
}
  \item{u}{
A positive number being added to the diagonal elements of the precision matrix, to control the magnitude of partial correlations. The default value is 0.1.
}
  \item{g}{
For \code{"cluster"} or \code{"hub"} graph, \code{g} is the number of hubs or clusters in the graph. The default value is about \code{d}/20 if \code{d}\eqn{ \ge} 40 and 2 if \code{d}\eqn{ <} 40. For \code{"band"} graph, \code{g} is the bandwidth and the default value is \eqn{1}. NOT applicable to \code{"random"} graph.
}
  \item{prob}{
For \code{"random"} graph, it is the probability that a pair of nodes has an edge. The default value is 3/\code{d}. For \code{"cluster"} graph, it is the probability that a pair of nodes has an edge in each cluster. The default value is 6\eqn{*}\code{g/d} if \code{d/g}\eqn{ \le} 30 and 0.3 if \code{d/g}\eqn{ >} 30. NOT applicable to \code{"hub"}, \code{"band"}, and \code{"scale-free"} graphs.
}
  \item{seed}{
Set seed for data generation. The default value is 1.
}
  \item{vis}{
Visualize the adjacency matrix of the true graph structure, the graph pattern, the covariance matrix and the empirical covariance matrix. The default value is \code{FALSE}.
}
  \item{verbose}{
If \code{verbose = FALSE}, tracing information printing is disabled. The default value is \code{TRUE}.
}
}

\details{
Given the adjacency matrix \code{theta}, the graph patterns are generated as below:\cr\cr
(I) \code{"random"}: Each pair of off-diagonal elements are randomly set \code{theta[i,j]=theta[j,i]=1} for \code{i!=j} with probability \code{prob}, and \code{0} other wise. It results in about \code{d*(d-1)*prob/2} edges in the graph.\cr\cr
(II)\code{"hub"}:The row/columns are evenly partitioned into \code{g} disjoint groups. Each group is associated with a "center" row \code{i} in that group. Each pair of off-diagonal elements are set \code{theta[i,j]=theta[j,i]=1} for \code{i!=j} if \code{j} also belongs to the same group as \code{i} and \code{0} otherwise. It results in \code{d - g} edges in the graph.\cr\cr
(III)\code{"cluster"}:The row/columns are evenly partitioned into \code{g} disjoint groups. Each pair of off-diagonal elements are set \code{theta[i,j]=theta[j,i]=1} for \code{i!=j} with the probability \code{prob}if both \code{i} and \code{j} belong to the same group, and \code{0} other wise. It results in about \code{g*(d/g)*(d/g-1)*prob/2} edges in the graph.\cr\cr
(IV)\code{"band"}: The off-diagonal elements are set to be \code{theta[i,j]=1} if \code{1<=|i-j|<=g} and \code{0} other wise. It results in \code{(2d-1-g)*g/2} edges in the graph.\cr\cr
(V) \code{"scale-free"}: The graph is generated using B-A algorithm. The initial graph has two connected nodes and each new node is connected to only one node in the existing graph with the probability proportional to the degree of the each node in the existing graph. It results in \code{d} edges in the graph.\cr\cr

The adjacency matrix \code{theta} has all diagonal elements equal to \code{0}. To obtain a positive definite covariance matrix, the smallest eigenvalue of \code{theta*v} (denoted by \code{e}) is computed. Then we set the covariance matrix equal to \code{cov2cor(solve(theta*v+(|e|+0.1+u)*I))} to generate multivariate normal data.
}

\value{
An object with S3 class "sim" is returned:
\item{data}{
The \code{n} by \code{d} matrix for the generated data
}
\item{sigma}{
The covariance matrix for the generated data
}
  \item{omega}{
The precision matrix for the generated data
}
  \item{sigmahat}{
The empirical covariance matrix for the generated data
}
  \item{theta}{
The adjacency matrix of true graph structure (in sparse matrix representation) for the generated data
}
}

\author{
Xingguo Li, Tuo Zhao, Tong Zhang and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

%\references{
%}
\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}
}
\examples{
## band graph with bandwidth 3
L = scio.generator(graph = "band", g = 3)
plot(L)

## random sparse graph
L = scio.generator(vis = TRUE)

## random dense graph
L = scio.generator(prob = 0.5, vis = TRUE)

## hub graph with 6 hubs
L = scio.generator(graph = "hub", g = 6, vis = TRUE)

## cluster graph with 8 clusters
L = scio.generator(graph = "cluster", g = 8, vis = TRUE)

## scale-free graphs
L = scio.generator(graph="scale-free", vis = TRUE)
}
