\name{scio.plot}
\alias{scio.plot}

\title{
Graph visualization for an object with S3 class \code{"scio"}
}

\description{
Implements the graph visualization using adjacency matrix. It can automatic organize 2D embedding layout.
}

\usage{
scio.plot(G, epsflag = FALSE, graph.name = "default", cur.num = 1, 
          location)
}

\arguments{
  \item{G}{
The adjacency matrix corresponding to the graph.
}
  \item{epsflag}{
If \code{epsflag = TRUE}, save the plot as an eps file in the target directory. The default value is \code{FALSE}.
}
  \item{graph.name}{
The name of the output eps files. The default value is "default".
}
  \item{cur.num}{
The number of plots saved as eps files. Only applicable when \code{epsflag = TRUE}. The default value is 1.
}
  \item{location}{
Target directory. The default value is the current working directory.
}
}

\details{
The user can change \code{cur.num} to plot several figures and select the best one. The implementation is based on the popular package "igraph".
}

\author{
Xingguo Li, Tuo Zhao, Tong Zhang and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}
}

\examples{
## visualize the hub graph
L = scio.generator(graph = "hub")
scio.plot(L$theta)

## visualize the band graph
L = scio.generator(graph = "band",g=5)
scio.plot(L$theta)

## visualize the cluster graph
L = scio.generator(graph = "cluster")
scio.plot(L$theta)

#show working directory
getwd()
#plot 5 graphs and save the plots as eps files in the working directory  
scio.plot(L$theta, epsflag = TRUE, cur.num = 5)
}
