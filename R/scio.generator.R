#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# scio.generator(): Data generator                                                 #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 10th, 2014                                                             #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

## Main function
scio.generator <- function(n = 200, d = 50, graph = "random", v = NULL, u = NULL, g = NULL, 
                            prob = NULL, seed = NULL, vis = FALSE, verbose = TRUE){	
  gcinfo(FALSE)
  if(verbose) cat("Generating data from the multivariate normal distribution with the", graph,"graph structure...\n")
  
  if(graph!="random" && graph!="hub" && graph!="cluster" && graph!="band" && graph!="scale-free"){
    cat("\"graph\" must be one of \"random\", \"hub\", \"cluster\", \"band\" and \"scale-free\" \n")
    cat("More on help(scio.generator) \n")
    return(NULL)
  }
  if(graph=="hub"||graph=="cluster"){
    if(d<4){
      cat("d is too small, d>=4 required for",graph,"\n")
      cat("More on help(scio.generator) \n")
      return(NULL)
    }
  }
  if(graph=="random"||graph=="band"||graph=="scale-free"){
    if(d<3){
      cat("d is too small, d>=3 required for",graph,"\n")
      cat("More on help(scio.generator) \n")
      return(NULL)
    }
  }
  if(is.null(seed)) seed = 1
  set.seed(seed)
  if(is.null(g)){
    g = 1
    if(graph == "hub" || graph == "cluster"){
      if(d > 40)	g = ceiling(d/20)
      if(d <= 40) g = 2
    }
  }
    
  if(graph == "random"){
    if(is.null(prob))	prob = min(1, 3/d)
    prob = sqrt(prob/2)*(prob<0.5)+(1-sqrt(0.5-0.5*prob))*(prob>=0.5)
  }
  
  if(graph == "cluster"){
    if(is.null(prob)){
      if(d/g > 30)	prob = 0.3
      if(d/g <= 30)	prob = min(1,6*g/d)
    }
    prob = sqrt(prob/2)*(prob<0.5)+(1-sqrt(0.5-0.5*prob))*(prob>=0.5)
  }  
  
  
  # parition variables into groups
  g.large = d%%g
  g.small = g - g.large
  n.small = floor(d/g)
  n.large = n.small+1
  g.list = c(rep(n.small,g.small),rep(n.large,g.large))
  g.ind = rep(c(1:g),g.list)
  rm(g.large,g.small,n.small,n.large,g.list)
  gc()
  
  # build the graph structure
  theta = matrix(0,d,d);
  if(graph == "band"){
    if(is.null(u)) u = 0.1
    if(is.null(v)) v = 0.3
    for(i in 1:g){
      diag(theta[1:(d-i),(1+i):d]) = 1
      diag(theta[(1+i):d,1:(d-1)]) = 1
    }	
  }
  if(graph == "cluster"){
    if(is.null(u)) u = 0.1
    if(is.null(v)) v = 0.3
    for(i in 1:g){
      tmp = which(g.ind==i)
      tmp2 = matrix(runif(length(tmp)^2,0,0.5),length(tmp),length(tmp))
      tmp2 = tmp2 + t(tmp2)		 	
      theta[tmp,tmp][tmp2<prob] = 1
      rm(tmp,tmp2)
      gc()
    }
  }
  if(graph == "hub"){
    if(is.null(u)) u = 0.1
    if(is.null(v)) v = 0.3
    for(i in 1:g){
      tmp = which(g.ind==i)
      theta[tmp[1],tmp] = 1
      theta[tmp,tmp[1]] = 1
      rm(tmp)
      gc()
    }
  }
  if(graph == "random"){
    if(is.null(u)) u = 0.1
    if(is.null(v)) v = 0.3
    
    tmp = matrix(runif(d^2,0,0.5),d,d)
    tmp = tmp + t(tmp)
    theta[tmp < prob] = 1
    #theta[tmp >= tprob] = 0
    rm(tmp)
    gc()
  }
  
  if(graph == "scale-free"){
    if(is.null(u)) u = 0.1
    if(is.null(v)) v = 0.3
    out = .C("SFGen",dd0=as.integer(2),dd=as.integer(d),G=as.integer(theta),seed=as.integer(seed),package="flare")
    theta = matrix(as.numeric(out$G),d,d)
  }
  if(graph=="band"||graph=="cluster"||graph=="hub"||graph=="random"||graph=="scale-free") {
    diag(theta) = 0
    omega = theta*v
    
    # make omega positive definite and standardized
    diag(omega) = abs(min(eigen(omega)$values)) + 0.1 + u
    sigma = cov2cor(solve(omega))
    omega = solve(sigma)
  }
    
  # generate multivariate normal data
  x = mvrnorm(n,rep(0,d),sigma)
  sigmahat = cor(x)
  
  # graph and covariance visulization
  if(vis == TRUE){
    fullfig = par(mfrow = c(2, 2), pty = "s", omi=c(0.3,0.3,0.3,0.3), 
                  mai = c(0.3,0.3,0.3,0.3))
    fullfig[1] = image(theta, col = gray.colors(256),  main = "Adjacency Matrix")
    
    fullfig[2] = image(sigma, col = gray.colors(256), main = "Covariance Matrix")
    g = graph.adjacency(theta, mode="undirected", diag=FALSE)
    layout.grid = layout.fruchterman.reingold(g)
    
    fullfig[3] = plot(g, layout=layout.grid, edge.color='gray50',vertex.color="red", 
                      vertex.size=3, vertex.label=NA,main = "Graph Pattern")
    
    fullfig[4] = image(sigmahat, col = gray.colors(256), main = "Empirical Matrix")
    rm(fullfig,g,layout.grid)
    gc()
  }
  if(verbose) cat("done.\n")
  rm(vis,verbose)
  gc()
  
  sim = list(data = x, sigma = sigma, sigmahat = sigmahat, omega = omega, 
             theta = Matrix(theta,sparse = TRUE), sparsity= sum(theta)/(d*(d-1)), 
             graph.type=graph, prob = prob)
  class(sim) = "sim" 
  return(sim)
}


print.sim = function(x, ...){
  cat("Simulated data generated by scio.generator()\n")
  cat("Sample size: n =", nrow(x$data), "\n")
  cat("Dimension: d =", ncol(x$data), "\n")
  cat("Graph type = ", x$graph.type, "\n")
  cat("Sparsity level:", sum(x$theta)/ncol(x$data)/(ncol(x$data)-1),"\n")
}

plot.sim = function(x, ...){
  gcinfo(FALSE)	
  par = par(mfrow = c(2, 2), pty = "s", omi=c(0.3,0.3,0.3,0.3), 
            mai = c(0.3,0.3,0.3,0.3))
  image(as.matrix(x$theta), col = gray.colors(256),  main = "Adjacency Matrix")
  image(x$sigma, col = gray.colors(256), main = "Covariance Matrix")
  g = graph.adjacency(x$theta, mode="undirected", diag=FALSE)
  layout.grid = layout.fruchterman.reingold(g)
  
  plot(g, layout=layout.grid, edge.color='gray50',vertex.color="red", vertex.size=3, 
       vertex.label=NA,main = "Graph Pattern")
  rm(g, layout.grid)
  gc()
  image(x$sigmahat, col = gray.colors(256), main = "Empirical Covariance Matrix")
}
