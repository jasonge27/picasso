#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# picasso.scio(): The user interface for scio()                                    #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 2nd, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

picasso.scio <- function(X,
                         lambda = NULL,
                         nlambda = NULL,
                         lambda.min.ratio = NULL,
                         lambda.min = NULL,
                         method = "l1",
                         alg = "cyclic",
                         opt = "naive",
                         gamma = 3,
                         sym = "or",
                         truncation = 0, 
                         max.act.in = 3, 
                         prec = 1e-4,
                         max.ite = 1e3,
                         standardize = FALSE,
                         perturb = TRUE,
                         verbose = TRUE)
{
  begt=Sys.time()
  n = nrow(X)
  d = ncol(X)
  if(verbose)
    cat("Sparse column inverse operator \n")
  if(n==0 || d==0) {
    cat("No data input.\n")
    return(NULL)
  }
  if(method!="l1" && method!="mcp" && method!="scad"){
    cat(" Wrong \"method\" input. \n \"method\" should be one of \"l1\", \"mcp\" and \"scad\".\n", 
        method,"does not exist. \n")
    return(NULL)
  }
  if(alg!="cyclic" && alg!="greedy" && alg!="proximal" && alg!="random" && alg!="hybrid"){
    cat(" Wrong \"alg\" input. \n \"alg\" should be one of \"cyclic\", \"greedy\", \"proximal\", \"random\" and \"hybrid\".\n", 
        alg,"does not exist. \n")
    return(NULL)
  }
  maxdf = max(n,d)
  est = list()
  est$cov.input = isSymmetric(X)
  correlation = FALSE
  if(est$cov.input)
  {
    if(verbose) {
      cat("The input is identified as the covriance matrix.\n")
    }
    if(method=="slasso") {
      cat("The input for \"slasso\" cannot be covriance matrix.\n")
      return(NULL)
    }
    if(correlation)
      S = cov2cor(X)
    else
      S = X
  }
  if(!est$cov.input)
  {
    if(standardize)
      X = scale(X)
    
    if(correlation)
      S = cor(X)
    else
      S = cov(X)
  }
  
  if(!is.null(lambda)) nlambda = length(lambda)
  if(is.null(lambda))
  {
    if(is.null(nlambda))
      nlambda = 100
    lambda.max = max(max(S-diag(d)),-min(S-diag(d)))
    if(is.null(lambda.min)){
      if(is.null(lambda.min.ratio)){
        lambda.min = 0.05*lambda.max
      }else{
        lambda.min = min(lambda.min.ratio*lambda.max, lambda.max)
      }
    }
    if(lambda.min>=lambda.max) cat("\"lambda.min\" is too small. \n")
    lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
    rm(lambda.max,lambda.min,lambda.min.ratio)
    gc()
  }
  
  est$lambda = lambda
  est$nlambda = nlambda
  
  if (is.logical(perturb)) {
    if (perturb) { 
      perturb = 1/sqrt(n)
    } else {
      perturb = 0
    }
  }
  S = S + diag(d)*perturb
  
  if(method=="l1") {
    method.flag = 1
  }
  if(method=="mcp") {
    method.flag = 2
    if (gamma<=1) {
      cat("gamma > 1 is required for MCP. Set to the default value 3. \n")
      gamma = 3
    }
  }
  if(method=="scad") {
    method.flag = 3
    if (gamma<=2) {
      cat("gamma > 2 is required for SCAD. Set to the default value 3. \n")
      gamma = 3
    }
  }
  
  out = scio.sc(S, lambda, nlambda, gamma, d, maxdf, prec, max.ite, verbose, alg, method.flag, max.act.in, truncation)
  
  est$ite = out$ite
  
  for(j in 1:d) {
    if(out$col_cnz[j+1]>out$col_cnz[j])
    {
      idx.tmp = (out$col_cnz[j]+1):out$col_cnz[j+1]
      ord = order(out$row_idx[idx.tmp])
      out$row_idx[idx.tmp] = out$row_idx[ord + out$col_cnz[j]]
      out$x[idx.tmp] = out$x[ord + out$col_cnz[j]]
    }
  }
  G = new("dgCMatrix", Dim = as.integer(c(d*nlambda,d)), x = as.vector(out$x[1:out$col_cnz[d+1]]),
          p = as.integer(out$col_cnz), i = as.integer(out$row_idx[1:out$col_cnz[d+1]]))
  
  est$beta = list()
  est$path = list()
  est$df = matrix(0,d,nlambda)
  est$sparsity = rep(0,nlambda) 
  for(i in 1:nlambda) {
    est$beta[[i]] = G[((i-1)*d+1):(i*d),]
    est$path[[i]] = abs(est$beta[[i]])
    est$df[,i] = apply(sign(est$path[[i]]),2,sum)
    
    if(sym == "or")
      est$path[[i]] = sign(est$path[[i]] + t(est$path[[i]]))
    if(sym == "and")
      est$path[[i]] = sign(est$path[[i]] * t(est$path[[i]]))
    est$sparsity[i] = sum(est$path[[i]])/d/(d-1)
  }
  rm(G)
  runt=Sys.time()-begt
  est$runtime = runt
  est$obj = out$obj
  est$runt = out$runt
  est$sigma = S
  est$method = method
  est$alg = alg
  est$gamma = gamma
  est$sym = sym
  est$verbose = verbose
  est$standardize = standardize
  est$correlation = correlation
  est$perturb = perturb
  class(est) = "scio"
  gc()
  return(est)
}

print.scio <- function(x, ...)
{  
  cat("\n SCIO options summary: \n")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda,digits=3))
  cat("Method=", x$method, "\n")
  cat("Path length:",x$nlambda,"\n")
  cat("Graph dimension:",ncol(x$X),"\n")
  cat("Sparsity level:",min(x$sparsity),"----->",max(x$sparsity),"\n")
}

plot.scio = function(x, align = FALSE, ...){
  gcinfo(FALSE)
  
  if(x$nlambda == 1)  par(mfrow = c(1, 2), pty = "s", omi=c(0.3,0.3,0.3,0.3), mai = c(0.3,0.3,0.3,0.3))
  if(x$nlambda == 2)	par(mfrow = c(1, 3), pty = "s", omi=c(0.3,0.3,0.3,0.3), mai = c(0.3,0.3,0.3,0.3))
  if(x$nlambda >= 3)	par(mfrow = c(1, 4), pty = "s", omi=c(0.3,0.3,0.3,0.3), mai = c(0.3,0.3,0.3,0.3))
  
  if(x$nlambda <= 3)	z.final = 1:x$nlambda
  
  if(x$nlambda >=4){
    z.max = max(x$sparsity)
    z.min = min(x$sparsity)
    z = z.max - z.min
    z.unique = unique(c(which(x$sparsity>=(z.min + 0.03*z))[1],which(x$sparsity>=(z.min + 0.07*z))[1],which(x$sparsity>=(z.min + 0.15*z))[1]))
    
    
    if(length(z.unique) == 1){
      if(z.unique<(x$nlambda-1))	z.final = c(z.unique,z.unique+1,z.unique+2)
      if(z.unique==(x$nlambda-1)) z.final = c(z.unique-1,z.unique,z.unique+1)
      if(z.unique==x$nlambda) 	z.final = c(z.unique-2,z.unique-1,z.unique)
    }
    
    if(length(z.unique) == 2){
      if(diff(z.unique)==1){
        if(z.unique[2]<x$nlambda) z.final = c(z.unique,z.unique[2]+1) 
        if(z.unique[2]==x$nlambda) z.final = c(z.unique[1]-1,z.unique)
      }
      if(diff(z.unique)>1) z.final = c(z.unique[1],z.unique[1]+1,z.unique[2])
    }
    
    if(length(z.unique) == 3) z.final = z.unique
    
    rm(z.max,z.min,z,z.unique)
    gc()
    
  }
  plot(x$lambda, x$sparsity, log = "x", xlab = "Regularization Parameter", ylab = "Sparsity Level", type = "l",xlim = rev(range(x$lambda)), main = "Sparsity vs. Regularization")
  
  lines(x$lambda[z.final],x$sparsity[z.final],type = "p")
  
  if(align){
    layout.grid = layout.fruchterman.reingold(graph.adjacency(as.matrix(x$path[[z.final[length(z.final)]]]), mode="undirected", diag=FALSE))
    for(i in z.final){
      g = graph.adjacency(as.matrix(x$path[[i]]), mode="undirected", diag=FALSE)
      plot(g, layout=layout.grid, edge.color='gray50',vertex.color="red", vertex.size=3, vertex.label=NA, main = paste("lambda = ",as.character(round(x$lambda[i],3)),sep = ""))
      rm(g)
      gc()
    }
    rm(layout.grid)
  }
  if(!align){
    for(i in z.final){
      g = graph.adjacency(as.matrix(x$path[[i]]), mode="undirected", diag=FALSE)
      layout.grid = layout.fruchterman.reingold(g)
      plot(g, layout=layout.grid, edge.color='gray50',vertex.color="red", vertex.size=3, vertex.label=NA, main = paste("lambda = ",as.character(round(x$lambda[i],3)),sep = ""))
      rm(g,layout.grid)
      gc()
    }
  }
  if(align) cat("Three plotted graphs are aligned according to the third graph\n")
}
