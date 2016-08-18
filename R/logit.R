#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# logit(): Logistic regression                                                     #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 22nd, 2014                                                             #
# Version: 0.2.0                                                                   #
#----------------------------------------------------------------------------------#

logit.cyclic <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,verbose,
                         method.flag, max.act.in, truncation)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("L1 regularization via cyclic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("MCP regularization via cyclic active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("SCAD regularization via cyclic active set identification and coordinate descent\n")
  }
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  str=.C("picasso_logit_cyclic", as.double(Y), as.double(X), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(size.act),
         as.double(obj), as.double(runt), as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), 
         as.integer(method.flag), as.integer(max.act.in), as.double(truncation), 
         PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act,
         obj = obj, runt = runt))
}

logit.greedy <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,verbose,method.flag)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("L1 regularization via greedy active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("MCP regularization via greedy active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("SCAD regularization via greedy active set identification and coordinate descent\n")
  }
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  str=.C("picasso_logit_greedy", as.double(Y), as.double(X), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(size.act),
         as.double(obj), as.double(runt), as.double(lambda), as.integer(nlambda), as.double(gamma), 
         as.integer(max.ite), as.double(prec), as.integer(method.flag), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act,
         obj = obj, runt = runt))
}

logit.prox <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,verbose,method.flag)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("L1 regularization via proximal gradient active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("MCP regularization via proximal gradient active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("SCAD regularization via proximal gradient active set identification and coordinate descent\n")
  }
  #L = max(colSums(abs(crossprod(X))))/n
  #L = eigen(crossprod(X)/n)$values[1]
  L = d/20
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  str=.C("picasso_logit_prox", as.double(Y), as.double(X), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(size.act), 
         as.double(obj), as.double(runt), as.double(lambda), as.integer(nlambda), as.double(gamma), 
         as.integer(max.ite), as.double(prec), as.double(L), as.integer(method.flag), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act,
         obj = obj, runt = runt))
}

logit.stoc <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,verbose,method.flag,
                       max.act.in, truncation)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("L1 regularization via stochastic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("MCP regularization via stochastic active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("SCAD regularization via stochastic active set identification and coordinate descent\n")
  }
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  str=.C("picasso_logit_stoc", as.double(Y), as.double(X), as.double(beta), 
         as.double(beta.intcpt), as.integer(n), as.integer(d), as.integer(ite.lamb), 
         as.integer(ite.cyc), as.integer(size.act), as.double(obj), 
         as.double(runt), as.double(lambda), as.integer(nlambda), as.double(gamma), 
         as.integer(max.ite), as.double(prec), as.integer(method.flag), 
         as.integer(max.act.in), as.double(truncation), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act,
         obj = obj, runt = runt))
}
