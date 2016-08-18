#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# lasso(): Lasso                                                                   #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Sep 1st, 2015                                                              #
# Version: 0.4.5                                                                   #
#----------------------------------------------------------------------------------#

lasso.sc.cov <- function(Y, X, XY, lambda, nlambda, gamma, n, d, df, max.ite, prec, verbose, 
                         alg, method.flag, max.act.in, truncation)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("L1 regularization via",alg,"active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("MCP regularization via",alg,"active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("SCAD regularization via",alg,"active set identification and coordinate descent\n")
  }
  if(alg=="cyclic") alg.flag=1
  if(alg=="greedy") alg.flag=2
  if(alg=="proximal") alg.flag=3
  if(alg=="random") alg.flag=4 
  if(alg=="hybrid") alg.flag=5
  L = d*n
  maxdf = min(n,d)
  beta = rep(0,maxdf*nlambda)
  beta.intcpt = rep(0,nlambda)
  beta.idx = rep(0,maxdf*nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = rep(0,nlambda)
  obj = rep(0,nlambda)
  col.cnz = rep(0,nlambda+1)
  cnz = 0
  err = 0
  str=.C("picasso_lasso_sc_cov", as.double(Y), as.double(X), as.double(XY), 
         as.double(beta), as.double(beta.intcpt), as.integer(beta.idx), 
         as.integer(cnz), as.integer(col.cnz), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.double(obj), as.double(runt), as.integer(err), as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), as.integer(method.flag), 
         as.double(truncation), as.integer(n), as.integer(d), as.integer(df), 
         as.integer(max.act.in), as.integer(alg.flag), as.double(L), PACKAGE="picasso")
  ite = list()
  ite[[1]] = unlist(str[9])
  ite[[2]] = unlist(str[10])
  obj = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[12]),ncol=nlambda,byrow = FALSE)
#   
#   if(err==1)
#     cat("Parameters are too dense")
  return(list(beta=unlist(str[4]), intcpt=unlist(str[5]), beta.idx=unlist(str[6]),
              ite=ite, obj = obj, runt = runt, cnz = unlist(str[7]),
              col.cnz = unlist(str[8]), err = unlist(str[13])))
}
