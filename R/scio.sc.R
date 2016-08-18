#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# scio(): Sparse Column Inverse Operator                                           #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 22nd, 2014                                                             #
# Version: 0.2.0                                                                   #
#----------------------------------------------------------------------------------#

scio.sc <- function(S, lambda, nlambda, gamma, d, maxdf, prec, max.ite, verbose, 
                           alg, method.flag, max.act.in,truncation)
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
  if(alg=="proximal") 
    L = eigen(S, only.values=TRUE)$values[1]
  else
    L = 0;
  nlambda = length(lambda)
  ite.lamb = rep(0,d*nlambda)
  ite.cyc = rep(0,d*nlambda)
  obj = array(0,dim=c(d,nlambda))
  runt = array(0,dim=c(d,nlambda))
  x = array(0,dim=c(d,maxdf,nlambda))
  col_cnz = rep(0,d+1)
  row_idx = rep(0,d*maxdf*nlambda)
  begt=Sys.time()
  str=.C("picasso_scio_sc", as.double(S), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.double(x), as.integer(col_cnz), as.integer(row_idx),
         as.double(obj), as.double(runt), as.integer(d), as.double(lambda), 
         as.integer(nlambda), as.integer(max.ite), as.double(prec), as.double(gamma), 
         as.integer(method.flag), as.double(truncation), as.integer(max.act.in), 
         as.integer(alg.flag), as.double(L), PACKAGE="picasso")
  runt1=Sys.time()-begt
  ite = list()
  ite[[1]] = matrix(unlist(str[2]), byrow = FALSE, ncol = nlambda)
  ite[[2]] = matrix(unlist(str[3]), byrow = FALSE, ncol = nlambda)
  x = unlist(str[4])
  col_cnz = unlist(str[5])
  row_idx = unlist(str[6])
  obj = matrix(unlist(str[7]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[8]),ncol=nlambda,byrow = FALSE)
  return(list(ite=ite, obj=obj,runt1=runt1,x=x, col_cnz=col_cnz, row_idx=row_idx, runt=runt))
}
