gaussian_solver <- function(Y, X, lambda, nlambda, gamma, n, d, df, max.ite, prec, 
                    verbose, standardize, intercept, method.flag, type.gaussian)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via active set identification and coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization via active set identification and coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization via active set identification and coordinate descent\n")
  }
 
  beta = rep(0, d*nlambda)
  beta.intcpt = rep(0, nlambda)
  beta.idx = rep(0, d*nlambda)
  ite.lamb = rep(0, nlambda)
  ite.cyc = rep(0, nlambda)
  runt = rep(0, nlambda)
  obj = rep(0, nlambda)
  col.cnz = rep(0, nlambda+1)
  cnz = 0
  err = 0

  if (type.gaussian == "covariance"){
     str=.C("picasso_gaussian_cov", 
         as.double(Y), as.double(X), 
         as.integer(n), as.integer(d),  
         as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec),
         as.integer(method.flag), as.integer(intercept),
         as.double(beta), as.double(beta.intcpt), as.integer(ite.lamb), 
         as.integer(beta.idx), 
        as.double(runt), 
        PACKAGE="picasso")
   } else {
     #print(n)
     #print(d)
     str=.C("picasso_gaussian_naive", 
           as.double(Y), as.double(X), 
         as.integer(n), as.integer(d),  
         as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec),
         as.integer(method.flag), as.integer(intercept),
         as.double(beta), as.double(beta.intcpt), as.integer(ite.lamb), 
         as.integer(beta.idx), 
        as.double(runt), 
          PACKAGE="picasso")
   }
  
 
  runt = matrix(unlist(str[16]), ncol = nlambda, byrow = FALSE)

  return(list(beta = unlist(str[12]), intcpt = unlist(str[13]), beta.idx = unlist(str[15]),
              ite = unlist(str[14]),  runt = runt, 
              err=0) # TODO: adding error message
              )
}
