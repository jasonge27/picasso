gaussian_solver <- function(Y, X, lambda, nlambda, gamma, n, d, df, max.ite, prec, 
                    verbose, standardize, method.flag, type.gaussian)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via active set identification and coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization via active set identification and coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization via active set identification and coordinate descent\n")
  }
 
  maxdf = min(n, d)
  beta = rep(0, maxdf*nlambda)
  beta.intcpt = rep(0, nlambda)
  beta.idx = rep(0, maxdf*nlambda)
  ite.lamb = rep(0, nlambda)
  ite.cyc = rep(0, nlambda)
  runt = rep(0, nlambda)
  obj = rep(0, nlambda)
  col.cnz = rep(0, nlambda+1)
  cnz = 0
  err = 0

  if (type.gaussian == "covariance"){
     str=.C("picasso_gaussian_cov", as.double(Y), as.double(X),
         as.double(beta), as.double(beta.intcpt), as.integer(beta.idx), 
         as.integer(cnz), as.integer(col.cnz), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.double(obj), as.double(runt), as.integer(err), as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), as.integer(method.flag), 
          as.integer(n), as.integer(d), as.integer(df), 
          as.integer(verbose), as.integer(standardize), PACKAGE="picasso")
   } else {
     str=.C("picasso_gaussian_naive", as.double(Y), as.double(X), 
         as.double(beta), as.double(beta.intcpt), as.integer(beta.idx), 
         as.integer(cnz), as.integer(col.cnz), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.double(obj), as.double(runt), as.integer(err), as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), as.integer(method.flag), 
          as.integer(n), as.integer(d), as.integer(df), 
          as.integer(verbose), as.integer(standardize), PACKAGE="picasso")
   }
  
 
  ite = list()
  ite[[1]] = unlist(str[8])
  ite[[2]] = unlist(str[9])
  obj = matrix(unlist(str[10]), ncol = nlambda, byrow = FALSE)
  runt = matrix(unlist(str[11]), ncol = nlambda, byrow = FALSE)
#   
#   if(err==1)
#     cat("Parameters are too dense")
  return(list(beta = unlist(str[3]), intcpt = unlist(str[4]), beta.idx = unlist(str[5]),
              ite = ite, obj = obj, runt = runt, cnz = unlist(str[6]),
              col.cnz = unlist(str[7]), err = unlist(str[12])))
}
