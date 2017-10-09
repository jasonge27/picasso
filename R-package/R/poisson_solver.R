poisson_solver <- function(Y, X, lambda, nlambda, gamma, n, d, 
                          max.ite, prec, intercept, verbose, method.flag)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via greedy active set identification and coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization via greedy active set identification and coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization via greedy active set identification and coordinate descent\n")
  }
  beta = matrix(0, nrow = d, ncol = nlambda)
  beta.intcpt = rep(0, nlambda)
  size.act = rep(0, nlambda)
  ite.lamb = rep(0, nlambda)
  ite.cyc = rep(0, nlambda)
  runt = matrix(0, 1, nlambda)
  obj = matrix(0, 1, nlambda)
  str=.C("picasso_poisson_solver", 
    as.double(Y), as.double(X), 
         as.integer(n), as.integer(d), 
         as.double(lambda), as.integer(nlambda),
         as.double(gamma), as.integer(max.ite),
         as.double(prec), as.integer(method.flag),
         as.integer(intercept),
         as.double(beta), as.double(beta.intcpt), 
         as.integer(ite.lamb), as.integer(size.act), 
         as.double(runt),
         PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for (i in 1:nlambda){
    beta.i = unlist(str[12])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[13])
  ite.lamb = unlist(str[14])
  size.act = unlist(str[15])
  runt = matrix(unlist(str[16]), ncol = nlambda, byrow = FALSE)
  return(list(beta = beta.list, intcpt = beta.intcpt, ite=ite.lamb, size.act = size.act,
         runt = runt))
}
