sqrtlasso_solver <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,
    intercept, verbose, method.flag, dfmax)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via greedy active set identification and coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization via greedy active set identification and coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization via greedy active set identification and coordinate descent\n")
  }

  out <- .Call("picasso_sqrtlasso_call",
    Y, X,
    as.integer(n), as.integer(d),
    lambda, as.integer(nlambda),
    as.double(gamma), as.integer(max.ite),
    as.double(prec), as.integer(method.flag),
    as.integer(intercept),
    as.integer(dfmax),
    PACKAGE = "picasso"
  )

  num.fit <- out$num_fit

  return(list(beta = out$beta, intcpt = out$intcpt[1:num.fit],
         ite = out$ite_lamb[1:num.fit],
         size.act = out$size_act[1:num.fit],
         runt = out$runt[1:num.fit],
         num.fit = num.fit))
}
