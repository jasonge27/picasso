gaussian_solver <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,
                    verbose, intercept, method.flag, type.gaussian, dfmax)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via active set identification and coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization via active set identification and coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization via active set identification and coordinate descent\n")
  }

  solver_name <- if (type.gaussian == "covariance") "picasso_gaussian_cov_call" else "picasso_gaussian_naive_call"
  out <- .Call(solver_name,
    Y, X,
    as.integer(n), as.integer(d),
    lambda, as.integer(nlambda),
    as.double(gamma), as.integer(max.ite), as.double(prec),
    as.integer(method.flag), as.integer(intercept),
    as.integer(dfmax),
    PACKAGE = "picasso"
  )

  return(list(
    beta = out$beta,
    intcpt = out$intcpt,
    ite = out$ite_lamb,
    num.fit = out$num_fit,
    err = 0
  ))
}
