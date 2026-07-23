sqrtlasso_solver <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,
    intercept, verbose, method.flag, dfmax, lla.max.stages = 3L)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via active-set quadratic-MM updates\n")
    if (method.flag == 2)
      cat("MCP regularization via active-set quadratic-MM updates and adaptive LLA\n")
    if (method.flag == 3)
      cat("SCAD regularization via active-set quadratic-MM updates and adaptive LLA\n")
  }

  out <- .Call("picasso_sqrtlasso_lla_call",
    as.double(Y), X,
    as.integer(n), as.integer(d),
    as.double(lambda), as.integer(nlambda),
    as.double(gamma), as.integer(max.ite),
    as.double(prec), as.integer(method.flag),
    as.integer(intercept),
    as.integer(dfmax),
    as.integer(lla.max.stages),
    PACKAGE = "picasso"
  )

  .picasso_scalar_lla_result(
    out, lambda, nlambda, lla.max.stages, "Sqrt-lasso"
  )
}
