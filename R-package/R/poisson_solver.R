poisson_solver <- function(Y, X, lambda, nlambda, gamma, n, d,
                          max.ite, prec, intercept, verbose, method.flag, dfmax,
                          offset = NULL, lla.max.stages = 3L)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via active-set Proximal Newton/IRLS\n")
    if (method.flag == 2)
      cat("MCP regularization via active-set Proximal Newton/IRLS and adaptive LLA\n")
    if (method.flag == 3)
      cat("SCAD regularization via active-set Proximal Newton/IRLS and adaptive LLA\n")
  }

  if (is.null(offset)) offset <- rep(0.0, n)

  out <- .Call("picasso_poisson_lla_call",
    as.double(Y), X,
    as.integer(n), as.integer(d),
    as.double(lambda), as.integer(nlambda),
    as.double(gamma), as.integer(max.ite),
    as.double(prec), as.integer(method.flag),
    as.integer(intercept),
    as.integer(dfmax),
    as.double(offset),
    as.integer(lla.max.stages),
    PACKAGE = "picasso"
  )

  .picasso_scalar_lla_result(
    out, lambda, nlambda, lla.max.stages, "Poisson"
  )
}
