multinomial_solver <- function(Y_int, X, lambda, nlambda, gamma,
                               n, d, K, max.ite,
                               prec, intercept, verbose, method.flag, dfmax)
{
  if (verbose) {
    if (method.flag == 1)
      cat("L1 regularization (multinomial) via coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization (multinomial) via coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization (multinomial) via coordinate descent\n")
  }

  out <- .Call("picasso_multinomial_call",
    as.double(Y_int), X,
    as.integer(n), as.integer(d), as.integer(K),
    lambda, as.integer(nlambda),
    as.double(gamma), as.integer(max.ite),
    as.double(prec), as.integer(method.flag),
    as.integer(intercept),
    as.integer(dfmax),
    PACKAGE = "picasso"
  )

  num.fit <- out$num_fit

  return(list(
    beta    = out$beta,          # d * K * nlambda flat
    intcpt  = out$intcpt,        # K * nlambda flat
    ite     = out$ite_lamb[seq_len(num.fit)],
    size.act = out$size_act[seq_len(num.fit)],
    runt    = out$runt[seq_len(num.fit)],
    num.fit = num.fit
  ))
}
