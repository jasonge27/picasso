picasso.sqrtlasso <- function(X,
                          Y,
                          lambda = NULL,
                          nlambda = NULL,
                          lambda.min.ratio = NULL,
                          method="l1",
                          gamma = 3,
                          dfmax = NULL,
                          standardize = TRUE,
                          intercept = TRUE,
                          prec = 1e-7,
                          max.ite = 1e4,
                          verbose = FALSE,
                          lla.max.stages = 3L,
                          fast.mode = FALSE)
{
  prec <- .picasso_resolve_precision(prec, fast.mode, "sqrtlasso")
  lla.max.stages <- .picasso_validate_lla_max_stages(lla.max.stages)
  standardize <- .picasso_validate_flag(standardize, "standardize")
  intercept <- .picasso_validate_flag(intercept, "intercept")
  verbose <- .picasso_validate_flag(verbose, "verbose")
  max.ite <- .picasso_validate_positive_integer(max.ite, "max.ite")
  dfmax <- .picasso_validate_nonnegative_integer(
    dfmax, "dfmax", allow.null = TRUE
  )
  dims = .picasso_validate_design(X)
  n = dims$n
  d = dims$d
  if (!is.numeric(Y) || length(Y) != n || anyNA(Y) ||
      any(!is.finite(Y))) {
    stop(sprintf("Y must be a finite numeric vector of length %d.", n))
  }
  Yb = as.double(Y)

  begt = Sys.time()

  if (verbose)
    cat("Sparse sqrt lasso regression. \n")

  design = .picasso_prepare_design(X, standardize, center = intercept)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  yy = Yb
  
  lambda.max = if (is.null(lambda)) {
    eta0 <- .picasso_null_eta(yy, "sqrtlasso", intercept = intercept)
    residual0 <- yy - eta0
    L0 = sqrt(sum(residual0 * residual0) / n)
    if (L0 == 0) 0 else max(abs(crossprod(xx, residual0 / n))) / L0
  } else {
    0.0
  }
  lambda.info = .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda = lambda.info$lambda
  nlambda = lambda.info$nlambda

  method.info = .picasso_method_flag(method, gamma)
  method.flag = method.info$flag
  gamma = method.info$gamma
  
  dfmax.int <- if (is.null(dfmax)) -1L else dfmax

  out = sqrtlasso_solver(yy, xx, lambda, nlambda, gamma,
              n, d, max.ite, prec, intercept, verbose,
              method.flag, dfmax.int, lla.max.stages)
  
  # truncate to actual number of lambdas fit (early stopping)
  num.fit = out$num.fit
  if (num.fit < nlambda) {
    lambda = lambda[1:num.fit]
    nlambda = num.fit
  }

  est = list()
  beta.raw = matrix(out$beta[1:(d * nlambda)], nrow = d, ncol = nlambda, byrow = FALSE)
  df = as.integer(colSums(beta.raw != 0))
  scaled = .picasso_rescale_solution(beta.raw, out$intcpt, standardize, xinvc.vec, xm)

  runt = Sys.time()-begt
  est$runt = out$runt
  est$beta = Matrix(scaled$beta)
  est$intercept = if (intercept) scaled$intercept else
    rep(0.0, length(scaled$intercept))
  est$lambda = lambda
  est$nlambda = nlambda
  est$df = df
  est$method = method
  est$alg = "active-set-quadratic-mm"

  est$ite =out$ite
  est$lla.max.stages = out$lla.max.stages
  est$status = out$status
  est$status.code = out$status.code
  est$failure = out$failure
  est$diagnostics = out$diagnostics
  est$verbose = verbose
  est$runtime = runt
  est$fast.mode = fast.mode
  est$prec = prec

  est$nulldev <- .picasso_null_deviance(
    Yb, "sqrtlasso", intercept = intercept
  )
  fit_dev <- 0.5 * out$smooth.objective^2
  est$dev.ratio <- if (est$nulldev > 0) {
    pmax(0, pmin(1, 1 - fit_dev / est$nulldev))
  } else {
    rep(0.0, length(fit_dev))
  }

  class(est) = "sqrtlasso"
  return(est)
}

print.sqrtlasso <- function(x, ...)
{  
  .picasso_print_summary(x, " SQRT Lasso options summary: ")
}

plot.sqrtlasso <- function(x, ...)
{
  .picasso_plot_path(x)
}

coef.sqrtlasso <- function(object, lambda.idx = NULL, beta.idx = NULL, ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.sqrtlasso <- function(object, newdata, lambda.idx = NULL, Y.pred.idx = NULL,
                              type = "response", s = NULL, ...)
{
  .picasso_predict(
    object,
    newdata,
    lambda.idx,
    Y.pred.idx,
    type = type,
    s = s
  )
}
