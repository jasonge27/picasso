picasso.logit <- function(X,
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
                          offset = NULL,
                          lla.max.stages = 3L,
                          fast.mode = FALSE)
{
  prec <- .picasso_resolve_precision(prec, fast.mode, "binomial")
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
  if (length(Y) != n || anyNA(Y) ||
      (is.numeric(Y) && any(!is.finite(Y)))) {
    stop(sprintf("Y must contain %d finite, non-missing observations.", n))
  }
  Y = as.factor(Y)
  if (length(levels(Y)) != 2){
    stop(sprintf(
      "Response vector must contain exactly 2 levels; found %d.",
      length(levels(Y))
    ))
  }
  Yb = rep(0, n)
  Yb[which(Y == levels(Y)[2])] = 1

  begt = Sys.time()

  if (verbose)
    cat("Sparse logistic regression. \n")

  offset.vec <- .picasso_validate_offset(offset, n, "binomial")
  design = .picasso_prepare_design(X, standardize, center = intercept)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  yy = Yb
  
  lambda.max = if (is.null(lambda)) {
    eta0 <- .picasso_null_eta(yy, "binomial", offset.vec, intercept)
    mu0 <- stats::plogis(eta0)
    max(abs(crossprod(xx, (yy - mu0) / n)))
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
  out = logit_solver(yy, xx, lambda, nlambda, gamma,
              n, d, max.ite, prec, intercept, verbose,
              method.flag, dfmax.int, offset.vec, lla.max.stages)
  
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
  est$beta = Matrix(scaled$beta)
  est$intercept = if (intercept) scaled$intercept else
    rep(0.0, length(scaled$intercept))
  est$lambda = lambda
  est$nlambda = nlambda
  est$df = df
  est$method = method
  est$alg = "actnewton"
  est$runt = out$runt
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
  est$offset.used = !is.null(offset)
  est$levels = levels(Y)

  est$nulldev <- .picasso_null_deviance(
    Yb, "binomial", offset = offset.vec, intercept = intercept
  )
  fit_dev <- out$smooth.objective
  est$dev.ratio <- pmax(0, pmin(1, 1 - fit_dev / est$nulldev))

  class(est) = "logit"
  return(est)
}

print.logit <- function(x, ...)
{  
  .picasso_print_summary(x, " Logit options summary: ", method_label = "Method", show_alg = TRUE)
}

plot.logit <- function(x, ...)
{
  .picasso_plot_path(x)
}

coef.logit <- function(object, lambda.idx = NULL, beta.idx = NULL, ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.logit <- function(object, newdata, lambda.idx = NULL, p.pred.idx = NULL,
                          type = "response", s = NULL, newoffset = NULL, ...)
{
  type <- .picasso_validate_choice(
    type, c("response", "link", "class", "nonzero"), "type"
  )
  if (type == "class") {
    link <- .picasso_predict(
      object, newdata, lambda.idx, p.pred.idx,
      transform = identity,
      type = "link",
      s = s,
      newoffset = newoffset
    )
    return(matrix(as.integer(link > 0), nrow = nrow(link),
                  dimnames = list(NULL, colnames(link))))
  }
  .picasso_predict(
    object,
    newdata,
    lambda.idx,
    p.pred.idx,
    transform = stats::plogis,
    type = type,
    s = s,
    newoffset = newoffset
  )
}
