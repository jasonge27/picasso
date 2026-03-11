picasso.gaussian <- function(X,
                          Y,
                          lambda = NULL,
                          nlambda = NULL,
                          lambda.min.ratio = NULL,
                          method = "l1",
                          type.gaussian = NULL,
                          gamma = 3,
                          df = NULL,
                          dfmax = NULL,
                          standardize = TRUE,
                          intercept = TRUE,
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = FALSE)
{
  begt = Sys.time()
  dims = .picasso_validate_design(X)
  n = dims$n
  d = dims$d
  if (verbose)
    cat("Sparse linear regression. \n")

  if (is.null(type.gaussian)) {
    type.gaussian = "naive"
  }

  if (type.gaussian!="naive" && type.gaussian!="covariance") {
    stop(sprintf(
      "Invalid `type.gaussian`: %s. Must be one of: naive, covariance.",
      type.gaussian
    ))
  }

  
  res.sd = FALSE 

  design = .picasso_prepare_design(X, standardize)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  if (standardize) {
    ym = mean(Y)
    y1 = Y-ym
    if (res.sd){
      sdy = sqrt(sum(y1^2)/(n-1))
      yy = y1/sdy
    } else {
      sdy = 1
      yy = y1
    }
  } else {
    sdy = 1
    yy = Y
  }

  # `df` is accepted for backward compatibility and is currently unused.
  
  est = list()
  xy = crossprod(xx,yy)
  lambda.max = max(abs(xy/n))
  lambda.info = .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda = lambda.info$lambda
  nlambda = lambda.info$nlambda

  method.info = .picasso_method_flag(method, gamma)
  method.flag = method.info$flag
  gamma = method.info$gamma

  dfmax.int <- if (is.null(dfmax)) as.integer(-1) else as.integer(dfmax)

  out = gaussian_solver(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose,
                       intercept, method.flag, type.gaussian, dfmax.int)

  if (out$err == 1) {
    stop("Parameters are too dense. Please choose larger `lambda`.")
  }
  if (out$err == 2) {
    warning("`df` may be too small. You may choose larger `df`.", call. = FALSE)
  }

  # truncate to actual number of lambdas fit (early stopping)
  num.fit = out$num.fit
  if (num.fit < nlambda) {
    lambda = lambda[1:num.fit]
    nlambda = num.fit
  }

  beta.raw = matrix(out$beta[1:(d * nlambda)], nrow = d, ncol = nlambda, byrow = FALSE)
  scaled = .picasso_rescale_solution(beta.raw, out$intcpt[1:nlambda], standardize, xinvc.vec, xm)

  est$beta = Matrix(scaled$beta)
  est$intercept = if (standardize) scaled$intercept + ym else scaled$intercept
  est$lambda = lambda * sdy
  est$df = colSums(beta.raw != 0)

  est$ite = out$ite[1:nlambda]

  runt = Sys.time()-begt

  est$nlambda = nlambda
  est$gamma = gamma
  est$method = method
  est$alg = paste("actgd", type.gaussian, sep = "-")
  est$verbose = verbose
  est$runtime = runt
  class(est) = "gaussian"
  return(est)
}

print.gaussian <- function(x, ...)
{  
  .picasso_print_summary(x, " Lasso options summary: ", method_label = "Method", show_alg = TRUE)
}

plot.gaussian <- function(x, ...)
{
  .picasso_plot_path(x)
}

coef.gaussian <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.gaussian <- function(object, newdata, lambda.idx = c(1:3), Y.pred.idx = c(1:5), ...)
{
  .picasso_predict(
    object,
    newdata,
    lambda.idx,
    Y.pred.idx,
    default_response_idx = c(1:5)
  )
}
