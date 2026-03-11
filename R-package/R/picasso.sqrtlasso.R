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
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = FALSE)
{
  dims = .picasso_validate_design(X)
  n = dims$n
  d = dims$d
  Yb = Y

  begt = Sys.time()

  if (verbose)
    cat("Sparse sqrt lasso regression. \n")

  design = .picasso_prepare_design(X, standardize)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  yy = Yb
  
  L0 = sqrt(sum(yy*yy)/n)
  lambda.max = max(abs(crossprod(xx,yy/n)))/L0
  lambda.info = .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda = lambda.info$lambda
  nlambda = lambda.info$nlambda

  method.info = .picasso_method_flag(method, gamma)
  method.flag = method.info$flag
  gamma = method.info$gamma
  
  dfmax.int <- if (is.null(dfmax)) as.integer(-1) else as.integer(dfmax)

  out = sqrtlasso_solver(yy, xx, lambda, nlambda, gamma,
              n, d, max.ite, prec, intercept, verbose,
              method.flag, dfmax.int)
  
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
  est$intercept = scaled$intercept
  est$lambda = lambda
  est$nlambda = nlambda
  est$df = df
  est$method = method
  est$alg = "actnewton"

  est$ite =out$ite
  est$verbose = verbose
  est$runtime = runt
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

coef.sqrtlasso <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.sqrtlasso <- function(object, newdata, lambda.idx = c(1:3), Y.pred.idx = c(1:5), ...)
{
  .picasso_predict(
    object,
    newdata,
    lambda.idx,
    Y.pred.idx,
    default_response_idx = c(1:5)
  )
}
