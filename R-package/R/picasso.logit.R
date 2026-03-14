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
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = FALSE,
                          offset = NULL)
{
  dims = .picasso_validate_design(X)
  n = dims$n
  d = dims$d
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

  design = .picasso_prepare_design(X, standardize)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  yy = Yb
  
  lambda.max = max(abs(crossprod(xx,yy/n)))
  lambda.info = .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda = lambda.info$lambda
  nlambda = lambda.info$nlambda

  method.info = .picasso_method_flag(method, gamma)
  method.flag = method.info$flag
  gamma = method.info$gamma
  
  dfmax.int <- if (is.null(dfmax)) as.integer(-1) else as.integer(dfmax)
  offset.vec <- if (is.null(offset)) rep(0.0, n) else as.double(offset)

  out = logit_solver(yy, xx, lambda, nlambda, gamma,
              n, d, max.ite, prec, intercept, verbose,
              method.flag, dfmax.int, offset.vec)
  
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
  est$intercept = scaled$intercept
  est$lambda = lambda
  est$nlambda = nlambda
  est$df = df
  est$method = method
  est$alg = "actnewton"
 
  est$ite =out$ite
  est$verbose = verbose
  est$runtime = runt

  est$nulldev <- .picasso_null_deviance(Yb, "binomial")
  fit_dev <- .picasso_fit_deviance(Yb, X, as.matrix(est$beta), est$intercept, "binomial")
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

coef.logit <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.logit <- function(object, newdata, lambda.idx = c(1:3), p.pred.idx = c(1:5),
                          type = "response", s = NULL, ...)
{
  if (type == "class") {
    probs <- .picasso_predict(
      object, newdata, lambda.idx, p.pred.idx,
      default_response_idx = c(1:5),
      transform = function(z) exp(z) / (1 + exp(z)),
      type = "response",
      s = s
    )
    return(matrix(as.integer(probs > 0.5), nrow = nrow(probs),
                  dimnames = list(NULL, colnames(probs))))
  }
  .picasso_predict(
    object,
    newdata,
    lambda.idx,
    p.pred.idx,
    default_response_idx = c(1:5),
    transform = function(z) exp(z) / (1 + exp(z)),
    type = type,
    s = s
  )
}
