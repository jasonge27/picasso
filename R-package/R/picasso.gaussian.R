.picasso_prepare_gaussian_design <- function(X, standardize, intercept) {
  X <- as.matrix(X)
  # Avoid copy-on-write for an already-double caller matrix. Integer designs
  # still need one owning conversion before entering the native double API.
  if (!is.double(X))
    storage.mode(X) <- "double"

  n <- nrow(X)
  d <- ncol(X)
  # Native Gaussian objectives now profile the intercept themselves. Raw
  # unstandardized data can therefore pass through without an n-by-d centering
  # copy. Original column means are needed only to undo standardized centering.
  if (standardize && intercept) {
    standardized <- .picasso_standardize(X)
    return(list(
      xx = standardized$xx,
      xm = as.numeric(standardized$xm),
      xinvc.vec = standardized$xinvc.vec
    ))
  }

  xm <- rep(0.0, d)
  xx <- X
  xinvc.vec <- rep(1.0, d)

  if (standardize) {
    divisor <- max(n - 1L, 1L)
    xinvc.vec[] <- 0.0
    for (j in seq_len(d)) {
      column <- xx[, j]
      maximum <- max(abs(column))
      if (maximum > 0) {
        relative.norm <- sqrt(sum((column / maximum)^2) / divisor)
        if (relative.norm > 0) {
          xinvc.vec[j] <- (1 / maximum) / relative.norm
          xx[, j] <- column * xinvc.vec[j]
        }
      }
    }
  }

  list(
    xx = xx,
    xm = xm,
    xinvc.vec = xinvc.vec
  )
}


.picasso_resolve_gaussian_type <- function(type.gaussian, n, d, lambda) {
  # Preserve the historical direct-call meaning of NULL while making the
  # public default explicit and inspectable as "auto".
  if (is.null(type.gaussian)) {
    return("naive")
  }
  type.gaussian <- .picasso_validate_choice(
    type.gaussian, c("auto", "naive", "covariance"), "type.gaussian"
  )
  if (type.gaussian != "auto") {
    return(type.gaussian)
  }

  # Covariance updates amortize their lazy Gram-column cache on small designs,
  # sparse regularization paths, or very tall problems. Denser paths need a
  # more conservative shape threshold because many cached columns can erase
  # that benefit. Limit a fully populated cache to 8 MiB; users can still
  # request covariance explicitly outside these guardrails.
  max.covariance.features <- 1024L
  enough.path.reuse <- length(lambda) >= 8L
  has.usable.features <- d > 0L
  cache.within.budget <- d <= max.covariance.features
  not.wide <- n >= d
  lambda.ratio <- if (length(lambda) > 0L && lambda[1L] > 0) {
    lambda[length(lambda)] / lambda[1L]
  } else {
    0.0
  }
  ratio.tolerance <- 1e-12
  small.design <- d <= 160L
  sparse.path <- lambda.ratio + ratio.tolerance >= 0.10
  moderately.tall.default.path <-
    lambda.ratio + ratio.tolerance >= 0.05 && n >= 4 * as.double(d)
  very.tall <- n >= 16 * as.double(d)
  use.covariance <- small.design || sparse.path ||
    moderately.tall.default.path || very.tall
  if (has.usable.features && enough.path.reuse && cache.within.budget &&
      not.wide && use.covariance) {
    "covariance"
  } else {
    "naive"
  }
}


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
                          prec = 1e-7,
                          max.ite = 1e4,
                          verbose = FALSE,
                          fast.mode = FALSE)
{
  prec <- .picasso_resolve_precision(prec, fast.mode, "gaussian")
  standardize <- .picasso_validate_flag(standardize, "standardize")
  intercept <- .picasso_validate_flag(intercept, "intercept")
  verbose <- .picasso_validate_flag(verbose, "verbose")
  max.ite <- .picasso_validate_positive_integer(max.ite, "max.ite")
  dfmax <- .picasso_validate_nonnegative_integer(
    dfmax, "dfmax", allow.null = TRUE
  )
  begt = Sys.time()
  dims = .picasso_validate_design(X)
  n = dims$n
  d = dims$d
  if (!is.numeric(Y) || length(Y) != n || anyNA(Y) ||
      any(!is.finite(Y))) {
    stop(sprintf("Y must contain %d finite numeric values.", n))
  }
  Y <- as.double(Y)
  if (verbose)
    cat("Sparse linear regression. \n")

  # Scaling and centering remain independent public choices. The native
  # objective profiles an included intercept; wrapper centering is needed only
  # when coefficients are standardized and later rescaled.
  design = .picasso_prepare_gaussian_design(X, standardize, intercept)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  # `df` is accepted for backward compatibility and is currently unused.
  
  est = list()
  lambda.max = if (!is.null(lambda)) {
    0.0
  } else {
    null.residual = if (intercept) Y - mean(Y) else Y
    max(abs(crossprod(xx, null.residual) / n))
  }
  lambda.info = .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda = lambda.info$lambda
  nlambda = lambda.info$nlambda
  requested.type.gaussian <- if (is.null(type.gaussian)) {
    "naive"
  } else {
    type.gaussian
  }
  type.gaussian <- .picasso_resolve_gaussian_type(
    type.gaussian, n, d, lambda
  )

  method.info = .picasso_method_flag(method, gamma)
  method.flag = method.info$flag
  gamma = method.info$gamma

  dfmax.int <- if (is.null(dfmax)) -1L else dfmax

  out = gaussian_solver(
    Y, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose,
    intercept, method.flag, type.gaussian, dfmax.int
  )

  # truncate to actual number of lambdas fit (early stopping)
  num.fit = out$num.fit
  if (length(num.fit) != 1L || is.na(num.fit) || num.fit < 1L ||
      num.fit > nlambda) {
    stop("Gaussian solver returned no usable lambda values.")
  }
  if (num.fit < nlambda) {
    lambda = lambda[1:num.fit]
    nlambda = num.fit
  }

  beta.raw <- matrix(
    out$beta[seq_len(d * nlambda)], nrow = d, ncol = nlambda
  )
  beta <- if (standardize) beta.raw * xinvc.vec else beta.raw

  intcpt.raw <- out$intcpt[seq_len(nlambda)]
  fitted.intercept <- if (intercept) {
    intcpt.raw - drop(crossprod(xm, beta))
  } else {
    rep(0.0, nlambda)
  }

  est$beta = Matrix(beta)
  est$intercept = fitted.intercept
  est$lambda = lambda
  est$df = as.integer(colSums(beta != 0))

  est$ite = out$ite[1:nlambda]
  est$status = out$status
  est$status.code = out$status.code
  est$failure = out$failure

  runt = Sys.time()-begt

  est$nlambda = nlambda
  est$gamma = gamma
  est$method = method
  est$type.gaussian.requested = requested.type.gaussian
  est$type.gaussian = type.gaussian
  est$alg = paste("actgd", type.gaussian, sep = "-")
  est$verbose = verbose
  est$runtime = runt
  est$fast.mode = fast.mode
  est$prec = prec

  est$nulldev <- .picasso_null_deviance(
    Y, "gaussian", intercept = intercept
  )
  fit_dev <- 0.5 * out$smooth.objective[seq_len(nlambda)]
  est$dev.ratio <- if (est$nulldev > 0) {
    pmax(0, pmin(1, 1 - fit_dev / est$nulldev))
  } else {
    rep(0.0, nlambda)
  }

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

coef.gaussian <- function(object, lambda.idx = NULL, beta.idx = NULL, ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.gaussian <- function(object, newdata, lambda.idx = NULL, Y.pred.idx = NULL,
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
