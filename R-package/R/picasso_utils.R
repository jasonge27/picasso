.picasso_method_flag <- function(method, gamma) {
  if (method == "l1") {
    return(list(flag = 1L, gamma = gamma))
  }

  if (method == "mcp") {
    if (gamma <= 1) {
      warning("gamma > 1 is required for MCP. Set to default value 3.")
      gamma <- 3
    }
    return(list(flag = 2L, gamma = gamma))
  }

  if (method == "scad") {
    if (gamma <= 2) {
      warning("gamma > 2 is required for SCAD. Set to default value 3.")
      gamma <- 3
    }
    return(list(flag = 3L, gamma = gamma))
  }

  stop(sprintf("Invalid `method`: %s. Must be one of: l1, mcp, scad.", method))
}


.picasso_standardize <- function(X) {
  n <- nrow(X)
  d <- ncol(X)

  out <- .Call("picasso_standardize_call",
    X, as.integer(n), as.integer(d),
    PACKAGE = "picasso"
  )

  list(
    xx = matrix(out$xx, nrow = n, ncol = d, byrow = FALSE),
    xm = matrix(out$xm, nrow = 1),
    xinvc.vec = out$xinvc
  )
}


.picasso_validate_design <- function(X) {
  n <- nrow(X)
  d <- ncol(X)
  if (n == 0 || d == 0) {
    stop("No data input.")
  }
  list(n = n, d = d)
}


.picasso_prepare_design <- function(X, standardize) {
  if (standardize) {
    std <- .picasso_standardize(X)
    return(std)
  }

  list(
    xx = X,
    xm = matrix(0, nrow = 1, ncol = ncol(X)),
    xinvc.vec = rep(1, ncol(X))
  )
}


.picasso_lambda_path <- function(lambda, nlambda, lambda.min.ratio, lambda.max) {
  if (!is.null(lambda)) {
    return(list(lambda = lambda, nlambda = length(lambda)))
  }

  if (is.null(nlambda)) {
    nlambda <- 100L
  }

  if (is.null(lambda.min.ratio)) {
    lambda.min <- 0.05 * lambda.max
  } else {
    lambda.min <- min(lambda.min.ratio * lambda.max, lambda.max)
  }

  if (lambda.min >= lambda.max) {
    stop(sprintf(
      "Invalid `lambda.min.ratio`: generated lambda.min (%.4g) must be smaller than lambda.max (%.4g).",
      lambda.min, lambda.max
    ))
  }

  list(
    lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda)),
    nlambda = nlambda
  )
}


.picasso_rescale_solution <- function(beta.raw, intcpt.raw, standardize, xinvc.vec, xm) {
  if (standardize) {
    beta <- beta.raw * xinvc.vec
    intercept <- intcpt.raw - as.numeric(xm %*% beta)
  } else {
    beta <- beta.raw
    intercept <- intcpt.raw
  }

  list(beta = beta, intercept = intercept)
}


.picasso_runtime_unit <- function(runtime) {
  as.character(units(runtime))
}


.picasso_print_summary <- function(x, header, method_label = NULL, show_alg = FALSE) {
  cat("\n", header, "\n", sep = "")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda, digits = 3))
  if (!is.null(method_label)) {
    cat(method_label, "=", x$method, "\n")
  }
  if (show_alg) {
    cat("Alg =", x$alg, "\n")
  }
  cat("Degree of freedom:", min(x$df), "----->", max(x$df), "\n")
  cat("Runtime:", x$runtime, " ", .picasso_runtime_unit(x$runtime), "\n")
  invisible(x)
}


.picasso_plot_path <- function(x) {
  matplot(
    x$lambda,
    t(x$beta),
    type = "l",
    main = "Regularization Path",
    xlab = "Regularization Parameter",
    ylab = "Coefficient"
  )
  invisible(NULL)
}


.picasso_validate_indices <- function(idx, n, name) {
  if (length(idx) == 0L) {
    stop(sprintf("`%s` must contain at least one index.", name))
  }

  if (any(idx < 1L) || any(idx > n)) {
    stop(sprintf("`%s` contains out-of-range indices. Valid range is 1..%d.", name, n))
  }
}


.picasso_extract_coef <- function(object, lambda.idx, beta.idx) {
  lambda.idx <- as.integer(lambda.idx)
  beta.idx <- as.integer(beta.idx)
  .picasso_validate_indices(lambda.idx, object$nlambda, "lambda.idx")
  .picasso_validate_indices(beta.idx, nrow(object$beta), "beta.idx")

  beta.block <- as.matrix(object$beta[beta.idx, lambda.idx, drop = FALSE])
  coef.mat <- rbind(
    "(Intercept)" = as.numeric(object$intercept[lambda.idx]),
    beta.block
  )

  rownames(coef.mat)[-1] <- paste0("beta[", beta.idx, "]")
  colnames(coef.mat) <- paste0("lambda[", lambda.idx, "]")
  coef.mat
}


.picasso_predict <- function(object, newdata, lambda.idx, response.idx,
                             default_response_idx, transform = identity) {
  lambda.idx <- as.integer(lambda.idx)
  .picasso_validate_indices(lambda.idx, object$nlambda, "lambda.idx")

  pred.n <- nrow(newdata)
  lambda.n <- length(lambda.idx)

  intcpt <- matrix(
    rep(object$intercept[lambda.idx], pred.n),
    nrow = pred.n,
    ncol = lambda.n,
    byrow = TRUE
  )

  linear <- newdata %*% object$beta[, lambda.idx] + intcpt
  pred <- as.matrix(transform(linear))

  is_default_idx <- length(response.idx) == length(default_response_idx) &&
    isTRUE(all(response.idx == default_response_idx))

  if (!is_default_idx) {
    response.idx <- as.integer(response.idx)
    .picasso_validate_indices(response.idx, pred.n, "response.idx")
    pred <- pred[response.idx, , drop = FALSE]
  }

  colnames(pred) <- paste0("lambda[", lambda.idx, "]")
  pred
}
