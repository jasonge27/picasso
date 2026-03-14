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


# Proper Poisson deviance: 2 * mean(y*log(y/mu) - (y-mu)), always >= 0.
# Convention: 0*log(0) = 0.
.picasso_poisson_dev <- function(y, mu) {
  mu <- pmax(mu, 1e-15)
  term <- ifelse(y > 0, y * log(y / mu) - (y - mu), mu - y)
  2 * mean(term)
}


.picasso_null_deviance <- function(Y, family, offset = NULL) {
  Y <- as.numeric(Y)
  n <- length(Y)
  if (family %in% c("gaussian", "sqrtlasso")) {
    mu0 <- mean(Y)
    sum((Y - mu0)^2) / (2 * n)
  } else if (family == "binomial") {
    p0 <- pmax(pmin(mean(Y), 1 - 1e-8), 1e-8)
    -mean(Y * log(p0) + (1 - Y) * log(1 - p0))
  } else if (family == "poisson") {
    if (!is.null(offset) && any(offset != 0)) {
      # Null model with offset: log(mu_i) = offset_i + c
      # MLE: sum(exp(offset_i + c)) = sum(y_i) => c = log(mean(y) / mean(exp(offset)))
      exp_off  <- exp(offset)
      mean_eoff <- mean(exp_off)
      c0 <- if (mean_eoff > 0 && mean(Y) > 0) log(mean(Y) / mean_eoff) else 0
      mu0 <- exp_off * exp(c0)
    } else {
      mu0 <- rep(max(mean(Y), 1e-15), n)
    }
    .picasso_poisson_dev(Y, mu0)
  } else {
    NA_real_
  }
}


.picasso_fit_deviance <- function(Y, X, beta_mat, intercept_vec, family,
                                  offset = NULL) {
  Y <- as.numeric(Y)
  n <- nrow(X)
  nlambda <- ncol(beta_mat)
  off <- if (!is.null(offset) && any(offset != 0)) offset else rep(0, n)
  eta_mat <- X %*% beta_mat + matrix(rep(intercept_vec, each = n), nrow = n) +
             matrix(rep(off, nlambda), nrow = n)
  if (family %in% c("gaussian", "sqrtlasso")) {
    colSums((Y - eta_mat)^2) / (2 * n)
  } else if (family == "binomial") {
    vapply(seq_len(nlambda), function(k) {
      p <- pmax(pmin(1 / (1 + exp(-eta_mat[, k])), 1 - 1e-8), 1e-8)
      -mean(Y * log(p) + (1 - Y) * log(1 - p))
    }, numeric(1))
  } else if (family == "poisson") {
    vapply(seq_len(nlambda), function(k) {
      mu <- exp(eta_mat[, k])
      .picasso_poisson_dev(Y, mu)
    }, numeric(1))
  } else {
    rep(NA_real_, nlambda)
  }
}


# Resolve s= (lambda values) to interpolated beta/intercept columns.
# Returns a list(beta_mat, intercept_vec, col_names) ready for prediction.
# lambda path is assumed decreasing: lams[1] >= lams[2] >= ... >= lams[nlambda].
.picasso_resolve_s <- function(object, s) {
  lams    <- object$lambda
  nlam    <- length(lams)
  beta_m  <- as.matrix(object$beta)        # d x nlambda
  intcpt  <- as.numeric(object$intercept)  # nlambda

  interp_betas   <- matrix(0, nrow(beta_m), length(s))
  interp_intcpts <- numeric(length(s))
  col_names      <- character(length(s))
  interpolated   <- logical(length(s))

  for (k in seq_along(s)) {
    sv <- s[k]

    if (sv >= lams[1]) {
      # beyond sparse end — clamp
      interp_betas[, k]   <- beta_m[, 1]
      interp_intcpts[k]   <- intcpt[1]
      col_names[k]        <- paste0("s=", sv)
    } else if (sv <= lams[nlam]) {
      # beyond dense end — clamp
      interp_betas[, k]   <- beta_m[, nlam]
      interp_intcpts[k]   <- intcpt[nlam]
      col_names[k]        <- paste0("s=", sv)
    } else {
      # find i_lo: largest index with lams[i_lo] >= sv (i.e. lambda value >= sv)
      i_lo <- max(which(lams >= sv))
      i_hi <- i_lo + 1L

      if (abs(lams[i_lo] - sv) < 1e-12) {
        # exact match
        interp_betas[, k]   <- beta_m[, i_lo]
        interp_intcpts[k]   <- intcpt[i_lo]
        col_names[k]        <- paste0("s=", sv)
      } else {
        # linear interpolation
        alpha <- (lams[i_lo] - sv) / (lams[i_lo] - lams[i_hi])
        interp_betas[, k]   <- (1 - alpha) * beta_m[, i_lo] + alpha * beta_m[, i_hi]
        interp_intcpts[k]   <- (1 - alpha) * intcpt[i_lo]   + alpha * intcpt[i_hi]
        col_names[k]        <- paste0("s=", sv)
        interpolated[k]     <- TRUE
      }
    }
  }

  if (any(interpolated)) {
    interp_vals <- s[interpolated]
    message(sprintf(
      "Note: %d value(s) of s (%s) not in the lambda path; predictions obtained by linear interpolation.",
      sum(interpolated),
      paste(signif(interp_vals, 4), collapse = ", ")
    ))
  }

  list(beta_mat = interp_betas, intercept_vec = interp_intcpts,
       col_names = col_names)
}


.picasso_predict <- function(object, newdata, lambda.idx = NULL, response.idx,
                             default_response_idx, transform = identity,
                             type = "response", s = NULL) {
  # --- s= path (lambda values, with interpolation) ---
  if (!is.null(s)) {
    if (type == "nonzero") {
      # nonzero: use nearest lambda for each s value (interpolation of support is undefined)
      return(lapply(s, function(sv) {
        idx <- which.min(abs(object$lambda - sv))
        which(abs(object$beta[, idx]) > 1e-8)
      }))
    }

    res    <- .picasso_resolve_s(object, s)
    bm     <- res$beta_mat
    iv     <- res$intercept_vec
    pred.n <- nrow(newdata)
    ns     <- length(s)

    if (type == "link") transform <- identity
    intcpt_mat <- matrix(rep(iv, each = pred.n), nrow = pred.n)
    pred <- as.matrix(transform(newdata %*% bm + intcpt_mat))
    colnames(pred) <- res$col_names

    is_default_idx <- length(response.idx) == length(default_response_idx) &&
      isTRUE(all(response.idx == default_response_idx))
    if (!is_default_idx) {
      response.idx <- as.integer(response.idx)
      .picasso_validate_indices(response.idx, pred.n, "response.idx")
      pred <- pred[response.idx, , drop = FALSE]
    }
    return(pred)
  }

  # --- lambda.idx= path (integer indices, original behaviour) ---
  if (type == "nonzero") {
    lambda.idx <- as.integer(lambda.idx)
    .picasso_validate_indices(lambda.idx, object$nlambda, "lambda.idx")
    return(lapply(lambda.idx, function(i) which(abs(object$beta[, i]) > 1e-8)))
  }

  lambda.idx <- as.integer(lambda.idx)
  .picasso_validate_indices(lambda.idx, object$nlambda, "lambda.idx")

  if (type == "link") transform <- identity

  pred.n   <- nrow(newdata)
  lambda.n <- length(lambda.idx)

  intcpt <- matrix(
    rep(object$intercept[lambda.idx], pred.n),
    nrow = pred.n,
    ncol = lambda.n,
    byrow = TRUE
  )

  linear <- newdata %*% object$beta[, lambda.idx] + intcpt
  pred   <- as.matrix(transform(linear))

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
