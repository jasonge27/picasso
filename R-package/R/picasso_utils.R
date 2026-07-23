.PICASSO_HIGH_PRECISION <- 1e-7
.PICASSO_FAST_PRECISION <- 1e-4
.PICASSO_FAST_POISSON_PRECISION <- 4e-4


.picasso_validate_choice <- function(value, choices, name) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !(value %in% choices)) {
    stop(sprintf(
      "%s must be one of: %s.", name, paste(choices, collapse = ", ")
    ), call. = FALSE)
  }
  value
}


.picasso_validate_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop(sprintf("%s must be TRUE or FALSE.", name), call. = FALSE)
  }
  value
}


.picasso_validate_positive_integer <- function(value, name,
                                                allow.null = FALSE) {
  if (is.null(value) && allow.null) return(NULL)
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value <= 0 || value != floor(value) ||
      value > .Machine$integer.max) {
    stop(sprintf("%s must be a positive finite integer.", name),
         call. = FALSE)
  }
  as.integer(value)
}


.picasso_validate_nonnegative_integer <- function(value, name,
                                                   allow.null = FALSE) {
  if (is.null(value) && allow.null) return(NULL)
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value < 0 || value != floor(value) ||
      value > .Machine$integer.max) {
    stop(sprintf("%s must be a nonnegative finite integer.", name),
         call. = FALSE)
  }
  as.integer(value)
}


.picasso_fast_precision <- function(family) {
  # Gaussian and glmnet both stop its coordinate descent by a scaled
  # objective-change criterion, for which 1e-7 is already the aligned preset.
  # The Newton/IRLS families use an approximately absolute KKT tolerance.
  # Poisson needs 4e-4 to match glmnet's observed external KKT accuracy;
  # binomial, multinomial, and square-root loss retain 1e-4.
  if (identical(family, "gaussian")) {
    .PICASSO_HIGH_PRECISION
  } else if (identical(family, "poisson")) {
    .PICASSO_FAST_POISSON_PRECISION
  } else {
    .PICASSO_FAST_PRECISION
  }
}


.picasso_resolve_precision <- function(prec, fast.mode, family) {
  fast.mode <- .picasso_validate_flag(fast.mode, "fast.mode")
  if (!is.numeric(prec) || length(prec) != 1L || is.na(prec) ||
      !is.finite(prec) || prec <= 0) {
    stop("prec must be a positive finite numeric scalar.")
  }
  prec <- as.double(prec)
  if (!fast.mode) return(prec)

  fast.precision <- .picasso_fast_precision(family)
  presets <- unique(c(.PICASSO_HIGH_PRECISION, fast.precision))
  matches.preset <- any(abs(prec - presets) <= 1e-7 * presets)
  if (!matches.preset) {
    stop(paste0(
      "fast.mode = TRUE fixes prec at ", format(fast.precision),
      " for family = \"", family, "\"; remove the custom prec value or ",
      "set fast.mode = FALSE."
    ))
  }
  fast.precision
}


.picasso_method_flag <- function(method, gamma) {
  method <- .picasso_validate_choice(
    method, c("l1", "mcp", "scad"), "method"
  )
  if (!is.numeric(gamma) || length(gamma) != 1L || is.na(gamma) ||
      !is.finite(gamma))
    stop("gamma must be a single finite numeric value.")
  if (method == "l1") {
    return(list(flag = 1L, gamma = gamma))
  }

  if (method == "mcp") {
    if (gamma <= 1)
      stop("gamma must be greater than 1 for MCP.")
    return(list(flag = 2L, gamma = gamma))
  }

  if (method == "scad") {
    if (gamma <= 2)
      stop("gamma must be greater than 2 for SCAD.")
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
    xx = out$xx,
    xm = out$xm,
    xinvc.vec = out$xinvc
  )
}


.picasso_validate_design <- function(X) {
  if (!is.numeric(X) || length(dim(X)) != 2L) {
    stop("X must be a numeric matrix.")
  }
  n <- nrow(X)
  d <- ncol(X)
  if (n == 0L || d == 0L) {
    stop("No data input.")
  }
  if (anyNA(X) || any(!is.finite(X))) {
    stop("X must contain only finite values.")
  }
  list(n = n, d = d)
}


.picasso_prepare_design <- function(X, standardize, center = TRUE) {
  X <- as.matrix(X)
  # storage.mode<- duplicates shared R objects even when their type already
  # is double. Preserve a read-only double matrix and coerce only integer
  # inputs, which avoids an n-by-d copy on the common path.
  if (!is.double(X))
    storage.mode(X) <- "double"

  if (standardize && center) {
    std <- .picasso_standardize(X)
    return(std)
  }

  if (standardize) {
    n <- nrow(X)
    d <- ncol(X)
    divisor <- max(n - 1L, 1L)
    xinvc.vec <- numeric(d)
    for (j in seq_len(d)) {
      column <- X[, j]
      maximum <- max(abs(column))
      if (maximum > 0) {
        relative.norm <- sqrt(sum((column / maximum)^2) / divisor)
        xinvc.vec[j] <- (1 / maximum) / relative.norm
      }
    }
    return(list(
      xx = sweep(X, 2L, xinvc.vec, `*`),
      xm = matrix(0, nrow = 1L, ncol = d),
      xinvc.vec = xinvc.vec
    ))
  }

  list(
    xx = X,
    xm = matrix(0, nrow = 1, ncol = ncol(X)),
    xinvc.vec = rep(1, ncol(X))
  )
}


.picasso_lambda_path <- function(lambda, nlambda, lambda.min.ratio, lambda.max) {
  if (!is.null(lambda)) {
    if (!is.numeric(lambda) || length(dim(lambda)) != 0L ||
        length(lambda) == 0L || anyNA(lambda) || any(!is.finite(lambda)) ||
        any(lambda < 0)) {
      stop("lambda must be a nonempty vector of finite nonnegative values.")
    }
    if (length(lambda) > 1L && any(diff(lambda) >= 0)) {
      stop("lambda values must be strictly decreasing.")
    }
    lambda <- as.double(lambda)
    return(list(lambda = lambda, nlambda = length(lambda)))
  }

  nlambda <- .picasso_validate_positive_integer(
    nlambda, "nlambda", allow.null = TRUE
  )
  if (!is.null(lambda.min.ratio)) {
    if (!is.numeric(lambda.min.ratio) || length(lambda.min.ratio) != 1L ||
        is.na(lambda.min.ratio) || !is.finite(lambda.min.ratio) ||
        lambda.min.ratio <= 0 || lambda.min.ratio >= 1) {
      stop(
        "lambda.min.ratio must be a finite numeric scalar strictly between 0 and 1.",
        call. = FALSE
      )
    }
    lambda.min.ratio <- as.double(lambda.min.ratio)
  }

  if (is.null(nlambda)) {
    nlambda <- 100L
  }

  if (length(lambda.max) != 1L || !is.numeric(lambda.max) ||
      is.na(lambda.max) || !is.finite(lambda.max) || lambda.max < 0)
    stop("lambda.max must be a finite nonnegative scalar.")
  if (lambda.max == 0) {
    path <- if (nlambda == 1L) 0 else
      seq(.Machine$double.eps, 0, length.out = nlambda)
    return(list(lambda = path, nlambda = nlambda))
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


.picasso_indices <- function(index, n, name, default.length = NULL) {
  if (is.null(index) && !is.null(default.length)) {
    return(seq_len(min(as.integer(default.length), n)))
  }
  if (length(index) == 0L) {
    stop(sprintf("`%s` must contain at least one index.", name))
  }
  if ((!is.numeric(index) && !is.integer(index)) || is.factor(index) ||
      anyNA(index) || any(!is.finite(index)) ||
      any(index != floor(index))) {
    stop(sprintf("`%s` must contain finite integer indices.", name))
  }
  if (any(index < 1) || any(index > n) ||
      any(index > .Machine$integer.max)) {
    stop(sprintf("`%s` contains out-of-range indices. Valid range is 1..%d.", name, n))
  }
  as.integer(index)
}


.picasso_extract_coef <- function(object, lambda.idx, beta.idx) {
  lambda.idx <- .picasso_indices(
    lambda.idx, object$nlambda, "lambda.idx", default.length = 3L
  )
  beta.idx <- .picasso_indices(
    beta.idx, nrow(object$beta), "beta.idx", default.length = 3L
  )

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


# Add this response-only constant to mean(mu - y * eta), then multiply by two,
# to recover the conventional Poisson deviance used by the public R API.
.picasso_poisson_saturated_constant <- function(y) {
  y <- as.numeric(y)
  mean(ifelse(y > 0, y * log(y) - y, 0.0))
}


.picasso_poisson_mean <- function(eta) {
  mu <- suppressWarnings(exp(eta))
  if (any(!is.finite(mu)))
    stop("Poisson linear predictor is too large for a finite response mean.")
  mu
}


.picasso_binomial_nll_from_eta <- function(y, eta) {
  if (!is.numeric(y) || is.null(y) || anyNA(y) || any(!is.finite(y)) ||
      any(!(y %in% c(0, 1)))) {
    stop("Binomial response must contain finite zero/one values.")
  }
  if (!is.numeric(eta) || !(is.null(dim(eta)) || length(dim(eta)) == 2L))
    stop("Binomial linear predictor must be a numeric vector or matrix.")

  eta.matrix <- if (is.null(dim(eta))) {
    matrix(as.double(eta), ncol = 1L)
  } else {
    matrix(as.double(eta), nrow = nrow(eta), ncol = ncol(eta))
  }
  if (nrow(eta.matrix) != length(y) || ncol(eta.matrix) == 0L ||
      any(!is.finite(eta.matrix))) {
    stop(paste0(
      "Binomial response and linear predictor must be finite and have ",
      "matching rows."
    ))
  }

  # For y=0 this is softplus(eta); for y=1 it is softplus(-eta).
  # Writing the two cases this way avoids both tail clipping and cancellation.
  signed.eta <- eta.matrix * (1 - 2 * as.double(y))
  loss <- pmax(signed.eta, 0) + log1p(exp(-abs(signed.eta)))
  value <- colMeans(loss)
  if (any(!is.finite(value)))
    stop("Binomial negative log-likelihood is not finite.")
  value
}


.picasso_poisson_deviance_from_eta <- function(y, eta, mu = NULL) {
  if (!is.numeric(y) || is.null(y) || anyNA(y) || any(!is.finite(y)) ||
      any(y < 0)) {
    stop("Poisson response must contain finite nonnegative values.")
  }
  if (!is.numeric(eta) || !(is.null(dim(eta)) || length(dim(eta)) == 2L))
    stop("Poisson linear predictor must be a numeric vector or matrix.")

  eta.matrix <- if (is.null(dim(eta))) {
    matrix(as.double(eta), ncol = 1L)
  } else {
    matrix(as.double(eta), nrow = nrow(eta), ncol = ncol(eta))
  }
  if (nrow(eta.matrix) != length(y) || ncol(eta.matrix) == 0L ||
      any(!is.finite(eta.matrix))) {
    stop(paste0(
      "Poisson response and linear predictor must be finite and have ",
      "matching rows."
    ))
  }

  fitted.mean <- if (is.null(mu)) {
    .picasso_poisson_mean(eta.matrix)
  } else {
    if (!is.numeric(mu) || !(is.null(dim(mu)) || length(dim(mu)) == 2L))
      stop("Poisson fitted mean must be a numeric vector or matrix.")
    mu.matrix <- if (is.null(dim(mu))) {
      matrix(as.double(mu), ncol = 1L)
    } else {
      matrix(as.double(mu), nrow = nrow(mu), ncol = ncol(mu))
    }
    if (!identical(dim(mu.matrix), dim(eta.matrix)) || anyNA(mu.matrix) ||
        any(!is.finite(mu.matrix)) || any(mu.matrix < 0)) {
      stop(paste0(
        "Poisson fitted mean must be finite, nonnegative, and match the ",
        "linear predictor shape."
      ))
    }
    mu.matrix
  }

  terms <- fitted.mean
  positive <- y > 0
  if (any(positive)) {
    response <- as.double(y[positive])
    terms[positive, ] <-
      response * (log(response) - eta.matrix[positive, , drop = FALSE]) -
      response + fitted.mean[positive, , drop = FALSE]
  }
  value <- 2 * colMeans(terms)
  if (any(!is.finite(value)))
    stop("Poisson deviance is not finite.")
  pmax(value, 0)
}


.picasso_multinomial_nll_from_logits <- function(response, logits) {
  if (!is.numeric(logits) || length(dim(logits)) != 2L ||
      nrow(logits) == 0L || ncol(logits) == 0L || anyNA(logits) ||
      any(!is.finite(logits))) {
    stop("Multinomial logits must be a nonempty finite numeric matrix.")
  }
  if (!is.numeric(response) || length(response) != nrow(logits) ||
      anyNA(response) || any(!is.finite(response)) ||
      any(response != floor(response)) ||
      any(response < 1L | response > ncol(logits))) {
    stop(paste0(
      "Multinomial response must contain valid one-based integer class codes."
    ))
  }

  response <- as.integer(response)
  row.maximum <- apply(logits, 1L, max)
  exponential.sum <- rowSums(exp(logits - row.maximum))
  true.logit <- logits[cbind(seq_len(nrow(logits)), response)]
  # Keep the common row shift out of a later subtraction. This matches the
  # native objective and remains accurate even when the shift is very large.
  row.loss <- log(exponential.sum) + (row.maximum - true.logit)
  value <- mean(row.loss)
  if (!is.finite(value))
    stop("Multinomial negative log-likelihood is not finite.")
  value
}


.picasso_validate_offset <- function(offset, n, family) {
  if (is.null(offset)) {
    return(rep(0.0, n))
  }
  if (!is.numeric(offset) || length(dim(offset)) != 0L ||
      length(offset) != n || anyNA(offset) || any(!is.finite(offset))) {
    stop(sprintf(
      "offset for %s regression must be a finite numeric vector of length %d.",
      family, n
    ))
  }
  as.double(offset)
}


.picasso_prediction_offset <- function(object, newoffset, n) {
  if (is.null(newoffset)) {
    if (isTRUE(object$offset.used)) {
      stop(
        "newoffset must be provided when predicting from a model fitted with offset.",
        call. = FALSE
      )
    }
    return(NULL)
  }
  if (!is.numeric(newoffset) || length(dim(newoffset)) != 0L ||
      length(newoffset) != n || anyNA(newoffset) ||
      any(!is.finite(newoffset))) {
    stop(sprintf(
      "newoffset must be a finite numeric vector of length %d.", n
    ), call. = FALSE)
  }
  as.double(newoffset)
}


.picasso_binomial_response_codes <- function(y, levels, n = NULL,
                                              name = "response") {
  if (!is.null(n) && length(y) != n)
    stop(sprintf("`%s` must have length %d.", name, n))
  if (length(y) == 0L || anyNA(y))
    stop(sprintf("`%s` must be nonempty and contain no missing values.", name))
  numeric.codes <- is.numeric(y) && all(is.finite(y)) &&
    all(y %in% c(0, 1))
  if (is.null(levels)) {
    if (numeric.codes) return(as.numeric(y))
    stop(paste0(
      "This legacy binomial fit has no stored class map; `", name,
      "` must use numeric 0/1 labels."
    ))
  }
  if (length(levels) != 2L)
    stop("The fitted binomial model does not contain a two-level class map.")

  labels <- as.character(y)
  matched <- match(labels, levels)
  if (anyNA(matched) && numeric.codes) {
    # Preserve the long-standing public convention that numeric zero/one is an
    # accepted encoded response even when the fitted labels are, e.g., no/yes.
    return(as.numeric(y))
  }
  if (anyNA(matched)) {
    unknown <- unique(labels[is.na(matched)])
    stop(sprintf(
      "`%s` contains class values absent from the fitted model: %s.",
      name, paste(unknown, collapse = ", ")
    ))
  }
  as.numeric(matched - 1L)
}


.picasso_null_eta <- function(Y, family, offset = NULL, intercept = TRUE) {
  Y <- as.numeric(Y)
  n <- length(Y)
  off <- if (is.null(offset)) rep(0.0, n) else as.double(offset)

  if (family %in% c("gaussian", "sqrtlasso")) {
    return(rep(if (intercept) mean(Y) else 0.0, n))
  }

  if (family == "binomial") {
    if (!intercept) {
      return(off)
    }
    target <- sum(Y)
    lower <- -40 - max(off)
    upper <- 40 - min(off)
    shift <- stats::uniroot(
      function(value) sum(stats::plogis(off + value)) - target,
      lower = lower, upper = upper, tol = .Machine$double.eps^0.5
    )$root
    return(off + shift)
  }

  if (family == "poisson") {
    if (!intercept) {
      return(off)
    }
    maximum <- max(off)
    log.mean.exp <- maximum + log(mean(exp(off - maximum)))
    shift <- log(mean(Y)) - log.mean.exp
    return(off + shift)
  }

  rep(NA_real_, n)
}


.picasso_null_deviance <- function(Y, family, offset = NULL,
                                   intercept = TRUE) {
  Y <- as.numeric(Y)
  n <- length(Y)
  eta0 <- .picasso_null_eta(Y, family, offset, intercept)
  if (family %in% c("gaussian", "sqrtlasso")) {
    sum((Y - eta0)^2) / (2 * n)
  } else if (family == "binomial") {
    .picasso_binomial_nll_from_eta(Y, eta0)
  } else if (family == "poisson") {
    .picasso_poisson_deviance_from_eta(Y, eta0)
  } else {
    NA_real_
  }
}


.picasso_fit_deviance <- function(Y, X, beta_mat, intercept_vec, family,
                                  offset = NULL) {
  Y <- as.numeric(Y)
  n <- nrow(X)
  nlambda <- ncol(beta_mat)
  off <- if (is.null(offset)) rep(0.0, n) else as.double(offset)
  eta_mat <- X %*% beta_mat + matrix(rep(intercept_vec, each = n), nrow = n) +
             matrix(rep(off, nlambda), nrow = n)
  if (family %in% c("gaussian", "sqrtlasso")) {
    colSums((Y - eta_mat)^2) / (2 * n)
  } else if (family == "binomial") {
    .picasso_binomial_nll_from_eta(Y, eta_mat)
  } else if (family == "poisson") {
    .picasso_poisson_deviance_from_eta(Y, eta_mat)
  } else {
    rep(NA_real_, nlambda)
  }
}


.picasso_prediction_newdata <- function(object, newdata) {
  if (!is.numeric(newdata) || length(dim(newdata)) != 2L) {
    stop("`newdata` must be a numeric matrix.", call. = FALSE)
  }
  if (nrow(newdata) == 0L) {
    stop("`newdata` must contain at least one row.", call. = FALSE)
  }
  expected <- nrow(object$beta)
  if (ncol(newdata) != expected) {
    stop(sprintf(
      "`newdata` has %d columns; the fitted model expects %d.",
      ncol(newdata), expected
    ), call. = FALSE)
  }
  if (anyNA(newdata) || any(!is.finite(newdata))) {
    stop("`newdata` must contain only finite values.", call. = FALSE)
  }
  newdata <- as.matrix(newdata)
  if (!is.double(newdata))
    storage.mode(newdata) <- "double"
  newdata
}


.picasso_validate_s <- function(s) {
  if (!is.numeric(s) || length(dim(s)) != 0L || length(s) == 0L ||
      anyNA(s) || any(!is.finite(s)) || any(s < 0)) {
    stop("`s` must contain finite nonnegative lambda values.", call. = FALSE)
  }
  as.double(s)
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
                             transform = identity, type = "response", s = NULL,
                             newoffset = NULL) {
  type <- .picasso_validate_choice(
    type, c("response", "link", "nonzero"), "type"
  )

  # --- s= path (lambda values, with interpolation) ---
  if (!is.null(s)) {
    s <- .picasso_validate_s(s)
    if (type == "nonzero") {
      # nonzero: use nearest lambda for each s value (interpolation of support is undefined)
      return(lapply(s, function(sv) {
        idx <- which.min(abs(object$lambda - sv))
        which(abs(object$beta[, idx]) > 1e-8)
      }))
    }

    newdata <- .picasso_prediction_newdata(object, newdata)
    res    <- .picasso_resolve_s(object, s)
    bm     <- res$beta_mat
    iv     <- res$intercept_vec
    pred.n <- nrow(newdata)
    offset <- .picasso_prediction_offset(object, newoffset, pred.n)

    if (type == "link") transform <- identity
    intcpt_mat <- matrix(rep(iv, each = pred.n), nrow = pred.n)
    linear <- newdata %*% bm + intcpt_mat
    if (!is.null(offset)) linear <- linear + offset
    pred <- as.matrix(transform(linear))
    colnames(pred) <- res$col_names

    if (!is.null(response.idx)) {
      response.idx <- .picasso_indices(
        response.idx, pred.n, "response.idx"
      )
      pred <- pred[response.idx, , drop = FALSE]
    }
    return(pred)
  }

  # --- lambda.idx= path (integer indices, original behaviour) ---
  lambda.idx <- .picasso_indices(
    lambda.idx, object$nlambda, "lambda.idx", default.length = 3L
  )
  if (type == "nonzero") {
    return(lapply(lambda.idx, function(i) which(abs(object$beta[, i]) > 1e-8)))
  }

  newdata <- .picasso_prediction_newdata(object, newdata)
  if (type == "link") transform <- identity

  pred.n   <- nrow(newdata)
  lambda.n <- length(lambda.idx)
  offset   <- .picasso_prediction_offset(object, newoffset, pred.n)

  intcpt <- matrix(
    rep(object$intercept[lambda.idx], pred.n),
    nrow = pred.n,
    ncol = lambda.n,
    byrow = TRUE
  )

  linear <- newdata %*% object$beta[, lambda.idx] + intcpt
  if (!is.null(offset)) linear <- linear + offset
  # Matrix-backed coefficient paths can yield a dense Matrix object. Base
  # transforms such as stats::plogis() require an ordinary numeric matrix.
  pred   <- as.matrix(transform(as.matrix(linear)))

  if (!is.null(response.idx)) {
    response.idx <- .picasso_indices(
      response.idx, pred.n, "response.idx"
    )
    pred <- pred[response.idx, , drop = FALSE]
  }

  colnames(pred) <- paste0("lambda[", lambda.idx, "]")
  pred
}
