cv.picasso <- function(X, Y, ..., nfolds = 10, foldid = NULL,
                       type.measure = "default") {
  n <- nrow(X)
  if (is.null(foldid)) {
    foldid <- sample(rep(seq_len(nfolds), length.out = n))
  } else {
    nfolds <- length(unique(foldid))
  }

  # Fit on full data to get lambda path
  fit_full <- picasso(X, Y, ...)
  lambda   <- fit_full$lambda
  nlambda  <- fit_full$nlambda
  family   <- fit_full$family

  if (is.null(family)) family <- "gaussian"

  # type.measure defaults
  if (type.measure == "default") {
    type.measure <- if (family == "binomial") "class" else "deviance"
  }

  # Cross-validation loop
  cv_mat <- matrix(NA_real_, nrow = nfolds, ncol = nlambda)

  for (fold in seq_len(nfolds)) {
    test_idx  <- which(foldid == fold)
    train_idx <- which(foldid != fold)

    X_train <- X[train_idx, , drop = FALSE]
    Y_train <- if (is.matrix(Y)) Y[train_idx, , drop = FALSE] else Y[train_idx]
    X_test  <- X[test_idx,  , drop = FALSE]
    Y_test  <- if (is.matrix(Y)) Y[test_idx,  , drop = FALSE] else Y[test_idx]

    # Fit on training fold with same lambda path
    fit_fold <- tryCatch(
      picasso(X_train, Y_train, lambda = lambda, ...),
      error = function(e) NULL
    )
    if (is.null(fit_fold)) next

    n_fit   <- fit_fold$nlambda
    Y_test_vec <- as.numeric(Y_test)
    n_test     <- length(Y_test_vec)

    beta_sub   <- as.matrix(fit_fold$beta)[, seq_len(n_fit), drop = FALSE]
    intcpt_sub <- as.numeric(fit_fold$intercept)[seq_len(n_fit)]
    eta_mat    <- X_test %*% beta_sub +
                  matrix(rep(intcpt_sub, each = n_test), nrow = n_test)

    for (k in seq_len(n_fit)) {
      eta <- as.numeric(eta_mat[, k])
      cv_mat[fold, k] <- switch(
        type.measure,
        "deviance" = {
          if (family %in% c("gaussian", "sqrtlasso"))
            mean((Y_test_vec - eta)^2) / 2
          else if (family == "binomial") {
            p <- pmax(pmin(1 / (1 + exp(-eta)), 1 - 1e-8), 1e-8)
            -mean(Y_test_vec * log(p) + (1 - Y_test_vec) * log(1 - p))
          } else if (family == "poisson")
            .picasso_poisson_dev(Y_test_vec, exp(eta))
          else NA_real_
        },
        "mse"  = mean((Y_test_vec - eta)^2),
        "mae"  = mean(abs(Y_test_vec - eta)),
        "class" = {
          if (family == "binomial")
            mean(as.integer(1 / (1 + exp(-eta)) > 0.5) != Y_test_vec)
          else NA_real_
        },
        NA_real_
      )
    }
  }

  cvm  <- colMeans(cv_mat, na.rm = TRUE)
  cvsd <- apply(cv_mat, 2, sd, na.rm = TRUE)
  cvup <- cvm + cvsd
  cvlo <- cvm - cvsd

  # lambda.min: lambda with minimum CV error
  min_idx      <- which.min(cvm)
  lambda.min   <- lambda[min_idx]

  # lambda.1se: largest lambda within 1 SE of minimum
  cutoff       <- cvm[min_idx] + cvsd[min_idx]
  lambda.1se   <- max(lambda[cvm <= cutoff])

  nzero <- as.integer(colSums(fit_full$beta != 0))

  result <- list(
    lambda      = lambda,
    cvm         = cvm,
    cvsd        = cvsd,
    cvup        = cvup,
    cvlo        = cvlo,
    nzero       = nzero,
    lambda.min  = lambda.min,
    lambda.1se  = lambda.1se,
    name        = type.measure,
    picasso.fit = fit_full
  )
  class(result) <- "cv.picasso"
  result
}


print.cv.picasso <- function(x, ...) {
  cat("Cross-validated picasso fit\n")
  cat(sprintf("  Measure:     %s\n", x$name))
  cat(sprintf("  lambda.min:  %.6g  (index %d)\n",
              x$lambda.min, which(x$lambda == x$lambda.min)))
  cat(sprintf("  lambda.1se:  %.6g  (index %d)\n",
              x$lambda.1se, which(x$lambda == x$lambda.1se)))
  cat(sprintf("  nlambda:     %d\n", length(x$lambda)))
  invisible(x)
}


plot.cv.picasso <- function(x, sign.lambda = 1, ...) {
  log_lambda <- sign.lambda * log(x$lambda)
  ylim_range <- range(c(x$cvup, x$cvlo), na.rm = TRUE)

  plot(log_lambda, x$cvm,
       type = "o", pch = 20,
       xlab = if (sign.lambda == 1) "log(lambda)" else "-log(lambda)",
       ylab = x$name,
       main = "Cross-Validation Error",
       ylim = ylim_range, ...)

  # Error bars
  for (i in seq_along(log_lambda)) {
    lines(c(log_lambda[i], log_lambda[i]),
          c(x$cvlo[i], x$cvup[i]),
          col = "grey60")
  }

  # Vertical lines for lambda.min and lambda.1se
  abline(v = sign.lambda * log(x$lambda.min), lty = 2, col = "red")
  abline(v = sign.lambda * log(x$lambda.1se), lty = 2, col = "blue")

  invisible(NULL)
}


coef.cv.picasso <- function(object, s = c("lambda.min", "lambda.1se"), ...) {
  s <- match.arg(s)
  lam <- object[[s]]
  idx <- which.min(abs(object$lambda - lam))
  coef(object$picasso.fit, lambda.idx = idx, ...)
}


predict.cv.picasso <- function(object, newdata,
                                s = c("lambda.min", "lambda.1se"),
                                type = "response", ...) {
  s <- match.arg(s)
  lam <- object[[s]]
  predict(object$picasso.fit, newdata, s = lam, type = type, ...)
}
