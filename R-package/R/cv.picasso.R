.picasso_cv_require_usable_status <- function(fit, context) {
  status.code <- fit$status.code
  if (is.null(status.code)) {
    return(invisible(NULL))
  }
  usable <- length(status.code) == 1L && !is.na(status.code) &&
    status.code %in% c(0L, 1L, 10L)
  if (!usable) {
    status.label <- if (is.null(fit$status) || length(fit$status) != 1L)
      "unknown" else as.character(fit$status)
    stop(sprintf(
      paste0("%s stopped with status '%s' (code %s); ",
             "cross-validation requires a usable fitted path."),
      context, status.label, paste(status.code, collapse = ", ")
    ), call. = FALSE)
  }
  invisible(NULL)
}


.PICASSO_CV_BLOCK_BYTES <- 8 * 1024^2


.picasso_cv_scalar_fold_loss <- function(
    X, Y, beta, intercept, n.fit, family, measure, offset = NULL,
    block.bytes = .PICASSO_CV_BLOCK_BYTES) {
  if (length(block.bytes) != 1L || !is.numeric(block.bytes) ||
      is.na(block.bytes) || !is.finite(block.bytes) || block.bytes <= 0) {
    stop("block.bytes must be one finite positive number.")
  }
  n <- nrow(X)
  d <- ncol(X)
  if (length(n.fit) != 1L || is.na(n.fit) || n.fit < 1L ||
      n.fit != floor(n.fit) || ncol(beta) < n.fit ||
      length(intercept) < n.fit) {
    stop("Fitted scalar path has inconsistent coefficient dimensions.")
  }
  n.fit <- as.integer(n.fit)

  evaluate.block <- function(eta.mat) {
    result <- numeric(ncol(eta.mat))
    for (k in seq_len(ncol(eta.mat))) {
      eta <- as.numeric(eta.mat[, k])
      response.fit <- if (family == "binomial") {
        stats::plogis(eta)
      } else if (family == "poisson") {
        .picasso_poisson_mean(eta)
      } else {
        eta
      }
      result[k] <- switch(
        measure,
        "deviance" = {
          if (family %in% c("gaussian", "sqrtlasso")) {
            mean((Y - eta)^2) / 2
          } else if (family == "binomial") {
            .picasso_binomial_nll_from_eta(Y, eta)
          } else if (family == "poisson") {
            .picasso_poisson_deviance_from_eta(
              Y, eta, mu = response.fit
            )
          } else {
            NA_real_
          }
        },
        "mse" = mean((Y - response.fit)^2),
        "mae" = mean(abs(Y - response.fit)),
        "class" = {
          if (family == "binomial") {
            mean(as.integer(eta > 0) != Y)
          } else {
            NA_real_
          }
        },
        NA_real_
      )
    }
    result
  }

  # Account for both the dense coefficient slice and its predictor. Small
  # folds retain the established single-GEMM path and reduction order.
  full.working.bytes <- 8 * (as.double(n) + d) * n.fit
  if (full.working.bytes <= block.bytes) {
    beta.sub <- as.matrix(beta)[, seq_len(n.fit), drop = FALSE]
    intercept.sub <- as.numeric(intercept)[seq_len(n.fit)]
    eta.mat <- X %*% beta.sub +
      matrix(rep(intercept.sub, each = n), nrow = n)
    if (family %in% c("binomial", "poisson") && !is.null(offset)) {
      eta.mat <- eta.mat + matrix(rep(offset, n.fit), nrow = n)
    }
    return(evaluate.block(eta.mat))
  }

  block.columns <- as.integer(max(1, min(
    n.fit,
    floor(as.double(block.bytes) / (8 * (max(n, 1L) + max(d, 1L))))
  )))
  result <- numeric(n.fit)
  for (block.start in seq.int(1L, n.fit, by = block.columns)) {
    block.stop <- min(n.fit, block.start + block.columns - 1L)
    indices <- seq.int(block.start, block.stop)
    beta.block <- as.matrix(beta[, indices, drop = FALSE])
    intercept.block <- as.numeric(intercept)[indices]
    eta.mat <- X %*% beta.block +
      matrix(rep(intercept.block, each = n), nrow = n)
    if (family %in% c("binomial", "poisson") && !is.null(offset)) {
      eta.mat <- eta.mat + matrix(
        rep(offset, length(indices)), nrow = n
      )
    }
    result[indices] <- evaluate.block(eta.mat)
  }
  result
}


cv.picasso <- function(X, Y, ..., nfolds = 10, foldid = NULL,
                       type.measure = "default", fast.mode = FALSE) {
  if (!is.numeric(X) || length(dim(X)) != 2L || nrow(X) == 0L || ncol(X) == 0L)
    stop("X must be a nonempty numeric matrix.")
  n <- nrow(X)
  response.length <- if (is.matrix(Y)) nrow(Y) else length(Y)
  if (response.length != n)
    stop(sprintf("Y must contain %d observations.", n))

  dots <- list(...)
  # Use the same precision preset for the full fit and every fold fit.
  dots$fast.mode <- fast.mode

  # Fit on full data to establish the common lambda path and global class map.
  full.args <- c(list(X = X, Y = Y), dots)
  fit_full <- do.call(picasso, full.args)
  lambda <- fit_full$lambda
  nlambda <- fit_full$nlambda
  family <- fit_full$family
  if (is.null(family)) family <- "gaussian"
  # Freeze the full-data Gaussian auto decision for every fold. This keeps the
  # numerical backend stable even though training folds have fewer rows.
  if (family == "gaussian") {
    dots$type.gaussian <- fit_full$type.gaussian
  }
  .picasso_cv_require_usable_status(
    fit_full, sprintf("Full-data %s fit", family)
  )

  if (length(type.measure) != 1L || !is.character(type.measure) ||
      is.na(type.measure))
    stop("type.measure must be a single character value.")
  if (type.measure == "default") {
    type.measure <- if (family %in% c("binomial", "multinomial"))
      "class" else "deviance"
  }
  type.measure <- match.arg(
    type.measure, c("deviance", "mse", "mae", "class")
  )
  if (family == "multinomial" &&
      !(type.measure %in% c("deviance", "class")))
    stop("Multinomial cross-validation supports deviance or class loss.")
  if (type.measure == "class" &&
      !(family %in% c("binomial", "multinomial")))
    stop("Class loss is available only for binomial or multinomial models.")

  global.levels <- NULL
  global.response <- NULL
  if (family == "binomial") {
    global.levels <- fit_full$levels
    global.response <- factor(as.character(Y), levels = global.levels)
    if (anyNA(global.response))
      stop("Y contains values outside the full-data binomial class map.")
  } else if (family == "multinomial") {
    global.levels <- fit_full$levels
    global.response <- factor(as.character(Y), levels = global.levels)
    if (anyNA(global.response))
      stop("Y contains values outside the full-data multinomial class map.")
  }

  if (is.null(foldid)) {
    if (length(nfolds) != 1L || !is.numeric(nfolds) || is.na(nfolds) ||
        !is.finite(nfolds) || nfolds != floor(nfolds) ||
        nfolds < 2L || nfolds > n)
      stop(sprintf("nfolds must be an integer between 2 and %d.", n))
    nfolds <- as.integer(nfolds)
    if (family %in% c("binomial", "multinomial")) {
      class.counts <- table(global.response)
      if (any(class.counts < 2L)) {
        limiting <- names(class.counts)[class.counts < 2L]
        stop(sprintf(
          paste0("Stratified categorical CV requires at least two ",
                 "observations per class; too few in: %s."),
          paste(limiting, collapse = ", ")
        ))
      }
      foldid <- integer(n)
      next.fold <- sample.int(nfolds, 1L)
      for (level in global.levels) {
        index <- sample(which(global.response == level))
        labels <- ((seq_along(index) + next.fold - 2L) %% nfolds) + 1L
        foldid[index] <- labels
        next.fold <- ((next.fold + length(index) - 1L) %% nfolds) + 1L
      }
    } else {
      foldid <- sample(rep(seq_len(nfolds), length.out = n))
    }
  } else {
    if ((!is.numeric(foldid) && !is.integer(foldid)) || is.factor(foldid) ||
        length(foldid) != n || anyNA(foldid) || any(!is.finite(foldid)) ||
        any(foldid != floor(foldid)) || any(foldid < 1L))
      stop("foldid must be a length-n vector of positive integer fold labels.")
    foldid <- as.integer(foldid)
    observed.folds <- sort(unique(foldid))
    if (!identical(observed.folds, seq_len(max(observed.folds))))
      stop("foldid labels must be consecutive integers starting at 1.")
    nfolds <- length(observed.folds)
    if (nfolds < 2L)
      stop("foldid must define at least two folds.")
  }

  # Cross-validation loop
  cv_mat <- matrix(NA_real_, nrow = nfolds, ncol = nlambda)
  common.nlambda <- nlambda

  for (fold in seq_len(nfolds)) {
    test_idx  <- which(foldid == fold)
    train_idx <- which(foldid != fold)

    X_train <- X[train_idx, , drop = FALSE]
    Y_train <- if (is.matrix(Y)) Y[train_idx, , drop = FALSE] else Y[train_idx]
    X_test  <- X[test_idx,  , drop = FALSE]
    Y_test  <- if (is.matrix(Y)) Y[test_idx,  , drop = FALSE] else Y[test_idx]

    if (family %in% c("binomial", "multinomial")) {
      missing.classes <- setdiff(
        global.levels, unique(as.character(global.response[train_idx]))
      )
      if (length(missing.classes) > 0L)
        stop(sprintf(
          "Training fold %d is missing %s class(es): %s.",
          fold, family, paste(missing.classes, collapse = ", ")
        ))
    }

    if (family == "binomial") {
      Y_train <- factor(
        as.character(global.response[train_idx]), levels = global.levels
      )
      Y_test <- factor(
        as.character(global.response[test_idx]), levels = global.levels
      )
    } else if (family == "multinomial") {
      Y_train <- factor(
        as.character(global.response[train_idx]), levels = global.levels
      )
      Y_test <- factor(
        as.character(global.response[test_idx]), levels = global.levels
      )
    }

    fold.args <- dots
    fold.args$X <- X_train
    fold.args$Y <- Y_train
    fold.args$lambda <- lambda
    fold.args$nlambda <- NULL
    fold.args$lambda.min.ratio <- NULL
    if (!is.null(fold.args$offset)) {
      if (length(fold.args$offset) != n)
        stop(sprintf("offset must have length %d for cross-validation.", n))
      fold.args$offset <- fold.args$offset[train_idx]
    }

    fit_fold <- tryCatch(
      do.call(picasso, fold.args),
      error = function(e) {
        stop(sprintf(
          "%s fit failed in fold %d: %s",
          family, fold, conditionMessage(e)
        ), call. = FALSE)
      }
    )
    .picasso_cv_require_usable_status(
      fit_fold, sprintf("%s fit in fold %d", family, fold)
    )

    n_fit   <- fit_fold$nlambda
    if (n_fit < 1L || n_fit > nlambda)
      stop(sprintf("Fold %d returned invalid nlambda=%s.", fold, n_fit))
    if (n_fit != nlambda) {
      if (!isTRUE(fit_fold$path.early.stopped))
        stop(sprintf(
          paste0("%s fold %d covered only %d/%d lambdas; ",
                 "cross-validation requires a usable common path."),
          family, fold, n_fit, nlambda
        ))
      common.nlambda <- min(common.nlambda, n_fit)
    }

    if (family == "multinomial") {
      if (!identical(fit_fold$levels, global.levels))
        stop(sprintf("Multinomial class mapping changed in fold %d.", fold))
      response <- .picasso_multinomial_response_codes(
        Y_test, global.levels, nrow(X_test), "fold response"
      )
      for (k in seq_len(n_fit)) {
        logits <- .picasso_multinomial_logits_one(fit_fold, X_test, k)
        cv_mat[fold, k] <- if (type.measure == "deviance") {
          .picasso_multinomial_nll_from_logits(response, logits)
        } else {
          mean(max.col(logits, ties.method = "first") != response)
        }
      }
      next
    }

    Y_test_vec <- if (family == "binomial") {
      as.numeric(Y_test) - 1.0
    } else {
      as.numeric(Y_test)
    }
    test.offset <- if (family %in% c("binomial", "poisson") &&
                       !is.null(dots$offset)) {
      dots$offset[test_idx]
    } else {
      NULL
    }
    cv_mat[fold, seq_len(n_fit)] <- .picasso_cv_scalar_fold_loss(
      X_test, Y_test_vec, fit_fold$beta, fit_fold$intercept, n_fit,
      family, type.measure, test.offset
    )
  }

  # glmnet-style deviance stopping may truncate otherwise successful folds at
  # different points.  Because every truncation is a lambda-path prefix, the
  # shortest successful prefix is the exact common path across all folds.
  if (common.nlambda < nlambda) {
    cv_mat <- cv_mat[, seq_len(common.nlambda), drop = FALSE]
    lambda <- lambda[seq_len(common.nlambda)]
    nlambda <- common.nlambda
  }

  valid.fold.count <- colSums(is.finite(cv_mat))
  if (any(valid.fold.count != nfolds)) {
    first.incomplete <- which(valid.fold.count != nfolds)[1L]
    stop(sprintf(
      paste0("Cross-validation covered only %d/%d folds at lambda index %d; ",
             "increase dfmax or use a shorter lambda path."),
      valid.fold.count[first.incomplete], nfolds, first.incomplete
    ))
  }
  cvm  <- colMeans(cv_mat, na.rm = TRUE)
  cvsd <- apply(cv_mat, 2, sd, na.rm = TRUE) / sqrt(valid.fold.count)
  cvup <- cvm + cvsd
  cvlo <- cvm - cvsd

  # lambda.min: lambda with minimum CV error
  min_idx      <- which.min(cvm)
  lambda.min   <- lambda[min_idx]

  # lambda.1se: largest lambda within 1 SE of minimum
  cutoff       <- cvm[min_idx] + cvsd[min_idx]
  lambda.1se   <- max(lambda[cvm <= cutoff])

  nzero <- if (family == "multinomial") {
    vapply(seq_len(nlambda), function(index) {
      sum(vapply(fit_full$beta, function(beta) {
        as.integer(sum(abs(beta[, index]) > 1e-8))
      }, integer(1)))
    }, integer(1))
  } else {
    as.integer(colSums(fit_full$beta != 0))
  }

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
    foldid      = foldid,
    family      = family,
    fast.mode   = fit_full$fast.mode,
    prec        = fit_full$prec,
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
  positive.lambda <- x$lambda[x$lambda > 0]
  lambda.floor <- if (length(positive.lambda) > 0L) {
    max(min(positive.lambda) / 10, .Machine$double.xmin)
  } else {
    .Machine$double.xmin
  }
  safe.log.lambda <- function(lambda) log(pmax(lambda, lambda.floor))
  log_lambda <- sign.lambda * safe.log.lambda(x$lambda)
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
  abline(v = sign.lambda * safe.log.lambda(x$lambda.min),
         lty = 2, col = "red")
  abline(v = sign.lambda * safe.log.lambda(x$lambda.1se),
         lty = 2, col = "blue")

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
