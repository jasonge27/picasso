.PICASSO_ASSESS_BLOCK_BYTES <- 8 * 1024^2


.picasso_binary_confusion_table <- function(predicted, actual.bin.offset) {
  structure(
    tabulate(1L + predicted + actual.bin.offset, nbins = 4L),
    dim = c(2L, 2L),
    dimnames = list(predicted = c("0", "1"), actual = c("0", "1")),
    class = "table"
  )
}


.picasso_multinomial_confusion_table <- function(
    predicted, response, levels) {
  levels <- as.character(levels)
  K <- length(levels)
  structure(
    tabulate(predicted + K * (response - 1L), nbins = K * K),
    dim = c(K, K),
    dimnames = list(predicted = levels, actual = levels),
    class = "table"
  )
}


.picasso_scalar_assessment_metrics <- function(
    newx, newy, beta.mat, intercept, family, offset = NULL,
    block.bytes = .PICASSO_ASSESS_BLOCK_BYTES) {
  n <- nrow(newx)
  nlambda <- ncol(beta.mat)
  if (nlambda < 1L || length(intercept) != nlambda) {
    stop("Fitted scalar path has inconsistent coefficient dimensions.")
  }
  if (length(block.bytes) != 1L || !is.numeric(block.bytes) ||
      is.na(block.bytes) || !is.finite(block.bytes) || block.bytes <= 0) {
    stop("block.bytes must be one finite positive number.")
  }
  full.predictor.bytes <- 8 * as.double(n) * nlambda

  # Preserve the established single-GEMM path, including its low overhead and
  # reduction order, whenever its predictor already fits within the budget.
  if (full.predictor.bytes <= block.bytes) {
    eta <- newx %*% beta.mat +
      matrix(rep(intercept, each = n), nrow = n)
    if (!is.null(offset)) eta <- eta + offset
    if (family %in% c("gaussian", "sqrtlasso")) {
      mse <- colMeans((newy - eta)^2)
      return(list(
        mse = mse,
        mae = colMeans(abs(newy - eta)),
        deviance = mse / 2
      ))
    }
    if (family == "binomial") {
      return(list(
        deviance = .picasso_binomial_nll_from_eta(newy, eta),
        class = vapply(seq_len(nlambda), function(k) {
          pred.class <- as.integer(eta[, k] > 0)
          mean(pred.class != newy)
        }, numeric(1))
      ))
    }
    if (family == "poisson") {
      fitted.mean <- .picasso_poisson_mean(eta)
      return(list(
        deviance = .picasso_poisson_deviance_from_eta(
          newy, eta, mu = fitted.mean
        ),
        mse = colMeans((newy - fitted.mean)^2)
      ))
    }
    stop(sprintf("Unsupported scalar assessment family '%s'.", family))
  }

  # A blocked matrix product materializes both the coefficient slice and its
  # n-by-block predictor. Bound their combined size, not just eta.
  block.columns <- as.integer(max(1, min(
    nlambda,
    floor(as.double(block.bytes) /
            (8 * (max(n, 1L) + nrow(beta.mat))))
  )))

  metrics <- if (family %in% c("gaussian", "sqrtlasso")) {
    list(mse = numeric(nlambda), mae = numeric(nlambda),
         deviance = numeric(nlambda))
  } else if (family == "binomial") {
    list(deviance = numeric(nlambda), class = numeric(nlambda))
  } else if (family == "poisson") {
    list(deviance = numeric(nlambda), mse = numeric(nlambda))
  } else {
    stop(sprintf("Unsupported scalar assessment family '%s'.", family))
  }

  for (block.start in seq.int(1L, nlambda, by = block.columns)) {
    block.stop <- min(nlambda, block.start + block.columns - 1L)
    indices <- seq.int(block.start, block.stop)
    beta.block <- beta.mat[, indices, drop = FALSE]
    eta <- newx %*% beta.block
    eta <- sweep(eta, 2L, intercept[indices], FUN = "+")
    if (!is.null(offset)) eta <- eta + offset

    if (family %in% c("gaussian", "sqrtlasso")) {
      metrics$mse[indices] <- colMeans((newy - eta)^2)
      metrics$mae[indices] <- colMeans(abs(newy - eta))
      metrics$deviance[indices] <- metrics$mse[indices] / 2
    } else if (family == "binomial") {
      metrics$deviance[indices] <-
        .picasso_binomial_nll_from_eta(newy, eta)
      metrics$class[indices] <- vapply(seq_along(indices), function(k) {
        pred.class <- as.integer(eta[, k] > 0)
        mean(pred.class != newy)
      }, numeric(1))
    } else {
      fitted.mean <- .picasso_poisson_mean(eta)
      metrics$deviance[indices] <- .picasso_poisson_deviance_from_eta(
        newy, eta, mu = fitted.mean
      )
      metrics$mse[indices] <- colMeans((newy - fitted.mean)^2)
    }
  }

  metrics
}


.picasso_binomial_confusion_tables <- function(
    newx, newy, beta.mat, intercept, offset = NULL,
    block.bytes = .PICASSO_ASSESS_BLOCK_BYTES) {
  n <- nrow(newx)
  nlambda <- ncol(beta.mat)
  if (nlambda < 1L || length(intercept) != nlambda) {
    stop("Fitted binomial path has inconsistent coefficient dimensions.")
  }
  if (length(block.bytes) != 1L || !is.numeric(block.bytes) ||
      is.na(block.bytes) || !is.finite(block.bytes) || block.bytes <= 0) {
    stop("block.bytes must be one finite positive number.")
  }

  # Keep the established single-GEMM implementation whenever its avoidable
  # output workspace fits.  beta.mat is already a selected dense matrix at
  # this point, so counting it here would make wide problems block needlessly.
  # eta is double and predictions are integer: (8 + 4) * n * nlambda bytes.
  actual.bin.offset <- 2L * as.integer(newy)
  full.workspace.bytes <- 12 * as.double(n) * as.double(nlambda)
  if (full.workspace.bytes <= block.bytes) {
    eta <- newx %*% beta.mat +
      matrix(rep(intercept, each = n), nrow = n)
    if (!is.null(offset)) eta <- eta + offset

    predictions <- matrix(as.integer(eta > 0), nrow = n)
    return(lapply(seq_len(ncol(predictions)), function(k) {
      .picasso_binary_confusion_table(
        predictions[, k], actual.bin.offset
      )
    }))
  }

  # Once blocking is necessary, budget the double predictor and selected
  # coefficient-column copy.  This keeps the proven faster tall-problem block
  # width while the full-path gate above accounts for integer predictions.
  block.column.bytes <- 8 * (
    as.double(n) + as.double(nrow(beta.mat))
  )
  block.columns <- as.integer(max(1, min(
    nlambda,
    floor(as.double(block.bytes) / max(block.column.bytes, 1))
  )))
  result <- vector("list", nlambda)

  for (block.start in seq.int(1L, nlambda, by = block.columns)) {
    block.stop <- min(nlambda, block.start + block.columns - 1L)
    indices <- seq.int(block.start, block.stop)
    beta.block <- beta.mat[, indices, drop = FALSE]
    eta <- newx %*% beta.block +
      matrix(rep(intercept[indices], each = n), nrow = n)
    if (!is.null(offset)) eta <- eta + offset
    predictions <- matrix(as.integer(eta > 0), nrow = n)

    for (k in seq_along(indices)) {
      result[[indices[k]]] <- .picasso_binary_confusion_table(
        predictions[, k], actual.bin.offset
      )
    }
  }

  result
}


assess.picasso <- function(object, newx, newy, newoffset = NULL, ...) {
  family <- object$family
  if (is.null(family))
    stop("object must have a $family field set by picasso()")

  if (missing(newx) || is.null(newx))
    stop("newx must be provided")
  if (missing(newy) || is.null(newy))
    stop("newy must be provided")

  if (family == "multinomial") {
    if (!is.null(newoffset))
      stop("newoffset is supported only for binomial or Poisson models.")
    newx <- .picasso_multinomial_newdata(object, newx)
    response <- .picasso_multinomial_response_codes(
      newy, object$levels, nrow(newx), "newy"
    )
    lambda.idx <- seq_len(object$nlambda)
    deviance <- numeric(object$nlambda)
    class.loss <- numeric(object$nlambda)
    for (index in lambda.idx) {
      logits <- .picasso_multinomial_logits_one(object, newx, index)
      deviance[index] <- .picasso_multinomial_nll_from_logits(
        response, logits
      )
      class.loss[index] <- mean(
        max.col(logits, ties.method = "first") != response
      )
    }
    result <- list(lambda = object$lambda, deviance = deviance,
                   class = class.loss)
    class(result) <- "assess.picasso"
    return(result)
  }

  if (!is.numeric(newx) || length(dim(newx)) != 2L || nrow(newx) == 0L ||
      anyNA(newx) || any(!is.finite(newx)))
    stop("newx must be a nonempty finite numeric matrix.")
  if (ncol(newx) != nrow(object$beta))
    stop(sprintf(
      "newx has %d columns; the fitted model expects %d.",
      ncol(newx), nrow(object$beta)
    ))
  n       <- nrow(newx)
  newy <- if (family == "binomial") {
    .picasso_binomial_response_codes(newy, object$levels, n, "newy")
  } else {
    if (!is.numeric(newy) || length(newy) != n || anyNA(newy) ||
        any(!is.finite(newy)))
      stop(sprintf("newy must be a finite numeric vector of length %d.", n))
    as.numeric(newy)
  }
  if (family == "poisson" && any(newy < 0))
    stop("newy must contain nonnegative values for Poisson assessment.")
  offset <- if (family %in% c("binomial", "poisson")) {
    .picasso_prediction_offset(object, newoffset, n)
  } else {
    if (!is.null(newoffset))
      stop("newoffset is supported only for binomial or Poisson models.")
    NULL
  }

  beta_mat   <- as.matrix(object$beta)
  intcpt_vec <- as.numeric(object$intercept)
  metrics <- .picasso_scalar_assessment_metrics(
    newx, newy, beta_mat, intcpt_vec, family, offset
  )
  result <- c(list(lambda = object$lambda), metrics)

  class(result) <- "assess.picasso"
  result
}


print.assess.picasso <- function(x, ...) {
  cat("assess.picasso result:\n")
  metrics <- setdiff(names(x), "lambda")
  for (m in metrics) {
    cat(sprintf("  %s: range [%.4g, %.4g]\n",
                m, min(x[[m]]), max(x[[m]])))
  }
  invisible(x)
}


confusion.picasso <- function(object, newx, newy, lambda.idx = NULL,
                              newoffset = NULL, ...) {
  family <- object$family
  if (is.null(family) || !(family %in% c("binomial", "multinomial")))
    stop("confusion.picasso supports only binomial or multinomial family")

  if (missing(newx) || is.null(newx))
    stop("newx must be provided")
  if (missing(newy) || is.null(newy))
    stop("newy must be provided")

  if (family == "multinomial") {
    if (!is.null(newoffset))
      stop("newoffset is supported only for binomial confusion matrices.")
    newx <- .picasso_multinomial_newdata(object, newx)
    response <- .picasso_multinomial_response_codes(
      newy, object$levels, nrow(newx), "newy"
    )
    if (is.null(lambda.idx)) {
      lambda.idx <- seq_len(object$nlambda)
    } else {
      lambda.idx <- .picasso_multinomial_indices(
        lambda.idx, object$nlambda, "lambda.idx"
      )
    }
    result <- lapply(lambda.idx, function(index) {
      logits <- .picasso_multinomial_logits_one(object, newx, index)
      .picasso_multinomial_confusion_table(
        max.col(logits, ties.method = "first"),
        response,
        object$levels
      )
    })
    names(result) <- paste0("lambda[", lambda.idx, "]")
    return(result)
  }

  if (!is.numeric(newx) || length(dim(newx)) != 2L || nrow(newx) == 0L ||
      anyNA(newx) || any(!is.finite(newx)))
    stop("newx must be a nonempty finite numeric matrix.")
  if (ncol(newx) != nrow(object$beta))
    stop(sprintf(
      "newx has %d columns; the fitted model expects %d.",
      ncol(newx), nrow(object$beta)
    ))
  newy <- .picasso_binomial_response_codes(
    newy, object$levels, nrow(newx), "newy"
  )
  if (is.null(lambda.idx)) {
    lambda.idx <- seq_len(object$nlambda)
  } else {
    lambda.idx <- .picasso_multinomial_indices(
      lambda.idx, object$nlambda, "lambda.idx"
    )
  }

  n         <- nrow(newx)
  offset <- .picasso_prediction_offset(object, newoffset, n)
  beta_sub  <- as.matrix(object$beta[, lambda.idx, drop = FALSE])
  intcpt_sub <- as.numeric(object$intercept)[lambda.idx]
  .picasso_binomial_confusion_tables(
    newx, newy, beta_sub, intcpt_sub, offset
  )
}
