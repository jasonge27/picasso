assess.picasso <- function(object, newx, newy, ...) {
  family <- object$family
  if (is.null(family))
    stop("object must have a $family field set by picasso()")

  if (missing(newx) || is.null(newx))
    stop("newx must be provided")

  n       <- nrow(newx)
  nlambda <- object$nlambda
  newy    <- as.numeric(newy)

  beta_mat   <- as.matrix(object$beta)
  intcpt_vec <- as.numeric(object$intercept)
  eta_mat    <- newx %*% beta_mat +
                matrix(rep(intcpt_vec, each = n), nrow = n)

  result <- list(lambda = object$lambda)

  if (family %in% c("gaussian", "sqrtlasso")) {
    result$mse      <- colMeans((newy - eta_mat)^2)
    result$mae      <- colMeans(abs(newy - eta_mat))
    result$deviance <- result$mse / 2
  } else if (family == "binomial") {
    result$deviance <- vapply(seq_len(nlambda), function(k) {
      p <- pmax(pmin(1 / (1 + exp(-eta_mat[, k])), 1 - 1e-8), 1e-8)
      -mean(newy * log(p) + (1 - newy) * log(1 - p))
    }, numeric(1))
    result$class <- vapply(seq_len(nlambda), function(k) {
      pred_class <- as.integer(1 / (1 + exp(-eta_mat[, k])) > 0.5)
      mean(pred_class != newy)
    }, numeric(1))
  } else if (family == "poisson") {
    result$deviance <- vapply(seq_len(nlambda), function(k) {
      .picasso_poisson_dev(newy, exp(eta_mat[, k]))
    }, numeric(1))
    result$mse <- colMeans((newy - exp(eta_mat))^2)
  } else if (family == "multinomial") {
    stop("assess.picasso for multinomial: use predict() with type='class'")
  }

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


confusion.picasso <- function(object, newx, newy, lambda.idx = NULL, ...) {
  family <- object$family
  if (is.null(family) || !(family %in% c("binomial", "multinomial")))
    stop("confusion.picasso supports only binomial or multinomial family")

  if (missing(newx) || is.null(newx))
    stop("newx must be provided")

  newy <- as.numeric(newy)
  if (is.null(lambda.idx)) lambda.idx <- seq_len(object$nlambda)
  lambda.idx <- as.integer(lambda.idx)

  n         <- nrow(newx)
  beta_sub  <- as.matrix(object$beta)[, lambda.idx, drop = FALSE]
  intcpt_sub <- as.numeric(object$intercept)[lambda.idx]
  eta_mat   <- newx %*% beta_sub +
               matrix(rep(intcpt_sub, each = n), nrow = n)

  if (family == "binomial") {
    pred_mat <- matrix(
      as.integer(1 / (1 + exp(-eta_mat)) > 0.5),
      nrow = n
    )
  } else {
    stop("multinomial confusion not yet implemented here; use predict(type='class')")
  }

  lapply(seq_len(ncol(pred_mat)), function(k) {
    table(predicted = pred_mat[, k], actual = newy)
  })
}
