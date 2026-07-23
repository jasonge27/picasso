#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: r_native_loss_benchmark.R PACKAGE_LIB OUTPUT_CSV")
}

.libPaths(c(normalizePath(args[[1L]]), .libPaths()))
suppressPackageStartupMessages(library(picasso))

Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)

standardize <- function(x) {
  centered <- sweep(x, 2L, colMeans(x), "-")
  scale <- sqrt(colMeans(centered * centered))
  scale[scale <= sqrt(.Machine$double.eps)] <- 1
  sweep(centered, 2L, scale, "/")
}

make_design <- function(n, p, seed) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), n, p)
  for (j in 2:p) x[, j] <- 0.2 * x[, j - 1L] + sqrt(0.96) * x[, j]
  standardize(x)
}

make_lambda <- function(x, y, family, length = 45L) {
  n <- nrow(x)
  if (family == "binomial") {
    gradient <- crossprod(x, as.numeric(y) - 1L - mean(as.numeric(y) - 1L)) / n
  } else if (family == "poisson") {
    gradient <- crossprod(x, y - mean(y)) / n
  } else if (family == "multinomial") {
    encoded <- model.matrix(~ y - 1)
    gradient <- crossprod(x, sweep(encoded, 2L, colMeans(encoded), "-")) / n
  } else {
    gradient <- crossprod(x, as.numeric(y) - mean(as.numeric(y))) / n
  }
  top <- max(abs(gradient))
  exp(seq(log(top), log(top * 0.02), length.out = length))
}

make_response <- function(x, family, seed) {
  set.seed(seed)
  beta <- numeric(ncol(x))
  beta[seq_len(min(30L, ncol(x)))] <-
    rep(c(1, -1), length.out = min(30L, ncol(x))) *
    seq(1.0, 0.25, length.out = min(30L, ncol(x)))
  eta <- drop(x %*% beta)
  eta <- eta / max(1, stats::sd(eta))
  if (family %in% c("gaussian", "sqrtlasso")) return(eta + rnorm(nrow(x)))
  if (family == "binomial") {
    return(factor(rbinom(nrow(x), 1L, stats::plogis(eta)), levels = 0:1))
  }
  if (family == "poisson") return(rpois(nrow(x), exp(0.2 + 0.4 * eta)))

  k <- 4L
  coefficients <- matrix(rnorm(ncol(x) * k), ncol(x), k)
  coefficients[-seq_len(min(30L, ncol(x))), ] <- 0
  logits <- x %*% coefficients
  logits <- sweep(logits, 1L, apply(logits, 1L, max), "-")
  probabilities <- exp(logits)
  probabilities <- probabilities / rowSums(probabilities)
  draws <- vapply(seq_len(nrow(x)), function(i) {
    sample.int(k, 1L, prob = probabilities[i, ])
  }, integer(1))
  factor(draws, levels = seq_len(k))
}

x <- make_design(4000L, 120L, 73001L)
families <- c("gaussian", "binomial", "poisson", "sqrtlasso", "multinomial")
rows <- list()
index <- 0L

for (family in families) {
  y <- make_response(x, family, 73100L + match(family, families))
  lambda.family <- if (family == "sqrtlasso") "gaussian" else family
  lambda <- make_lambda(x, y, lambda.family)
  run <- function() {
    gc(FALSE)
    timing <- system.time(fit <- picasso(
      X = x, Y = y, family = family, method = "l1", lambda = lambda,
      standardize = FALSE, intercept = TRUE, max.ite = 10000L,
      verbose = FALSE, fast.mode = TRUE
    ))
    c(
      elapsed = unname(timing[["elapsed"]]),
      before_deviance = as.numeric(fit$runtime, units = "secs"),
      nlambda = fit$nlambda,
      deviance_checksum = sum(fit$dev.ratio)
    )
  }
  invisible(run())
  for (repetition in seq_len(5L)) {
    value <- run()
    index <- index + 1L
    rows[[index]] <- data.frame(
      family = family,
      repetition = repetition,
      elapsed = value[["elapsed"]],
      before_deviance = value[["before_deviance"]],
      postfit_deviance = value[["elapsed"]] - value[["before_deviance"]],
      nlambda = as.integer(value[["nlambda"]]),
      deviance_checksum = value[["deviance_checksum"]]
    )
  }
}

result <- do.call(rbind, rows)
write.csv(result, args[[2L]], row.names = FALSE)
print(aggregate(
  cbind(elapsed, before_deviance, postfit_deviance) ~ family,
  result, median
))
