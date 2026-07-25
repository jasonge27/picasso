args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: benchmark.R repetitions output.rds")
}
repetitions <- as.integer(args[[1L]])
output <- args[[2L]]
if (is.na(repetitions) || repetitions < 1L) {
  stop("repetitions must be positive")
}

suppressPackageStartupMessages(library(picasso))
set.seed(20260720)
n <- 120000L
d <- 20L
nlambda <- 100L
x <- matrix(rnorm(n * d), nrow = n, ncol = d)
signal <- drop(x[, 1L] * 0.25 - x[, 2L] * 0.15)
y <- signal + sin(seq_len(n) / 37) * 0.2
foldid <- rep(1:2, length.out = n)
lambda <- exp(seq(log(10), log(1), length.out = nlambda))

elapsed <- numeric(repetitions)
fit <- NULL
for (iteration in seq_len(repetitions)) {
  invisible(gc())
  elapsed[[iteration]] <- system.time({
    fit <- cv.picasso(
      x, y, family = "gaussian", lambda = lambda, foldid = foldid,
      type.measure = "mse", standardize = FALSE, intercept = TRUE,
      type.gaussian = "covariance", fast.mode = TRUE, max.ite = 1000L
    )
  })[["elapsed"]]
}

summary <- list(
  lambda = fit$lambda,
  cvm = fit$cvm,
  cvsd = fit$cvsd,
  cvup = fit$cvup,
  cvlo = fit$cvlo,
  nzero = fit$nzero,
  lambda.min = fit$lambda.min,
  lambda.1se = fit$lambda.1se,
  foldid = fit$foldid,
  family = fit$family,
  fast.mode = fit$fast.mode,
  prec = fit$prec,
  beta = as.matrix(fit$picasso.fit$beta),
  intercept = as.numeric(fit$picasso.fit$intercept)
)
saveRDS(summary, output, version = 3L)
cat(sprintf(
  "median=%.9f min=%.9f max=%.9f nlambda=%d checksum=%.17g\n",
  median(elapsed), min(elapsed), max(elapsed), length(fit$lambda),
  sum(fit$cvm) + sum(fit$cvsd) + sum(summary$beta) + sum(summary$intercept)
))
