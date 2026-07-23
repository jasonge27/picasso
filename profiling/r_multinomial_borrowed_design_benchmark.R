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
n <- 100000L
d <- 100L
K <- 3L
set.seed(20260720)
x <- matrix(rnorm(n * d), nrow = n, ncol = d)
labels <- factor(rep(c("a", "b", "c"), length.out = n),
                 levels = c("a", "b", "c"))
probe.index <- unique(pmax(1L, pmin(length(x), c(
  1L, 2L, n, n + 1L, length(x) %/% 2L, length(x) - 1L, length(x)
))))
input.checksum <- c(sum = sum(x), samples = x[probe.index])

elapsed <- numeric(repetitions)
fit <- NULL
for (iteration in seq_len(repetitions)) {
  invisible(gc())
  elapsed[[iteration]] <- system.time({
    fit <- picasso(
      x, labels, family = "multinomial", lambda = 10,
      standardize = FALSE, intercept = TRUE,
      fast.mode = TRUE, verbose = FALSE
    )
  })[["elapsed"]]
}
stopifnot(identical(
  input.checksum,
  c(sum = sum(x), samples = x[probe.index])
))

summary <- list(
  lambda = fit$lambda,
  beta = lapply(fit$beta, as.matrix),
  intercept = fit$intercept,
  dev.ratio = fit$dev.ratio,
  df = fit$df,
  nlambda = fit$nlambda,
  status = fit$status,
  status.code = fit$status.code,
  diagnostics = within(fit$diagnostics, runtime <- NULL),
  input.checksum = input.checksum
)
saveRDS(summary, output, version = 3L)
cat(sprintf(
  "median=%.9f min=%.9f max=%.9f checksum=%.17g\n",
  median(elapsed), min(elapsed), max(elapsed),
  sum(vapply(summary$beta, sum, numeric(1))) +
    sum(unlist(summary$intercept)) + sum(summary$dev.ratio)
))
