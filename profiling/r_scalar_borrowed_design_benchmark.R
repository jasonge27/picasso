args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop(paste(
    "usage: Rscript r_scalar_borrowed_design_benchmark.R",
    "<family> <shape> <short|path> <repetitions> <output.rds>"
  ))
}

family <- args[[1L]]
shape <- args[[2L]]
mode <- match.arg(args[[3L]], c("short", "path"))
repetitions <- as.integer(args[[4L]])
output <- args[[5L]]
stopifnot(family %in% c("gaussian", "binomial", "poisson", "sqrtlasso"))
stopifnot(is.finite(repetitions), repetitions >= 1L)

suppressPackageStartupMessages(library(picasso))

dimensions <- switch(
  shape,
  tall = c(n = 100000L, d = 100L),
  wide = c(n = 1000L, d = 10000L),
  dense_path = c(n = 4000L, d = 500L),
  stop("unknown shape")
)
n <- dimensions[["n"]]
d <- dimensions[["d"]]
set.seed(20260720 + n + d)
x <- matrix(rnorm(n * d), nrow = n, ncol = d)
signal <- numeric(d)
signal[seq_len(min(d, 12L))] <-
  seq(0.55, -0.25, length.out = min(d, 12L))
eta <- drop(x %*% signal)
y <- switch(
  family,
  gaussian = eta + rnorm(n, sd = 0.7),
  binomial = as.numeric(runif(n) < plogis(eta)),
  poisson = rpois(n, exp(pmax(-2, pmin(2, eta)))),
  sqrtlasso = eta + rnorm(n, sd = 0.7)
)

lambda <- if (mode == "short") {
  10
} else {
  limits <- switch(
    family,
    gaussian = c(0.8, 0.04),
    binomial = c(0.2, 0.006),
    poisson = c(0.5, 0.015),
    sqrtlasso = c(0.8, 0.04)
  )
  exp(seq(log(limits[[1L]]), log(limits[[2L]]), length.out = 30L))
}

# Avoid a full-size checksum temporary: /usr/bin/time -l should measure the
# solver's design copy, not a preceding x*x allocation.
probe_index <- unique(pmax(1L, pmin(length(x), c(
  1L, 2L, n, n + 1L, length(x) %/% 2L, length(x) - 1L, length(x)
))))
input_checksum <- c(sum = sum(x), samples = x[probe_index])
invisible(gc(reset = TRUE))

elapsed <- numeric(repetitions)
fit <- NULL
for (iteration in seq_len(repetitions)) {
  invisible(gc())
  elapsed[[iteration]] <- system.time({
    fit <- picasso(
      x, y, family = family, lambda = lambda,
      standardize = FALSE, intercept = TRUE,
      type.gaussian = "naive", fast.mode = TRUE,
      verbose = FALSE
    )
  })[["elapsed"]]
}
stopifnot(identical(
  input_checksum,
  c(sum = sum(x), samples = x[probe_index])
))

result <- list(
  metadata = list(
    family = family,
    shape = shape,
    mode = mode,
    n = n,
    d = d,
    repetitions = repetitions,
    package_version = as.character(packageVersion("picasso")),
    R = R.version.string,
    platform = R.version$platform,
    blas = unname(extSoftVersion()[["BLAS"]])
  ),
  elapsed = elapsed,
  summary = list(
    lambda = fit$lambda,
    beta = as.matrix(fit$beta),
    intercept = fit$intercept,
    dev.ratio = fit$dev.ratio,
    df = fit$df,
    nlambda = fit$nlambda,
    status = fit$status,
    status.code = fit$status.code,
    path.early.stopped = fit$path.early.stopped,
    input_checksum = input_checksum
  )
)
saveRDS(result, output, version = 3L)
cat(sprintf(
  paste0(
    "family=%s shape=%s mode=%s median=%.9f min=%.9f max=%.9f ",
    "checksum=%.17g\n"
  ),
  family, shape, mode, median(elapsed), min(elapsed), max(elapsed),
  sum(result$summary$beta) + sum(result$summary$intercept) +
    sum(result$summary$dev.ratio)
))
