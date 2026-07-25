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
n <- 100000L
d <- 20L
nlambda <- 100L
x <- matrix(rnorm(n * d), nrow = n, ncol = d)
beta <- matrix(rnorm(d * nlambda, sd = 0.03), nrow = d)
intercept <- seq(-0.2, 0.2, length.out = nlambda)
y <- factor(ifelse(seq_len(n) %% 3L == 0L, "yes", "no"),
            levels = c("no", "yes"))
object <- list(
  family = "binomial", levels = levels(y), nlambda = nlambda,
  lambda = exp(seq(log(1), log(0.1), length.out = nlambda)),
  beta = beta, intercept = intercept, offset.used = FALSE
)

elapsed <- numeric(repetitions)
result <- NULL
for (iteration in seq_len(repetitions)) {
  invisible(gc())
  elapsed[[iteration]] <- system.time({
    result <- confusion.picasso(object, x, y)
  })[["elapsed"]]
}

saveRDS(result, output, version = 3L)
cat(sprintf(
  "median=%.9f min=%.9f max=%.9f tables=%d checksum=%.17g\n",
  median(elapsed), min(elapsed), max(elapsed), length(result),
  sum(vapply(result, sum, numeric(1)))
))
