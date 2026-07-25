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
suppressPackageStartupMessages(library(Matrix))
set.seed(20260720)
n <- 100000L
d <- 20L
K <- 8L
nlambda <- 100L
levels <- paste0("class [", seq_len(K), "]")
newx <- matrix(rnorm(n * d), n, d)
beta <- lapply(seq_len(K), function(k) {
  Matrix(matrix(rnorm(d * nlambda, sd = 0.08), d, nlambda))
})
intercept <- lapply(seq_len(K), function(k) {
  rnorm(nlambda, sd = 0.05)
})
object <- structure(list(
  beta = beta,
  intercept = intercept,
  lambda = exp(seq(log(2), log(0.01), length.out = nlambda)),
  nlambda = nlambda,
  K = K,
  levels = levels,
  family = "multinomial"
), class = "multinomial")
newy <- levels[sample.int(K, n, replace = TRUE)]

invisible(confusion.picasso(
  object, newx[seq_len(100L), , drop = FALSE], newy[seq_len(100L)]
))
invisible(gc())

elapsed <- numeric(repetitions)
result <- NULL
for (iteration in seq_len(repetitions)) {
  invisible(gc())
  elapsed[[iteration]] <- system.time({
    result <- confusion.picasso(object, newx, newy)
  })[["elapsed"]]
}

saveRDS(result, output, version = 3L)
cat(sprintf(
  "median=%.6f min=%.6f max=%.6f tables=%d output=%s\n",
  median(elapsed), min(elapsed), max(elapsed), length(result), output
))
