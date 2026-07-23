test_that("sqrt-lasso stops cleanly before the wide interpolation boundary", {
  set.seed(1)
  n <- 80L
  p <- 240L
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  x <- scale(x, center = TRUE, scale = FALSE)
  x <- sweep(x, 2L, sqrt(colMeans(x^2)), "/")
  beta <- c(seq(1, 0.4, length.out = 12L), rep(0, p - 12L))
  y <- drop(x %*% beta) + rnorm(n)

  residual <- y - mean(y)
  loss <- sqrt(mean(residual^2))
  lambda.max <- max(abs(crossprod(x, residual) / n)) / loss
  lambda <- exp(seq(log(lambda.max), log(0.3 * lambda.max), length.out = 35L))

  fit <- expect_no_warning(picasso(
    x, y, family = "sqrtlasso", method = "l1", lambda = lambda,
    standardize = FALSE, intercept = TRUE, fast.mode = TRUE,
    max.ite = 10000L
  ))

  expect_identical(fit$status, "completed")
  expect_null(fit$failure)
  expect_lt(fit$nlambda, length(lambda))
  expect_gt(tail(fit$dev.ratio, 1L), 0.999)
  expect_lte(max(fit$diagnostics$kkt), fit$prec * 1.001)
})
