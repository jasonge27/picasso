test_that("sqrt-lasso accepts the exact constant-response solution", {
  set.seed(731)
  n <- 37L
  d <- 6L
  x <- matrix(rnorm(n * d), nrow = n, ncol = d)
  y <- rep(0.125, n)
  lambda <- c(0.4, 0.1, 0.0)

  for (method in c("l1", "mcp", "scad")) {
    fit <- picasso(
      x, y, family = "sqrtlasso", method = method,
      lambda = lambda, intercept = TRUE, standardize = TRUE,
      prec = 1e-8, max.ite = 1000L
    )
    label <- paste("sqrt-lasso", method)

    expect_identical(fit$status, "completed", info = label)
    expect_identical(fit$status.code, 0L, info = label)
    expect_identical(fit$nlambda, length(lambda), info = label)
    expect_equal(as.matrix(fit$beta), matrix(0, d, length(lambda)),
                 tolerance = 0, info = label)
    expect_equal(fit$intercept, rep(y[1L], length(lambda)),
                 tolerance = 0, info = label)
    expect_equal(fit$diagnostics$iterations, rep(0L, length(lambda)),
                 info = label)
    expect_equal(fit$diagnostics$nonzero, rep(0L, length(lambda)),
                 info = label)
    expect_true(all(is.finite(as.matrix(fit$diagnostics))), info = label)
    expect_equal(fit$diagnostics$objective, rep(0, length(lambda)),
                 tolerance = 0, info = label)
    expect_equal(fit$diagnostics$kkt, rep(0, length(lambda)),
                 tolerance = 0, info = label)
    expect_equal(fit$diagnostics$stationarity, rep(0, length(lambda)),
                 tolerance = 0, info = label)
    expect_equal(fit$nulldev, 0, tolerance = 0, info = label)
    expect_equal(fit$dev.ratio, rep(0, length(lambda)),
                 tolerance = 0, info = label)
    expect_equal(
      as.numeric(predict(fit, x[1:4, , drop = FALSE], lambda.idx = 1:3,
                         Y.pred.idx = 1:4)),
      rep(y[1L], 12L), tolerance = 0, info = label
    )
  }
})
