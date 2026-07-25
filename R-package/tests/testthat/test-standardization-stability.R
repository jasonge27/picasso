extreme_finite_design <- function(n = 18L) {
  fraction <- seq(0.55, 0.95, length.out = n)
  cbind(
    same.sign = 1e308 * fraction,
    wide.center = 1e308 * c(-1, rep(1, n - 1L)),
    ordinary = seq(-2, 2, length.out = n),
    constant = rep(1e308, n)
  )
}


test_that("centered standardization is stable for extreme finite columns", {
  x <- extreme_finite_design()
  n <- nrow(x)
  design <- picasso:::.picasso_prepare_design(
    x, standardize = TRUE, center = TRUE
  )

  expect_true(all(is.finite(design$xx)))
  expect_true(all(is.finite(design$xm)))
  expect_true(all(is.finite(design$xinvc.vec)))
  expect_equal(
    as.numeric(design$xm[1L, 1L]),
    1e308 * mean(seq(0.55, 0.95, length.out = n)),
    tolerance = 2e-15
  )
  expect_equal(colMeans(design$xx)[1:3], rep(0, 3), tolerance = 2e-15)
  expect_equal(colSums(design$xx[, 1:3, drop = FALSE]^2),
               rep(n - 1, 3), tolerance = 2e-13)
  expect_equal(design$xx[, 4L], rep(0, n), tolerance = 0)
  expect_equal(design$xinvc.vec[4L], 0, tolerance = 0)

  ordinary.reference <- as.numeric(scale(x[, "ordinary"]))
  expect_equal(design$xx[, 3L], ordinary.reference,
               tolerance = 2e-15)
  expect_equal(design$xm[1L, 3L], mean(x[, "ordinary"]),
               tolerance = 2e-15)
  expect_equal(design$xinvc.vec[3L], 1 / sd(x[, "ordinary"]),
               tolerance = 2e-15)
})


test_that("Gaussian centered standardization is stable for extreme finite columns", {
  x <- extreme_finite_design()
  shared <- picasso:::.picasso_prepare_design(
    x, standardize = TRUE, center = TRUE
  )
  gaussian <- picasso:::.picasso_prepare_gaussian_design(
    x, standardize = TRUE, intercept = TRUE
  )

  expect_equal(gaussian$xx, shared$xx, tolerance = 0)
  expect_equal(gaussian$xm, as.numeric(shared$xm), tolerance = 0)
  expect_equal(gaussian$xinvc.vec, shared$xinvc.vec, tolerance = 0)
  expect_true(all(is.finite(gaussian$xx)))
  expect_true(all(is.finite(gaussian$xm)))
  expect_true(all(is.finite(gaussian$xinvc.vec)))
})


test_that("Gaussian fits an extreme finite centered design", {
  x <- extreme_finite_design()
  y <- seq(-1, 1, length.out = nrow(x))

  for (gaussian.type in c("naive", "covariance")) {
    fit <- picasso(
      x, y, family = "gaussian", type.gaussian = gaussian.type,
      method = "l1", lambda = c(0.2, 0.1), standardize = TRUE,
      intercept = TRUE, prec = 1e-7, max.ite = 2000L
    )

    expect_identical(fit$nlambda, 2L)
    expect_true(all(is.finite(as.matrix(fit$beta))))
    expect_true(all(is.finite(fit$intercept)))
    expect_true(all(is.finite(fit$dev.ratio)))
    prediction <- predict(
      fit, x, lambda.idx = seq_len(fit$nlambda),
      Y.pred.idx = seq_len(nrow(x)), type = "response"
    )
    expect_true(all(is.finite(prediction)))
  }
})


test_that("scalar GLM fits an extreme finite centered design", {
  x <- extreme_finite_design()
  y <- factor(rep(c("no", "yes", "no"), length.out = nrow(x)))
  fit <- picasso(
    x, y, family = "binomial", method = "l1",
    lambda = c(0.2, 0.1), standardize = TRUE, intercept = TRUE,
    prec = 1e-7, max.ite = 2000L
  )

  expect_identical(fit$status, "completed")
  expect_identical(fit$status.code, 0L)
  expect_identical(fit$nlambda, 2L)
  expect_true(all(is.finite(as.matrix(fit$beta))))
  expect_true(all(is.finite(fit$intercept)))
  expect_true(all(is.finite(fit$dev.ratio)))
  for (lambda.index in seq_len(fit$nlambda)) {
    probability <- predict(
      fit, x, lambda.idx = lambda.index,
      p.pred.idx = seq_len(nrow(x)), type = "response"
    )
    expect_true(all(is.finite(probability)))
    expect_true(all(probability >= 0 & probability <= 1))
  }
})


test_that("multinomial fits an extreme finite centered design", {
  x <- extreme_finite_design()
  y <- factor(rep(c("alpha", "beta", "gamma"), length.out = nrow(x)))
  fit <- picasso(
    x, y, family = "multinomial", method = "l1",
    lambda = c(0.2, 0.1), standardize = TRUE, intercept = TRUE,
    prec = 1e-7, max.ite = 2000L
  )

  expect_identical(fit$status, "completed")
  expect_identical(fit$status.code, 0L)
  expect_identical(fit$nlambda, 2L)
  expect_true(all(vapply(fit$beta, function(value) {
    all(is.finite(as.matrix(value)))
  }, logical(1))))
  expect_true(all(vapply(fit$intercept, function(value) {
    all(is.finite(value))
  }, logical(1))))
  expect_true(all(is.finite(fit$dev.ratio)))
  for (lambda.index in seq_len(fit$nlambda)) {
    probability <- predict(
      fit, x, lambda.idx = lambda.index, type = "response"
    )
    expect_true(all(is.finite(probability)))
    expect_equal(rowSums(probability), rep(1, nrow(x)), tolerance = 1e-12)
  }
})
