test_that("Gaussian iteration limits retain only converged prefixes", {
  x <- matrix(c(
     1.0,  1.1,  0.9,
     2.0,  2.1,  1.8,
     3.0,  3.2,  2.7,
     4.0,  4.1,  3.7,
    -1.0, -1.2, -0.8,
    -2.0, -2.1, -1.7,
    -3.0, -3.1, -2.8,
    -4.0, -4.2, -3.6
  ), ncol = 3L, byrow = TRUE)
  y <- c(3.0, 5.9, 9.2, 12.1, -3.2, -6.1, -9.0, -12.2)

  for (backend in c("naive", "covariance")) {
    limited <- NULL
    expect_warning(
      limited <- picasso(
        x, y, lambda = c(50, 0.05), family = "gaussian",
        type.gaussian = backend, standardize = FALSE, intercept = FALSE,
        max.ite = 1L
      ),
      "successful 1/2-lambda prefix"
    )
    expect_identical(limited$nlambda, 1L, info = backend)
    expect_identical(limited$status, "inner_iteration_limit", info = backend)
    expect_identical(limited$status.code, 4L, info = backend)
    expect_identical(limited$failure$lambda.index, 2L, info = backend)
    expect_equal(as.matrix(limited$beta), matrix(0, 3L, 1L), info = backend)

    expect_error(
      picasso(
        x, y, lambda = 0.05, family = "gaussian",
        type.gaussian = backend, standardize = FALSE, intercept = FALSE,
        max.ite = 1L
      ),
      "inner_iteration_limit.*before completing"
    )

    completed <- picasso(
      x, y, lambda = c(50, 0.05), family = "gaussian",
      type.gaussian = backend, standardize = FALSE, intercept = FALSE,
      max.ite = 1000L
    )
    expect_identical(completed$nlambda, 2L, info = backend)
    expect_identical(completed$status, "completed", info = backend)
    expect_identical(completed$status.code, 0L, info = backend)
    expect_null(completed$failure, info = backend)
  }
})
