legacy.scalar.assessment.metrics <- function(
    newx, newy, beta.mat, intercept, family, offset = NULL) {
  n <- nrow(newx)
  nlambda <- ncol(beta.mat)
  eta <- newx %*% beta.mat +
    matrix(rep(intercept, each = n), nrow = n)
  if (!is.null(offset)) eta <- eta + offset

  if (family %in% c("gaussian", "sqrtlasso")) {
    mse <- colMeans((newy - eta)^2)
    list(mse = mse, mae = colMeans(abs(newy - eta)), deviance = mse / 2)
  } else if (family == "binomial") {
    list(
      deviance = picasso:::.picasso_binomial_nll_from_eta(newy, eta),
      class = vapply(seq_len(nlambda), function(k) {
        mean(as.integer(eta[, k] > 0) != newy)
      }, numeric(1))
    )
  } else {
    fitted.mean <- picasso:::.picasso_poisson_mean(eta)
    list(
      deviance = picasso:::.picasso_poisson_deviance_from_eta(
        newy, eta, mu = fitted.mean
      ),
      mse = colMeans((newy - fitted.mean)^2)
    )
  }
}


test_that("scalar assessment blocks preserve every family metric", {
  set.seed(20260719)
  n <- 37L
  d <- 6L
  nlambda <- 7L
  newx <- matrix(rnorm(n * d), nrow = n)
  beta.mat <- matrix(rnorm(d * nlambda, sd = 0.12), nrow = d)
  intercept <- seq(-0.25, 0.2, length.out = nlambda)
  offset <- seq(-0.15, 0.1, length.out = n)
  responses <- list(
    gaussian = rnorm(n),
    sqrtlasso = rnorm(n),
    binomial = as.numeric(seq_len(n) %% 3L == 0L),
    poisson = as.numeric(seq_len(n) %% 5L)
  )

  for (family in names(responses)) {
    family.offset <- if (family %in% c("binomial", "poisson")) offset else NULL
    expected <- legacy.scalar.assessment.metrics(
      newx, responses[[family]], beta.mat, intercept, family, family.offset
    )
    one.gemm <- picasso:::.picasso_scalar_assessment_metrics(
      newx, responses[[family]], beta.mat, intercept, family, family.offset,
      block.bytes = 8 * n * nlambda
    )
    two.columns <- picasso:::.picasso_scalar_assessment_metrics(
      newx, responses[[family]], beta.mat, intercept, family, family.offset,
      block.bytes = 8 * (n + d) * 2
    )
    one.column <- picasso:::.picasso_scalar_assessment_metrics(
      newx, responses[[family]], beta.mat, intercept, family, family.offset,
      block.bytes = 1
    )

    expect_identical(names(one.gemm), names(expected), info = family)
    expect_identical(one.gemm, expected, info = family)
    expect_equal(two.columns, expected, tolerance = 1e-13, info = family)
    expect_equal(one.column, expected, tolerance = 1e-13, info = family)

    if (family %in% c("binomial", "poisson")) {
      expected.no.offset <- legacy.scalar.assessment.metrics(
        newx, responses[[family]], beta.mat, intercept, family
      )
      blocked.no.offset <- picasso:::.picasso_scalar_assessment_metrics(
        newx, responses[[family]], beta.mat, intercept, family,
        block.bytes = 1
      )
      expect_equal(blocked.no.offset, expected.no.offset, tolerance = 1e-13,
                   info = paste(family, "without offset"))
    }
  }
})


test_that("single-row single-lambda assessment supports both execution paths", {
  x <- matrix(2, nrow = 1L, ncol = 1L)
  beta <- matrix(0.25, nrow = 1L, ncol = 1L)
  expected <- legacy.scalar.assessment.metrics(
    x, 1.5, beta, -0.1, "gaussian"
  )
  one.gemm <- picasso:::.picasso_scalar_assessment_metrics(
    x, 1.5, beta, -0.1, "gaussian", block.bytes = 8
  )
  one.block <- picasso:::.picasso_scalar_assessment_metrics(
    x, 1.5, beta, -0.1, "gaussian", block.bytes = 1
  )

  expect_identical(one.gemm, expected)
  expect_equal(one.block, expected, tolerance = 1e-13)
})


test_that("public scalar assessment keeps its result contract", {
  set.seed(20260720)
  n <- 29L
  d <- 5L
  nlambda <- 4L
  newx <- matrix(rnorm(n * d), nrow = n)
  beta.mat <- matrix(rnorm(d * nlambda, sd = 0.08), nrow = d)
  intercept <- seq(-0.1, 0.15, length.out = nlambda)
  lambda <- seq(0.4, 0.1, length.out = nlambda)
  offset <- seq(-0.1, 0.1, length.out = n)
  cases <- list(
    gaussian = rnorm(n),
    sqrtlasso = rnorm(n),
    binomial = factor(rep(c("no", "yes"), length.out = n),
                      levels = c("no", "yes")),
    poisson = as.numeric(seq_len(n) %% 4L)
  )

  for (family in names(cases)) {
    response <- cases[[family]]
    numeric.response <- if (family == "binomial") {
      as.numeric(response) - 1
    } else {
      response
    }
    family.offset <- if (family %in% c("binomial", "poisson")) offset else NULL
    object <- list(
      family = family,
      beta = Matrix::Matrix(beta.mat),
      intercept = intercept,
      lambda = lambda,
      nlambda = nlambda,
      offset.used = !is.null(family.offset)
    )
    if (family == "binomial") object$levels <- levels(response)
    expected.metrics <- legacy.scalar.assessment.metrics(
      newx, numeric.response, beta.mat, intercept, family, family.offset
    )
    actual <- assess.picasso(
      object, newx, response, newoffset = family.offset
    )

    expect_s3_class(actual, "assess.picasso")
    expect_identical(names(actual), c("lambda", names(expected.metrics)),
                     info = family)
    expect_equal(unclass(actual), c(list(lambda = lambda), expected.metrics),
                 tolerance = 1e-15, info = family)
  }
})


test_that("assessment block budget validation is explicit", {
  x <- matrix(1, nrow = 2L, ncol = 1L)
  for (bad.budget in list(0, -1, NA_real_, Inf, numeric())) {
    expect_error(
      picasso:::.picasso_scalar_assessment_metrics(
        x, c(0, 1), matrix(0, 1L, 1L), 0, "gaussian",
        block.bytes = bad.budget
      ),
      "block.bytes"
    )
  }
})
