legacy.scalar.cv.loss <- function(
    x, y, beta, intercept, family, measure, offset = NULL, n.fit = ncol(beta)) {
  n <- length(y)
  beta.sub <- as.matrix(beta)[, seq_len(n.fit), drop = FALSE]
  intercept.sub <- as.numeric(intercept)[seq_len(n.fit)]
  eta.mat <- x %*% beta.sub +
    matrix(rep(intercept.sub, each = n), nrow = n)
  if (family %in% c("binomial", "poisson") && !is.null(offset)) {
    eta.mat <- eta.mat + matrix(rep(offset, n.fit), nrow = n)
  }

  vapply(seq_len(n.fit), function(k) {
    eta <- as.numeric(eta.mat[, k])
    response.fit <- if (family == "binomial") {
      stats::plogis(eta)
    } else if (family == "poisson") {
      picasso:::.picasso_poisson_mean(eta)
    } else {
      eta
    }
    switch(
      measure,
      deviance = {
        if (family %in% c("gaussian", "sqrtlasso")) {
          mean((y - eta)^2) / 2
        } else if (family == "binomial") {
          picasso:::.picasso_binomial_nll_from_eta(y, eta)
        } else {
          picasso:::.picasso_poisson_deviance_from_eta(
            y, eta, mu = response.fit
          )
        }
      },
      mse = mean((y - response.fit)^2),
      mae = mean(abs(y - response.fit)),
      class = mean(as.integer(eta > 0) != y)
    )
  }, numeric(1))
}


test_that("scalar CV lambda blocks preserve every supported loss", {
  set.seed(20260720)
  n <- 37L
  d <- 6L
  nlambda <- 7L
  n.fit <- 5L
  x <- matrix(rnorm(n * d), nrow = n)
  beta <- Matrix::Matrix(matrix(rnorm(d * nlambda, sd = 0.12), nrow = d))
  intercept <- seq(-0.25, 0.2, length.out = nlambda)
  offset <- seq(-0.15, 0.1, length.out = n)
  responses <- list(
    gaussian = rnorm(n),
    sqrtlasso = rnorm(n),
    binomial = as.numeric(seq_len(n) %% 3L == 0L),
    poisson = as.numeric(seq_len(n) %% 5L)
  )
  measures <- list(
    gaussian = c("deviance", "mse", "mae"),
    sqrtlasso = c("deviance", "mse", "mae"),
    binomial = c("deviance", "mse", "mae", "class"),
    poisson = c("deviance", "mse", "mae")
  )

  for (family in names(responses)) {
    family.offset <- if (family %in% c("binomial", "poisson")) offset else NULL
    for (measure in measures[[family]]) {
      expected <- legacy.scalar.cv.loss(
        x, responses[[family]], beta, intercept, family, measure,
        family.offset, n.fit
      )
      one.gemm <- picasso:::.picasso_cv_scalar_fold_loss(
        x, responses[[family]], beta, intercept, n.fit, family, measure,
        family.offset, block.bytes = 8 * (n + d) * n.fit
      )
      two.columns <- picasso:::.picasso_cv_scalar_fold_loss(
        x, responses[[family]], beta, intercept, n.fit, family, measure,
        family.offset, block.bytes = 8 * (n + d) * 2
      )
      one.column <- picasso:::.picasso_cv_scalar_fold_loss(
        x, responses[[family]], beta, intercept, n.fit, family, measure,
        family.offset, block.bytes = 1
      )

      expect_identical(one.gemm, expected,
                       info = paste(family, measure, "single GEMM"))
      if (measure == "class") {
        expect_identical(two.columns, expected, info = family)
        expect_identical(one.column, expected, info = family)
      } else {
        expect_equal(two.columns, expected, tolerance = 1e-13,
                     info = paste(family, measure, "two columns"))
        expect_equal(one.column, expected, tolerance = 1e-13,
                     info = paste(family, measure, "one column"))
      }
    }

    if (family %in% c("binomial", "poisson")) {
      expected <- legacy.scalar.cv.loss(
        x, responses[[family]], beta, intercept, family, "deviance",
        n.fit = n.fit
      )
      actual <- picasso:::.picasso_cv_scalar_fold_loss(
        x, responses[[family]], beta, intercept, n.fit, family, "deviance",
        block.bytes = 1
      )
      expect_equal(actual, expected, tolerance = 1e-13,
                   info = paste(family, "without offset"))
    }
  }
})


test_that("scalar CV blocking handles edge shapes and a zero intercept", {
  x <- matrix(c(2, -1), nrow = 1L)
  beta <- Matrix::Matrix(matrix(c(0.25, -0.5), nrow = 2L))
  expected <- legacy.scalar.cv.loss(
    x, 1.5, beta, 0, "gaussian", "mae", n.fit = 1L
  )
  one.gemm <- picasso:::.picasso_cv_scalar_fold_loss(
    x, 1.5, beta, 0, 1L, "gaussian", "mae",
    block.bytes = 8 * (nrow(x) + ncol(x))
  )
  forced.block <- picasso:::.picasso_cv_scalar_fold_loss(
    x, 1.5, beta, 0, 1L, "gaussian", "mae", block.bytes = 1
  )

  expect_identical(one.gemm, expected)
  expect_identical(forced.block, expected)
})


test_that("cv.picasso routes scalar fold prediction through blocking", {
  set.seed(20260720)
  n <- 48L
  d <- 4L
  x <- matrix(rnorm(n * d), nrow = n)
  signal <- drop(x %*% c(0.7, -0.5, 0.3, 0))
  foldid <- rep(1:3, length.out = n)
  lambda <- c(0.22, 0.13, 0.075, 0.04)
  offset <- seq(-0.12, 0.12, length.out = n)
  cases <- list(
    gaussian = list(
      y = signal + 0.15 * sin(seq_len(n)), measure = "mae", offset = NULL
    ),
    binomial = list(
      y = factor(signal + offset > median(signal + offset)),
      measure = "deviance", offset = offset
    )
  )
  run.cv <- function(case, family) {
    arguments <- list(
      X = x, Y = case$y, family = family, lambda = lambda,
      foldid = foldid, type.measure = case$measure,
      standardize = FALSE, max.ite = 3000L
    )
    if (!is.null(case$offset)) arguments$offset <- case$offset
    do.call(cv.picasso, arguments)
  }
  expected <- lapply(names(cases), function(family) {
    run.cv(cases[[family]], family)
  })
  names(expected) <- names(cases)

  original <- picasso:::.picasso_cv_scalar_fold_loss
  calls <- 0L
  testthat::local_mocked_bindings(
    .picasso_cv_scalar_fold_loss = function(...) {
      calls <<- calls + 1L
      original(..., block.bytes = 1)
    },
    .package = "picasso"
  )
  actual <- lapply(names(cases), function(family) {
    run.cv(cases[[family]], family)
  })
  names(actual) <- names(cases)

  expect_identical(calls, 2L * 3L)
  for (family in names(cases)) {
    expect_identical(actual[[family]]$lambda, expected[[family]]$lambda,
                     info = family)
    expect_identical(actual[[family]]$nzero, expected[[family]]$nzero,
                     info = family)
    expect_identical(actual[[family]]$foldid, expected[[family]]$foldid,
                     info = family)
    for (field in c("cvm", "cvsd", "cvup", "cvlo", "lambda.min",
                    "lambda.1se")) {
      expect_equal(actual[[family]][[field]], expected[[family]][[field]],
                   tolerance = 1e-13, info = paste(family, field))
    }
  }
})


test_that("scalar CV block budget validation is explicit", {
  x <- matrix(1, nrow = 2L, ncol = 1L)
  beta <- Matrix::Matrix(matrix(0, nrow = 1L, ncol = 1L))
  for (bad.budget in list(0, -1, NA_real_, Inf, numeric())) {
    expect_error(
      picasso:::.picasso_cv_scalar_fold_loss(
        x, c(0, 1), beta, 0, 1L, "gaussian", "deviance",
        block.bytes = bad.budget
      ),
      "block.bytes"
    )
  }
})
