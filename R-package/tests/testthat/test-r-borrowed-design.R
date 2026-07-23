scalar_borrowed_fixture <- function() {
  set.seed(20260720)
  x <- matrix(rnorm(48L * 6L), nrow = 48L)
  eta <- 0.4 + x[, 1L] - 0.6 * x[, 2L]
  list(
    x = x,
    gaussian = eta + rnorm(48L, sd = 0.2),
    binomial = as.numeric(eta > median(eta)),
    poisson = as.numeric((seq_len(48L) %% 4L) + 1L),
    sqrtlasso = eta + rnorm(48L, sd = 0.3)
  )
}


test_that("double design preparation preserves no-op matrix storage", {
  skip_if_not(capabilities("profmem"))

  x <- matrix(as.double(seq_len(30L)), nrow = 10L)
  token <- tracemem(x)
  on.exit(untracemem(x), add = TRUE)
  prepared <- picasso:::.picasso_prepare_design(
    x, standardize = FALSE, center = TRUE
  )
  expect_identical(tracemem(prepared$xx), token)

  gaussian.x <- x + 0
  gaussian.token <- tracemem(gaussian.x)
  on.exit(untracemem(gaussian.x), add = TRUE)
  gaussian.prepared <- picasso:::.picasso_prepare_gaussian_design(
    gaussian.x, standardize = FALSE, intercept = TRUE
  )
  expect_identical(tracemem(gaussian.prepared$xx), gaussian.token)

  mock.scalar <- structure(
    list(beta = Matrix::Matrix(matrix(0, nrow = 3L, ncol = 1L))),
    class = "gaussian"
  )
  scalar.newdata <- x
  scalar.token <- tracemem(scalar.newdata)
  on.exit(untracemem(scalar.newdata), add = TRUE)
  scalar.prepared <- picasso:::.picasso_prediction_newdata(
    mock.scalar, scalar.newdata
  )
  expect_identical(tracemem(scalar.prepared), scalar.token)

  mock.multinomial <- structure(
    list(beta = list(Matrix::Matrix(matrix(0, nrow = 3L, ncol = 1L)))),
    class = "multinomial"
  )
  multinomial.newdata <- x
  multinomial.token <- tracemem(multinomial.newdata)
  on.exit(untracemem(multinomial.newdata), add = TRUE)
  multinomial.prepared <- picasso:::.picasso_multinomial_newdata(
    mock.multinomial, multinomial.newdata
  )
  expect_identical(tracemem(multinomial.prepared), multinomial.token)
})


test_that("integer design preparation converts once without mutating input", {
  x <- matrix(seq_len(30L), nrow = 10L)
  before <- serialize(x, NULL, version = 3L)

  prepared <- picasso:::.picasso_prepare_design(
    x, standardize = FALSE, center = TRUE
  )
  gaussian.prepared <- picasso:::.picasso_prepare_gaussian_design(
    x, standardize = FALSE, intercept = TRUE
  )

  expect_true(is.double(prepared$xx))
  expect_true(is.double(gaussian.prepared$xx))
  expect_identical(prepared$xx, matrix(as.double(x), nrow = nrow(x)))
  expect_identical(gaussian.prepared$xx, prepared$xx)
  expect_identical(serialize(x, NULL, version = 3L), before)
})


test_that("R scalar fits borrow X read-only across preprocessing modes", {
  fixture <- scalar_borrowed_fixture()
  path <- c(0.8, 0.4, 0.2)

  for (family in c("gaussian", "binomial", "poisson", "sqrtlasso")) {
    for (standardize in c(FALSE, TRUE)) {
      for (intercept in c(FALSE, TRUE)) {
        x <- fixture$x
        before <- serialize(x, NULL, version = 3L)
        fit <- picasso(
          x, fixture[[family]], family = family, lambda = path,
          standardize = standardize, intercept = intercept,
          type.gaussian = "naive", fast.mode = TRUE
        )
        expect_identical(
          serialize(x, NULL, version = 3L), before,
          info = paste(family, standardize, intercept)
        )
        expect_true(
          all(is.finite(as.matrix(fit$beta))),
          info = paste(family, standardize, intercept)
        )
        expect_true(
          all(is.finite(fit$intercept)),
          info = paste(family, standardize, intercept)
        )
      }
    }
  }
})


test_that("R multinomial fits keep caller X read-only across preprocessing", {
  fixture <- scalar_borrowed_fixture()
  response <- factor(rep(c("a", "b", "c"), length.out = nrow(fixture$x)))

  for (standardize in c(FALSE, TRUE)) {
    for (intercept in c(FALSE, TRUE)) {
      x <- fixture$x
      before <- serialize(x, NULL, version = 3L)
      fit <- picasso(
        x, response, family = "multinomial", lambda = 10,
        standardize = standardize, intercept = intercept,
        fast.mode = TRUE
      )
      expect_identical(
        serialize(x, NULL, version = 3L), before,
        info = paste("multinomial", standardize, intercept)
      )
      expect_true(
        all(vapply(fit$beta, function(value) {
          all(is.finite(as.matrix(value)))
        }, logical(1L))),
        info = paste("multinomial", standardize, intercept)
      )
    }
  }
})


test_that("materialized ALTREP-derived designs match ordinary doubles", {
  values <- as.double(seq_len(120L))
  x.altrep <- matrix(values, nrow = 40L, ncol = 3L)
  x.dense <- x.altrep + 0
  y <- 0.2 + x.dense[, 1L] / 50 - x.dense[, 2L] / 80
  path <- c(0.5, 0.2, 0.08)

  altrep.before <- serialize(x.altrep, NULL, version = 3L)
  fit.altrep <- picasso(
    x.altrep, y, family = "gaussian", lambda = path,
    standardize = FALSE, type.gaussian = "naive"
  )
  fit.dense <- picasso(
    x.dense, y, family = "gaussian", lambda = path,
    standardize = FALSE, type.gaussian = "naive"
  )

  expect_identical(serialize(x.altrep, NULL, version = 3L), altrep.before)
  expect_identical(as.matrix(fit.altrep$beta), as.matrix(fit.dense$beta))
  expect_identical(fit.altrep$intercept, fit.dense$intercept)
  expect_identical(fit.altrep$dev.ratio, fit.dense$dev.ratio)
})


test_that("integer and double public designs produce identical paths", {
  n <- 60L
  x.integer <- matrix(
    as.integer((seq_len(n * 4L) * 7L) %% 13L - 6L),
    nrow = n, ncol = 4L
  )
  x.double <- matrix(as.double(x.integer), nrow = n, ncol = 4L)
  eta <- 0.3 + x.double[, 1L] / 5 - x.double[, 2L] / 8
  responses <- list(
    gaussian = eta + sin(seq_len(n)) / 10,
    binomial = as.numeric(eta > median(eta)),
    poisson = as.numeric((seq_len(n) %% 4L) + 1L),
    sqrtlasso = eta + cos(seq_len(n)) / 8,
    multinomial = factor(rep(c("a", "b", "c"), length.out = n))
  )

  for (standardize in c(FALSE, TRUE)) {
    for (family in names(responses)) {
      common <- list(
        Y = responses[[family]], family = family, lambda = 10,
        standardize = standardize, intercept = TRUE, fast.mode = TRUE,
        type.gaussian = "naive"
      )
      integer.fit <- do.call(picasso, c(list(X = x.integer), common))
      double.fit <- do.call(picasso, c(list(X = x.double), common))
      label <- paste(family, "standardize", standardize)

      if (family == "multinomial") {
        expect_identical(
          lapply(integer.fit$beta, as.matrix),
          lapply(double.fit$beta, as.matrix),
          info = label
        )
        expect_identical(integer.fit$intercept, double.fit$intercept,
                         info = label)
      } else {
        expect_identical(as.matrix(integer.fit$beta),
                         as.matrix(double.fit$beta), info = label)
        expect_identical(integer.fit$intercept, double.fit$intercept,
                         info = label)
      }
      expect_identical(integer.fit$dev.ratio, double.fit$dev.ratio,
                       info = label)
      expect_identical(integer.fit$df, double.fit$df, info = label)
      expect_identical(integer.fit$status.code, double.fit$status.code,
                       info = label)
    }
  }
})
