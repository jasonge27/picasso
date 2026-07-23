gaussian_interface_fixture <- function() {
  index <- seq_len(72L)
  position <- (index - 36.5) / 36
  x <- cbind(
    linear = 2.5 + position,
    sine = sin(2 * pi * index / 72),
    cosine = cos(4 * pi * index / 72) + 0.15 * sin(0.37 * index)
  )
  y <- 4.25 + drop(x %*% c(1.3, -0.85, 0.55)) +
    0.03 * sin(0.41 * index)
  list(x = x, y = y)
}


gaussian_oracle_design <- function(x, standardize, intercept) {
  n <- nrow(x)
  xm <- if (intercept) colMeans(x) else rep(0.0, ncol(x))
  xx <- if (intercept) sweep(x, 2L, xm, `-`) else x
  multiplier <- rep(1.0, ncol(x))

  if (standardize) {
    divisor <- max(n - 1L, 1L)
    multiplier[] <- 0.0
    for (j in seq_len(ncol(x))) {
      column.norm <- sqrt(sum(xx[, j]^2) / divisor)
      if (column.norm > 0) {
        multiplier[j] <- 1 / column.norm
        xx[, j] <- xx[, j] * multiplier[j]
      }
    }
  }

  list(xx = xx, multiplier = multiplier)
}


test_that("Gaussian auto mode uses the calibrated guarded policy", {
  path.05 <- exp(seq(0, log(0.05), length.out = 8L))
  path.03 <- exp(seq(0, log(0.03), length.out = 8L))
  path.20 <- exp(seq(0, log(0.20), length.out = 8L))
  resolve <- picasso:::.picasso_resolve_gaussian_type

  expect_identical(resolve("auto", 120L, 120L, path.03), "covariance")
  expect_identical(resolve("auto", 250L, 250L, path.20), "covariance")
  expect_identical(resolve("auto", 1000L, 250L, path.05), "covariance")
  expect_identical(resolve("auto", 2000L, 250L, path.03), "naive")
  expect_identical(resolve("auto", 4000L, 250L, path.03), "covariance")
  expect_identical(resolve("auto", 10000L, 1025L, path.20), "naive")
  expect_identical(resolve("auto", 100L, 0L, path.20), "naive")
  expect_identical(resolve("auto", 1000L, 100L, path.05[1:7]), "naive")
  expect_identical(resolve("naive", 4000L, 250L, path.03), "naive")
  expect_identical(resolve("covariance", 10L, 250L, path.03), "covariance")
  expect_identical(resolve(NULL, 4000L, 250L, path.03), "naive")
  expect_error(resolve("other", 4000L, 250L, path.03), "auto, naive")
})


test_that("Gaussian public auto mode records and matches its resolved backend", {
  expect_identical(formals(picasso)$type.gaussian, "auto")
  set.seed(20260718)
  n <- 80L
  d <- 8L
  x <- matrix(rnorm(n * d), nrow = n)
  y <- 0.6 + x[, 1L] - 0.4 * x[, 2L] + rnorm(n)

  automatic <- picasso(x, y, family = "gaussian", nlambda = 8L)
  explicit <- picasso(
    x, y, family = "gaussian", nlambda = 8L,
    type.gaussian = "covariance"
  )
  expect_identical(automatic$type.gaussian.requested, "auto")
  expect_identical(automatic$type.gaussian, "covariance")
  expect_identical(automatic$alg, "actgd-covariance")
  expect_identical(as.matrix(automatic$beta), as.matrix(explicit$beta))
  expect_identical(automatic$intercept, explicit$intercept)
  expect_identical(automatic$lambda, explicit$lambda)

  short.path <- picasso(x, y, family = "gaussian", nlambda = 7L)
  expect_identical(short.path$type.gaussian, "naive")
  forced.naive <- picasso(
    x, y, family = "gaussian", nlambda = 8L, type.gaussian = "naive"
  )
  expect_identical(forced.naive$type.gaussian, "naive")
  legacy.null <- picasso(
    x, y, family = "gaussian", nlambda = 8L, type.gaussian = NULL
  )
  expect_identical(legacy.null$type.gaussian.requested, "naive")
  expect_identical(legacy.null$type.gaussian, "naive")

  # Backend selection is penalty-independent. This exact comparison guards
  # against nonconvex MCP/SCAD fits taking a different path merely because
  # the public default was resolved automatically.
  for (nonconvex.method in c("mcp", "scad")) {
    automatic.nonconvex <- picasso(
      x, y, family = "gaussian", method = nonconvex.method, nlambda = 8L
    )
    explicit.nonconvex <- picasso(
      x, y, family = "gaussian", method = nonconvex.method, nlambda = 8L,
      type.gaussian = "covariance"
    )
    expect_identical(
      as.matrix(automatic.nonconvex$beta),
      as.matrix(explicit.nonconvex$beta)
    )
    expect_identical(
      automatic.nonconvex$intercept, explicit.nonconvex$intercept
    )
  }
})


test_that("Gaussian cross-validation freezes the full-data auto backend", {
  set.seed(20260720)
  n <- 644L
  d <- 161L
  x <- matrix(rnorm(n * d), nrow = n)
  y <- x[, 1L] - 0.5 * x[, 2L] + rnorm(n)
  foldid <- rep(1:2, length.out = n)

  automatic <- cv.picasso(
    x, y, family = "gaussian", nlambda = 8L,
    lambda.min.ratio = 0.05, foldid = foldid
  )
  explicit <- cv.picasso(
    x, y, family = "gaussian", nlambda = 8L,
    lambda.min.ratio = 0.05, foldid = foldid,
    type.gaussian = "covariance"
  )
  expect_identical(automatic$picasso.fit$type.gaussian, "covariance")
  expect_identical(automatic$lambda, explicit$lambda)
  expect_identical(automatic$cvm, explicit$cvm)
  expect_identical(automatic$cvsd, explicit$cvsd)
})


test_that("Gaussian standardize and intercept combinations match lm", {
  fixture <- gaussian_interface_fixture()
  x <- fixture$x
  y <- fixture$y

  for (standardize in c(FALSE, TRUE)) {
    for (intercept in c(FALSE, TRUE)) {
      oracle.x <- if (intercept) cbind(`(Intercept)` = 1, x) else x
      oracle <- lm.fit(oracle.x, y)
      expected.intercept <- if (intercept) unname(oracle$coefficients[1L]) else 0
      expected.beta <- if (intercept) {
        unname(oracle$coefficients[-1L])
      } else {
        unname(oracle$coefficients)
      }

      for (type.gaussian in c("naive", "covariance")) {
        label <- paste(
          "standardize", standardize, "intercept", intercept, type.gaussian
        )
        fit <- picasso(
          x, y, family = "gaussian", method = "l1", lambda = 0,
          type.gaussian = type.gaussian,
          standardize = standardize, intercept = intercept,
          prec = 1e-11, max.ite = 50000L
        )

        expect_equal(
          as.numeric(fit$beta[, 1L]), expected.beta,
          tolerance = 5e-5, info = label
        )
        expect_equal(
          fit$intercept[1L], expected.intercept,
          tolerance = 2e-5, info = label
        )
        expect_equal(
          drop(x %*% as.numeric(fit$beta[, 1L]) + fit$intercept[1L]),
          unname(oracle$fitted.values), tolerance = 5e-5, info = label
        )
        if (!intercept) {
          expect_identical(fit$intercept, 0.0, info = label)
        }
      }
    }
  }
})


test_that("Gaussian lambda, null deviance, and KKT use the fitted model space", {
  fixture <- gaussian_interface_fixture()
  x <- fixture$x
  y <- fixture$y
  n <- nrow(x)

  for (standardize in c(FALSE, TRUE)) {
    for (intercept in c(FALSE, TRUE)) {
      label <- paste("standardize", standardize, "intercept", intercept)
      oracle <- gaussian_oracle_design(x, standardize, intercept)
      null.residual <- if (intercept) y - mean(y) else y
      expected.lambda.max <- max(abs(crossprod(
        oracle$xx, null.residual
      ))) / n
      expected.nulldev <- mean(null.residual^2) / 2

      fit <- picasso(
        x, y, family = "gaussian", method = "l1",
        nlambda = 3L, lambda.min.ratio = 0.4,
        standardize = standardize, intercept = intercept,
        prec = 1e-10, max.ite = 50000L
      )

      expect_equal(
        fit$lambda[1L], expected.lambda.max,
        tolerance = 1e-11, info = label
      )
      expect_equal(fit$nulldev, expected.nulldev,
                   tolerance = 1e-12, info = label)
      expect_equal(as.numeric(fit$beta[, 1L]), rep(0, ncol(x)),
                   tolerance = 1e-12, info = label)
      expect_equal(
        fit$intercept[1L], if (intercept) mean(y) else 0,
        tolerance = 1e-12, info = label
      )

      fitted <- x %*% as.matrix(fit$beta) +
        matrix(rep(fit$intercept, each = n), nrow = n)
      fit.deviance <- colMeans((y - fitted)^2) / 2
      expected.ratio <- if (expected.nulldev > 0) {
        pmax(0, pmin(1, 1 - fit.deviance / expected.nulldev))
      } else {
        rep(0, fit$nlambda)
      }
      expect_equal(fit$dev.ratio, expected.ratio,
                   tolerance = 1e-10, info = label)

      last <- fit$nlambda
      beta.raw <- as.numeric(fit$beta[, last])
      if (standardize) {
        beta.raw <- beta.raw / oracle$multiplier
      }
      residual <- y - drop(
        x %*% as.numeric(fit$beta[, last]) + fit$intercept[last]
      )
      correlation <- drop(crossprod(oracle$xx, residual)) / n
      active <- abs(beta.raw) > 1e-7
      if (any(active)) {
        expect_equal(
          unname(correlation[active]),
          fit$lambda[last] * sign(beta.raw[active]),
          tolerance = 3e-5, info = label
        )
      }
      expect_true(
        all(abs(correlation[!active]) <= fit$lambda[last] + 3e-5),
        info = label
      )
    }
  }
})


test_that("Gaussian constant columns and one-row data remain finite", {
  index <- seq_len(40L)
  varying <- (index - mean(index)) / 10
  x <- cbind(constant = 11, varying = varying, zero = 0)
  y <- 3.5 + 1.75 * varying + 0.02 * sin(index)

  for (standardize in c(FALSE, TRUE)) {
    for (type.gaussian in c("naive", "covariance")) {
      fit <- picasso(
        x, y, family = "gaussian", lambda = 0,
        standardize = standardize, intercept = TRUE,
        type.gaussian = type.gaussian,
        prec = 1e-11, max.ite = 50000L
      )
      oracle <- lm.fit(cbind(1, varying), y)
      expect_identical(as.numeric(fit$beta[c(1L, 3L), 1L]), c(0, 0))
      expect_equal(as.numeric(fit$beta[2L, 1L]),
                   unname(oracle$coefficients[2L]),
                   tolerance = 2e-6)
      expect_equal(fit$intercept[1L], unname(oracle$coefficients[1L]),
                   tolerance = 2e-6)
      expect_true(all(is.finite(c(as.matrix(fit$beta), fit$intercept))))
    }
  }

  one.x <- matrix(c(3, -2, 0), nrow = 1L)
  for (standardize in c(FALSE, TRUE)) {
    for (intercept in c(FALSE, TRUE)) {
      fit <- picasso(
        one.x, 7, family = "gaussian", lambda = 100,
        standardize = standardize, intercept = intercept
      )
      expect_true(all(is.finite(c(
        as.matrix(fit$beta), fit$intercept, fit$nulldev, fit$dev.ratio
      ))))
      expect_identical(fit$intercept, if (intercept) 7.0 else 0.0)
      expect_identical(as.numeric(fit$beta[, 1L]), rep(0, ncol(one.x)))
      expect_equal(
        fit$nulldev, if (intercept) 0 else 49 / 2, tolerance = 0
      )
      expect_identical(fit$dev.ratio, 0.0)
    }
  }

  constant.response <- picasso(
    cbind(varying, constant = 4), rep(6, length(varying)),
    family = "gaussian", nlambda = 3L,
    standardize = FALSE, intercept = TRUE
  )
  expect_true(all(is.finite(c(
    as.matrix(constant.response$beta), constant.response$intercept,
    constant.response$lambda, constant.response$dev.ratio
  ))))
  expect_identical(
    as.numeric(constant.response$beta),
    rep(0, nrow(constant.response$beta) * constant.response$nlambda)
  )
  expect_identical(constant.response$intercept, rep(6.0, 3L))
  expect_identical(constant.response$dev.ratio, rep(0.0, 3L))
})
