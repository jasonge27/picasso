native_loss_fixture <- function() {
  set.seed(20260717)
  n <- 72L
  p <- 7L
  x <- matrix(rnorm(n * p), n, p)
  signal <- 0.7 * x[, 1L] - 0.45 * x[, 2L]
  offset <- seq(-0.25, 0.25, length.out = n)
  list(
    x = x,
    gaussian = signal + rnorm(n, sd = 0.6),
    binomial = factor(
      rbinom(n, 1L, plogis(-0.1 + signal + offset)), levels = 0:1
    ),
    poisson = rpois(n, exp(0.15 + 0.35 * signal + offset)),
    offset = offset
  )
}


expect_native_deviance_matches_prediction <- function(
    fit, x, y, family, offset = NULL, tolerance = 1e-9) {
  encoded <- if (family == "binomial") as.integer(y) - 1L else as.numeric(y)
  explicit <- picasso:::.picasso_fit_deviance(
    encoded, x, as.matrix(fit$beta), fit$intercept, family,
    offset = offset
  )
  expected.ratio <- if (fit$nulldev > 0) {
    pmax(0, pmin(1, 1 - explicit / fit$nulldev))
  } else {
    rep(0, fit$nlambda)
  }
  expect_equal(fit$dev.ratio, expected.ratio, tolerance = tolerance)
  invisible(explicit)
}


test_that("Gaussian native MSE replaces post-fit training predictions", {
  dat <- native_loss_fixture()
  lambda <- c(0.35, 0.22, 0.14)

  for (update in c("naive", "covariance")) {
    for (penalty in c("l1", "mcp", "scad")) {
      fit <- picasso(
        dat$x, dat$gaussian, family = "gaussian", method = penalty,
        type.gaussian = update, lambda = lambda, standardize = TRUE,
        prec = 1e-7, max.ite = 5000L
      )
      expect_native_deviance_matches_prediction(
        fit, dat$x, dat$gaussian, "gaussian", tolerance = 2e-10
      )
    }
  }
})


test_that("scalar ActNewton native losses match explicit predictions", {
  dat <- native_loss_fixture()
  lambda <- c(0.18, 0.11, 0.07)

  for (penalty in c("l1", "mcp", "scad")) {
    binomial <- picasso(
      dat$x, dat$binomial, family = "binomial", method = penalty,
      lambda = lambda, standardize = TRUE, offset = dat$offset,
      prec = 1e-6, max.ite = 5000L, lla.max.stages = 8L
    )
    binomial.explicit <- expect_native_deviance_matches_prediction(
      binomial, dat$x, dat$binomial, "binomial", dat$offset,
      tolerance = 2e-10
    )
    expect_equal(
      binomial$diagnostics$smooth.objective, binomial.explicit,
      tolerance = 2e-10, info = penalty
    )

    poisson <- picasso(
      dat$x, dat$poisson, family = "poisson", method = penalty,
      lambda = lambda, standardize = TRUE, offset = dat$offset,
      prec = 1e-6, max.ite = 5000L, lla.max.stages = 8L
    )
    poisson.explicit <- expect_native_deviance_matches_prediction(
      poisson, dat$x, dat$poisson, "poisson", dat$offset,
      tolerance = 2e-8
    )
    poisson.native <- pmax(
      0,
      2 * (
        poisson$diagnostics$smooth.objective +
          picasso:::.picasso_poisson_saturated_constant(dat$poisson)
      )
    )
    expect_equal(
      poisson.native, poisson.explicit,
      tolerance = 2e-8, info = penalty
    )

    sqrtlasso <- picasso(
      dat$x, dat$gaussian, family = "sqrtlasso", method = penalty,
      lambda = lambda, standardize = TRUE, prec = 1e-6,
      max.ite = 5000L, lla.max.stages = 8L
    )
    sqrt.explicit <- expect_native_deviance_matches_prediction(
      sqrtlasso, dat$x, dat$gaussian, "sqrtlasso", tolerance = 2e-8
    )
    expect_equal(
      0.5 * sqrtlasso$diagnostics$smooth.objective^2,
      sqrt.explicit, tolerance = 2e-8, info = penalty
    )
  }
})
