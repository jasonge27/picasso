test_that("link-scale loss helpers preserve ordinary-data values", {
  binomial.response <- c(0, 1, 1, 0)
  binomial.eta <- cbind(
    c(-2, -0.5, 1.25, 3),
    rep(0, 4)
  )
  expect_equal(
    picasso:::.picasso_binomial_nll_from_eta(
      binomial.response, binomial.eta
    ),
    c(1.1003803570355486, 0.6931471805599453),
    tolerance = 1e-15
  )

  poisson.response <- c(0, 1, 2, 4)
  poisson.eta <- cbind(
    c(-2, -0.5, log(2), log(3.5)),
    rep(0, 4)
  )
  expected.poisson <- c(
    0.13799575672366809,
    1.9657359027997265
  )
  expect_equal(
    picasso:::.picasso_poisson_deviance_from_eta(
      poisson.response, poisson.eta
    ),
    expected.poisson,
    tolerance = 1e-15
  )
  expect_equal(
    picasso:::.picasso_poisson_deviance_from_eta(
      poisson.response, poisson.eta, mu = exp(poisson.eta)
    ),
    expected.poisson,
    tolerance = 1e-15
  )

  multinomial.logits <- rbind(
    c(0.2, -0.4, 0.1),
    c(-1, 0.5, 0.3),
    c(2, -0.25, 0.75)
  )
  expect_equal(
    picasso:::.picasso_multinomial_nll_from_logits(
      c(1, 3, 2), multinomial.logits
    ),
    1.4640368467170874,
    tolerance = 1e-15
  )
})


test_that("link-scale losses retain finite extreme information", {
  expect_equal(
    picasso:::.picasso_binomial_nll_from_eta(
      c(0, 1), c(1000, -1000)
    ),
    1000,
    tolerance = 1e-14
  )
  expect_equal(
    picasso:::.picasso_poisson_deviance_from_eta(1, -1000),
    1998,
    tolerance = 1e-14
  )
  expect_equal(
    picasso:::.picasso_multinomial_nll_from_logits(
      2, matrix(c(0, -1000, -2), nrow = 1L)
    ),
    1000.126928011043,
    tolerance = 1e-13
  )

  common.shift <- 2^53
  shifted.logits <- matrix(
    c(common.shift, common.shift - 2, common.shift - 4), nrow = 1L
  )
  expect_identical(
    as.numeric(shifted.logits - common.shift), c(0, -2, -4)
  )
  expect_equal(
    picasso:::.picasso_multinomial_nll_from_logits(1, shifted.logits),
    0.14293162849989952,
    tolerance = 2e-15
  )

  expect_error(
    picasso:::.picasso_poisson_deviance_from_eta(1, 1000),
    "too large for a finite response mean"
  )
})


test_that("null and explicit path deviances use link-scale losses", {
  expect_equal(
    picasso:::.picasso_null_deviance(
      c(0, 1), "binomial", offset = c(1000, -1000), intercept = FALSE
    ),
    1000,
    tolerance = 1e-14
  )
  expect_equal(
    picasso:::.picasso_null_deviance(
      1, "poisson", offset = -1000, intercept = FALSE
    ),
    1998,
    tolerance = 1e-14
  )

  expect_equal(
    picasso:::.picasso_fit_deviance(
      c(0, 1), matrix(c(1, -1), ncol = 1L),
      matrix(1000, nrow = 1L), 0, "binomial"
    ),
    1000,
    tolerance = 1e-14
  )
  expect_equal(
    picasso:::.picasso_fit_deviance(
      1, matrix(1, nrow = 1L), matrix(-1000, nrow = 1L),
      0, "poisson"
    ),
    1998,
    tolerance = 1e-14
  )
})


test_that("public assessment routes deviance through link-scale losses", {
  binomial.object <- list(
    family = "binomial",
    levels = c("0", "1"),
    nlambda = 1L,
    lambda = 0.1,
    beta = matrix(1000, nrow = 1L),
    intercept = 0,
    offset.used = FALSE
  )
  binomial.assessment <- assess.picasso(
    binomial.object,
    matrix(c(1, -1), ncol = 1L),
    c(0, 1)
  )
  expect_equal(binomial.assessment$deviance, 1000, tolerance = 1e-14)

  poisson.object <- list(
    family = "poisson",
    nlambda = 1L,
    lambda = 0.1,
    beta = matrix(-1000, nrow = 1L),
    intercept = 0,
    offset.used = FALSE
  )
  poisson.assessment <- assess.picasso(
    poisson.object, matrix(1, nrow = 1L), 1
  )
  expect_equal(poisson.assessment$deviance, 1998, tolerance = 1e-14)

  multinomial.object <- list(
    family = "multinomial",
    levels = c("alpha", "beta", "gamma"),
    K = 3L,
    nlambda = 1L,
    lambda = 0.1,
    beta = lapply(seq_len(3L), function(index) matrix(0, nrow = 1L)),
    intercept = list(0, -1000, -2)
  )
  multinomial.assessment <- assess.picasso(
    multinomial.object, matrix(0, nrow = 1L), "beta"
  )
  expect_equal(
    multinomial.assessment$deviance,
    1000.126928011043,
    tolerance = 1e-13
  )
})
