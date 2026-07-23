fast_mode_fixture <- function() {
  set.seed(20260716)
  n <- 72L
  p <- 8L
  x <- matrix(rnorm(n * p), n, p)
  signal <- drop(x[, seq_len(4L)] %*% c(0.9, -0.7, 0.5, -0.35))
  multinomial <- factor(
    max.col(cbind(signal, -0.6 * signal + x[, 5L], x[, 6L]) +
              matrix(rnorm(n * 3L, sd = 0.45), n, 3L)),
    levels = seq_len(3L)
  )
  list(
    x = x,
    gaussian = signal + 0.2 * sin(seq_len(n)),
    binomial = factor(as.integer(signal + 0.4 * x[, 7L] > median(signal))),
    poisson = as.integer(pmin(8, rpois(n, exp(pmin(1, 0.2 + 0.25 * signal))))),
    sqrtlasso = 1.5 + signal + 0.15 * cos(seq_len(n)),
    multinomial = multinomial,
    lambda = c(0.24, 0.14, 0.08)
  )
}


fast_mode_fit <- function(family, method = "l1", ..., fast.mode = FALSE,
                          prec = NULL) {
  fixture <- fast_mode_fixture()
  arguments <- list(
    fixture$x, fixture[[family]], family = family,
    lambda = fixture$lambda, method = method, standardize = TRUE,
    max.ite = 5000L, fast.mode = fast.mode, ...
  )
  if (!is.null(prec)) arguments$prec <- prec
  do.call(picasso, arguments)
}


expect_same_fast_mode_fit <- function(left, right, family) {
  expect_identical(left$lambda, right$lambda, info = family)
  expect_identical(left$nlambda, right$nlambda, info = family)
  expect_identical(left$intercept, right$intercept, info = family)
  if (family == "multinomial") {
    expect_identical(length(left$beta), length(right$beta), info = family)
    for (klass in seq_along(left$beta)) {
      expect_identical(
        as.matrix(left$beta[[klass]]), as.matrix(right$beta[[klass]]),
        info = paste(family, "class", klass)
      )
    }
  } else {
    expect_identical(as.matrix(left$beta), as.matrix(right$beta), info = family)
  }
}


test_that("fast mode defaults off and maps to the documented precision", {
  expect_identical(formals(picasso)$fast.mode, FALSE)
  expect_identical(formals(picasso)$prec, 1e-7)
  for (function.name in c(
      "picasso.gaussian", "picasso.logit", "picasso.poisson",
      "picasso.sqrtlasso", "picasso.multinomial")) {
    function.formals <- formals(get(function.name, envir = asNamespace("picasso")))
    expect_identical(function.formals$fast.mode, FALSE, info = function.name)
    expect_identical(function.formals$prec, 1e-7, info = function.name)
  }

  for (family in c(
      "gaussian", "binomial", "poisson", "sqrtlasso", "multinomial")) {
    high.default <- fast_mode_fit(family)
    high.explicit <- fast_mode_fit(family, prec = 1e-7)
    expect_false(high.default$fast.mode, info = family)
    expect_identical(high.default$prec, 1e-7, info = family)
    expect_same_fast_mode_fit(high.default, high.explicit, family)

    fast <- fast_mode_fit(family, fast.mode = TRUE)
    fast.precision <- if (family == "gaussian") {
      1e-7
    } else if (family == "poisson") {
      4e-4
    } else {
      1e-4
    }
    fast.reference <- fast_mode_fit(family, prec = fast.precision)
    expect_true(fast$fast.mode, info = family)
    expect_identical(fast$prec, fast.precision, info = family)
    expect_same_fast_mode_fit(fast, fast.reference, family)
  }

  for (method in c("mcp", "scad")) {
    fast <- fast_mode_fit("poisson", method = method, fast.mode = TRUE)
    reference <- fast_mode_fit("poisson", method = method, prec = 4e-4)
    expect_identical(fast$prec, 4e-4, info = method)
    expect_same_fast_mode_fit(fast, reference, "poisson")
  }
})


test_that("fast mode rejects ambiguous or invalid precision settings", {
  fixture <- fast_mode_fixture()
  for (bad in list(1, NA, c(TRUE, FALSE), "yes")) {
    expect_error(
      picasso(fixture$x, fixture$gaussian, fast.mode = bad),
      "fast.mode"
    )
  }
  expect_error(
    picasso(
      fixture$x, fixture$gaussian, fast.mode = TRUE, prec = 1e-6
    ),
    "fixes prec"
  )
  expect_error(
    picasso(
      fixture$x, fixture$gaussian, fast.mode = TRUE, prec = 1e-4
    ),
    "gaussian"
  )
  expect_s3_class(
    picasso(
      fixture$x, fixture$gaussian, lambda = fixture$lambda,
      fast.mode = TRUE, prec = 1e-7
    ),
    "gaussian"
  )
  expect_s3_class(
    picasso(
      fixture$x, fixture$binomial, family = "binomial",
      lambda = fixture$lambda, fast.mode = TRUE,
      prec = exp(log(1e-4))
    ),
    "logit"
  )
  expect_s3_class(
    picasso(
      fixture$x, fixture$poisson, family = "poisson",
      lambda = fixture$lambda, fast.mode = TRUE,
      prec = exp(log(4e-4))
    ),
    "poisson"
  )
})


test_that("cross-validation propagates fast mode to every fit", {
  fixture <- fast_mode_fixture()
  foldid <- rep(1:3, length.out = nrow(fixture$x))
  fast <- cv.picasso(
    fixture$x, fixture$gaussian, family = "gaussian",
    lambda = fixture$lambda, foldid = foldid, fast.mode = TRUE
  )
  reference <- cv.picasso(
    fixture$x, fixture$gaussian, family = "gaussian",
    lambda = fixture$lambda, foldid = foldid, prec = 1e-7
  )
  expect_true(fast$fast.mode)
  expect_identical(fast$prec, 1e-7)
  expect_true(fast$picasso.fit$fast.mode)
  expect_identical(fast$picasso.fit$prec, 1e-7)
  expect_identical(fast$cvm, reference$cvm)
  expect_identical(fast$cvsd, reference$cvsd)

  fast.class <- cv.picasso(
    fixture$x, fixture$binomial, family = "binomial",
    lambda = fixture$lambda, foldid = foldid, fast.mode = TRUE,
    type.measure = "class"
  )
  reference.class <- cv.picasso(
    fixture$x, fixture$binomial, family = "binomial",
    lambda = fixture$lambda, foldid = foldid, prec = 1e-4,
    type.measure = "class"
  )
  expect_identical(fast.class$cvm, reference.class$cvm)
  expect_identical(fast.class$cvsd, reference.class$cvsd)

  fast.poisson <- cv.picasso(
    fixture$x, fixture$poisson, family = "poisson",
    lambda = fixture$lambda, foldid = foldid, fast.mode = TRUE
  )
  reference.poisson <- cv.picasso(
    fixture$x, fixture$poisson, family = "poisson",
    lambda = fixture$lambda, foldid = foldid, prec = 4e-4
  )
  expect_identical(fast.poisson$prec, 4e-4)
  expect_identical(fast.poisson$cvm, reference.poisson$cvm)
  expect_identical(fast.poisson$cvsd, reference.poisson$cvsd)
})
