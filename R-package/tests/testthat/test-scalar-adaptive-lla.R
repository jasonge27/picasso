scalar_lla_fixture <- function() {
  n <- 180L
  d <- 12L
  index <- seq_len(n)
  x <- sapply(seq_len(d), function(feature) {
    sin(0.07 * index * feature) +
      0.25 * cos(0.13 * (index + feature + 1))
  })
  x <- scale(x)
  beta <- c(1, -0.8, 0.65, -0.5, rep(0, d - 4L))
  signal <- drop(x %*% beta)
  draw <- (0.6180339887498949 * index) %% 1
  list(
    x = x,
    binomial = as.integer(draw < plogis(-0.2 + signal)),
    poisson = floor(exp(0.15 + 0.28 * signal) + draw),
    sqrtlasso = 0.4 + signal + 0.35 * sin(0.31 * index)
  )
}


fit_scalar_lla <- function(family, method, lla.max.stages = 3L, ...) {
  fixture <- scalar_lla_fixture()
  picasso(
    fixture$x, fixture[[family]], family = family, method = method,
    nlambda = 4L, lambda.min.ratio = 0.37, prec = 1e-5,
    max.ite = 1000L, lla.max.stages = lla.max.stages, ...
  )
}


test_that("scalar adaptive-LLA native routines are registered", {
  registered <- getDLLRegisteredRoutines(
    getLoadedDLLs()[["picasso"]]
  )$.Call
  expect_identical(registered[["picasso_logit_lla_call"]]$numParameters, 14L)
  expect_identical(
    registered[["picasso_poisson_lla_call"]]$numParameters, 14L
  )
  expect_identical(
    registered[["picasso_sqrtlasso_lla_call"]]$numParameters, 13L
  )
  expect_identical(formals(picasso)$lla.max.stages, 3L)
  expect_identical(formals(picasso:::picasso.logit)$lla.max.stages, 3L)
  expect_identical(formals(picasso:::picasso.poisson)$lla.max.stages, 3L)
  expect_identical(formals(picasso:::picasso.sqrtlasso)$lla.max.stages, 3L)
})


test_that("R model interfaces reject non-finite gamma values", {
  fixture <- scalar_lla_fixture()
  for (family in c("gaussian", "binomial", "poisson", "sqrtlasso")) {
    for (method in c("mcp", "scad")) {
      for (bad.gamma in c(Inf, -Inf, NaN, NA_real_)) {
        expect_error(
          picasso(
            fixture$x, fixture[[if (family == "gaussian") "sqrtlasso" else family]],
            family = family, method = method, gamma = bad.gamma,
            lambda = c(0.3, 0.15)
          ),
          "single finite numeric", info = paste(family, method, bad.gamma)
        )
      }
    }
  }
})


test_that("binomial, poisson, and sqrt-lasso use certified adaptive LLA", {
  for (family in c("binomial", "poisson", "sqrtlasso")) {
    l1 <- fit_scalar_lla(family, "l1")
    expect_identical(l1$status, "completed", info = family)
    expect_identical(l1$status.code, 0L, info = family)
    expect_true(all(l1$diagnostics$lla.stages == 1L), info = family)
    expect_lte(max(l1$diagnostics$kkt), 1e-5)
    expect_null(l1$failure, info = family)

    for (method in c("mcp", "scad")) {
      capped <- fit_scalar_lla(family, method, 3L)
      adaptive <- fit_scalar_lla(family, method, 25L)
      label <- paste(family, method)

      expect_identical(capped$status, "lla_stationarity_limit", info = label)
      expect_identical(capped$status.code, 10L, info = label)
      expect_true(all(capped$diagnostics$lla.stages == 3L), info = label)
      expect_null(capped$failure, info = label)

      expect_identical(adaptive$status, "completed", info = label)
      expect_identical(adaptive$status.code, 0L, info = label)
      expect_true(all(adaptive$diagnostics$lla.stages >= 3L), info = label)
      expect_true(all(adaptive$diagnostics$lla.stages <= 25L), info = label)
      expect_lte(max(adaptive$diagnostics$kkt), 1e-5)
      expect_lte(max(adaptive$diagnostics$stationarity), 1e-5)
      expect_lte(
        max(adaptive$diagnostics$objective - capped$diagnostics$objective),
        1e-7
      )
      expect_null(adaptive$failure, info = label)
      expect_identical(adaptive$lla.max.stages, 25L, info = label)
    }
  }
})


test_that("scalar adaptive LLA validates budgets and preserves failures", {
  fixture <- scalar_lla_fixture()
  for (family in c("binomial", "poisson", "sqrtlasso")) {
    for (bad in list(2L, 3.5, Inf, NA_real_)) {
      expect_error(
        picasso(
          fixture$x, fixture[[family]], family = family, method = "mcp",
          nlambda = 1L, lla.max.stages = bad
        ),
        "lla.max.stages",
        info = family
      )
    }
  }

  lambda.max <- max(abs(crossprod(
    fixture$x, fixture$binomial / nrow(fixture$x)
  )))
  partial <- NULL
  expect_warning(
    partial <- picasso(
      fixture$x, fixture$binomial,
      family = "binomial", method = "mcp",
      lambda = c(lambda.max, 0.2 * lambda.max),
      prec = 1e-7, max.ite = 1L
    ),
    "successful 1/2-lambda prefix"
  )
  expect_identical(partial$status, "subproblem_failed")
  expect_identical(partial$status.code, 3L)
  expect_identical(partial$nlambda, 1L)
  expect_identical(partial$failure$lambda.index, 2L)
  expect_identical(partial$failure$stage, 1L)
  expect_true(is.na(partial$failure$diagnostics$iterations))
  expect_true(is.na(partial$failure$diagnostics$nonzero))

  expect_error(
    picasso(
      fixture$x, fixture$binomial,
      family = "binomial", method = "mcp",
      lambda = 0.2 * lambda.max, prec = 1e-7, max.ite = 1L
    ),
    "before completing a lambda value"
  )
})


test_that("scalar R bridges reject malformed vectors before native access", {
  fixture <- scalar_lla_fixture()
  n <- nrow(fixture$x)
  d <- ncol(fixture$x)
  lambda <- c(0.3, 0.15)
  call_logit <- function(y = as.double(fixture$binomial),
                         x = as.double(fixture$x),
                         offset = numeric(n)) {
    .Call(
      "picasso_logit_lla_call", y, x, as.integer(n), as.integer(d),
      as.double(lambda), 2L, 3.0, 1000L, 1e-5, 1L, 1L, -1L,
      offset, 3L, PACKAGE = "picasso"
    )
  }

  expect_error(call_logit(offset = 0), "offset.*length n")
  expect_error(call_logit(y = as.integer(fixture$binomial)),
               "response must be a double")
  expect_error(call_logit(x = as.double(fixture$x)[-1L]),
               "design.*n\\*d")

  call_gaussian <- function(symbol, y = as.double(fixture$sqrtlasso),
                            x = as.double(fixture$x),
                            lambda.values = as.double(lambda)) {
    .Call(
      symbol, y, x, as.integer(n), as.integer(d), lambda.values, 2L,
      3.0, 1000L, 1e-5, 1L, 1L, -1L, PACKAGE = "picasso"
    )
  }
  for (symbol in c("picasso_gaussian_naive_call",
                   "picasso_gaussian_cov_call")) {
    expect_error(call_gaussian(symbol, y = as.integer(fixture$sqrtlasso)),
                 "response must be a double")
    expect_error(call_gaussian(symbol, x = as.double(fixture$x)[-1L]),
                 "design.*n\\*d")
    expect_error(call_gaussian(symbol, lambda.values = as.integer(lambda)),
                 "lambda must be a double")
  }

  call_legacy_glm <- function(symbol, y, x = as.double(fixture$x),
                              offset = numeric(n)) {
    .Call(
      symbol, as.double(y), x, as.integer(n), as.integer(d),
      as.double(lambda), 2L, 3.0, 1000L, 1e-5, 1L, 1L, -1L,
      offset, PACKAGE = "picasso"
    )
  }
  expect_error(
    call_legacy_glm("picasso_logit_call", fixture$binomial,
                    x = as.double(fixture$x)[-1L]),
    "design.*n\\*d"
  )
  expect_error(
    call_legacy_glm("picasso_poisson_call", fixture$poisson, offset = 0),
    "offset.*length n"
  )
  expect_error(
    .Call(
      "picasso_sqrtlasso_call", as.double(fixture$sqrtlasso),
      as.double(fixture$x)[-1L], as.integer(n), as.integer(d),
      as.double(lambda), 2L, 3.0, 1000L, 1e-5, 1L, 1L, -1L,
      PACKAGE = "picasso"
    ),
    "design.*n\\*d"
  )

  expect_error(
    .Call(
      "picasso_standardize_call", as.integer(fixture$x),
      as.integer(n), as.integer(d), PACKAGE = "picasso"
    ),
    "design must be a double"
  )
  expect_error(
    .Call(
      "picasso_standardize_call", as.double(fixture$x)[-1L],
      as.integer(n), as.integer(d), PACKAGE = "picasso"
    ),
    "design.*n\\*d"
  )

  integer.x <- matrix(as.integer(round(10 * fixture$x)), nrow = n)
  integer.responses <- list(
    binomial = as.integer(fixture$binomial),
    poisson = as.integer(fixture$poisson),
    sqrtlasso = as.integer(round(fixture$sqrtlasso))
  )
  for (family in names(integer.responses)) {
    fit <- picasso(
      integer.x, integer.responses[[family]], family = family,
      method = "l1", lambda = c(10L, 5L), max.ite = 1000L
    )
    expect_identical(fit$nlambda, 2L, info = family)
    expect_true(all(is.finite(as.matrix(fit$beta))), info = family)
  }
})


test_that("scalar no-intercept standardization preserves the origin", {
  fixture <- scalar_lla_fixture()
  n <- nrow(fixture$x)
  shifted.x <- sweep(
    fixture$x, 2L, seq(0.5, 2.0, length.out = ncol(fixture$x)), `+`
  )
  scaled <- picasso:::.picasso_prepare_design(
    shifted.x, standardize = TRUE, center = FALSE
  )
  expect_equal(as.numeric(scaled$xm), rep(0, ncol(shifted.x)), tolerance = 0)

  for (family in c("binomial", "poisson", "sqrtlasso")) {
    response <- fixture[[family]]
    fit <- picasso(
      shifted.x, response, family = family, method = "l1",
      nlambda = 3L, lambda.min.ratio = 0.5,
      standardize = TRUE, intercept = FALSE,
      prec = 1e-5, max.ite = 1000L
    )
    expect_equal(fit$intercept, rep(0, fit$nlambda), tolerance = 0,
                 info = family)
    expect_identical(fit$df[1L], 0L, info = family)

    if (family == "binomial") {
      residual0 <- response - 0.5
      expected.null <- log(2)
      expected.lambda <- max(abs(crossprod(
        scaled$xx, residual0 / n
      )))
    } else if (family == "poisson") {
      residual0 <- response - 1
      expected.null <- picasso:::.picasso_poisson_dev(
        response, rep(1, n)
      )
      expected.lambda <- max(abs(crossprod(
        scaled$xx, residual0 / n
      )))
    } else {
      residual0 <- response
      scale0 <- sqrt(mean(residual0^2))
      expected.null <- mean(residual0^2) / 2
      expected.lambda <- max(abs(crossprod(
        scaled$xx, residual0 / n
      ))) / scale0
    }
    expect_equal(fit$lambda[1L], expected.lambda, tolerance = 1e-10,
                 info = family)
    expect_equal(fit$nulldev, expected.null, tolerance = 1e-10,
                 info = family)
  }

  one.row <- picasso:::.picasso_prepare_design(
    matrix(3, nrow = 1L), standardize = TRUE, center = FALSE
  )
  expect_true(all(is.finite(one.row$xx)))
  expect_true(all(is.finite(one.row$xinvc.vec)))
})


test_that("scalar offsets are validated and reflected in binomial deviance", {
  fixture <- scalar_lla_fixture()
  n <- nrow(fixture$x)
  for (family in c("binomial", "poisson")) {
    expect_error(
      picasso(fixture$x, fixture[[family]], family = family, offset = 0),
      "offset.*length"
    )
    bad.offset <- numeric(n)
    bad.offset[3L] <- Inf
    expect_error(
      picasso(
        fixture$x, fixture[[family]], family = family,
        offset = bad.offset
      ),
      "finite numeric"
    )
  }

  offset <- seq(-1.1, 0.9, length.out = n)
  fit <- picasso(
    fixture$x, fixture$binomial, family = "binomial", method = "l1",
    nlambda = 3L, lambda.min.ratio = 0.5, offset = offset,
    prec = 1e-5, max.ite = 1000L
  )
  target <- sum(fixture$binomial)
  shift <- uniroot(
    function(value) sum(plogis(offset + value)) - target,
    c(-40 - max(offset), 40 - min(offset))
  )$root
  null.eta <- offset + shift
  null.deviance <- mean(
    log1p(exp(null.eta)) - fixture$binomial * null.eta
  )
  eta <- fixture$x %*% as.matrix(fit$beta) +
    matrix(rep(fit$intercept, each = n), nrow = n) + offset
  fit.deviance <- vapply(seq_len(fit$nlambda), function(index) {
    mean(log1p(exp(eta[, index])) - fixture$binomial * eta[, index])
  }, numeric(1))
  expect_equal(fit$nulldev, null.deviance, tolerance = 1e-8)
  expect_equal(
    fit$dev.ratio,
    pmax(0, pmin(1, 1 - fit.deviance / null.deviance)),
    tolerance = 1e-8
  )
})


test_that("binomial and Poisson prediction applies validated newoffset", {
  fixture <- scalar_lla_fixture()
  n <- nrow(fixture$x)
  rows <- seq_len(11L)
  newx <- fixture$x[rows, , drop = FALSE]
  newoffset <- seq(-0.7, 0.6, length.out = length(rows))

  for (family in c("binomial", "poisson")) {
    training.offset <- if (family == "binomial") numeric(n) else
      seq(-0.4, 0.5, length.out = n)
    fit <- picasso(
      fixture$x, fixture[[family]], family = family, method = "l1",
      lambda = c(0.2, 0.1), offset = training.offset,
      prec = 1e-5, max.ite = 1000L
    )
    expect_true(isTRUE(fit$offset.used), info = family)
    expect_error(
      predict(fit, newx, lambda.idx = 1L, type = "link"),
      "newoffset must be provided", info = family
    )
    expect_error(
      predict(
        fit, newx, lambda.idx = 1L, type = "link", newoffset = 0
      ),
      "length 11", info = family
    )
    bad.offset <- newoffset
    bad.offset[2L] <- Inf
    expect_error(
      predict(
        fit, newx, lambda.idx = 1L, type = "link",
        newoffset = bad.offset
      ),
      "finite numeric vector", info = family
    )

    expected.link <- drop(
      newx %*% as.matrix(fit$beta)[, 1L, drop = FALSE]
    ) + fit$intercept[1L] + newoffset
    actual.link <- drop(predict(
      fit, newx, lambda.idx = 1L, type = "link",
      newoffset = newoffset
    ))
    expect_equal(actual.link, expected.link, tolerance = 1e-12, info = family)
    expect_equal(
      drop(predict(
        fit, newx, s = fit$lambda[1L], type = "link",
        newoffset = newoffset
      )),
      expected.link,
      tolerance = 1e-12,
      info = paste(family, "s path")
    )

    expected.response <- if (family == "binomial") {
      plogis(expected.link)
    } else {
      exp(expected.link)
    }
    expect_equal(
      drop(predict(
        fit, newx, lambda.idx = 1L, type = "response",
        newoffset = newoffset
      )),
      expected.response,
      tolerance = 1e-12,
      info = family
    )
    if (family == "poisson") {
      overflow.offset <- rep(1000, nrow(newx))
      expect_error(
        predict(
          fit, newx, lambda.idx = 1L, type = "response",
          newoffset = overflow.offset
        ),
        "too large for a finite response mean"
      )
      expect_error(
        assess.picasso(fit, newx, fixture[[family]][rows],
                       newoffset = overflow.offset),
        "too large for a finite response mean"
      )
    }
    newy <- fixture[[family]][rows]
    expect_error(
      assess.picasso(fit, newx, newy),
      "newoffset must be provided", info = family
    )
    assessment <- assess.picasso(
      fit, newx, newy, newoffset = newoffset
    )
    expected.deviance <- if (family == "binomial") {
      -mean(newy * log(plogis(expected.link)) +
              (1 - newy) * log1p(-plogis(expected.link)))
    } else {
      picasso:::.picasso_poisson_dev(newy, exp(expected.link))
    }
    expect_equal(
      assessment$deviance[1L], expected.deviance,
      tolerance = 1e-12, info = paste(family, "assessment")
    )
    if (family == "binomial") {
      expect_identical(
        drop(predict(
          fit, newx, lambda.idx = 1L, type = "class",
          newoffset = newoffset
        )),
        as.integer(expected.link > 0)
      )
      expect_error(
        confusion.picasso(fit, newx, newy, lambda.idx = 1L),
        "newoffset must be provided"
      )
      observed.confusion <- confusion.picasso(
        fit, newx, newy, lambda.idx = 1L, newoffset = newoffset
      )[[1L]]
      expected.confusion <- table(
        predicted = as.integer(expected.link > 0), actual = newy
      )
      expect_identical(observed.confusion, expected.confusion)
    }

    no.offset.fit <- picasso(
      fixture$x, fixture[[family]], family = family, method = "l1",
      lambda = c(0.2, 0.1), prec = 1e-5, max.ite = 1000L
    )
    expect_false(isTRUE(no.offset.fit$offset.used), info = family)
    default.link <- drop(predict(
      no.offset.fit, newx, lambda.idx = 1L, type = "link"
    ))
    expect_equal(
      drop(predict(
        no.offset.fit, newx, lambda.idx = 1L, type = "link",
        newoffset = numeric(length(rows))
      )),
      default.link,
      tolerance = 0,
      info = family
    )
    expect_equal(
      drop(predict(
        no.offset.fit, newx, lambda.idx = 1L, type = "link",
        newoffset = newoffset
      )),
      default.link + newoffset,
      tolerance = 1e-12,
      info = family
    )
  }
})


test_that("binomial response and class prediction support multiple lambdas", {
  fixture <- scalar_lla_fixture()
  rows <- seq_len(12L)
  newx <- fixture$x[rows, , drop = FALSE]
  fit <- picasso(
    fixture$x, fixture$binomial,
    family = "binomial", method = "l1",
    lambda = c(0.5, 0.35, 0.25), prec = 1e-7, max.ite = 1000L
  )

  link <- predict(fit, newx, lambda.idx = 1:2, type = "link")
  response <- predict(fit, newx, lambda.idx = 1:2, type = "response")
  class <- predict(fit, newx, lambda.idx = 1:2, type = "class")

  expect_equal(response, stats::plogis(link), tolerance = 0)
  expect_identical(unname(class), unname(matrix(
    as.integer(link > 0), nrow = nrow(link), dimnames = dimnames(link)
  )))
})


test_that("binomial confusion has fixed predicted-by-observed axes", {
  fixture <- scalar_lla_fixture()
  rows <- which(fixture$binomial == 1L)[seq_len(8L)]
  fit <- picasso(
    fixture$x, fixture$binomial,
    family = "binomial", method = "l1", lambda = c(0.5, 0.25)
  )
  confusion <- confusion.picasso(
    fit, fixture$x[rows, , drop = FALSE], fixture$binomial[rows],
    lambda.idx = 1L
  )[[1L]]

  expect_identical(dim(confusion), c(2L, 2L))
  expect_identical(rownames(confusion), c("0", "1"))
  expect_identical(colnames(confusion), c("0", "1"))
  expect_identical(sum(confusion), length(rows))
})


test_that("binomial factor CV uses the fitted zero-one class map", {
  fixture <- scalar_lla_fixture()
  rows <- seq_len(90L)
  x <- fixture$x[rows, , drop = FALSE]
  y <- fixture$binomial[rows]
  y.factor <- factor(ifelse(y == 0L, "no", "yes"),
                     levels = c("no", "yes"))
  foldid <- ((rows - 1L) %% 3L) + 1L

  for (measure in c("class", "deviance")) {
    numeric.cv <- cv.picasso(
      x, y, family = "binomial", method = "l1",
      lambda = c(0.2, 0.1), foldid = foldid,
      type.measure = measure, prec = 1e-5, max.ite = 1000L
    )
    factor.cv <- cv.picasso(
      x, y.factor, family = "binomial", method = "l1",
      lambda = c(0.2, 0.1), foldid = foldid,
      type.measure = measure, prec = 1e-5, max.ite = 1000L
    )
    expect_equal(factor.cv$cvm, numeric.cv$cvm, tolerance = 1e-12,
                 info = measure)
    expect_equal(factor.cv$cvsd, numeric.cv$cvsd, tolerance = 1e-12,
                 info = measure)
  }

  reversed.factor <- factor(
    ifelse(y == 0L, "no", "yes"), levels = c("yes", "no")
  )
  reversed.codes <- as.numeric(reversed.factor) - 1.0
  reversed.fit <- picasso(
    x, reversed.factor, family = "binomial", method = "l1",
    lambda = c(0.2, 0.1), prec = 1e-5, max.ite = 1000L
  )
  expect_identical(reversed.fit$levels, c("yes", "no"))
  expect_equal(
    assess.picasso(reversed.fit, x, reversed.factor)$deviance,
    assess.picasso(reversed.fit, x, reversed.codes)$deviance,
    tolerance = 0
  )
  reversed.cv <- cv.picasso(
    x, reversed.factor, family = "binomial", method = "l1",
    lambda = c(0.2, 0.1), foldid = foldid,
    type.measure = "deviance", prec = 1e-5, max.ite = 1000L
  )
  encoded.cv <- cv.picasso(
    x, reversed.codes, family = "binomial", method = "l1",
    lambda = c(0.2, 0.1), foldid = foldid,
    type.measure = "deviance", prec = 1e-5, max.ite = 1000L
  )
  expect_equal(reversed.cv$cvm, encoded.cv$cvm, tolerance = 1e-12)
})


test_that("GLM CV MSE and MAE use the response scale", {
  fixture <- scalar_lla_fixture()
  rows <- seq_len(60L)
  x <- fixture$x[rows, , drop = FALSE]
  foldid <- ((rows - 1L) %% 3L) + 1L
  lambda <- c(0.3, 0.15)

  for (configuration in list(
      list(family = "binomial", measure = "mse"),
      list(family = "poisson", measure = "mae"))) {
    family <- configuration$family
    measure <- configuration$measure
    y <- fixture[[family]][rows]
    cv <- cv.picasso(
      x, y, family = family, method = "l1", lambda = lambda,
      foldid = foldid, type.measure = measure,
      prec = 1e-5, max.ite = 1000L
    )
    fold.loss <- matrix(NA_real_, nrow = 3L, ncol = length(lambda))
    for (fold in seq_len(3L)) {
      train <- foldid != fold
      test <- !train
      fold.fit <- picasso(
        x[train, , drop = FALSE], y[train], family = family,
        method = "l1", lambda = lambda, prec = 1e-5, max.ite = 1000L
      )
      eta <- x[test, , drop = FALSE] %*% as.matrix(fold.fit$beta) +
        matrix(rep(fold.fit$intercept, each = sum(test)), nrow = sum(test))
      response <- if (family == "binomial") {
        stats::plogis(eta)
      } else {
        picasso:::.picasso_poisson_mean(eta)
      }
      error <- y[test] - response
      fold.loss[fold, ] <- if (measure == "mse") {
        colMeans(error^2)
      } else {
        colMeans(abs(error))
      }
    }
    expect_equal(cv$cvm, colMeans(fold.loss), tolerance = 1e-12,
                 info = paste(family, measure))
  }

  expect_error(
    cv.picasso(
      x, fixture$sqrtlasso[rows], family = "gaussian",
      lambda = lambda, foldid = foldid, type.measure = "class"
    ),
    "only for binomial or multinomial"
  )
})


test_that("binomial assessment maps fitted factor labels to zero and one", {
  fixture <- scalar_lla_fixture()
  rows <- seq_len(90L)
  x <- fixture$x[rows, , drop = FALSE]
  y.numeric <- fixture$binomial[rows]
  y.factor <- factor(ifelse(y.numeric == 0L, "no", "yes"),
                     levels = c("no", "yes"))
  fit <- picasso(
    x, y.factor, family = "binomial", method = "l1",
    lambda = c(0.2, 0.1), prec = 1e-5, max.ite = 1000L
  )

  numeric.assessment <- assess.picasso(fit, x, y.numeric)
  factor.assessment <- assess.picasso(fit, x, y.factor)
  expect_equal(factor.assessment$deviance, numeric.assessment$deviance,
               tolerance = 0)
  expect_equal(factor.assessment$class, numeric.assessment$class,
               tolerance = 0)
  expect_identical(
    confusion.picasso(fit, x, y.factor),
    confusion.picasso(fit, x, y.numeric)
  )
  expect_error(
    assess.picasso(fit, x, rep("unknown", nrow(x))),
    "absent from the fitted model"
  )

  legacy.fit <- fit
  legacy.fit$levels <- NULL
  expect_equal(
    assess.picasso(legacy.fit, x, y.numeric)$deviance,
    numeric.assessment$deviance,
    tolerance = 0
  )
  expect_error(
    assess.picasso(legacy.fit, x, y.factor),
    "legacy binomial fit"
  )
  expect_error(
    assess.picasso(fit, x, y.factor[-1L]),
    "length 90"
  )
})


test_that("constant-count Poisson paths have finite zero deviance ratios", {
  fixture <- scalar_lla_fixture()
  rows <- seq_len(60L)
  fit <- picasso(
    fixture$x[rows, , drop = FALSE], rep(1, length(rows)),
    family = "poisson", method = "l1", lambda = c(0.2, 0.1),
    prec = 1e-5, max.ite = 1000L
  )
  expect_equal(fit$nulldev, 0, tolerance = 1e-14)
  expect_identical(fit$dev.ratio, rep(0.0, fit$nlambda))
  expect_true(all(is.finite(fit$dev.ratio)))
  expect_error(
    assess.picasso(fit, fixture$x[rows, , drop = FALSE], rep(1, 30L)),
    "length 60"
  )
  expect_error(
    assess.picasso(fit, fixture$x[rows, , drop = FALSE],
                   c(-1, rep(1, 59L))),
    "nonnegative"
  )
})


test_that("automatic binomial CV stratifies rare classes", {
  fixture <- scalar_lla_fixture()
  rows <- seq_len(24L)
  x <- fixture$x[rows, , drop = FALSE]
  y <- c(rep(0, 22L), 1, 1)

  set.seed(20260716)
  fit <- cv.picasso(
    x, y, family = "binomial", method = "l1",
    lambda = c(1.0, 0.5), nfolds = 4L,
    type.measure = "class", prec = 1e-5, max.ite = 1000L
  )
  for (fold in seq_len(4L)) {
    expect_equal(length(unique(y[fit$foldid != fold])), 2L)
  }

  expect_error(
    cv.picasso(
      x, c(rep(0, 23L), 1), family = "binomial", method = "l1",
      lambda = c(1.0, 0.5), nfolds = 4L,
      type.measure = "class", prec = 1e-5, max.ite = 1000L
    ),
    "at least two observations per class"
  )

  expect_error(
    cv.picasso(
      x, y, family = "binomial", method = "l1",
      lambda = c(1.0, 0.5),
      foldid = c(rep(1L, 22L), 2L, 2L),
      type.measure = "class", prec = 1e-5, max.ite = 1000L
    ),
    "missing binomial class"
  )
})


test_that("cross-validation rejects scalar hard failures", {
  fixture <- scalar_lla_fixture()
  lambda.max <- max(abs(crossprod(
    fixture$x, fixture$binomial / nrow(fixture$x)
  )))
  expect_error(
    suppressWarnings(cv.picasso(
      fixture$x, fixture$binomial,
      family = "binomial", method = "mcp", nfolds = 2L,
      lambda = c(lambda.max, 0.2 * lambda.max),
      prec = 1e-7, max.ite = 1L
    )),
    "Full-data binomial fit stopped"
  )

  expect_silent(picasso:::.picasso_cv_require_usable_status(
    list(status.code = 10L, status = "lla_stationarity_limit"),
    "binomial fit in fold 1"
  ))
  expect_error(
    picasso:::.picasso_cv_require_usable_status(
      list(status.code = 3L, status = "subproblem_failed"),
      "binomial fit in fold 1"
    ),
    "requires a usable fitted path"
  )

  split.y <- rep(c(0L, 1L), each = 12L)
  split.x <- fixture$x[seq_along(split.y), , drop = FALSE]
  expect_error(
    cv.picasso(
      split.x, split.y, family = "binomial", method = "l1",
      nlambda = 2L, lambda.min.ratio = 0.5,
      foldid = split.y + 1L
    ),
    "missing binomial class"
  )
})


test_that("stage budget reaches CV fits and does not alter Gaussian", {
  fixture <- scalar_lla_fixture()
  set.seed(31)
  cv <- cv.picasso(
    fixture$x, fixture$binomial,
    family = "binomial", method = "mcp", nfolds = 2L,
    nlambda = 3L, lambda.min.ratio = 0.5,
    prec = 1e-5, max.ite = 1000L, lla.max.stages = 8L
  )
  expect_identical(cv$picasso.fit$lla.max.stages, 8L)
  expect_true(cv$picasso.fit$status.code %in% c(0L, 10L))

  gaussian3 <- picasso(
    fixture$x, fixture$sqrtlasso,
    family = "gaussian", method = "mcp", nlambda = 4L,
    lla.max.stages = 3L
  )
  gaussian9 <- picasso(
    fixture$x, fixture$sqrtlasso,
    family = "gaussian", method = "mcp", nlambda = 4L,
    lla.max.stages = 9L
  )
  expect_equal(gaussian9$lambda, gaussian3$lambda, tolerance = 0)
  expect_equal(
    as.matrix(gaussian9$beta), as.matrix(gaussian3$beta), tolerance = 0
  )
  expect_equal(gaussian9$intercept, gaussian3$intercept, tolerance = 0)
})


test_that("penalty controls and algorithm metadata are strict and truthful", {
  fixture <- scalar_lla_fixture()

  expect_error(
    picasso:::.picasso_method_flag("mcp", 1),
    "greater than 1"
  )
  expect_error(
    picasso:::.picasso_method_flag("scad", 2),
    "greater than 2"
  )
  expect_equal(
    picasso:::.picasso_method_flag("mcp", 1 + .Machine$double.eps)$gamma,
    1 + .Machine$double.eps,
    tolerance = 0
  )
  expect_equal(
    picasso:::.picasso_method_flag(
      "scad", 2 + 2 * .Machine$double.eps
    )$gamma,
    2 + 2 * .Machine$double.eps,
    tolerance = 0
  )

  expect_error(
    picasso(
      fixture$x, fixture$sqrtlasso, family = "gaussian",
      nlambda = 2L, lla.max.stages = 2L
    ),
    "equal to 3"
  )

  sqrt.fit <- picasso(
    fixture$x, fixture$sqrtlasso, family = "sqrtlasso",
    method = "l1", lambda = c(0.5, 0.35)
  )
  expect_identical(sqrt.fit$alg, "active-set-quadratic-mm")
})
