scalar_contract_fixture <- function() {
  x <- matrix(seq_len(24L) / 24, nrow = 8L, ncol = 3L)
  list(
    x = x,
    gaussian = seq_len(8L) / 4,
    binomial = rep(c(0L, 1L), 4L),
    poisson = rep(c(1L, 2L), 4L),
    sqrtlasso = seq_len(8L) / 5,
    multinomial = factor(rep(c("a", "b", "c", "a"), 2L))
  )
}


mock_scalar_fit <- function(class = "gaussian") {
  structure(
    list(
      beta = Matrix::Matrix(
        matrix(c(1, 0, -0.5, 0, 0.25, 0), nrow = 3L, ncol = 2L)
      ),
      intercept = c(0.1, -0.2),
      lambda = c(1, 0.5),
      nlambda = 2L,
      offset.used = FALSE,
      levels = c("no", "yes")
    ),
    class = class
  )
}


test_that("public fit choices and integer controls are strict", {
  fixture <- scalar_contract_fixture()

  expect_error(
    picasso(fixture$x, fixture$gaussian, family = c("gaussian", "poisson")),
    "family must be one of"
  )
  expect_error(
    picasso(fixture$x, fixture$gaussian, family = NA_character_),
    "family must be one of"
  )
  expect_error(
    picasso(fixture$x, fixture$gaussian, method = c("l1", "mcp")),
    "method must be one of"
  )
  expect_error(
    picasso(fixture$x, fixture$gaussian, method = NA_character_),
    "method must be one of"
  )

  for (family in c(
    "gaussian", "binomial", "poisson", "sqrtlasso", "multinomial"
  )) {
    response <- fixture[[family]]
    label <- paste("family", family)
    expect_error(
      picasso(fixture$x, response, family = family, max.ite = 1.5),
      "max.ite must be a positive finite integer", info = label
    )
    expect_error(
      picasso(fixture$x, response, family = family, dfmax = -1),
      "dfmax must be a nonnegative finite integer", info = label
    )
    expect_error(
      picasso(fixture$x, response, family = family, dfmax = 1.5),
      "dfmax must be a nonnegative finite integer", info = label
    )
    expect_error(
      picasso(fixture$x, response, family = family, standardize = 1),
      "standardize must be TRUE or FALSE", info = label
    )
    expect_error(
      picasso(fixture$x, response, family = family, intercept = NA),
      "intercept must be TRUE or FALSE", info = label
    )
    expect_error(
      picasso(fixture$x, response, family = family, verbose = c(TRUE, FALSE)),
      "verbose must be TRUE or FALSE", info = label
    )
  }
})


test_that("generated path controls reject invalid values consistently", {
  fixture <- scalar_contract_fixture()

  for (family in c(
    "gaussian", "binomial", "poisson", "sqrtlasso", "multinomial"
  )) {
    response <- fixture[[family]]
    label <- paste("family", family)
    expect_error(
      picasso(
        fixture$x, response, family = family, standardize = FALSE,
        nlambda = 1.5
      ),
      "nlambda must be a positive finite integer", info = label
    )
    expect_error(
      picasso(
        fixture$x, response, family = family, standardize = FALSE,
        lambda.min.ratio = 0
      ),
      "lambda.min.ratio.*strictly between 0 and 1", info = label
    )
  }

  explicit <- picasso:::.picasso_lambda_path(
    c(1, 0.5), nlambda = 0, lambda.min.ratio = Inf, lambda.max = NA_real_
  )
  expect_identical(explicit$lambda, c(1, 0.5))
  expect_identical(explicit$nlambda, 2L)
})


test_that("scalar coefficient and prediction indices are strict", {
  fit <- mock_scalar_fit()
  newx <- matrix(seq_len(12L) / 10, nrow = 4L, ncol = 3L)

  expect_error(
    coef(fit, lambda.idx = 1.5, beta.idx = 1L),
    "lambda.idx.*finite integer indices"
  )
  expect_error(
    coef(fit, lambda.idx = 1L, beta.idx = NA_real_),
    "beta.idx.*finite integer indices"
  )
  expect_error(
    predict(fit, newx, lambda.idx = 1.5),
    "lambda.idx.*finite integer indices"
  )
  expect_error(
    predict(fit, newx, lambda.idx = 1L, Y.pred.idx = 1.5),
    "response.idx.*finite integer indices"
  )
  expect_error(
    predict(fit, newx, lambda.idx = 3L),
    "lambda.idx.*out-of-range"
  )

  warning.seen <- FALSE
  expect_error(
    withCallingHandlers(
      coef(fit, lambda.idx = 1e100, beta.idx = 1L),
      warning = function(w) {
        warning.seen <<- TRUE
        invokeRestart("muffleWarning")
      }
    ),
    "lambda.idx.*out-of-range"
  )
  expect_false(warning.seen)
})


test_that("scalar default indices are bounded by short fitted dimensions", {
  newx <- matrix(seq_len(8L) / 10, nrow = 4L, ncol = 2L)

  for (fit.class in c("gaussian", "logit", "poisson", "sqrtlasso")) {
    fit <- mock_scalar_fit(fit.class)
    fit$beta <- fit$beta[1:2, , drop = FALSE]

    extracted <- coef(fit)
    expect_identical(dim(extracted), c(3L, 2L), info = fit.class)

    predicted <- predict(fit, newx)
    expect_identical(dim(predicted), c(4L, 2L), info = fit.class)
    expect_identical(
      colnames(predicted), c("lambda[1]", "lambda[2]"), info = fit.class
    )
  }
})


test_that("explicit scalar prediction row indices are not default sentinels", {
  newx <- matrix(seq_len(14L) / 10, nrow = 7L, ncol = 2L)

  for (fit.class in c("gaussian", "logit", "poisson", "sqrtlasso")) {
    fit <- mock_scalar_fit(fit.class)
    fit$beta <- fit$beta[1:2, , drop = FALSE]
    row.argument <- if (fit.class %in% c("gaussian", "sqrtlasso")) {
      "Y.pred.idx"
    } else {
      "p.pred.idx"
    }

    full <- predict(fit, newx, lambda.idx = 1L)
    selected <- do.call(
      predict,
      c(list(object = fit, newdata = newx, lambda.idx = 1L),
        stats::setNames(list(1:5), row.argument))
    )

    expect_identical(dim(full), c(7L, 1L), info = fit.class)
    expect_identical(dim(selected), c(5L, 1L), info = fit.class)
    expect_equal(unname(selected), unname(full[1:5, , drop = FALSE]),
                 tolerance = 0, info = fit.class)

    full.s <- predict(fit, newx, s = fit$lambda[1L])
    selected.s <- do.call(
      predict,
      c(list(object = fit, newdata = newx, s = fit$lambda[1L]),
        stats::setNames(list(1:5), row.argument))
    )
    expect_identical(dim(selected.s), c(5L, 1L), info = fit.class)
    expect_equal(unname(selected.s), unname(full.s[1:5, , drop = FALSE]),
                 tolerance = 0, info = fit.class)
  }
})


test_that("scalar prediction validates type, s, and newdata", {
  fit <- mock_scalar_fit()
  newx <- matrix(seq_len(12L) / 10, nrow = 4L, ncol = 3L)

  expect_error(
    predict(fit, newx, lambda.idx = 1L, type = "unknown"),
    "type must be one of"
  )
  expect_error(
    predict(fit, newx, lambda.idx = 1L, type = c("link", "response")),
    "type must be one of"
  )
  expect_error(predict(fit, newx, s = numeric()), "s.*finite nonnegative")
  expect_error(predict(fit, newx, s = NA_real_), "s.*finite nonnegative")
  expect_error(predict(fit, newx, s = -0.1), "s.*finite nonnegative")
  expect_error(predict(fit, newx, s = "0.5"), "s.*finite nonnegative")

  expect_error(
    predict(fit, as.numeric(newx), lambda.idx = 1L),
    "newdata.*numeric matrix"
  )
  expect_error(
    predict(fit, matrix(numeric(), nrow = 0L, ncol = 3L), lambda.idx = 1L),
    "newdata.*at least one row"
  )
  expect_error(
    predict(fit, newx[, 1:2, drop = FALSE], lambda.idx = 1L),
    "newdata.*expects 3"
  )
  newx.na <- newx
  newx.na[1L] <- NA_real_
  expect_error(
    predict(fit, newx.na, lambda.idx = 1L),
    "newdata.*only finite"
  )
  newx.inf <- newx
  newx.inf[1L] <- Inf
  expect_error(
    predict(fit, newx.inf, lambda.idx = 1L),
    "newdata.*only finite"
  )

  expected.support <- list(c(1L, 3L))
  expect_identical(
    predict(fit, NULL, lambda.idx = 1L, type = "nonzero"),
    expected.support
  )
  expect_identical(
    predict(fit, "newdata is unused", lambda.idx = 1L, type = "nonzero"),
    expected.support
  )
  expect_identical(
    predict(
      fit, NULL, lambda.idx = 1.5, s = fit$lambda[1L], type = "nonzero"
    ),
    expected.support
  )

  # Preserve the documented scalar behavior: s takes precedence over an
  # otherwise invalid lambda.idx value.
  expect_equal(
    unname(predict(fit, newx, lambda.idx = 1.5, s = fit$lambda[1L])),
    unname(predict(fit, newx, lambda.idx = 1L)),
    tolerance = 0
  )
  expect_silent(
    predict(fit, matrix(as.integer(newx * 10), nrow = 4L), lambda.idx = 1L)
  )

  logit.fit <- mock_scalar_fit("logit")
  expect_error(
    predict(
      logit.fit, newx, lambda.idx = 1L,
      type = c("class", "response")
    ),
    "type must be one of"
  )
})


test_that("CV plotting handles paths containing zero lambda", {
  cv <- structure(
    list(
      lambda = c(1, 0),
      cvm = c(0.8, 0.7),
      cvsd = c(0.1, 0.1),
      cvup = c(0.9, 0.8),
      cvlo = c(0.7, 0.6),
      lambda.min = 0,
      lambda.1se = 1,
      name = "deviance"
    ),
    class = "cv.picasso"
  )

  output <- tempfile(fileext = ".pdf")
  grDevices::pdf(output)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_silent(plot(cv))

  cv$lambda <- c(0, 0)
  cv$lambda.1se <- 0
  expect_silent(plot(cv))
})
