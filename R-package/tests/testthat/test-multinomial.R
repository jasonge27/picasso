multinomial_fixture <- function(n.per.class = 10L, d = 5L) {
  set.seed(20260715)
  classes <- c("alpha", "beta", "gamma")
  y <- factor(
    rep(classes, each = n.per.class),
    levels = c(classes, "unused")
  )
  x <- matrix(rnorm(length(y) * d, sd = 0.6), length(y), d)
  signal.column <- match(as.character(y), classes)
  x[cbind(seq_len(nrow(x)), signal.column)] <-
    x[cbind(seq_len(nrow(x)), signal.column)] + 1.8
  list(x = x, y = y, classes = classes)
}


test_that("native routines are registered and dynamic lookup is disabled", {
  dll <- getLoadedDLLs()[["picasso"]]
  registered <- getDLLRegisteredRoutines(dll)
  expect_true("picasso_multinomial_call" %in% names(registered$.Call))
  expect_identical(
    registered$.Call[["picasso_multinomial_call"]]$numParameters, 15L
  )
  expect_false(dll[["dynamicLookup"]])
})


test_that("adaptive LLA stationarity-limit status is mapped", {
  expect_identical(
    picasso:::.picasso_multinomial_status_label(10L),
    "lla_stationarity_limit"
  )
})


fit_small_multinomial <- function(method = "l1", nlambda = 3L, ...) {
  dat <- multinomial_fixture()
  picasso(
    dat$x, dat$y,
    family = "multinomial",
    method = method,
    nlambda = nlambda,
    lambda.min.ratio = 0.5,
    prec = 1e-4,
    max.ite = 1000L,
    ...
  )
}


test_that("L1, MCP, and SCAD multinomial paths fit and predict", {
  dat <- multinomial_fixture()

  for (penalty in c("l1", "mcp", "scad")) {
    fit <- fit_small_multinomial(penalty, nlambda = 3L)
    expect_s3_class(fit, "multinomial")
    expect_identical(fit$family, "multinomial")
    expect_identical(fit$alg, "multinomial-proximal-newton")
    if (penalty == "l1") {
      expect_identical(fit$status, "completed")
      expect_identical(fit$status.code, 0L)
    } else {
      expect_true(fit$status.code %in% c(0L, 10L))
      expect_true(fit$status %in% c("completed", "lla_stationarity_limit"))
    }
    expect_null(fit$failure)
    expect_identical(fit$lla.max.stages, 3L)
    expect_equal(nrow(fit$diagnostics), fit$nlambda)
    expect_equal(fit$diagnostics$lambda, fit$lambda)
    expect_true(all(fit$diagnostics$runtime >= 0))
    expect_identical(fit$levels, dat$classes)
    expect_equal(length(fit$beta), length(dat$classes))
    expect_true(all(vapply(fit$beta, function(beta) {
      all(is.finite(as.matrix(beta)))
    }, logical(1))))

    probability <- predict(
      fit, dat$x[1:4, , drop = FALSE],
      lambda.idx = fit$nlambda, type = "response"
    )
    expect_equal(dim(probability), c(4L, 3L))
    expect_equal(colnames(probability), dat$classes)
    expect_equal(rowSums(probability), rep(1, 4), tolerance = 1e-10)
  }
})


test_that("native multinomial smooth loss matches explicit path predictions", {
  dat <- multinomial_fixture()
  truth <- match(as.character(dat$y), dat$classes)

  for (penalty in c("l1", "mcp", "scad")) {
    fit <- fit_small_multinomial(penalty, nlambda = 3L)
    explicit.nll <- vapply(seq_len(fit$nlambda), function(index) {
      probability <- predict(
        fit, dat$x, lambda.idx = index, type = "response"
      )
      -mean(log(pmax(
        probability[cbind(seq_len(nrow(dat$x)), truth)], 1e-15
      )))
    }, numeric(1))

    expect_equal(
      fit$diagnostics$smooth.nll, explicit.nll,
      tolerance = 2e-12, info = penalty
    )
    expect_equal(
      fit$dev.ratio,
      pmax(0, pmin(1, 1 - explicit.nll / fit$nulldev)),
      tolerance = 2e-12, info = penalty
    )
  }
})


test_that("saturated multinomial paths stop cleanly and remain CV-usable", {
  dat <- multinomial_fixture()
  design <- picasso:::.picasso_prepare_design(
    dat$x, standardize = TRUE, center = TRUE
  )$xx
  codes <- as.integer(droplevels(dat$y)) - 1L
  proportions <- tabulate(codes + 1L, nbins = 3L) / length(codes)
  lambda.max <- max(vapply(seq_len(3L), function(k) {
    max(abs(crossprod(
      design, as.numeric(codes == (k - 1L)) - proportions[k]
    ))) / length(codes)
  }, numeric(1)))
  requested <- 0.45 * lambda.max * (1 - 1e-7 * seq.int(0L, 11L))

  explicit <- picasso(
    dat$x, dat$y, family = "multinomial", lambda = requested,
    prec = 5e-7, max.ite = 5000L
  )
  expect_false(explicit$path.early.stopped)
  expect_identical(explicit$requested.nlambda, 12L)
  expect_identical(explicit$nlambda, 12L)

  fit <- picasso(
    dat$x, dat$y, family = "multinomial", nlambda = 100L,
    lambda.min.ratio = 1e-4, prec = 5e-7, max.ite = 5000L
  )
  expect_identical(fit$status, "completed")
  expect_true(fit$path.early.stopped)
  expect_identical(fit$requested.nlambda, 100L)
  expect_gte(fit$nlambda, 5L)
  expect_lt(fit$nlambda, fit$requested.nlambda)
  expect_equal(nrow(fit$diagnostics), fit$nlambda)

  set.seed(20260716)
  cv <- cv.picasso(
    dat$x, dat$y, family = "multinomial", nlambda = 100L,
    lambda.min.ratio = 1e-4, nfolds = 2L,
    prec = 5e-7, max.ite = 5000L
  )
  expect_gte(length(cv$lambda), 5L)
  expect_lte(length(cv$lambda), fit$nlambda)
  expect_true(all(is.finite(cv$cvm)))
})


test_that("multinomial LLA stage budget defaults to three and is configurable", {
  expect_identical(formals(picasso)$lla.max.stages, 3L)
  expect_identical(formals(picasso:::picasso.multinomial)$lla.max.stages, 3L)

  default <- fit_small_multinomial("scad", nlambda = 3L)
  extended <- fit_small_multinomial(
    "scad", nlambda = 3L, lla.max.stages = 8L
  )
  expect_identical(default$lla.max.stages, 3L)
  expect_identical(extended$lla.max.stages, 8L)
  expect_identical(default$nlambda, 3L)
  expect_identical(extended$nlambda, 3L)
  expect_identical(default$status.code, 10L)
  expect_identical(default$status, "lla_stationarity_limit")
  expect_identical(extended$status.code, 0L)
  expect_lte(max(extended$diagnostics$stationarity), 1e-4)
  expect_null(default$failure)
  expect_null(extended$failure)

  dat <- multinomial_fixture()
  set.seed(17)
  cv <- cv.picasso(
    dat$x, dat$y, family = "multinomial", method = "scad",
    nfolds = 2L, nlambda = 3L, lambda.min.ratio = 0.5,
    prec = 1e-4, max.ite = 1000L, lla.max.stages = 3L
  )
  expect_identical(cv$picasso.fit$status.code, 10L)
  expect_identical(length(cv$lambda), 3L)

  for (bad in list(2L, 3.5, Inf, NA_real_)) {
    expect_error(
      picasso(
        dat$x, dat$y, family = "multinomial", nlambda = 1L,
        lla.max.stages = bad
      ),
      "lla.max.stages"
    )
  }
})


test_that("multinomial status distinguishes dfmax and solver failure", {
  dat <- multinomial_fixture()
  reference <- fit_small_multinomial(nlambda = 3L)

  limited <- picasso(
    dat$x, dat$y,
    family = "multinomial",
    nlambda = 6L,
    lambda.min.ratio = 0.1,
    dfmax = 0L,
    prec = 1e-4,
    max.ite = 300L
  )
  expect_identical(limited$status, "dfmax_reached")
  expect_identical(limited$status.code, 1L)
  expect_null(limited$failure)
  expect_equal(nrow(limited$diagnostics), limited$nlambda)
  expect_lte(limited$nlambda, 6L)

  expect_error(
    picasso(
      dat$x, dat$y,
      family = "multinomial",
      lambda = tail(reference$lambda, 1L),
      prec = 1e-8,
      max.ite = 1L
    ),
    "status.*code.*before completing"
  )

  partial <- NULL
  expect_warning(
    partial <- picasso(
      dat$x, dat$y,
      family = "multinomial",
      lambda = c(reference$lambda[1L], tail(reference$lambda, 1L)),
      prec = 1e-8,
      max.ite = 1L
    ),
    "successful.*lambda prefix"
  )
  expect_gt(partial$status.code, 1L)
  expect_identical(partial$nlambda, 1L)
  expect_equal(nrow(partial$diagnostics), 1L)
  expect_identical(partial$failure$lambda.index, 2L)
  expect_equal(partial$failure$lambda, tail(reference$lambda, 1L))
  expect_true(is.na(partial$failure$diagnostics$iterations))
  expect_true(is.na(partial$failure$diagnostics$nonzero))
  expect_gte(partial$failure$diagnostics$inner.sweeps, 1)

  expect_error(
    suppressWarnings(cv.picasso(
      dat$x, dat$y,
      family = "multinomial",
      lambda = c(reference$lambda[1L], tail(reference$lambda, 1L)),
      nfolds = 2L,
      prec = 1e-8,
      max.ite = 1L
    )),
    "Full-data multinomial fit stopped"
  )
})


test_that("zero-gradient multinomial data receive a valid lambda path", {
  y <- factor(rep(c("alpha", "beta", "gamma"), each = 4L))
  x <- matrix(1, length(y), 3L)

  fit <- picasso(
    x, y, family = "multinomial", nlambda = 3L,
    prec = 1e-4, max.ite = 100L
  )
  expect_equal(fit$lambda, seq(.Machine$double.eps, 0, length.out = 3L))
  expect_true(all(diff(fit$lambda) < 0))
  expect_identical(fit$status, "completed")
  probability <- predict(fit, x[1:2, , drop = FALSE], lambda.idx = 3L)
  expect_equal(rowSums(probability), c(1, 1), tolerance = 1e-12)

  single <- picasso(
    x, y, family = "multinomial", nlambda = 1L,
    prec = 1e-4, max.ite = 100L
  )
  expect_identical(single$lambda, 0)
})


test_that("no-intercept multinomial fits scale about the origin", {
  y <- factor(c(rep("alpha", 8L), rep("beta", 3L), "gamma"))
  n <- length(y)
  x <- cbind(
    constant = rep(5, n),
    shifted = 10 + seq_len(n),
    oscillating = 3 + rep(c(-1, 1), length.out = n)
  )

  origin.design <- picasso:::.picasso_prepare_design(
    x, standardize = TRUE, center = FALSE
  )
  expect_equal(origin.design$xm, matrix(0, 1L, ncol(x)))
  expect_equal(unname(colSums(origin.design$xx^2)), rep(n - 1, ncol(x)),
               tolerance = 1e-12)
  expect_true(all(origin.design$xx[, "constant"] != 0))
  expect_true(all(abs(colMeans(origin.design$xx)) > 0))

  y.code <- as.integer(y)
  uniform <- rep(1 / nlevels(y), nlevels(y))
  expected.lambda.max <- max(vapply(seq_len(nlevels(y)), function(k) {
    max(abs(crossprod(
      origin.design$xx, as.numeric(y.code == k) - uniform[k]
    ))) / n
  }, numeric(1)))

  no.intercept <- picasso(
    x, y, family = "multinomial", intercept = FALSE,
    standardize = TRUE, nlambda = 2L, lambda.min.ratio = 0.5,
    prec = 1e-7, max.ite = 1000L
  )
  expect_equal(no.intercept$lambda[1L], expected.lambda.max,
               tolerance = 1e-12)
  expect_true(all(vapply(no.intercept$intercept, function(value) {
    identical(as.numeric(value), rep(0, no.intercept$nlambda))
  }, logical(1))))
  expect_equal(no.intercept$nulldev, log(nlevels(y)), tolerance = 1e-12)
  expect_equal(no.intercept$diagnostics$objective[1L], no.intercept$nulldev,
               tolerance = 1e-10)
  expect_equal(no.intercept$dev.ratio[1L], 0, tolerance = 1e-10)

  centered.design <- picasso:::.picasso_prepare_design(
    x, standardize = TRUE, center = TRUE
  )
  empirical <- tabulate(y.code, nbins = nlevels(y)) / n
  expected.intercept.lambda.max <- max(vapply(
    seq_len(nlevels(y)), function(k) {
      max(abs(crossprod(
        centered.design$xx, as.numeric(y.code == k) - empirical[k]
      ))) / n
    }, numeric(1)
  ))
  with.intercept <- picasso(
    x, y, family = "multinomial", intercept = TRUE,
    standardize = TRUE, nlambda = 1L,
    prec = 1e-7, max.ite = 1000L
  )
  expect_equal(with.intercept$lambda, expected.intercept.lambda.max,
               tolerance = 1e-12)
  expect_equal(
    with.intercept$nulldev,
    -mean(log(empirical[y.code])),
    tolerance = 1e-12
  )
})


test_that("short multinomial paths have safe dynamic defaults", {
  dat <- multinomial_fixture()
  path <- fit_small_multinomial(nlambda = 2L)
  fit <- picasso(
    dat$x, dat$y,
    family = "multinomial",
    lambda = path$lambda[1L],
    prec = 1e-4,
    max.ite = 300L
  )

  coefficient <- coef(fit)
  expect_length(coefficient, 3L)
  expect_true(all(vapply(coefficient, ncol, integer(1)) == 1L))

  probability <- predict(fit, dat$x[1:2, , drop = FALSE])
  expect_equal(dim(probability), c(2L, 3L))
  link <- predict(fit, dat$x[1:2, , drop = FALSE], type = "link")
  expect_equal(colnames(link), dat$classes)
})


test_that("multinomial wrapper rejects ambiguous and malformed inputs", {
  dat <- multinomial_fixture()

  expect_error(
    picasso(dat$x, dat$y, family = "multinomial", lambda = c(0.1, 0.2)),
    "strictly decreasing"
  )
  expect_error(
    picasso(dat$x, dat$y, family = "multinomial", dfmax = -1),
    "dfmax"
  )
  expect_error(
    picasso(dat$x, dat$y, family = "multinomial", nlambda = 0),
    "nlambda"
  )
  expect_error(
    picasso(dat$x, dat$y, family = "multinomial", offset = rep(0, nrow(dat$x))),
    "not supported"
  )
  bad.y <- as.numeric(dat$y)
  bad.y[1L] <- Inf
  expect_error(
    picasso(dat$x, bad.y, family = "multinomial"),
    "finite class values"
  )

  fit <- fit_small_multinomial(nlambda = 2L)
  expect_error(predict(fit, dat$x[, -1, drop = FALSE]), "expects")
  expect_error(predict(fit, dat$x, type = "invalid"), "arg")
  expect_error(predict(fit, dat$x, lambda.idx = 0), "out-of-range")
  expect_error(predict(fit, dat$x, lambda.idx = 1.5), "integer indices")
  expect_error(predict(fit, dat$x, s = NA_real_), "finite")
  expect_error(
    predict(fit, dat$x, lambda.idx = 1L, s = fit$lambda[1L]),
    "only one"
  )
  expect_error(coef(fit, beta.idx = ncol(dat$x) + 1L), "out-of-range")
})


test_that("assessment and confusion matrices support multinomial fits", {
  dat <- multinomial_fixture()
  fit <- fit_small_multinomial(nlambda = 3L)

  assessment <- assess.picasso(fit, dat$x, dat$y)
  expect_s3_class(assessment, "assess.picasso")
  expect_length(assessment$deviance, fit$nlambda)
  expect_length(assessment$class, fit$nlambda)
  expect_true(all(is.finite(assessment$deviance)))
  expect_true(all(assessment$class >= 0 & assessment$class <= 1))

  confusion <- confusion.picasso(fit, dat$x, dat$y, lambda.idx = c(1L, 3L))
  expect_named(confusion, c("lambda[1]", "lambda[3]"))
  expect_true(all(vapply(confusion, function(tab) {
    identical(dim(tab), c(3L, 3L)) && sum(tab) == nrow(dat$x)
  }, logical(1))))
  subset <- dat$y != "gamma"
  subset.confusion <- confusion.picasso(
    fit, dat$x[subset, , drop = FALSE], dat$y[subset], lambda.idx = 1L
  )[[1L]]
  expect_identical(dim(subset.confusion), c(3L, 3L))
  expect_identical(unname(subset.confusion[, "gamma"]), c(0L, 0L, 0L))
  expect_error(
    assess.picasso(fit, dat$x, rep("unknown", nrow(dat$x))),
    "absent from the fitted model"
  )
})


test_that("multinomial cross-validation is stratified and uses softmax loss", {
  dat <- multinomial_fixture(n.per.class = 9L)
  set.seed(17)
  cv.class <- cv.picasso(
    dat$x, dat$y,
    family = "multinomial",
    nfolds = 3L,
    nlambda = 2L,
    lambda.min.ratio = 0.6,
    type.measure = "default",
    prec = 1e-4,
    max.ite = 300L
  )
  expect_identical(cv.class$family, "multinomial")
  expect_identical(cv.class$name, "class")
  expect_true(all(table(droplevels(dat$y), cv.class$foldid) > 0L))
  expect_length(cv.class$nzero, length(cv.class$lambda))
  manual.nzero <- vapply(seq_along(cv.class$lambda), function(index) {
    sum(vapply(cv.class$picasso.fit$beta, function(beta) {
      sum(abs(beta[, index]) > 1e-8)
    }, numeric(1)))
  }, numeric(1))
  expect_equal(as.numeric(cv.class$nzero), manual.nzero)

  fold.loss <- matrix(NA_real_, nrow = 3L, ncol = length(cv.class$lambda))
  for (fold in seq_len(3L)) {
    train <- cv.class$foldid != fold
    test <- !train
    fold.fit <- picasso(
      dat$x[train, , drop = FALSE], dat$y[train],
      family = "multinomial", lambda = cv.class$lambda,
      prec = 1e-4, max.ite = 300L
    )
    predicted <- predict(
      fold.fit, dat$x[test, , drop = FALSE],
      lambda.idx = seq_along(cv.class$lambda), type = "class"
    )
    fold.loss[fold, ] <- vapply(predicted, function(value) {
      mean(as.character(value) != as.character(dat$y[test]))
    }, numeric(1))
  }
  expected.se <- apply(fold.loss, 2L, sd) / sqrt(nrow(fold.loss))
  expect_equal(cv.class$cvsd, expected.se, tolerance = 1e-12)
  expect_equal(cv.class$cvup, cv.class$cvm + expected.se,
               tolerance = 1e-12)
  expect_equal(cv.class$cvlo, cv.class$cvm - expected.se,
               tolerance = 1e-12)

  set.seed(17)
  cv.deviance <- cv.picasso(
    dat$x, dat$y,
    family = "multinomial",
    nfolds = 3L,
    nlambda = 2L,
    lambda.min.ratio = 0.6,
    type.measure = "deviance",
    prec = 1e-4,
    max.ite = 300L
  )
  expect_identical(cv.deviance$name, "deviance")
  expect_true(all(is.finite(cv.deviance$cvm)))
  expect_true(all(cv.deviance$cvm >= 0))
})


test_that("multinomial fold IDs are strict and missing training classes fail", {
  dat <- multinomial_fixture(n.per.class = 6L)
  expect_error(
    cv.picasso(
      dat$x, dat$y, family = "multinomial", foldid = rep(c(1L, 3L), 9L),
      nlambda = 1L, prec = 1e-4, max.ite = 300L
    ),
    "consecutive integers"
  )

  custom.fold <- integer(nrow(dat$x))
  custom.fold[dat$y == "alpha"] <- rep(c(1L, 2L), length.out = 6L)
  custom.fold[dat$y == "beta"] <- rep(c(2L, 3L), length.out = 6L)
  custom.fold[dat$y == "gamma"] <- rep(c(1L, 3L), length.out = 6L)
  custom <- cv.picasso(
    dat$x, dat$y,
    family = "multinomial", foldid = custom.fold,
    nlambda = 1L, prec = 1e-4, max.ite = 300L
  )
  expect_identical(custom$picasso.fit$levels, dat$classes)
  expect_true(all(vapply(seq_len(3L), function(fold) {
    length(unique(dat$y[custom.fold == fold])) < 3L
  }, logical(1))))

  bad.fold <- rep(2L, nrow(dat$x))
  bad.fold[dat$y == "gamma"] <- 1L
  expect_error(
    cv.picasso(
      dat$x, dat$y, family = "multinomial", foldid = bad.fold,
      nlambda = 1L, prec = 1e-4, max.ite = 300L
    ),
    "Training fold 1.*gamma"
  )
})


test_that("ordinary Gaussian assessment and cross-validation still run", {
  set.seed(44)
  x <- matrix(rnorm(80), 20L, 4L)
  y <- x[, 1L] - 0.5 * x[, 2L] + rnorm(20L, sd = 0.2)
  fit <- picasso(x, y, family = "gaussian", nlambda = 2L)
  assessment <- assess.picasso(fit, x, y)
  expect_length(assessment$mse, fit$nlambda)
  covariance.fit <- picasso(
    x, y, family = "gaussian", type.gaussian = "covariance", nlambda = 2L
  )
  expect_equal(dim(covariance.fit$beta), dim(fit$beta))

  set.seed(45)
  cv <- cv.picasso(x, y, family = "gaussian", nfolds = 2L, nlambda = 2L)
  expect_identical(cv$family, "gaussian")
  expect_true(all(is.finite(cv$cvm)))
})
