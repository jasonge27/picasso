streaming_multinomial_model <- function(mode = c("ordinary", "ties", "extreme")) {
  mode <- match.arg(mode)
  classes <- c("zeta", "alpha", "mu")
  d <- 3L
  nlambda <- 4L
  beta <- lapply(seq_len(3L), function(class) matrix(0, d, nlambda))
  intercept <- lapply(seq_len(3L), function(class) numeric(nlambda))

  if (mode == "ordinary") {
    beta[[1L]][1L, ] <- c(0.4, 0.2, -0.1, -0.3)
    beta[[2L]][2L, ] <- c(-0.2, 0.1, 0.3, 0.5)
    beta[[3L]][3L, ] <- c(0.1, -0.3, 0.2, -0.4)
    intercept <- list(
      c(0.1, 0.2, 0.31, 0.4),
      c(-0.2, 0, 0.2, 0.4),
      c(0.3, 0.1, -0.1, -0.3)
    )
  } else if (mode == "extreme") {
    beta[[1L]][1L, ] <- 500
    beta[[2L]][1L, ] <- -500
    beta[[3L]][2L, ] <- 400
  }

  structure(
    list(
      beta = beta,
      intercept = intercept,
      lambda = c(4, 2, 1, 0.5),
      nlambda = nlambda,
      K = 3L,
      levels = classes,
      family = "multinomial"
    ),
    class = "multinomial"
  )
}


streaming_multinomial_data <- function() {
  list(
    x = rbind(
      c(1, 0, -1), c(0, 1, 1), c(-1, -1, 0),
      c(2, -2, 0.5), c(0, 0, 0)
    ),
    y = factor(
      c("zeta", "alpha", "mu", "zeta", "alpha"),
      levels = c("mu", "alpha", "zeta", "unused")
    )
  )
}


legacy_multinomial_logits <- function(object, newdata, lambda.idx) {
  lapply(lambda.idx, function(index) {
    logits <- matrix(0, nrow(newdata), object$K)
    for (class in seq_len(object$K)) {
      logits[, class] <-
        as.numeric(newdata %*% object$beta[[class]][, index]) +
        object$intercept[[class]][index]
    }
    colnames(logits) <- object$levels
    logits
  })
}


legacy_multinomial_confusion <- function(predicted, response, levels) {
  table(
    predicted = factor(levels[predicted], levels = levels),
    actual = factor(levels[response], levels = levels)
  )
}


test_that("tabulated multinomial confusion tables match factor tables exactly", {
  cases <- list(
    list(
      levels = c("negative", "positive"),
      predicted = c(1L, 1L, 2L, 2L),
      response = c(1L, 2L, 1L, 2L)
    ),
    list(
      levels = setNames(
        c("class one", "beta/[2]", "third | class"),
        c("first name", "second name", "third name")
      ),
      predicted = c(1L, 3L, 1L, 3L, 3L),
      response = c(3L, 1L, 3L, 3L, 1L)
    ),
    list(
      levels = c("a", "b", "c", "d"),
      predicted = c(4L, 2L, 4L, 1L, 2L, 4L),
      response = c(1L, 4L, 2L, 1L, 4L, 2L)
    )
  )

  for (case in cases) {
    expected <- legacy_multinomial_confusion(
      case$predicted, case$response, case$levels
    )
    actual <- picasso:::.picasso_multinomial_confusion_table(
      case$predicted, case$response, case$levels
    )
    expect_identical(actual, expected)
  }
})


test_that("tabulated confusion preserves path order, repeats, labels, and ties", {
  data <- streaming_multinomial_data()
  lambda.idx <- c(4L, 2L, 4L, 1L)

  for (mode in c("ordinary", "ties")) {
    object <- streaming_multinomial_model(mode)
    object$levels <- c("class one", "beta/[2]", "third | class")
    response <- match(as.character(data$y), c("zeta", "alpha", "mu"))
    newy <- object$levels[response]
    logits <- legacy_multinomial_logits(object, data$x, lambda.idx)
    expected <- lapply(logits, function(eta) {
      legacy_multinomial_confusion(
        max.col(eta, ties.method = "first"), response, object$levels
      )
    })
    names(expected) <- paste0("lambda[", lambda.idx, "]")

    actual <- confusion.picasso(
      object, data$x, newy, lambda.idx = lambda.idx
    )
    expect_identical(actual, expected, info = mode)
  }
})


test_that("one-lambda logits preserve the public path helper contract", {
  data <- streaming_multinomial_data()
  for (mode in c("ordinary", "ties", "extreme")) {
    object <- streaming_multinomial_model(mode)
    index <- c(4L, 2L, 1L)
    expected <- legacy_multinomial_logits(object, data$x, index)
    expect_identical(
      picasso:::.picasso_multinomial_logits(object, data$x, index),
      expected,
      info = mode
    )
    expect_identical(
      picasso:::.picasso_multinomial_logits_one(object, data$x, 2L),
      expected[[2L]],
      info = mode
    )
  }
})


test_that("streamed scoring is serialization-equivalent on edge cases", {
  data <- streaming_multinomial_data()
  response <- match(as.character(data$y), c("zeta", "alpha", "mu"))

  for (mode in c("ordinary", "ties", "extreme")) {
    object <- streaming_multinomial_model(mode)
    legacy.logits <- legacy_multinomial_logits(
      object, data$x, seq_len(object$nlambda)
    )
    expected.assessment <- structure(
      list(
        lambda = object$lambda,
        deviance = vapply(legacy.logits, function(logits) {
          picasso:::.picasso_multinomial_nll_from_logits(response, logits)
        }, numeric(1)),
        class = vapply(legacy.logits, function(logits) {
          mean(max.col(logits, ties.method = "first") != response)
        }, numeric(1))
      ),
      class = "assess.picasso"
    )
    expect_identical(
      assess.picasso(object, data$x, data$y), expected.assessment,
      info = mode
    )

    expected.class <- lapply(legacy.logits[c(4L, 2L)], function(logits) {
      factor(
        max.col(logits, ties.method = "first"),
        levels = seq_len(object$K), labels = object$levels
      )
    })
    expect_identical(
      predict(object, data$x, lambda.idx = c(4L, 2L), type = "class"),
      expected.class,
      info = mode
    )

    actual <- factor(object$levels[response], levels = object$levels)
    expected.confusion <- lapply(c(4L, 1L), function(index) {
      predicted <- factor(
        object$levels[max.col(
          legacy.logits[[index]], ties.method = "first"
        )],
        levels = object$levels
      )
      table(predicted = predicted, actual = actual)
    })
    names(expected.confusion) <- c("lambda[4]", "lambda[1]")
    expect_identical(
      confusion.picasso(
        object, data$x, data$y, lambda.idx = c(4L, 1L)
      ),
      expected.confusion,
      info = mode
    )
  }
})


test_that("class scoring streams logits while retaining softmax tie semantics", {
  data <- streaming_multinomial_data()
  object <- streaming_multinomial_model("ordinary")
  original.softmax <- picasso:::.picasso_multinomial_softmax
  softmax.calls <- 0L

  local_mocked_bindings(
    .picasso_multinomial_softmax = function(...) {
      softmax.calls <<- softmax.calls + 1L
      original.softmax(...)
    },
    .picasso_multinomial_logits = function(...) {
      stop("scoring retained a full logits path")
    },
    .package = "picasso"
  )

  expect_silent(assess.picasso(object, data$x, data$y))
  expect_silent(confusion.picasso(
    object, data$x, data$y, lambda.idx = c(1L, 4L)
  ))
  expect_silent(predict(
    object, data$x, lambda.idx = c(1L, 4L), type = "class"
  ))
  expect_identical(softmax.calls, 0L)

  scoring.functions <- list(
    picasso:::assess.picasso,
    picasso:::confusion.picasso,
    picasso:::cv.picasso
  )
  expect_true(all(vapply(scoring.functions, function(fun) {
    !grepl(
      ".picasso_multinomial_logits(",
      paste(deparse(body(fun)), collapse = "\n"),
      fixed = TRUE
    )
  }, logical(1))))
})


near_tie_multinomial_model <- function(value = 0.125 * .Machine$double.eps) {
  classes <- c("first", "second", "third")
  structure(list(
    beta = lapply(seq_len(3L), function(class) matrix(0, 1L, 1L)),
    intercept = list(0, value, -2),
    lambda = 1,
    nlambda = 1L,
    K = 3L,
    levels = classes,
    family = "multinomial",
    status = "completed",
    status.code = 0L,
    path.early.stopped = FALSE,
    fast.mode = FALSE,
    prec = 1e-4
  ), class = "multinomial")
}


test_that("near ties classify from the largest finite logit", {
  object <- near_tie_multinomial_model()
  x <- matrix(1, 1L, 1L)
  logits <- picasso:::.picasso_multinomial_logits_one(object, x, 1L)
  probability <- picasso:::.picasso_multinomial_softmax(logits)

  # The finite logits differ, but the softmax values round to an exact tie.
  # This is the compatibility boundary that direct max.col(logits) misses.
  expect_identical(max.col(logits, ties.method = "first"), 2L)
  expect_identical(max.col(probability, ties.method = "first"), 1L)
  expect_identical(
    as.character(predict(object, x, type = "class")), "second"
  )
  expect_identical(assess.picasso(object, x, "first")$class, 1)
  confusion <- confusion.picasso(object, x, "first")[[1L]]
  expect_identical(unname(confusion["second", "first"]), 1L)

  y <- factor(
    c(rep("first", 6L), rep("second", 4L), rep("third", 2L)),
    levels = object$levels
  )
  foldid <- unlist(lapply(c(6L, 4L, 2L), function(size) {
    rep(seq_len(2L), length.out = size)
  }))
  fake.fit <- function(X, Y, ..., lambda = NULL, nlambda = NULL,
                       lambda.min.ratio = NULL) {
    result <- near_tie_multinomial_model()
    result$levels <- levels(droplevels(as.factor(Y)))
    result
  }
  local_mocked_bindings(picasso = fake.fit, .package = "picasso")
  cv <- cv.picasso(
    matrix(1, length(y), 1L), y, foldid = foldid,
    family = "multinomial", nlambda = 1L, type.measure = "class"
  )
  expect_identical(cv$cvm, 2 / 3)
})


test_that("near-tie compatibility boundary is explicit", {
  fractions <- c(0.125, 0.25, 0.5, 1, 2)
  choices <- vapply(fractions, function(fraction) {
    logits <- matrix(c(0, fraction * .Machine$double.eps, -2), 1L)
    probability <- picasso:::.picasso_multinomial_softmax(logits)
    c(
      direct = max.col(logits, ties.method = "first"),
      legacy = max.col(probability, ties.method = "first")
    )
  }, integer(2L))
  expect_identical(unname(choices["direct", ]), rep(2L, 5L))
  expect_identical(unname(choices["legacy", ]), c(1L, 1L, 2L, 2L, 2L))
})


test_that("NaN and positive-infinite fitted logits still fail class scoring", {
  x <- matrix(1, 6L, 1L)
  y <- factor(
    rep(c("first", "second", "third"), each = 2L),
    levels = c("first", "second", "third")
  )
  foldid <- rep(seq_len(2L), 3L)

  for (bad in c(Inf, NaN)) {
    object <- near_tie_multinomial_model(bad)
    expect_error(
      predict(object, x, type = "class"),
      "logits must be finite"
    )
    expect_error(
      assess.picasso(object, x, y),
      "logits must be finite"
    )
    expect_error(
      confusion.picasso(object, x, y),
      "logits must be finite"
    )

    fake.fit <- function(X, Y, ..., lambda = NULL, nlambda = NULL,
                         lambda.min.ratio = NULL) {
      result <- near_tie_multinomial_model(bad)
      result$levels <- levels(droplevels(as.factor(Y)))
      result
    }
    local_mocked_bindings(picasso = fake.fit, .package = "picasso")
    expect_error(
      cv.picasso(
        x, y, foldid = foldid, family = "multinomial",
        nlambda = 1L, type.measure = "class"
      ),
      "logits must be finite"
    )
    # Deviance no longer forms the unused probability matrix, so malformed
    # fitted logits fail at the NLL contract instead of the softmax contract.
    expect_error(
      cv.picasso(
        x, y, foldid = foldid, family = "multinomial",
        nlambda = 1L, type.measure = "deviance"
      ),
      "logits must be finite"
    )
  }
})


test_that("all multinomial scoring rejects non-finite logits", {
  object <- near_tie_multinomial_model(-Inf)
  x <- matrix(1, 1L, 1L)
  expect_error(
    predict(object, x, type = "class"), "logits must be finite"
  )
  expect_error(
    confusion.picasso(object, x, "first"), "logits must be finite"
  )
  expect_error(
    assess.picasso(object, x, "first"), "logits must be finite"
  )
})


test_that("multinomial CV class and deviance losses remain exact", {
  set.seed(20260719)
  x <- matrix(rnorm(90), 30L, 3L)
  y <- factor(
    rep(c("zeta", "alpha", "mu"), each = 10L),
    levels = c("mu", "zeta", "alpha", "unused")
  )
  foldid <- rep(seq_len(3L), 10L)

  for (measure in c("class", "deviance")) {
    set.seed(101)
    actual <- cv.picasso(
      x, y, family = "multinomial", foldid = foldid,
      nlambda = 2L, lambda.min.ratio = 0.7,
      type.measure = measure, prec = 1e-4, max.ite = 300L
    )
    expected <- matrix(NA_real_, 3L, length(actual$lambda))
    for (fold in seq_len(3L)) {
      train <- foldid != fold
      fit <- picasso(
        x[train, , drop = FALSE], y[train], family = "multinomial",
        lambda = actual$lambda, prec = 1e-4, max.ite = 300L
      )
      response <- match(as.character(y[!train]), fit$levels)
      logits <- legacy_multinomial_logits(
        fit, x[!train, , drop = FALSE], seq_along(actual$lambda)
      )
      expected[fold, ] <- vapply(logits, function(eta) {
        if (measure == "class") {
          mean(max.col(eta, ties.method = "first") != response)
        } else {
          picasso:::.picasso_multinomial_nll_from_logits(response, eta)
        }
      }, numeric(1))
    }
    expect_equal(actual$cvm, colMeans(expected), tolerance = 1e-12)
    expect_equal(
      actual$cvsd,
      apply(expected, 2L, stats::sd) / sqrt(nrow(expected)),
      tolerance = 1e-12
    )
  }
})
