legacy.binomial.confusion <- function(
    object, newx, newy, lambda.idx = NULL, newoffset = NULL) {
  if (!is.numeric(newx) || length(dim(newx)) != 2L || nrow(newx) == 0L ||
      anyNA(newx) || any(!is.finite(newx))) {
    stop("newx must be a nonempty finite numeric matrix.")
  }
  if (ncol(newx) != nrow(object$beta)) {
    stop(sprintf(
      "newx has %d columns; the fitted model expects %d.",
      ncol(newx), nrow(object$beta)
    ))
  }
  newy <- picasso:::.picasso_binomial_response_codes(
    newy, object$levels, nrow(newx), "newy"
  )
  if (is.null(lambda.idx)) {
    lambda.idx <- seq_len(object$nlambda)
  } else {
    lambda.idx <- picasso:::.picasso_multinomial_indices(
      lambda.idx, object$nlambda, "lambda.idx"
    )
  }

  n <- nrow(newx)
  offset <- picasso:::.picasso_prediction_offset(object, newoffset, n)
  beta.sub <- as.matrix(object$beta)[, lambda.idx, drop = FALSE]
  intercept.sub <- as.numeric(object$intercept)[lambda.idx]
  eta <- newx %*% beta.sub +
    matrix(rep(intercept.sub, each = n), nrow = n)
  if (!is.null(offset)) eta <- eta + offset

  predictions <- matrix(as.integer(eta > 0), nrow = n)
  lapply(seq_len(ncol(predictions)), function(k) {
    table(
      predicted = factor(predictions[, k], levels = 0:1),
      actual = factor(newy, levels = 0:1)
    )
  })
}


expect.confusion.bytes <- function(
    object, newx, newy, lambda.idx = NULL, newoffset = NULL) {
  expected <- legacy.binomial.confusion(
    object, newx, newy, lambda.idx, newoffset
  )
  actual <- confusion.picasso(
    object, newx, newy, lambda.idx, newoffset
  )
  # identical() asserts exact values, dims, dimnames, and class.  Comparing
  # serialize() bytes instead is representation-fragile: R 4.6 keeps ALTREP
  # strings inside the table() oracle that older R materialized.
  expect_identical(actual, expected)
  expect_null(names(actual))
  expect_true(all(vapply(actual, inherits, logical(1), what = "table")))
}


expect.confusion.helper.bytes <- function(
    object, newx, newy, lambda.idx, newoffset = NULL, block.bytes) {
  expected <- legacy.binomial.confusion(
    object, newx, newy, lambda.idx, newoffset
  )
  response <- picasso:::.picasso_binomial_response_codes(
    newy, object$levels, nrow(newx), "newy"
  )
  offset <- picasso:::.picasso_prediction_offset(
    object, newoffset, nrow(newx)
  )
  beta <- as.matrix(object$beta[, lambda.idx, drop = FALSE])
  intercept <- as.numeric(object$intercept)[lambda.idx]
  actual <- picasso:::.picasso_binomial_confusion_tables(
    newx, response, beta, intercept, offset, block.bytes
  )
  expect_identical(actual, expected)
  expect_null(names(actual))
}


test_that("binomial confusion column streaming is byte-identical", {
  x <- cbind(
    c(-2, -1, 0, 1, 2, -3, 3),
    c(1, -1, 1, -1, 0, 0.5, -0.5),
    c(0, 2, -2, 1, -1, 3, -3)
  )
  beta <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0.25, -0.5, 0.125
  ), nrow = 3L)
  object <- list(
    family = "binomial",
    levels = c("no", "yes"),
    nlambda = ncol(beta),
    lambda = seq(0.4, 0.1, length.out = ncol(beta)),
    beta = beta,
    intercept = c(0, .Machine$double.xmin, -.Machine$double.xmin, 0.1),
    offset.used = FALSE
  )
  y.numeric <- c(0, 1, 0, 1, 1, 0, 1)
  y.factor <- factor(ifelse(y.numeric == 0, "no", "yes"),
                     levels = object$levels)

  expect.confusion.bytes(object, x, y.numeric)
  expect.confusion.bytes(object, x, y.factor)
  expect.confusion.bytes(object, x, y.factor, 3L)
  expect.confusion.bytes(object, x, y.numeric, c(4L, 1L, 3L))
  expect.confusion.bytes(object, x, y.factor, c(2L, 2L, 4L, 1L))

  missing.class <- rep("yes", nrow(x))
  expect.confusion.bytes(object, x, missing.class, c(1L, 4L))

  sparse.object <- object
  sparse.object$beta <- Matrix::Matrix(beta, sparse = TRUE)
  expect.confusion.bytes(sparse.object, x, y.factor, c(4L, 2L, 4L))

  zero.tables <- confusion.picasso(object, x, y.numeric, 1:3)
  expect_identical(unname(zero.tables[[1L]]["1", ]), c(0L, 0L))
  expect_identical(unname(zero.tables[[2L]]["0", ]), c(0L, 0L))
  expect_identical(unname(zero.tables[[3L]]["1", ]), c(0L, 0L))
})


test_that("binomial confusion streaming preserves offset and validation", {
  x <- matrix(c(-2, -1, 0, 1, 2, 3), ncol = 1L)
  object <- list(
    family = "binomial", levels = c("low", "high"), nlambda = 3L,
    lambda = c(0.3, 0.2, 0.1),
    beta = Matrix::Matrix(matrix(c(0, 0.5, -0.25), nrow = 1L),
                          sparse = TRUE),
    intercept = c(0, 0.1, -0.1), offset.used = TRUE
  )
  y <- factor(c("low", "high", "low", "high", "high", "low"),
              levels = object$levels)
  offset <- c(0, .Machine$double.xmin, -.Machine$double.xmin, 0.2, -0.2, 0)
  expect.confusion.bytes(object, x, y, c(3L, 1L, 3L), offset)
  expect.confusion.bytes(object, x, as.numeric(y) - 1, NULL, offset)

  expect_error(confusion.picasso(object, x, y), "newoffset must be provided")
  expect_error(
    confusion.picasso(object, x, y, newoffset = offset[-1L]),
    "length 6"
  )
  expect_error(confusion.picasso(object, as.numeric(x), y, newoffset = offset),
               "numeric matrix")
  expect_error(confusion.picasso(object, cbind(x, x), y, newoffset = offset),
               "expects 1")
  expect_error(confusion.picasso(object, x, y[-1L], newoffset = offset),
               "length 6")
  expect_error(
    confusion.picasso(object, x, rep("unknown", nrow(x)), newoffset = offset),
    "absent from the fitted model"
  )
  expect_error(confusion.picasso(object, x, y, integer(), offset),
               "at least one index")
  expect_error(confusion.picasso(object, x, y, c(1, NA), offset),
               "finite integer indices")
  expect_error(confusion.picasso(object, x, y, 4L, offset),
               "out-of-range")
})


test_that("binomial confusion uses exact small and blocked paths", {
  x <- cbind(
    c(-2, -1, 0, 1, 2, 3, -3),
    c(1, -1, 1, -1, 0, 0.5, -0.5),
    c(0, 2, -2, 1, -1, 3, -3)
  )
  beta <- matrix(c(
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0.25, -0.5, 0.125,
    -0.4, 0.2, 0.05
  ), nrow = 3L)
  object <- list(
    family = "binomial", levels = c("no", "yes"),
    nlambda = ncol(beta), lambda = seq(0.5, 0.1, length.out = ncol(beta)),
    beta = beta,
    intercept = c(0, .Machine$double.xmin, -.Machine$double.xmin, 0.1, -0.2),
    offset.used = TRUE
  )
  y <- factor(c("no", "yes", "no", "yes", "yes", "no", "yes"),
              levels = object$levels)
  offset <- c(0, .Machine$double.xmin, -.Machine$double.xmin,
              0.2, -0.2, 0, 0.1)
  indices <- c(5L, 1L, 3L, 3L, 2L)
  workspace.bytes <-
    12 * as.double(nrow(x)) * as.double(length(indices))

  # The exact boundary takes the legacy full-matrix path; one byte below it
  # forces blocking while preserving ordering, repeats, and eta > 0 ties.
  expect.confusion.helper.bytes(
    object, x, y, indices, offset, workspace.bytes
  )
  expect.confusion.helper.bytes(
    object, x, y, indices, offset, workspace.bytes - 1
  )

  # A sub-column budget exercises the at-least-one-column guard.
  expect.confusion.helper.bytes(object, x, y, indices, offset, 1)

  sparse.object <- object
  sparse.object$beta <- Matrix::Matrix(beta, sparse = TRUE)
  expect.confusion.helper.bytes(
    sparse.object, x, y, c(4L, 2L, 4L), offset, 1
  )
})


test_that("binomial confusion workspace boundaries select exact block widths", {
  selected.widths <- integer()
  gate.method.name <- "[.picasso_confusion_gate_probe"
  gate.method.existed <- exists(
    gate.method.name, envir = .GlobalEnv, inherits = FALSE
  )
  if (gate.method.existed) {
    old.gate.method <- get(gate.method.name, envir = .GlobalEnv)
  }
  on.exit({
    if (gate.method.existed) {
      assign(gate.method.name, old.gate.method, envir = .GlobalEnv)
    } else if (exists(gate.method.name, envir = .GlobalEnv,
                      inherits = FALSE)) {
      rm(list = gate.method.name, envir = .GlobalEnv)
    }
  }, add = TRUE)
  gate.method <- function(x, i, j, ..., drop = TRUE) {
    selected.widths <<- c(selected.widths, length(j))
    NextMethod("[")
  }
  assign(gate.method.name, gate.method, envir = .GlobalEnv)

  # In a wide problem beta is already materialized.  Its size must not force
  # blocking while the avoidable eta-plus-prediction workspace still fits.
  n <- 2L
  d <- 1000L
  nlambda <- 3L
  x <- matrix(0, nrow = n, ncol = d)
  beta <- matrix(0, nrow = d, ncol = nlambda)
  beta.probe <- structure(
    beta,
    class = c("picasso_confusion_gate_probe", "matrix", "array")
  )
  intercept <- c(-1, 0, 1)
  y <- c(0L, 1L)
  full.bytes <- 12 * as.double(n) * as.double(nlambda)

  full <- picasso:::.picasso_binomial_confusion_tables(
    x, y, beta.probe, intercept, block.bytes = full.bytes
  )
  expect_length(selected.widths, 0L)

  selected.widths <- integer()
  blocked <- picasso:::.picasso_binomial_confusion_tables(
    x, y, beta.probe, intercept, block.bytes = full.bytes - 1
  )
  expect_identical(selected.widths, rep(1L, nlambda))
  expect_identical(blocked, full)

  # Below the full-path boundary, a block contains exactly as many columns as
  # fit in the double predictor plus the double coefficient-column copy.
  n <- 100L
  d <- 2L
  nlambda <- 10L
  x <- matrix(seq_len(n * d) / 100, nrow = n, ncol = d)
  beta <- matrix(seq_len(d * nlambda) / 1000,
                 nrow = d, ncol = nlambda)
  beta.probe <- structure(
    beta,
    class = c("picasso_confusion_gate_probe", "matrix", "array")
  )
  intercept <- seq(-0.1, 0.1, length.out = nlambda)
  y <- rep(0:1, length.out = n)
  block.column.bytes <- 8 * (as.double(n) + as.double(d))

  selected.widths <- integer()
  two.columns <- picasso:::.picasso_binomial_confusion_tables(
    x, y, beta.probe, intercept,
    block.bytes = 2 * block.column.bytes
  )
  expect_identical(selected.widths, rep(2L, nlambda / 2L))

  selected.widths <- integer()
  one.column <- picasso:::.picasso_binomial_confusion_tables(
    x, y, beta.probe, intercept,
    block.bytes = 2 * block.column.bytes - 1
  )
  expect_identical(selected.widths, rep(1L, nlambda))
  expect_identical(one.column, two.columns)
})


test_that("binomial confusion helper validates its workspace budget", {
  x <- matrix(c(-1, 0, 1), ncol = 1L)
  beta <- matrix(c(0, 1), nrow = 1L)
  y <- c(0L, 1L, 1L)

  for (bad in list(0, -1, NA_real_, Inf, numeric())) {
    expect_error(
      picasso:::.picasso_binomial_confusion_tables(
        x, y, beta, c(0, 0), block.bytes = bad
      ),
      "block.bytes"
    )
  }
})


test_that("binary tabulation preserves table bytes for every class pattern", {
  cases <- list(
    list(predicted = c(0L, 0L, 1L, 1L), actual = c(0L, 1L, 0L, 1L)),
    list(predicted = rep(0L, 5L), actual = rep(0L, 5L)),
    list(predicted = rep(1L, 5L), actual = rep(1L, 5L)),
    list(predicted = c(1L, 0L, 1L), actual = rep(0L, 3L)),
    list(predicted = c(0L, 1L, 0L), actual = rep(1L, 3L))
  )

  for (case in cases) {
    expected <- table(
      predicted = factor(case$predicted, levels = 0:1),
      actual = factor(case$actual, levels = 0:1)
    )
    actual <- picasso:::.picasso_binary_confusion_table(
      case$predicted, 2L * case$actual
    )
    expect_identical(actual, expected)
  }
})


test_that("binary tabulation preserves factor labels and subnormal logit ties", {
  tiny <- .Machine$double.xmin * .Machine$double.eps
  x <- matrix(c(-1, 0, 1), ncol = 1L)
  beta <- matrix(c(0, tiny, -tiny), nrow = 1L)
  object <- list(
    family = "binomial", levels = c("negative", "positive"),
    nlambda = 3L, lambda = c(0.3, 0.2, 0.1), beta = beta,
    intercept = c(0, tiny, -tiny), offset.used = FALSE
  )
  y <- factor(c("negative", "positive", "positive"),
              levels = object$levels)

  expect.confusion.bytes(object, x, y, c(2L, 1L, 3L, 2L))
  expect.confusion.bytes(
    object, x, factor(rep("positive", 3L), levels = object$levels),
    c(3L, 1L)
  )
})
