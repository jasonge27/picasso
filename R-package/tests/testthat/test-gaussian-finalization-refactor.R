gaussian_refactor_contract_names <- c(
  "beta", "intercept", "lambda", "df", "ite", "status", "status.code",
  "nlambda", "gamma", "method", "type.gaussian.requested",
  "type.gaussian", "alg", "verbose", "runtime", "fast.mode", "prec",
  "nulldev", "dev.ratio", "family"
)


legacy_gaussian_finalize <- function(flat.beta, d, nlambda, standardize,
                                     multiplier) {
  beta.raw <- matrix(0.0, nrow = d, ncol = nlambda)
  beta.raw[] <- flat.beta[seq_len(d * nlambda)]
  beta <- matrix(0.0, nrow = d, ncol = nlambda)
  beta[seq_len(d), ] <- if (standardize) {
    sweep(beta.raw, 1L, multiplier, `*`)
  } else {
    beta.raw
  }
  list(
    beta = Matrix::Matrix(beta),
    df = as.integer(colSums(beta != 0))
  )
}


refactored_gaussian_finalize <- function(flat.beta, d, nlambda, standardize,
                                         multiplier) {
  beta.raw <- matrix(
    flat.beta[seq_len(d * nlambda)], nrow = d, ncol = nlambda
  )
  beta <- if (standardize) beta.raw * multiplier else beta.raw
  list(
    beta = Matrix::Matrix(beta),
    df = as.integer(colSums(beta != 0))
  )
}


test_that("Gaussian finalization is serialization-identical to the legacy path", {
  d <- 7L
  requested.nlambda <- 9L
  flat.beta <- rep(c(0, -3, 2, 0, 1e-200, -1e-120, 4), requested.nlambda)
  multiplier <- c(0, 1, 0.5, 2, 1e120, 1e-120, 3)

  for (nlambda in c(1L, 5L, requested.nlambda)) {
    for (standardize in c(FALSE, TRUE)) {
      legacy <- legacy_gaussian_finalize(
        flat.beta, d, nlambda, standardize, multiplier
      )
      refactored <- refactored_gaussian_finalize(
        flat.beta, d, nlambda, standardize, multiplier
      )
      expect_identical(
        serialize(refactored, NULL, version = 3L),
        serialize(legacy, NULL, version = 3L),
        info = paste("nlambda", nlambda, "standardize", standardize)
      )
      expect_identical(refactored$df, as.integer(colSums(
        as.matrix(refactored$beta) != 0
      )))
    }
  }
})


test_that("Gaussian public result fields and shapes survive finalization cleanup", {
  index <- seq_len(48L)
  x.integer <- cbind(
    constant = rep.int(7L, length(index)),
    linear = as.integer(index - 24L),
    alternating = rep(c(-2L, 3L), length.out = length(index)),
    zero = integer(length(index))
  )
  y <- 2.75 + 0.04 * x.integer[, "linear"] -
    0.3 * x.integer[, "alternating"]

  for (type.gaussian in c("naive", "covariance")) {
    for (standardize in c(FALSE, TRUE)) {
      for (intercept in c(FALSE, TRUE)) {
        label <- paste(type.gaussian, standardize, intercept)
        fit <- picasso(
          x.integer, y, family = "gaussian", lambda = 0.25,
          type.gaussian = type.gaussian, standardize = standardize,
          intercept = intercept, prec = 1e-10, max.ite = 10000L
        )

        expect_identical(names(fit), gaussian_refactor_contract_names,
                         info = label)
        expect_s3_class(fit, "gaussian", exact = TRUE)
        expect_s4_class(fit$beta, "Matrix")
        expect_identical(dim(fit$beta), c(ncol(x.integer), 1L), info = label)
        expect_identical(length(fit$intercept), 1L, info = label)
        expect_identical(fit$nlambda, 1L, info = label)
        expect_identical(fit$status, "completed", info = label)
        expect_identical(fit$status.code, 0L, info = label)
        expect_null(fit$failure, info = label)
        expect_identical(fit$df, as.integer(colSums(
          as.matrix(fit$beta) != 0
        )), info = label)
        expected.intercept <- if (intercept) {
          as.numeric(
            mean(y) - crossprod(colMeans(x.integer), as.matrix(fit$beta))
          )
        } else {
          0.0
        }
        expect_equal(
          fit$intercept, expected.intercept, tolerance = 1e-12, info = label
        )
        expect_true(all(is.finite(c(
          as.matrix(fit$beta), fit$intercept, fit$dev.ratio
        ))), info = label)
      }
    }
  }

  design <- picasso:::.picasso_prepare_gaussian_design(
    x.integer, standardize = TRUE, intercept = TRUE
  )
  expect_named(design, c("xx", "xm", "xinvc.vec"), ignore.order = FALSE)
})


test_that("Gaussian dfmax returns a compact, contract-preserving prefix", {
  set.seed(20260719)
  n <- 80L
  d <- 12L
  x <- matrix(rnorm(n * d), nrow = n)
  y <- x[, 1L] + 0.8 * x[, 2L] + 0.6 * x[, 3L] + rnorm(n, sd = 0.1)
  requested.nlambda <- 40L

  for (type.gaussian in c("naive", "covariance")) {
    for (standardize in c(FALSE, TRUE)) {
      for (intercept in c(FALSE, TRUE)) {
        label <- paste(type.gaussian, standardize, intercept)
        fit <- picasso(
          x, y, family = "gaussian", nlambda = requested.nlambda,
          lambda.min.ratio = 0.001, dfmax = 1L,
          type.gaussian = type.gaussian, standardize = standardize,
          intercept = intercept, prec = 1e-8, max.ite = 10000L
        )

        expect_lt(fit$nlambda, requested.nlambda)
        expect_gt(fit$nlambda, 0L)
        expect_identical(fit$status, "dfmax_reached", info = label)
        expect_identical(fit$status.code, 1L, info = label)
        expect_null(fit$failure, info = label)
        expect_identical(names(fit), gaussian_refactor_contract_names,
                         info = label)
        expect_identical(dim(fit$beta), c(d, fit$nlambda), info = label)
        expect_identical(length(fit$intercept), fit$nlambda, info = label)
        expect_identical(length(fit$lambda), fit$nlambda, info = label)
        expect_identical(length(fit$ite), fit$nlambda, info = label)
        expect_identical(length(fit$dev.ratio), fit$nlambda, info = label)
        expect_identical(fit$df, as.integer(colSums(
          as.matrix(fit$beta) != 0
        )), info = label)
      }
    }
  }
})
