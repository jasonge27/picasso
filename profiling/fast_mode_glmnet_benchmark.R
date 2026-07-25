#!/usr/bin/env Rscript

# Reproducible fast-mode comparison between PICASSO and glmnet.
#
# Usage:
#   Rscript profiling/fast_mode_glmnet_benchmark.R case \
#     <family> <shape> <seed> <out_dir> <picasso_lib> <glmnet_lib>
#   Rscript profiling/fast_mode_glmnet_benchmark.R aggregate <out_dir>
#   Rscript profiling/fast_mode_glmnet_benchmark.R run-all \
#     <out_dir> <picasso_lib> <glmnet_lib>
#   Rscript profiling/fast_mode_glmnet_benchmark.R profile \
#     <family> <shape> <seed> <out_dir> <picasso_lib> <glmnet_lib>
#   Rscript profiling/fast_mode_glmnet_benchmark.R profile-all \
#     <out_dir> <picasso_lib> <glmnet_lib>
#   Rscript profiling/fast_mode_glmnet_benchmark.R native-loop \
#     <family> <shape> <seed> <seconds> <picasso_lib> <glmnet_lib>

options(stringsAsFactors = FALSE, warn = 1)
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)

arguments <- commandArgs(trailingOnly = TRUE)
if (!length(arguments)) stop("A mode is required.")
mode <- arguments[[1L]]

load_packages <- function(picasso_lib, glmnet_lib = NULL) {
  libraries <- c(picasso_lib, glmnet_lib, .libPaths())
  .libPaths(unique(libraries[nzchar(libraries)]))
  suppressPackageStartupMessages(library(picasso))
  if (!is.null(glmnet_lib)) {
    suppressPackageStartupMessages(library(glmnet))
  }
  invisible(NULL)
}

benchmark_environment <- function() {
  file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE),
                        value = TRUE)
  script <- if (length(file_argument) == 1L) {
    normalizePath(sub("^--file=", "", file_argument), mustWork = FALSE)
  } else {
    NA_character_
  }
  repository <- if (!is.na(script)) {
    normalizePath(file.path(dirname(script), ".."), mustWork = FALSE)
  } else {
    NA_character_
  }
  git_output <- function(arguments) {
    if (is.na(repository)) return(character())
    tryCatch(
      suppressWarnings(system2(
        "git", c("-C", repository, arguments), stdout = TRUE, stderr = FALSE
      )),
      error = function(condition) character()
    )
  }
  commit <- git_output(c("rev-parse", "HEAD"))
  status <- git_output(c("status", "--porcelain"))
  loaded <- getLoadedDLLs()
  native_path <- if ("picasso" %in% names(loaded)) {
    loaded[["picasso"]][["path"]]
  } else {
    NA_character_
  }
  glmnet_dll <- grep("glmnet", names(loaded), ignore.case = TRUE, value = TRUE)
  glmnet_path <- if (length(glmnet_dll)) {
    loaded[[glmnet_dll[[1L]]]][["path"]]
  } else {
    NA_character_
  }
  blas <- tryCatch(unname(extSoftVersion()[["BLAS"]]),
                   error = function(condition) NA_character_)
  cpu <- unname(Sys.info()[["machine"]])
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    detected <- tryCatch(
      suppressWarnings(system2(
        "sysctl", c("-n", "machdep.cpu.brand_string"),
        stdout = TRUE, stderr = FALSE
      )),
      error = function(condition) character()
    )
    if (length(detected) && nzchar(detected[[1L]])) cpu <- detected[[1L]]
    if (identical(cpu, unname(Sys.info()[["machine"]]))) {
      hardware <- tryCatch(
        suppressWarnings(system2(
          "system_profiler", "SPHardwareDataType",
          stdout = TRUE, stderr = FALSE
        )),
        error = function(condition) character()
      )
      chip <- grep("^[[:space:]]*Chip:", hardware, value = TRUE)
      if (length(chip)) {
        detected <- sub("^[^:]+:[[:space:]]*", "", chip[[1L]])
        if (nzchar(detected)) cpu <- detected
      }
    }
  } else if (file.exists("/proc/cpuinfo")) {
    cpu_info <- readLines("/proc/cpuinfo", warn = FALSE)
    detected <- sub("^[^:]+:[[:space:]]*", "", grep(
      "^(model name|Hardware)[[:space:]]*:", cpu_info, value = TRUE
    ))
    if (length(detected) && nzchar(detected[[1L]])) cpu <- detected[[1L]]
  }
  thread_names <- c(
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS"
  )
  r_config <- function(name) {
    value <- tryCatch(
      suppressWarnings(system2(
        file.path(R.home("bin"), "R"), c("CMD", "config", name),
        stdout = TRUE, stderr = FALSE
      )),
      error = function(condition) character()
    )
    if (length(value)) paste(value, collapse = " ") else NA_character_
  }
  data.frame(
    git_commit = if (length(commit)) commit[[1L]] else NA_character_,
    git_dirty = if (length(commit)) length(status) > 0L else NA,
    benchmark_script_md5 = if (!is.na(script) && file.exists(script)) {
      unname(tools::md5sum(script))
    } else {
      NA_character_
    },
    native_library_md5 = if (!is.na(native_path) && file.exists(native_path)) {
      unname(tools::md5sum(native_path))
    } else {
      NA_character_
    },
    glmnet_library_md5 = if (!is.na(glmnet_path) && file.exists(glmnet_path)) {
      unname(tools::md5sum(glmnet_path))
    } else {
      NA_character_
    },
    operating_system = paste(
      unname(Sys.info()[c("sysname", "release")]), collapse = " "
    ),
    machine = unname(Sys.info()[["machine"]]),
    cpu = cpu,
    cxx17 = r_config("CXX17"),
    cxx17flags = r_config("CXX17FLAGS"),
    blas = blas %||% NA_character_,
    thread_environment = paste(
      paste0(thread_names, "=", Sys.getenv(thread_names)), collapse = ";"
    )
  )
}

shape_definition <- function(shape) {
  if (identical(shape, "tall")) {
    list(shape = shape, n_train = 4000L, n_val = 1000L, n_test = 2000L,
         p = 120L, rho = 0.20)
  } else if (identical(shape, "wide")) {
    list(shape = shape, n_train = 350L, n_val = 350L, n_test = 1000L,
         p = 2200L, rho = 0.45)
  } else {
    stop("shape must be 'tall' or 'wide'.")
  }
}

ar1_design <- function(n, p, rho) {
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  if (p > 1L && rho != 0) {
    innovation <- sqrt(1 - rho * rho)
    for (j in 2:p) {
      x[, j] <- rho * x[, j - 1L] + innovation * x[, j]
    }
  }
  x
}

standardize_split <- function(x_train, x_val, x_test) {
  center <- colMeans(x_train)
  centered <- sweep(x_train, 2L, center, "-")
  scale <- sqrt(colMeans(centered * centered))
  scale[!is.finite(scale) | scale <= sqrt(.Machine$double.eps)] <- 1
  transform <- function(x) {
    sweep(sweep(x, 2L, center, "-"), 2L, scale, "/")
  }
  list(train = transform(x_train), val = transform(x_val),
       test = transform(x_test))
}

scale_signal <- function(beta, x, target_sd) {
  current <- stats::sd(drop(x %*% beta))
  if (is.finite(current) && current > 0) beta * target_sd / current else beta
}

softmax <- function(eta) {
  shifted <- sweep(eta, 1L, apply(eta, 1L, max), "-")
  exponential <- exp(shifted)
  exponential / rowSums(exponential)
}

sample_multinomial <- function(probability) {
  uniform <- runif(nrow(probability))
  cumulative <- t(apply(probability, 1L, cumsum))
  1L + rowSums(cumulative < uniform)
}

make_split_problem <- function(family, shape, seed, classes = 4L) {
  definition <- shape_definition(shape)
  set.seed(as.integer(seed))
  total <- definition$n_train + definition$n_val + definition$n_test
  raw_x <- ar1_design(total, definition$p, definition$rho)
  train_index <- seq_len(definition$n_train)
  val_index <- definition$n_train + seq_len(definition$n_val)
  test_index <- definition$n_train + definition$n_val +
    seq_len(definition$n_test)
  x <- standardize_split(
    raw_x[train_index, , drop = FALSE],
    raw_x[val_index, , drop = FALSE],
    raw_x[test_index, , drop = FALSE]
  )
  sparse <- min(30L, definition$p)
  beta <- numeric(definition$p)
  beta[seq_len(sparse)] <- rep(c(1, -1), length.out = sparse) *
    seq(1.2, 0.35, length.out = sparse)

  if (family %in% c("gaussian", "sqrtlasso")) {
    beta <- scale_signal(beta, x$train, 2.0)
    all_x <- rbind(x$train, x$val, x$test)
    response <- drop(all_x %*% beta) + rnorm(total)
  } else if (family == "binomial") {
    beta <- scale_signal(beta, x$train, 1.25)
    all_x <- rbind(x$train, x$val, x$test)
    eta <- -0.15 + drop(all_x %*% beta)
    response <- factor(rbinom(total, 1L, plogis(eta)), levels = 0:1)
  } else if (family == "poisson") {
    beta <- scale_signal(beta, x$train, 0.50)
    all_x <- rbind(x$train, x$val, x$test)
    eta <- log(1.7) + drop(all_x %*% beta)
    response <- rpois(total, exp(eta))
    if (sum(response[train_index]) == 0) response[train_index[1L]] <- 1
  } else if (family == "multinomial") {
    beta_matrix <- matrix(0, definition$p, classes)
    raw_beta <- matrix(rnorm(sparse * classes), sparse, classes)
    raw_beta <- sweep(raw_beta, 1L, rowMeans(raw_beta), "-")
    beta_matrix[seq_len(sparse), ] <- raw_beta
    train_eta <- x$train %*% beta_matrix
    signal_sd <- stats::sd(as.numeric(train_eta))
    if (is.finite(signal_sd) && signal_sd > 0) beta_matrix <- beta_matrix / signal_sd
    all_x <- rbind(x$train, x$val, x$test)
    eta <- all_x %*% beta_matrix
    intercept <- seq(-0.2, 0.2, length.out = classes)
    intercept <- intercept - mean(intercept)
    eta <- sweep(eta, 2L, intercept, "+")
    codes <- sample_multinomial(softmax(eta))
    labels <- paste0("c", seq_len(classes))
    response <- factor(codes, levels = seq_len(classes), labels = labels)
  } else {
    stop("Unsupported family: ", family)
  }

  list(
    x_train = x$train, x_val = x$val, x_test = x$test,
    y_train = response[train_index], y_val = response[val_index],
    y_test = response[test_index], definition = definition
  )
}

lambda_maximum <- function(x, y, family) {
  n <- nrow(x)
  if (family == "gaussian") {
    residual <- as.numeric(y) - mean(as.numeric(y))
    max(abs(crossprod(x, residual) / n))
  } else if (family == "sqrtlasso") {
    residual <- as.numeric(y) - mean(as.numeric(y))
    scale <- sqrt(mean(residual * residual))
    if (scale == 0) 0 else max(abs(crossprod(x, residual) / n)) / scale
  } else if (family == "binomial") {
    yy <- as.integer(y) - 1L
    max(abs(crossprod(x, yy - mean(yy)) / n))
  } else if (family == "poisson") {
    yy <- as.numeric(y)
    max(abs(crossprod(x, yy - mean(yy)) / n))
  } else if (family == "multinomial") {
    yy <- stats::model.matrix(~ y - 1)
    residual <- sweep(yy, 2L, colMeans(yy), "-")
    max(abs(crossprod(x, residual) / n))
  } else {
    stop("Unsupported family: ", family)
  }
}

make_lambda <- function(x, y, family, path_length = 45L, ratio = NULL) {
  # The square-root loss becomes numerically delicate much earlier on the
  # wide fixture.  Its path is not cross-package comparable, so use a still
  # informative but safer tail while retaining the shared-family 0.02 tail.
  if (is.null(ratio)) ratio <- if (family == "sqrtlasso") 0.30 else 0.02
  top <- lambda_maximum(x, y, family)
  stopifnot(is.finite(top), top > 0)
  exp(seq(log(top), log(top * ratio), length.out = path_length))
}

fast_precision <- function(family) {
  if (family == "gaussian") 1e-7
  else if (family == "poisson") 4e-4
  else 1e-4
}

fit_picasso <- function(x, y, family, lambda, gaussian_type = "auto") {
  fit <- picasso::picasso(
    X = x, Y = y, family = family, method = "l1", lambda = lambda,
    standardize = FALSE, intercept = TRUE, fast.mode = TRUE,
    max.ite = 10000L, verbose = FALSE, type.gaussian = gaussian_type
  )
  stopifnot(isTRUE(fit$fast.mode))
  expected_precision <- fast_precision(family)
  stopifnot(isTRUE(all.equal(as.numeric(fit$prec), expected_precision)))
  fit
}

fit_glmnet <- function(x, y, family, lambda, gaussian_type = NULL) {
  arguments <- list(
    x = x, y = y, family = family, alpha = 1, lambda = lambda,
    standardize = FALSE, intercept = TRUE,
    penalty.factor = rep(1, ncol(x)), relax = FALSE,
    control = list(
      thresh = 1e-7, maxit = 1000000L, fdev = 0, devmax = 1,
      mnlam = as.integer(length(lambda)), dfmax = as.integer(ncol(x) + 1L),
      pmax = as.integer(ncol(x)), trace.it = 0L
    )
  )
  if (family == "binomial") arguments$type.logistic <- "Newton"
  if (family == "multinomial") arguments$type.multinomial <- "ungrouped"
  if (!is.null(gaussian_type)) arguments$type.gaussian <- gaussian_type
  fit <- do.call(glmnet::glmnet, arguments)
  if (!is.null(fit$jerr)) stopifnot(fit$jerr == 0)
  fit
}

assert_path <- function(fit, lambda, label) {
  actual <- as.numeric(fit$lambda)
  tolerance <- 1e-12 * max(1, max(lambda))
  if (length(actual) != length(lambda) ||
      max(abs(actual - lambda)) > tolerance) {
    stop(sprintf("%s path mismatch: got %d of %d", label,
                 length(actual), length(lambda)))
  }
  invisible(TRUE)
}

extract_solution <- function(fit, package, family, response_levels = NULL) {
  if (package == "picasso") {
    if (family == "multinomial") {
      beta <- array(0, c(nrow(fit$beta[[1L]]), fit$K, fit$nlambda))
      for (class_index in seq_len(fit$K)) {
        beta[, class_index, ] <- as.matrix(fit$beta[[class_index]])
      }
      intercept <- do.call(rbind, fit$intercept)
    } else {
      beta <- as.matrix(fit$beta)
      intercept <- as.numeric(fit$intercept)
    }
  } else {
    if (family == "multinomial") {
      classes <- names(fit$beta)
      order <- match(response_levels, classes)
      if (anyNA(order)) stop("glmnet multinomial class ordering mismatch.")
      beta <- array(0, c(nrow(fit$beta[[1L]]), length(order), length(fit$lambda)))
      for (class_index in seq_along(order)) {
        beta[, class_index, ] <- as.matrix(fit$beta[[order[class_index]]])
      }
      intercept <- as.matrix(fit$a0)[order, , drop = FALSE]
    } else {
      beta <- as.matrix(fit$beta)
      intercept <- as.numeric(fit$a0)
    }
  }
  list(beta = beta, intercept = intercept, lambda = as.numeric(fit$lambda))
}

stable_logistic_loss <- function(eta, y) {
  mean(pmax(eta, 0) + log1p(exp(-abs(eta))) - y * eta)
}

objective_kkt_one <- function(x, y, family, beta, intercept, lambda,
                              active_tolerance = 1e-10) {
  n <- nrow(x)
  if (family == "gaussian") {
    residual <- intercept + drop(x %*% beta) - as.numeric(y)
    loss <- mean(residual * residual) / 2
    gradient <- drop(crossprod(x, residual)) / n
    intercept_gradient <- mean(residual)
  } else if (family == "sqrtlasso") {
    residual <- intercept + drop(x %*% beta) - as.numeric(y)
    scale <- sqrt(mean(residual * residual))
    loss <- scale
    if (scale > 0) {
      gradient <- drop(crossprod(x, residual)) / (n * scale)
      intercept_gradient <- mean(residual) / scale
    } else {
      gradient <- rep(0, ncol(x))
      intercept_gradient <- 0
    }
  } else if (family == "binomial") {
    yy <- as.integer(y) - 1L
    eta <- intercept + drop(x %*% beta)
    probability <- plogis(eta)
    loss <- stable_logistic_loss(eta, yy)
    gradient <- drop(crossprod(x, probability - yy)) / n
    intercept_gradient <- mean(probability - yy)
  } else if (family == "poisson") {
    yy <- as.numeric(y)
    eta <- intercept + drop(x %*% beta)
    mean_value <- exp(pmin(eta, 700))
    loss <- mean(mean_value - yy * eta)
    gradient <- drop(crossprod(x, mean_value - yy)) / n
    intercept_gradient <- mean(mean_value - yy)
  } else if (family == "multinomial") {
    yy <- stats::model.matrix(~ y - 1)
    eta <- sweep(x %*% beta, 2L, intercept, "+")
    probability <- softmax(eta)
    row_maximum <- apply(eta, 1L, max)
    chosen <- eta[cbind(seq_len(n), as.integer(y))]
    loss <- mean(row_maximum + log(rowSums(exp(eta - row_maximum))) - chosen)
    gradient <- crossprod(x, probability - yy) / n
    intercept_gradient <- colMeans(probability - yy)
  } else {
    stop("Unsupported family: ", family)
  }

  beta_matrix <- as.matrix(beta)
  gradient_matrix <- as.matrix(gradient)
  cutoff <- active_tolerance * max(1, max(abs(beta_matrix)))
  active <- abs(beta_matrix) > cutoff
  residual <- matrix(0, nrow(beta_matrix), ncol(beta_matrix))
  residual[active] <- abs(
    gradient_matrix[active] + lambda * sign(beta_matrix[active])
  )
  residual[!active] <- pmax(abs(gradient_matrix[!active]) - lambda, 0)
  absolute_kkt <- max(c(residual, abs(intercept_gradient)))
  normalizer <- max(lambda, max(abs(gradient_matrix)), 1e-12)
  c(
    objective = loss + lambda * sum(abs(beta_matrix)), loss = loss,
    penalty = lambda * sum(abs(beta_matrix)), kkt = absolute_kkt,
    relative_kkt = absolute_kkt / normalizer,
    intercept_kkt = max(abs(intercept_gradient)), nonzero = sum(active)
  )
}

path_diagnostics <- function(solution, x, y, family, package, shape, seed) {
  rows <- vector("list", length(solution$lambda))
  for (lambda_index in seq_along(solution$lambda)) {
    beta <- if (family == "multinomial") {
      solution$beta[, , lambda_index]
    } else {
      solution$beta[, lambda_index]
    }
    intercept <- if (family == "multinomial") {
      solution$intercept[, lambda_index]
    } else {
      solution$intercept[lambda_index]
    }
    diagnostic <- objective_kkt_one(
      x, y, family, beta, intercept, solution$lambda[lambda_index]
    )
    rows[[lambda_index]] <- data.frame(
      family = family, shape = shape, seed = seed, package = package,
      lambda_index = lambda_index, lambda = solution$lambda[lambda_index],
      objective = diagnostic[["objective"]], loss = diagnostic[["loss"]],
      penalty = diagnostic[["penalty"]], kkt = diagnostic[["kkt"]],
      relative_kkt = diagnostic[["relative_kkt"]],
      intercept_kkt = diagnostic[["intercept_kkt"]],
      nonzero = diagnostic[["nonzero"]]
    )
  }
  do.call(rbind, rows)
}

predict_path <- function(solution, x, family) {
  if (family == "multinomial") {
    lapply(seq_len(dim(solution$beta)[3L]), function(lambda_index) {
      softmax(sweep(
        x %*% solution$beta[, , lambda_index], 2L,
        solution$intercept[, lambda_index], "+"
      ))
    })
  } else {
    eta <- sweep(x %*% solution$beta, 2L, solution$intercept, "+")
    if (family == "binomial") plogis(eta)
    else if (family == "poisson") exp(pmin(eta, 700))
    else eta
  }
}

score_prediction <- function(y, prediction, family) {
  if (family %in% c("gaussian", "sqrtlasso")) {
    yy <- as.numeric(y)
    mse <- mean((yy - prediction)^2)
    c(mse = mse, rmse = sqrt(mse), mae = mean(abs(yy - prediction)),
      r2 = 1 - sum((yy - prediction)^2) / sum((yy - mean(yy))^2))
  } else if (family == "binomial") {
    yy <- as.integer(y) - 1L
    probability <- pmin(pmax(prediction, 1e-15), 1 - 1e-15)
    rank_value <- rank(probability)
    n1 <- sum(yy == 1L)
    n0 <- sum(yy == 0L)
    auc <- if (n1 && n0) {
      (sum(rank_value[yy == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
    } else {
      NA_real_
    }
    logloss <- -mean(yy * log(probability) + (1 - yy) * log1p(-probability))
    c(logloss = logloss, brier = mean((probability - yy)^2),
      error = mean((probability >= 0.5) != yy), auc = auc)
  } else if (family == "poisson") {
    yy <- as.numeric(y)
    mean_value <- pmax(prediction, 1e-15)
    unit <- ifelse(yy > 0, yy * log(yy / mean_value) - (yy - mean_value),
                   mean_value)
    c(deviance = 2 * mean(unit),
      nll = mean(mean_value - yy * log(mean_value) + lgamma(yy + 1)),
      mae = mean(abs(yy - mean_value)), mse = mean((yy - mean_value)^2))
  } else if (family == "multinomial") {
    codes <- as.integer(y)
    probability <- pmax(prediction, 1e-15)
    one_hot <- matrix(0, length(codes), ncol(probability))
    one_hot[cbind(seq_along(codes), codes)] <- 1
    c(
      logloss = -mean(log(probability[cbind(seq_along(codes), codes)])),
      brier = mean(rowSums((probability - one_hot)^2)),
      error = mean(max.col(probability, ties.method = "first") != codes)
    )
  } else {
    stop("Unsupported family: ", family)
  }
}

primary_metric <- function(family) {
  if (family %in% c("gaussian", "sqrtlasso")) "mse"
  else if (family == "poisson") "deviance"
  else "logloss"
}

score_path <- function(y, prediction, family) {
  if (family == "multinomial") {
    do.call(rbind, lapply(prediction, function(value) {
      score_prediction(y, value, family)
    }))
  } else {
    do.call(rbind, lapply(seq_len(ncol(prediction)), function(index) {
      score_prediction(y, prediction[, index], family)
    }))
  }
}

elapsed_fit <- function(package, x, y, family, lambda, gaussian_type) {
  start <- proc.time()[["elapsed"]]
  fit <- if (package == "picasso") {
    fit_picasso(x, y, family, lambda, gaussian_type %||% "auto")
  } else {
    fit_glmnet(x, y, family, lambda, gaussian_type)
  }
  elapsed <- proc.time()[["elapsed"]] - start
  list(fit = fit, elapsed = elapsed)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

model_object_mb <- function(fit) {
  # glmnet calls created through do.call() can retain x/y inside fit$call.
  # Remove the captured language object before comparing returned payloads.
  measured <- fit
  if (!is.null(measured$call)) measured$call <- NULL
  as.numeric(object.size(measured)) / 1024^2
}

time_block <- function(package, x, y, family, lambda, gaussian_type,
                       batch_size) {
  gc(FALSE)
  start <- proc.time()[["elapsed"]]
  fit <- NULL
  for (index in seq_len(batch_size)) {
    fit <- if (package == "picasso") {
      fit_picasso(x, y, family, lambda, gaussian_type %||% "auto")
    } else {
      fit_glmnet(x, y, family, lambda, gaussian_type)
    }
  }
  elapsed <- proc.time()[["elapsed"]] - start
  assert_path(fit, lambda, paste(package, family))
  list(fit = fit, per_fit = elapsed / batch_size, block = elapsed)
}

benchmark_pair <- function(problem, family, shape, seed, lambda, config,
                           picasso_type, glmnet_type) {
  x <- problem$x_train
  y <- problem$y_train
  warm_picasso <- elapsed_fit(
    "picasso", x, y, family, lambda, picasso_type
  )
  warm_glmnet <- elapsed_fit(
    "glmnet", x, y, family, lambda, glmnet_type
  )
  assert_path(warm_picasso$fit, lambda, "PICASSO warm-up")
  assert_path(warm_glmnet$fit, lambda, "glmnet warm-up")
  target_seconds <- 0.15
  batch <- c(
    picasso = min(25L, max(1L, ceiling(target_seconds /
      max(warm_picasso$elapsed, 0.001)))),
    glmnet = min(25L, max(1L, ceiling(target_seconds /
      max(warm_glmnet$elapsed, 0.001))))
  )
  order <- c(
    "picasso", "glmnet", "glmnet", "picasso",
    "picasso", "glmnet", "glmnet", "picasso",
    "picasso", "glmnet", "glmnet", "picasso",
    "picasso", "glmnet"
  )
  counts <- c(picasso = 0L, glmnet = 0L)
  rows <- vector("list", length(order))
  last <- list(picasso = warm_picasso$fit, glmnet = warm_glmnet$fit)
  for (order_index in seq_along(order)) {
    package <- order[[order_index]]
    counts[package] <- counts[package] + 1L
    gaussian_type <- if (package == "picasso") picasso_type else glmnet_type
    measured <- time_block(
      package, x, y, family, lambda, gaussian_type, batch[[package]]
    )
    last[[package]] <- measured$fit
    native_seconds <- if (package == "picasso") {
      as.numeric(measured$fit$runtime, units = "secs")
    } else {
      NA_real_
    }
    rows[[order_index]] <- data.frame(
      family = family, shape = shape, seed = seed, config = config,
      package = package, repetition = counts[[package]],
      batch_size = batch[[package]], seconds = measured$per_fit,
      block_seconds = measured$block,
      object_mb = model_object_mb(measured$fit),
      picasso_gaussian_backend = if (
        package == "picasso" && family == "gaussian"
      ) measured$fit$type.gaussian else NA_character_,
      reported_picasso_seconds = native_seconds,
      path_fitted = length(measured$fit$lambda)
    )
  }
  list(timing = do.call(rbind, rows), fits = last)
}

benchmark_picasso_only <- function(problem, family, shape, seed, lambda) {
  warm <- elapsed_fit("picasso", problem$x_train, problem$y_train,
                      family, lambda, "auto")
  assert_path(warm$fit, lambda, "PICASSO warm-up")
  batch <- min(25L, max(1L, ceiling(0.15 / max(warm$elapsed, 0.001))))
  rows <- vector("list", 7L)
  last <- warm$fit
  for (repetition in seq_len(7L)) {
    measured <- time_block(
      "picasso", problem$x_train, problem$y_train, family, lambda,
      "auto", batch
    )
    last <- measured$fit
    rows[[repetition]] <- data.frame(
      family = family, shape = shape, seed = seed,
      config = "fast_explicit_path_auto",
      package = "picasso", repetition = repetition, batch_size = batch,
      seconds = measured$per_fit, block_seconds = measured$block,
      object_mb = model_object_mb(measured$fit),
      picasso_gaussian_backend = NA_character_,
      reported_picasso_seconds = as.numeric(measured$fit$runtime, units = "secs"),
      path_fitted = length(measured$fit$lambda)
    )
  }
  list(timing = do.call(rbind, rows), fits = list(picasso = last))
}

fit_counters <- function(fit, package, family, shape, seed) {
  if (package == "picasso") {
    diagnostics <- fit$diagnostics
    data.frame(
      family = family, shape = shape, seed = seed, package = package,
      total_passes = sum(as.numeric(fit$ite), na.rm = TRUE),
      total_outer_iterations = if (!is.null(diagnostics$outer.iterations))
        sum(diagnostics$outer.iterations, na.rm = TRUE) else NA_real_,
      total_inner_sweeps = if (!is.null(diagnostics$inner.sweeps))
        sum(diagnostics$inner.sweeps, na.rm = TRUE) else NA_real_,
      total_coordinate_updates = if (!is.null(diagnostics$coordinate.updates))
        sum(diagnostics$coordinate.updates, na.rm = TRUE) else NA_real_,
      native_lambda_seconds = if (!is.null(fit[["runt", exact = TRUE]]))
        sum(as.numeric(fit[["runt", exact = TRUE]]), na.rm = TRUE) else NA_real_,
      glmnet_npasses = NA_real_
    )
  } else {
    data.frame(
      family = family, shape = shape, seed = seed, package = package,
      total_passes = NA_real_, total_outer_iterations = NA_real_,
      total_inner_sweeps = NA_real_, total_coordinate_updates = NA_real_,
      native_lambda_seconds = NA_real_, glmnet_npasses = fit$npasses %||% NA_real_
    )
  }
}

run_case <- function(family, shape, seed, out_dir) {
  supported <- c("gaussian", "binomial", "poisson", "sqrtlasso", "multinomial")
  stopifnot(family %in% supported, shape %in% c("tall", "wide"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  problem <- make_split_problem(family, shape, seed)
  requested <- make_lambda(problem$x_train, problem$y_train, family)
  probe <- fit_picasso(
    problem$x_train, problem$y_train, family, requested, "auto"
  )
  lambda <- as.numeric(probe$lambda)
  if (!length(lambda)) stop("PICASSO produced an empty lambda path.")
  message(sprintf(
    "%s/%s seed=%d: common path %d/%d",
    family, shape, seed, length(lambda), length(requested)
  ))

  timing_parts <- list()
  if (family == "sqrtlasso") {
    main <- benchmark_picasso_only(problem, family, shape, seed, lambda)
    timing_parts[[1L]] <- main$timing
    main_fits <- main$fits
  } else {
    main <- benchmark_pair(
      problem, family, shape, seed, lambda,
      "fast_explicit_path_auto", "auto",
      if (family == "gaussian") NULL else NULL
    )
    timing_parts[[1L]] <- main$timing
    main_fits <- main$fits
    if (family == "gaussian" && shape == "tall") {
      matched_naive <- benchmark_pair(
        problem, family, shape, seed, lambda, "matched_naive",
        "naive", "naive"
      )
      matched_covariance <- benchmark_pair(
        problem, family, shape, seed, lambda, "matched_covariance",
        "covariance", "covariance"
      )
      timing_parts[[2L]] <- matched_naive$timing
      timing_parts[[3L]] <- matched_covariance$timing
    }
  }
  timing <- do.call(rbind, timing_parts)

  response_levels <- if (is.factor(problem$y_train)) levels(problem$y_train) else NULL
  solutions <- list()
  diagnostics <- list()
  counters <- list()
  packages <- names(main_fits)
  for (package in packages) {
    solution <- extract_solution(
      main_fits[[package]], package, family, response_levels
    )
    assert_path(main_fits[[package]], lambda, paste(package, "final"))
    solutions[[package]] <- solution
    diagnostics[[package]] <- path_diagnostics(
      solution, problem$x_train, problem$y_train, family, package, shape, seed
    )
    counters[[package]] <- fit_counters(
      main_fits[[package]], package, family, shape, seed
    )
  }

  selected <- list()
  metric_rows <- list()
  validation_paths <- list()
  test_paths <- list()
  metric_index <- 0L
  for (package in packages) {
    validation_paths[[package]] <- predict_path(
      solutions[[package]], problem$x_val, family
    )
    test_paths[[package]] <- predict_path(
      solutions[[package]], problem$x_test, family
    )
    validation_scores <- score_path(
      problem$y_val, validation_paths[[package]], family
    )
    primary <- primary_metric(family)
    chosen <- which.min(validation_scores[, primary])
    test_prediction <- if (family == "multinomial") {
      test_paths[[package]][[chosen]]
    } else {
      test_paths[[package]][, chosen]
    }
    test_scores <- score_prediction(problem$y_test, test_prediction, family)
    selected[[package]] <- data.frame(
      family = family, shape = shape, seed = seed, package = package,
      lambda_index = chosen, lambda = lambda[chosen],
      validation_metric = primary,
      validation_value = validation_scores[chosen, primary]
    )
    for (metric in names(test_scores)) {
      metric_index <- metric_index + 1L
      metric_rows[[metric_index]] <- data.frame(
        family = family, shape = shape, seed = seed, package = package,
        metric = metric, value = unname(test_scores[[metric]])
      )
    }
  }

  comparison <- NULL
  if (all(c("picasso", "glmnet") %in% packages)) {
    picasso_diagnostic <- diagnostics$picasso
    glmnet_diagnostic <- diagnostics$glmnet
    objective_difference <- abs(
      picasso_diagnostic$objective - glmnet_diagnostic$objective
    )
    relative_objective_difference <- objective_difference / pmax(
      1, abs(picasso_diagnostic$objective), abs(glmnet_diagnostic$objective)
    )
    prediction_difference <- if (family == "multinomial") {
      vapply(seq_along(test_paths$picasso), function(index) {
        max(abs(test_paths$picasso[[index]] - test_paths$glmnet[[index]]))
      }, numeric(1))
    } else {
      apply(abs(test_paths$picasso - test_paths$glmnet), 2L, max)
    }
    comparison <- data.frame(
      family = family, shape = shape, seed = seed,
      max_relative_objective_difference = max(relative_objective_difference),
      median_relative_objective_difference = median(relative_objective_difference),
      max_test_path_prediction_difference = max(prediction_difference),
      median_test_path_prediction_difference = median(prediction_difference),
      selected_same_lambda = selected$picasso$lambda_index ==
        selected$glmnet$lambda_index
    )
  }

  result <- list(
    metadata = cbind(data.frame(
      family = family, shape = shape, seed = seed,
      n = problem$definition$n_train, p = problem$definition$p,
      requested_lambda = length(requested), common_lambda = length(lambda),
      picasso_version = as.character(utils::packageVersion("picasso")),
      glmnet_version = if (family == "sqrtlasso") NA_character_ else
        as.character(utils::packageVersion("glmnet")),
      r_version = R.version.string,
      resolved_gaussian_backend = if (family == "gaussian") {
        backend <- main_fits$picasso$type.gaussian
        stopifnot(length(backend) == 1L,
                  backend %in% c("naive", "covariance"))
        backend
      } else {
        NA_character_
      }
    ), benchmark_environment()),
    timing = timing,
    diagnostics = do.call(rbind, diagnostics),
    counters = do.call(rbind, counters),
    selected = do.call(rbind, selected),
    metrics = do.call(rbind, metric_rows),
    comparison = comparison
  )
  file <- file.path(out_dir, sprintf("case_%s_%s_%d.rds", family, shape, seed))
  saveRDS(result, file)
  message("wrote ", file)
  invisible(result)
}

bind_component <- function(results, component) {
  values <- lapply(results, `[[`, component)
  values <- values[!vapply(values, is.null, logical(1))]
  if (!length(values)) NULL else do.call(rbind, values)
}

group_summary <- function(data, keys, value, summary_function) {
  split_data <- split(data, data[keys], drop = TRUE)
  rows <- lapply(split_data, function(part) {
    cbind(part[1L, keys, drop = FALSE], summary_function(part[[value]]))
  })
  rownames_result <- do.call(rbind, rows)
  rownames(rownames_result) <- NULL
  rownames_result
}

aggregate_results <- function(out_dir) {
  files <- Sys.glob(file.path(out_dir, "case_*.rds"))
  if (!length(files)) stop("No case RDS files found in ", out_dir)
  results <- lapply(files, readRDS)
  metadata <- bind_component(results, "metadata")
  timing <- bind_component(results, "timing")
  diagnostics <- bind_component(results, "diagnostics")
  counters <- bind_component(results, "counters")
  # Only multinomial currently records genuine per-lambda native timings.
  # The scalar ActNewton runtime vectors are zero sentinels, and Gaussian has
  # no per-lambda runtime field.
  counters$native_lambda_seconds[
    counters$package == "picasso" & counters$family != "multinomial"
  ] <- NA_real_
  selected <- bind_component(results, "selected")
  metrics <- bind_component(results, "metrics")
  comparison <- bind_component(results, "comparison")

  timing_summary <- group_summary(
    timing, c("family", "shape", "config", "package"), "seconds",
    function(value) data.frame(
      observations = length(value), median_seconds = median(value),
      q1_seconds = unname(stats::quantile(value, 0.25)),
      q3_seconds = unname(stats::quantile(value, 0.75))
    )
  )
  object_summary <- group_summary(
    timing, c("family", "shape", "config", "package"), "object_mb",
    function(value) data.frame(object_mb = median(value))
  )
  timing_summary <- merge(
    timing_summary, object_summary,
    by = c("family", "shape", "config", "package"), sort = FALSE
  )
  picasso_time <- timing_summary[timing_summary$package == "picasso", ]
  glmnet_time <- timing_summary[timing_summary$package == "glmnet", ]
  speed <- merge(
    picasso_time, glmnet_time,
    by = c("family", "shape", "config"), suffixes = c("_picasso", "_glmnet")
  )
  speed$picasso_over_glmnet <- speed$median_seconds_picasso /
    speed$median_seconds_glmnet

  kkt_summary <- group_summary(
    diagnostics, c("family", "shape", "package"), "relative_kkt",
    function(value) data.frame(
      max_relative_kkt = max(value), median_relative_kkt = median(value)
    )
  )
  absolute_kkt <- group_summary(
    diagnostics, c("family", "shape", "package"), "kkt",
    function(value) data.frame(max_absolute_kkt = max(value))
  )
  kkt_summary <- merge(
    kkt_summary, absolute_kkt,
    by = c("family", "shape", "package"), sort = FALSE
  )

  metric_summary <- group_summary(
    metrics, c("family", "shape", "package", "metric"), "value",
    function(value) data.frame(mean = mean(value), sd = stats::sd(value))
  )
  metric_picasso <- metric_summary[metric_summary$package == "picasso", ]
  metric_glmnet <- metric_summary[metric_summary$package == "glmnet", ]
  metric_comparison <- merge(
    metric_picasso, metric_glmnet,
    by = c("family", "shape", "metric"),
    suffixes = c("_picasso", "_glmnet")
  )
  metric_comparison$difference_picasso_minus_glmnet <-
    metric_comparison$mean_picasso - metric_comparison$mean_glmnet

  write.csv(metadata, file.path(out_dir, "metadata.csv"), row.names = FALSE)
  write.csv(timing, file.path(out_dir, "runtime_raw.csv"), row.names = FALSE)
  write.csv(timing_summary, file.path(out_dir, "runtime_summary.csv"), row.names = FALSE)
  write.csv(speed, file.path(out_dir, "runtime_comparison.csv"), row.names = FALSE)
  write.csv(diagnostics, file.path(out_dir, "kkt_objective_path.csv"), row.names = FALSE)
  write.csv(kkt_summary, file.path(out_dir, "kkt_summary.csv"), row.names = FALSE)
  write.csv(counters, file.path(out_dir, "solver_counters.csv"), row.names = FALSE)
  write.csv(selected, file.path(out_dir, "selected_models.csv"), row.names = FALSE)
  write.csv(metrics, file.path(out_dir, "prediction_metrics_raw.csv"), row.names = FALSE)
  write.csv(metric_summary, file.path(out_dir, "prediction_metrics_summary.csv"), row.names = FALSE)
  write.csv(metric_comparison, file.path(out_dir, "prediction_comparison.csv"), row.names = FALSE)
  write.csv(comparison, file.path(out_dir, "path_parity.csv"), row.names = FALSE)

  cat("\nRuntime comparison (PICASSO / glmnet; >1 means glmnet is faster):\n")
  print(speed[, c("family", "shape", "config", "median_seconds_picasso",
                  "median_seconds_glmnet", "picasso_over_glmnet")],
        row.names = FALSE)
  cat("\nExternal KKT summary:\n")
  print(kkt_summary, row.names = FALSE)
  invisible(list(speed = speed, kkt = kkt_summary,
                 metrics = metric_comparison, parity = comparison))
}

run_all_cases <- function(out_dir, picasso_lib, glmnet_lib) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE),
                        value = TRUE)
  if (length(file_argument) != 1L) stop("Cannot determine benchmark script path.")
  script <- normalizePath(sub("^--file=", "", file_argument))
  executable <- file.path(R.home("bin"), "Rscript")
  families <- c("gaussian", "binomial", "poisson", "sqrtlasso", "multinomial")
  shapes <- c("tall", "wide")
  seeds <- 74001:74003
  total <- length(families) * length(shapes) * length(seeds)
  completed <- 0L
  for (family in families) {
    for (shape in shapes) {
      for (seed in seeds) {
        completed <- completed + 1L
        message(sprintf("[%d/%d] %s/%s seed=%d", completed, total,
                        family, shape, seed))
        status <- system2(
          executable,
          c("--vanilla", script, "case", family, shape, as.character(seed),
            out_dir, picasso_lib, glmnet_lib)
        )
        if (!identical(status, 0L)) {
          stop(sprintf("Case failed: %s/%s seed=%d", family, shape, seed))
        }
      }
    }
  }
  status <- system2(executable, c("--vanilla", script, "aggregate", out_dir))
  if (!identical(status, 0L)) stop("Aggregation failed.")
  invisible(NULL)
}

run_all_profiles <- function(out_dir, picasso_lib, glmnet_lib) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE),
                        value = TRUE)
  if (length(file_argument) != 1L) stop("Cannot determine benchmark script path.")
  script <- normalizePath(sub("^--file=", "", file_argument))
  executable <- file.path(R.home("bin"), "Rscript")
  families <- c("gaussian", "binomial", "poisson", "sqrtlasso", "multinomial")
  shapes <- c("tall", "wide")
  completed <- 0L
  for (family in families) {
    for (shape in shapes) {
      completed <- completed + 1L
      seed <- if (family == "sqrtlasso" && shape == "wide") 74002L else 74001L
      message(sprintf("[profile %d/10] %s/%s seed=%d", completed,
                      family, shape, seed))
      status <- system2(
        executable,
        c("--vanilla", script, "profile", family, shape, as.character(seed),
          out_dir, picasso_lib, glmnet_lib)
      )
      if (!identical(status, 0L)) {
        stop(sprintf("Profile failed: %s/%s", family, shape))
      }
    }
  }
  layer_files <- Sys.glob(file.path(out_dir, "layer_*.csv"))
  layer <- do.call(rbind, lapply(layer_files, read.csv))
  layer_summary <- group_summary(
    layer, c("family", "shape", "layer"), "seconds",
    function(value) data.frame(
      median_seconds = median(value),
      q1_seconds = unname(stats::quantile(value, 0.25)),
      q3_seconds = unname(stats::quantile(value, 0.75))
    )
  )
  public <- layer_summary[layer_summary$layer == "public", ]
  direct <- layer_summary[layer_summary$layer == "direct", ]
  comparison <- merge(public, direct, by = c("family", "shape"),
                      suffixes = c("_public", "_direct"))
  comparison$wrapper_seconds <- pmax(
    0, comparison$median_seconds_public - comparison$median_seconds_direct
  )
  comparison$direct_fraction <- comparison$median_seconds_direct /
    comparison$median_seconds_public
  write.csv(layer, file.path(out_dir, "layer_raw.csv"), row.names = FALSE)
  write.csv(layer_summary, file.path(out_dir, "layer_summary.csv"), row.names = FALSE)
  write.csv(comparison, file.path(out_dir, "layer_comparison.csv"), row.names = FALSE)
  print(comparison[, c("family", "shape", "median_seconds_public",
                       "median_seconds_direct", "wrapper_seconds",
                       "direct_fraction")], row.names = FALSE)
  invisible(NULL)
}

direct_picasso_fit <- function(problem, family, lambda,
                               gaussian_type = "naive") {
  x <- problem$x_train
  n <- nrow(x)
  d <- ncol(x)
  y <- problem$y_train
  if (family == "gaussian") {
    picasso:::gaussian_solver(
      as.double(y), x, lambda, length(lambda), 3, n, d, 10000L,
      fast_precision(family),
      FALSE, TRUE, 1L, gaussian_type, -1L
    )
  } else if (family == "binomial") {
    yy <- as.integer(y) - 1L
    picasso:::logit_solver(
      yy, x, lambda, length(lambda), 3, n, d, 10000L,
      fast_precision(family),
      TRUE, FALSE, 1L, -1L, rep(0, n), 3L
    )
  } else if (family == "poisson") {
    picasso:::poisson_solver(
      as.double(y), x, lambda, length(lambda), 3, n, d, 10000L,
      fast_precision(family),
      TRUE, FALSE, 1L, -1L, rep(0, n), 3L
    )
  } else if (family == "sqrtlasso") {
    picasso:::sqrtlasso_solver(
      as.double(y), x, lambda, length(lambda), 3, n, d, 10000L,
      fast_precision(family),
      TRUE, FALSE, 1L, -1L, 3L
    )
  } else if (family == "multinomial") {
    yy <- as.integer(y) - 1L
    picasso:::multinomial_solver(
      yy, x, lambda, length(lambda), 3, n, d, nlevels(y), 10000L,
      fast_precision(family), TRUE, FALSE, 1L, -1L, 3L, FALSE
    )
  } else {
    stop("Unsupported family: ", family)
  }
}

profile_runtime <- function(family, shape, seed, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  problem <- make_split_problem(family, shape, seed)
  requested <- make_lambda(problem$x_train, problem$y_train, family)
  probe <- fit_picasso(
    problem$x_train, problem$y_train, family, requested, "auto"
  )
  lambda <- as.numeric(probe$lambda)
  resolved_gaussian_type <- if (family == "gaussian") {
    probe$type.gaussian
  } else {
    "naive"
  }
  public_once <- elapsed_fit(
    "picasso", problem$x_train, problem$y_train, family, lambda, "auto"
  )
  start <- proc.time()[["elapsed"]]
  invisible(direct_picasso_fit(
    problem, family, lambda, resolved_gaussian_type
  ))
  direct_warm <- proc.time()[["elapsed"]] - start
  public_batch <- min(25L, max(1L, ceiling(0.20 /
    max(public_once$elapsed, 0.001))))
  direct_batch <- min(25L, max(1L, ceiling(0.20 / max(direct_warm, 0.001))))

  time_expression <- function(kind, batch) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    value <- NULL
    for (index in seq_len(batch)) {
      value <- if (kind == "public") {
        fit_picasso(
          problem$x_train, problem$y_train, family, lambda, "auto"
        )
      } else {
        direct_picasso_fit(
          problem, family, lambda, resolved_gaussian_type
        )
      }
    }
    (proc.time()[["elapsed"]] - start) / batch
  }
  profile_rows <- list()
  row_index <- 0L
  order <- rep(c("public", "direct", "direct", "public"), 4L)
  for (kind in order) {
    row_index <- row_index + 1L
    batch <- if (kind == "public") public_batch else direct_batch
    profile_rows[[row_index]] <- data.frame(
      family = family, shape = shape, seed = seed, layer = kind,
      seconds = time_expression(kind, batch), batch_size = batch
    )
  }
  rows <- do.call(rbind, profile_rows)

  rprof_file <- file.path(out_dir, sprintf("rprof_%s_%s_%d.out", family, shape, seed))
  iterations <- max(5L, ceiling(2 / max(public_once$elapsed, 0.001)))
  Rprof(rprof_file, interval = 0.001)
  for (index in seq_len(iterations)) {
    invisible(fit_picasso(
      problem$x_train, problem$y_train, family, lambda, "auto"
    ))
  }
  Rprof(NULL)
  summary <- summaryRprof(rprof_file)
  by_total <- head(summary$by.total, 30L)
  by_total$function_name <- rownames(by_total)
  rownames(by_total) <- NULL
  by_total$family <- family
  by_total$shape <- shape
  by_total$seed <- seed
  write.csv(rows, file.path(
    out_dir, sprintf("layer_%s_%s_%d.csv", family, shape, seed)
  ), row.names = FALSE)
  write.csv(by_total, file.path(
    out_dir, sprintf("rprof_%s_%s_%d.csv", family, shape, seed)
  ), row.names = FALSE)
  profile_metadata <- cbind(data.frame(
    family = family, shape = shape, seed = seed,
    resolved_gaussian_backend = if (family == "gaussian") {
      resolved_gaussian_type
    } else {
      NA_character_
    }
  ), benchmark_environment())
  write.csv(profile_metadata, file.path(
    out_dir, sprintf("profile_metadata_%s_%s_%d.csv", family, shape, seed)
  ), row.names = FALSE)
  invisible(rows)
}

native_loop <- function(family, shape, seed, seconds) {
  problem <- make_split_problem(family, shape, seed)
  requested <- make_lambda(problem$x_train, problem$y_train, family)
  probe <- fit_picasso(
    problem$x_train, problem$y_train, family, requested, "auto"
  )
  lambda <- as.numeric(probe$lambda)
  cat(sprintf("PROFILE_PID=%d\n", Sys.getpid()))
  flush.console()
  deadline <- proc.time()[["elapsed"]] + seconds
  fits <- 0L
  while (proc.time()[["elapsed"]] < deadline) {
    invisible(fit_picasso(
      problem$x_train, problem$y_train, family, lambda, "auto"
    ))
    fits <- fits + 1L
  }
  cat(sprintf("PROFILE_FITS=%d\n", fits))
  invisible(NULL)
}

if (mode == "case") {
  if (length(arguments) != 7L) stop("Invalid case arguments.")
  family <- arguments[[2L]]
  shape <- arguments[[3L]]
  seed <- as.integer(arguments[[4L]])
  out_dir <- arguments[[5L]]
  load_packages(arguments[[6L]], arguments[[7L]])
  run_case(family, shape, seed, out_dir)
} else if (mode == "aggregate") {
  if (length(arguments) != 2L) stop("Invalid aggregate arguments.")
  aggregate_results(arguments[[2L]])
} else if (mode == "run-all") {
  if (length(arguments) != 4L) stop("Invalid run-all arguments.")
  run_all_cases(arguments[[2L]], arguments[[3L]], arguments[[4L]])
} else if (mode == "profile") {
  if (length(arguments) != 7L) stop("Invalid profile arguments.")
  family <- arguments[[2L]]
  shape <- arguments[[3L]]
  seed <- as.integer(arguments[[4L]])
  out_dir <- arguments[[5L]]
  load_packages(arguments[[6L]], arguments[[7L]])
  profile_runtime(family, shape, seed, out_dir)
} else if (mode == "profile-all") {
  if (length(arguments) != 4L) stop("Invalid profile-all arguments.")
  run_all_profiles(arguments[[2L]], arguments[[3L]], arguments[[4L]])
} else if (mode == "native-loop") {
  if (length(arguments) != 7L) stop("Invalid native-loop arguments.")
  family <- arguments[[2L]]
  shape <- arguments[[3L]]
  seed <- as.integer(arguments[[4L]])
  seconds <- as.numeric(arguments[[5L]])
  load_packages(arguments[[6L]], arguments[[7L]])
  native_loop(family, shape, seed, seconds)
} else {
  stop("Unknown mode: ", mode)
}
