.picasso_multinomial_status_label <- function(code) {
  labels <- c(
    "completed",
    "dfmax_reached",
    "invalid_input",
    "outer_iteration_limit",
    "inner_iteration_limit",
    "line_search_failed",
    "no_descent_direction",
    "numerical_failure",
    "lla_majorization_failed",
    "exception",
    "lla_stationarity_limit"
  )
  if (length(code) != 1L || is.na(code) || code < 0L || code >= length(labels))
    stop(sprintf("Multinomial solver returned unknown status code %s.", code))
  labels[code + 1L]
}


.picasso_multinomial_diagnostics <- function(out, lambda, index,
                                              committed = TRUE) {
  if (length(index) == 0L) {
    return(data.frame(
      lambda.index = integer(), lambda = numeric(), iterations = integer(),
      nonzero = integer(), runtime = numeric(), outer.iterations = integer(),
      inner.sweeps = numeric(), coordinate.updates = numeric(),
      objective = numeric(), smooth.nll = numeric(), kkt = numeric(),
      stationarity = numeric()
    ))
  }
  data.frame(
    lambda.index = as.integer(index),
    lambda = as.numeric(lambda[index]),
    # Transactional solution counters are deliberately unwritten at a failed
    # point. Report NA there rather than presenting their zero sentinels as
    # evidence that the attempted solve performed no work.
    iterations = if (committed)
      as.integer(out$ite_lamb[index]) else rep(NA_integer_, length(index)),
    nonzero = if (committed)
      as.integer(out$size_act[index]) else rep(NA_integer_, length(index)),
    runtime = as.numeric(out$runt[index]),
    outer.iterations = as.integer(out$outer_ite[index]),
    inner.sweeps = as.numeric(out$inner_sweeps[index]),
    coordinate.updates = as.numeric(out$coordinate_updates[index]),
    objective = as.numeric(out$objective[index]),
    smooth.nll = as.numeric(out$smooth_nll[index]),
    kkt = as.numeric(out$kkt[index]),
    stationarity = as.numeric(out$stationarity[index]),
    check.names = FALSE
  )
}


multinomial_solver <- function(Y_int, X, lambda, nlambda, gamma,
                               n, d, K, max.ite,
                               prec, intercept, verbose, method.flag, dfmax,
                               lla.max.stages = 3L,
                               path.early.stop = TRUE)
{
  if (verbose) {
    if (method.flag == 1)
      cat("L1 regularization (multinomial) via Proximal Newton/IRLS\n")
    if (method.flag == 2)
      cat("MCP regularization (multinomial) via Proximal Newton/IRLS\n")
    if (method.flag == 3)
      cat("SCAD regularization (multinomial) via Proximal Newton/IRLS\n")
  }

  out <- .Call("picasso_multinomial_call",
    as.double(Y_int), X,
    as.integer(n), as.integer(d), as.integer(K),
    as.double(lambda), as.integer(nlambda),
    as.double(gamma), as.integer(max.ite),
    as.double(prec), as.integer(method.flag),
    as.integer(intercept),
    as.integer(dfmax),
    as.integer(lla.max.stages),
    as.logical(path.early.stop),
    PACKAGE = "picasso"
  )

  num.fit <- as.integer(out$num_fit[1L])
  status.code <- as.integer(out$status[1L])
  status <- .picasso_multinomial_status_label(status.code)
  if (is.na(num.fit) || num.fit < 0L || num.fit > nlambda)
    stop(sprintf("Multinomial solver returned invalid num_fit=%s.", num.fit))

  failed.zero <- as.integer(out$failed_lambda[1L])
  failed.stage.zero <- as.integer(out$failed_stage[1L])
  failed.index <- if (!is.na(failed.zero) && failed.zero >= 0L)
    failed.zero + 1L else NA_integer_
  failed.stage <- if (!is.na(failed.stage.zero) && failed.stage.zero >= 0L)
    failed.stage.zero + 1L else NA_integer_

  diagnostics <- .picasso_multinomial_diagnostics(
    out, lambda, seq_len(num.fit)
  )
  stationarity.limit <- identical(status.code, 10L)
  path.early.stopped <- status.code %in% c(0L, 10L) &&
    num.fit > 0L && num.fit < nlambda
  hard.failure <- status.code > 1L && !stationarity.limit
  failure <- NULL
  if (hard.failure) {
    failure.diagnostics <- if (!is.na(failed.index) &&
                                failed.index <= nlambda) {
      .picasso_multinomial_diagnostics(
        out, lambda, failed.index, committed = FALSE
      )
    } else {
      NULL
    }
    failure <- list(
      lambda.index = failed.index,
      lambda = if (!is.na(failed.index) && failed.index <= nlambda)
        lambda[failed.index] else NA_real_,
      stage = failed.stage,
      status = status,
      status.code = status.code,
      diagnostics = failure.diagnostics
    )
  }

  if (status.code == 1L && num.fit < 1L)
    stop("Multinomial solver reported dfmax before fitting any lambda values.")
  if (hard.failure) {
    location <- if (is.na(failed.index)) "" else
      sprintf(" at lambda index %d", failed.index)
    stage <- if (is.na(failed.stage)) "" else
      sprintf(", LLA stage %d", failed.stage)
    message <- sprintf(
      "Multinomial solver stopped with status '%s' (code %d)%s%s",
      status, status.code, location, stage
    )
    if (num.fit == 0L)
      stop(paste0(message, " before completing a lambda value."), call. = FALSE)
    warning(sprintf(
      "%s; returning the successful %d/%d-lambda prefix.",
      message, num.fit, nlambda
    ), call. = FALSE)
  }

  return(list(
    beta    = out$beta,          # d * K * nlambda flat
    intcpt  = out$intcpt,        # K * nlambda flat
    ite     = out$ite_lamb[seq_len(num.fit)],
    size.act = out$size_act[seq_len(num.fit)],
    runt    = out$runt[seq_len(num.fit)],
    num.fit = num.fit,
    status = status,
    status.code = status.code,
    failure = failure,
    diagnostics = diagnostics,
    smooth.nll = out$smooth_nll[seq_len(num.fit)],
    path.early.stopped = path.early.stopped,
    requested.nlambda = as.integer(nlambda),
    lla.max.stages = as.integer(lla.max.stages)
  ))
}
