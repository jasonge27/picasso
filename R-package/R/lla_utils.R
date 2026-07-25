.picasso_validate_lla_max_stages <- function(value) {
  if (length(value) != 1L || !is.numeric(value) || is.na(value) ||
      !is.finite(value) || value < 3 || value != floor(value) ||
      value > .Machine$integer.max) {
    stop("lla.max.stages must be a finite integer greater than or equal to 3.")
  }
  as.integer(value)
}


.picasso_scalar_lla_status_label <- function(code) {
  labels <- c(
    "completed",
    "dfmax_reached",
    "invalid_input",
    "subproblem_failed",
    "inner_iteration_limit",
    "line_search_failed",
    "no_descent_direction",
    "numerical_failure",
    "lla_majorization_failed",
    "exception",
    "lla_stationarity_limit",
    "interrupted"
  )
  if (length(code) != 1L || is.na(code) || code < 0L || code >= length(labels))
    stop(sprintf("Adaptive LLA solver returned unknown status code %s.", code))
  labels[code + 1L]
}


# The native path loops absorb a pending Ctrl-C via R_ToplevelExec and stop
# at the next lambda boundary with status code 11. Re-signal the interrupt
# here so the whole call chain (including cv.picasso fold loops) aborts the
# way an uncaught Ctrl-C would, and tryCatch(interrupt = ...) still works.
.picasso_signal_interrupt <- function() {
  stop(structure(
    class = c("interrupt", "condition"),
    list(message = "picasso fit interrupted by user.", call = NULL)
  ))
}


.picasso_scalar_lla_diagnostics <- function(out, lambda, index,
                                             committed = TRUE) {
  if (length(index) == 0L) {
    return(data.frame(
      lambda.index = integer(), lambda = numeric(), iterations = integer(),
      nonzero = integer(), runtime = numeric(), lla.stages = integer(),
      objective = numeric(), smooth.objective = numeric(), kkt = numeric(),
      stationarity = numeric()
    ))
  }
  data.frame(
    lambda.index = as.integer(index),
    lambda = as.numeric(lambda[index]),
    iterations = if (committed)
      as.integer(out$ite_lamb[index]) else rep(NA_integer_, length(index)),
    nonzero = if (committed)
      as.integer(out$size_act[index]) else rep(NA_integer_, length(index)),
    runtime = as.numeric(out$runt[index]),
    lla.stages = as.integer(out$lla_stages[index]),
    objective = as.numeric(out$objective[index]),
    smooth.objective = as.numeric(out$smooth_objective[index]),
    kkt = as.numeric(out$kkt[index]),
    stationarity = as.numeric(out$stationarity[index]),
    check.names = FALSE
  )
}


.picasso_scalar_lla_result <- function(out, lambda, nlambda,
                                        lla.max.stages, family.label) {
  num.fit <- as.integer(out$num_fit[1L])
  status.code <- as.integer(out$status[1L])
  status <- .picasso_scalar_lla_status_label(status.code)
  if (identical(status.code, 11L)) .picasso_signal_interrupt()
  if (is.na(num.fit) || num.fit < 0L || num.fit > nlambda)
    stop(sprintf("%s solver returned invalid num_fit=%s.", family.label, num.fit))

  failed.zero <- as.integer(out$failed_lambda[1L])
  failed.stage.zero <- as.integer(out$failed_stage[1L])
  failed.index <- if (!is.na(failed.zero) && failed.zero >= 0L)
    failed.zero + 1L else NA_integer_
  failed.stage <- if (!is.na(failed.stage.zero) && failed.stage.zero >= 0L)
    failed.stage.zero + 1L else NA_integer_
  diagnostics <- .picasso_scalar_lla_diagnostics(
    out, lambda, seq_len(num.fit)
  )

  usable.status <- status.code %in% c(0L, 1L, 10L)
  failure <- NULL
  if (!usable.status) {
    failure.diagnostics <- if (!is.na(failed.index) &&
                                failed.index <= nlambda) {
      .picasso_scalar_lla_diagnostics(
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
    location <- if (is.na(failed.index)) "" else
      sprintf(" at lambda index %d", failed.index)
    stage <- if (is.na(failed.stage)) "" else
      sprintf(", LLA stage %d", failed.stage)
    message <- sprintf(
      "%s solver stopped with status '%s' (code %d)%s%s",
      family.label, status, status.code, location, stage
    )
    if (num.fit == 0L)
      stop(paste0(message, " before completing a lambda value."), call. = FALSE)
    warning(sprintf(
      "%s; returning the successful %d/%d-lambda prefix.",
      message, num.fit, nlambda
    ), call. = FALSE)
  }
  if (usable.status && num.fit == 0L)
    stop(sprintf("%s solver returned no usable lambda values.", family.label))

  list(
    beta = out$beta,
    intcpt = out$intcpt[seq_len(num.fit)],
    ite = out$ite_lamb[seq_len(num.fit)],
    size.act = out$size_act[seq_len(num.fit)],
    runt = out$runt[seq_len(num.fit)],
    num.fit = num.fit,
    status = status,
    status.code = status.code,
    failure = failure,
    diagnostics = diagnostics,
    smooth.objective = out$smooth_objective[seq_len(num.fit)],
    lla.max.stages = as.integer(lla.max.stages)
  )
}
