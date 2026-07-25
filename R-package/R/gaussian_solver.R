gaussian_solver <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,
                    verbose, intercept, method.flag, type.gaussian, dfmax)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via active set identification and coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization via active set identification and coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization via active set identification and coordinate descent\n")
  }

  if (type.gaussian == "covariance") {
    out <- .Call("picasso_gaussian_cov_call",
      Y, X,
      as.integer(n), as.integer(d),
      lambda, as.integer(nlambda),
      as.double(gamma), as.integer(max.ite), as.double(prec),
      as.integer(method.flag), as.integer(intercept),
      as.integer(dfmax),
      PACKAGE = "picasso"
    )
  } else {
    out <- .Call("picasso_gaussian_naive_call",
      Y, X,
      as.integer(n), as.integer(d),
      lambda, as.integer(nlambda),
      as.double(gamma), as.integer(max.ite), as.double(prec),
      as.integer(method.flag), as.integer(intercept),
      as.integer(dfmax),
      PACKAGE = "picasso"
    )
  }

  num.fit <- as.integer(out$num_fit[1L])
  status.code <- as.integer(out$status[1L])
  status <- .picasso_scalar_lla_status_label(status.code)
  if (identical(status.code, 11L)) .picasso_signal_interrupt()
  if (is.na(num.fit) || num.fit < 0L || num.fit > nlambda) {
    stop(sprintf("Gaussian solver returned invalid num_fit=%s.", num.fit))
  }
  failed.zero <- as.integer(out$failed_lambda[1L])
  failed.index <- if (!is.na(failed.zero) && failed.zero >= 0L) {
    failed.zero + 1L
  } else {
    NA_integer_
  }
  usable.status <- status.code %in% c(0L, 1L)
  failure <- NULL
  if (!usable.status) {
    failure <- list(
      lambda.index = failed.index,
      lambda = if (!is.na(failed.index) && failed.index <= nlambda) {
        lambda[failed.index]
      } else {
        NA_real_
      },
      status = status,
      status.code = status.code
    )
    location <- if (is.na(failed.index)) "" else {
      sprintf(" at lambda index %d", failed.index)
    }
    message <- sprintf(
      "Gaussian solver stopped with status '%s' (code %d)%s",
      status, status.code, location
    )
    if (num.fit == 0L) {
      stop(paste0(message, " before completing a lambda value."),
           call. = FALSE)
    }
    warning(sprintf(
      "%s; returning the successful %d/%d-lambda prefix.",
      message, num.fit, nlambda
    ), call. = FALSE)
  }
  if (usable.status && num.fit == 0L) {
    stop("Gaussian solver returned no usable lambda values.", call. = FALSE)
  }

  return(list(
    beta = out$beta,
    intcpt = out$intcpt,
    ite = out$ite_lamb,
    smooth.objective = out$smooth_objective,
    num.fit = num.fit,
    status = status,
    status.code = status.code,
    failure = failure,
    err = status.code
  ))
}
