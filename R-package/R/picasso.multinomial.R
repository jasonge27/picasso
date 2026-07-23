.picasso_multinomial_indices <- function(index, n, name, default.length = 3L) {
  .picasso_indices(index, n, name, default.length = default.length)
}


.picasso_multinomial_newdata <- function(object, newdata) {
  if (is.null(newdata) || !is.numeric(newdata) || length(dim(newdata)) != 2L)
    stop("`newdata` must be a numeric matrix.")
  if (nrow(newdata) == 0L)
    stop("`newdata` must contain at least one row.")
  if (anyNA(newdata) || any(!is.finite(newdata)))
    stop("`newdata` must contain only finite values.")
  expected <- nrow(object$beta[[1L]])
  if (ncol(newdata) != expected)
    stop(sprintf(
      "`newdata` has %d columns; the fitted model expects %d.",
      ncol(newdata), expected
    ))
  newdata <- as.matrix(newdata)
  if (!is.double(newdata))
    storage.mode(newdata) <- "double"
  newdata
}


.picasso_multinomial_response_codes <- function(y, levels, n = NULL,
                                                 name = "response") {
  if (!is.null(n) && length(y) != n)
    stop(sprintf("`%s` must have length %d.", name, n))
  if (length(y) == 0L || anyNA(y))
    stop(sprintf("`%s` must be nonempty and contain no missing values.", name))
  labels <- as.character(y)
  codes <- match(labels, levels)
  if (anyNA(codes)) {
    unknown <- unique(labels[is.na(codes)])
    stop(sprintf(
      "`%s` contains class values absent from the fitted model: %s.",
      name, paste(unknown, collapse = ", ")
    ))
  }
  codes
}


.picasso_multinomial_softmax <- function(logits) {
  row.maximum <- apply(logits, 1L, max)
  probabilities <- exp(logits - row.maximum)
  probabilities <- probabilities / rowSums(probabilities)
  if (any(!is.finite(probabilities)))
    stop("Multinomial prediction produced non-finite probabilities.")
  probabilities
}


.picasso_multinomial_logits_one <- function(object, newdata, lambda.index) {
  logits <- matrix(0, nrow(newdata), object$K)
  for (k in seq_len(object$K)) {
    logits[, k] <- as.numeric(newdata %*% object$beta[[k]][, lambda.index]) +
      object$intercept[[k]][lambda.index]
  }
  if (anyNA(logits) || any(!is.finite(logits)))
    stop("Multinomial logits must be finite.")
  colnames(logits) <- object$levels
  logits
}


.picasso_multinomial_logits <- function(object, newdata, lambda.idx) {
  lapply(lambda.idx, function(lambda.index) {
    .picasso_multinomial_logits_one(object, newdata, lambda.index)
  })
}


picasso.multinomial <- function(X,
                                Y,
                                lambda = NULL,
                                nlambda = NULL,
                                lambda.min.ratio = NULL,
                                method = "l1",
                                gamma = 3,
                                dfmax = NULL,
                                standardize = TRUE,
                                intercept = TRUE,
                                prec = 1e-7,
                                max.ite = 1e4,
                                verbose = FALSE,
                                lla.max.stages = 3L,
                                fast.mode = FALSE)
{
  prec <- .picasso_resolve_precision(prec, fast.mode, "multinomial")
  automatic.lambda.path <- is.null(lambda)
  dims <- .picasso_validate_design(X)
  n <- dims$n
  d <- dims$d

  method <- .picasso_validate_choice(
    method, c("l1", "mcp", "scad"), "method"
  )
  max.ite <- .picasso_validate_positive_integer(max.ite, "max.ite")
  nlambda <- .picasso_validate_positive_integer(
    nlambda, "nlambda", allow.null = TRUE
  )
  standardize <- .picasso_validate_flag(standardize, "standardize")
  intercept <- .picasso_validate_flag(intercept, "intercept")
  verbose <- .picasso_validate_flag(verbose, "verbose")
  dfmax <- .picasso_validate_nonnegative_integer(
    dfmax, "dfmax", allow.null = TRUE
  )

  if (!is.atomic(Y) || is.list(Y))
    stop("Y must be an atomic response vector.")
  if (length(Y) != n)
    stop(sprintf("Y must have length %d to match the rows of X.", n))
  if (anyNA(Y))
    stop("Y must not contain missing values.")
  if (is.numeric(Y) && any(!is.finite(Y)))
    stop("Numeric Y must contain only finite class values.")
  if (length(gamma) != 1L || !is.numeric(gamma) ||
      is.na(gamma) || !is.finite(gamma))
    stop("gamma must be a finite numeric scalar.")
  lla.max.stages <- .picasso_validate_lla_max_stages(lla.max.stages)

  # Encode Y as 0-indexed integers
  Y_fac <- droplevels(as.factor(Y))
  K <- nlevels(Y_fac)
  if (K < 3)
    stop(sprintf(
      "picasso.multinomial requires >= 3 classes; found %d. Use family='binomial' for 2 classes.",
      K
    ))
  Y_int <- as.integer(Y_fac) - 1L  # 0..K-1

  begt <- Sys.time()
  if (verbose) cat("Sparse multinomial regression.\n")

  # Centering changes the model space when no intercept is fitted. In that
  # case standardization scales columns about the origin, retaining nonzero
  # constant columns as genuine predictors.
  design <- .picasso_prepare_design(X, standardize, center = intercept)
  xx     <- design$xx
  xm     <- design$xm
  xinvc.vec <- design$xinvc.vec

  # At the native zero model, fitted intercepts give empirical class
  # probabilities; without intercepts every class probability is 1/K.
  p0 <- if (intercept) {
    tabulate(Y_int + 1L, nbins = K) / n
  } else {
    rep(1 / K, K)
  }
  lambda.max <- 0.0
  if (is.null(lambda)) {
    for (k in seq_len(K)) {
      resid_k <- as.numeric(Y_int == (k - 1L)) - p0[k]
      gk      <- abs(crossprod(xx, resid_k)) / n
      lambda.max <- max(lambda.max, max(gk))
    }
  }

  lambda.info <- .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda      <- lambda.info$lambda
  nlambda     <- lambda.info$nlambda
  if (!is.numeric(lambda) || !is.null(dim(lambda)) || length(lambda) == 0L ||
      anyNA(lambda) ||
      any(!is.finite(lambda)) || any(lambda < 0))
    stop("lambda must be a numeric vector of finite nonnegative values.")
  if (length(lambda) > 1L && any(diff(lambda) >= 0))
    stop("lambda must be strictly decreasing.")

  method.info <- .picasso_method_flag(method, gamma)
  method.flag <- method.info$flag
  gamma       <- method.info$gamma

  dfmax.int <- if (is.null(dfmax)) -1L else dfmax

  out <- multinomial_solver(Y_int, xx, lambda, nlambda, gamma,
                            n, d, K, max.ite,
                            prec, intercept, verbose,
                            method.flag, dfmax.int, lla.max.stages,
                            automatic.lambda.path)

  num.fit <- out$num.fit
  if (length(num.fit) != 1L || is.na(num.fit) || num.fit <= 0)
    stop("Multinomial solver did not fit any lambda values.")
  if (num.fit > nlambda)
    stop(sprintf(
      "Multinomial solver returned %d fits, but only %d were requested.",
      num.fit, nlambda
    ))
  if (num.fit < nlambda) {
    lambda  <- lambda[seq_len(num.fit)]
    nlambda <- num.fit
  }

  # Reshape output: beta is d * K * nlambda flat (for each lambda: K*d)
  # Layout: beta[lambda * K * d + class * d + feat]
  beta_array   <- array(out$beta[seq_len(d * K * nlambda)],
                        dim = c(d, K, nlambda))
  intcpt_mat   <- matrix(out$intcpt[seq_len(K * nlambda)],
                         nrow = K, ncol = nlambda)

  # Rescale each class's coefficients back to original scale
  beta_list    <- vector("list", K)
  intcpt_list  <- vector("list", K)
  for (k in seq_len(K)) {
    beta_raw_k <- beta_array[, k, , drop = FALSE]
    dim(beta_raw_k) <- c(d, nlambda)
    beta_k <- if (standardize) beta_raw_k * xinvc.vec else beta_raw_k
    intcpt_k <- if (!intercept) {
      rep(0, nlambda)
    } else if (standardize) {
      intcpt_mat[k, ] - as.numeric(xm %*% beta_k)
    } else {
      intcpt_mat[k, ]
    }
    beta_list[[k]]   <- Matrix(beta_k)
    intcpt_list[[k]] <- intcpt_k
  }

  runt <- Sys.time() - begt

  est <- list(
    beta        = beta_list,       # list of K sparse matrices (d x nlambda)
    intercept   = intcpt_list,     # list of K vectors (length nlambda)
    lambda      = lambda,
    nlambda     = nlambda,
    df          = as.integer(out$size.act),
    method      = method,
    alg         = "multinomial-proximal-newton",
    K           = K,
    levels      = levels(Y_fac),
    ite         = out$ite,
    runt        = out$runt,
    lla.max.stages = lla.max.stages,
    status      = out$status,
    status.code = out$status.code,
    failure     = out$failure,
    diagnostics = out$diagnostics,
    path.early.stopped = out$path.early.stopped,
    requested.nlambda = out$requested.nlambda,
    fast.mode   = fast.mode,
    prec        = prec,
    verbose     = verbose,
    runtime     = runt,
    family      = "multinomial"
  )
  est$nulldev <- -mean(log(p0[Y_int + 1L]))
  # The native solver already evaluates the smooth loss at every committed
  # path point. Reusing it avoids rebuilding n-by-K logits and probabilities
  # for each lambda after fitting.
  fit.deviance <- out$smooth.nll
  est$dev.ratio <- if (est$nulldev > 0) {
    pmax(0, pmin(1, 1 - fit.deviance / est$nulldev))
  } else {
    rep(0, nlambda)
  }
  class(est) <- "multinomial"
  est
}


print.multinomial <- function(x, ...) {
  cat("\n Multinomial options summary:\n")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda, digits = 3))
  cat("Method =", x$method, "\n")
  if (!is.null(x$status) && !is.null(x$status.code))
    cat("Status =", x$status, sprintf("(code %d)\n", x$status.code))
  cat("Classes:", paste(x$levels, collapse = ", "), "\n")
  cat("Degree of freedom:", min(x$df), "----->", max(x$df), "\n")
  cat("Runtime:", x$runtime, " ", as.character(units(x$runtime)), "\n")
  invisible(x)
}


plot.multinomial <- function(x, which.class = 1, ...) {
  k <- .picasso_multinomial_indices(
    which.class, x$K, "which.class", default.length = 1L
  )
  if (length(k) != 1L)
    stop("`which.class` must contain exactly one index.")
  matplot(
    x$lambda,
    t(as.matrix(x$beta[[k]])),
    type = "l",
    main = sprintf("Regularization Path (class %s)", x$levels[k]),
    xlab = "Regularization Parameter",
    ylab = "Coefficient"
  )
  invisible(NULL)
}


coef.multinomial <- function(object, lambda.idx = NULL, beta.idx = NULL, ...) {
  lambda.idx <- .picasso_multinomial_indices(
    lambda.idx, object$nlambda, "lambda.idx"
  )
  beta.idx <- .picasso_multinomial_indices(
    beta.idx, nrow(object$beta[[1L]]), "beta.idx"
  )
  lapply(seq_len(object$K), function(k) {
    beta_k <- object$beta[[k]]
    intcpt_k <- object$intercept[[k]]
    beta.block <- as.matrix(beta_k[beta.idx, lambda.idx, drop = FALSE])
    coef.mat <- rbind(
      "(Intercept)" = as.numeric(intcpt_k[lambda.idx]),
      beta.block
    )
    rownames(coef.mat)[-1] <- paste0("beta[", beta.idx, "]")
    colnames(coef.mat) <- paste0("lambda[", lambda.idx, "]")
    coef.mat
  })
}


predict.multinomial <- function(object, newdata, lambda.idx = NULL,
                                 type = "response", s = NULL, ...) {
  if (missing(newdata))
    stop("`newdata` must be provided.")
  newdata <- .picasso_multinomial_newdata(object, newdata)
  type <- match.arg(type, c("response", "link", "class", "nonzero"))
  n_new <- nrow(newdata)
  K     <- object$K
  lams  <- object$lambda
  nlam  <- object$nlambda

  # Helper: linear predictor (n_new x K) for one set of betas/intercepts.
  # beta_list: list of K numeric vectors length d
  # intcpt_vec: numeric vector length K
  .lp <- function(beta_list, intcpt_vec) {
    lp <- matrix(0, n_new, K)
    for (k in seq_len(K))
      lp[, k] <- as.numeric(newdata %*% beta_list[[k]]) + intcpt_vec[k]
    if (anyNA(lp) || any(!is.finite(lp)))
      stop("Multinomial logits must be finite.")
    colnames(lp) <- object$levels
    lp
  }

  # Helper: softmax predict from lp_mat
  .predict_one <- function(lp_mat, sv_label) {
    if (type == "link") return(lp_mat)
    if (type == "class") {
      return(factor(
        max.col(lp_mat, ties.method = "first"),
        levels = seq_len(K), labels = object$levels
      ))
    }
    prob_mat <- .picasso_multinomial_softmax(lp_mat)
    colnames(prob_mat) <- object$levels
    if (type == "response") return(prob_mat)
    prob_mat
  }

  # ----- resolve request to a list of (beta_list, intcpt_vec) items -----
  if (!is.null(s)) {
    if (!is.null(lambda.idx))
      stop("Supply only one of `s` and `lambda.idx`.")
    if (!is.numeric(s) || length(s) == 0L || anyNA(s) ||
        any(!is.finite(s)) || any(s < 0))
      stop("`s` must contain finite nonnegative lambda values.")
    # s= path: lambda values with interpolation
    items <- lapply(s, function(sv) {
      if (type == "nonzero") {
        # nonzero: nearest lambda, no interpolation
        li <- which.min(abs(lams - sv))
        return(list(kind = "idx", li = li, label = paste0("s=", sv)))
      }
      if (sv >= lams[1]) {
        return(list(kind = "idx", li = 1L, label = paste0("s=", sv)))
      } else if (sv <= lams[nlam]) {
        return(list(kind = "idx", li = nlam, label = paste0("s=", sv)))
      } else {
        i_lo <- max(which(lams >= sv))
        i_hi <- i_lo + 1L
        if (abs(lams[i_lo] - sv) < 1e-12) {
          return(list(kind = "idx", li = i_lo, label = paste0("s=", sv)))
        }
        alpha <- (lams[i_lo] - sv) / (lams[i_lo] - lams[i_hi])
        return(list(kind = "interp", i_lo = i_lo, i_hi = i_hi,
                    alpha = alpha, label = paste0("s=", sv)))
      }
    })

    interp_flags <- vapply(items, function(it) it$kind == "interp", logical(1))
    if (any(interp_flags)) {
      interp_vals <- s[interp_flags]
      message(sprintf(
        "Note: %d value(s) of s (%s) not in the lambda path; predictions obtained by linear interpolation.",
        sum(interp_flags),
        paste(signif(interp_vals, 4), collapse = ", ")
      ))
    }

    result_list <- lapply(items, function(it) {
      if (it$kind == "idx") {
        li <- it$li
        if (type == "nonzero")
          return(lapply(seq_len(K),
                        function(k) which(abs(object$beta[[k]][, li]) > 1e-8)))
        beta_list  <- lapply(seq_len(K), function(k) as.numeric(object$beta[[k]][, li]))
        intcpt_vec <- vapply(seq_len(K), function(k) object$intercept[[k]][li], numeric(1))
      } else {
        a <- it$alpha
        beta_list <- lapply(seq_len(K), function(k) {
          (1 - a) * as.numeric(object$beta[[k]][, it$i_lo]) +
               a  * as.numeric(object$beta[[k]][, it$i_hi])
        })
        intcpt_vec <- vapply(seq_len(K), function(k) {
          (1 - a) * object$intercept[[k]][it$i_lo] +
               a  * object$intercept[[k]][it$i_hi]
        }, numeric(1))
      }
      .predict_one(.lp(beta_list, intcpt_vec), it$label)
    })

    return(if (length(s) == 1) result_list[[1]] else result_list)
  }

  # ----- lambda.idx= path (original behaviour) -----
  lambda.idx <- .picasso_multinomial_indices(
    lambda.idx, object$nlambda, "lambda.idx"
  )
  result_list <- lapply(lambda.idx, function(li) {
    if (type == "nonzero")
      return(lapply(seq_len(K),
                    function(k) which(abs(object$beta[[k]][, li]) > 1e-8)))
    beta_list  <- lapply(seq_len(K), function(k) as.numeric(object$beta[[k]][, li]))
    intcpt_vec <- vapply(seq_len(K), function(k) object$intercept[[k]][li], numeric(1))
    .predict_one(.lp(beta_list, intcpt_vec), paste0("lambda[", li, "]"))
  })

  if (length(lambda.idx) == 1) result_list[[1]] else result_list
}
