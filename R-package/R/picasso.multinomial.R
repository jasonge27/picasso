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
                                prec = 1e-4,
                                max.ite = 1e4,
                                verbose = FALSE)
{
  dims <- .picasso_validate_design(X)
  n <- dims$n
  d <- dims$d

  # Encode Y as 0-indexed integers
  Y_fac <- as.factor(Y)
  K <- nlevels(Y_fac)
  if (K < 3)
    stop(sprintf(
      "picasso.multinomial requires >= 3 classes; found %d. Use family='binomial' for 2 classes.",
      K
    ))
  Y_int <- as.integer(Y_fac) - 1L  # 0..K-1

  begt <- Sys.time()
  if (verbose) cat("Sparse multinomial regression.\n")

  design <- .picasso_prepare_design(X, standardize)
  xx     <- design$xx
  xm     <- design$xm
  xinvc.vec <- design$xinvc.vec

  # lambda.max: max over all (k, j) of |X^T (1(Y==k) - p_k0)| / n
  # where p_k0 = n_k / n
  p0 <- tabulate(Y_int + 1L, nbins = K) / n  # prior class probs
  lambda.max <- 0
  for (k in seq_len(K)) {
    resid_k <- as.numeric(Y_int == (k - 1L)) - p0[k]
    gk      <- abs(crossprod(xx, resid_k)) / n
    lambda.max <- max(lambda.max, max(gk))
  }

  lambda.info <- .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda      <- lambda.info$lambda
  nlambda     <- lambda.info$nlambda

  method.info <- .picasso_method_flag(method, gamma)
  method.flag <- method.info$flag
  gamma       <- method.info$gamma

  dfmax.int <- if (is.null(dfmax)) as.integer(-1) else as.integer(dfmax)

  out <- multinomial_solver(Y_int, xx, lambda, nlambda, gamma,
                            n, d, K, max.ite,
                            prec, intercept, verbose,
                            method.flag, dfmax.int)

  num.fit <- out$num.fit
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
    if (standardize) {
      beta_k    <- beta_raw_k * xinvc.vec
      intcpt_k  <- intcpt_mat[k, ] - as.numeric(xm %*% beta_k)
    } else {
      beta_k    <- beta_raw_k
      intcpt_k  <- intcpt_mat[k, ]
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
    alg         = "multinomial-cd",
    K           = K,
    levels      = levels(Y_fac),
    ite         = out$ite,
    verbose     = verbose,
    runtime     = runt
  )
  class(est) <- "multinomial"
  est
}


print.multinomial <- function(x, ...) {
  cat("\n Multinomial options summary:\n")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda, digits = 3))
  cat("Method =", x$method, "\n")
  cat("Classes:", paste(x$levels, collapse = ", "), "\n")
  cat("Degree of freedom:", min(x$df), "----->", max(x$df), "\n")
  cat("Runtime:", x$runtime, " ", as.character(units(x$runtime)), "\n")
  invisible(x)
}


plot.multinomial <- function(x, which.class = 1, ...) {
  k <- which.class
  if (k < 1 || k > x$K)
    stop(sprintf("which.class must be between 1 and %d", x$K))
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


coef.multinomial <- function(object, lambda.idx = 1:3, beta.idx = 1:3, ...) {
  lambda.idx <- as.integer(lambda.idx)
  beta.idx   <- as.integer(beta.idx)
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


predict.multinomial <- function(object, newdata, lambda.idx = 1:3,
                                 type = "response", s = NULL, ...) {
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
    lp
  }

  # Helper: softmax predict from lp_mat
  .predict_one <- function(lp_mat, sv_label) {
    if (type == "link") return(lp_mat)
    lp_mat <- lp_mat - apply(lp_mat, 1, max)
    exp_lp <- exp(lp_mat)
    prob_mat <- exp_lp / rowSums(exp_lp)
    if (type == "response") return(prob_mat)
    if (type == "class")
      return(factor(apply(prob_mat, 1, which.max),
                    levels = seq_len(K), labels = object$levels))
    prob_mat
  }

  # ----- resolve request to a list of (beta_list, intcpt_vec) items -----
  if (!is.null(s)) {
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
  lambda.idx <- as.integer(lambda.idx)
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
