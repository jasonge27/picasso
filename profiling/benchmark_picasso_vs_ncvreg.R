#!/usr/bin/env Rscript
# Comprehensive benchmark: picasso vs ncvreg
# Compare computation speed and objective value gap

library(picasso)
library(ncvreg)
library(Matrix)

cat("=== picasso vs ncvreg Benchmark ===\n")
cat("picasso version:", as.character(packageVersion("picasso")), "\n")
cat("ncvreg  version:", as.character(packageVersion("ncvreg")), "\n\n")

# ============================================================
# Helper: generate design matrix with correlation structure
# ============================================================
gen_data <- function(n, d, s, rho, family, seed = 42) {
  set.seed(seed)
  # Toeplitz correlation: Sigma_{ij} = rho^|i-j|
  if (rho == 0) {
    X <- matrix(rnorm(n * d), n, d)
  } else {
    # Use Cholesky of Toeplitz matrix
    Sigma_half <- chol(rho^abs(outer(1:d, 1:d, "-")))
    X <- matrix(rnorm(n * d), n, d) %*% Sigma_half
  }
  # Standardize columns
  X <- scale(X)

  beta_true <- rep(0, d)
  support <- sample(1:d, s)
  beta_true[support] <- runif(s, 0.5, 2) * sample(c(-1, 1), s, replace = TRUE)

  if (family == "gaussian") {
    y <- as.numeric(X %*% beta_true + rnorm(n) * 0.5)
  } else if (family == "binomial") {
    eta <- as.numeric(X %*% beta_true)
    prob <- 1 / (1 + exp(-eta))
    y <- rbinom(n, 1, prob)
  } else if (family == "poisson") {
    # Use smaller coefficients to avoid overflow
    beta_true[support] <- runif(s, 0.1, 0.5) * sample(c(-1, 1), s, replace = TRUE)
    eta <- as.numeric(X %*% beta_true)
    y <- rpois(n, exp(eta))
  }
  list(X = X, y = y, beta_true = beta_true)
}

# ============================================================
# Helper: compute penalized objective value
# ============================================================
compute_penalty <- function(beta, lambda, gamma, method) {
  abeta <- abs(beta)
  if (method == "scad") {
    pen <- ifelse(abeta <= lambda,
                  lambda * abeta,
                  ifelse(abeta <= gamma * lambda,
                         -(abeta^2 - 2 * gamma * lambda * abeta + lambda^2) / (2 * (gamma - 1)),
                         (gamma + 1) * lambda^2 / 2))
  } else if (method == "mcp") {
    pen <- ifelse(abeta <= gamma * lambda,
                  lambda * abeta - abeta^2 / (2 * gamma),
                  gamma * lambda^2 / 2)
  } else {
    pen <- lambda * abeta
  }
  sum(pen)
}

compute_obj <- function(X, y, beta, intercept, lambda, gamma, method, family) {
  n <- nrow(X)
  eta <- as.numeric(X %*% beta) + intercept

  if (family == "gaussian") {
    loss <- sum((y - eta)^2) / (2 * n)
  } else if (family == "binomial") {
    loss <- -mean(y * eta - log(1 + exp(eta)))
  } else if (family == "poisson") {
    loss <- -mean(y * eta - exp(eta))
  }

  pen <- compute_penalty(beta, lambda, gamma, method)
  loss + pen
}

# ============================================================
# Main benchmark function
# ============================================================
run_benchmark <- function(n, d, s, rho, family, method, gamma = 3.7, nlambda = 100) {

  dat <- gen_data(n, d, s, rho, family)
  X <- dat$X
  y <- dat$y

  # --- picasso ---
  t_picasso <- system.time({
    fit_p <- tryCatch(
      picasso(X, y, family = family, method = method, gamma = gamma,
              nlambda = nlambda, prec = 1e-7),
      error = function(e) NULL
    )
  })["elapsed"]

  # --- ncvreg ---
  ncvreg_penalty <- toupper(method)  # "SCAD" or "MCP"
  ncvreg_family <- family
  t_ncvreg <- system.time({
    fit_n <- tryCatch({
      if (family == "gaussian") {
        ncvreg(X, y, family = "gaussian", penalty = ncvreg_penalty,
               gamma = gamma, nlambda = nlambda, eps = 1e-7)
      } else if (family == "binomial") {
        ncvreg(X, y, family = "binomial", penalty = ncvreg_penalty,
               gamma = gamma, nlambda = nlambda, eps = 1e-7)
      } else if (family == "poisson") {
        ncvreg(X, y, family = "poisson", penalty = ncvreg_penalty,
               gamma = gamma, nlambda = nlambda, eps = 1e-7)
      }
    }, error = function(e) NULL)
  })["elapsed"]

  if (is.null(fit_p) || is.null(fit_n)) {
    return(data.frame(
      n = n, d = d, s = s, rho = rho, family = family, method = method,
      time_picasso = ifelse(is.null(fit_p), NA, t_picasso),
      time_ncvreg = ifelse(is.null(fit_n), NA, t_ncvreg),
      speedup = NA,
      nlam_compare = NA,
      mean_obj_gap = NA,
      max_obj_gap = NA,
      picasso_better_pct = NA,
      stringsAsFactors = FALSE
    ))
  }

  # --- Compare objective values at matched lambdas ---
  # Use intersection of lambda ranges
  lam_p <- fit_p$lambda
  lam_n <- fit_n$lambda

  # Get picasso beta/intercept
  beta_p <- as.matrix(fit_p$beta)  # d x nlambda
  if (family == "gaussian") {
    intcpt_p <- fit_p$intercept
  } else {
    intcpt_p <- fit_p$intercept
  }

  # Get ncvreg beta/intercept
  coef_n <- as.matrix(coef(fit_n))  # (d+1) x nlambda, first row is intercept
  intcpt_n <- coef_n[1, ]
  beta_n <- coef_n[-1, , drop = FALSE]

  # Match lambdas: for each picasso lambda, find closest ncvreg lambda
  obj_gap <- c()
  picasso_wins <- 0
  total_compare <- 0

  for (i in seq_along(lam_p)) {
    j <- which.min(abs(lam_n - lam_p[i]))
    if (abs(lam_n[j] - lam_p[i]) / max(lam_p[i], 1e-10) > 0.05) next  # skip if >5% mismatch

    lam <- (lam_p[i] + lam_n[j]) / 2

    obj_picasso <- compute_obj(X, y, beta_p[, i], intcpt_p[i], lam, gamma, method, family)
    obj_ncvreg  <- compute_obj(X, y, beta_n[, j], intcpt_n[j], lam, gamma, method, family)

    gap <- obj_picasso - obj_ncvreg
    obj_gap <- c(obj_gap, gap)
    total_compare <- total_compare + 1
    if (gap < -1e-10) picasso_wins <- picasso_wins + 1
  }

  data.frame(
    n = n, d = d, s = s, rho = rho, family = family, method = method,
    time_picasso = round(t_picasso, 4),
    time_ncvreg = round(t_ncvreg, 4),
    speedup = round(t_ncvreg / t_picasso, 2),
    nlam_compare = total_compare,
    mean_obj_gap = round(mean(obj_gap), 6),
    max_obj_gap = round(max(abs(obj_gap)), 6),
    picasso_better_pct = round(picasso_wins / max(total_compare, 1) * 100, 1),
    stringsAsFactors = FALSE
  )
}

# ============================================================
# Experimental settings
# ============================================================
settings <- expand.grid(
  n       = c(200, 500, 1000),
  d       = c(500, 1000, 2000),
  s       = c(10, 20),
  rho     = c(0, 0.5, 0.8),
  family  = c("gaussian", "binomial", "poisson"),
  method  = c("scad", "mcp"),
  stringsAsFactors = FALSE
)

# Filter out impractical combos: s should be < d/10 roughly, and n > s
settings <- settings[settings$s < settings$d & settings$s < settings$n, ]

cat(sprintf("Total experiments: %d\n\n", nrow(settings)))

# ============================================================
# Run all benchmarks
# ============================================================
results <- data.frame()
for (i in seq_len(nrow(settings))) {
  cfg <- settings[i, ]
  cat(sprintf("[%3d/%d] n=%4d d=%4d s=%2d rho=%.1f %-9s %-4s ... ",
              i, nrow(settings), cfg$n, cfg$d, cfg$s, cfg$rho, cfg$family, cfg$method))
  flush.console()

  res <- tryCatch(
    run_benchmark(cfg$n, cfg$d, cfg$s, cfg$rho, cfg$family, cfg$method),
    error = function(e) {
      cat("ERROR:", conditionMessage(e), "\n")
      data.frame(
        n = cfg$n, d = cfg$d, s = cfg$s, rho = cfg$rho,
        family = cfg$family, method = cfg$method,
        time_picasso = NA, time_ncvreg = NA, speedup = NA,
        nlam_compare = NA, mean_obj_gap = NA, max_obj_gap = NA,
        picasso_better_pct = NA, stringsAsFactors = FALSE
      )
    }
  )

  cat(sprintf("picasso=%.3fs ncvreg=%.3fs speedup=%.1fx obj_gap=%.2e\n",
              res$time_picasso, res$time_ncvreg, res$speedup, res$mean_obj_gap))

  results <- rbind(results, res)
}

# ============================================================
# Summary tables
# ============================================================
cat("\n\n========================================\n")
cat("        FULL RESULTS TABLE\n")
cat("========================================\n")
print(results, row.names = FALSE, right = FALSE)

# Summary by family and method
cat("\n\n========================================\n")
cat("  SUMMARY: Average by Family x Method\n")
cat("========================================\n")
agg <- aggregate(
  cbind(time_picasso, time_ncvreg, speedup, mean_obj_gap, max_obj_gap, picasso_better_pct)
  ~ family + method, data = results, FUN = function(x) mean(x, na.rm = TRUE)
)
agg$speedup <- round(agg$speedup, 2)
agg$mean_obj_gap <- formatC(agg$mean_obj_gap, format = "e", digits = 2)
agg$max_obj_gap <- formatC(agg$max_obj_gap, format = "e", digits = 2)
agg$picasso_better_pct <- round(agg$picasso_better_pct, 1)
print(agg, row.names = FALSE, right = FALSE)

# Summary by correlation
cat("\n\n========================================\n")
cat("  SUMMARY: Average by Correlation (rho)\n")
cat("========================================\n")
agg2 <- aggregate(
  cbind(time_picasso, time_ncvreg, speedup, mean_obj_gap, picasso_better_pct)
  ~ rho, data = results, FUN = function(x) mean(x, na.rm = TRUE)
)
agg2$speedup <- round(agg2$speedup, 2)
agg2$mean_obj_gap <- formatC(agg2$mean_obj_gap, format = "e", digits = 2)
agg2$picasso_better_pct <- round(agg2$picasso_better_pct, 1)
print(agg2, row.names = FALSE, right = FALSE)

# Summary by dimension
cat("\n\n========================================\n")
cat("  SUMMARY: Average by (n, d)\n")
cat("========================================\n")
agg3 <- aggregate(
  cbind(time_picasso, time_ncvreg, speedup, mean_obj_gap)
  ~ n + d, data = results, FUN = function(x) mean(x, na.rm = TRUE)
)
agg3$speedup <- round(agg3$speedup, 2)
agg3$mean_obj_gap <- formatC(agg3$mean_obj_gap, format = "e", digits = 2)
print(agg3, row.names = FALSE, right = FALSE)

# Save results
write.csv(results, "benchmark_results.csv", row.names = FALSE)
cat("\nResults saved to benchmark_results.csv\n")
