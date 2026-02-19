library(picasso)

# Reproducible tutorial examples for Gaussian, binomial, and Poisson models.
set.seed(2016)

generate_design <- function(n, d, corr = 0.5) {
  scale(matrix(rnorm(n * d), n, d) + corr * rnorm(n)) / sqrt(n - 1) * sqrt(n)
}

show_path_summary <- function(fit, label, idx = 30) {
  i <- min(idx, fit$nlambda)
  cat("\n====", label, "====\n")
  cat("First 5 lambdas:\n")
  print(head(fit$lambda, 5))
  cat("First 5 df values:\n")
  print(head(fit$df, 5))
  cat("Selected lambda index:", i, "\n")
  print(fit$lambda[i])
  print(fit$beta[, i])
  print(fit$intercept[i])
}

################################################################
## Sparse linear regression
n <- 100
d <- 80
s <- 20
X <- generate_design(n, d, corr = 0.5)
beta <- c(runif(s), rep(0, d - s))
Y <- X %*% beta + rnorm(n)

fit_g_l1_naive <- picasso(X, Y, family = "gaussian", method = "l1",
                          type.gaussian = "naive", nlambda = 100)
fit_g_l1_cov <- picasso(X, Y, family = "gaussian", method = "l1",
                        type.gaussian = "covariance", nlambda = 100)
fit_g_mcp <- picasso(X, Y, family = "gaussian", method = "mcp", nlambda = 100)
fit_g_scad <- picasso(X, Y, family = "gaussian", method = "scad", nlambda = 100)

show_path_summary(fit_g_l1_naive, "Gaussian / L1 (naive)")

## Optional: visualize solution paths
plot(fit_g_l1_naive)
plot(fit_g_l1_cov)
plot(fit_g_mcp)
plot(fit_g_scad)

################################################################
## Sparse logistic regression
X <- generate_design(n, d, corr = 0.5)
beta <- c(runif(s), rep(0, d - s))
p <- 1 / (1 + exp(-X %*% beta))
Y <- rbinom(n, 1, p)

fit_b_l1 <- picasso(X, Y, family = "binomial", method = "l1", nlambda = 100)
fit_b_mcp <- picasso(X, Y, family = "binomial", method = "mcp", nlambda = 100)
fit_b_scad <- picasso(X, Y, family = "binomial", method = "scad", nlambda = 100)

show_path_summary(fit_b_l1, "Binomial / L1")

## Fitted Bernoulli probabilities on training data
prob_train <- fit_b_l1$p
print(dim(prob_train))

## Optional: visualize solution paths
plot(fit_b_l1)
plot(fit_b_mcp)
plot(fit_b_scad)

################################################################
## Sparse poisson regression
X <- generate_design(n, d, corr = 0.5)
beta <- c(runif(s), rep(0, d - s)) / sqrt(s)
eta <- X %*% beta + rnorm(n)
Y <- rpois(n, lambda = exp(eta))

fit_p_l1 <- picasso(X, Y, family = "poisson", method = "l1", nlambda = 100)
fit_p_mcp <- picasso(X, Y, family = "poisson", method = "mcp", nlambda = 100)
fit_p_scad <- picasso(X, Y, family = "poisson", method = "scad", nlambda = 100)

show_path_summary(fit_p_l1, "Poisson / L1")

## Optional: visualize solution paths
plot(fit_p_l1)
plot(fit_p_mcp)
plot(fit_p_scad)
