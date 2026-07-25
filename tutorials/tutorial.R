library(picasso)

set.seed(20260719)
n <- 120L
d <- 24L
X <- matrix(rnorm(n * d), nrow = n)
beta <- c(1.2, -0.8, 0.5, rep(0, d - 3L))

## Gaussian: the public default chooses a memory-bounded backend automatically.
y.gaussian <- drop(0.4 + X %*% beta + rnorm(n, sd = 0.7))
fit.gaussian <- picasso(
  X, y.gaussian, family = "gaussian", method = "l1", nlambda = 16L
)
cat(
  "Gaussian backend:", fit.gaussian$type.gaussian.requested, "->",
  fit.gaussian$type.gaussian, "\n"
)
coef(
  fit.gaussian,
  lambda.idx = c(1L, fit.gaussian$nlambda),
  beta.idx = 1:4
)
predict(
  fit.gaussian, X[1:6, , drop = FALSE],
  s = tail(fit.gaussian$lambda, 1L),
  Y.pred.idx = 1:6
)
assess.picasso(fit.gaussian, X, y.gaussian)

## Explicit Gaussian backends remain available for reproducible comparisons.
fit.gaussian.naive <- picasso(
  X, y.gaussian, nlambda = 16L, type.gaussian = "naive"
)
fit.gaussian.covariance <- picasso(
  X, y.gaussian, nlambda = 16L, type.gaussian = "covariance"
)

## Binomial: MCP/SCAD use adaptive LLA around active-set Newton subproblems.
eta.binomial <- drop(X %*% beta)
y.binomial <- factor(
  ifelse(runif(n) < plogis(eta.binomial), "case", "control")
)
fit.binomial <- picasso(
  X, y.binomial, family = "binomial", method = "mcp",
  nlambda = 16L, lla.max.stages = 3L, fast.mode = TRUE
)
fit.binomial$status
head(fit.binomial$diagnostics)
predict(
  fit.binomial, X[1:6, , drop = FALSE],
  lambda.idx = fit.binomial$nlambda, type = "response",
  p.pred.idx = 1:6
)

## Poisson offsets are added on the link scale and must be supplied again for
## new-data prediction or assessment.
exposure <- runif(n, 0.5, 2.0)
offset <- log(exposure)
y.poisson <- rpois(n, exp(offset + 0.2 + X[, 1L] - 0.5 * X[, 2L]))
fit.poisson <- picasso(
  X, y.poisson, family = "poisson", nlambda = 16L, offset = offset
)
predict(
  fit.poisson, X[1:6, , drop = FALSE],
  lambda.idx = fit.poisson$nlambda, type = "response",
  newoffset = offset[1:6], p.pred.idx = 1:6
)
assess.picasso(fit.poisson, X, y.poisson, newoffset = offset)

## Square-root lasso uses active-set quadratic-MM updates and shares the
## adaptive-LLA diagnostics interface.
fit.sqrt <- picasso(
  X, y.gaussian, family = "sqrtlasso", nlambda = 16L
)
fit.sqrt$status

## Multinomial: labels are retained and probabilities are class-coupled.
scores <- cbind(
  red = X[, 1L] + 0.3 * X[, 2L],
  green = -X[, 1L] + 0.4 * X[, 3L],
  blue = -0.5 * X[, 2L] - 0.4 * X[, 3L]
)
y.multinomial <- factor(colnames(scores)[max.col(
  scores + matrix(rnorm(length(scores), sd = 0.5), nrow = n)
)])
fit.multinomial <- picasso(
  X, y.multinomial, family = "multinomial", nlambda = 16L,
  fast.mode = TRUE
)
predict(
  fit.multinomial, X[1:6, , drop = FALSE],
  lambda.idx = fit.multinomial$nlambda, type = "response"
)
predict(
  fit.multinomial, X[1:6, , drop = FALSE],
  lambda.idx = fit.multinomial$nlambda, type = "class"
)
confusion.picasso(
  fit.multinomial, X, y.multinomial,
  lambda.idx = fit.multinomial$nlambda
)

## Cross-validation uses one common lambda path. Categorical folds are
## stratified automatically; R fold IDs are one-based when supplied manually.
cv.gaussian <- cv.picasso(
  X, y.gaussian, family = "gaussian", nlambda = 16L, nfolds = 3L
)
cv.gaussian$lambda.min
coef(cv.gaussian, s = "lambda.1se", beta.idx = 1:4)

## Coefficient-path plots are available for every family.
plot(fit.gaussian)
plot(fit.multinomial, which.class = 1L)
