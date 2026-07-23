picasso <- function(X,
                    Y,
                    lambda = NULL,
                    nlambda = 100,
                    lambda.min.ratio = 0.05,
                    family = "gaussian",
                    method = "l1",
                    type.gaussian = "auto",
                    gamma = 3,
                    df = NULL,
                    dfmax = NULL,
                    standardize = TRUE,
                    intercept = TRUE,
                    prec = 1e-7,
                    max.ite = 1e3,
                    verbose = FALSE,
                    offset = NULL,
                    lla.max.stages = 3L,
                    fast.mode = FALSE)
{
  supported.family <- c(
    "gaussian", "binomial", "poisson", "sqrtlasso", "multinomial"
  )
  family <- .picasso_validate_choice(family, supported.family, "family")
  method <- .picasso_validate_choice(
    method, c("l1", "mcp", "scad"), "method"
  )
  prec <- .picasso_resolve_precision(prec, fast.mode, family)
  lla.max.stages <- .picasso_validate_lla_max_stages(lla.max.stages)

  if (!is.null(offset) && !(family %in% c("binomial", "poisson"))) {
    stop(sprintf("offset is not supported for family = \"%s\".", family))
  }

  if (family == "gaussian") {
    if (!is.matrix(Y))
      Y = as.matrix(Y)
    if (ncol(Y) != 1)
      stop("Only univariate response is supported for family = \"gaussian\" in this version.")

    out = picasso.gaussian(X = X, Y = Y, lambda = lambda, nlambda = nlambda,
                        lambda.min.ratio = lambda.min.ratio,
                        method = method, type.gaussian = type.gaussian, gamma = gamma, df = df,
                        dfmax = dfmax,
                        standardize = standardize,  intercept= intercept,
                        prec = prec,
                        max.ite = max.ite, verbose = verbose,
                        fast.mode = fast.mode)
  } else if (family == "binomial") {
    out = picasso.logit(X = X, Y = Y, lambda = lambda, nlambda = nlambda,
                        lambda.min.ratio = lambda.min.ratio,
                        method = method, gamma = gamma, dfmax = dfmax,
                        standardize = standardize, intercept=intercept,
                        prec = prec, max.ite = max.ite, verbose = verbose,
                        offset = offset,
                        lla.max.stages = lla.max.stages,
                        fast.mode = fast.mode)
  } else if (family == "sqrtlasso"){
    if(!is.matrix(Y))
      Y = as.matrix(Y)

    out = picasso.sqrtlasso(X = X, Y = Y, lambda = lambda, nlambda = nlambda,
                        lambda.min.ratio = lambda.min.ratio,
                        method = method, gamma = gamma, dfmax = dfmax,
                        standardize = standardize, intercept=intercept,
                        prec = prec, max.ite = max.ite, verbose = verbose,
                        lla.max.stages = lla.max.stages,
                        fast.mode = fast.mode)
  } else if (family=="poisson") {
    out = picasso.poisson(X = X, Y=Y, lambda = lambda, nlambda = nlambda,
                        lambda.min.ratio = lambda.min.ratio,
                       method = method, gamma = gamma, dfmax = dfmax,
                       standardize = standardize,
                       intercept = intercept,
                       prec = prec, max.ite = max.ite,
                       verbose = verbose,
                       offset = offset,
                       lla.max.stages = lla.max.stages,
                       fast.mode = fast.mode)
  } else if (family == "multinomial") {
    out = picasso.multinomial(X = X, Y = Y, lambda = lambda, nlambda = nlambda,
                              lambda.min.ratio = lambda.min.ratio,
                              method = method, gamma = gamma, dfmax = dfmax,
                              standardize = standardize, intercept = intercept,
                              prec = prec, max.ite = max.ite,
                              lla.max.stages = lla.max.stages,
                              verbose = verbose,
                              fast.mode = fast.mode)
  }
  out$family = family
  return(out)
}
