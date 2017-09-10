picasso <- function(X, 
                    Y, 
                    lambda = NULL,
                    nlambda = 100,
                    lambda.min.ratio = 0.05,
                    lambda.min = NULL,
                    family = "gaussian",
                    method = "l1",
                    type.gaussian = "naive",
                    gamma = 3,
                    df = NULL,
                    standardize = TRUE,
                    prec = 1e-7,
                    max.ite = 1e3,
                    verbose = FALSE)
{
  if (family != "gaussian" && family != "binomial" &&  family != "poisson" && family != "sqrtlasso"){
    cat(" Wrong \"family\" input. \n \"family\" should be 
           one of \"gaussian\", \"binomial\" and \"poisson\".\n", 
        family," is not supported in this version. \n")
    return(NULL)
  }
  if (family == "gaussian") {
    if (!is.matrix(Y)) 
      Y = as.matrix(Y)
    p = ncol(Y)
    if (p == 1)
      out = picasso.gaussian(X = X, Y = Y, lambda = lambda, nlambda = nlambda, 
                          lambda.min.ratio = lambda.min.ratio, lambda.min = lambda.min, 
                          method = method, type.gaussian = type.gaussian, gamma = gamma, df = df, 
                          standardize = standardize,  prec = prec, 
                          max.ite = max.ite, verbose = verbose)
  }
  
  if (family == "binomial") {
    if(!is.matrix(Y))
      Y = as.matrix(Y)
    
    out = picasso.logit(X = X, Y = Y, lambda = lambda, nlambda = nlambda, 
                        lambda.min.ratio = lambda.min.ratio, lambda.min = lambda.min, 
                        method = method, gamma = gamma, standardize = standardize, 
                        prec = prec, max.ite = max.ite, verbose = verbose)
  }

  if (family == "sqrtlasso"){
    if(!is.matrix(Y))
      Y = as.matrix(Y)
    
    out = picasso.sqrtlasso(X = X, Y = Y, lambda = lambda, nlambda = nlambda, 
                        lambda.min.ratio = lambda.min.ratio, lambda.min = lambda.min, 
                        method = method, gamma = gamma, standardize = standardize, 
                        prec = prec, max.ite = max.ite, verbose = verbose)
  }

  if(family=="poisson"){
    out = picasso.poisson(X = X, Y=Y, lambda = lambda, nlambda = nlambda, 
                        lambda.min.ratio = lambda.min.ratio,
                       lambda.min = lambda.min, method = method, gamma = gamma, 
                       prec = prec, max.ite = max.ite, 
                       standardize = standardize, verbose = verbose)
  }
  out$family = family
  return(out)
}
