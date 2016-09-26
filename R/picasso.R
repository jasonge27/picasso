picasso <- function(X, 
                    Y, 
                    lambda = NULL,
                    nlambda = NULL,
                    lambda.min.ratio = NULL,
                    lambda.min = NULL,
                    family = "gaussian",
                    method = "l1",
                    alg = "greedy",
                    opt = NULL,
                    gamma = 3,
                    df = NULL,
                    sym = "or",
                    standardize = TRUE,
                    perturb = TRUE,
                    max.act.in = 3,
                    truncation = 1e-2, 
                    prec = 1e-4,
                    max.ite = 1e3,
                    verbose = TRUE)
{
  if(family!="gaussian" && family!="binomial" &&  family != "poisson"){
    cat(" Wrong \"family\" input. \n \"family\" should be one of \"gaussian\", \"binomial\" and \"poisson\".\n", 
        family,"does not exist. \n")
    return(NULL)
  }
  if(family=="gaussian"){
    if(is.matrix(Y)==FALSE) {
      Y = as.matrix(Y)
    }
    p = ncol(Y)
    if(p==1){
      out = picasso.gaussian(X = X, Y = Y, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
                          lambda.min = lambda.min, method = method, opt = opt, gamma = gamma, df = df, 
                          standardize = standardize, max.act.in = max.act.in,  prec = prec, 
                          max.ite = max.ite, verbose = verbose)
    }
  }
  
  if(family=="binomial"){
    if(is.matrix(Y)==FALSE) {
      Y = as.matrix(Y)
    }
    out = picasso.logit(X = X, Y = Y, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
                        lambda.min = lambda.min, method = method,gamma = gamma, standardize = standardize, 
                        max.act.in = max.act.in, truncation = truncation, prec = prec, max.ite = max.ite, 
                        verbose = verbose)
  }

  if(family=="poisson"){
    out = picasso.poisson(X = X, Y=Y, lambda = lambda, nlambda = nlambda, 
                        lambda.min.ratio = lambda.min.ratio,
                       lambda.min = lambda.min, method = method, gamma = gamma, 
                       max.act.in = max.act.in, prec = prec, max.ite = max.ite, 
                       standardize = standardize, verbose = verbose)
  }
  out$family = family
  return(out)
}
