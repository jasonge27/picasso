picasso.gaussian <- function(X, 
                          Y, 
                          lambda = NULL,
                          nlambda = NULL,
                          lambda.min.ratio = NULL,
                          lambda.min = NULL,
                          method = "l1",
                          type.gaussian = NULL,
                          gamma = 3,
                          df = NULL,
                          standardize = TRUE,
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = FALSE)
{
  begt = Sys.time()
  n = nrow(X)
  d = ncol(X)
  if (verbose)
    cat("Sparse linear regression. \n")

  if (is.null(type.gaussian)) {
    if (n < 500) {
      type.gaussian = "covariance"
    } else {
      type.gaussian = "naive" 
    }
  }

  if (n == 0 || d == 0) {
    cat("No data input.\n")
    return(NULL)
  }

  if (method != "l1" && method != "mcp" && method != "scad"){
    cat(" Wrong \"method\" input. \n \"method\" 
          should be one of \"l1\", \"mcp\", \"scad\".\n", 
        method," is not supported in this version. \n")
    return(NULL)
  }
 
  if (type.gaussian!="naive" && type.gaussian!="covariance") {
    cat(" Wrong \"type.gaussian\" input. \n \"type.gaussian\" should 
          be one of \"naive\" and \"covariance\".\n", 
        type.gaussian," is not supported in this version. \n")
    return(NULL)
  }

  
  res.sd = FALSE 

  if (standardize) {
    xx = rep(0,n*d)
    xm = rep(0,d)
    xinvc.vec = rep(0,d)
    str = .C("standardize_design", as.double(X), as.double(xx), as.double(xm), 
              as.double(xinvc.vec), as.integer(n), as.integer(d), PACKAGE="picasso")
    xx = matrix(unlist(str[2]), nrow=n, ncol=d, byrow=FALSE)
    xm = matrix(unlist(str[3]), nrow=1)
    xinvc.vec = unlist(str[4])
    
    ym = mean(Y)
    y1 = Y-ym
    if (res.sd){
      sdy = sqrt(sum(y1^2)/(n-1))
      yy = y1/sdy
    } else {
      sdy = 1
      yy = y1
    }
  } else {
    xinvc.vec = rep(1,d)
    sdy = 1
    xx = X
    yy = Y
  }

  if (is.null(df)) {
    df = d
  }
  
  est = list()
  if (!is.null(lambda)) nlambda = length(lambda)

  if (is.null(lambda)){
    if (is.null(nlambda))
      nlambda = 100

    xy = crossprod(xx,yy)
    lambda.max = max(abs(xy/n))

    if (is.null(lambda.min)){
      if (is.null(lambda.min.ratio)){
        lambda.min = 0.05*lambda.max
      } else {
        lambda.min = min(lambda.min.ratio*lambda.max, lambda.max)
      }
    }
    if (lambda.min >= lambda.max) 
      cat("\"lambda.min\" is too small. \n")
    lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
  }

  if (method == "l1" || method == "mcp" || method == "scad") {
    if (method == "l1") {
      method.flag = 1
    }
    if (method == "mcp") {
      method.flag = 2
      if (gamma <= 1) {
        cat("gamma > 1 is required for MCP. Set to the default value 3. \n")
        gamma = 3
      }
    }
    if (method == "scad") {
      method.flag = 3
      if (gamma <= 2) {
        cat("gamma > 2 is required for SCAD. Set to the default value 3. \n")
        gamma = 3
      }
    }

    out = gaussian_solver(yy, xx, lambda, nlambda, gamma, n, d, df, max.ite, prec, verbose, 
                         standardize, method.flag, type.gaussian)

    if (out$err == 1)
      cat("Error! Parameters are too dense. Please choose larger \"lambda\". \n")
    if (out$err == 2)
      cat("Warning! \"df\" may be too small. You may choose larger \"df\". \n")
  }
  
  est$beta = new("dgCMatrix", Dim = as.integer(c(d,nlambda)),
            x = as.vector(out$beta[1:out$col.cnz[nlambda+1]]),
             p=as.integer(out$col.cnz),
             i = as.integer(out$beta.idx[1:out$col.cnz[nlambda+1]]))

  est$df = rep(0,nlambda)
  for (i in 1:nlambda)
    est$df[i] = out$col.cnz[i+1] - out$col.cnz[i]

  est$intercept = matrix(0, nrow=1, ncol=nlambda)

  if (standardize){
    for (k in 1:nlambda){
      est$beta[,k] = xinvc.vec * est$beta[,k]*sdy
      est$intercept[k] = ym - as.numeric(xm%*%est$beta[,k]) + out$intcpt[k]*sdy
    }
    est$lambda = lambda * sdy
  } else {
    for (k in 1:nlambda){
      est$intercept[k] = out$intcpt[k]
    }
    est$lambda = lambda
  }

  est$ite = out$ite
  est$runt = out$runt
  
  runt = Sys.time()-begt
  
  est$nlambda = nlambda
  est$gamma = gamma
  est$method = method
  est$verbose = verbose
  est$runtime = runt
  class(est) = "gaussian"
  return(est)
}

print.gaussian <- function(x, ...)
{  
  cat("\n Lasso options summary: \n")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda,digits=3))
  cat("Method =", x$method, "\n")
  cat("Alg =", x$alg, "\n")
  cat("Degree of freedom:",min(x$df),"----->",max(x$df),"\n")
  if (units.difftime(x$runtime)=="secs") unit="secs"
  if (units.difftime(x$runtime)=="mins") unit="mins"
  if (units.difftime(x$runtime)=="hours") unit="hours"
  cat("Runtime:",x$runtime," ",unit,"\n")
}

plot.gaussian <- function(x, ...)
{
  matplot(x$lambda, t(x$beta), type="l", main="Regularization Path",
          xlab="Regularization Parameter", ylab="Coefficient")
}

coef.gaussian <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
{
  lambda.n = length(lambda.idx)
  beta.n = length(beta.idx)
  cat("\n Values of estimated coefficients: \n")
  cat(" index     ")
  for (i in 1:lambda.n){
    cat("",formatC(lambda.idx[i], digits=5, width=10),"")
  }
  cat("\n")
  cat(" lambda    ")
  for (i in 1:lambda.n){
    cat("",formatC(object$lambda[lambda.idx[i]], digits=4, width=10),"")
  }
  cat("\n")
  cat(" intercept ")
  for (i in 1:lambda.n){
    cat("",formatC(object$intercept[i], digits=4, width=10),"")
  }
  cat("\n")
  for (i in 1:beta.n){
    cat(" beta",formatC(beta.idx[i], digits=5, width=-5))
    for (j in 1:lambda.n){
      cat("",formatC(object$beta[beta.idx[i],lambda.idx[j]], digits=4, width=10),"")
    }
    cat("\n")
  }
}

predict.gaussian <- function(object, newdata, lambda.idx = c(1:3), Y.pred.idx = c(1:5), ...)
{
  pred.n = nrow(newdata)
  lambda.n = length(lambda.idx)
  Y.pred.n = length(Y.pred.idx)
  intcpt = matrix(rep(object$intercept[,lambda.idx],pred.n),nrow=pred.n,
                  ncol=lambda.n,byrow=T)
  Y.pred = newdata%*%object$beta[,lambda.idx] + intcpt
  cat("\n Values of predicted responses: \n")
  cat("   index   ")
  for (i in 1:lambda.n){
    cat("",formatC(lambda.idx[i], digits=5, width=10),"")
  }
  cat("\n")
  cat("   lambda  ")
  for (i in 1:lambda.n){
    cat("",formatC(object$lambda[lambda.idx[i]], digits=4, width=10),"")
  }
  cat("\n")
  for (i in 1:Y.pred.n){
    cat("    Y",formatC(Y.pred.idx[i], digits=5, width=-5))
    for (j in 1:lambda.n){
      cat("",formatC(Y.pred[Y.pred.idx[i],j], digits=4, width=10),"")
    }
    cat("\n")
  }
  return(Y.pred)
}