test_fishnet <- function(n = 10000, p = 5000, c = 0.5, nlambda = 100){
  library(picasso)
  library(glmnet)
  set.seed(111)
  
  s = 20
  X = scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(s)
  true_beta = runif(s)
  
  param = X[,1:s]%*%true_beta+rnorm(n)
  Y = rpois(n, exp(param))
  
  cat("glmnet timing:\n")
  print(system.time(fitg<-glmnet(X, Y, family="poisson", 
                                  lambda.min.ratio=0.01, standardize=TRUE,
                                  thresh =1e-7, nlambda=nlambda)))
  
  cat("picasso timing:\n")
  print(system.time(fitp<-picasso(X, Y, family="poisson", 
                                  lambda.min.ratio=0.01,nlambda=nlambda,standardize=TRUE,
                                  verbose=FALSE, prec=1e-7)))
  
  cat("---------------------------------------\n")
  cat("comparisons of objective fuction values\n")
  objg = rep(0,nlambda)
  objp = rep(0,nlambda)
  for(i in 1:nlambda){
    rp = X%*%fitp$beta[,i]+fitp$intercept[i]
    objp[i] = sum(exp(rp)-Y*rp)/n + fitp$lambda[i]*sum(abs(fitp$beta[,i]))
    rg = X%*%fitg$beta[,i]+fitg$a0[i]
    objg[i] = sum(exp(rg)-Y*rg)/n+fitg$lambda[i]*sum(abs(fitg$beta[,i]))
  }
  
  print(mean(abs(objp-objg)/abs(objg)))
  
}

esterror <- function(b0, beta){
   nlambda = dim(beta)[2]
   err = 1e20
   for (i in 1:nlambda){
     err = min(err, sum((beta[,i]-b0)^2)/sum(b0^2))
   }
   return(err)
}

test_elnet_naive <- function(n = 10000, p = 5000, c = 0.5, nlambda = 200){
  library(picasso)
  library(glmnet)
  library(ncvreg)
  set.seed(111)
  
  X=scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(n-1)*sqrt(n)
  s = 20
  true_beta = c(runif(s), rep(0, p-s))
  Y = X%*%true_beta + rnorm(n)
  
  cat("picasso timing:\n")
  print(system.time(fitp<-picasso(X,Y,family="gaussian", type.gaussian = 'naive',
                                  lambda.min.ratio=0.05, standardize = FALSE,
                                  prec=1e-7,nlambda=nlambda)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp$beta))
  
  cat("glmnet timing:\n")
  print(system.time(fitg<-glmnet(X,Y,family="gaussian", type.gaussian = 'naive',
                                 lambda = fitp$lambda,
                                 standardize=FALSE, thresh=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitg$beta))
  
  cat("ncvreg timing:\n")
  print(system.time(fitncv<-ncvreg(X, Y, family="gaussian", penalty="lasso",
                                 lambda = fitp$lambda,
                                 standardize=FALSE, eps=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitncv$beta[2:(p+1),]))
  
  
  cat("compare obj function values:\n")
  objg = rep(0,nlambda)
  objp = rep(0,nlambda)
  for(i in 1:nlambda){
    rp = X%*%fitp$beta[,i]+fitp$intercept[i]
    objp[i] = sum((Y-rp)^2)/(2*n)+fitp$lambda[i]*sum(abs(fitp$beta[,i]))
    rg = X%*%fitg$beta[,i]+fitg$a0[i]
    objg[i] =  sum((Y-rg)^2)/(2*n)+fitg$lambda[i]*sum(abs(fitg$beta[,i]))
  }

  print(mean((objp-objg)/abs(objg)))
}

test_elnet_cov <- function(n = 300, p = 100, c = 0.5, nlambda = 100, ratio = 0.01){
  library(picasso)
  library(glmnet)
  set.seed(111)
  
  X = scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(n-1)*sqrt(n)
  cor(X[,1],X[,2])
  s = 20
  true_beta = c(runif(s), rep(0, p-s)) 
  Y = X%*%true_beta + rnorm(n)
  
  
  cat("picasso timing:\n")
  print(system.time(fitp<-picasso(X, Y, family="gaussian", type.gaussian = 'covariance',
                                  lambda.min.ratio=ratio, standardize=FALSE,
                                  verbose=FALSE, prec=1e-7, nlambda=nlambda)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp$beta))
  
  cat("glmnet timing:\n")
  print(system.time(fitg<-glmnet(X,Y,family="gaussian", type.gaussian = "covariance",
                                   lambda = fitp$lambda,
                                   standardize = FALSE, thresh = 1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitg$beta))
    
  cat("compare obj function values:\n")
  objg = rep(0,nlambda)
  objp = rep(0,nlambda)
  for(i in 1:nlambda){
    rp = X%*%fitp$beta[,i]+fitp$intercept[i]
    objp[i] = sum((Y-rp)^2)/(2*n)+fitp$lambda[i]*sum(abs(fitp$beta[,i]))
    rg = X%*%fitg$beta[,i]+fitg$a0[i]
    objg[i] =  sum((Y-rg)^2)/(2*n)+fitg$lambda[i]*sum(abs(fitg$beta[,i]))
  }
  
  print(mean((objp-objg)/abs(objg)))
}

test_lognet <- function(n = 10000, p = 5000, c = 1.0, nlambda = 100, ratio=0.01){
  library(glmnet)
  library(picasso)
  library(ncvreg)
  set.seed(111)
  
  X=scale(matrix(rnorm(n*p),n,p)+ c*rnorm(n))/sqrt(n-1)*sqrt(n)
  s = 20
  true_beta = c(runif(s), rep(0, p-s)) 
  Y=X%*%true_beta+rnorm(n)>.5
  
  cat("picasso timing:\n")
  print(system.time(fitp<-picasso(X,Y,family="binomial",
                                  lambda.min.ratio=ratio,standardize=FALSE,verbose=FALSE,prec=1e-7,nlambda=nlambda)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp$beta))
  
  cat("glmnet timing:\n")
  print(system.time(fitg<-glmnet(X,Y,family="binomial",
                                 lambda=fitp$lambda, standardize=FALSE,thresh=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitg$beta))
  
  cat('ncvreg timing:\n')
  print(system.time(fitncv<-ncvreg(X,Y,family="binomial",
                                 lambda=fitp$lambda, standardize=FALSE,thresh=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp$beta[2:(p+1),])) 
  
  cat("---------------------------------------\n")
  cat("comparisons of objective fuction values\n")
  objg = rep(0,nlambda)
  objp = rep(0,nlambda)
  for(i in 1:nlambda){
    rp = X%*%fitp$beta[,i]+fitp$intercept[i]
    objp[i] = sum(log(1+exp(rp))-Y*rp)/n+fitp$lambda[i]*sum(abs(fitp$beta[,i]))
    rg = X%*%fitg$beta[,i]+fitg$a0[i]
    objg[i] = sum(log(1+exp(rg))-Y*rg)/n+fitg$lambda[i]*sum(abs(fitg$beta[,i]))
  }
  
  print(mean(abs(objp-objg)/abs(objg)))
  #cat("---------------------------------------\n")
  #print(objp)
  #cat("---------------------------------------\n")
  #print(objg)
}

test_elnet_naive_nonlinear <- function(n = 500, p = 5000, c = 0.1, nlambda = 100, verb=FALSE){
  library(picasso)
  library(ncvreg)
  set.seed(111)
  
  X=scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(n-1)*sqrt(n)
  s = 20
  true_beta = c(runif(s), rep(0, p-s))
  Y = X%*%true_beta + rnorm(n)
  
  cat("picasso timing for mcp penalty:\n")
  print(system.time(fitp.mcp<-picasso(X,Y,family="gaussian", type.gaussian='naive', method="mcp",
                                  lambda.min.ratio=0.05, gamma =3 ,
                                  prec=1e-7,nlambda=nlambda)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp.mcp$beta))
  
  cat("ncvreg timing for mcp penalty:\n")
  print(system.time(fitncv.mcp<-ncvreg(X,Y, family="gaussian", penalty="MCP",
                                      lambda = fitp.mcp$lambda, gamma = 3,
                                      eps=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitncv.mcp$beta[2:(p+1),]))
  
  cat("picasso timing for scad penalty:\n")
  print(system.time(fitp.scad<-picasso(X,Y,family="gaussian", type.gaussian='naive', method="scad",
                                  lambda.min.ratio=0.05, gamma = 3,
                                  prec=1e-7,nlambda=nlambda)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp.scad$beta))
  
  cat("ncvreg timing for scad penalty:\n")
  print(system.time(fitncv.scad<-ncvreg(X,Y, family="gaussian", penalty="SCAD",
                                        lambda = fitp.scad$lambda,  gamma =3,
                                        eps=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitncv.scad$beta[2:(p+1),]))
}

test_elnet_cov_nonlinear <- function(n = 400, p = 1000, c = 1.0, nlambda = 100, verb=FALSE){
  library(picasso)
  set.seed(111)
  
  X = scale(matrix(rnorm(n*p),n,p) + c*rnorm(n))/sqrt(n-1)*sqrt(n)
  true_beta = runif(20)
  Y = X[,1:20] %*% true_beta + rnorm(n)
  
  cat("picasso timing for mcp penalty:\n")
  print(system.time(fitp.mcp<-picasso(X,Y,family="gaussian", type.gaussian='cov', method="mcp",
                                      lambda.min.ratio=0.001, standardize=FALSE,
                                      verbose=verb,prec=1e-7,nlambda=nlambda)))
  
  cat("picasso timing for scad penalty:\n")
  print(system.time(fitp.scad<-picasso(X,Y,family="gaussian", type.gaussian='cov', method="scad",
                                       lambda.min.ratio=0.001, standardize=FALSE,verbose=verb,
                                       prec=1e-7,nlambda=nlambda)))
}

test_lognet_nonlinear <- function(n = 10000, p = 5000, c = 1.0, nlambda = 100, ratio = 0.05, verb=FALSE){
  library(picasso)
  set.seed(111)
  
  X = scale(matrix(rnorm(n*p),n,p) + c*rnorm(n))
  s = 20
  true_beta = c(runif(s), rep(0, p-s))
  Y = X %*% true_beta + rnorm(n) > .5
  
  cat("picasso timing for mcp penalty:\n")
  print(system.time(fitp.mcp<-picasso(X,Y,family="binomial", method="mcp",
                                  lambda.min.ratio=ratio, verbose=FALSE, prec=1e-7, gamma =3,
                                  nlambda=nlambda)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp.mcp$beta))

  cat("ncvreg timing for mcp penalty:\n")
  print(system.time(fitncv.mcp<-ncvreg(X,Y, family="binomial", penalty="MCP",
                                       lambda = fitp.mcp$lambda, gamma = 3,
                                       eps=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitncv.mcp$beta[2:(p+1),]))
  
  cat("picasso timing for scad penalty:\n")
  print(system.time(fitp.scad<-picasso(X,Y,family="binomial", method="scad",
                                  lambda.min.ratio=ratio, gamma = 3,
                                  prec=1e-7,nlambda=nlambda)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitp.scad$beta))
  
  cat("ncvreg timing for scad penalty:\n")
  print(system.time(fitncv.scad<-ncvreg(X,Y, family="binomial", penalty="SCAD",
                                        lambda = fitp.scad$lambda, gamma = 3,
                                        eps=1e-7)))
  cat("best estimation error along the path:\n")
  print(esterror(true_beta, fitncv.scad$beta[2:(p+1),]))
  
  cat("glmnet L1 timing:\n")
  print(system.time(fitg<-glmnet(X,Y,family="binomial",
                                 lambda=fitp$lambda, standardize=FALSE,thresh=1e-7)))
  cat("best estimation for LASSO:\n")
  print(esterror(true_beta, fitg$beta))
  
  
}

test_fishnet_nonlinear <- function(n = 10000, p = 5000, c= 1.0, nlambda = 100, verb=FALSE){
  library(picasso)
  library(glmnet)
  set.seed(111)
  
  s = 20
  X = scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(s)
  true_beta = runif(s)
  
  param = X[,1:s]%*%true_beta+rnorm(n)
  Y = rpois(n, exp(param))
  
  cat("poisson regression with mcp timing:\n")
  print(system.time(fitp.mcp<-picasso(X, Y, family="poisson", method = "mcp",
                                  lambda.min.ratio=0.01,nlambda=nlambda,standardize=TRUE,
                                  verbose=FALSE, prec=1e-7)))
  
  cat("poisson regression with scad timing:\n")
  print(system.time(fitp.scad<-picasso(X, Y, family="poisson", method = "scad",
                                  lambda.min.ratio=0.01,nlambda=nlambda,standardize=TRUE,
                                  verbose=FALSE, prec=1e-7)))

}