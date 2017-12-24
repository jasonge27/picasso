generate_sim_lognet <- function(n, d, c, seed=1024) {
  set.seed(seed)
  cor.X <- c 
  S <- matrix(cor.X,d,d) + (1-cor.X)*diag(d)
  R <- chol(S)

  X <- scale(matrix(rnorm(n*d),n,d)%*%R)*sqrt(n-1)/sqrt(n)
  attributes(X) <- NULL
  X <- matrix(X, n,d)

  s <- 20
  true_beta <- c(runif(s), rep(0, d-s)) 

  # strictly seperable
  Y <- rbinom(n=n, size=1, p = 1/(1+exp(-X%*%true_beta)))

  return(list(X=X, Y=c(Y), true_beta=true_beta))
}

generate_sim <- function(n, d, c, seed=1024) {
  set.seed(seed)
  cor.X <- c 

  S <- matrix(cor.X,d,d) + (1-cor.X)*diag(d)
  R <- chol(S)

  X <- scale(matrix(rnorm(n*d),n,d)%*%R)*sqrt(n-1)/sqrt(n)
  attributes(X) <- NULL
  X <- matrix(X, n,d)

  s <- 20
  true_beta <- c(runif(s), rep(0, d-s)) 
  Y <- X%*%true_beta+rnorm(n)*5
  return(list(X=X, Y=c(Y), true_beta = true_beta))
}

esterror <- function(b0, beta){
   nlambda = dim(beta)[2]
   err = 1e20
   for (i in 1:nlambda){
     err = min(err, sum((beta[,i]-b0)^2)/sum(b0^2))
   }
   return(err)
}


test_sqrt_mse <- function(n = 500, p = 800, c = 0.5, nlambda = 20){
  library(flare)
  library(scalreg)
  set.seed(1024)
  df <- generate_sim(n, p, c)

  ratio = 0.01
  trialN = 2 

  obj.picasso <- rep(0, trialN)
  time.picasso <- rep(0, trialN)

  obj.flare <- rep(0, trialN)
  time.flare <- rep(0, trialN)

  obj.scalreg <- rep(0, trialN)
  time.scalreg <- rep(0, trialN)

  for (i in 1:trialN){
    t <- system.time(
            fitp<- picasso(df$X, df$Y, family="sqrtlasso", standardize=FALSE, nlambda=nlambda, 
                      prec=1e-7,
                lambda.min.ratio = 0.1))

    time.picasso[i] = t[1] 

    obj.picasso[i] <- sqrt(sum((df$Y - df$X %*% fitp$beta[, nlambda] - fitp$intercept[nlambda])^2)/n) + 
                                  fitp$lambda[nlambda]*sum(abs(fitp$beta[, nlambda]))

  

    t <- system.time(fitflare <- slim(df$X, df$Y, lambda = fitp$lambda, 
                        method='lq', q=2, prec=1e-1, verbose=FALSE))
    time.flare[i] = t[1] 

    obj.flare[i] <- sqrt(sum((df$Y - df$X %*% fitflare$beta[, nlambda] - fitflare$intercept[nlambda])^2)/n ) + 
                                        fitp$lambda[nlambda]*sum(abs(fitflare$beta[, nlambda]))


    t <- system.time(fits <- scalreg(df$X, df$Y, 
              lam0 = fitp$lambda[nlambda], LSE = FALSE))
    time.scalreg[i] = t[1]
                                       
    obj.scalreg[i] <- sqrt(sum((df$Y - df$X %*% fits$coefficients)^2)/n ) + 
                                        fitp$lambda[nlambda]*sum(abs(fits$coefficients)) 
    
  }
  cat("picasso\n")
  cat(paste("time", mean(time.picasso), "std", sd(time.picasso), "\n", sep=','))
  cat(paste("obj", mean(obj.picasso), "std", sd(obj.picasso), "\n", sep=','))


  cat("flare\n")
  cat(paste("time", mean(time.flare), "std", sd(time.flare), "\n", sep=','))
  cat(paste("obj", mean(obj.flare), "obj", sd(obj.flare), "\n", sep=','))

  cat("scalreg\n")
  cat(paste("time", mean(time.scalreg), "std", sd(time.scalreg), "\n", sep=','))
  cat(paste("obj", mean(obj.scalreg), "std", sd(obj.scalreg), "\n", sep=','))
}

test_elnet<- function(n = 10000, p = 5000, c = 0.5, nlambda = 20){
  library(picasso)
  library(glmnet)
  library(ncvreg)
  
  df <- generate_sim(n, p, c)

  ratio = 0.1
  trialN = 2 

  time.picasso <- rep(0, trialN)
  obj.picasso <- rep(0, trialN)

  time.glmnet<- rep(0, trialN)
  obj.glmnet<- rep(0, trialN)

  time.ncvreg <- rep(0, trialN)
  obj.ncvreg <- rep(0, trialN)

  idx <- nlambda 
  for (i in 1:trialN){
    t <- system.time(
            fitp<- picasso(df$X, df$Y, family="gaussian", standardize=FALSE, nlambda=nlambda, 
                      prec=1e-6,
                lambda.min.ratio = 0.1))

    time.picasso[i] <- t[1]
    obj.picasso[i] <- (sum((df$Y - df$X %*% fitp$beta[, idx] - fitp$intercept[idx])^2)/n) + 
                                  fitp$lambda[idx]*sum(abs(fitp$beta[, idx]))
  

    t <- system.time(fitglmnet<- glmnet(df$X, df$Y, lambda = fitp$lambda))
    time.glmnet[i] = t[1] 
    obj.glmnet[i] <- (sum((df$Y - df$X %*% fitglmnet$beta[, idx] - fitglmnet$a0[idx])^2)/n) + 
                                  fitglmnet$lambda[idx]*sum(abs(fitglmnet$beta[, idx]))



    t <- system.time(fitncvreg <- ncvreg(df$X, df$Y, lambda = fitp$lambda, penalty='lasso'))
    time.ncvreg[i] = t[1]
    obj.ncvreg[i] <- (sum((df$Y - df$X %*% fitncvreg$beta[2:(p+1), idx] - fitncvreg$beta[1,idx])^2)/n) + 
                                  fitncvreg$lambda[idx]*sum(abs(fitncvreg$beta[2:(p+1), idx]))
                                       
    
  }
  
  cat("picasso\n")
  cat(paste("time", mean(time.picasso), "std", sd(time.picasso), "\n", sep=','))
  cat(paste("obj", mean(obj.picasso), "std", sd(obj.picasso), "\n", sep=','))


  cat("glmnet\n")
  cat(paste("time", mean(time.glmnet), "std", sd(time.glmnet), "\n", sep=','))
  cat(paste("obj", mean(obj.glmnet), "obj", sd(obj.glmnet), "\n", sep=','))

  cat("ncvreg\n")
  cat(paste("time", mean(time.ncvreg), "std", sd(time.ncvreg), "\n", sep=','))
  cat(paste("obj", mean(obj.ncvreg), "std", sd(obj.ncvreg), "\n", sep=','))

}


test_elnet_nonlinear <- function(n = 10000, p = 5000, c = 0.5, nlambda = 20, penalty='scad'){
  library(picasso)
  library(ncvreg)
  
  df <- generate_sim(n, p, c)
  
  ratio = 0.1
  trialN = 2 

  time.picasso <- rep(0, trialN)
  obj.picasso <- rep(0, trialN)

  time.ncvreg <- rep(0, trialN)
  obj.ncvreg <- rep(0, trialN)

  idx <- nlambda
  for (i in 1:trialN){
    t <- system.time(
            fitp<- picasso(df$X, df$Y, family="gaussian", method=tolower(penalty), 
                      standardize=FALSE, nlambda=nlambda, gamma=3,
                      prec=1e-6,
                lambda.min.ratio = 0.1))

    time.picasso[i] = t[1] 
    obj.picasso[i] <- (sum((df$Y - df$X %*% fitp$beta[, idx] - fitp$intercept[idx])^2)/n) + 
                                 calc.penalty(fitp$lambda[idx], gamma=3, fitp$beta[,idx], penalty) 


    tryCatch({
      t <- system.time(fitncvreg <- ncvreg(df$X, df$Y, family="gaussian", 
                penalty=toupper(penalty), gamma=3, lambda = fitp$lambda))
      time.ncvreg[i] <- t[1]

      obj.ncvreg[i] <- (sum((df$Y - df$X %*% fitncvreg$beta[2:(p+1), idx] - fitncvreg$beta[1, idx])^2)/n) + 
                                  calc.penalty(fitncvreg$lambda[idx], gamma=3, fitncvreg$beta[2:(p+1),idx], penalty)},
      error=function(e){
        print("ncvreg runs into error")
        print(e)}
    )
  }
  
    
  cat("picasso\n")
  cat(paste("time", mean(time.picasso), "std", sd(time.picasso), "\n", sep=','))
  cat(paste("obj", mean(obj.picasso), "std", sd(obj.picasso), "\n", sep=','))

  cat("ncvreg\n")
  cat(paste("time", mean(time.ncvreg), "std", sd(time.ncvreg), "\n", sep=','))
  cat(paste("obj", mean(obj.ncvreg), "std", sd(obj.ncvreg), "\n", sep=','))

}


test_lognet <- function(n = 10000, p = 5000, c = 1.0, nlambda = 20, ratio=0.1){
  library(glmnet)
  library(picasso)
  library(ncvreg)
  
  df <- generate_sim_lognet(n, p, c)
  
  ratio = 0.1
  trialN = 2 

  obj.picasso <- rep(0, trialN)
  time.picasso <- rep(0, trialN)
  esterr.picasso <- rep(0, trialN)

  obj.glmnet<- rep(0, trialN)
  time.glmnet<- rep(0, trialN)
  esterr.glmnet<- rep(0, trialN)

  obj.ncvreg<- rep(0, trialN)
  time.ncvreg <- rep(0, trialN)
  esterr.ncvreg <- rep(0, trialN)

  idx = nlambda 

  for (i in 1:trialN){
    t <- system.time(
            fitp<- picasso(df$X, df$Y, family="binomial", standardize=FALSE, nlambda=nlambda, 
                      prec=1e-6,
                lambda.min.ratio = 0.1))

    time.picasso[i] = t[1] 

    rp = df$X%*%fitp$beta[,idx]+fitp$intercept[idx]
    obj.picasso[i] = sum(log(1+exp(rp))-df$Y*rp)/n+fitp$lambda[idx]*sum(abs(fitp$beta[,idx]))
  

    t <- system.time(fitglmnet<- glmnet(df$X, df$Y, family='binomial', lambda = fitp$lambda))
    time.glmnet[i] = t[1] 

    rp = df$X%*%fitglmnet$beta[,idx]+fitglmnet$a0[idx]
    obj.glmnet[i] <- sum(log(1+exp(rp))-df$Y*rp)/n+fitglmnet$lambda[idx]*sum(abs(fitglmnet$beta[,idx]))


    tryCatch({
      t <- system.time(fitncvreg <- ncvreg(df$X, df$Y, family='binomial', 
              penalty='lasso',
              eps=1e-4,
              lambda=fitp$lambda))
      time.ncvreg[i] = t[1]
                                       
      rp = df$X%*%fitncvreg$beta[2:(p+1),idx]+fitncvreg$beta[1,idx]
      obj.ncvreg[i] <- sum(log(1+exp(rp))-df$Y*rp)/n+fitncvreg$lambda[idx]*sum(abs(fitncvreg$beta[2:(p+1),idx]))
      },
      error=function(e){
        print("ncvreg runs into error")
        print(e)}
    )
  }

    
  cat("picasso\n")
  cat(paste("time", mean(time.picasso), "std", sd(time.picasso), "\n", sep=','))
  cat(paste("obj", mean(obj.picasso), "std", sd(obj.picasso), "\n", sep=','))


  cat("glmnet\n")
  cat(paste("time", mean(time.glmnet), "std", sd(time.glmnet), "\n", sep=','))
  cat(paste("obj", mean(obj.glmnet), "obj", sd(obj.glmnet), "\n", sep=','))

  cat("ncvreg\n")
  cat(paste("time", mean(time.ncvreg), "std", sd(time.ncvreg), "\n", sep=','))
  cat(paste("obj", mean(obj.ncvreg), "std", sd(obj.ncvreg), "\n", sep=','))

}


calc.penalty <- function(lambda, gamma, beta, penalty){
  d <- length(beta)
  value <- 0
  beta <- abs(beta)
  for (i in 1:d){
    if (penalty=='mcp'){
      if (beta[i] > gamma*lambda){
        value <- value + lambda*lambda*gamma/2 
      } else {
        value <- value + lambda*(beta[i] - beta[i]*beta[i]/(2*lambda*gamma))
      }
    } else if (penalty == 'scad') {
      if (beta[i] > gamma*lambda){
        value <- value + (gamma+1)*lambda*lambda/2
      } else if (beta[i] > lambda ){
        value <- value - (beta[i]*beta[i] - 2*lambda*gamma*beta[i] + lambda*lambda)/(2*(gamma-1)) 
      } else {
        value <- value + lambda*beta[i]
      }
    }
  }
  return(value)
}

test_lognet_nonlinear <- function(n = 10000, p = 5000, c = 1.0, nlambda = 20, penalty){
  library(picasso)
  library(ncvreg)
  library(cvplogistic)
  
  df <- generate_sim_lognet(n, p, c)
  
  ratio = 0.1
  trialN = 2 

  time.picasso <- rep(0, trialN)
  obj.picasso <- rep(0, trialN)


  time.ncvreg <- rep(0, trialN)
  obj.ncvreg <- rep(0, trialN)

  time.cvp <- rep(0, trialN)
  obj.cvp <- rep(0, trialN)


  idx <- nlambda

  picasso.prec <- 1e-4
  if (penalty == 'mcp')
    picasso.prec <- 1e-3

  for (i in 1:trialN){
    t <- system.time(
            fitp<- picasso(df$X, df$Y, family="binomial", standardize=FALSE, nlambda=nlambda, 
                      prec=picasso.prec, method=tolower(penalty), gamma=3,
                lambda.min.ratio = 0.1))

    time.picasso[i] = t[1]  

    rp = df$X%*%fitp$beta[,idx]+fitp$intercept[idx]
    obj.picasso[i] = sum(log(1+exp(rp))-df$Y*rp)/n +
                      calc.penalty(fitp$lambda[idx], gamma=3, fitp$beta[,idx], penalty)


    tryCatch(
      {
      t <- system.time(fitncvreg <- ncvreg(df$X, df$Y, family='binomial', penalty=toupper(penalty), 
                gamma=3, eps=1e-6, 
                lambda = fitp$lambda))
      time.ncvreg[i] = t[1]
                                       
      rp = df$X%*%fitncvreg$beta[2:(p+1),idx]+fitncvreg$beta[1,idx]
      obj.ncvreg[i] = sum(log(1+exp(rp))-df$Y*rp)/n +
                      calc.penalty(fitncvreg$lambda[idx], gamma=3, fitncvreg$beta[2:(p+1),idx], penalty)},
      error=function(e){
        print("ncvreg runs into error")
        print(e)}
    )

 
    t <- system.time(fitcvp <- cvplogistic(df$Y, df$X, penalty=tolower(penalty), epsilon=1e-2, 
            nlambda=nlambda, lambda.min=fitp$lambda[nlambda]))
    time.cvp[i] <- t[1]  

    rp <- df$X%*%fitcvp[[1]][2:(p+1),idx]+fitcvp[[1]][1, idx]
    obj.cvp[i] <- sum(log(1+exp(rp))-df$Y*rp)/n +
                      calc.penalty(fitcvp[[2]][idx], gamma=3, fitcvp[[1]][2:(p+1),idx], penalty)
    
  }
  
    
  cat("picasso\n")
  cat(paste("time", mean(time.picasso), "std", sd(time.picasso), "\n", sep=','))
  cat(paste("obj", mean(obj.picasso), "std", sd(obj.picasso), "\n", sep=','))


  cat("ncvreg\n")
  cat(paste("time", mean(time.ncvreg), "std", sd(time.ncvreg), "\n", sep=','))
  cat(paste("obj", mean(obj.ncvreg), "std", sd(obj.ncvreg), "\n", sep=','))


  #cat("cvp\n")
  #cat(paste("time", mean(time.cvp), "std", sd(time.cvp), "\n", sep=','))
  #cat(paste("obj,", mean(obj.cvp), "std", sd(obj.cvp), "\n", sep=','))

}