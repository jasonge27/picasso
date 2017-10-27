set.seed(2016)
library(glmnet)


n <- 1000; p <- 1000; c <- 0.1
# n sample number, p dimension, c correlation parameter
X <- scale(matrix(rnorm(n*p),n,p)+c*rnorm(n))/sqrt(n-1)*sqrt(n) # n is smaple number, 
s <- 20  # sparsity level
true_beta <- c(runif(s), rep(0, p-s))
Y <- X%*%true_beta + rnorm(n)
fitg<-glmnet(X,Y,family="gaussian")

 # the minimal estimation error |\hat{beta}-beta| / |beta|
cat("min estimation error using glmnet\n")
min(apply(abs(fitg$beta - true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))

library(picasso)
fitp <- picasso(X, Y, family="gaussian", method="scad")
cat("min estimation error using picasso\n")
min(apply(abs(fitp$beta-true_beta), MARGIN=2, FUN=sum))/sum(abs(true_beta))

cat("\n-----------------------------------------------------------\n")
cat("-------Testing Logistic Regression with L1 Penalty------\n")
source("test_picasso.R")

cat("------Comparisions of estimation errors.------\n")
cat("-----------------------\n")
cat("---------n=2000, p=1000, c=0.1---------\n")
test_lognet(n=2000, p=1000, c=0.1)
cat("-----------------------\n")
cat("---------n=2000, p=1000, c=0.5---------\n")
test_lognet(n=2000, p=1000, c=0.5)
cat("-----------------------\n")
cat("---------n=2000, p=1000, c=1.0---------\n")
test_lognet(n=2000, p=1000, c=1.0)

cat("\n-----------------------------------------------------------\n")
cat("-------Testing Logistic Regression with SCAD/MCP Penalty------\n")
cat("-----------------------\n")
cat("---------n=3000, p=3000, c=0.1---------\n")
test_lognet_nonlinear(n=3000, p =3000, c=0.1)
cat("-----------------------\n")
cat("---------n=3000, p=3000, c=0.5---------\n")
test_lognet_nonlinear(n=3000, p =3000, c=0.5)
cat("-----------------------\n")
cat("---------n=3000, p=3000, c=1.0---------\n")
test_lognet_nonlinear(n=3000, p =3000, c=1.0)


cat("\n-----------------------------------------------------------\n")
cat("-------Testing Square Root Lasso------\n")
cat("-----------------------\n")
cat("---------n=500, p=400, c=0.5---------\n")
test_sqrt_mse(n=500, p =400, c=0.5)
cat("-----------------------\n")
cat("---------n=500, p=800, c=0.5---------\n")
test_sqrt_mse(n=500, p =800, c=0.5)
cat("-----------------------\n")
cat("---------n=500, p=1600, c=0.5---------\n")
test_sqrt_mse(n=500, p =1600, c=0.5)
