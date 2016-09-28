\name{predict.poisson}
\alias{predict.poisson}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Prediction for an object with S3 class \code{"poisson"}
}
\description{
Predicting responses of the given design data.
}
\usage{
\method{predict}{poisson}(object, newdata, lambda.idx = c(1:3), p.pred.idx = c(1:5), \dots)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
  An object with S3 class \code{"poisson"} 
  }
  \item{newdata}{
  An optional data frame in which to look for variables with which to predict. If omitted, the traning data of the are used.
  }
  \item{lambda.idx}{
  The indices of the regularizaiton parameters in the solution path to be displayed. The default values are \code{c(1:3)}.
  }
  \item{p.pred.idx}{
  The indices of the predicted response vectors in the solution path to be displayed. The default values are \code{c(1:5)}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
\code{predict.poisson} produces predicted response mean (which is also the parameter for poisson distribution) for the \code{newdata} from the estimated \code{beta} values in the \code{object}, i.e.
\deqn{
    \hat{p} = e^{\hat{\beta}_0 + X_{new} \hat{\beta}}.
    }\cr
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{p.pred}{
  The predicted response mean vectors based on the estimated models.
  }
}
\author{
Jason Ge, Xingguo Li, Mengdi Wang, Tong Zhang, Han Liu and Tuo Zhao\cr
Maintainer: Jason Ge <jiange@princeton.edu>
}

\examples{
## generate data
set.seed(123)
n = 100
d = 200
d1 = 10
rho0 = 0.3
lambda = c(3:1)*sqrt(log(d)/n)
Sigma = matrix(0,nrow=d,ncol=d)
Sigma[1:d1,1:d1] = rho0
diag(Sigma) = 1
mu = rep(0,d)
X = mvrnorm(n=2*n,mu=mu,Sigma=Sigma)
X.fit = X[1:n,]
X.pred = X[(n+1):(2*n),]
beta = c(rep(sqrt(1/3),3),rep(0,d-3))
Y.fit = exp(X.fit\%*\%beta)/(1+exp(X.fit\%*\%beta))

## Regression with "l1".
out=picasso(X=X.fit,Y=Y.fit,lambda=lambda,family="binomial")

## Display results
p=predict(out,X.pred)
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}