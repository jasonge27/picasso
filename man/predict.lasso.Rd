\name{predict.lasso}
\alias{predict.lasso}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Prediction for an object with S3 class \code{"lasso"}
}
\description{
Predicting responses of the given design data.
}
\usage{
\method{predict}{lasso}(object, newdata, lambda.idx = c(1:3), Y.pred.idx = c(1:5), \dots)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
  An object with S3 class \code{"lasso"} 
  }
  \item{newdata}{
  An optional data frame in which to look for variables with which to predict. If omitted, the traning data of the are used.
  }
  \item{lambda.idx}{
  The indices of the regularizaiton parameters in the solution path to be displayed. The default values are \code{c(1:3)}.
  }
  \item{Y.pred.idx}{
  The indices of the predicted response vectors in the solution path to be displayed. The default values are \code{c(1:5)}.
  }
  \item{\dots}{
  Arguments to be passed to methods.
  }
}
\details{
\code{predict.lasso} produces predicted values of the responses of the \code{newdata} from the estimated \code{beta} values in the \code{object}, i.e.
\deqn{
    \hat{Y} = \hat{\beta}_0 + X_{new} \hat{\beta}.
    }\cr
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{Y.pred}{
  The predicted response vectors based on the estimated models.
  }
}
\author{
Xingguo Li, Tuo Zhao, Tong Zhang and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
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
eps = rt(n=n,df=n-1)
beta = c(rep(sqrt(1/3),3),rep(0,d-3))
Y.fit = X.fit\%*\%beta+eps

## Regression with "l1".
out=picasso(X=X.fit,Y=Y.fit,lambda=lambda,family="gaussian")

## Display results
Y=predict(out,X.pred)
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}