\name{predict.poisson}
\alias{predict.poisson}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Prediction for an object with S3 class \code{"poisson"}
}
\description{
Predict responses for new design data.
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
  Optional matrix/data frame used for prediction. If omitted, the training design matrix from fitting is used.
  }
  \item{lambda.idx}{
  The indices of the regularization parameters in the solution path to be displayed. The default values are \code{c(1:3)}.
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
Maintainer: Tuo Zhao <tourzhao@gatech.edu>
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{picasso}} and \code{\link{picasso-package}}.
}