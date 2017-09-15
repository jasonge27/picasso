#ifndef PICASSO_OBJECTIVE_H
#define PICASSO_OBJECTIVE_H

#include <vector>

namespace picasso{

class ModelParam{
public:
  int d;
  std::vector<double> coef;
  double intercept;

  ModelParam(int dim){
    d = dim;
    coef.resize(d, 0);
    intercept = 0.0;
  }
};

enum ObjType {MSE, Logistic, Poission, SqrtMSE};

class ObjFunction {
private:
  int n; // sample number
  int d; // sample dimension

  std::vector<std::vector<double> > X;
  std::vector<double> Y;

  std::vector<double> gr;

  ObjType obj_type;
  ModelParam model_param;

  double deviance;

public:
  ObjFunction(ObjType obj_type, const double * xmat, 
      const double * y, int n, int d):model_param(d){
    this->obj_type = obj_type;
    this->d = d;
    this->n = n;
    Y.resize(n);
    X.resize(d);

    for (int i = 0; i < n; i++)
      Y[i] = y[i];
    
    for (int j = 0; j < d; j++){
      X[j].resize(n);
      for (int i = 0; i < n; i++)
        X[j][i] = xmat[j*n+i];
    }
  };

  int get_dim() {return d;}

  double get_grad(int idx){return gr[idx]);

  // fabs(null fvalue - saturated fvalue)
  double get_deviance(){return(deviance)} ;

  double get_model_coef(int idx){
    return((idx<0) ? model_param.intercept : model_param.beta[idx])
  }
  void set_model_coef(double value, int idx){
    if (idx >= 0) 
      model_param.beta[idx]] = value;
    else
      model_param.intercept = value;
  }

  ModelParam get_model_param() {return(m_model_param);};

  // reset model param and also update related aux vars 
  virtual void set_model_param(ModelParam &other_param) = 0; 
  
  // coordinate descent
  virtual double coordinate_descent(int idx, double thr) = 0;
  
  // update intercept term
  virtual void intercept_update() = 0;

  // update gradient and other aux vars
  virtual void update_auxiliary() = 0;

  // compute quadratic change of fvalue on the idx dimension
  virtual double get_local_change(double old, int idx) = 0;

  // unpenalized function value
  virtual double eval() = 0;


  ~ObjFunction() {};
}; 

} // namespace picasso

#endif // PICASSO_OBJECTIVE_H