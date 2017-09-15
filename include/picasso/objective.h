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

  std::vector<double> m_X;
  std::vector<double> m_Y;

  ObjType m_obj_type;
  ModelParam m_model_param;
  

public:
  ObjFunction(ObjType obj_type, const double * xmat, const double * y, int n, int d);

  ModelParam get_model_param() {return m_model_param;};
  virtual void set_model_param(ModelParam &other_param) = 0; // TODO
  
  virtual double coordinate_descent(int idx, double thr) = 0;
  virtual void intercept_update() = 0;

  virtual void update_auxiliary() = 0;

  virtual double get_local_change(double old, int idx) = 0;

  virtual std::vector<double> get_grad() = 0;
  virtual double get_grad(int idx) = 0;

  virtual double get_model_coef(int idx) = 0;
  virtual double eval() = 0;

  ~ObjFunction() {};
}; 

} // namespace picasso

#endif // PICASSO_OBJECTIVE_H