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

  ModelParam get_model_param();
  
  double coordinate_descent(int idx, double thr);
  void intercept_update()
  double get_local_change();
  double get_local_change(ModelParam model_param, int idx); 

  vector<double> get_grad();
  double get_grad(int idx);

  double get_model_coef(int idx);
  double eval();
};

}

#endif PICASSO_OBJECTIVE_H