#include <picasso/objective.h>

namespace picasso{

class LogisticObjective: public ObjFunction {
private:
  std::vector<double> p;
  std::vector<double> w;
  std::vector<double> Xb;
  std::vector<double> r;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant 
  double a, g; 
  double local_change;

public:
  double coordinate_descent(int idx, double thr){
    /*
    g = 0.0;
    a = 0.0;

    double tmp;
    for (int i = 0; i < n; i++){
      tmp = w[i]*X[idx*n+i]*X[idx*n+i];
      g += tmp*m_model_param.beta[idx] + r[i]*X[idx*n+i];
      a += tmp;
    }
    g = g / n;
    a = a / n;

    tmp = beta[idx];
    if (fabs(g) > thr){
      beta[idx] = soft_thresh_l1(g, thr) / a;
    } else {
      beta[idx] = 0.0;
    }

    for (int i = 0; i < n; i++)
      Xb[i] = Xb[i] + (model_param.beta[idx] - tmp)*X[idx*n+i];
    
    for (int i = 0; i < n; i++)
      r[i] = r[i] - w[i]*X[idx*n+i] * (model_param.beta[idx] - tmp);
    
    local_change = 0.0;
    tmp = ()*/
    return 0;
  }


};

} // namespace picasso