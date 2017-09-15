#include <picasso/objective.h>

class LogisticObjective: public ObjFunction {
private:
  std::vector<double> p;
  std::vector<double> w;
  std::vector<double> Xb;
  std::vector<double> r;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant 
  double a, g; 
  double wXX;
  double sum_r;
  double sum_w;

public:
  double coordinate_descent(int idx, double thr){
    g = 0.0;
    a = 0.0;

    double tmp;
    for (int i = 0; i < n; i++){
      tmp = w[i]*X[idx*n+i]*X[idx*n+i];
      g += tmp*model_param.beta[idx] + r[i]*X[idx*n+i];
      a += tmp;
    }
    g = g / n;
    a = a / n;

    tmp = beta[idx];
    if (fabs(g) > thr)
      model_param.beta[idx] = soft_thresh_l1(g, thr) / a;
    else 
      model_param.beta[idx] = 0.0;
    
    for (int i = 0; i < n; i++)
      Xb[i] = Xb[i] + (model_param.beta[idx]-tmp)*X[idx*n+i];
    
    sum_r = 0.0;
    for (int i = 0; i < n; i++){
      r[i] = r[i] - w[i]*X[idx*n+i]*(model_param.beta[idx]-tmp);
      sum_r += r[i];
    }
    
    return model_param.beta[idx];
  }

  void update_auxiliary(){
    sum_w = 0.0;
    for (int i = 0; i < n; i++){
      p[i] = 1.0/(1.0 + exp(-model_param.intercept - Xb[i]));
      w[i] = p[i]*(1-p[i]);
      sum_w += w[i];
    }

    for (int idx = 0; idx < d; idx++){
      wXX[idx] = 0.0;
      for (int i = 0; i < n; i++)
        wXX[idx] += w[i]*X[idx*n+i]*X[idx*n+i];
    }
  }

  void intercept_update(){
    tmp = sum_r / sum_w; 

    sum_r = 0.0;
    for (i = 0; i < n; i++){
      r[i] = r[i] - tmp * w[i];
      sum_r += r[i];
    }
            
    model_param.intercept += tmp;
  }

  double get_local_change(double old, int idx){
    if (idx >= 0){
      double tmp = old - model_param.beta[idx];
      return wXX[idx]*tmp*tmp/(2*n); 
    } else {
      double tmp = old - model_param.intercept;
      return sum_w*tmp*tmp/(2*n); 
    }
  }

  double get_global_change(double old, int idx){
    if (idx >= 0){
      double tmp = model_param.beta[idx] - old;
      return wXX[idx]*tmp*tmp/(2*n);
    } else {
      double tmp = model_param.intercept - old;
      dev = sum_w*tmp*tmp/(2*n);
      return dev; 
    }
  } 
}