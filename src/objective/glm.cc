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
  double local_change;

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
    
    for (int i = 0; i < n; i++)
      r[i] = r[i] - w[i]*X[idx*n+i]*(model_param.beta[idx]-tmp);
    
    local_change = 0.0;
    tmp = (model_param.beta[idx]-tmp)*(model_param.beta[idx]-tmp);
    for (int i = 0; i < n; i++)
      local_change += w[i]*X[idx*n+i]*X[idx*n+i];
    local_change = local_change * tmp / (2*n);

    return model_param.beta[idx];
  }

  void intercept_update(){
    double sum_r = 0.0;
    double sum_w = 0.0;
    for (int i = 0; i < n; i++){
      sum_r += r[i];
      sum_w += w[i];
    }
         
    tmp = sum_r / sum_w; 
    for (i = 0; i < n; i++)
      r[i] = r[i] - tmp * w[i];
            
    model_param.intercept += tmp;
    local_change = sum_w * tmp*tmp/ (2*n);
  }

  double get_global_change(ModelParam old_model_param, int idx) {
    double global_change = 0.0;
    double dev = 0.0;

    for (int i = 0; i < d; i++){
      double tmp = model_param.beta[i] - old_model_param.beta[i];
      tmp = tmp*tmp;
      dev = 0.0;
      for (int j = 0; j < n; j++)
        dev += w[j] * X[j*n+i] * X[j*n+i] * tmp;
      dev = dev / (2*n);

      if (dev > global_change)
        global_change = dev;
    }
  
    tmp = model_param.intercept - old_model_param.intercept;
    dev = sum_w * tmp*tmp/(2*n);
    if (dev > global_change)
      global_change = dev;

    return global_change;
  }

}