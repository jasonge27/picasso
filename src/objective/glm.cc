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
  double wXX;
  double sum_r;
  double sum_w;

public:
  LogisticObjective(ObjType obj_type, const double *xmat, const double *y, int n, int d):
    ObjFunction(obj_type, xmat, y, n, d){
    a = 0.0;
    g = 0.0;
    p.resize(d);
    w.resize(n);
    Wb.resize(n);
    r.resize(n);

    if (m_param.include_intercept){
      double avr_y = 0.0;
      for (int i = 0; i < n; i++)
       avr_y += Y[i];
      avr_y = avr_y/n;
      model_param.intercept = log(avr_y/(1-avr_y));
    }

    update_auxiliary();
  }

  double coordinate_descent(int idx, double thr){
    g = 0.0;
    a = 0.0;

    double tmp;
    // g = (<wXX, model_param.beta> + <r, X>)/n
    // a = sum(wXX)/n
    for (int i = 0; i < n; i++){
      tmp = w[i]*X[idx*n+i]*X[idx*n+i];
      g += tmp * model_param.beta[idx] + r[i]*X[idx*n+i];
      a += wXX[idx];
    }
    g = g / n;
    a = a / n;

    tmp = beta[idx];
    if (fabs(g) > thr)
      model_param.beta[idx] = soft_thresh_l1(g, thr) / a;
    else 
      model_param.beta[idx] = 0.0;
    
    // Xb += delta*X[idx*n]
    for (int i = 0; i < n; i++)
      Xb[i] = Xb[i] + (model_param.beta[idx]-tmp)*X[idx*n+i];
    
    sum_r = 0.0;
    // r -= delta*w*X 
    for (int i = 0; i < n; i++){
      r[i] = r[i] - w[i]*X[idx*n+i]*(model_param.beta[idx]-tmp);
      sum_r += r[i];
    }
    
    return(model_param.beta[idx]);
  }

  void intercept_update(){
    tmp = sum_r / sum_w; 
    model_param.intercept += tmp;

    sum_r = 0.0;
    for (i = 0; i < n; i++){
      r[i] = r[i] - tmp * w[i];
      sum_r += r[i];
    }
  }

  void update_auxiliary(){
    sum_w = 0.0;
    sum_r = 0.0;
    for (int i = 0; i < n; i++){
      p[i] = 1.0/(1.0 + exp(-model_param.intercept - Xb[i]));
      w[i] = p[i]*(1-p[i]);
      r[i] = Y[i] - p[i];
      sum_w += w[i];
      sum_r += r[i];
    }

    for (int idx = 0; idx < d; idx++){
      wXX[idx] = 0.0;
      gr[idx] = 0.0;
      for (int i = 0; i < n; i++){
        wXX[idx] += w[i]*X[idx*n+i]*X[idx*n+i];
        gr[idx] += r[i]*X[idx*n+i];
      }
      gr[idx] = gr[idx] / n;
    }
  }

  double get_local_change(double old, int idx){
    if (idx >= 0){
      double tmp = old - model_param.beta[idx];
      return(wXX[idx]*tmp*tmp/(2*n)); 
    } else {
      double tmp = old - model_param.intercept;
      return(sum_w*tmp*tmp/(2*n)); 
    }
  }

  double eval() {
    double v = 0.0;
    for (int i = 0; i < n; i++){
        v -= Y[i]*(model_param.intercpet+Xb[i]); 
    }

    for (int i = 0; i < n; i++)
    if (p[i] > 1e-8) {
        v -= (log(p[i]) - model_param.intercept - Xb[i]);
    }

    v = v/n;
    return(v);
  }
}
} // namespace picasso