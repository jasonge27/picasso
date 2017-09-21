#include <picasso/objective.hpp>

namespace picasso {
class SqrtMSEObjective : public ObjFunction {
 private:
  std::vector<double> w;
  std::vector<double> Xb;
  std::vector<double> r;

  // wXX[j] = sum(w*X[j]*X[j])
  std::vector<double> wXX;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant
  double a, g;
  double L;  // sqrt(MSE)
  double sum_r;
  double sum_r2;

 public:
  SqrtMSEObjective(const double *xmat, const double *y, int n, int d)
      : ObjFunction(xmat, y, n, d) {
    a = 0.0;
    g = 0.0;
    L = 0.0;
    w.resize(n);
    Xb.resize(n);
    r.resize(n);

    if (m_param.include_intercept) {
      double avr_y = 0.0;
      for (int i = 0; i < n; i++) avr_y += Y[i];
      avr_y = avr_y / n;
      model_param.intercept = avr_y;
    }

    update_auxiliary();

    // saturated fvalue = 0
    deviance = fabs(eval());
  }

  double coordinate_descent(RegFunction *regfunc, int idx) {
    g = 0.0;
    a = 0.0;

    double tmp;
    // g = (<wXX, model_param.beta> + <r, X>)/n
    // a = sum(wXX)/n
    for (int i = 0; i < n; i++) {
      tmp = w[i] * X[idx][i] * X[idx][i];
      g += tmp * model_param.beta[idx] + r[i] * X[idx][i];
      a += wXX[idx];
    }
    g = g / (n * L);
    a = a / (n * L);

    tmp = beta[idx];
    if (fabs(g) > thr)
      model_param.beta[idx] = regfunc->threshold(g) / a;
    else
      model_param.beta[idx] = 0.0;

    tmp = model_param.beta[idx] - tmp;
    // Xb += delta*X[idx*n]
    for (int i = 0; i < n; i++) Xb[i] = Xb[i] + tmp * X[idx][i];

    sum_r = 0.0;
    // r -= delta*X
    for (int i = 0; i < n; i++) {
      r[i] = r[i] - X[idx][i] * tmp;
      sum_r += r[i];
    }

    return (model_param.beta[idx]);
  }

  void intercept_update() {
    tmp = sum_r / n;
    model_param.intercept += tmp;

    for (int i = 0; i < n; i++) {
      r[i] = r[i] - tmp;
      sum_r = 0.0;
    }
  }

  void set_model_param(ModelParam &other_param) {
    model_param = other_param;
    for (int i = 0; i < n; i++) {
      Xb[i] = 0.0;
      for (int j = 0; j < d; j++) Xb[i] = X[j][i] * model_param.beta[j];
    }
  }

  void update_auxiliary() {
    sum_r = 0.0;
    for (int i = 0; i < n; i++) {
      r[i] = Y[i] - Xb[i];
      sum_r += r[i];
      sum_r2 += r[i] * r[i];
    }
    L = sqrt(sum_r2 / n);

    for (int i = 0; i < n; i++) {
      w[i] = 1 - r[i] * r[i] / sum_r2;

      for (int idx = 0; idx < d; idx++) {
        wXX[idx] = 0.0;
        gr[idx] = 0.0;
        for (int i = 0; i < n; i++) {
          wXX[idx] += w[i] * X[idx][i] * X[idx][i];
          gr[idx] += r[i] * X[idx][i];
        }
        gr[idx] = gr[idx] / (n * L);
      }
    }
  }

  double get_local_change(double old, int idx) {
    if (idx >= 0) {
      double tmp = old - model_param.beta[idx];
      return (wXX[idx] * tmp * tmp / (2 * L * n));
    } else {
      double tmp = old - model_param.intercept;
      return (fabs(tmp));
    }
  }

  void eval() { return (L); }
};

}  // namespace picasso