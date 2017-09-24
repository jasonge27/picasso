#include <picasso/objective.hpp>

namespace picasso {
class GLMObjective : public ObjFunction {
 private:
  std::vector<double> p;
  std::vector<double> w;
  std::vector<double> Xb;
  std::vector<double> r;

  // wXX[j] = sum(w*X[j]*X[j])
  std::vector<double> wXX;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant
  double a, g;
  double sum_r;
  double sum_w;

 public:
  GLMObjective(const double *xmat, const double *y, int n, int d)
      : ObjFunction(xmat, y, n, d) {
    /*
    a = 0.0;
    g = 0.0;
    p.resize(d);
    w.resize(n);
    Xb.resize(n);
    r.resize(n);

    if (m_param.include_intercept) {
      double avr_y = 0.0;
      for (int i = 0; i < n; i++) avr_y += Y[i];
      avr_y = avr_y / n;
      model_param.intercept = log(avr_y / (1 - avr_y));
    }

    update_auxiliary();

    // saturated fvalue = 0
    deviance = fabs(eval());
    */
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
    g = g / n;
    a = a / n;

    tmp = beta[idx];
    if (fabs(g) > thr)
      model_param.beta[idx] = regfunc->threshold(g) / a;
    else
      model_param.beta[idx] = 0.0;

    // Xb += delta*X[idx*n]
    for (int i = 0; i < n; i++)
      Xb[i] = Xb[i] + (model_param.beta[idx] - tmp) * X[idx][i];

    sum_r = 0.0;
    // r -= delta*w*X
    for (int i = 0; i < n; i++) {
      r[i] = r[i] - w[i] * X[idx][i] * (model_param.beta[idx] - tmp);
      sum_r += r[i];
    }

    return (model_param.beta[idx]);
  }

  void intercept_update() {
    tmp = sum_r / sum_w;
    model_param.intercept += tmp;

    sum_r = 0.0;
    for (int i = 0; i < n; i++) {
      r[i] = r[i] - tmp * w[i];
      sum_r += r[i];
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
    update_key_aux();
    sum_w = 0.0;
    sum_r = 0.0;
    for (int i = 0; i < n; i++) {
      r[i] = Y[i] - p[i];
      sum_w += w[i];
      sum_r += r[i];
    }

    for (int idx = 0; idx < d; idx++) {
      wXX[idx] = 0.0;
      gr[idx] = 0.0;
      for (int i = 0; i < n; i++) {
        wXX[idx] += w[i] * X[idx][i] * X[idx][i];
        gr[idx] += r[i] * X[idx][i];
      }
      gr[idx] = gr[idx] / n;
    }
  }

  double get_local_change(double old, int idx) {
    if (idx >= 0) {
      double tmp = old - model_param.beta[idx];
      return (wXX[idx] * tmp * tmp / (2 * n));
    } else {
      double tmp = old - model_param.intercept;
      return (sum_w * tmp * tmp / (2 * n));
    }
  }
};

class LogisticObjective : public GLMObjective {
 public:
  void update_key_aux() {
    for (int i = 0; i < n; i++) {
      p[i] = 1.0 / (1.0 + exp(-model_param.intercept - Xb[i]));
      w[i] = p[i] * (1 - p[i]);
    }
  }
  double eval() {
    double v = 0.0;
    for (int i = 0; i < n; i++) v -= Y[i] * (model_param.intercept + Xb[i]);

    for (int i = 0; i < n; i++)
      if (p[i] > 1e-8) v -= (log(p[i]) - model_param.intercept - Xb[i]);

    return (v / n);
  }
};

class PoissonObjective : public ObjFunction {
 public:
  void update_key_aux() {
    for (int i = 0; i < n; i++) {
      p[i] = exp(model_param.intercept + Xb[i]);
      w[i] = p[i];
    }
  }

  double eval() {
    double v = 0.0;
    for (int i = 0; i < n; i++)
      v = v + p[i] - Y[i] * (model_param.intercept + Xb[i]);
    return (v / n);
  }
};

}  // namespace picasso