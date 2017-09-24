#ifndef PICASSO_OBJECTIVE_H
#define PICASSO_OBJECTIVE_H

#include <cmath>
#include <vector>

namespace picasso {

class ModelParam {
 public:
  int d;
  std::vector<double> beta;
  double intercept;

  ModelParam(int dim) {
    d = dim;
    beta.resize(d, 0);
    intercept = 0.0;
  }
};

double threshold_l1(double x, double thr) {
  if (x > thr)
    return x - thr;
  else if (x < -thr)
    return x + thr;
  else
    return 0;
}

class RegFunction {
 public:
  virtual double threshold(double x) = 0;
};

class RegL1 : public RegFunction {
 private:
  double m_lambda;

 public:
  void set_param(double lambda) { m_lambda = lambda; }

  double threshold(double x) { return threshold_l1(x, m_lambda); }
};

class RegSCAD : public RegFunction {
 private:
  double m_lambda;
  double m_gamma;

 public:
  void set_param(double lambda, double gamma) {
    m_lambda = lambda;
    m_gamma = gamma;
  };

  double threshold(double x) {
    if (fabs(x) > fabs(m_gamma * m_lambda)) {
      return x;
    } else {
      if (fabs(x) > fabs(2 * m_lambda)) {
        return threshold_l1(x, m_gamma * m_lambda / (m_gamma - 1)) /
               (1 - 1 / (m_gamma - 1));
      } else {
        return threshold_l1(x, m_lambda);
      }
    }
  };
};

class RegMCP : public RegFunction {
 private:
  double m_lambda;
  double m_gamma;

 public:
  void set_param(double lambda, double gamma) {
    m_lambda = lambda;
    m_gamma = gamma;
  }

  double threshold(double x) {
    if (fabs(x) > fabs(m_gamma * m_lambda)) {
      return x;
    } else {
      if (fabs(x) > fabs(2 * m_lambda)) {
        return threshold_l1(x, m_gamma * m_lambda / (m_gamma - 1)) /
               (1 - 1 / (m_gamma - 1));
      } else {
        return threshold_l1(x, m_lambda);
      }
    }
  }
};

class ObjFunction {
 private:
  int n;  // sample number
  int d;  // sample dimension

  std::vector<std::vector<double> > X;
  std::vector<double> Y;

  std::vector<double> gr;

  ModelParam model_param;

  double deviance;

 public:
  ObjFunction(const double *xmat, const double *y, int n, int d)
      : model_param(d) {
    this->d = d;
    this->n = n;
    Y.resize(n);
    X.resize(d);
    gr.resize(d);

    for (int i = 0; i < n; i++) Y[i] = y[i];

    for (int j = 0; j < d; j++) {
      X[j].resize(n);
      for (int i = 0; i < n; i++) X[j][i] = xmat[j * n + i];
    }
  };

  int get_dim() { return d; }

  double get_grad(int idx) { return gr[idx]; };

  // fabs(null fvalue - saturated fvalue)
  double get_deviance() { return (deviance); };

  double get_model_coef(int idx) {
    return ((idx < 0) ? model_param.intercept : model_param.beta[idx]);
  }
  void set_model_coef(double value, int idx) {
    if (idx >= 0)
      model_param.beta[idx] = value;
    else
      model_param.intercept = value;
  }

  ModelParam get_model_param() { return model_param; };

  // reset model param and also update related aux vars
  virtual void set_model_param(ModelParam &other_param) = 0;

  // coordinate descent
  virtual double coordinate_descent(RegFunction *regfun, int idx) = 0;

  // update intercept term
  virtual void intercept_update() = 0;

  // update gradient and other aux vars
  virtual void update_key_aux() = 0;
  virtual void update_auxiliary() = 0;

  // compute quadratic change of fvalue on the idx dimension
  virtual double get_local_change(double old, int idx) = 0;

  // unpenalized function value
  virtual double eval() = 0;

  ~ObjFunction(){};
};

}  // namespace picasso

#endif  // PICASSO_OBJECTIVE_H