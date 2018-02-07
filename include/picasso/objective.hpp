#ifndef PICASSO_OBJECTIVE_H
#define PICASSO_OBJECTIVE_H

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include <ctime>


namespace picasso {

class ModelParam {
 public:
  int d;
  Eigen::ArrayXd beta;
  double intercept;

  ModelParam(int dim) {
    d = dim;
    beta.resize(d);
    beta.setZero();
    intercept = 0.0;
  }
};

class RegFunction {
 public:
  virtual double threshold(double x) = 0;
  virtual void set_param(double lambda, double gamma) = 0;
  virtual double get_lambda() = 0;

  virtual ~RegFunction(){};

  double threshold_l1(double x, double thr) {
    if (x > thr)
      return x - thr;
    else if (x < -thr)
      return x + thr;
    else
      return 0;
  }
};

class RegL1 : public RegFunction {
 private:
  double m_lambda;

 public:
  void set_param(double lambda, double gamma) { m_lambda = lambda; }
  double get_lambda() { return m_lambda; };
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
  double get_lambda() { return m_lambda; };

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
  double get_lambda() { return m_lambda; };

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
 protected:
  int n;  // sample number
  int d;  // sample dimension

  Eigen::ArrayXXd X;
  Eigen::ArrayXd Y;

  Eigen::ArrayXd gr;
  Eigen::ArrayXd Xb;

  ModelParam model_param;

  double deviance;

 public:
  ObjFunction(const double *xmat, const double *y, int n, int d, bool usePypthon=false)
      : model_param(d) {
    this->d = d;
    this->n = n;
    Y.resize(n);
    gr.resize(d);

    Xb.resize(n);
    Xb.setZero();

    for (int i = 0; i < n; i++) Y[i] = y[i];

    X.resize(n, d);
    if(!usePypthon)
      for (int j = 0; j < d; j++) {
        for (int i = 0; i < n; i++) X(i, j) = xmat[j * n + i];
      }
    else
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) X(i, j) = xmat[i * d + j];
      }
  };

  int get_dim() { return d; }
  int get_sample_num() { return n; }

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
  Eigen::ArrayXd get_model_Xb() const { return Xb; };

  const ModelParam &get_model_param_ref() { return model_param; };
  const Eigen::ArrayXd &get_model_Xb_ref() const { return Xb; };

  // reset model param and also update related aux vars
  void set_model_param(ModelParam &other_param) {
    model_param.d = other_param.d;
    model_param.beta = other_param.beta;
    model_param.intercept = other_param.intercept;
  };

  void set_model_Xb(Eigen::ArrayXd &other_Xb) { Xb = other_Xb; };

  // coordinate descent
  virtual double coordinate_descent(RegFunction *regfun, int idx) = 0;

  // update intercept term
  virtual void intercept_update() = 0;

  // update gradient and other aux vars
  virtual void update_auxiliary() = 0;
  virtual void update_gradient(int idx){};

  // compute quadratic change of fvalue on the idx dimension
  virtual double get_local_change(double old, int idx) = 0;

  // unpenalized function value
  virtual double eval() = 0;

  virtual ~ObjFunction(){};
};

class GLMObjective : public ObjFunction {
 protected:
  Eigen::ArrayXd p, w, r;

  // wXX[j] = sum(w*X[j]*X[j])
  Eigen::ArrayXd wXX;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant
  double a, g;
  double sum_r;
  double sum_w;

 public:
  GLMObjective(const double *xmat, const double *y, int n, int d,
               bool include_intercept=false, bool usePypthon=false);

  double coordinate_descent(RegFunction *regfunc, int idx);

  void intercept_update();
  void update_gradient(int);

  double get_local_change(double old, int idx);
};

class LogisticObjective : public GLMObjective {
 public:
  LogisticObjective(const double *xmat, const double *y, int n, int d,
                    bool include_intercept=false, bool usePypthon=false);

  void update_auxiliary();

  double eval();
};

class PoissonObjective : public GLMObjective {
 public:
  PoissonObjective(const double *xmat, const double *y, int n, int d,
                   bool include_intercept=false, bool usePypthon=false);

  void update_auxiliary();

  double eval();
};


class SqrtMSEObjective : public ObjFunction {
 private:
  Eigen::ArrayXd r;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant
  double a, g;
  double L;  // sqrt(MSE)
  double sum_r;
  double sum_r2;

 public:
  SqrtMSEObjective(const double *xmat, const double *y, int n, int d,
                   bool include_intercept=false, bool usePypthon=false);

  double coordinate_descent(RegFunction *regfunc, int idx);

  void intercept_update();

  void update_auxiliary();
  void update_gradient(int idx);

  double get_local_change(double old, int idx);

  double eval();
};

class GaussianNaiveUpdateObjective final : public ObjFunction {
 private:
  Eigen::ArrayXd r, XX;

 public:
  GaussianNaiveUpdateObjective(const double *xmat, const double *y, int n,
                               int d, bool include_intercept=false, bool usePypthon=false);
  double coordinate_descent(RegFunction *regfunc, int idx);

  void intercept_update();
  void update_auxiliary();
  void update_gradient(int idx);

  double get_local_change(double old, int idx);

  double eval();
};

}  // namespace picasso

#endif  // PICASSO_OBJECTIVE_H
