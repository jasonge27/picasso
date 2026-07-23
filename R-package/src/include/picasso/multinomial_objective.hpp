#ifndef PICASSO_MULTINOMIAL_OBJECTIVE_HPP
#define PICASSO_MULTINOMIAL_OBJECTIVE_HPP

#include <Eigen/Dense>

#include <vector>

namespace picasso {

// Smooth multinomial negative log-likelihood and its exact derivatives.
// Coefficients use d-by-K storage and observations use n-by-K storage.  All
// objective and derivative values are averages over observations.
class MultinomialObjective {
 public:
  MultinomialObjective(Eigen::MatrixXd x,
                       Eigen::VectorXi labels,
                       int num_classes);

  // Solvers borrow this objective and may key immutable path caches by its
  // address.  Copy construction creates a distinct objective, but replacing
  // the data of an already-borrowed object would invalidate those caches.
  MultinomialObjective(const MultinomialObjective &) = default;
  MultinomialObjective &operator=(const MultinomialObjective &) = delete;
  MultinomialObjective &operator=(MultinomialObjective &&) = delete;

  int sample_num() const { return m_n; }
  int feature_num() const { return m_d; }
  int class_num() const { return m_k; }
  const Eigen::MatrixXd &design_matrix() const { return m_x; }
  const Eigen::VectorXi &labels() const { return m_labels; }

  void linear_predictor(const Eigen::MatrixXd &beta,
                        const Eigen::VectorXd &intercept,
                        Eigen::MatrixXd *logits) const;

  // Computes a row-wise stable softmax.  log_sum_exp may be null.
  static void softmax_logsumexp(const Eigen::MatrixXd &logits,
                                Eigen::MatrixXd *probabilities,
                                Eigen::VectorXd *log_sum_exp = 0);

  double negative_log_likelihood(
      const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
      Eigen::MatrixXd *probabilities = 0) const;

  double negative_log_likelihood_from_logits(
      const Eigen::MatrixXd &logits,
      Eigen::MatrixXd *probabilities = 0) const;

  void smooth_gradient(const Eigen::MatrixXd &beta,
                       const Eigen::VectorXd &intercept,
                       Eigen::MatrixXd *beta_gradient,
                       Eigen::VectorXd *intercept_gradient,
                       Eigen::MatrixXd *probabilities = 0) const;

  void smooth_gradient_from_probabilities(
      const Eigen::MatrixXd &probabilities,
      Eigen::MatrixXd *beta_gradient,
      Eigen::VectorXd *intercept_gradient) const;

  // Refreshes the intercept gradient and only the selected feature rows.
  // beta_gradient must already have shape d-by-K; unselected rows are left
  // untouched so an active-set solver can retain its last full-gradient
  // certificate without allocating another dense d-by-K workspace.
  void smooth_gradient_from_probabilities_on_active_features(
      const Eigen::MatrixXd &probabilities,
      const std::vector<unsigned char> &active_features,
      Eigen::MatrixXd *beta_gradient,
      Eigen::VectorXd *intercept_gradient) const;

  // Applies each row's exact categorical covariance
  // diag(p_i) - p_i p_i^T without materializing a K-by-K matrix.
  static void apply_probability_hessian(
      const Eigen::MatrixXd &probabilities,
      const Eigen::MatrixXd &linear_direction,
      Eigen::MatrixXd *weighted_direction);

  // Applies the full parameter Hessian at a fixed probability matrix.  This
  // is the coupled IRLS Hessian, including the cross-class -p_i p_i^T term.
  void hessian_vector_product(
      const Eigen::MatrixXd &probabilities,
      const Eigen::MatrixXd &beta_direction,
      const Eigen::VectorXd &intercept_direction,
      Eigen::MatrixXd *beta_hessian_vector,
      Eigen::VectorXd *intercept_hessian_vector) const;

 private:
  void validate_parameters(const Eigen::MatrixXd &beta,
                           const Eigen::VectorXd &intercept) const;
  void validate_probabilities(const Eigen::MatrixXd &probabilities) const;

  int m_n;
  int m_d;
  int m_k;
  Eigen::MatrixXd m_x;
  Eigen::VectorXi m_labels;
};

}  // namespace picasso

#endif  // PICASSO_MULTINOMIAL_OBJECTIVE_HPP
