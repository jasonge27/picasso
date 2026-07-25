#include <picasso/multinomial_objective.hpp>

#include "../internal/multinomial_problem_view.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace picasso {

namespace {

template <typename Design>
void multinomial_view_linear_predictor_with_design(
    const Design &design, const Eigen::MatrixXd &beta,
    const Eigen::VectorXd &intercept, Eigen::MatrixXd *logits) {
  logits->noalias() = design * beta;
  logits->rowwise() += intercept.transpose();
}

template <typename Design>
void multinomial_view_smooth_gradient_with_design(
    const Design &design, const Eigen::MatrixXd &residual, int sample_num,
    Eigen::MatrixXd *beta_gradient,
    Eigen::VectorXd *intercept_gradient) {
  const double inverse_n = 1.0 / static_cast<double>(sample_num);
  beta_gradient->noalias() = inverse_n * design.transpose() * residual;
  *intercept_gradient =
      inverse_n * residual.colwise().sum().transpose();
}

template <typename Design>
void multinomial_view_active_gradient_with_design(
    const Design &design, const Eigen::MatrixXd &residual,
    const std::vector<unsigned char> &active_features, int sample_num,
    int feature_num, int class_num, Eigen::MatrixXd *beta_gradient,
    Eigen::VectorXd *intercept_gradient) {
  const double inverse_n = 1.0 / static_cast<double>(sample_num);
  *intercept_gradient =
      inverse_n * residual.colwise().sum().transpose();

  std::vector<int> active_indices;
  active_indices.reserve(active_features.size());
  for (int feature = 0; feature < feature_num; ++feature) {
    if (active_features[static_cast<std::size_t>(feature)] != 0)
      active_indices.push_back(feature);
  }
  const int active_count = static_cast<int>(active_indices.size());
  if (active_count == 0) return;

  if (active_count <= 4) {
    for (int index = 0; index < active_count; ++index) {
      const int feature = active_indices[static_cast<std::size_t>(index)];
      beta_gradient->row(feature).noalias() =
          inverse_n * design.col(feature).transpose() * residual;
    }
    return;
  }

  Eigen::MatrixXd active_design(sample_num, active_count);
  for (int index = 0; index < active_count; ++index)
    active_design.col(index) =
        design.col(active_indices[static_cast<std::size_t>(index)]);
  Eigen::MatrixXd active_gradient(active_count, class_num);
  active_gradient.noalias() =
      inverse_n * active_design.transpose() * residual;
  for (int index = 0; index < active_count; ++index) {
    beta_gradient->row(
        active_indices[static_cast<std::size_t>(index)]) =
        active_gradient.row(index);
  }
}

template <typename Design>
void multinomial_view_hessian_vector_product_with_design(
    const Design &design, const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &beta_direction,
    const Eigen::VectorXd &intercept_direction, int sample_num,
    Eigen::MatrixXd *beta_hessian_vector,
    Eigen::VectorXd *intercept_hessian_vector) {
  Eigen::MatrixXd linear_direction;
  linear_direction.noalias() = design * beta_direction;
  linear_direction.rowwise() += intercept_direction.transpose();

  Eigen::MatrixXd weighted_direction;
  detail::MultinomialProblemView::apply_probability_hessian(
      probabilities, linear_direction, &weighted_direction);

  const double inverse_n = 1.0 / static_cast<double>(sample_num);
  beta_hessian_vector->noalias() =
      inverse_n * design.transpose() * weighted_direction;
  *intercept_hessian_vector =
      inverse_n * weighted_direction.colwise().sum().transpose();
}

}  // namespace

namespace detail {

MultinomialProblemView::MultinomialProblemView(
    const Eigen::MatrixXd &design, const Eigen::VectorXi &labels,
    int num_classes, const void *identity)
    : m_design_data(design.data()),
      m_labels(&labels),
      m_n(static_cast<int>(design.rows())),
      m_d(static_cast<int>(design.cols())),
      m_k(num_classes),
      m_identity(identity) {
  validate_problem_shape();
}

MultinomialProblemView::MultinomialProblemView(
    const double *column_major_design, int sample_num, int feature_num,
    const Eigen::VectorXi &labels, int num_classes, const void *identity)
    : m_design_data(column_major_design),
      m_labels(&labels),
      m_n(sample_num),
      m_d(feature_num),
      m_k(num_classes),
      m_identity(identity) {
  validate_problem_shape();
}

void MultinomialProblemView::validate_problem_shape() const {
  if (m_n <= 0 || m_d <= 0 || m_k < 2)
    throw std::invalid_argument("multinomial dimensions must be positive");
  if (m_labels == 0 || m_labels->size() != m_n)
    throw std::invalid_argument(
        "multinomial label length does not match X");
  if (m_identity == 0)
    throw std::invalid_argument("multinomial problem identity is null");
  if (m_design_data == 0)
    throw std::invalid_argument("multinomial design pointer is null");
  if (!design_pointer_is_aligned(m_design_data))
    throw std::invalid_argument(
        "multinomial design pointer does not satisfy Eigen alignment");
}

void MultinomialProblemView::validate_parameters(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept) const {
  if (beta.rows() != m_d || beta.cols() != m_k)
    throw std::invalid_argument("multinomial beta must have shape d-by-K");
  if (intercept.size() != m_k)
    throw std::invalid_argument("multinomial intercept must have length K");
}

void MultinomialProblemView::validate_probabilities(
    const Eigen::MatrixXd &probabilities) const {
  if (probabilities.rows() != m_n || probabilities.cols() != m_k)
    throw std::invalid_argument(
        "multinomial probabilities must have shape n-by-K");
}

void MultinomialProblemView::linear_predictor(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Eigen::MatrixXd *logits) const {
  if (logits == 0)
    throw std::invalid_argument("multinomial logits output is null");
  validate_parameters(beta, intercept);
  const ConstDesignMap design = mapped_design_matrix();
  multinomial_view_linear_predictor_with_design(
      design, beta, intercept, logits);
}

void MultinomialProblemView::softmax_logsumexp(
    const Eigen::MatrixXd &logits, Eigen::MatrixXd *probabilities,
    Eigen::VectorXd *log_sum_exp) {
  if (probabilities == 0)
    throw std::invalid_argument("multinomial probability output is null");
  if (logits.rows() <= 0 || logits.cols() <= 0)
    throw std::invalid_argument("multinomial logits must be nonempty");

  probabilities->resize(logits.rows(), logits.cols());
  if (log_sum_exp != 0) log_sum_exp->resize(logits.rows());
  for (Eigen::Index i = 0; i < logits.rows(); ++i) {
    const double row_max = logits.row(i).maxCoeff();
    double exponential_sum = 0.0;
    for (Eigen::Index k = 0; k < logits.cols(); ++k) {
      const double value = std::exp(logits(i, k) - row_max);
      (*probabilities)(i, k) = value;
      exponential_sum += value;
    }
    if (!std::isfinite(row_max) || !std::isfinite(exponential_sum) ||
        exponential_sum <= 0.0)
      throw std::invalid_argument("multinomial logits must be finite");
    probabilities->row(i) /= exponential_sum;
    if (log_sum_exp != 0)
      (*log_sum_exp)[i] = row_max + std::log(exponential_sum);
  }
}

double MultinomialProblemView::negative_log_likelihood(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Eigen::MatrixXd *probabilities) const {
  Eigen::MatrixXd logits;
  linear_predictor(beta, intercept, &logits);
  return negative_log_likelihood_from_logits(logits, probabilities);
}

double MultinomialProblemView::negative_log_likelihood_from_logits(
    const Eigen::MatrixXd &logits, Eigen::MatrixXd *probabilities) const {
  if (logits.rows() != m_n || logits.cols() != m_k)
    throw std::invalid_argument(
        "multinomial logits must have shape n-by-K");
  if (probabilities != 0) probabilities->resize(m_n, m_k);
  double loss = 0.0;
  for (int i = 0; i < m_n; ++i) {
    const double row_max = logits.row(i).maxCoeff();
    double exponential_sum = 0.0;
    for (int k = 0; k < m_k; ++k) {
      const double value = std::exp(logits(i, k) - row_max);
      if (probabilities != 0) (*probabilities)(i, k) = value;
      exponential_sum += value;
    }
    if (!std::isfinite(row_max) || !std::isfinite(exponential_sum) ||
        exponential_sum <= 0.0)
      throw std::invalid_argument("multinomial logits must be finite");
    if (probabilities != 0) probabilities->row(i) /= exponential_sum;
    loss += std::log(exponential_sum) +
            (row_max - logits(i, (*m_labels)[i]));
  }
  return loss / static_cast<double>(m_n);
}

void MultinomialProblemView::smooth_gradient(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Eigen::MatrixXd *beta_gradient, Eigen::VectorXd *intercept_gradient,
    Eigen::MatrixXd *probabilities) const {
  if (beta_gradient == 0 || intercept_gradient == 0)
    throw std::invalid_argument("multinomial gradient output is null");
  Eigen::MatrixXd logits;
  Eigen::MatrixXd local_probabilities;
  Eigen::MatrixXd *probability_output =
      probabilities == 0 ? &local_probabilities : probabilities;
  linear_predictor(beta, intercept, &logits);
  softmax_logsumexp(logits, probability_output);
  smooth_gradient_from_probabilities(*probability_output, beta_gradient,
                                     intercept_gradient);
}

void MultinomialProblemView::smooth_gradient_from_probabilities(
    const Eigen::MatrixXd &probabilities, Eigen::MatrixXd *beta_gradient,
    Eigen::VectorXd *intercept_gradient) const {
  if (beta_gradient == 0 || intercept_gradient == 0)
    throw std::invalid_argument("multinomial gradient output is null");
  validate_probabilities(probabilities);
  Eigen::MatrixXd residual = probabilities;
  for (int i = 0; i < m_n; ++i) residual(i, (*m_labels)[i]) -= 1.0;
  const ConstDesignMap design = mapped_design_matrix();
  multinomial_view_smooth_gradient_with_design(
      design, residual, m_n, beta_gradient, intercept_gradient);
}

void MultinomialProblemView::
    smooth_gradient_from_probabilities_on_active_features(
        const Eigen::MatrixXd &probabilities,
        const std::vector<unsigned char> &active_features,
        Eigen::MatrixXd *beta_gradient,
        Eigen::VectorXd *intercept_gradient) const {
  if (beta_gradient == 0 || intercept_gradient == 0)
    throw std::invalid_argument("multinomial gradient output is null");
  if (active_features.size() != static_cast<std::size_t>(m_d))
    throw std::invalid_argument(
        "multinomial active-feature mask has the wrong length");
  if (beta_gradient->rows() != m_d || beta_gradient->cols() != m_k)
    throw std::invalid_argument(
        "multinomial active beta-gradient output has the wrong shape");
  validate_probabilities(probabilities);
  Eigen::MatrixXd residual = probabilities;
  for (int i = 0; i < m_n; ++i) residual(i, (*m_labels)[i]) -= 1.0;
  const ConstDesignMap design = mapped_design_matrix();
  multinomial_view_active_gradient_with_design(
      design, residual, active_features, m_n, m_d, m_k, beta_gradient,
      intercept_gradient);
}

void MultinomialProblemView::apply_probability_hessian(
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &linear_direction,
    Eigen::MatrixXd *weighted_direction) {
  if (weighted_direction == 0)
    throw std::invalid_argument("multinomial Hessian output is null");
  if (probabilities.rows() != linear_direction.rows() ||
      probabilities.cols() != linear_direction.cols())
    throw std::invalid_argument(
        "multinomial probability and direction shapes differ");
  weighted_direction->resize(probabilities.rows(), probabilities.cols());
  for (Eigen::Index i = 0; i < probabilities.rows(); ++i) {
    double probability_dot_direction = 0.0;
    for (Eigen::Index k = 0; k < probabilities.cols(); ++k)
      probability_dot_direction +=
          probabilities(i, k) * linear_direction(i, k);
    for (Eigen::Index k = 0; k < probabilities.cols(); ++k) {
      (*weighted_direction)(i, k) =
          probabilities(i, k) *
          (linear_direction(i, k) - probability_dot_direction);
    }
  }
}

void MultinomialProblemView::hessian_vector_product(
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &beta_direction,
    const Eigen::VectorXd &intercept_direction,
    Eigen::MatrixXd *beta_hessian_vector,
    Eigen::VectorXd *intercept_hessian_vector) const {
  if (beta_hessian_vector == 0 || intercept_hessian_vector == 0)
    throw std::invalid_argument(
        "multinomial Hessian-vector output is null");
  validate_probabilities(probabilities);
  validate_parameters(beta_direction, intercept_direction);
  const ConstDesignMap design = mapped_design_matrix();
  multinomial_view_hessian_vector_product_with_design(
      design, probabilities, beta_direction, intercept_direction, m_n,
      beta_hessian_vector, intercept_hessian_vector);
}

}  // namespace detail

namespace {

detail::MultinomialProblemView objective_problem_view(
    const MultinomialObjective &objective) {
  return detail::MultinomialProblemView(
      objective.design_matrix(), objective.labels(), objective.class_num(),
      &objective);
}

}  // namespace

MultinomialObjective::MultinomialObjective(Eigen::MatrixXd x,
                                           Eigen::VectorXi labels,
                                           int num_classes)
    : m_n(static_cast<int>(x.rows())),
      m_d(static_cast<int>(x.cols())),
      m_k(num_classes),
      m_x(std::move(x)),
      m_labels(std::move(labels)) {
  if (m_n <= 0 || m_d <= 0 || m_k < 2)
    throw std::invalid_argument("multinomial dimensions must be positive");
  if (m_labels.size() != m_n)
    throw std::invalid_argument("multinomial label length does not match X");
  for (int i = 0; i < m_n; ++i) {
    if (m_labels[i] < 0 || m_labels[i] >= m_k)
      throw std::invalid_argument("multinomial label is out of range");
  }
}

void MultinomialObjective::validate_parameters(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept) const {
  objective_problem_view(*this).validate_parameters(beta, intercept);
}

void MultinomialObjective::validate_probabilities(
    const Eigen::MatrixXd &probabilities) const {
  objective_problem_view(*this).validate_probabilities(probabilities);
}

void MultinomialObjective::linear_predictor(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Eigen::MatrixXd *logits) const {
  objective_problem_view(*this).linear_predictor(beta, intercept, logits);
}

void MultinomialObjective::softmax_logsumexp(
    const Eigen::MatrixXd &logits, Eigen::MatrixXd *probabilities,
    Eigen::VectorXd *log_sum_exp) {
  detail::MultinomialProblemView::softmax_logsumexp(
      logits, probabilities, log_sum_exp);
}

double MultinomialObjective::negative_log_likelihood(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Eigen::MatrixXd *probabilities) const {
  return objective_problem_view(*this).negative_log_likelihood(
      beta, intercept, probabilities);
}

double MultinomialObjective::negative_log_likelihood_from_logits(
    const Eigen::MatrixXd &logits, Eigen::MatrixXd *probabilities) const {
  return objective_problem_view(*this).negative_log_likelihood_from_logits(
      logits, probabilities);
}

void MultinomialObjective::smooth_gradient(
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Eigen::MatrixXd *beta_gradient, Eigen::VectorXd *intercept_gradient,
    Eigen::MatrixXd *probabilities) const {
  objective_problem_view(*this).smooth_gradient(
      beta, intercept, beta_gradient, intercept_gradient, probabilities);
}

void MultinomialObjective::smooth_gradient_from_probabilities(
    const Eigen::MatrixXd &probabilities, Eigen::MatrixXd *beta_gradient,
    Eigen::VectorXd *intercept_gradient) const {
  objective_problem_view(*this).smooth_gradient_from_probabilities(
      probabilities, beta_gradient, intercept_gradient);
}

void MultinomialObjective::
    smooth_gradient_from_probabilities_on_active_features(
        const Eigen::MatrixXd &probabilities,
        const std::vector<unsigned char> &active_features,
        Eigen::MatrixXd *beta_gradient,
        Eigen::VectorXd *intercept_gradient) const {
  objective_problem_view(*this)
      .smooth_gradient_from_probabilities_on_active_features(
          probabilities, active_features, beta_gradient,
          intercept_gradient);
}

void MultinomialObjective::apply_probability_hessian(
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &linear_direction,
    Eigen::MatrixXd *weighted_direction) {
  detail::MultinomialProblemView::apply_probability_hessian(
      probabilities, linear_direction, weighted_direction);
}

void MultinomialObjective::hessian_vector_product(
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &beta_direction,
    const Eigen::VectorXd &intercept_direction,
    Eigen::MatrixXd *beta_hessian_vector,
    Eigen::VectorXd *intercept_hessian_vector) const {
  objective_problem_view(*this).hessian_vector_product(
      probabilities, beta_direction, intercept_direction,
      beta_hessian_vector, intercept_hessian_vector);
}

}  // namespace picasso
