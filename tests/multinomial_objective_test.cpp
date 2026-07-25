#include <picasso/multinomial_objective.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

template <typename Derived>
double max_abs(const Eigen::MatrixBase<Derived> &value) {
  return value.cwiseAbs().maxCoeff();
}

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

double parameter_inner_product(const Eigen::MatrixXd &beta_a,
                               const Eigen::VectorXd &intercept_a,
                               const Eigen::MatrixXd &beta_b,
                               const Eigen::VectorXd &intercept_b) {
  return (beta_a.array() * beta_b.array()).sum() +
         intercept_a.dot(intercept_b);
}

Eigen::VectorXd pack_parameters(const Eigen::MatrixXd &beta,
                                const Eigen::VectorXd &intercept) {
  const int d = static_cast<int>(beta.rows());
  const int k = static_cast<int>(beta.cols());
  Eigen::VectorXd packed((d + 1) * k);
  for (int klass = 0; klass < k; ++klass) {
    for (int j = 0; j < d; ++j)
      packed[klass * (d + 1) + j] = beta(j, klass);
    packed[klass * (d + 1) + d] = intercept[klass];
  }
  return packed;
}

}  // namespace

int main() {
  bool ok = true;

  Eigen::MatrixXd extreme_logits(2, 3);
  extreme_logits << 1000.0, 1001.0, 999.0,
                    -1000.0, -999.0, -1002.0;
  Eigen::MatrixXd extreme_probabilities;
  Eigen::VectorXd extreme_log_sum_exp;
  picasso::MultinomialObjective::softmax_logsumexp(
      extreme_logits, &extreme_probabilities, &extreme_log_sum_exp);
  const double softmax_sum_error =
      max_abs(extreme_probabilities.rowwise().sum() -
              Eigen::VectorXd::Ones(extreme_logits.rows()));
  ok &= require(extreme_probabilities.allFinite() &&
                    extreme_log_sum_exp.allFinite(),
                "extreme finite logits must produce finite softmax outputs");
  ok &= require(extreme_probabilities.minCoeff() >= 0.0 &&
                    extreme_probabilities.maxCoeff() <= 1.0,
                "softmax probabilities must lie in [0, 1]");
  ok &= require(softmax_sum_error < 1e-14,
                "softmax rows must sum to one");

  Eigen::MatrixXd extreme_x = Eigen::MatrixXd::Zero(1, 1);
  Eigen::VectorXi extreme_label(1);
  extreme_label << 2;
  picasso::MultinomialObjective extreme_objective(extreme_x, extreme_label, 3);
  Eigen::MatrixXd separated_logits(1, 3);
  separated_logits << 1000.0, 0.0, -1000.0;
  const double separated_loss =
      extreme_objective.negative_log_likelihood_from_logits(separated_logits);
  ok &= require(std::isfinite(separated_loss) &&
                    std::fabs(separated_loss - 2000.0) < 1e-12,
                "NLL must not clip a very unlikely true class");

  Eigen::MatrixXd x(5, 2);
  x << -1.2, 0.4,
        0.3, -0.8,
        1.1, 0.7,
        -0.5, 1.3,
        0.9, -1.1;
  Eigen::VectorXi labels(5);
  labels << 0, 2, 1, 2, 0;
  picasso::MultinomialObjective objective(x, labels, 3);

  Eigen::MatrixXd base_logits(5, 3);
  base_logits << -1.0, 0.0, 1.0,
                 0.5, -0.5, 1.5,
                 2.0, 1.0, -1.0,
                 -2.0, 0.0, 3.0,
                 1.0, -2.0, 0.0;
  Eigen::MatrixXd base_probabilities;
  Eigen::MatrixXd shifted_probabilities;
  const double base_loss = objective.negative_log_likelihood_from_logits(
      base_logits, &base_probabilities);
  const Eigen::MatrixXd shifted_logits =
      (base_logits.array() + 1e15).matrix();
  const double shifted_loss = objective.negative_log_likelihood_from_logits(
      shifted_logits, &shifted_probabilities);
  const double shift_loss_error = std::fabs(base_loss - shifted_loss);
  const double shift_probability_error =
      max_abs(base_probabilities - shifted_probabilities);
  ok &= require(shift_loss_error < 1e-14,
                "NLL must be invariant to a large common class shift");
  ok &= require(shift_probability_error < 1e-14,
                "probabilities must be invariant to a common class shift");

  const double nonfinite_values[] = {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()};
  for (int value_index = 0; value_index < 2; ++value_index) {
    Eigen::MatrixXd invalid_logits = base_logits;
    invalid_logits(1, 1) = nonfinite_values[value_index];
    bool nll_rejected = false;
    bool softmax_rejected = false;
    try {
      objective.negative_log_likelihood_from_logits(invalid_logits);
    } catch (const std::invalid_argument &) {
      nll_rejected = true;
    }
    try {
      Eigen::MatrixXd ignored;
      picasso::MultinomialObjective::softmax_logsumexp(invalid_logits,
                                                       &ignored);
    } catch (const std::invalid_argument &) {
      softmax_rejected = true;
    }
    ok &= require(nll_rejected && softmax_rejected,
                  "NLL and softmax must reject nonfinite logits");
  }

  Eigen::MatrixXd beta(2, 3);
  beta << 0.2, -0.3, 0.1,
          -0.4, 0.5, -0.2;
  Eigen::VectorXd intercept(3);
  intercept << 0.1, -0.2, 0.3;
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(beta, intercept, &beta_gradient,
                            &intercept_gradient, &probabilities);
  const double gradient_class_sum_error = std::max(
      max_abs(beta_gradient.rowwise().sum()),
      std::fabs(intercept_gradient.sum()));
  ok &= require(gradient_class_sum_error < 1e-14,
                "smooth gradient must sum to zero across classes");

  const double epsilon = 1e-6;
  double gradient_error = 0.0;
  for (int j = 0; j < beta.rows(); ++j) {
    for (int klass = 0; klass < beta.cols(); ++klass) {
      Eigen::MatrixXd plus = beta;
      Eigen::MatrixXd minus = beta;
      plus(j, klass) += epsilon;
      minus(j, klass) -= epsilon;
      const double numeric =
          (objective.negative_log_likelihood(plus, intercept) -
           objective.negative_log_likelihood(minus, intercept)) /
          (2.0 * epsilon);
      gradient_error = std::max(
          gradient_error,
          std::fabs(numeric - beta_gradient(j, klass)));
    }
  }
  for (int klass = 0; klass < intercept.size(); ++klass) {
    Eigen::VectorXd plus = intercept;
    Eigen::VectorXd minus = intercept;
    plus[klass] += epsilon;
    minus[klass] -= epsilon;
    const double numeric =
        (objective.negative_log_likelihood(beta, plus) -
         objective.negative_log_likelihood(beta, minus)) /
        (2.0 * epsilon);
    gradient_error = std::max(
        gradient_error,
        std::fabs(numeric - intercept_gradient[klass]));
  }
  ok &= require(gradient_error < 1e-7,
                "smooth gradient must match central finite differences");

  // The active-set gradient primitive must reproduce the corresponding rows
  // of a full GEMM while leaving every inactive row untouched.  Use a
  // noncontiguous mask so this also catches accidental prefix/block updates.
  Eigen::MatrixXd masked_x(7, 5);
  masked_x << -1.2, 0.4, 0.7, -0.3, 1.1,
               0.3, -0.8, 0.2, 1.4, -0.5,
               1.1, 0.7, -1.3, 0.6, 0.9,
              -0.5, 1.3, 0.8, -1.0, 0.2,
               0.9, -1.1, 0.4, 0.5, -0.7,
              -0.2, 0.6, -0.9, 1.2, 0.3,
               0.8, -0.4, 1.0, -0.6, -1.2;
  Eigen::VectorXi masked_labels(7);
  masked_labels << 0, 2, 1, 2, 0, 1, 2;
  picasso::MultinomialObjective masked_objective(masked_x, masked_labels, 3);
  Eigen::MatrixXd masked_beta(5, 3);
  masked_beta << 0.2, -0.3, 0.1,
                -0.4, 0.5, -0.2,
                 0.7, -0.1, -0.4,
                -0.2, 0.6, 0.3,
                 0.1, -0.5, 0.8;
  Eigen::VectorXd masked_intercept(3);
  masked_intercept << 0.2, -0.4, 0.1;
  Eigen::MatrixXd masked_probabilities;
  Eigen::MatrixXd masked_full_gradient;
  Eigen::VectorXd masked_full_intercept_gradient;
  masked_objective.smooth_gradient(
      masked_beta, masked_intercept, &masked_full_gradient,
      &masked_full_intercept_gradient, &masked_probabilities);

  const double inactive_sentinel = 123.25;
  std::vector<unsigned char> subset_mask(5, 0);
  subset_mask[0] = 1;
  subset_mask[2] = 1;
  subset_mask[4] = 1;
  Eigen::MatrixXd masked_subset_gradient =
      Eigen::MatrixXd::Constant(5, 3, inactive_sentinel);
  Eigen::VectorXd masked_subset_intercept_gradient;
  masked_objective.smooth_gradient_from_probabilities_on_active_features(
      masked_probabilities, subset_mask, &masked_subset_gradient,
      &masked_subset_intercept_gradient);
  double active_gradient_error = 0.0;
  bool inactive_rows_untouched = true;
  for (int feature = 0; feature < 5; ++feature) {
    if (subset_mask[static_cast<std::size_t>(feature)] != 0) {
      active_gradient_error = std::max(
          active_gradient_error,
          max_abs(masked_subset_gradient.row(feature) -
                  masked_full_gradient.row(feature)));
    } else {
      inactive_rows_untouched =
          inactive_rows_untouched &&
          masked_subset_gradient.row(feature)
                  .array()
                  .operator==(inactive_sentinel)
                  .all();
    }
  }
  const double active_intercept_gradient_error = max_abs(
      masked_subset_intercept_gradient - masked_full_intercept_gradient);
  ok &= require(active_gradient_error < 2e-15 &&
                    active_intercept_gradient_error < 2e-15 &&
                    inactive_rows_untouched,
                "active gradient must match full rows and preserve inactive rows");

  std::vector<unsigned char> empty_mask(5, 0);
  Eigen::MatrixXd empty_gradient =
      Eigen::MatrixXd::Constant(5, 3, inactive_sentinel);
  Eigen::VectorXd empty_intercept_gradient;
  masked_objective.smooth_gradient_from_probabilities_on_active_features(
      masked_probabilities, empty_mask, &empty_gradient,
      &empty_intercept_gradient);
  ok &= require(
      empty_gradient.array().operator==(inactive_sentinel).all() &&
          max_abs(empty_intercept_gradient - masked_full_intercept_gradient) <
              2e-15,
      "empty active mask must update only the intercept gradient");

  std::vector<unsigned char> dense_mask(5, 1);
  Eigen::MatrixXd dense_active_gradient =
      Eigen::MatrixXd::Constant(5, 3, inactive_sentinel);
  Eigen::VectorXd dense_active_intercept_gradient;
  masked_objective.smooth_gradient_from_probabilities_on_active_features(
      masked_probabilities, dense_mask, &dense_active_gradient,
      &dense_active_intercept_gradient);
  ok &= require(max_abs(dense_active_gradient - masked_full_gradient) < 2e-15 &&
                    max_abs(dense_active_intercept_gradient -
                            masked_full_intercept_gradient) < 2e-15,
                "dense active mask must match the full gradient");

  bool bad_mask_rejected = false;
  bool bad_output_rejected = false;
  try {
    std::vector<unsigned char> bad_mask(4, 1);
    masked_objective.smooth_gradient_from_probabilities_on_active_features(
        masked_probabilities, bad_mask, &dense_active_gradient,
        &dense_active_intercept_gradient);
  } catch (const std::invalid_argument &) {
    bad_mask_rejected = true;
  }
  try {
    Eigen::MatrixXd bad_output(4, 3);
    masked_objective.smooth_gradient_from_probabilities_on_active_features(
        masked_probabilities, dense_mask, &bad_output,
        &dense_active_intercept_gradient);
  } catch (const std::invalid_argument &) {
    bad_output_rejected = true;
  }
  ok &= require(bad_mask_rejected && bad_output_rejected,
                "active gradient must reject invalid mask and output shapes");

  Eigen::MatrixXd direction_a(2, 3);
  direction_a << 0.7, -0.2, 0.4,
                 -0.1, 0.6, -0.5;
  Eigen::VectorXd intercept_direction_a(3);
  intercept_direction_a << 0.3, -0.4, 0.2;
  Eigen::MatrixXd hessian_beta_a;
  Eigen::VectorXd hessian_intercept_a;
  objective.hessian_vector_product(
      probabilities, direction_a, intercept_direction_a, &hessian_beta_a,
      &hessian_intercept_a);

  Eigen::MatrixXd gradient_plus;
  Eigen::MatrixXd gradient_minus;
  Eigen::VectorXd intercept_gradient_plus;
  Eigen::VectorXd intercept_gradient_minus;
  objective.smooth_gradient(
      beta + epsilon * direction_a,
      intercept + epsilon * intercept_direction_a, &gradient_plus,
      &intercept_gradient_plus);
  objective.smooth_gradient(
      beta - epsilon * direction_a,
      intercept - epsilon * intercept_direction_a, &gradient_minus,
      &intercept_gradient_minus);
  const Eigen::MatrixXd numeric_beta_hvp =
      (gradient_plus - gradient_minus) / (2.0 * epsilon);
  const Eigen::VectorXd numeric_intercept_hvp =
      (intercept_gradient_plus - intercept_gradient_minus) /
      (2.0 * epsilon);
  const double hvp_error =
      std::max(max_abs(numeric_beta_hvp - hessian_beta_a),
               max_abs(numeric_intercept_hvp - hessian_intercept_a));
  ok &= require(hvp_error < 1e-7,
                "Hessian-vector product must match gradient differences");

  const int n = static_cast<int>(x.rows());
  const int d = static_cast<int>(x.cols());
  const int k = static_cast<int>(probabilities.cols());
  const int block_size = d + 1;
  Eigen::MatrixXd dense_hessian =
      Eigen::MatrixXd::Zero(block_size * k, block_size * k);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd augmented_x(block_size);
    augmented_x.head(d) = x.row(i).transpose();
    augmented_x[d] = 1.0;
    const Eigen::VectorXd p = probabilities.row(i).transpose();
    Eigen::MatrixXd class_hessian = p.asDiagonal();
    class_hessian.noalias() -= p * p.transpose();
    for (int left_class = 0; left_class < k; ++left_class) {
      for (int right_class = 0; right_class < k; ++right_class) {
        dense_hessian.block(left_class * block_size,
                            right_class * block_size, block_size,
                            block_size).noalias() +=
            class_hessian(left_class, right_class) *
            (augmented_x * augmented_x.transpose()) /
            static_cast<double>(n);
      }
    }
  }
  const Eigen::VectorXd packed_direction_a =
      pack_parameters(direction_a, intercept_direction_a);
  const Eigen::VectorXd packed_hvp_a =
      pack_parameters(hessian_beta_a, hessian_intercept_a);
  const double dense_hessian_error =
      max_abs(dense_hessian * packed_direction_a - packed_hvp_a);
  ok &= require(dense_hessian_error < 1e-14,
                "matrix-free HVP must match the explicit dense Hessian");

  const double psd_quadratic = parameter_inner_product(
      direction_a, intercept_direction_a, hessian_beta_a,
      hessian_intercept_a);
  ok &= require(psd_quadratic >= -1e-14,
                "multinomial Hessian must be positive semidefinite");

  Eigen::MatrixXd direction_b(2, 3);
  direction_b << -0.3, 0.9, 0.2,
                  0.8, -0.7, 0.1;
  Eigen::VectorXd intercept_direction_b(3);
  intercept_direction_b << -0.2, 0.5, 0.4;
  Eigen::MatrixXd hessian_beta_b;
  Eigen::VectorXd hessian_intercept_b;
  objective.hessian_vector_product(
      probabilities, direction_b, intercept_direction_b, &hessian_beta_b,
      &hessian_intercept_b);
  const double adjoint_left = parameter_inner_product(
      direction_a, intercept_direction_a, hessian_beta_b,
      hessian_intercept_b);
  const double adjoint_right = parameter_inner_product(
      direction_b, intercept_direction_b, hessian_beta_a,
      hessian_intercept_a);
  const double self_adjoint_error = std::fabs(adjoint_left - adjoint_right);
  ok &= require(self_adjoint_error < 1e-14,
                "Hessian-vector operator must be self-adjoint");

  Eigen::VectorXd common_feature_direction(2);
  common_feature_direction << 0.6, -0.9;
  Eigen::MatrixXd class_shift_direction(2, 3);
  for (int klass = 0; klass < class_shift_direction.cols(); ++klass)
    class_shift_direction.col(klass) = common_feature_direction;
  const Eigen::VectorXd class_shift_intercept =
      Eigen::VectorXd::Constant(3, 0.7);
  Eigen::MatrixXd class_shift_beta_hvp;
  Eigen::VectorXd class_shift_intercept_hvp;
  objective.hessian_vector_product(
      probabilities, class_shift_direction, class_shift_intercept,
      &class_shift_beta_hvp, &class_shift_intercept_hvp);
  const double class_shift_zero_error =
      std::max(max_abs(class_shift_beta_hvp),
               max_abs(class_shift_intercept_hvp));
  ok &= require(class_shift_zero_error < 1e-14,
                "common class-shift directions must lie in the Hessian null space");

  std::cout << "gradient_max_abs_error=" << gradient_error << "\n"
            << "hvp_max_abs_error=" << hvp_error << "\n"
            << "dense_hessian_max_abs_error=" << dense_hessian_error << "\n"
            << "psd_quadratic=" << psd_quadratic << "\n"
            << "self_adjoint_error=" << self_adjoint_error << "\n"
            << "class_shift_zero_error=" << class_shift_zero_error << "\n"
            << "common_shift_loss_error=" << shift_loss_error << "\n"
            << "gradient_class_sum_error=" << gradient_class_sum_error << "\n"
            << "active_gradient_max_abs_error=" << active_gradient_error
            << "\n"
            << "softmax_row_sum_error=" << softmax_sum_error << "\n";
  return ok ? 0 : 1;
}
