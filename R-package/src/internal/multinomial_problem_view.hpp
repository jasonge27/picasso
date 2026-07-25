#ifndef PICASSO_INTERNAL_MULTINOMIAL_PROBLEM_VIEW_HPP
#define PICASSO_INTERNAL_MULTINOMIAL_PROBLEM_VIEW_HPP

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace picasso {
namespace detail {

#if defined(__GNUC__) || defined(__clang__)
#  define PICASSO_MULTINOMIAL_PRIVATE __attribute__((visibility("hidden")))
#else
#  define PICASSO_MULTINOMIAL_PRIVATE
#endif

// Private, synchronous view of one immutable multinomial problem.  Public
// MultinomialObjective instances continue to own MatrixXd/VectorXi values and
// retain their historical layout and copy semantics.  The synchronous R C API
// may borrow an aligned column-major buffer; row-major Python input and
// misaligned C buffers are copied into an owning MatrixXd first.
class PICASSO_MULTINOMIAL_PRIVATE MultinomialProblemView {
 public:
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
  enum { kDesignMapAlignment = Eigen::AlignedMax };
#else
  // Eigen 3.2 exposes only the historical 16-byte Aligned map option.
  enum { kDesignMapAlignment = Eigen::Aligned };
#endif
  typedef Eigen::Map<
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor>,
      kDesignMapAlignment>
      ConstDesignMap;

  MultinomialProblemView(const Eigen::MatrixXd &design,
                         const Eigen::VectorXi &labels, int num_classes,
                         const void *identity);

  MultinomialProblemView(const double *column_major_design, int sample_num,
                         int feature_num, const Eigen::VectorXi &labels,
                         int num_classes, const void *identity);

  int sample_num() const { return m_n; }
  int feature_num() const { return m_d; }
  int class_num() const { return m_k; }
  const Eigen::VectorXi &labels() const { return *m_labels; }
  const void *identity() const { return m_identity; }

  static std::size_t design_alignment_bytes() {
#if EIGEN_VERSION_AT_LEAST(3, 3, 0) && EIGEN_MAX_ALIGN_BYTES > 0
    return static_cast<std::size_t>(EIGEN_MAX_ALIGN_BYTES);
#elif EIGEN_VERSION_AT_LEAST(3, 3, 0)
    return sizeof(double);
#else
    return 16;
#endif
  }

  static bool design_pointer_is_aligned(const double *design) {
    if (design == 0) return false;
    const std::size_t alignment = design_alignment_bytes();
    if (alignment <= sizeof(double)) return true;
    return reinterpret_cast<std::uintptr_t>(design) %
               static_cast<std::uintptr_t>(alignment) ==
           0;
  }

  ConstDesignMap mapped_design_matrix() const {
    return ConstDesignMap(m_design_data, m_n, m_d);
  }

  void validate_parameters(const Eigen::MatrixXd &beta,
                           const Eigen::VectorXd &intercept) const;

  void validate_probabilities(
      const Eigen::MatrixXd &probabilities) const;

  void linear_predictor(const Eigen::MatrixXd &beta,
                        const Eigen::VectorXd &intercept,
                        Eigen::MatrixXd *logits) const;

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

  void smooth_gradient_from_probabilities_on_active_features(
      const Eigen::MatrixXd &probabilities,
      const std::vector<unsigned char> &active_features,
      Eigen::MatrixXd *beta_gradient,
      Eigen::VectorXd *intercept_gradient) const;

  static void apply_probability_hessian(
      const Eigen::MatrixXd &probabilities,
      const Eigen::MatrixXd &linear_direction,
      Eigen::MatrixXd *weighted_direction);

  void hessian_vector_product(
      const Eigen::MatrixXd &probabilities,
      const Eigen::MatrixXd &beta_direction,
      const Eigen::VectorXd &intercept_direction,
      Eigen::MatrixXd *beta_hessian_vector,
      Eigen::VectorXd *intercept_hessian_vector) const;

 private:
  void validate_problem_shape() const;

  const double *m_design_data;
  const Eigen::VectorXi *m_labels;
  int m_n;
  int m_d;
  int m_k;
  const void *m_identity;
};

#undef PICASSO_MULTINOMIAL_PRIVATE

}  // namespace detail
}  // namespace picasso

#endif  // PICASSO_INTERNAL_MULTINOMIAL_PROBLEM_VIEW_HPP
