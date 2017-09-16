#ifndef PICASSO_ACTNEWTON_H
#define PICASSO_ACTNEWTON_H

#include <string>
#include <cmath>

#include <picasso/actnewton.h>

namespace picasso {
namespace solver {
enum RegType {L1, SCAD, MCP};

// training parameters
class ActNewtonTrainParam {
public:
  /*! number of regularization parameters */
  unsigned num_lambda;

  /*! the last paramter on the regularization path */
  double target_lambda;

  /*! type of regularization terms */
  RegType reg_type;

  /*! gamma param for SCAD and MCP regularization */
  double reg_gamma;

  /*！ rounds of relaxation when solving SCAD and MCP penalty */
  unsigned num_relaxation_round;

  /*! precision of optimization */
  double prec;

  /*! max number of iteration for innner loop */
  int max_iter;

  /*! whether or not to add intercept term */
  bool include_intercept;

  std::vector<double> lambdas;

  ActNewtonTrainParam(); 

  void configure(const std::vector<std::pair<std::string, std::string> >& cfg);
  
  void set_lambdas(const double * lambda_path, int n);
  
  std::vector<double> get_lambda_path() const;
};

class ActNewtonSolver {
private:
  const ActNewtonTrainParam m_param;
  const ObjFunction * m_obj;

  std::vector<int> itercnt_path;
  std::vector<ModelParam> solution_path;

public:
  ActNewtonSolver(ObjFunction * obj, ActNewtonTrainParam param);

  void solve(ObjFunction* obj);
};

} // namespace solver
} // namespace picasso


#endif // PICASSO_ACTNEWTON_H