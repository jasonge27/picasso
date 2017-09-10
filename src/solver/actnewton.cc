#include <picasso/actnewton.h>
#include <dmlc/parameter.h>

namespace picasso {
namespace solver {
DMLC_REGISTRY_FILE_TAG(solver);

// model parameters
struct SpLinearModelParam: public dmlc::Parameter<SpLinearModelParam> {
  // feature dimension
  unsigned dim_feature;

  SpLinearModelParam() {
    std::memset(this, 0, sizeof(SpLinearModelParam));
  }

  DMLC_DECLARE_PARAMETER(SpLinearModelParam) {
    DMLC_DECLARE_FIELD(dim_feature).set_lower_bound(0)
      .describe("Dimension of features.");
  }
};

enum RegType {L1, SCAD, MCP};

// training parameters
struct ActNewtonTrainParam: public dmlc::Parameter<ActNewtonTrainParam> {
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

  DMLC_DECLARE_PARAMETER(ActNewtonTrainParam) {
    DMLC_DECLARE_FIELD(num_lambda).set_lower_bound(0).set_default(100)
      .describe("number of parameters on the regularization path.");
    DMLC_DECLARE_FIELD(target_lambda).set_lower_bound(0).set_default(1e-3)
      .describe("the smallest parameter on the regularization path");
    DMLC_DECLARE_FIELD(reg_type).set_default(L1)
      .describe("the type of regularization");
    DMLC_DECLARE_FIELD(num_relaxation_round).set_lower_bound(0).set_default(3)
      .describe("number of rounds for multistage convex relaxation.");
    DMLC_DECLARE_FIELD(prec).set_lower_bound(0).set_default(1e-4)
      .describe("precision of optimization solution.")
  }
}

class ActNewtonSolver {
private:
  ActNewtonTrainParam m_param;
  const ObjFunction * m_obj;
public:
  ActNewtonSolver(ObjFunction * obj):m_obj(obj) {}

  void configure(const std::vector<std::pair<std::string, std::string> >& cfg){
    m_param.init(cfg);
  }

  void solve(ObjFunction* obj){
    std::vector<double> & gr = obj->grad();
    const std::vector<double> & lambda_path = param.lambda_path();
    
    model_before_relaxation = obj->model_params();
    obj->init_active_set();
    for (auto lambda_iter = lambda_path.begin(); 
          lambda_iter != lambda_path.end(); lambda_iter++){
      obj->set_model_params(model_before_relaxation);  

      if (lambda_iter != lambda_path.begin())
        obj->reset_active_set();

      int relaxation_count = 0;
      while (relaxation_count < param.num_relaxation_round) {
        relaxation_count++;
        
      }
    }
  }
}

}
}