#include <picasso/actnewton.h>
#include <dmlc/parameter.h>

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

  std::vector<double> lambdas;

  ActiveNewtonTrainParam() {
    num_lambda = 100;
    target_lambda = 1e-6;
    reg_type = L1;
    reg_gamma = 3.0;
    num_relaxation_round = 3;
    prec = 1e-4;
    lambdas.clear();
  }

  void configure(const std::vector<std::pair<std::string, std::string> >& cfg){
    for (auto iter = cfg.begin(); iter != cfg.end(); iter++){
      if (iter.first == "nlambda")
        num_lambda = stoi(iter.second)
      else if (iter.first == "target_lambda")
        target_lambda = stof(iter.second)
      else if (iter.first == "reg_type"){
        if (iter.second == "L1")
          reg_type = L1;
        else if (iter.second == "SCAD")
          reg_type = SCAD;
        else if (iter.second == "MCP")
          reg_type = MCP;
        else 
          // throw exception
      }
      else 
        // TODO
    }
  }

  void set_lambdas(const double * lambda_path, int n) {
    lambdas.resize(n);
    for (int i = 0; i < n; i++)
      lambdas[i] = lambda_path[i];
    num_lambda = lambdas.size();
    target_lambda = lambdas[num_lambda-1];
  }
}

class ActNewtonSolver {
private:
  const ActNewtonTrainParam m_param;
  const ObjFunction * m_obj;

public:
  ActNewtonSolver(ObjFunction * obj, ActNewtonTrainParam param):
    m_obj(obj), m_param(param) {}

  void solve(ObjFunction* obj){
    std::vector<double> & gr = obj->grad();
    double dev_null = obj->eval(); // initial un-penalized fvalue

    const std::vector<double> & lambda_path = param.lambda_path();

    std::vector<int> active_set;
    active_set.resize(d);
    
    // model parameters on the master branch
    ModelParam model_master = obj->model_params();

    obj->init_active_set();

    for (auto lambda_iter = lambda_path.begin(); 
          lambda_iter != lambda_path.end(); lambda_iter++){
      obj->set_model_params(model_master);  

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