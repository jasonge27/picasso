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

  /*! max number of iteration for innner loop */
  int max_iter;

  std::vector<double> lambdas;

  ActiveNewtonTrainParam() {
    num_lambda = 100;
    target_lambda = 1e-6;
    reg_type = L1;
    reg_gamma = 3.0;
    num_relaxation_round = 3;
    prec = 1e-4;
    max_iter = 1000;
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

  std::vector<int> itercnt_path;
  std::vector<ModelParam> solution_path;

public:
  ActNewtonSolver(ObjFunction * obj, ActNewtonTrainParam param):
    m_obj(obj), m_param(param) {}

  void solve(ObjFunction* obj){
    std::vector<double> & gr = obj->get_grad();
    unsigned int d = gr.size();

    const std::vector<double> & lambdas = param.get_lambda_path();
    itercnt_path.resize(lambdas.size(), 0);

    double dev_null = obj->eval(); // initial un-penalized fvalue
    double dev_thr = dev_null * m_param.prec;

    // actset_indcat[i] == 1 if i is in the active set
    std::vector<int> actset_indcat(d, 0); 
    // actset_idx <- which(actset_indcat==1) 
    std::vector<int> actset_idx; 
    
    // model parameters on the master path 
    // each master parameter is relaxed into SCAD/MCP parameter
    ModelParam model_master = obj->get_model_params();

    obj->init_active_set();

    std::vector<double> stage_lambdas(d, 0);
    for (int i = 0; i < lambdas.size(); i++){
      // start with the previous solution on the master path
      obj->set_model_params(model_master);  

      // init the active set
      double threshold = 2*lambdas[i];
      if (i > 0) threshold -= lambdas[i-1];
      for (int j = 0; j < d; ++j){
        stage_lambdas[j] = lambdas[i];

        if (gr[j] > threshold)
          actset_indcat[j] = 1;
      }

      // loop level 0: multistage convex relaxation
      int loopcnt_level_0 = 0;
      while (loopcnt_level_0 < m_param.num_relaxation_round) {
        loopcnt_level_0++;

        // loop level 1: active set update 
        int loopcnt_level_1 = 0;
        bool terminate_loop_level_1 = true;
        while (loopcnt_level_1< m_param.max_iter){
          loopcnt_level_1++;
          terminate_loop_level_1 = true;

          ModelParam model_param_level_1 = obj->get_model_param();

          actset_idx.clear();
          for (int j = 0; j < d; j++)
            if (actset_indcat[j]) {
              double updated_coord = obj->coordinate_descent(j, stage_lambda[j]);
              
              if (fabs(updated_coord) > 0)
                actset_idx.push_back(j);
            }
          
          // loop level 2: proximal newton on active set
          int loopcnt_level_2 = 0;
          bool terminate_loop_level_2 = true;
          while (loopcnt_level_2 < m_param.max_iter){
            loopcnt_level_2++;
            terminate_loop_level_2 = true;

            for (int k = 0; k < actset_idx.size(); k++){
              int idx = actset_idx[k];
            
              double updated_coord = obj->coordinate_descent(idx, stage_lambda[idx]);

              if (obj->get_local_change() > dev_thr) 
                terminate_loop_level_2 = false;
            }

            if (m_param.intercept){
              obj->intercept_update();
              if (obj->get_local_change() > dev_thr)
                terminate_loop_level_2 = false;
            }

            if (terminate_loop_level_2)
              break;
          }

          itercnt_path[i] += loopcnt_level_2;

          // check stopping criterion 1: fvalue change
          for (int k = 0; k < actset_idx.size(); ++k)
            if (obj->get_local_change(model_param_level_1, actset_idx[k]) > dev_thr)
              terminate_loop_level_1 = false;

          // check stopping criterion 2: active set change 
          for (int k = 0; k < d; k++)
            if (actset_indcat[k] == 0){
              gr[k] = obj->get_grad(k);
              if (gr[k] > stage_lambda[k]){
                actset_indcat[k] = 1;
                terminate_loop_level_1 = false;
              }
            }

          if (terminate_loop_level_1)
           break;
        }

        if (loopcnt_level_0 == 1)
          model_master = obj->get_model_params();
        
        if (m_param.reg_type == L1)
          break;

        // update stage lambda
        for (int j = 0; j < d; j++){
          double beta = obj->get_model_coef(j);

          switch (m_param.reg_type) {
            case MCP: 
               stage_lambda[j] = (fabs(beta) > lambda[i] * m_param.gamma) 
                  ? 0.0 : lambda[i] - fabs(beta)/m_param.gamma; 
            case SCAD: 
               stage_lambda[j] = (fabs(beta) > lambda[i] * m_param.gamma) 
                  ? 0.0 : ((fabs(beta) > lambda[i]) 
                  ? ((lambda[i]*m_param.gamma - fabs(beta))/(m_param.gamma-1)):lambda[i]) 
            default: 
               stage_lambda[j] = lambda[i];
          }
        }
      }

      solution_path.push_back(obj->get_model_param());
    }
  }
}

}
}