#include <picasso/actnewton.h>

namespace picasso {
namespace solver {
enum RegType {L1, SCAD, MCP};

// training parameters
ActNewtonTrainParam::ActNewtonTrainParam() {
  num_lambda = 100;
  target_lambda = 1e-6;
  reg_type = L1;
  reg_gamma = 3.0;
  num_relaxation_round = 3;
  prec = 1e-4;
  max_iter = 1000;
  include_intercept = true;
  lambdas.clear();
}

ActNewtonTrainParam::configure(const std::vector<std::pair<std::string, std::string> >& cfg){
  for (auto iter = cfg.begin(); iter != cfg.end(); iter++){
    if (iter->first == "nlambda")
      num_lambda = stoi(iter->second);
    else if (iter->first == "target_lambda")
      target_lambda = stof(iter->second);
    else if (iter->first == "reg_type"){
      if (iter->second == "L1")
        reg_type = L1;
      else if (iter->second == "SCAD")
        reg_type = SCAD;
      else if (iter->second == "MCP")
        reg_type = MCP;
      else 
        {/* throw exception */}
    } else {
      /* TODO */
    }
  }
}

ActNewtonTrainParam::set_lambdas(const double * lambda_path, int n) {
  lambdas.resize(n);
  for (int i = 0; i < n; i++)
    lambdas[i] = lambda_path[i];
    num_lambda = lambdas.size();
    target_lambda = lambdas[num_lambda-1];
  }
  
  std::vector<double> get_lambda_path() const
  {
    // TODO
    std::vector<double> emptyvec;
    return emptyvec;
  }
};

class ActNewtonSolver {
private:
  const ActNewtonTrainParam m_param;
  const ObjFunction * m_obj;

  std::vector<int> itercnt_path;
  std::vector<ModelParam> solution_path;

public:
  ActNewtonSolver(ObjFunction * obj, ActNewtonTrainParam param):
    m_param(param),m_obj(obj)  {}

  void solve(ObjFunction* obj){
    unsigned int d = obj->get_dim();

    const std::vector<double> & lambdas = m_param.get_lambda_path();
    itercnt_path.resize(lambdas.size(), 0);

    double dev_thr = obj->get_deviance() * m_param.prec;

    // actset_indcat[i] == 1 if i is in the active set
    std::vector<int> actset_indcat(d, 0); 
    // actset_idx <- which(actset_indcat==1) 
    std::vector<int> actset_idx; 
    
    std::vector<double> old_coef(d);

    // model parameters on the master path 
    // each master parameter is relaxed into SCAD/MCP parameter
    ModelParam model_master = obj->get_model_param();

    std::vector<double> stage_lambdas(d, 0);
    for (int i = 0; i < lambdas.size(); i++){
      // start with the previous solution on the master path
      obj->set_model_param(model_master);  

      // calculating gradients and other auxiliary vars such as r
      obj->update_auxiliary();

      // init the active set
      double threshold = 2*lambdas[i];
      if (i > 0) threshold -= lambdas[i-1];
      for (int j = 0; j < d; ++j){
        stage_lambdas[j] = lambdas[i];

        if (obj->get_grad[j] > threshold)
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

          double old_intcpt = obj->get_model_coef(-1);
          for (int j = 0; j < d; j++)
            old_coef[j] = obj->get_model_coef(j);

          // initialize actset_idx
          actset_idx.clear();
          for (int j = 0; j < d; j++)
            if (actset_indcat[j]) {
              double updated_coord = obj->coordinate_descent(j, stage_lambdas[j]);
              
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
            
              double old_beta = obj->get_model_coef(idx);
              double updated_coord = obj->coordinate_descent(idx, stage_lambdas[idx]);

              if (obj->get_local_change(old_beta, idx) > dev_thr) 
                terminate_loop_level_2 = false;
            }

            if (m_param.include_intecept){
              double old_intcpt = obj->get_model_coef(-1);
              obj->intercept_update();
              if (obj->get_local_change(old_intcpt, -1) > dev_thr)
                terminate_loop_level_2 = false;
            }

            if (terminate_loop_level_2)
              break;
          }

          itercnt_path[i] += loopcnt_level_2;

          // check stopping criterion 1: fvalue change
          for (int k = 0; k < actset_idx.size(); ++k){
            int idx = actset_idx[k];
            if (obj->get_local_change(old_coef[idx], idx) > dev_thr)
              terminate_loop_level_1 = false;
          }
          if ((m_param.include_intercept) && 
              (obj->get_local_change(old_intcpt, -1) > dev_thr))
            terminate_loop_level_1 = false;

          // recompute grad, second order coef w jand other aux vars
          obj->update_auxiliary();

          // check stopping criterion 2: active set change 
          for (int k = 0; k < d; k++)
            if (actset_indcat[k] == 0){
              if (fabs(obj->get_grad(k)) > stage_lambdas[k]){
                actset_indcat[k] = 1;
                terminate_loop_level_1 = false;
              }
            }

          if (terminate_loop_level_1)
           break;
        }

        if (loopcnt_level_0 == 1)
          model_master = obj->get_model_param();
        
        if (m_param.reg_type == L1)
          break;

        // update stage lambda
        for (int j = 0; j < d; j++){
          double beta = obj->get_model_coef(j);

          switch (m_param.reg_type) {
            case MCP: 
               stage_lambdas[j] = (fabs(beta) > lambdas[i] * m_param.reg_gamma) 
                  ? 0.0 : lambdas[i] - fabs(beta)/m_param.reg_gamma; 
            case SCAD: 
               stage_lambdas[j] = (fabs(beta) > lambdas[i] * m_param.reg_gamma) 
                  ? 0.0 : ((fabs(beta) > lambdas[i]) 
                  ? ((lambdas[i]*m_param.reg_gamma - fabs(beta))/(m_param.reg_gamma-1)):lambdas[i]); 
            default: 
               stage_lambdas[j] = lambdas[i];
          }
        }
      }

      solution_path.push_back(obj->get_model_param());
    }
  }
};

} // namespace solver
} // namespace picasso
