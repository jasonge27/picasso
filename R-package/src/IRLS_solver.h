void calc_IRLS_coef(const double *  w, const double *  X, 
     double *  r,  double *  beta, 
     int k,  int n, 
    double * g, double * a);

void update_residual(double * r, const double *  w, 
    const double *  X, const double delta, 
    const int k, const int n);

void solve_weighted_lasso_with_naive_update(const double* X, 
    const double* w, // w = p * (1-p)
    const double* lambda, 
    const int n, const int d,
    const int max_ite, const double prec, const double dev_null,
    double* beta, double* Xb, int * active_set, 
    double * r, // r = y - p 
    double* intcpt, int* set_act, int* act_size, 
    double* runt, 
    int* inner_loop_count, int verbose);