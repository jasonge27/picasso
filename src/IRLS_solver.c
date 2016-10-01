#include "mathutils.h"
#include "IRLS_solver.h"

void solve_weighted_lasso_with_naive_update(const double* X, 
    const double* w, // w = p * (1-p)
    const double* lambda, 
    const int n, const int d,
    const int max_ite, const double prec, const double dev_null,
    double* beta, double* Xb, int * active_set, 
    double * r, // r = y - p 
    double* intcpt, int* set_act, int* act_size, 
    double* runt, 
    int* inner_loop_count,
    int verbose){

    int i,  k, m, size_a;
    int c_idx;

    double g, tmp;
    clock_t start, stop;
    
    start = clock();

    int loopcnt = 0;

    double a =0.0;
    size_a = 0;
  
    double sum_w = 0.0;
    double thr = 0.0;
    for (i = 0; i < n; i++){
        sum_w += w[i];
    }

    for (k = 0; k < d; k++)
     if (active_set[k] == 1){ // skip inactive set
        calc_IRLS_coef(w, X, r, beta, k, n, &g, &a);

        tmp  = beta[k];
        if (k == 0){ 
            thr = lambda[k];
        } else {
            thr = 2*lambda[k] - lambda[k-1];
        }
        if (2*fabs(g) > thr){
            set_act[size_a] = k;
            size_a += 1;
            
            beta[k] = soft_thresh_l1(2*g, thr) / (2*a);
        } else {
            beta[k] = 0.0;
        }

        if (tmp == beta[k])
            continue;

        X_beta_update(Xb, X+k*n, -tmp, n);
        X_beta_update(Xb, X+k*n, beta[k], n);

        update_residual(r, w, X, beta[k]-tmp, k, n);
    }



    double dev_local;
    int terminate_loop;   
    double sum_r; 
    double dev_thr = prec * dev_null;
    while (loopcnt < max_ite) {  
        
        
        loopcnt ++;
        terminate_loop = 1;
        for (m = 0; m < size_a; m++) {
            c_idx = set_act[m];
                
            calc_IRLS_coef(w, X, r, beta, c_idx, n, &g, &a);

            tmp  = beta[c_idx];

            if (2*fabs(g) > lambda[c_idx]){
                beta[c_idx] = soft_thresh_l1(2*g, lambda[c_idx]) / (2*a);                
            } else{
                beta[c_idx] = 0.0;
            }

            if (tmp == beta[c_idx])
                continue;
            
            update_residual(r, w, X, beta[c_idx]-tmp, c_idx, n);

            X_beta_update(Xb, X+c_idx*n, -tmp, n);
            X_beta_update(Xb, X+c_idx*n, beta[c_idx], n);

            dev_local = 0.0;
            tmp = (beta[c_idx]-tmp)*(beta[c_idx]-tmp);
            for (i = 0; i < n; i++){
                dev_local += w[i]*X[c_idx*n+i]*X[c_idx*n+i];
            }
            dev_local = dev_local * tmp / (2*n);
            if (dev_local > dev_thr){
                terminate_loop = 0;
            }
        }
       
        //update intercept
        sum_r = 0.0;
        for (i = 0; i < n; i++){
            sum_r += r[i];
        } 
        tmp = sum_r / sum_w; 
        if (sum_r != 0){
            for (i = 0; i < n; i++){
                r[i] = r[i] - tmp * w[i];
            }
            (*intcpt) += tmp;
            dev_local = sum_w * tmp*tmp/ (2*n);
            if (dev_local > dev_thr){
                terminate_loop = 0;
            }
        }
        
        if (terminate_loop){
            break;
        }           
    }
    if (verbose){
        Rprintf("---innner loop %d\n", loopcnt);
    }

    *inner_loop_count = loopcnt;
       

    stop = clock();
    *runt = (double)(stop - start)/CLOCKS_PER_SEC;
    *act_size = size_a;
}


void calc_IRLS_coef(const double *  w, const double *  X, 
     double *  r,  double *  beta, 
     int k,  int n, 
    double * g, double * a){
    (*g) = 0.0;
    (*a) = 0.0;

    int i;
    double tmp;
    for (i = 0; i < n; i++){
        tmp = w[i]*X[k*n+i]*X[k*n+i];
        (*g) += tmp*beta[k] + r[i]*X[k*n+i];
        (*a) += tmp;
    }
    (*g) = (*g) / (2*n);
    (*a) = (*a) / (2*n);
}

void update_residual(double * r, const double *  w, 
    const double *  X, const double delta, 
    const int k, const int n){
    int i;
    for (i = 0; i < n; i++){
        r[i] = r[i] - w[i] * X[k*n+i] * delta; // delta = beta_new(k) - beta_old(k)
    }
}