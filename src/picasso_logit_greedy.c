#include "mymath.h"
#include <R.h>

void calc_IRLS_coef( double *  w,  double *  X, 
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

// Iterative reweighted least square solver with warm start \beta0 and intcpt0
// min \frac{1}{2n} \sum_{i=1}^N w_i(r_i - X_i^T \beta + intcept)^2 + \lambda * |\beta|
void solve_IRLS_with_warmstart(const double* X, 
    const double* w, // w = p * (1-p)
    const double lambda, 
    const int n, const int d,
    const int max_ite, const double prec, const double dev_null,
    double* beta, double* Xb, 
    double * r, // r = y - p 
    double* intcpt, int* set_act, int* act_size, 
    double* runt, 
    int* inner_loop_count){

    int i, j, k, m, size_a;
    int c_idx, idx;

    double g, tmp;
    clock_t start, stop;
    
    start = clock();

    int loopcnt = 0;

    double a =0.0;
    size_a = 0;
  
    double sum_w = 0.0;
    for (i = 0; i < n; i++){
        sum_w += w[i];
    }

    for (k = 0; k < d; k++){
        calc_IRLS_coef(w, X, r, beta, k, n, &g, &a);
        
        tmp  = beta[k];
        if (2*fabs(g) > lambda){
            set_act[size_a] = k;
            size_a += 1;

            beta[k] = soft_thresh_l1(2*g, lambda) / (2*a);
        } else {
            beta[k] = 0.0;
        }

        if (tmp == beta[k])
            continue;

        X_beta_update(Xb, X+k*n, -tmp, n);
        X_beta_update(Xb, X+k*n, beta[k], n);

        update_residual(r, w, X, beta[k]-tmp, k, n);
    }



    double dev_local, dev_change;   
    double sum_r; 
    while (loopcnt < max_ite) {  
       // Rprintf("loopcnt: %d, active set size:%d\n", loopcnt, size_a);
        loopcnt ++;
        dev_change = 0;
        for (m = 0; m < size_a; m++) {
            c_idx = set_act[m];
                
            calc_IRLS_coef(w, X, r, beta, c_idx, n, &g, &a);

            tmp  = beta[c_idx];

            if (2*fabs(g) > lambda){
                beta[c_idx] = soft_thresh_l1(2*g, lambda) / (2*a);                
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
            if (dev_local > dev_change){
                dev_change = dev_local;
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
            if (dev_local > dev_change){
                dev_change = dev_local;
            }
        }
        
      //  Rprintf("inner loop, dev_change:%f, dev_null:%f\n", dev_change, dev_null);
        if (dev_change < prec * dev_null){
            break;
        }           
    }

    *inner_loop_count = loopcnt;
       

    stop = clock();
    *runt = (double)(stop - start)/CLOCKS_PER_SEC;
    *act_size = size_a;
}

void picasso_logit_greedy(double *Y, double *X, double *beta, double *intcpt, 
    int *nn, int *dd, int *ite_lamb, int *ite_cyc, int *size_act, 
    double *obj, double *runt, double *lambda, int *nnlambda, double *ggamma, 
    int *mmax_ite, double *pprec, int *fflag){
    
    int i, j, k, s, n, d, nlambda, size_a;
    double tmp, ilambda, ilambda0;
     
    n = *nn;
    d = *dd;
    int max_ite1 = *mmax_ite;
    int max_ite2 = *mmax_ite;
    double prec1 = *pprec;
    double prec2 = *pprec;
    nlambda = *nnlambda;

    int *set_act = (int *) Calloc(d, int);
    
    double *beta_old = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);

    double *p = (double *) Calloc(n, double);
    double *Xb = (double *) Calloc(n, double);
    double *w = (double *) Calloc(n, double);
    double *r = (double *) Calloc(n, double);


    double dev_null = 0.0;
    double dev_sat = 0.0;

    double beta0_null = 0.0;
    double avr_y = 0.0;
    for (i = 0; i < n; i++){
        avr_y += Y[i];
    }
    avr_y = avr_y / n;
    beta0_null = log(avr_y /(1-avr_y));
    dev_null = -(avr_y * beta0_null + log(1 - avr_y)); // dev_null > 0

    dev_sat = dev_null; 

    int outer_loop_count;
    double dev_local, dev_change;

    //max_ite1 = 4;

    for (i=0; i<nlambda; i++) {
      //  Rprintf("%f\n", lambda[i]);
      //  ilambda0 = lambda[i];
      //  ilambda = lambda[i] / w;
        if(i == 0) {
            intcpt[i] = 0;
            for (j = 0; j < d; j++){
                beta1[j] = 0.0;
            }
            for (j = 0; j < n; j++){
                Xb[j] = 0.0;
            }
        } else {
        //    intcpt[i] = intcpt[i-1] - sum_vec_dif(p, Y, n)/wn;
            intcpt[i] = intcpt[i-1];
        }

       // prec1 = (1 + prec2 * 10) * ilambda;
        prec1 = prec2;
        
        outer_loop_count = 0;
        while (outer_loop_count < max_ite1) {
            outer_loop_count++;

            p_update(p, Xb, intcpt[i], n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            for (j = 0; j < n; j++){
                w[j] = p[j] * (1 - p[j]);
                r[j] = Y[j] - p[j];
            }

            for (j = 0; j < d; j++){
                beta_old[j] = beta1[j];
            }

            solve_IRLS_with_warmstart(X, w, lambda[i], 
                n, d,
                max_ite2,  
                prec2, dev_null,
                beta1, Xb, r,
                &intcpt[i], 
                set_act, 
                &size_act[i], // active set size
                &runt[i],  // total run time
                &ite_cyc[i] // innner loop counter
            ); 
            
            dev_change = 0.0;
            for (s = 0 ; s < size_act[i]; s++){
                k = set_act[s];
                tmp = (beta1[k]-beta_old[k]);
                tmp = tmp*tmp;
                dev_local = 0.0;
                for (j = 0; j < n; j++){
                    dev_local += w[i]*X[k*n+j]*X[k*n+j]*tmp;
                }
                dev_local = dev_local / (2*n);
                if (dev_local > dev_change){
                    dev_change = dev_local;
                }        
            }
            
          //  Rprintf("dev_change:%f, dev_null:%f\n", dev_change, dev_null);
            if (dev_change < prec1 * dev_null){
                break;
            }

        }
        ite_lamb[i] = outer_loop_count;

        vec_copy(beta1, beta+i*d, set_act, size_act[i]);
    }
    
    Free(beta_old);
    Free(beta1);
    Free(set_act);
    Free(p);
    Free(Xb);
    Free(w);
    Free(r);
}
