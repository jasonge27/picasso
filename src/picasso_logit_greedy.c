#include "mymath.h"
#include <R.h>

void picasso_logit_greedy(double *Y, double * X, double * beta, 
    double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, 
    int * size_act, double *obj, double *runt, double *lambda, int *nnlambda, 
    double *ggamma, int *mmax_ite, double *pprec, int *fflag){
    
    int i, k, m, n, d, max_ite1, max_ite2, nlambda, size_a, size_a1, comb_flag, match, ite1, ite2, c_idx, idx, flag;
    double gamma, w, wn, g, prec1, prec2, ilambda, ilambda0, tmp, dif1, dif2;
    clock_t start, stop;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    w = 0.25;
    wn = w*(double)n;
    gamma = *ggamma;
    flag = *fflag;
    
    double *beta2 = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    int *set_act = (int *) Calloc(d, int);
    double *grad = (double *) Calloc(d, double);
    double *p = (double *) Calloc(n, double);
    double *p_Y = (double *) Calloc(n, double);
    double *Xb = (double *) Calloc(n, double);
    start = clock();
    size_a = 0;
    
    double function_value = 0.0;
    double function_value_change = 0.0;
    double beta1_backup = 0.0;
    double a = 0;

    for (i=0; i<nlambda; i++) {
        Rprintf("lambda: %.3f \n", lambda[i]);
        ilambda0 = lambda[i];
        ilambda = lambda[i]/w;
        if(i>0) {
            intcpt[i] = intcpt[i-1] - sum_vec_dif(p,Y,n)/wn;
        }
        prec1 = (1+prec2*10)*ilambda;
        ite1 = 0;
        dif1 = prec1*2;
       // Rprintf("%f %f %d %d\n", dif1, prec1, ite1, max_ite1);
        while (dif1>prec1 && ite1<max_ite1) {
           // Rprintf("***outer loop***\n");
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            dif_vec(p_Y, p, Y, n); // p_Y = p - Y
            if(flag==1){
                get_grad_logit_l1_vec(grad, p_Y, X, n, d); // grad = <p-Y, X>/n
            }
            if(flag==2){
                get_grad_logit_mcp_vec(grad, p_Y, X, beta1, ilambda0, gamma, n, d); // grad = <p-Y, X>/n + h_grad(mcp)
            }
            if(flag==3){
                get_grad_logit_scad_vec(grad, p_Y, X, beta1, ilambda0, gamma, n, d); // grad = <p-Y, X>/n + h_grad(scad)
            }
            idx = max_abs_idx(grad, d);
            
            for (int ii = 0; ii < d; ii++){
                Rprintf("%.2f ", grad[ii]);
            }
            Rprintf("%d \n", idx);

            comb_flag = 1;
            if(size_a>0){
                match = is_match(idx,set_act,size_a);
                if(match == 1) {
                    comb_flag = 0;
                }
            }
            if(comb_flag==1){
                set_act[size_a] = idx;
                size_a++;
            }
            ite2 = 0;
            dif2 = 1;

            Rprintf("%d\n", size_a);
            // this while loop is IRLS on the fixed support,
            // cd until convergence (in terms of change of beta)
            function_value_change = 0.0;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;

                if (flag == 1){
                    function_value = get_function_value_l1(p, Y, Xb, beta1, intcpt[i],  n, ilambda); 
                } else if (flag == 2){
                    function_value = get_function_value_mcp(p, Y, Xb, beta1, intcpt[i], n, ilambda0, gamma );
                } else if (flag == 3){
                    function_value = get_function_value_scad(p, Y, Xb, beta1, intcpt[i], n, ilambda0, gamma);
                }
                // Rprintf("fvalue:%f,  size_a:%d\n", function_value, size_a);

                for (m=0; m<size_a; m++) {
                    c_idx = set_act[m];
                    p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                    dif_vec(p_Y, p, Y, n); // p_Y = p - Y

                    a = get_cord_hessian(p, X, c_idx, n);
                    Rprintf("a: %f\n", a);
                    if (fabs(a) < 1e-20){
                        continue;
                    }

                    if(flag==1){
                        g = get_grad_logit_l1(p_Y, X+c_idx*n, n); // g = <p-Y, X>/n
                    }
                    if(flag==2){
                        g = get_grad_logit_mcp(p_Y, X+c_idx*n, beta1[c_idx], ilambda0, gamma, n); // g = <p-Y, X>/n + h_grad(mcp)
                    }
                    if(flag==3){
                        g = get_grad_logit_scad(p_Y, X+c_idx*n, beta1[c_idx], ilambda0, gamma, n); // g = <p-Y, X>/n + h_grad(scad)
                    }

                    beta1_backup = beta1[c_idx];
                    tmp = g - 2*a*beta1[c_idx];
                    X_beta_update(Xb, X+c_idx*n, -beta1[c_idx], n); // X*beta = X*beta-X[,c_idx]*beta1[c_idx]
                    beta1[c_idx] = -(soft_thresh_l1(tmp, ilambda))/(2*a);
                    X_beta_update(Xb, X+c_idx*n, beta1[c_idx], n); // X*beta = X*beta+X[,c_idx]*beta1[c_idx]

                    function_value_change += a*(beta1[c_idx]-beta1_backup)*(beta1[c_idx]-beta1_backup) 
                                            + g*(beta1[c_idx]-beta1_backup) 
                                            + ilambda*(fabs(beta1_backup) - fabs(beta1[c_idx]));
                }
                ite2++;
                dif2 = dif_2norm(beta1, beta0, set_act, size_a);
                vec_copy(beta1, beta0, set_act, size_a);

                if (fabs(function_value_change) < 1e-6 *fabs(function_value)){
                 //   Rprintf("fv_change:%f  fv:%f\n", function_value_change, function_value);
                    break;
                }
            }
            ite_cyc[i] += ite2;
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            dif_vec(p_Y, p, Y, n); // p_Y = p - Y
            if(flag==1){
                get_grad_logit_l1_vec(grad, p_Y, X, n, d); // grad = <p-Y, X>/n
            }
            if(flag==2){
                get_grad_logit_mcp_vec(grad, p_Y, X, beta1, ilambda0, gamma, n, d); // grad = <p-Y, X>/n + h_grad(mcp)
            }
            if(flag==3){
                get_grad_logit_scad_vec(grad, p_Y, X, beta1, ilambda0, gamma, n, d); // grad = <p-Y, X>/n + h_grad(scad)
            }
            dif1 = max_abs_vec(grad, d);
          //  Rprintf("---dif1: %f\n", dif1);
            //dif1 = dif_2norm(beta1, beta2, set_act, size_a);
            //vec_copy(beta1, beta2, set_act, size_a);
            size_a1 = 0;
            for (k=0; k<size_a; k++) {
                c_idx = set_act[k];
                if(beta1[c_idx]!=0){
                    set_act[size_a1] = c_idx;
                    size_a1++;
                }
            }
            size_a = size_a1;
            ite1++;
        }
        ite_lamb[i] = ite1;
        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        vec_copy(beta1, beta+i*d, set_act, size_a);
        size_act[i] = size_a;
    }
    
    Free(beta2);
    Free(beta1);
    Free(beta0);
    Free(grad);
    Free(set_act);
    Free(p);
    Free(p_Y);
    Free(Xb);
}
