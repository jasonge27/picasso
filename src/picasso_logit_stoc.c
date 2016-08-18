#include "mymath.h"

void picasso_logit_stoc(double *Y, double * X, double * beta, double * intcpt, int * nn, int * dd, int * ite_lamb, int * ite_cyc, int * size_act, double *obj, double *runt, double *lambda, int *nnlambda, double *ggamma, int *mmax_ite, double *pprec, int *fflag, int *mmax_act_in, double *ttrunc){
    
    int i, j, j1, k, m, n, d, max_ite1, max_ite2, nlambda, size_a, size_a1, match, ite1, ite2, c_idx, flag, max_act_in, act_in;
    double gamma, w, wn, g, prec1, prec2, ilambda, ilambda0, tmp, dif1, dif2, ilambda1, trunc;
    clock_t start, stop;
    
    n = *nn;
    d = *dd;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec1 = *pprec;
    prec2 = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    flag = *fflag;
    w = 0.25;
    wn = w*(double)n;
    max_act_in = *mmax_act_in;
    trunc = *ttrunc;
    
    double *beta2 = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    int *set_act = (int *) Calloc(d, int);
    int *set_idx = (int *) Calloc(d, int);
    double *p = (double *) Calloc(n, double);
    double *p_Y = (double *) Calloc(n, double);
    double *Xb = (double *) Calloc(n, double);
    double *grad = (double *) Calloc(d, double);
    start = clock();
    size_a = 0;
    for(i=0;i<d;i++){
        set_idx[i] = i;
    }
    
    for (i=0; i<nlambda; i++) {
        ilambda0 = lambda[i];
        ilambda = lambda[i]/w;
        ilambda1 = (1+trunc)*ilambda;
        prec1 = (1+trunc)*ilambda;
        ite1 = 0;
        dif1 = prec1*2;
        while (dif1>prec1 && ite1<max_ite1) {
            intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;
            p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            dif_vec(p_Y, p, Y, n); // p_Y = p - Y
            act_in = 0;
            shuffle(set_idx, d);
            for(j1=0; j1<d; j1++){
                j = set_idx[j1];
                match = is_match(j,set_act,size_a);
                if(match == 0){ // if j in set_act
                    if(flag==1){
                        g = get_grad_logit_l1(p_Y, X+j*n, n); // g = <p-Y, X>/n
                    }
                    if(flag==2){
                        g = get_grad_logit_mcp(p_Y, X+j*n, beta1[j], ilambda0, gamma, n); // g = <p-Y, X>/n + h_grad(mcp)
                    }
                    if(flag==3){
                        g = get_grad_logit_scad(p_Y, X+j*n, beta1[j], ilambda0, gamma, n); // g = <p-Y, X>/n + h_grad(scad)
                    }
                    tmp = beta1[j] - g/w;
                    if(fabs(tmp)>ilambda1){
                        set_act[size_a] = j;
                        size_a++;
                        act_in++;
                    }
                    if(act_in == max_act_in){
                        break;
                    }
                }
            }
            ite2 = 0;
            dif2 = 1;
            while (dif2>prec2 && ite2<max_ite2) {
                intcpt[i] = intcpt[i] - sum_vec_dif(p,Y,n)/wn;
                for (m=0; m<size_a; m++) {
                    c_idx = set_act[m];
                    p_update(p,Xb,intcpt[i],n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                    dif_vec(p_Y, p, Y, n); // p_Y = p - Y
                    if(flag==1){
                        g = get_grad_logit_l1(p_Y, X+c_idx*n, n); // g = <p-Y, X>
                    }
                    if(flag==2){
                        g = get_grad_logit_mcp(p_Y, X+c_idx*n, beta1[c_idx], ilambda0, gamma, n); // g = <p-Y, X>/n + h_grad(mcp)
                    }
                    if(flag==3){
                        g = get_grad_logit_scad(p_Y, X+c_idx*n, beta1[c_idx], ilambda0, gamma, n); // g = <p-Y, X>/n + h_grad(scad)
                    }
                    tmp = beta1[c_idx] - g/w;
                    X_beta_update(Xb, X+c_idx*n, -beta1[c_idx], n); // X*beta = X*beta-X[,c_idx]*beta1[c_idx]
                    beta1[c_idx] = soft_thresh_l1(tmp, ilambda);
                    X_beta_update(Xb, X+c_idx*n, beta1[c_idx], n); // X*beta = X*beta+X[,c_idx]*beta1[c_idx]
                }
                ite2++;
                dif2 = dif_2norm(beta1, beta0, set_act, size_a);
                vec_copy(beta1, beta0, set_act, size_a);
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
    Free(set_act);
    Free(set_idx);
    Free(p);
    Free(p_Y);
    Free(Xb);
    Free(grad);
}
