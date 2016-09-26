#include "mymath.h"

void picasso_gaussian_cov(double *Y, double * X, double * beta,
    double * intcpt, int * beta_idx, int * cnzz, int * col_cnz,
    int * ite_lamb, int * ite_cyc, double *obj, double *runt, int * err,
    double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec,
    int *fflag, int * nn, int * dd, int * ddf, int *mmax_act_in, 
    int *vverbose, int * sstandardized){
    int i, j, idx, n, d, d4, df, df1, max_ite1, max_ite2, nlambda, ite1, ite2, flag, act_in, hybrid, cnz, act_size, act_size1,  max_act_in, alg, total_df;
    double gamma, prec2, ilambda, ilambda1, ilambda2, dif2, dbn, lamb_max, cutoff, trunc, intcpt_tmp, L;
    clock_t start, stop;
    int verbose = (*vverbose);

   
    
    n = *nn;
    d = *dd;
    d4 = 4*d;
    df = *ddf;
    df1 = df+1;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec2 = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    flag = *fflag;
    L = n;
    alg = 1; // 1:cyclic 2:greedy 3:proximal 4:random 5:hybrid
    dbn = (double)n;

    max_act_in = *mmax_act_in;

    trunc = 0;
    total_df = min_int(d,n)*nlambda;


    start = clock();
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    double *beta_tild = (double *) Calloc(d, double);
    double *old_beta = (double *) Calloc(d, double);
    
    int *set_idx = (int *) Calloc(d, int);
    
    int *set_act1 = (int *) Calloc(d, int);
    int *set_actidx = (int *) Calloc(df1, int);
    int *set_actidx1 = (int *) Calloc(df1, int);
    
    double *res = (double *) Calloc(n, double);
    
    int *idxmaxgd = (int *) Calloc(max_act_in, int);
    
    double *setmaxgd = (double *) Calloc(max_act_in, double);
    
    double *grad = (double *) Calloc(d, double);
    
    double **XX = (double **) Calloc(df1, double *);
    int *XX_act_idx = (int *) Calloc(d, int);
    int *set_actidx_all = (int *) Calloc(df1, int);
    int act_size_all;

    
    double S[d];
    double XY[d];
    // double XX[df1][df1];
    // XX: set of xx^T values
    // XX_act_idx: the value of j is the index of jth coef in XX
    // set_actidx_all: the overall active coef
    // act_size_all: the overall number of active coef
    for(i=0;i<d;i++){
        set_act1[i] = 0;
        beta1[i] = 0;
        beta0[i] = 0;
        grad[i] = XY[i];
        XX_act_idx[i] = d4;
        S[i] = vec_inprod(X+i*n,X+i*n,n);
        XY[i] = vec_inprod(Y, X+i*n, n); // XY[i] = <Y, X[,i]>
    }
    
    for(i=0;i<df1;i++){
        set_actidx[i] = d4;
        XX[i] = (double *) Calloc(df1, double);
        for(j=0;j<df1;j++){
            XX[i][j] = 0;
        }
    }
    if(alg==4) for(i=0;i<d;i++) set_idx[i] = i;
    
    cnz = 0;
    act_size = 0;
    act_size_all = 0;
    double tmp_change = 0.0;
    double fchange =  0.0;
    double beta_cached = 0.0;

    for (i=0; i<nlambda; i++) {
      
        ilambda = lambda[i]*dbn;
       
        cutoff = 0;
        if (i != 0) {
            // Determine eligible set
            if (flag==1) cutoff = (2*lambda[i] - lambda[i-1])*dbn;
            if (flag==2) cutoff = (lambda[i] + gamma/(gamma-1)*(lambda[i] - lambda[i-1]))*dbn;
            if (flag==3) cutoff = (lambda[i] + gamma/(gamma-2)*(lambda[i] - lambda[i-1]))*dbn;
            intcpt[i] = intcpt[i-1];
        } else {
            // Determine eligible set
            lamb_max = 0;
            for (j=0; j<d; j++) if (fabs(grad[j]) > lamb_max) lamb_max = fabs(grad[j]);
            lamb_max = lamb_max/dbn;
            if (flag==1) cutoff = (2*lambda[i] - lamb_max)*dbn;
            if (flag==2) cutoff = (lambda[i] + gamma/(gamma-1)*(lambda[i] - lamb_max))*dbn;
            if (flag==3) cutoff = (lambda[i] + gamma/(gamma-2)*(lambda[i] - lamb_max))*dbn;
        }
        act_in=0;
        for (j=0; j<d; j++)
            if (fabs(grad[j]) > cutoff) {
                if(set_act1[j] == 0){
                    if(XX_act_idx[j]==d4){
                        if(act_size_all==df){
                            *err = 2;
                        }
                        if(act_size_all<df){
                            set_act1[j] = 1;
                            set_actidx[act_size] = j;
                            
                            act_size++;
                            act_in++;
                            
                            XX_act_idx[j] = act_size_all;
                            set_actidx_all[act_size_all] = j;
                            
                            updateXX(XX,XX_act_idx,X,set_actidx_all,act_size_all,n,df);
                            XX[act_size_all][act_size_all] = S[j];
                            act_size_all++;
                        }
                    }
                    else{
                        set_act1[j] = 1;
                        set_actidx[act_size] = j;
                        act_size++;
                        act_in++;
                    }
                }
            }

        ite1 = 0;
        prec2 = lambda[i]*(*pprec)*1e2;
        // outer loop begins here
        while (ite1 < max_ite1) {
            ite2 = 0;
            act_size1 = 0;

            for(j=0; j < act_size; j++){
                idx = set_actidx[j];
                if(set_act1[idx] == 1){
                    set_actidx1[act_size1] = idx;
                    act_size1++;
                }
            }

            for (j = 0; j < d; j++){
                old_beta[j] = beta1[j];
            }
            

            while (ite2 < max_ite2)  { 
                intcpt_tmp = cal_intcpt(XX, XX_act_idx, XY[d], set_actidx1, act_size1, beta1, df, dbn);
                if(intcpt_tmp-intcpt[i] != 0){
                    grad_ud(grad, XX, XX_act_idx, intcpt_tmp-intcpt[i], set_actidx1, act_size1, df); // grad[j] = grad[j]-intcpt[i]*sum(X_:j) on active set
                    intcpt[i] = intcpt_tmp;
                }

                fchange = -1.0;
                for (j=0; j<act_size1; j++) {
                    idx = set_actidx1[j];
                    grad_ud(grad, XX, XX_act_idx, -beta1[idx], set_actidx1, act_size1, XX_act_idx[idx]); // grad[] = grad[]+beta1[idx]*XX[idx][] on active set
                    
                    beta_cached = beta1[idx];

                    if(flag==1) beta1[idx] = soft_thresh_l1(grad[idx]/S[idx], ilambda/S[idx]);
                    if(flag==2) beta1[idx] = soft_thresh_mcp(grad[idx]/S[idx], ilambda/S[idx], gamma);
                    if(flag==3) beta1[idx] = soft_thresh_scad(grad[idx]/S[idx], ilambda/S[idx], gamma);
                    if(beta1[idx]==0) set_act1[idx] = 0;
                    else {
                        set_act1[idx] = 1;
                        grad_ud(grad, XX, XX_act_idx, beta1[idx], set_actidx1, act_size1, XX_act_idx[idx]); // grad[] = grad[]-beta1[idx]*XX[idx][] on active set
                    }

                    tmp_change = S[idx] * (beta_cached - beta1[idx]) * (beta_cached - beta1[idx]);
                    if (tmp_change > fchange){
                        fchange = tmp_change;
                    }
                }

                ite2++;
                if ((fchange >=0) && (fchange < prec2)){
                    break;
                }

                
                vec_copy(beta1, beta0, set_actidx, act_size);
                act_size1 = 0;
                for(j=0;j<act_size;j++){
                    idx = set_actidx[j];
                    if(set_act1[idx] == 1){
                        set_actidx1[act_size1] = idx;
                        act_size1++;
                    }
                }
                
            }

            if (verbose)
                Rprintf("---ite2=%d\n", ite2);
            ite_cyc[i] += ite2;
            
            // update the active set
            intcpt[i] = cal_intcpt(XX, XX_act_idx, XY[d], set_actidx1, act_size1, beta1, df, dbn);
            res_ud(res, Y, X, beta1, intcpt[i], set_actidx1, act_size1, n); // res = Y-X*beta1-intcpt
            act_in = 0;

            // check stopping critierion
            ite1++;
            fchange = -1.0;
            for (j = 0; j <d; j++){
                tmp_change = S[j] * (old_beta[j] - beta1[j])*(old_beta[j] - beta1[j]);
                if (tmp_change > fchange){
                    fchange = tmp_change;
                }
            }
            if ((fchange >=0) && (fchange < prec2)){
                break;
            }

            //ud_act_greedy_cov(X,XX,XX_act_idx,set_actidx_all,S,beta1,
            //    idxmaxgd,setmaxgd,res,grad,set_act1,gamma,ilambda,flag,
            //    &act_in,&act_size_all,df,d4,max_act_in,d,n,err);

            act_size = 0;
            for(j=0;j<d;j++){
                if(set_act1[j] == 1){
                    set_actidx[act_size] = j;
                    act_size++;
                }
            }
           
            if(act_in==0) break;
        }
        ite_lamb[i] = ite1;
        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        for(j=0;j<act_size;j++){
            if(cnz==total_df){
                *err = 1;
                break;
            }
            idx = set_actidx[j];
            beta[cnz] = beta1[idx];
            beta_idx[cnz] = idx;//i*d+idx;
            cnz++;
        }
        col_cnz[i+1] = cnz;
        if(*err==1) break;
    }
    if (verbose)
        Rprintf("-ite1=%d\n", ite1);
    *cnzz = cnz;
    Free(beta1);
    Free(beta0);
    Free(beta_tild);
    Free(set_idx);
    Free(set_act1);
    Free(set_actidx);
    Free(set_actidx1);
    Free(set_actidx_all);
    Free(idxmaxgd);
    Free(setmaxgd);
    Free(res);
    Free(grad);
    Free(XX_act_idx);
    for(i=1;i<df1;i++){
        Free(XX[i]);
    }
    Free(XX);
}
