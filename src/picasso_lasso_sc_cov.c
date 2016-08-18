#include "mymath.h"

void picasso_lasso_sc_cov(double *Y, double * X, double * XY, double * beta, double * intcpt, int * beta_idx, int * cnzz, int * col_cnz, int * ite_lamb, int * ite_cyc, double *obj, double *runt, int * err, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec, int *fflag, double *ttrunc, int * nn, int * dd, int * ddf, int *mmax_act_in, int * aalg, double *LL){
    
    int i, j, idx, n, d, d4, df, df1, max_ite1, max_ite2, nlambda, ite1, ite2, flag, act_in, hybrid, cnz, act_size, act_size1, act_size_all, max_act_in, alg, total_df;
    double gamma, prec2, ilambda, ilambda1, ilambda2, dif2, dbn, lamb_max, cutoff, trunc, intcpt_tmp, L;
    clock_t start, stop;
    
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
    L = *LL;
    alg = *aalg; // 1:cyclic 2:greedy 3:proximal 4:random 5:hybrid
    dbn = (double)n;
    max_act_in = *mmax_act_in;
    trunc = *ttrunc;
    total_df = min_int(d,n)*nlambda;
    
    start = clock();
    //double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    double *beta_tild = (double *) Calloc(d, double);
    int *set_idx = (int *) Calloc(d, int);
    int *set_act1 = (int *) Calloc(d, int);
    int *set_actidx = (int *) Calloc(df1, int);
    int *set_actidx1 = (int *) Calloc(df1, int);
    int *set_actidx_all = (int *) Calloc(df1, int);
    double *res = (double *) Calloc(n, double);
    int *idxmaxgd = (int *) Calloc(max_act_in, int);
    double *setmaxgd = (double *) Calloc(max_act_in, double);
    double *grad = (double *) Calloc(d, double);
    double **XX = (double **) Calloc(df1, double *);
    int *XX_act_idx = (int *) Calloc(d, int);
    double S[d];
    //double XX[df1][df1];
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
        //printf("%d:%f ",i,S[i]);
    }
    
    for(i=0;i<df1;i++){
        set_actidx[i] = d4;
        XX[i] = (double *) Calloc(df1, double);
        for(j=0;j<df1;j++){
            //*(*(XX+i)+j) = 0;
            XX[i][j] = 0;
            //printf("%f ",XX[i][j]);
        }
    }
    if(alg==4) for(i=0;i<d;i++) set_idx[i] = i;
    //printf("start df=%d ",df);
    
    cnz = 0;
    act_size = 0;
    act_size_all = 0;
    for (i=0; i<nlambda; i++) {
        if(alg==4) shuffle(set_idx, d);
        ilambda = lambda[i]*dbn;
        if(alg==5){
            ilambda1 = ilambda*(1+sqrt(trunc));
            ilambda2 = ilambda*(1+trunc);
        }
        else
            ilambda1 = ilambda*(1+trunc);
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
        // update XX^T
        act_in=0;
        //printf("i=%d ",i);
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
                            //S[j] = vec_inprod(X+j*n,X+j*n,n);
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
        //printf("act_size=%d \n",act_size);
        
        //act_size = 0;
        //for(j=0;j<d;j++){
        //    if(set_act1[j] == 1){
        //        //printf("j=%d ",j);
        //        set_actidx[act_size] = j;
        //        act_size++;
        //    }
        //}
        
        //printf("i=%d,actin=%d,act_size=%d act_size_all=%d \n",i,act_in,act_size,act_size_all);
        //printf("1 %f,%f,%f,%f,%f \n",grad[0],grad[1],grad[2],grad[3],grad[4]);
        ite1 = 0;
        prec2 = lambda[i]*(*pprec)*1e2;
        hybrid = 1;
        while (ite1 < max_ite1) {
            ite2 = 0;
            dif2 = 1e3;
            act_size1 = 0;
            for(j=0;j<act_size;j++){
                idx = set_actidx[j];
                if(set_act1[idx] == 1){
                    set_actidx1[act_size1] = idx;
                    act_size1++;
                }
            }
            //printf("ite2=%d act_size1=%d %f %f %f  \n",ite2,act_size1,beta1[1],beta1[4],beta1[6]);
            //printf("ite1=%d, act_size1=%d ", ite1,act_size1);
            // update the active coordinate
            while (dif2 > prec2 && ite2 < max_ite2) {
                intcpt_tmp = cal_intcpt(XX, XX_act_idx, XY[d], set_actidx1, act_size1, beta1, df, dbn);
                //printf("intcpti=%f ",intcpt[i]);
                if(intcpt_tmp-intcpt[i] != 0){
                    grad_ud(grad, XX, XX_act_idx, intcpt_tmp-intcpt[i], set_actidx1, act_size1, df); // grad[j] = grad[j]-intcpt[i]*sum(X_:j) on active set
                    intcpt[i] = intcpt_tmp;
                }
                //printf("intcpttmp=%f ",intcpt_tmp);
                //printf("ite2=%d intcpt=%f ", ite2,intcpt[i]);
                for (j=0; j<act_size1; j++) {
                    idx = set_actidx1[j];
                    grad_ud(grad, XX, XX_act_idx, -beta1[idx], set_actidx1, act_size1, XX_act_idx[idx]); // grad[] = grad[]+beta1[idx]*XX[idx][] on active set
                    if(flag==1) beta1[idx] = soft_thresh_l1(grad[idx]/S[idx], ilambda/S[idx]);
                    if(flag==2) beta1[idx] = soft_thresh_mcp(grad[idx]/S[idx], ilambda/S[idx], gamma);
                    if(flag==3) beta1[idx] = soft_thresh_scad(grad[idx]/S[idx], ilambda/S[idx], gamma);
                    //printf("idx=%d, beta=%f, grad=%f  ",j,beta1[j],grad[j]);
                    if(beta1[idx]==0) set_act1[idx] = 0;
                    else {
                        //printf("idx=%d, beta=%f, grad=%f  ", idx, beta1[idx], grad[idx]);
                        set_act1[idx] = 1;
                        grad_ud(grad, XX, XX_act_idx, beta1[idx], set_actidx1, act_size1, XX_act_idx[idx]); // grad[] = grad[]-beta1[idx]*XX[idx][] on active set
                    }
                }
                ite2++;
                //dif2 = dif_2norm_dense(beta1, beta0, d);
                dif2 = max_abs_vec_dif_act(beta1, beta0, set_actidx, act_size);
                //printf("act_size=%d, act_size1=%d, dif2=%f, prec2=%f   ",act_size,act_size1, dif2, prec2);
                vec_copy(beta1, beta0, set_actidx, act_size);
                act_size1 = 0;
                for(j=0;j<act_size;j++){
                    idx = set_actidx[j];
                    if(set_act1[idx] == 1){
                        set_actidx1[act_size1] = idx;
                        act_size1++;
                        //printf("act_size=%d,set_actidx=%d ",act_size,set_actidx[j]);
                    }
                }
                //printf("ite2=%d act_size1=%d dif=%f  \n",ite2,act_size1,dif2);
                
            }
            //printf("ite1=%d %f %f %f  \n",ite1,beta1[1],beta1[4],beta1[6]);
            //for(j=0;j<d;j++){
            //	if(beta1[j]!=0)
            //		printf("j=%d beta=%f  ",j,beta1[j]);
            //}
            ite_cyc[i] += ite2;
            
            // update the active set
            intcpt[i] = cal_intcpt(XX, XX_act_idx, XY[d], set_actidx1,act_size1, beta1, df, dbn);
            res_ud(res, Y, X, beta1, intcpt[i], set_actidx1, act_size1, n); // res = Y-X*beta1-intcpt
            //printf("intcpt=%f res %f,%f,%f,%f,%f,%f,%f \n",intcpt[i],res[0],res[1],res[2],res[3],res[4],res[5],res[6]);
            //printf("intcpt=%f res ",intcpt[i]);
            //for(j=0;j<n;j++){
            //    printf("%f ",res[j]);
            //}
            //printf("\n");
            //printf("4 %f,%f,%f,%f,%f,%f,%f \n",grad[0],grad[1],grad[2],grad[3],grad[4],grad[5],grad[6]);
            //printf("beta %f %f %f  \n",beta1[1],beta1[4],beta1[6]);
            act_in = 0;
            //for(j=0;j<n;j++){
            //    for(k=0;k<d;k++){
            //        printf("X[%d][%d]=%f ",j,k,X[k*n+j]);
            //    }
            //    printf("\n");
            //}
            if(alg==1) ud_act_cyclic_cov(X,XX,XX_act_idx,set_actidx_all,S,beta1,res,grad,set_act1,gamma,ilambda1,ilambda,flag,&act_in,&act_size_all,df,d4,d,n,err);
            if(alg==2) ud_act_greedy_cov(X,XX,XX_act_idx,set_actidx_all,S,beta1,idxmaxgd,setmaxgd,res,grad,set_act1,gamma,ilambda,flag,&act_in,&act_size_all,df,d4,max_act_in,d,n,err);
            if(alg==3) ud_act_prox_cov(X,XX,XX_act_idx,set_actidx_all,S,beta1,beta_tild,idxmaxgd,setmaxgd,res,grad,set_act1,gamma,L,ilambda,flag,&act_in,&act_size_all,df,d4,max_act_in,d,n,err);
            if(alg==4) ud_act_stoc_cov(X,XX,XX_act_idx,set_actidx_all,S,beta1,res,grad,set_act1,set_idx,gamma,ilambda1,ilambda,flag,&act_in,&act_size_all,df,d4,d,n,err);
            if(alg==5) ud_act_hybrid_cov(X,XX,XX_act_idx,set_actidx_all,S,beta1,idxmaxgd,setmaxgd,res,grad,set_act1,gamma,ilambda1,ilambda,flag,&act_in,&act_size_all,df,d4,max_act_in,hybrid,d,n,err);
            
            //printf("2 %f,%f,%f,%f,%f,%f,%f \n",grad[0],grad[1],grad[2],grad[3],grad[4],grad[5],grad[6]);
            //printf("i=%d,actin=%d,act_size=%d act_size1=%d act_size_all=%d \n",i,act_in,act_size,act_size1,act_size_all);
            //printf("beta %f %f %f  \n",beta1[1],beta1[4],beta1[6]);
            act_size = 0;
            for(j=0;j<d;j++){
                if(set_act1[j] == 1){
                    //printf("j=%d ",j);
                    set_actidx[act_size] = j;
                    act_size++;
                }
            }
            ite1++;
            if(alg==5){
                if(hybrid==1) {
                    if(act_in==0)
                        hybrid = 2;
                }
                if(hybrid==2){
                    if(act_in==0)
                        break;
                }
            }
            else{
                if(act_in==0) break;
            }
            break;
        }
        //printf("i=%d,ite1=%d,ite2=%d,act_size=%d,act_size_all=%d \n\n",i,ite1,ite2,act_size,act_size_all);
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
        //if(i==0) break;
    }
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
