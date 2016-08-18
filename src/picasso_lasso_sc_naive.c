#include "mymath.h"

void picasso_lasso_sc_naive(double *Y, double * X, double * S, double * beta, double * intcpt, int * beta_idx, int * cnzz, int * col_cnz, int * ite_lamb, int * ite_cyc, double *obj, double *runt, int * err, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec, int *fflag, double *ttrunc, int * nn, int * dd, int *mmax_act_in, int * aalg, double *LL){
    
    int i, j, n, d, max_ite1, max_ite2, nlambda, ite1, ite2, flag, act_in, hybrid, cnz, max_act_in, alg, total_df;
    double gamma, prec2, ilambda, ilambda1, ilambda2, dif2, dbn, lamb_max, cutoff, trunc, L;
    clock_t start, stop;
    
    n = *nn;
    d = *dd;
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
    
    //printf("start \n");
    start = clock();
    //double *beta2 = (double *) malloc(d*sizeof(double));
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    double *beta_tild = (double *) Calloc(d, double);
    int *set_idx = (int *) Calloc(d, int);
    int *set_act1 = (int *) Calloc(d, int);
    double *res = (double *) Calloc(n, double);
    int *idxmaxgd = (int *) Calloc(max_act_in, int);
    double *setmaxgd = (double *) Calloc(max_act_in, double);
    double *grad = (double *) Calloc(d, double);
    for(i=0;i<n;i++){
        res[i] = Y[i];
    }
    for(i=0;i<d;i++){
        set_act1[i] = 0;
        beta1[i] = 0;
        beta0[i] = 0;
    }
    if(alg==4) for(i=0;i<d;i++) set_idx[i] = i;
    vec_mat_prod(grad, res, X, n, d); // grad = X^T res
    //printf("alg=%d,max_act_in=%d,L=%f \n",alg,max_act_in,L);
    
    cnz = 0;
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
                set_act1[j] = 1;
                act_in++;
                //if(i==1){
                //	printf("i=%d, j=%d,grad=%f \n", i,j,grad[j]);
                //}
            }
        //printf("i=%d,actin=%d \n",i,act_in);
        //printf("1 %f,%f,%f,%f,%f \n",grad[0],grad[1],grad[2],grad[3],grad[4]);
        ite1 = 0;
        prec2 = lambda[i]*(*pprec)*1e2;
        hybrid = 1;
        while (ite1 < max_ite1) {
            ite2 = 0;
            dif2 = 1e3;
            //printf("ite2=%d act_in=%d %f %f %f  \n",ite2,act_in,beta1[1],beta1[4],beta1[6]);
            // update the active coordinate
            while (dif2 > prec2 && ite2 < max_ite2) {
                intcpt[i] = mean(res, n);
                //printf("intcpti=%f ",intcpt[i]);
                dif_vec_const(res, intcpt[i], n); //res = res - intcpt[i]
                //printf("ite2=%d intcpt=%f ", ite2,intcpt[i]);
                for (j=0; j<d; j++) {
                    if (set_act1[j]==1) {
                        dif_vec_vec(res, X+j*n, -beta1[j], n); //res = res+beta1[j]*X[,j]
                        grad[j] = vec_inprod(res, X+j*n, n);
                        if(flag==1) beta1[j] = soft_thresh_l1(grad[j]/S[j], ilambda/S[j]);
                        if(flag==2) beta1[j] = soft_thresh_mcp(grad[j]/S[j], ilambda/S[j], gamma);
                        if(flag==3) beta1[j] = soft_thresh_scad(grad[j]/S[j], ilambda/S[j], gamma);
                        if(beta1[j]==0) set_act1[j] = 0;
                        else {
                            //printf("j=%d, beta=%f, grad=%f  ", j, beta1[j], grad[j]);
                            dif_vec_vec(res, X+j*n, beta1[j], n); //res = res-beta1[j]*X[,j]
                        }
                    }
                }
                dif_vec_const(res, -intcpt[i], n); //res = res + intcpt[i]
                ite2++;
                //dif2 = dif_2norm_dense(beta1, beta0, d);
                dif2 = max_abs_vec_dif(beta1, beta0, d);
                //printf("dif=%f \n",dif2);
                vec_copy_dense(beta1, beta0, d);
                act_in = 0;
                for(j=0;j<d;j++){
                    if(set_act1[j] == 1){
                        act_in++;
                    }
                }
                //printf("ite2=%d act_size=%d dif=%f %f %f %f  \n",ite2,act_in,dif2,beta1[1],beta1[4],beta1[6]);
            }
            //printf("ite1=%d %f %f %f  \n",ite1,beta1[1],beta1[4],beta1[6]);
            ite_cyc[i] += ite2;
            
            // update the active set
            intcpt[i] = mean(res, n);
            dif_vec_const(res, intcpt[i], n); //res = res - intcpt[i]
            //printf("intcpt=%f res %f,%f,%f,%f,%f,%f,%f \n",intcpt[i],res[0],res[1],res[2],res[3],res[4],res[5],res[6]);
            //printf("4 %f,%f,%f,%f,%f,%f,%f \n",grad[0],grad[1],grad[2],grad[3],grad[4],grad[5],grad[6]);
            act_in = 0;
            if(alg==1) ud_act_cyclic(X,S,beta1,res,grad,set_act1,gamma,ilambda1,ilambda,flag,&act_in,d,n);
            if(alg==2) ud_act_greedy(X,S,beta1,idxmaxgd,setmaxgd,res,grad,set_act1,gamma,ilambda,flag,&act_in,max_act_in,d,n);
            if(alg==3) ud_act_prox(X,S,beta1,beta_tild,idxmaxgd,setmaxgd,res,grad,set_act1,gamma,L,ilambda,flag,&act_in,max_act_in,d,n);
            if(alg==4) ud_act_stoc(X,S,beta1,res,grad,set_act1,set_idx,gamma,ilambda1,ilambda,flag,&act_in,d,n);
            if(alg==5) ud_act_hybrid(X,S,beta1,idxmaxgd,setmaxgd,res,grad,set_act1,gamma,ilambda1,ilambda,flag,&act_in,max_act_in,hybrid,d,n);
            
            //printf("2 %f,%f,%f,%f,%f,%f,%f \n",grad[0],grad[1],grad[2],grad[3],grad[4],grad[5],grad[6]);
            //printf("i=%d,actin=%d \n",i,act_in);
            //printf("%f %f %f  \n",beta1[1],beta1[4],beta1[6]);
            dif_vec_const(res, -intcpt[i], n); //res = res + intcpt[i]
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
        }
        //printf("\n i=%d,ite1=%d,ite2=%d \n\n",i,ite1,ite2);
        ite_lamb[i] = ite1;
        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        for(j=0;j<d;j++){
            if (set_act1[j]!=0){
                if(cnz==total_df){
                    *err = 1;
                    break;
                }
                beta[cnz] = beta1[j];
                beta_idx[cnz] = j;
                cnz++;
            }
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
    Free(idxmaxgd);
    Free(setmaxgd);
    Free(res);
    Free(grad);
}
