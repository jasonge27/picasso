#include "mymath.h"

void picasso_scio_sc(double * S, int * ite_lamb, int * ite_cyc, double * x, int *col_cnz, int *row_idx, double *obj, double *runt, int * dd, double *lambda, int *nnlambda, int *mmax_ite, double *pprec, double * ggamma, int *fflag, double *ttrunc, int *mmax_act_in, int * aalg, double *LL){
    
    int i, j, m, d, d_sq, col, max_ite1, max_ite2, nlambda, act_in, size_a1, size_a2, match, ite1, ite2, cnz, flag, max_act_in, alg, hybrid;
    double prec2, ilambda, dif2, gamma, ilambda1, ilambda2, lamb_max, dbn, cutoff, trunc, L;
    clock_t start, stop;
    
    d = *dd;
    d_sq = d*d;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec2 = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    L = *LL;
    alg = *aalg; // 1:cyclic 2:greedy 3:proximal 4:random 5:hybrid
    flag = *fflag;
    trunc = *ttrunc;
    max_act_in = *mmax_act_in;
    dbn = 1;
    
    double *beta1 = (double *) Calloc(d, double);
    double *beta0 = (double *) Calloc(d, double);
    double *beta_tild = (double *) Calloc(d, double);
    int *set_idx = (int *) Calloc(d, int);
    int *set_act = (int *) Calloc(d, int);
    int *set_act1 = (int *) Calloc(d, int);
    double *eye = (double *) Calloc(d, double);
    int *idxmaxgd = (int *) Calloc(max_act_in, int);
    double *setmaxgd = (double *) Calloc(max_act_in, double);
    double *grad = (double *) Calloc(d, double);
    
    cnz = 0;
    for (col=0; col<d; col++) {
        
        start = clock();
        for(i=0;i<d;i++){
            set_act1[i] = 0;
            beta1[i] = 0;
            beta0[i] = 0;
            eye[i] = 0;
            grad[i] = 0;
        }
        if(alg==4) for(i=0;i<d;i++) set_idx[i] = i;
        eye[col] = 1;
        set_act1[col] = 1;
        grad[col] = 1;
        beta1[col] = 1;
        beta0[col] = 1;
        size_a1 = 0;
        //grad_scio(grad, eye, S, beta1, set_act, size_a, d);
        
        for (i=0; i<nlambda; i++) {
            if(alg==4) shuffle(set_idx, d);
            ilambda = lambda[i];
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
            size_a2 = 0;
            for (j=0; j<d; j++){
                if(set_act1[j] == 0){
                    if (fabs(grad[j]) > cutoff) {
                        set_act1[j] = 1;
                        act_in++;
                        set_act[size_a2] = j;
                        size_a2++;
                    }
                }
                else{
                    set_act[size_a2] = j;
                    size_a2++;
                }
            }
            
            ite1 = 0;
            prec2 = lambda[i]*(*pprec)*1e2;
            hybrid = 1;
            while (ite1 < max_ite1){
                ite2 = 0;
                dif2 = 1e3;
                while (dif2 > prec2 && ite2 < max_ite2) {
                    for (m=0; m<size_a2; m++) {
                        j = set_act[m];
                        grad[j] = res(eye[j], S+j*d, beta1, set_act, size_a2, j);
                        if(flag==1) beta1[j] = soft_thresh_l1(grad[j]/S[j*d+j], ilambda/S[j*d+j]);
                        if(flag==2) beta1[j] = soft_thresh_mcp(grad[j]/S[j*d+j], ilambda/S[j*d+j], gamma);
                        if(flag==3) beta1[j] = soft_thresh_scad(grad[j]/S[j*d+j], ilambda/S[j*d+j], gamma);
                        if(beta1[j]==0) set_act1[j] = 0;
                    }
                    ite2++;
                    //dif2 = dif_2norm_dense(beta1, beta0, d);
                    dif2 = max_abs_vec_dif(beta1, beta0, d);
                    //printf("dif=%f \n",dif2);
                    vec_copy_dense(beta1, beta0, d);
                    size_a1 = 0;
                    for(m=0; m<size_a2; m++){
                        j = set_act[m];
                        if(set_act1[j] == 1){
                            set_act[size_a1] = j;
                            size_a1++;
                        }
                    }
                    size_a2 = size_a1;
                }
                ite_cyc[i*d+col] += ite2;
                act_in = 0;
                grad_scio(grad, eye, S, beta1, set_act, size_a1, d);
                if(alg==1) ud_act_cyclic_scio(S,beta1,grad,set_act1,gamma,ilambda1,ilambda,flag,&act_in,d);
                if(alg==2) ud_act_greedy_scio(S,beta1,idxmaxgd,setmaxgd,grad,set_act1,gamma,ilambda,flag,&act_in,max_act_in,d);
                if(alg==3) ud_act_prox_scio(S,beta1,beta_tild,idxmaxgd,setmaxgd,grad,set_act1,gamma,L,ilambda,flag,&act_in,max_act_in,d);
                if(alg==4) ud_act_stoc_scio(S,beta1,grad,set_act1,set_idx,gamma,ilambda1,ilambda,flag,&act_in,d);
                if(alg==5) ud_act_hybrid_scio(S,beta1,idxmaxgd,setmaxgd,grad,set_act1,gamma,ilambda1,ilambda,flag,&act_in,max_act_in,hybrid,d);
                
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
            
            ite_lamb[i*d+col] = ite1;
            stop = clock();
            runt[i*d+col] = (double)(stop - start)/CLOCKS_PER_SEC;
            
            for(m=0; m<size_a2; m++) {
                j = set_act[m];
                x[cnz] = beta1[j];
                row_idx[cnz] = i*d+j;
                cnz++;
            }
        }
        col_cnz[col+1]=cnz;
    }
    
    Free(beta1);
    Free(beta0);
    Free(beta_tild);
    Free(set_idx);
    Free(set_act);
    Free(set_act1);
    Free(idxmaxgd);
    Free(setmaxgd);
    Free(eye);
    Free(grad);
}
