#include "mymath.h"

// sum(max(|vi|-lambda, 0)) - z
double func1(double lambda, double * v, double z, int n){
    int i;
    double tmp=0;
    
    for(i=0; i<n ; i++){
        tmp += max(fabs(v[i])-lambda, 0);
    }
    return tmp-z;
}

double mod_bisec1(double * v, double z, int n){
    int cnt;
    double vmax, eps, lamb1, lamb2, lamb3, tmpfun1;

    if(l1norm(v,n)<=z) return 0;
    else{
        eps = 1e-5;
        vmax = max_abs_vec(v, n); //printf("vmax=%.12f,abs(v)=%.12f, z=%f \n",vmax,l1norm(v,n),z);
        if(vmax<=0 ) return 0;
        lamb1 = max(0,vmax-z);
        lamb2 = vmax;
        lamb3 = vmax/2;
        tmpfun1 = func1(lamb3, v, z, n);
        //printf("cnt=%d, lamb1=%f, lamb2=%f, lamb3=%f, func1=%f, func2=%f, func3=%f \n", cnt,lamb1,lamb2,lamb3,fun1(lamb1, v, z, n),fun1(lamb2, v, z, n),tmpfun1); 
        //sleep(10);
        cnt = 0;
        while(fabs(tmpfun1)>eps){
            if(tmpfun1>0) lamb1 = lamb3;
            if(tmpfun1<0) lamb2 = lamb3;
            lamb3 = (lamb1+lamb2)/2;
            tmpfun1 = func1(lamb3, v, z, n);
            cnt++;
            //if(cnt % 1 ==0) 
        }
        return lamb3;
    }
}

// sum(max(|vi|-lambda, 0)) - z
void func_S(double *lamb_S, double *lamb2, double *v, int *n1, int *n2, double *f2, double *f_der2, double *b2){
    int i;
    double v_sum;

    v_sum = 0;
    for(i=*n2; i>=*n1; i--){
        if(v[i]>*lamb_S && v[i]<=*lamb2){
            v_sum += v[i];
        }
        else break;
    }
    *f_der2 += i - (*n2);
    *b2 += v_sum;
    *f2 = (*lamb_S)*(*f_der2)+(*b2);
    *n2 = i;
    *lamb2 = *lamb_S;
}

void func_T(double *lamb_T, double *lamb1, double *v, int *n1, int *n2, double *f1, double *f_der1, double *b1){
    int i;
    double v_sum;

    v_sum = 0;
    for(i=(*n1)+1; i<=*n2; i++){
        if(v[i]>*lamb1 && v[i]<=*lamb_T){
            v_sum += v[i];
        }
        else break;
    }
    *f_der1 += i - (*n1) - 1;
    *b1 -= v_sum;
    *f1 = (*lamb_T)*(*f_der1)+(*b1);
    *n1 = i-1;
    *lamb1 = *lamb_T;
}

void func_M(double *lamb_M, double *lamb2, double *v, int *n1, int *n2, double *f2, double *f_der2, double *b2, int *n, double *f, double *f_der, double *b){
    int i;
    double v_sum;

    v_sum = 0;
    for(i=*n2; i>=*n1; i--){
        if(v[i]>*lamb_M && v[i]<=*lamb2){
            v_sum += v[i];
        }
        else break;
    }
    *f_der = *f_der2 + i - (*n2);
    *b = *b2 + v_sum;
    *f = (*lamb_M)*(*f_der)+(*b);
    *n = i;
}

void init_func(double lambda, double *v, int *n1, int *n2, double *f, double *f_der, double *b, double z){
    int i;
    double v_sum;

    if(*n1 == *n2) {f=0;}
    v_sum = 0;
    for(i=*n2; i>=*n1; i--){
        if(v[i]>lambda){
            v_sum += v[i];
        }
        else break;
    }
    *n1 = i;
    *f_der = i - (*n2);
    *b = v_sum - z;
    *f = lambda*(*f_der) + (*b);
}

double mod_bisec2(double * v, double z, int n){
    int i, cnt, max_cnt, n1, n2, n3, n_s, n_v;
    double f1, f2, f3, f_der1, f_der2, f_der3, b1, b2, b3;
    double vmax, eps, lamb1, lamb2, lamb3, lamb_T, lamb_S;

    if(l1norm(v,n)<=z) {
        return 0;
    }
    else{
        eps = 1e-5; max_cnt = 5;
        double *v_0 = (double*) malloc(n*sizeof(double));
        double *v_s = (double*) malloc(n*sizeof(double));
        max_fabs_vc(v, v_0, &vmax, &n_v, n, z);//printf("vmax=%.12f,abs(v)=%.12f, z=%f \n",vmax,l1norm(v,n),z);
        max_selc(v_0, vmax, v_s, n_v, &n_s, z);
        //fabs_vc(v, v_0, n);//printf("vmax=%.12f,abs(v)=%.12f, z=%f \n",vmax,l1norm(v,n),z);
        //vmax = max_vec(v_0, n);
        //max_selc(v_0, vmax, v_s, n, &n_s, z);
        free(v_0); 
        if(n_s==0) {free(v_s); return 0;}
        if(vmax == 0) {free(v_s); return 0;}
        if(n_s==1) {free(v_s); return vmax-1;}
        double *v_1 = (double*) malloc(n_s*sizeof(double));
        n = n_s;
        for(i=0;i<n;i++) v_1[i] = v_s[i];
        free(v_s); 
        sort_up_bubble(v_1, n);
        lamb1 = max(0, v_1[n-1]-z);
        lamb2 = v_1[n-1];
        n1 = 0; n2 = n-1;
        init_func(lamb1, v_1, &n1, &n2, &f1, &f_der1, &b1, z);
        if(fabs(f1)<eps) {free(v_1); return lamb1;}
//printf("\n n_s = %d \n",n_s);
//for(i=0;i<n;i++){ printf("v1[%d]=%f, ",i,v_1[i]);} printf("\n");
//printf("lamb1=%f,lamb2=%f,n1=%d,n2=%d,f1=%f,f_der1=%f,b1=%f,z=%f\n",lamb1,lamb2,n1,n2,f1,f_der1,b1,z);
        f2 = -z; b2 = -z; f_der2 = 0;
        lamb_T = lamb1-f1/f_der1;
        lamb_S = lamb2-f2*(lamb2-lamb1)/(f2-f1);
        lamb3 = (lamb_T+lamb_S)/2;
        if(fabs(lamb3 - lamb_S)<eps) {free(v_1); return lamb3;}
        func_T(&lamb_T, &lamb1, v_1, &n1, &n2, &f1, &f_der1, &b1);
        if(fabs(f1)<eps) {free(v_1); return lamb1;}
        if(n1==n2) {free(v_1); return lamb2-f2*(lamb2-lamb1)/(f2-f1);}
        func_S(&lamb_S, &lamb2, v_1, &n1, &n2, &f2, &f_der2, &b2);
        if(fabs(f2)<eps) {free(v_1); return lamb2;}
        if(n1==n2) {free(v_1); return lamb2-f2*(lamb2-lamb1)/(f2-f1);}
        func_M(&lamb3, &lamb2, v_1, &n1, &n2, &f2, &f_der2, &b2, &n3, &f3, &f_der3, &b3);
        if(fabs(f3)<eps) {free(v_1); return lamb3;}
        if(n1==n2) {free(v_1); return lamb3-f3*(lamb3-lamb1)/(f3-f1);}
        cnt = 0;
        while(fabs(f3)>eps && cnt<max_cnt){
            if(f3>0) {lamb1 = lamb3; f1=f3; f_der1=f_der3; b1=b3; n1=n3;}
            if(f3<0) {lamb2 = lamb3; f2=f3; f_der2=f_der3; b2=b3; n2=n3;}
            if(f_der2==0) lamb_T = lamb1-f1/f_der1;
            else lamb_T = max(lamb1-f1/f_der1, lamb2-f2/f_der2);
            lamb_S = lamb2-f2*(lamb2-lamb1)/(f2-f1);
            lamb3 = (lamb_T+lamb_S)/2;
            if(fabs(lamb3 - lamb_S)<eps) {free(v_1); return lamb3;}
//printf("cnt=%d,lamb_T=%f,lamb_S=%f,lambd3=%f,f1=%f,f_der1=%f,b1=%f,f2=%f,f_der2=%f,b2=%f \n",cnt,lamb_T,lamb_S,lamb3,f1,f_der1,b1,f2,f_der2,b2);
            func_T(&lamb_T, &lamb1, v_1, &n1, &n2, &f1, &f_der1, &b1);
//printf("      lamb_1=%f,n1=%d,n2=%d,f1=%f,f_der1=%f,b1=%f \n",lamb1,n1,n2,f1,f_der1,b1);
            if(fabs(f1)<eps) {free(v_1); return lamb1;}
            if(n1==n2) {free(v_1); return lamb2-f2*(lamb2-lamb1)/(f2-f1);}
            func_S(&lamb_S, &lamb2, v_1, &n1, &n2, &f2, &f_der2, &b2);
//printf("      lamb_2=%f,n1=%d,n2=%d,f2=%f,f_der2=%f,b2=%f \n",lamb2,n1,n2,f2,f_der2,b2);
            if(fabs(f2)<eps) {free(v_1); return lamb2;}
            if(n1==n2) {free(v_1); return lamb2-f2*(lamb2-lamb1)/(f2-f1);}
            func_M(&lamb3, &lamb2, v_1, &n1, &n2, &f2, &f_der2, &b2, &n3, &f3, &f_der3, &b3);
//printf("      lamb_3=%f,n1=%d,n2=%d,f3=%f,f_der3=%f,b3=%f \n",lamb3,n1,n2,f3,f_der3,b3);
//if(cnt>0) getchar();
            if(fabs(f3)<eps) {free(v_1); return lamb3;}
            if(n1==n2) {free(v_1); return lamb3-f3*(lamb3-lamb1)/(f3-f1);}
            cnt++;
        }
//if(cnt==max_cnt) printf(".");
        free(v_1);
        return lamb3;
    }
}

void euc_proj(double * v, double z, int n){
    int i;
    double lambda;

    lambda = mod_bisec2(v, z, n);
    for(i=0; i<n; i++)
        v[i] = sign(v[i])*max(fabs(v[i])-lambda, 0);
}
