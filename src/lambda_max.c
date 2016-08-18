#include "mymath.h"

void lambda_max(double * lambda_max, double * X, double * Y, int * nn, int * dd){
    int i,j,in,n,d;
    n = *nn;
    d = *dd;
    double tmp;
    
    *lambda_max = 0;
    n = *nn;
    d = *dd;
    for(i=0; i<d; i++){
        tmp = 0;
        in = i*n;
        for(j=0; j<n; j++){
            tmp += X[in+j]*Y[j];
        }
        if(fabs(tmp)>*lambda_max)
            *lambda_max = fabs(tmp);
    }
}
