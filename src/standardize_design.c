#include "mymath.h"

void standardize_design(double * X, double * xx, double * xm, double * xinvc, int * nn, int * dd) {
    int i,j,jn,n,d;
    
    n = *nn;
    d = *dd;
    
    for (j=0; j<d; j++) {
        // Center
        xm[j] = 0;
        jn = j*n;
        for (i=0; i<n; i++) {
            xm[j] += X[jn+i];
        }
        xm[j] = xm[j] / n;
        for (i=0; i<n; i++) xx[jn+i] = X[jn+i] - xm[j];
        
        // Scale
        xinvc[j] = 0;
        for (i=0; i<n; i++) {
            xinvc[j] += pow(xx[jn+i], 2);
        }
        xinvc[j] = 1/sqrt(xinvc[j]/(n-1));
        for (i=0; i<n; i++) {
            xx[jn+i] = xx[jn+i]*xinvc[j];
            //if(i<3 && j<3) printf("%f ",xx[jn+i]);
        }
    }
}
