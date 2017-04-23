#ifndef MYMATH_H
#define MYMATH_H
#define MATHLIB_STANDALONE
#include "Rmath.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <R.h>
#include "R_ext/BLAS.h"
#include "R_ext/Lapack.h"
#include "Rinternals.h"
#include "R_ext/Rdynload.h"
#include <R_ext/Applic.h>

void coordinate_update(double * beta, double gr, double S, 
                        int standardized, double lambda);

double truncate(double x, double a);

int min_int(int a, int b);

double penalty_derivative(int method_flag, double x, double lambda, double gamma);

double get_penalty_value(int method_flag, double x, double lambda, double gamma);

double get_penalized_logistic_loss(int method_flag, double *p, double * Y, double * Xb, double * beta, 
                                double intcpt, int n, int d, double lambda, double gamma);


double get_penalized_poisson_loss(int method_flag, double *p, double * Y, double * Xb, double * beta, 
                                double intcpt, int n, int d, double lambda, double gamma);

double mean(double *x, int n);

double vec_inprod(double *x, double *y, int n);

// copy x to y
void vec_copy(double *x, double *y, int *xa_idx, int n);

// x[i] = soft_l1(y,lamb);
double soft_thresh_l1(double y, double lamb);

// x[i] = soft_scad(y,lamb);
double soft_thresh_scad(double y, double lamb, double gamma);

// x[i] = soft_mcp(y,lamb);
double soft_thresh_mcp(double y, double lamb, double gamma);

double soft_thresh_gr_l1(double y, double lamb, double beta, double dbn1);

double soft_thresh_gr_mcp(double y, double lamb, double beta, double gamma, double dbn1);

double soft_thresh_gr_scad(double y, double lamb, double beta, double gamma, double dbn1);

// Xb = Xb+X*beta
void X_beta_update(double *Xb, const double *X, double beta, int n);

// p[i] = 1/(1+exp(-intcpt-Xb[i]))
void p_update(double *p, double *Xb, double intcpt, int n);

void standardize_design(double * X, double * xx, double * xm, double * xinvc, int * nn, int * dd);

#endif