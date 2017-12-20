#include "mathutils.h"

void picasso_gaussian_naive(double *Y, double *X, double *beta, double *intcpt,
                            int *beta_idx, int *cnzz, int *col_cnz,
                            int *ite_lamb, int *ite_cyc, double *obj,
                            double *runt, int *err, double *lambda,
                            int *nnlambda, double *ggamma, int *mmax_ite,
                            double *pprec, int *fflag, int *nn, int *dd,
                            int *ddf, int *vverbose, int *sstandardized) {
  int i, j, k, n, s, d, df, nlambda;
  int max_ite1, max_ite2, max_ite3, ite1, ite2, ite3, flag;
  int cnz, total_df;
  double gamma, prec;
  clock_t start, stop;
  int verbose = (*vverbose);
  int standardized = (*sstandardized);

  n = *nn;
  d = *dd;
  df = *ddf;
  max_ite1 = *mmax_ite;
  max_ite2 = *mmax_ite;
  max_ite3 = *mmax_ite;
  prec = *pprec;
  nlambda = *nnlambda;
  gamma = *ggamma;
  flag = *fflag;

  total_df = min_int(d, n) * nlambda;

  start = clock();
  double *beta1 = (double *)Calloc(d, double);
  int *set_idx = (int *)Calloc(d, int);

  int *set_act = (int *)Calloc(d, int);
  int act_size = 0;

  int *active_set = (int *)Calloc(d, int);

  double *res = (double *)Calloc(n, double);
  double *grad = (double *)Calloc(d, double);
  double *S = (double *)Calloc(d, double);

  double r2 = 0;
  double gr, tmp;
  int terminate_loop, new_active_idx;

  // grad[j] = <res, X[,j]>
  for (i = 0; i < n; i++) {
    res[i] = Y[i];
  }
  prec = 0.0;
  for (i = 0; i < n; i++)
    prec += Y[i] * Y[i];
  prec = (*pprec) * sqrt(prec / n);

  for (i = 0; i < d; i++)
    grad[i] = vec_inprod(res, X + i * n, n) / n;

  for (i = 0; i < d; i++) {
    set_act[i] = 0;
    beta1[i] = 0;
    active_set[i] = 0;
    set_idx[i] = 0;

    // S[i] = <X[,i], X[,i]>/n
    if (standardized == 0)
      S[i] = vec_inprod(X + i * n, X + i * n, n) / n;
    else
      S[i] = 1.0;
  }

  cnz = 0;
  double tmp_change = 0.0;
  double beta_cached = 0.0;
  act_size = -1;
  int flag1 = 0;
  int flag2 = 1;
  for (i = 0; i < nlambda; i++) {
    if (verbose)
      Rprintf("lambda i:%f \n", lambda[i]);

    /*
for (j = 0; j < d; j++)
  if (active_set[j] == 0) {
    if (flag == 1)
      tmp = soft_thresh_l1(grad[j], lambda[i]);
    if (flag == 2)
      tmp = soft_thresh_mcp(grad[j], lambda[i], gamma);
    if (flag == 3)
      tmp = soft_thresh_scad(grad[j], lambda[i], gamma);

    if (fabs(tmp) > 1e-8)
      active_set[j] = 1;
  }
*/
    double thr = 0.0;
    if (i == 0) {
      double max_grad = 0.0;
      for (j = 0; j < d; j++)
        if (fabs(grad[j]) > max_grad)
          max_grad = fabs(grad[j]);
      if (flag == 1)
        thr = 2 * lambda[i] - max_grad;
      if (flag == 2)
        thr = lambda[i] + gamma / (gamma - 1) * (lambda[i] - max_grad);
      if (flag == 3)
        thr = lambda[i] + gamma / (gamma - 2) * (lambda[i] - max_grad);
    } else {
      if (flag == 1)
        thr = 2 * lambda[i] - lambda[i - 1];
      if (flag == 2)
        thr = lambda[i] + gamma / (gamma - 1) * (lambda[i] - lambda[i - 1]);
      if (flag == 3)
        thr = lambda[i] + gamma / (gamma - 2) * (lambda[i] - lambda[i - 1]);
    }

    for (j = 0; j < d; j++)
      if (fabs(grad[j]) > thr)
        active_set[j] = 1;

    ite1 = 0;

    while (ite1 < max_ite1) {
      ite1 += 1;

      // STEP1: constructing support set for active set minimization
      ite2 = 0;
      while (ite2 < max_ite2) {
        ite2 += 1;
        terminate_loop = 0;

        ite3 = 0;
        while (ite3 < max_ite3) {
          ite3 += 1;
          terminate_loop = 1;
          for (k = 0; k < act_size; k++) {
            j = set_act[k];

            beta_cached = beta1[j];
            gr = vec_inprod(res, X + j * n, n) / n;
            coordinate_update_nonlinear(&beta1[j], gr, S[j], standardized,
                                        lambda[i], gamma, flag);

            if (beta1[j] == beta_cached)
              continue;

            tmp = beta1[j] - beta_cached;
            r2 += tmp * (2 * gr - tmp);

            for (s = 0; s < n; s++)
              res[s] -= tmp * X[j * n + s]; // res = res - tmp * X[,j]

            tmp_change = tmp * tmp;
            if (standardized == 0)
              tmp_change = tmp_change * S[j];

            if (tmp_change > prec) {
              terminate_loop = 0;
            }
          }

          if (terminate_loop) {
            break;
          }
        }
        ite_cyc[i] += ite3;

        if (verbose)
          Rprintf("--------ite3:%d\n", ite3);

        new_active_idx = 0;
        for (j = 0; j < d; j++) {
          if ((!active_set[j]) || (set_idx[j]))
            continue;
          beta_cached = beta1[j];
          // gr = <res, X[,j]> / n
          gr = vec_inprod(res, X + j * n, n) / n;
          coordinate_update_nonlinear(&beta1[j], gr, S[j], standardized,
                                      lambda[i], gamma, flag);

          if (beta1[j] == beta_cached)
            continue;

          if (set_idx[j] == 0) {
            act_size += 1;
            set_act[act_size] = j;
            set_idx[j] = 1;
          }

          tmp = beta1[j] - beta_cached;
          r2 += tmp * (2 * gr - tmp);
          // res = res - tmp * X[,j]
          for (k = 0; k < n; k++)
            res[k] -= tmp * X[j * n + k];

          new_active_idx = 1;
        }

        if (!new_active_idx) {
          break;
        }
      }

      if (verbose)
        Rprintf("---ite2:%d\n", ite2);

      new_active_idx = 0;
      for (j = 0; j < d; j++)
        if (active_set[j] == 0) {
          grad[j] = fabs(vec_inprod(res, X + j * n, n)) / n;
          if (flag == 1)
            tmp = soft_thresh_l1(grad[j], lambda[i]);
          if (flag == 2)
            tmp = soft_thresh_mcp(grad[j], lambda[i], gamma);
          if (flag == 3)
            tmp = soft_thresh_scad(grad[j], lambda[i], gamma);

          if (fabs(tmp) > 1e-8) {
            active_set[j] = 1;
            set_idx[j] = 1;
            act_size += 1;
            set_act[act_size] = j;
            new_active_idx = 1;
          }
        }

      if (!new_active_idx) {
        break;
      }
    }

    if (verbose)
      Rprintf("-ite1=%d\n", ite1);
    ite_lamb[i] = ite_cyc[i];

    intcpt[i] = 0.0;
    for (j = 0; j < n; j++)
      intcpt[i] += res[j];
    intcpt[i] = intcpt[i] / n;

    stop = clock();
    runt[i] = (double)(stop - start) / CLOCKS_PER_SEC;
    for (j = 0; j < d; j++) {
      if ((set_idx[j] != 0) && (fabs(beta1[j]) > 1e-6)) {
        if (cnz == total_df) {
          *err = 1;
          break;
        }
        beta[cnz] = beta1[j];
        beta_idx[cnz] = j;
        cnz++;
      }
    }
    col_cnz[i + 1] = cnz;
    if (*err == 1)
      break;
  }
  *cnzz = cnz;

  Free(beta1);
  Free(active_set);

  // Free(old_beta);
  Free(set_idx);
  Free(set_act);

  Free(res);
  Free(grad);
  Free(S);
}
