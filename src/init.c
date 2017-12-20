#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .C calls */
extern void picasso_gaussian_cov(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void picasso_gaussian_naive(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void picasso_logit_solver(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void picasso_poisson_solver(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void picasso_sqrt_lasso_solver(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void standardize_design(void *, void *, void *, void *, void *, void *);

static const R_CMethodDef CEntries[] = {
    {"picasso_gaussian_cov",      (DL_FUNC) &picasso_gaussian_cov,      23},
    {"picasso_gaussian_naive",    (DL_FUNC) &picasso_gaussian_naive,    23},
    {"picasso_logit_solver",      (DL_FUNC) &picasso_logit_solver,      18},
    {"picasso_poisson_solver",    (DL_FUNC) &picasso_poisson_solver,    18},
    {"picasso_sqrt_lasso_solver", (DL_FUNC) &picasso_sqrt_lasso_solver, 18},
    {"standardize_design",        (DL_FUNC) &standardize_design,         6},
    {NULL, NULL, 0}
};

void R_init_picasso(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
