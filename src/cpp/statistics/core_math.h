#ifndef _UTEXAS_CORE_MATH_H_
#define _UTEXAS_CORE_MATH_H_

#include <boost/python.hpp>
// #include "aiter.h"
// #include "numpy.h"
#include <stdexcept>

// void init_random();
// double ranf();
// double approx_log_iv(double nu, double z);
// int rv_discrete_core(int d, aiter_d p);
// int rv_discrete(PyObject* p_object);
// double ll_vmf_core(int p, aiter_d mu, double kappa, aiter_l x, double xl);
// double ll_vmf(PyObject* mu_object, double kappa, PyObject* x_object, double xl);
// long double add_log(long double x, long double y);
// void normalize_log_core(int n, aiter_d x);
// void normalize_log(PyObject* x_object);

double ln_poch(double a, double x)
{
    gsl_sf_result result;
    int status = gsl_sf_lnpoch_e(a, x, &result);

    if(status)
        throw std::runtime_error(gsl_strerror(status));

    return result.val;
}

double ln_gamma(double v)
{
    gsl_sf_result result;
    int status = gsl_sf_lngamma_e(v, &result);

    if(status)
        throw std::runtime_error(gsl_strerror(status));

    return result.val;
}

#endif

