#include "core_math.h"
#include <cmath>
#include <ctime>
#include <boost/assert.hpp>

extern "C"
{

#include "dSFMT.h"

};

using namespace std;
using namespace boost;
using namespace boost::python;


// ---------
// VARIABLES
// ---------

namespace
{

const double EPSILON = 1e-4;
dsfmt_t dsfmt;

}


// ----------------
// VISIBLE ROUTINES
// ----------------

/*! \brief Seed the PRNG.
 */
void init_random()
{
    // FIXME better seed, perhaps taken as parameter; etc
    dsfmt_init_gen_rand(&dsfmt, time(NULL));
}

/*! \brief Return a random variate in [0.0, 1.0).
 */
double ranf(void)
{
    return dsfmt_genrand_close_open(&dsfmt);
}

/*! \brief Return an approximation to the Bessel function.
 *
 *  Computes the Abramowitz and Stegum approximation to the log modified bessel
 *  funtion of the first kind -- stable for high values of nu. See Chris Elkan's
 *  ICML 2006 paper on the EDCM distribution.
 */
double approx_log_iv(double nu, double z)
{
    double znu = z / nu;
    double alpha = 1.0 + znu * znu;
    double eta = sqrt(alpha) + log(znu) - log(1.0 + sqrt(alpha));

    return -log(sqrt(2.0 * M_PI * nu)) + nu * eta - 0.25 * log(alpha);
}

/*! \brief Return a discrete draw from a multinomial.
 */
int rv_discrete_core(int d, aiter_d p)
{
    double cut = ranf();

    BOOST_ASSERT(cut <= 1.0);
    BOOST_ASSERT(cut >= 0.0);

    for(int i = 0; i < d; ++i, ++p)
    {
        cut -= *p;

        if(cut < 0.0)
            return i;
    }
}

/*! \brief Return a discrete draw from a multinomial.
 */
int rv_discrete(PyObject* p_object)
{
    PyArrayObject* p = array_cast(p_object, PyArray_DOUBLE);

    BOOST_ASSERT(p->nd == 1);

    return rv_discrete_core(p->dimensions[0], aiter_d(p, 0));
}

/*! \brief Return the parent-conditioned log likelihood of \p x.
 *  \param xl: The magnitude of \p x.
 */
double ll_vmf_core(int n, aiter_d mu, double kappa, aiter_l x, double xl)
{
    double l_bessel = approx_log_iv(n / 2.0 - 1.0, kappa);
    double product = 0.0;

    for(int i = 0; i < n; ++i, ++mu, ++x)
    {
        product += *mu * *x;
    }

    return kappa * product / xl + (n / 2.0) * log(kappa / (2.0 * M_PI)) - log(kappa) - l_bessel;
}

/*! \brief Return the parent-conditioned log likelihood of \p x.
 *  \param xl: The magnitude of \p x.
 */
double ll_vmf(PyObject* mu_object, double kappa, PyObject* x_object, double xl)
{
    PyArrayObject* mu = array_cast(mu_object, PyArray_DOUBLE);
    PyArrayObject* x = array_cast(x_object, PyArray_LONG);

    BOOST_ASSERT(x->nd == 1);
    BOOST_ASSERT(mu->nd == 1);
    BOOST_ASSERT(x->dimensions[0] == mu->dimensions[0]);

    // compute the likelihood
    return ll_vmf_core(x->dimensions[0], aiter_d(mu, 0), kappa, aiter_l(x, 0), xl);
}

/*! \brief Convert out of log space and L1-normalize \p x, in place.
 *  \param n: The dimensionality of \p x.
 */
void normalize_log_core(int n, aiter_d x)
{
    long double s = 0.0;
    aiter_d x_ = x;

    for(int i = 0; i < n; ++i, ++x)
    {
        s = add_log(s, *x);
    }

    long double normalized_sum = 0;

    for(int i = 0; i < n; ++i, ++x_)
    {
        *x_ = exp(*x_ - s);
        normalized_sum += *x_;
    }

    BOOST_ASSERT(normalized_sum > 0.0);     // for nan
    BOOST_ASSERT(fabs(normalized_sum - 1.0) < EPSILON);
}

/*! \brief Convert out of log space and L1-normalize \p x, in place.
 */
void normalize_log(PyObject* x_object)
{
    PyArrayObject* x = array_cast(x_object, PyArray_DOUBLE);

    BOOST_ASSERT(x->nd == 1);

    normalize_log_core(x->dimensions[0], aiter_d(x, 0));
}

