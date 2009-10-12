/*! \file multinomial.h
 *  \brief The multinomial distribution.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_MULTINOMIAL_H_
#define _UTEXAS_MULTINOMIAL_H_

#include <gsl/gsl_sf.h>
#include <boost/python.hpp>
#include "numpy_array.h" // FIXME use eg CArray instead

namespace utexas
{
namespace ndmath
{

// --------
// ROUTINES
// --------

/*! \brief Calculate the log probability of the multinomial distribution.
 */
template<typename BetaArray, typename CountsArray>
double
multinomial_log_probability(
    const BetaArray& log_beta_D,
    const CountsArray& counts_D)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename BetaArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned long>::value));

    // mise en place
    size_t D = log_beta_D.template d<0>();

    BOOST_ASSERT(counts_D.template d<0>() == D);

    // calculate
    unsigned long n = 0;

    for(size_t d = D; d--;)
    {
        n += counts_D(d);
    }

    double lp = ln_gamma(n + 1);

    for(size_t d = D; d--;)
    {
        lp -= ln_gamma(counts_D(d) + 1);
        lp += log_beta_D(d) * counts_D(d);
    }

    return lp;
}

}
}

#endif

