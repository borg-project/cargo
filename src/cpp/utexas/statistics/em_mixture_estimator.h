/*! \file   utexas/statistics/em_mixture_estimator.h
 *  \brief  EM_MixtureEstimator.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_STATISTICS_EM_MIXTURE_ESTIMATOR_H_
#define _UTEXAS_STATISTICS_EM_MIXTURE_ESTIMATOR_H_

#include <utexas/statistics/vector_mixture.h>

namespace utexas
{
namespace statistics
{

//! Estimate a finite [linked] mixture distribution using EM.
template
<
    typename Component
>
VectorMixture<Component>
estimate_vector_mixture
(
    // FIXME surely we can make these parameters const, right?
    size_t K,                                    //!< the number of mixture components.
    FIXME samples,                               //!< the data to fit.
    boost::function<Component (FIXME)> estimator //!< the component estimator.
)
{
    // FIXME could initialize more intelligently
    // FIXME need a non-lame termination criterion

    // generate random initial component parameterizations
    NumpyArrayFY<double> pi;

    pi_K  = numpy.random.random(K)
    pi_K /= numpy.sum(pi_K)

    std::vector<C> components;

    for(size_t k = K; k--;)
    {
        components.push_back(estimator(samples));
    }

    // take some number of EM steps
    for(size_t i = 16; i--)
    {
        // evaluate the responsibilities
        r_NK = numpy.empty((N, K))

        for n in xrange(N)
        {
            for k in xrange(K)
            {
                r = pi_K[k] * numpy.exp(c[k].log_likelihood(samples[n]));

                // FIXME is this special case correct?
                if(r == 0.0)
                    r_NK[n, k] = float_finfo.tiny;
                else:
                    r_NK[n, k] = r;
            }
        }

        r_NK /= numpy.sum(r_NK, 1)[:, newaxis]

        // find the maximum-likelihood estimates of components
        for(size_t k = K; k--;)
        {
            components[k] = estimator(samples, r_NK[:, k]);
        }

        // find the maximum-likelihood pi values
        pi_K = numpy.sum(r_NK, 0) / N
    }

    // FIXME use rvalues (to allow copy elision)
    return VectorMixture(pi, components);
}

}
}

#endif

