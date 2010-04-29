/*! \file portfolio.h
 *  \brief Random code for portfolio work.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

// FIXME this code doesn't belong in cargo at all

#ifndef _UTEXAS_PORTFOLIO_H_
#define _UTEXAS_PORTFOLIO_H_

#include <map>
#include <vector>
#include <iostream>
#include <gsl/gsl_sf.h>
#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <utexas/statistics/dcm.h>
#include <utexas/statistics/core_math.h>
#include <utexas/python/numpy_array.h> // FIXME use eg CArray instead


namespace utexas
{
namespace ndmath
{

// ----------------------
// IMPLEMENTATION DETAILS
// ----------------------

namespace details
{

}


// --------
// ROUTINES
// --------

/*! \brief Calculate the posterior class probabilities.
 */
template<typename PiArray, typename SumArray, typename MixArray, typename CountsArray>
void
dcm_post_pi_K(
    PiArray& pi_K,
    const SumArray& sum_MK,
    const MixArray& mix_MKD,
    const CountsArray& counts_MD)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename PiArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename SumArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename MixArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::Element, unsigned long>::value));

    // mise en place
    size_t M = mix_MKD.template d<0>();
    size_t K = mix_MKD.template d<1>();
    size_t D = mix_MKD.template d<2>();

    BOOST_ASSERT(pi_K.template d<0>() == K);
    BOOST_ASSERT(counts_MD.template d<0>() == M);
    BOOST_ASSERT(counts_MD.template d<1>() == D);

    // calculate
    for(size_t k = K; k--;)
    {
        for(size_t m = M; m--;)
        {
            double ll = dcm_log_probability_mkd(sum_MK(m, k), mix_MKD, counts_MD, m, k);

            pi_K(k) *= exp(ll);
        }
    }

    double pi_K_sum = 0.0;

    for(size_t k = K; k--;)
    {
        pi_K_sum += pi_K(k);
    }

    for(size_t k = K; k--;)
    {
        pi_K(k) /= pi_K_sum;
    }
}

template<
    typename PiArray,
    typename SumArray,
    typename MixArray,
    typename CountsArray,
    typename OutArray>
void
dcm_model_predict(
    const PiArray& post_pi_K,
    SumArray& sum_MK,
    MixArray& mix_MKD,
    const CountsArray& counts_MD,
    OutArray& out_MD)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename PiArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename SumArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename MixArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::Element, unsigned long>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename OutArray::Element, double>::value));

    // mise en place
    size_t M = mix_MKD.template d<0>();
    size_t K = mix_MKD.template d<1>();
    size_t D = mix_MKD.template d<2>();

    BOOST_ASSERT(post_pi_K.template d<0>() == K);
    BOOST_ASSERT(sum_MK.template d<0>() == M);
    BOOST_ASSERT(sum_MK.template d<1>() == K);
    BOOST_ASSERT(counts_MD.template d<0>() == M);
    BOOST_ASSERT(counts_MD.template d<1>() == D);
    BOOST_ASSERT(out_MD.template d<0>() == M);
    BOOST_ASSERT(out_MD.template d<1>() == D);

    // generate posterior mixture components
    for(size_t m = M; m--;)
    {
        for(size_t d = D; d--;)
        {
            unsigned long count = counts_MD(m, d);

            for(size_t k = K; k--;)
            {
                sum_MK(m, k) += count;
                mix_MKD(m, k, d) += count;
            }
        }
    }

    // calculate outcome probabilities
    NumpyArrayFY<unsigned long, 1> outcome_D(D);

    for(size_t d = D; d--;)
    {
        // set up the hypothetical outcome vector
        for(size_t d_ = D; d_--;)
        {
            outcome_D(d_) = 0;
        }

        outcome_D(d) = 1;

        // calculate its likelihood
        for(size_t m = M; m--;)
        {
            out_MD(m, d) = 0.0;

            for(size_t k = K; k--;)
            {
                double ll = dcm_log_probability_mkd2(sum_MK(m, k), mix_MKD, outcome_D, m, k);

                out_MD(m, d) += post_pi_K(k) * exp(ll);
            }
        }
    }
}

template<
    typename PiArray,
    typename MixArray,
    typename CountsArray,
    typename OutArray>
void
multinomial_model_predict(
    PiArray& pi_K,
    const MixArray& mix_MKD,
    const CountsArray& counts_MD,
    OutArray& out_MD)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename PiArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename MixArray::Element, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::Element, unsigned long>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename OutArray::Element, double>::value));

    // mise en place
    size_t M = mix_MKD.template d<0>();
    size_t K = mix_MKD.template d<1>();
    size_t D = mix_MKD.template d<2>();

    BOOST_ASSERT(pi_K.template d<0>() == K);
    BOOST_ASSERT(counts_MD.template d<0>() == M);
    BOOST_ASSERT(counts_MD.template d<1>() == D);
    BOOST_ASSERT(out_MD.template d<0>() == M);
    BOOST_ASSERT(out_MD.template d<1>() == D);

    // calculate the posterior responsibilities
    for(size_t k = K; k--;)
    {
        for(size_t m = M; m--;)
        {
            double ll = multinomial_log_probability_mkd(mix_MKD, counts_MD, m, k);

            pi_K(k) *= exp(ll);
        }
    }

    double pi_K_sum = 0.0;

    for(size_t k = K; k--;)
    {
        pi_K_sum += pi_K(k);
    }

    for(size_t k = K; k--;)
    {
        pi_K(k) /= pi_K_sum;
    }

    // calculate outcome probabilities
    NumpyArrayFY<unsigned long, 1> outcome_D(D);

    for(size_t d = D; d--;)
    {
        // set up the hypothetical outcome vector
        for(size_t d_ = D; d_--;)
        {
            outcome_D(d_) = 0;
        }

        outcome_D(d) = 1;

        // calculate its likelihood
        for(size_t m = M; m--;)
        {
            out_MD(m, d) = 0.0;

            for(size_t k = K; k--;)
            {
                double ll = multinomial_log_probability_mkd2(mix_MKD, outcome_D, m, k);

                out_MD(m, d) += pi_K(k) * exp(ll);
            }
        }
    }
}

/*
template<
    typename PiArray,
    typename MixArray,
    typename CountsArray,
    typename OutArray>
void
multinomial_model_predict(
    PiArray& pi_K,
    const MixArray& mix_MKD,
    const CountsArray& counts_MD,
    OutArray& out_MD)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename PiArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename MixArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned long>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename OutArray::ElementType, double>::value));

    // mise en place
    size_t M = mix_MKD.template d<0>();
    size_t K = mix_MKD.template d<1>();
    size_t D = mix_MKD.template d<2>();

    BOOST_ASSERT(pi_K.template d<0>() == K);
    BOOST_ASSERT(counts_MD.template d<0>() == M);
    BOOST_ASSERT(counts_MD.template d<1>() == D);
    BOOST_ASSERT(out_MD.template d<0>() == M);
    BOOST_ASSERT(out_MD.template d<1>() == D);

    // calculate the posterior responsibilities
    for(action a)
    {
        mix_KD   = FIXME(a);
        counts_D = FIXME(a);

        for(size_t k = K; k--;)
        {
            pi_K(k) *= exp(multinomial_log_probability(mix_KD.slice(k), counts_D));
        }
    }

    double pi_K_sum = 0.0;

    for(size_t k = K; k--;)
    {
        pi_K_sum += pi_K(k);
    }

    for(size_t k = K; k--;)
    {
        pi_K(k) /= pi_K_sum;
    }

    // calculate the outcome probabilities
    for(action a)
    {
        D_a = FIXME;

        for(size_t d = D_a; d--;)
        {
            out_MD(m, d) = 0.0;

            for(size_t k = K; k--;)
            {
                out_MD(m, d) += pi_K(k) * exp(mix_MKD(m, k, d));
            }
        }
    }
}
*/

}
}

#endif

