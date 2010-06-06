/*! \file dcm.h
 *  \brief The Dirichlet compound multinomial (DCM) distribution.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifdef DOES_NOT_COMPILE
#ifndef _UTEXAS_DCM_H_
#define _UTEXAS_DCM_H_

#include <map>
#include <vector>
#include <iostream>
#include <gsl/gsl_sf.h>
#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <utexas/python/numpy_array.h>
#include <utexas/statistics/core_math.h>


namespace utexas
{
namespace ndmath
{

// ----------------------
// IMPLEMENTATION DETAILS
// ----------------------

namespace details
{

/*! \brief Compute the next value in the fixed-point iteration.
 */
template<
    typename AlphaArray,
    typename CountsArray,
    typename CountsSum1Array,
    typename WeightsArray>
double
minka_fixed_update(
    AlphaArray& alpha_D,
    const CountsArray& counts_ND,
    const CountsSum1Array& counts_sum1_N,
    const WeightsArray& weights_N,
    double weights_sum)
{
    // parameter sanity should be ensured by the caller
    size_t N = counts_ND.template d<0>();
    size_t D = counts_ND.template d<1>();

    // calculate the denominator
//    double alpha_sum = ndmath::sum(alpha_D);
    double alpha_sum = 0.0; // FIXME use the line above

    for(size_t d = D; d--;)
    {
        alpha_sum += alpha_D(d);
    }

    double denominator = 0.0;

    for(size_t n = N; n--;)
    {
        denominator += gsl_sf_psi(counts_sum1_N(n) + alpha_sum) * weights_N(n);
    }

    denominator -= weights_sum * gsl_sf_psi(alpha_sum);

    // calculate the numerator and update alpha
    double difference = 0.0;

    for(size_t d = D; d--;)
    {
        double numerator = 0.0;

        for(size_t n = N; n--;)
        {
            numerator += gsl_sf_psi(alpha_D(d) + counts_ND(n, d)) * weights_N(n);
        }

        numerator -= weights_sum * gsl_sf_psi(alpha_D(d));

        // update alpha
        double ratio = numerator / denominator;

        alpha_D(d) *= ratio;
        difference += fabs(ratio - 1.0);
    }

    // done
    return difference;
}

/*! \brief Results of precomputation for the Wallach recurrence estimator.
 */
struct PreWallachRecurrence
{
    typedef std::pair<unsigned int, double> NormCount;

    boost::scoped_array<NormCount> c_dot;
    size_t c_dot_size;
    boost::scoped_array<boost::scoped_array<NormCount> > c_k;
    boost::scoped_array<size_t> c_k_sizes;
};

}


// --------
// ROUTINES
// --------

/*! \brief Estimate the maximum likelihood DCM distribution.
 *
 *  Uses the Minka fixed-point estimation method.
 */
template<typename AlphaArray, typename CountsArray, typename WeightsArray>
void
estimate_dcm_minka_fixed(
    AlphaArray& alpha_D,
    const CountsArray& counts_ND,
    const WeightsArray& weights_N,
    double threshold,
    unsigned int cutoff)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename AlphaArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned int>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename WeightsArray::ElementType, double>::value));

    // mise en place
    size_t N = counts_ND.template d<0>();
    size_t D = counts_ND.template d<1>();

    BOOST_ASSERT(alpha_D.template d<0>() == D);
    BOOST_ASSERT(weights_N.template d<0>() == N);

    // precompute count norms
//    CArrayFY<unsigned int, 1> counts_sum1_N(N);
    NumpyArrayFY<unsigned int, 1> counts_sum1_N(N); // FIXME use CArrayFY

    for(size_t n = N; n--;)
    {
        counts_sum1_N(n) = 0;

        for(size_t d = D; d--;)
        {
            counts_sum1_N(n) += counts_ND(n, d);
        }
    }

    // the fixed-point loop
    // FIXME could do much more intelligent initialization
//    alpha = LiteralArrayFY<double, 1.0, 1>(D); // FIXME use this line instead of below
    for(size_t d = D; d--;)
    {
        alpha_D(d) = 1.0;
    }

//    double weights_sum = ndmath::sum(weights_N);
    double weights_sum = 0.0; // FIXME use line above

    for(size_t n = N; n--;)
    {
        weights_sum += weights_N(n);
    }

    double difference = threshold;

    for(unsigned int i = cutoff; i-- && difference >= threshold;)
    {
        difference = details::minka_fixed_update(alpha_D, counts_ND, counts_sum1_N, weights_N, weights_sum);

        // FIXME could do log-likelihood cutoff (but would that actually help?)
        // FIXME could instead do a mean-comparison cutoff above some max L1 norm

        // the digamma function is undefined at zero, so we take the approach
        // of flooring alpha at a small non-positive value; perhaps there is
        // a more principled approach to this issue
        for(size_t d = D; d--;)
        {
            if(alpha_D(d) < 1e-16)
                alpha_D(d) = 1e-16;
        }
    }
}

/*! \brief Precomputation for the Wallach DCM estimator.
 */
template<typename CountsArray, typename WeightsArray>
void
pre_estimate_dcm_wallach_recurrence(
    details::PreWallachRecurrence& pre,
    const CountsArray& counts_ND,
    const WeightsArray& weights_N)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned long>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename WeightsArray::ElementType, double>::value));

    // mise en place
    size_t N = counts_ND.template d<0>();
    size_t D = counts_ND.template d<1>();

    // precompute the unweighted norms
    typedef details::PreWallachRecurrence::NormCount NormCount;
    typedef
        std::map<NormCount::first_type, NormCount::second_type, std::greater<unsigned int> >
        NormCountsMap;

    NormCountsMap c_dot_map;    // STL all the way! bring it!
    NormCountsMap c_k_maps[D];  // C-style arrays! and C99 no less!

    for(size_t n = N; n--;)
    {
        unsigned int l1_norm = 0;
        double weight_n = weights_N(n);

        for(size_t d = D; d--;)
        {
            unsigned long count = counts_ND(n, d);

            l1_norm += count;

            if(count > 0)
                c_k_maps[d][count] += weight_n;
        }

        if(l1_norm > 0)
            c_dot_map[l1_norm] += weight_n;
    }

    // arrange them for estimation
    size_t i = 0;

    pre.c_dot_size = c_dot_map.size();
    pre.c_dot.reset(new NormCount[pre.c_dot_size]);

    BOOST_FOREACH(const NormCountsMap::value_type& item, c_dot_map)
    {
        pre.c_dot[i++] = item;
    }

    pre.c_k.reset(new boost::scoped_array<NormCount>[D]);
    pre.c_k_sizes.reset(new size_t[D]);

    for(size_t d = D; d--;)
    {
        i = 0;

        pre.c_k_sizes[d] = c_k_maps[d].size();

        pre.c_k[d].reset(new NormCount[pre.c_k_sizes[d]]);

        BOOST_FOREACH(const NormCountsMap::value_type& item, c_k_maps[d])
        {
            pre.c_k[d][i++] = item;
        }
    }
}

/*! \brief Estimate the maximum likelihood DCM distribution.
 *
 *  Uses Hanna Wallach's fixed-point estimator that exploits digamma recurrence.
 */
template<typename AlphaArray, typename CountsArray, typename WeightsArray>
void
estimate_dcm_wallach_recurrence(
    AlphaArray& alpha_D,
    const CountsArray& counts_ND,
    const WeightsArray& weights_N,
    double threshold,
    unsigned int cutoff)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename AlphaArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned long>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename WeightsArray::ElementType, double>::value));

    // mise en place
    size_t N = counts_ND.template d<0>();
    size_t D = counts_ND.template d<1>();

    BOOST_ASSERT(alpha_D.template d<0>() == D);
    BOOST_ASSERT(weights_N.template d<0>() == N);

    // precompute the weighted norms
    details::PreWallachRecurrence pre;

    pre_estimate_dcm_wallach_recurrence(pre, counts_ND, weights_N);

    // initialize alpha
    for(size_t d = D; d--;)
    {
        alpha_D(d) = 1.0;
    }

    // run the fixed-point iteration to convergence
    double difference = threshold;

    for(; cutoff-- && difference >= threshold;)
    {
        // compute sum_alpha
        double sum_alpha = 0.0;

        for(size_t d = D; d--;)
        {
            sum_alpha += alpha_D(d);
        }

        if(sum_alpha > 1e6)
        {
            // FIXME a bit of a hack
            break;
        }

        // compute the denominator; wallach_* are named as in (Wallach 2008)
        double wallach_s = 0.0;

        {
            double wallach_d = 0.0;
            size_t k = 0;

            for(size_t i = pre.c_dot_size; i--;)
            {
                details::PreWallachRecurrence::NormCount c_dot_item = pre.c_dot[i];

                for(; k < c_dot_item.first; ++k)
                {
                    wallach_d += 1.0 / (k + sum_alpha);
                }

                wallach_s += c_dot_item.second * wallach_d;
            }
        }

        // compute the numerator and update alpha
        difference = 0.0;

        for(size_t d = D; d--;)
        {
            // retrieve this weighted count
            const details::PreWallachRecurrence::NormCount* c_k_items = pre.c_k[d].get();
            size_t c_k_size = pre.c_k_sizes[d];

            // compute the numerator
            double wallach_s_k = 0.0;
            double wallach_d_k = 0.0;
            size_t k = 0;

            for(size_t i = c_k_size; i--;)
            {
                details::PreWallachRecurrence::NormCount c_k_item = c_k_items[i];

                for(; k < c_k_item.first; ++k)
                {
                    wallach_d_k += 1.0 / (k + alpha_D(d));
                }

                wallach_s_k += c_k_item.second * wallach_d_k;
            }

            // update this dimension of alpha
            double ratio = wallach_s_k / wallach_s;

            double old_FIXME = alpha_D(d);

            alpha_D(d) *= ratio;
            difference += fabs(ratio - 1.0);

            if(alpha_D(d) < 0.0 || !isfinite(alpha_D(d)))
            {
                std::cout << "oopsie... " << alpha_D(d) << " (" << ratio << "; old " << old_FIXME << ")\n";
                std::cout << wallach_s_k << " ---- " << wallach_s << "\n";
            }

            if(alpha_D(d) < boost::numeric::bounds<double>::smallest())
                alpha_D(d) = boost::numeric::bounds<double>::smallest();
        }
    }

//         double sum_alpha = 0.0;

//         for(size_t d = D; d--;)
//         {
//             sum_alpha += alpha_D(d);
//         }

//     std::cout << cutoff << " iterations from cutoff (sum is " << sum_alpha << ")\n";

    // the domain of the Dirichlet density is restricted to R^+
    for(size_t d = D; d--;)
    {
        BOOST_ASSERT(isfinite(alpha_D(d)));

        if(alpha_D(d) == 0.0)
            alpha_D(d) = boost::numeric::bounds<double>::smallest();

        BOOST_ASSERT(alpha_D(d) > 0.0);
    }
}

/*! \brief Calculate the log probability of the DCM distribution.
 */
template<typename AlphaArray, typename CountsArray>
double
dcm_log_probability(
    double sum_alpha,
    const AlphaArray& alpha_D,
    const CountsArray& counts_D)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename AlphaArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned long>::value));

    // mise en place
    size_t D = alpha_D.template d<0>();

    BOOST_ASSERT(counts_D.template d<0>() == D);

    // calculate
    unsigned long n = 0;

    for(size_t d = D; d--;)
    {
        n += counts_D(d);
    }

    double psigm = 0.0;

    for(size_t d = D; d--;)
    {
        psigm += ln_poch(alpha_D(d), counts_D(d));
    }

    return psigm - ln_poch(sum_alpha, n);
}

/*! \brief Calculate the log probability of the DCM distribution.
 *
 *  This method is exactly the same as the one above, but pulls counts
 *  from a matrix row. The code duplication is stupid. I'm in a hurry.
 */
template<typename AlphaArray, typename CountsArray>
double
dcm_log_probability_mkd(
    double sum_alpha,
    const AlphaArray& alpha_MKD,
    const CountsArray& counts_MD,
    size_t m,
    size_t k)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename AlphaArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned long>::value));

    // mise en place
    size_t M = alpha_MKD.template d<0>();
    size_t D = alpha_MKD.template d<2>();

    BOOST_ASSERT(counts_MD.template d<0>() == M);
    BOOST_ASSERT(counts_MD.template d<1>() == D);

    // calculate
    unsigned long n = 0;

    for(size_t d = D; d--;)
    {
        n += counts_MD(m, d);
    }

    double psigm = 0.0;

    for(size_t d = D; d--;)
    {
        psigm += ln_poch(alpha_MKD(m, k, d), counts_MD(m, d));
    }

    return psigm - ln_poch(sum_alpha, n);
}

/*! \brief Calculate the log probability of the DCM distribution.
 *
 *  This method is exactly the same as the one above, but uses a different
 *  count vector shape. The code duplication is stupid. I'm in a hurry.
 */
template<typename AlphaArray, typename CountsArray>
double
dcm_log_probability_mkd2(
    double sum_alpha,
    const AlphaArray& alpha_MKD,
    const CountsArray& counts_D,
    size_t m,
    size_t k)
{
    // sanity
    BOOST_STATIC_ASSERT((boost::is_same<typename AlphaArray::ElementType, double>::value));
    BOOST_STATIC_ASSERT((boost::is_same<typename CountsArray::ElementType, unsigned long>::value));

    // mise en place
    size_t D = alpha_MKD.template d<2>();

    BOOST_ASSERT(counts_D.template d<0>() == D);

    // calculate
    unsigned long n = 0;

    for(size_t d = D; d--;)
    {
        n += counts_D(d);
    }

    double psigm = 0.0;

    for(size_t d = D; d--;)
    {
        psigm += ln_poch(alpha_MKD(m, k, d), counts_D(d));
    }

    return psigm - ln_poch(sum_alpha, n);
}

}
}

#endif
#endif

