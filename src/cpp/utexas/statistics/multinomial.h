/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_STATISTICS_MULTINOMIAL_H_
#define _UTEXAS_STATISTICS_MULTINOMIAL_H_

#include <stdexcept>
#include <type_traits>
#include <gsl/gsl_rng.h>
#include <utexas/assert.h>
#include <utexas/array_packing.h>

namespace utexas
{

//! Multinomial distribution.
template<typename Array>
class Multinomial
{
    static_assert(std::is_same<typename Array::Element, double>::value, "need a double array");

    public:
        typedef Array Sample;

    private:
        typedef Array Beta;

    public:
        //! Construct.
        Multinomial
        (
            Beta beta //!< the distribution parameter vector.
        )
        :
            _beta(beta),
            _log_beta(beta.template d<0>())
            // FIXME should use some other rng instance, needs to use a different seed, etc etc.
        {
            UT_ASSERT(beta.nd() == 1);
            UT_ASSERT(beta.template packed<ARRAY_PACKED_C>());

            // FIXME L1 normalization should be optional
//             _beta     /= numpy.sum(beta)
//             _log_beta  = numpy.nan_to_num(numpy.log(self.__beta))
            // FIXME need to initialize _log_beta
        }

        // FIXME we want a move constructor?

        //! Destruct.
        ~Multinomial()
        {
        }

    public:
        //! Get a norm-1 sample from this distribution.
        Sample variate() const
        {
            return this->variate(1);
        }

        /*! \brief Get a sample from this distribution.
         */
        Sample
        variate
        (
            size_t N //!< the L1 norm of the count vectors drawn.
        )
        const
        {
            size_t K = _beta.template d<0>();
            Sample s   (K);

            // FIXME s needs to be contiguous (C-style), as does p, etc etc.
            // FIXME gsl error handling?

//             gsl_ran_multinomial(
//                 _r,
//                 K,
//                 N,
//                 const double p[],
//                 unsigned int n[]);

            // FIXME could just pull out the gsl code, right? (although it's potentially GPLed)
            throw std::runtime_error("whee!");
        }

        //! Get the nonzero dimension of a norm-1 sample.
        size_t indicator() const
        {
            // FIXME.
            return 0;
        }

        //! Return the log likelihood of \p sample under this distribution.
        double
        log_likelihood
        (
            const Sample& sample //!< the sample to consider.
        )
        {
            /*
            // sanity
            size_t D = _beta.template d<0>();

            UTEXAS_ASSERT(sample.template d<0>() == D);

            // calculate
            unsigned long n = 0;
            double        l = 0.0;

            for(size_t d = D; d--;)
            {
                n += sample(d);
                l -= ln_gamma(sample(d) + 1);
                l += _log_beta(d) * sample(d);
            }

            return l + ln_gamma(n + 1);
            */
            throw std::runtime_error("whee!");
        }

        //! Get the multinomial parameter vector.
        const Beta& get_beta() const
        {
            return _beta;
        }

        //!  Get the log of the multinomial parameter vector.
        const Beta& get_log_beta() const
        {
            return _log_beta;
        }

        // FIXME probably want an assignment operator or two, at least

    private:
        Beta     _beta;
        Beta     _log_beta;
};

}

#endif

