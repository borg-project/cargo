/*! \file   family.h
 *  \brief  Family.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_CARGO_STATISTICS_TUPLE_DISTRIBUTION_H_
#define _UTEXAS_CARGO_STATISTICS_TUPLE_DISTRIBUTION_H_

#include <tuple>

namespace utexas
{

/*! \brief Static collection of independent random variables.
 */
template
<
    typename ... S //!< types of samples
>
class TupleDistribution
{
    public:
        typedef std::tuple<S...> Sample;

    public:
        TupleDistribution()
        {
        }

    public:
        //! Sample from the distribution.
        Sample variate()
        {
        }

        //! Sample from the distribution.
    //     virtual void variates(RANGE) = 0;

        //!  Return the log likelihood of C{counts} under this distribution.
        double log_likelihood(
            const Sample& sample //!< a sample to consider.
            )
        {
        }

        //! Return the total log likelihood of many samples from this distribution.
        double total_log_likelihood(
            samples //!< a range of samples to consider.
            )
        {
        }
};

}

#endif

