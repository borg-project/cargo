/*! \file   family.h
 *  \brief  Family.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_CARGO_STATISTICS_FAMILY_H_
#define _UTEXAS_CARGO_STATISTICS_FAMILY_H_

namespace utexas
{

/*! \brief Interface to a family of distributions.
 *
 *  A distribution object is an instantiation of a distribution (eg N(1.0,
 *  1.0)); a distribution class corresponds to a family of distributions.
 */
template
<
    typename S //!< type of samples from this family
>
class Family
{
    //! Sample from the distribution.
    virtual S variate() = 0;

    //! Sample from the distribution.
    virtual void variate(S& sample) = 0;

    //! Sample from the distribution.
//     virtual void variates(RANGE) = 0;

    virtual double log_likelihood(self, sample):
        """
        Return the log likelihood of C{counts} under this distribution.
        """

        pass

    def total_log_likelihood(self, samples):
        """
        Return the total log likelihood of many samples from this distribution.
        """

        return sum(self.log_likelihood(s) for s in samples)
};

}

#endif

