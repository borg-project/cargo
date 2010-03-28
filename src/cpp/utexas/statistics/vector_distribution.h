/*! \file   vector_distribution.h
 *  \brief  VectorDistribution.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_CARGO_STATISTICS_VECTOR_DISTRIBUTION_H_
#define _UTEXAS_CARGO_STATISTICS_VECTOR_DISTRIBUTION_H_

#include <vector>
#include <boost/foreach.hpp>

namespace utexas
{

/*! \brief Dynamic collection of (independent) distributions.
 *  \sa    TupleDistribution
 */
template
<
    typename D //!< the type of all child distributions.
>
class VectorDistribution
{
    public:
        typedef std::vector<typename D::Sample> Sample;

    private:
        typedef std::vector<D> Distributions;

    public:
        //! Construct.
        VectorDistribution(Distributions d) :
            // ...
            _d(d)
        {
        }

        //! Construct.
        VectorDistribution(Distributions&& d) :
            // ...
            _d(d)
        {
        }

    public:
        //! Sample from the distribution.
        Sample variate() const
        {
            Sample s;

            BOOST_FOREACH(auto d, _d)
            {
                s.push_back(d.variate());
            }

            return s;
        }

        //!  Return the log likelihood of C{counts} under this distribution.
        double
        log_likelihood
        (
            const Sample& sample //!< a sample to consider.
        )
        const
        {
            double ll = 0.0;

            BOOST_FOREACH(auto d, _d)
            {
                ll += d.log_likelihood(sample);
            }

            return ll;
        }

    private:
        Distributions _d; //!< the child distributions.
};

}

#endif

