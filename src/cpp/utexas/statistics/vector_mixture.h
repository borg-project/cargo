/*! \file   vecotr_mixture.h
 *  \brief  VectorMixture.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_CARGO_STATISTICS_VECTOR_MIXTURE_H_
#define _UTEXAS_CARGO_STATISTICS_VECTOR_MIXTURE_H_

#include <vector>
#include <utexas/statistics/multinomial.h>

namespace utexas
{
namespace statistics
{

/*! \brief Dynamic mixture of distributions.
 *  \sa    TupleMixture.
 */
template
<
    typename Component //!< the type of all component distributions.
>
class VectorMixture
{
    public:
        typedef typename Component::Sample Sample;

    private:
        typedef Multinomial            Pi;
        typedef std::vector<Component> Components;

    public:
        //! Construct.
        VectorMixture
        (
            Pi          pi,        //!< the indicator distribution.
            Components  components //!< the mixture components.
        )
        :
            _pi(pi),
            _components(components)
        {
            UTEXAS_ASSERT(_pi.size()  > 0);
            UTEXAS_ASSERT(_pi.size() == _components.size());

            // FIXME should assert the _pi sums to ~1
            // FIXME should assert that the component distributions are compatible
            // FIXME (for static distributions, gcc can make that check a no-op)
        }

    public:
        //! Sample from this mixture distribution.
        Sample variate() const
        {
            size_t z = _pi.indicator_variate();

            UTEXAS_ASSERT(z < _components.size());

            return _components[z].variate();
        }

        //! Get the log likelihood of a sample.
        double
        log_likelihood
        (
            const Sample& sample //!< sample from this mixture.
        )
        const
        {
            UTEXAS_ASSERT(sample.size() == _components.size());

            // draw the sample(s)
            double ll = 0.0;

            for(size_t i = _components.size(); i--;)
            {
                // FIXME verify numerical integrity
                ll = add_log(ll, _pi.get_log_beta()(i) + _components[i].log_likelihood(sample));
            }

            return ll;
        }

    private:
        Pi         _pi;
        Components _components;
};

}
}

#endif

