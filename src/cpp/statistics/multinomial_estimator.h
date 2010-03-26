/*! \file   utexas/statistics/multinomial_estimator.h
 *  \brief  MultinomialEstimator.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_CARGO_STATISTICS_MULTINOMIAL_ESTIMATOR_H_
#define _UTEXAS_CARGO_STATISTICS_MULTINOMIAL_ESTIMATOR_H_

namespace utexas
{

/*! \brief Estimate the parameters of a multinomial distribution.
 *
 *  Extended to allow sample weighting for mixture-model EM.
 */
class MultinomialEstimator
{
    public:
        MultinomialEstimator(size_t d);

    public:
        Multinomial estimate(
            samples //!< the data.
            )
        {
            weights   = numpy.ones(counts.shape[0]) if weights is None else weights
        }

        //!  Return the estimated maximum likelihood distribution.
        Multinomial estimate(
            samples, //!< the data.
            weights  //!<
            )
        {
            Multinomial;

            weighted  = counts * weights[:, newaxis]
            mean      = numpy.sum(weighted, 0)
            mean     /= numpy.sum(mean)

            return Multinomial(mean)
        }

    private:
        size_t _d;
};

}

#endif

