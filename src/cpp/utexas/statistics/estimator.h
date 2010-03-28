/*! \file   estimator.h
 *  \brief  Estimator.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_CARGO_STATISTICS_ESTIMATOR_H_
#define _UTEXAS_CARGO_STATISTICS_ESTIMATOR_H_

namespace utexas
{

/*! \brief Estimate a distribution from samples.
 */
template<typename D, typename T>
class Estimator
{
	// domain of distribution D must be elements of T

	/*! \brief Return the estimated distribution.
	 */
    virtual FIXME estimate(const T& samples) = 0;
};

}

#endif

