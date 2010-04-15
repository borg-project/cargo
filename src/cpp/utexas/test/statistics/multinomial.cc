/*! \file   utexas/test/statistics/multinomial.cc
 *  \brief  Test associated source files.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#include <boost/python.hpp>
#include <boost/test/unit_test.hpp>
#include <utexas/python/numpy_array.h>
#include <utexas/statistics/multinomial.h>

using utexas::Multinomial;
using utexas::NumpyArrayFY;

BOOST_AUTO_TEST_CASE(variate)
{
    typedef NumpyArrayFY<double, 1> NumpyArrayD1;

    // set up a parameter vector
    NumpyArrayD1 beta(4);

    beta(0) = 0.0;
    beta(1) = 0.25;
    beta(2) = 0.50;
    beta(3) = 0.25;

    // test the distribution
    Multinomial<NumpyArrayD1> multinomial(beta);
    auto                      sample = multinomial.variate();
}

