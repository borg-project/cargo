/*! \file _statistics.cc
 *  \brief Compiled implementation of core statistics code.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "dcm.h"
#include "multinomial.h"
#include "numpy_array.h"

using namespace std;
using namespace boost;
using namespace boost::python;
using namespace utexas;
using namespace utexas::ndmath;


// ---------
// INTERNALS
// ---------

namespace
{

/*! \brief Wrap the gsl's log pochhammer function.
 */
double pochhammer_ln_py(double x, double n)
{
    return gsl_sf_lnpoch(x, n);
}

/*! \brief Wrap estimate_dcm_minka_fixed() for Python.
 */
handle<PyObject>
estimate_dcm_minka_fixed_py(
    handle<PyObject> counts_ND_py,
    handle<PyObject> weights_N_py,
    double threshold,
    unsigned int cutoff)
{
    NumpyArrayFY<unsigned int, 2> counts_ND(counts_ND_py);
    NumpyArrayFY<double, 1> weights_N(weights_N_py);
    NumpyArrayFY<double, 1> alpha_D(counts_ND.d<1>());

    estimate_dcm_minka_fixed(alpha_D, counts_ND, weights_N, threshold, cutoff);

    return alpha_D.get_wrapped();
}

/*! \brief Wrap estimate_dcm_wallach_recurrence() for Python.
 */
handle<PyObject>
estimate_dcm_wallach_recurrence_py(
    handle<PyObject> counts_ND_py,
    handle<PyObject> weights_N_py,
    double threshold,
    unsigned int cutoff)
{
    NumpyArrayFY<unsigned long, 2> counts_ND(counts_ND_py);
    NumpyArrayFY<double, 1> weights_N(weights_N_py);
    NumpyArrayFY<double, 1> alpha_D(counts_ND.d<1>());

    estimate_dcm_wallach_recurrence(alpha_D, counts_ND, weights_N, threshold, cutoff);

    return alpha_D.get_wrapped();
}

double
multinomial_log_probability_py(
    handle<PyObject> log_beta_D_py,
    handle<PyObject> counts_D_py)
{
    NumpyArrayFY<double, 1> log_beta_D(log_beta_D_py);
    NumpyArrayFY<unsigned long, 1> counts_D(counts_D_py);
 
    return multinomial_log_probability(log_beta_D, counts_D);
}

double
dcm_log_probability_py(
    double sum_alpha,
    handle<PyObject> alpha_D_py,
    handle<PyObject> counts_D_py)
{
    NumpyArrayFY<double, 1> alpha_D(alpha_D_py);
    NumpyArrayFY<unsigned long, 1> counts_D(counts_D_py);
 
    return dcm_log_probability(sum_alpha, alpha_D, counts_D);
}

}


// ------
// MODULE
// ------

/*! \brief Module initialization.
 */
BOOST_PYTHON_MODULE(_statistics)
{
    // setup
    import_array();
//    init_random();

    gsl_set_error_handler_off();

    // bind module methods
    def("pochhammer_ln", &pochhammer_ln_py);
    def("estimate_dcm_minka_fixed", &estimate_dcm_minka_fixed_py);
    def("estimate_dcm_wallach_recurrence", &estimate_dcm_wallach_recurrence_py);
    def("multinomial_log_probability", &multinomial_log_probability_py);
    def("dcm_log_probability", &dcm_log_probability_py);
}
 
