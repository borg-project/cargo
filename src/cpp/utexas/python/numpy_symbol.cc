/*! \file utexas/python/numpy_symbol.cc
 *  \brief Define the unique symbol for this module.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#include <boost/python.hpp>
#include "numpy_symbol.h"
#include <numpy/arrayobject.h>

__attribute__((__constructor__))
void set_up_ut_python()
{
    // FIXME shouldn't be necessary
    Py_Initialize();
    import_array();
}

