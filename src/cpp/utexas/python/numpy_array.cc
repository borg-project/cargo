/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#include "numpy_array.h"

namespace utexas
{
namespace details
{

template<>
bool numpy_array_fy_packed<ARRAY_PACKED_C>(PyObject* array)
{
    return PyArray_ISCONTIGUOUS(array);
}

template<>
bool numpy_array_fy_packed<ARRAY_PACKED_FORTRAN>(PyObject* array)
{
    return PyArray_ISFORTRAN(array);
}

}
}

