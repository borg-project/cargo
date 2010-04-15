/*! \file utexas/python/numpy_array.h
 *  \brief NumpyArrayFF et al.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef _UTEXAS_PYTHON_NUMPY_ARRAY_H_
#define _UTEXAS_PYTHON_NUMPY_ARRAY_H_

#include <boost/array.hpp>
#include <boost/assert.hpp>
#include <boost/python.hpp>
#include <boost/type_traits.hpp>
#include <utexas/array_packing.h>
#include <utexas/python/numpy.h>

namespace utexas
{

//PyArrayObject* array_cast(PyObject* o, int type_num);

//template<typename T>
//void assert_array_type(PyArrayObject* a)
//{
//    BOOST_ASSERT(false);
//}

//template<> void assert_array_type<double>(PyArrayObject* a);
//template<> void assert_array_type<long>(PyArrayObject* a);

namespace details
{

//! Is the array packed as specified?
template<unsigned int P>
bool numpy_array_fy_packed(PyObject* array)
{
    static_assert(P != P, "unsupported array packing");

    return false;
}

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

/*! \brief A multi-dimensional array with numpy storage.
 */
template<typename E, size_t ND>
class NumpyArrayFY
{
    public:
        typedef E Element;

    public:
        /*! \brief Wrap a nested Python sequence with an n-d array interface.
         *
         *  Builds or wraps a compatible, aligned numpy array from the sequence (if
         *  the passed object is already a compatible, aligned numpy array, no copy
         *  is made).
         */
        NumpyArrayFY(boost::python::handle<PyObject> data_py)// :
            // ...
        {
            // FIXME figure out typenum at compile time
            int typenum;

            if(boost::is_same<E, double>::value)
                typenum = NPY_DOUBLE;
            else if(boost::is_same<E, unsigned int>::value)
                typenum = NPY_UINT;
            else if(boost::is_same<E, unsigned long>::value)
                typenum = NPY_ULONG;
            else
                BOOST_ASSERT(false);

            PyArray_Descr* dtype = PyArray_DescrFromType(typenum);

            Py_INCREF(dtype);

            // FIXME don't require writeability for read-only arrays
            _wrapped =
                boost::python::handle<PyObject>(
                    PyArray_FromAny(
                        data_py.get(),
                        dtype,
                        ND,
                        ND,
                        NPY_ALIGNED | NPY_WRITEABLE,
                        NULL));
        }

        /*! \brief Allocate a new, empty numpy array for storage.
         */
        // FIXME dimensionality > 1
        NumpyArrayFY(npy_intp d0)
        {
            // FIXME figure out typenum at compile time
            int typenum;

            if(boost::is_same<E, double>::value)
                typenum = NPY_DOUBLE;
            else if(boost::is_same<E, unsigned int>::value)
                typenum = NPY_UINT;
            else if(boost::is_same<E, unsigned long>::value)
                typenum = NPY_ULONG;
            else
                BOOST_ASSERT(false);

            _wrapped = boost::python::handle<PyObject>(PyArray_SimpleNew(1, &d0, typenum));
        }

    public:
        //! Get the length of a particular dimension.
        template<size_t D>
        size_t d() const
        {
            BOOST_STATIC_ASSERT(D < ND);

            return PyArray_DIM(_wrapped.get(), D);
        }

        //! Get the number of dimensions.
        size_t nd() const
        {
            return ND;
        }

        //! Is the array packed in the specified order?
        template<unsigned int P>
        bool packed() const
        {
            return details::numpy_array_fy_packed<P>(_wrapped.get());
        }

        //! Get access to an array element.
        E& operator ()(size_t i0)
        {
            // FIXME this will cause explicit class instantiation to fail; find a better solution
            BOOST_STATIC_ASSERT(ND == 1);

            return *static_cast<E*>(PyArray_GETPTR1(_wrapped.get(), i0));
        }

        const E& operator ()(size_t i0) const
        {
            // FIXME this will cause explicit class instantiation to fail; find a better solution
            BOOST_STATIC_ASSERT(ND == 1);

            return *static_cast<E*>(PyArray_GETPTR1(_wrapped.get(), i0));
        }

        E& operator ()(size_t i0, size_t i1)
        {
            // FIXME this will cause explicit class instantiation to fail; find a better solution
            BOOST_STATIC_ASSERT(ND == 2);

            return *static_cast<E*>(PyArray_GETPTR2(_wrapped.get(), i0, i1));
        }

        const E& operator ()(size_t i0, size_t i1) const
        {
            // FIXME this will cause explicit class instantiation to fail; find a better solution
            BOOST_STATIC_ASSERT(ND == 2);

            return *static_cast<E*>(PyArray_GETPTR2(_wrapped.get(), i0, i1));
        }

        E& operator ()(size_t i0, size_t i1, size_t i2)
        {
            // FIXME this will cause explicit class instantiation to fail; find a better solution
            BOOST_STATIC_ASSERT(ND == 3);

            return *static_cast<E*>(PyArray_GETPTR3(_wrapped.get(), i0, i1, i2));
        }

        const E& operator ()(size_t i0, size_t i1, size_t i2) const
        {
            // FIXME this will cause explicit class instantiation to fail; find a better solution
            BOOST_STATIC_ASSERT(ND == 3);

            return *static_cast<E*>(PyArray_GETPTR3(_wrapped.get(), i0, i1, i2));
        }

        //! Get a handle on the wrapped numpy array.
        const boost::python::handle<PyObject>& get_wrapped() const
        {
            return _wrapped;
        }

    private:
        boost::python::handle<PyObject> _wrapped;
};

//    aiter(void* data, int stride) :
//        data((char*)data),
//        stride(stride) { }
//    aiter(PyArrayObject* a, int i) :
//        data(a->data + i * a->strides[0]),
//        stride(a->strides[0]) { assert_array_type<T>(a); }
//    aiter(PyArrayObject* a, int i, int j) :
//        data(a->data + i * a->strides[0] + j * a->strides[1]),
//        stride(a->strides[1]) { assert_array_type<T>(a); }
//    aiter(PyArrayObject* a, int i, int j, int k) :
//        data(a->data + i * a->strides[0] + j * a->strides[1] + k * a->strides[2]),
//        stride(a->strides[2]) { assert_array_type<T>(a); }

//    T& operator *()
//    {
//        return *(T*)(data);
//    }

//    aiter& operator ++()
//    {
//        data += stride;

//        return *this;
//    }

//    char* data;
//    int stride;
//typedef aiter<double> aiter_d;
//typedef aiter<long> aiter_l;

}

#endif

