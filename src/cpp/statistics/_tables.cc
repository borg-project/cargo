/*! \file _tables.cc
 *  \brief Compiled implementation of core tables code.
 *  \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#include <iostream>
#include <boost/python.hpp>
#include <boost/tuple/tuple.hpp>
#include <numpy/arrayobject.h>

using namespace std;
using namespace boost;
using namespace boost::python;


// ---------
// INTERNALS
// ---------

namespace
{

// FIXME this implementation is... pretty sad - bcs

/*! \brief Find the lower bound of a row range in a sorted leaf.
 */
long find_lower_row(const object& leaf, long nleft, long nright, const object& key, const object& get_key)
{
    if(nright < nleft)
    {
        return -1;
    }
    else
    {
        long ncenter = nleft + (nright - nleft) / 2;
//         object center_key = get_key(leaf[ncenter]);
        object center_key = get_key(leaf, ncenter);

        if(center_key < key)
        {
            return find_lower_row(leaf, ncenter + 1, nright, key, get_key);
        }
        else if(center_key == key)
        {
            if(nleft == ncenter)
                return ncenter;
            else
                return find_lower_row(leaf, nleft, ncenter, key, get_key);
        }
        else
        {
            return find_lower_row(leaf, nleft, ncenter - 1, key, get_key);
        }
    }
}

/*! \brief Find the upper bound of a row range in a sorted leaf.
 */
long find_upper_row(const object& leaf, long nleft, long nright, const object& key, const object& get_key)
{
    if(nright < nleft)
    {
        return -1;
    }
    else
    {
        long ncenter;

        if(nright == nleft + 1)
            ncenter = nright;
        else
            ncenter = nleft + (nright - nleft) / 2;

//         object center_key = get_key(leaf[ncenter]);
        object center_key = get_key(leaf, ncenter);

//         std::string s = extract<std::string>(str(center_key));
//         cout << ncenter << " " << s << "\n";

//         cout << nleft << " " << nright << " " << ncenter << "\n";

        if(center_key < key)
        {
            return find_upper_row(leaf, ncenter + 1, nright, key, get_key);
        }
        else if(center_key == key)
        {
            if(nright == ncenter)
                return ncenter;
            else
                return find_upper_row(leaf, ncenter, nright, key, get_key);
        }
        else
        {
            return find_upper_row(leaf, nleft, ncenter - 1, key, get_key);
        }
    }
}

/*! \brief Find a row or range of rows in a sorted leaf.
 *  \return: (lower_bound_index, upper_bound_index + 1)
 */
python::tuple find_rows_py(const object& leaf, const object& key, const object& get_key)
{
    boost::python::ssize_t len_leaf = len(leaf);

    if(len_leaf > 0)
    {
        long lower = find_lower_row(leaf, 0, len_leaf - 1, key, get_key);

        if(lower != -1)
        {
            long upper = find_upper_row(leaf, 0, len_leaf - 1, key, get_key);

            return python::make_tuple(lower, upper + 1);
        }
        else
        {
            return python::make_tuple(0, 0);
        }
    }
    else
    {
        return python::make_tuple(0, 0);
    }
}

}


// ------
// MODULE
// ------

/*! \brief Module initialization.
 */
BOOST_PYTHON_MODULE(_tables)
{
    // setup
    import_array();
//    init_random();

    // bind module methods
    def("find_rows", &find_rows_py);
}

