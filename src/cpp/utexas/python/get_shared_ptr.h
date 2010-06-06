/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef HEADER_C165F9CF_1FC3_49C2_9130_32E51A64A0E1
#define HEADER_C165F9CF_1FC3_49C2_9130_32E51A64A0E1

#include <memory>

namespace boost
{
namespace python
{

template<class T>
T* get_pointer(std::shared_ptr<T> const& p)
{
    return p.get();
}

}
}

#endif

