/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#ifndef HEADER_B0D8ACFF_4520_4387_96BD_A4ECE43EE572
#define HEADER_B0D8ACFF_4520_4387_96BD_A4ECE43EE572

#include <boost/python.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace utexas
{

//! A source of randomness for both C++ and Python.
class Randomness
{
    public:
        Randomness()
        {
        }
//             _r(gsl_rng_alloc(gsl_rng_default))

        virtual ~Randomness();
//             gsl_rng_free(_r);

    public:
        boost::mt19937        cpp_generator;
//         boost::python::object numpy_generator;
};

std::shared_ptr<Randomness> do_something_random(std::shared_ptr<Randomness> randomness);

}

#endif

