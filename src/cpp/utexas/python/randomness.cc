/*! \author Bryan Silverthorn <bcs@cargo-cult.org>
 */

#include "randomness.h"

using utexas::Randomness;


//
// DESTRUCTOR
//

Randomness::~Randomness()
{
}


//
// OTHER
//

namespace utexas
{

std::shared_ptr<Randomness> do_something_random(std::shared_ptr<Randomness> randomness)
{
    return randomness;
}

}

