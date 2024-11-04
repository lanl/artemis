//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef UTILS_EOS_HPP_
#define UTILS_EOS_HPP_

#include "artemis.hpp"

namespace ArtemisUtils {

// Maximum size of lambda array for optional extra EOS arguments.
// As it happens this must be >= 1 for device code.
static constexpr int lambda_max_vals = 1;

// Variant containing all EOSs to be used in Artemis.
using EOS = singularity::Variant<singularity::IdealGas>;

} // namespace ArtemisUtils

#endif // UTILS_EOS_HPP_
