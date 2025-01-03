//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Artemis includes
#include "advection.hpp"
#include "blast.hpp"
#include "conduction.hpp"
#include "constant.hpp"
#include "disk.hpp"
#include "dust_coagulation.hpp"
#include "gaussian_bump.hpp"
#include "linear_wave.hpp"
#include "shock.hpp"
#include "strat.hpp"
#include "thermalization.hpp"

using namespace parthenon::package::prelude;

namespace artemis {
//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator
//! \brief
template <Coordinates T>
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  std::string name = pin->GetString("artemis", "problem");
  if (name == "advection") {
    advection::ProblemGenerator<T>(pmb, pin);
  } else if (name == "blast") {
    blast::ProblemGenerator<T>(pmb, pin);
  } else if (name == "conduction") {
    cond::ProblemGenerator<T>(pmb, pin);
  } else if (name == "constant") {
    constant::ProblemGenerator<T>(pmb, pin);
  } else if (name == "disk") {
    disk::ProblemGenerator<T>(pmb, pin);
  } else if (name == "gaussian_bump") {
    gaussian_bump::ProblemGenerator<T>(pmb, pin);
  } else if (name == "linear_wave") {
    linear_wave::ProblemGenerator<T>(pmb, pin);
  } else if (name == "shock") {
    shock::ProblemGenerator<T>(pmb, pin);
  } else if (name == "strat") {
    strat::ProblemGenerator<T>(pmb, pin);
  } else if (name == "thermalization") {
    thermalization::ProblemGenerator<T>(pmb, pin);
  } else if (name == "dust_coagulation") {
    dust_coagulation::ProblemGenerator<T>(pmb, pin);
  } else {
    PARTHENON_FAIL("Invalid problem name!");
  }
}

} // namespace artemis

#endif // PGEN_PGEN_HPP_
