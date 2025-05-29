//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

// Artemis includes
#include "artemis.hpp"
#include "derived/fill_derived.hpp"
#include "radiation/imc/imc.hpp"

// Jaybenne includes
#include "jaybenne.hpp"

using namespace parthenon::driver::prelude;

namespace IMC {

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus IMC::JaybenneIMC
//! \brief Executes thermal IMC transport (Jaybenne) and syncs updated fields
template <Coordinates GEOM>
TaskListStatus JaybenneIMC(Mesh *pmesh, const Real time, const Real dt) {
  auto status = jaybenne::RadiationStep(pmesh, time, dt).Execute();
  if (status != TaskListStatus::complete) return status;
  status = ArtemisDerived::SyncFields<GEOM>(pmesh, time, dt).Execute();
  return status;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef Mesh M;
template TaskListStatus JaybenneIMC<G::cartesian>(M *pm, const Real t, const Real dt);
template TaskListStatus JaybenneIMC<G::cylindrical>(M *pm, const Real t, const Real dt);
template TaskListStatus JaybenneIMC<G::spherical1D>(M *pm, const Real t, const Real dt);
template TaskListStatus JaybenneIMC<G::spherical2D>(M *pm, const Real t, const Real dt);
template TaskListStatus JaybenneIMC<G::spherical3D>(M *pm, const Real t, const Real dt);
template TaskListStatus JaybenneIMC<G::axisymmetric>(M *pm, const Real t, const Real dt);

} // namespace IMC
