//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef SELF_GRAVITY_SELF_GRAVITY_HPP_
#define SELF_GRAVITY_SELF_GRAVITY_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "utils/units.hpp"

namespace SelfGravity {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            const ArtemisUtils::Constants &constants,
                                            const Packages_t &packages);

template <Coordinates GEOM>
void FillPoissonRHS(MeshData<Real> *md);

template <Coordinates GEOM>
TaskStatus SelfGravity(MeshData<Real> *md, const Real time, const Real dt);

void SolvePoisson(TaskCollection &tc, Mesh *pmesh, const Real time, const int stage);

} // namespace SelfGravity

#endif // SELF_GRAVITY_SELF_GRAVITY_HPP_
