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
#ifndef DUST_DUST_HPP_
#define DUST_DUST_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

using namespace parthenon::package::prelude;

namespace Dust {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md);

TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm);
TaskStatus FluxSource(MeshData<Real> *md, const Real dt);

void AddHistory(Coordinates coords, Params &params);

// OperatorSplit tasks
TaskListStatus OperatorSplitDust(Mesh *pm, parthenon::SimTime &tm);

template <Coordinates GEOM>
TaskStatus CoagulationOneStep(MeshData<Real> *md, const Real time, const Real dt);

} // namespace Dust

#endif // DUST_DUST_HPP_
