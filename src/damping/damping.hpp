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
#ifndef DAMPING_DAMPING_HPP_
#define DAMPING_DAMPING_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"

using namespace parthenon::package::prelude;

namespace Damping {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
TaskStatus DampingSource(MeshData<Real> *md, const Real time, const Real dt);

} // namespace Damping

#endif // DAMPING_DAMPING_HPP_
