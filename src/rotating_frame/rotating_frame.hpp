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
#ifndef ROTATING_FRAME_ROTATING_FRAME_HPP_
#define ROTATING_FRAME_ROTATING_FRAME_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"

using namespace parthenon::package::prelude;

namespace RotatingFrame {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus RotatingFrameForce(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION std::array<Real, 3> RotationVelocity(const std::array<Real, 3> &xv,
                                                            const Real omf) {
  // Return the rotational velocity

  // Empty constructor to get access to conversion routine
  geometry::Coords<GEOM> coords;

  const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

  const Real vp = omf * xcyl[0];

  std::array<Real, 3> vrot{ex1[1] * vp, ex2[1] * vp, ex3[1] * vp};
  return vrot;
}

} // namespace RotatingFrame

#endif // ROTATING_FRAME_ROTATING_FRAME_HPP_
