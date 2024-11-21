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
#ifndef ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_
#define ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_

#include "artemis.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class  TaskStatus ArtemisUtils::RiemannSolver
//! \brief Class that wraps templated Riemann solver call to allow for partial
//!        template specialization.
template <ReconstructionMethod R, CoordinateDirection DIR, Coordinates GEOM>
class Reconstruction {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql,
                                    parthenon::ScratchPad2D<Real> &qr) const {
    PARTHENON_FAIL("No default implementation!");
  }
};

} // namespace ArtemisUtils

// Partial specializations
#include "pcm.hpp"
#include "plm.hpp"
#include "ppm.hpp"

#endif // ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_
