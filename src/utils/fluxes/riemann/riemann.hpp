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
#ifndef ARTEMIS_UTILS_FLUXES_RIEMANN_RIEMANN_HPP_
#define ARTEMIS_UTILS_FLUXES_RIEMANN_RIEMANN_HPP_

// Artemis headers
#include "artemis.hpp"
#include "utils/eos/eos.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class  TaskStatus ArtemisUtils::RiemannSolver
//! \brief Class that wraps templated Riemann solver call to allow for partial
//!        template specialization.
template <RSolver R, Fluid FLUID_TYPE>
class RiemannSolver {
 public:
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void
  solve(const EOS &eos, parthenon::team_mbr_t const &member, const int b, const int k,
        const int j, const int il, const int iu, const int dir,
        const parthenon::ScratchPad2D<Real> &wl, const parthenon::ScratchPad2D<Real> &wr,
        const V1 &p, const V2 &q) const {
    PARTHENON_FAIL("No default implementation!");
  }
};

} // namespace ArtemisUtils

// Partial specializations
#include "hllc.hpp"
#include "hlle.hpp"
#include "llf.hpp"

#endif // ARTEMIS_UTILS_FLUXES_RIEMANN_RIEMANN_HPP_
