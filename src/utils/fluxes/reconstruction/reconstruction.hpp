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
#ifndef ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_
#define ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_

#include "artemis.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class  TaskStatus ArtemisUtils::Reconstruction
//! \brief Class that wraps templated Reconstruction method to allow for partial
template <ReconstructionMethod R, CoordinateDirection DIR, Coordinates GEOM>
class Reconstruction {
  template <typename V>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const int b, const int k, const int j,
             const int il, const int iu, const V &q, parthenon::ScratchPad2D<Real> &ql,
             parthenon::ScratchPad2D<Real> &qr) const {
    PARTHENON_FAIL("No default implementation!");
  }
};

//----------------------------------------------------------------------------------------
//! \class  TaskStatus ArtemisUtils::correct_recon
//! \brief Utility to zero radiation fluxes when near ~round-off
template <typename V>
KOKKOS_INLINE_FUNCTION void
correct_recon(parthenon::team_mbr_t const &member, const int dir, const int b,
              const int k, const int j, const int il, const int iu, const V &q,
              parthenon::ScratchPad2D<Real> &ql, parthenon::ScratchPad2D<Real> &qr) {
  const int nspecies = q.GetSize(b, rad::prim::energy());
  for (int n = 0; n < nspecies; ++n) {
    const int IFX = nspecies + (n * 3) + ((dir - 1));
    const int IFY = nspecies + (n * 3) + ((dir - 1) + 1) % 3;
    const int IFZ = nspecies + (n * 3) + ((dir - 1) + 2) % 3;
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                             [&](const int i) {
                               const int ipl = i + (dir == 1);
                               ql(IFX, ipl) *= (std::abs(ql(IFX, ipl)) > 1.0e-20);
                               ql(IFY, ipl) *= (std::abs(ql(IFY, ipl)) > 1.0e-20);
                               ql(IFZ, ipl) *= (std::abs(ql(IFZ, ipl)) > 1.0e-20);
                               qr(IFX, i) *= (std::abs(qr(IFX, i)) > 1.0e-20);
                               qr(IFY, i) *= (std::abs(qr(IFY, i)) > 1.0e-20);
                               qr(IFZ, i) *= (std::abs(qr(IFZ, i)) > 1.0e-20);
                             });
  }
}

} // namespace ArtemisUtils

// Partial specializations
#include "pcm.hpp"
#include "plm.hpp"
#include "ppm.hpp"

#endif // ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_
