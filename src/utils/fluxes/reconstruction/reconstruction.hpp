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
#include "utils/eos/eos.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class  TaskStatus ArtemisUtils::Reconstruction
//! \brief Class that wraps templated Reconstruction method to allow for partial
template <ReconstructionMethod R, CoordinateDirection DIR, Coordinates GEOM>
struct Reconstruction {
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
             const int b, const int k, const int j, const int il, const int iu,
             const V1 &q, const V2 &vg, parthenon::ScratchPad2D<Real> &ql,
             parthenon::ScratchPad2D<Real> &qr) const {
    PARTHENON_FAIL("No default implementation!");
  }
};

template <Coordinates GEOM, ReconstructionMethod R>
struct ReconGradient {
  template <typename V>
  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  operator()(const geometry::CoordParams &cpars, const V &q,
             const std::array<Real, 3> &dx, const int multi_d, const int three_d,
             const int b, const int n, const int k, const int j, const int i) const {
    PARTHENON_FAIL("No default implementation!");
  }
};

//----------------------------------------------------------------------------------------
//! \class  TaskStatus ArtemisUtils::post_recon
//! \brief Utility to apply floors, make thermodynamically consistent, or zero radiation
//! fluxes when near ~round-off
template <Fluid F, typename V, typename VC>
KOKKOS_INLINE_FUNCTION void
post_recon(const EOS &eos, const Real dfloor, const Real siefloor, const bool do_mhd,
           parthenon::team_mbr_t const &member, const int dir, const int b, const int k,
           const int j, const int il, const int iu, const V &q, const VC &qc,
           parthenon::ScratchPad2D<Real> &ql, parthenon::ScratchPad2D<Real> &qr) {
  using TE = parthenon::TopologicalElement;
  if constexpr (F == Fluid::radiation) {
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
  } else if constexpr (F == Fluid::gas) {
    // Make sure the reconstructed states make sense
    const int nspecies = q.GetSize(b, gas::prim::density());
    for (int n = 0; n < nspecies; ++n) {
      const int IDN = n;
      const int IPR = nspecies * 4 + n;
      const int ISE = nspecies * 5 + n;
      const int IBL = nspecies * 6 + n;
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
                                 const int ipl = i + (dir == 1);
                                 Real &dL = ql(IDN, ipl);
                                 Real &pL = ql(IPR, ipl);
                                 Real &eL = ql(ISE, ipl);
                                 Real &bL = ql(IBL, ipl);
                                 Real &dR = qr(IDN, i);
                                 Real &pR = qr(IPR, i);
                                 Real &eR = qr(ISE, i);
                                 Real &bR = qr(IBL, i);

                                 // Floor everything
                                 dL = std::max(dL, dfloor);
                                 dR = std::max(dR, dfloor);
                                 eL = std::max(eL, siefloor);
                                 eR = std::max(eR, siefloor);

                                 // Only correct these if something is wrong
                                 if ((pL <= 0.0) || (bL <= 0.0)) {
                                   pL = eos.PressureFromDensityInternalEnergy(dL, eL);
                                   bL = eos.BulkModulusFromDensityInternalEnergy(dL, eL);
                                 }
                                 if ((pR <= 0.0) || (bR <= 0.0)) {
                                   pR = eos.PressureFromDensityInternalEnergy(dR, eR);
                                   bR = eos.BulkModulusFromDensityInternalEnergy(dR, eR);
                                 }
                               });
    }
    // Replace the reconstructed B on the face in direction dir with the cons face value
    if (do_mhd) {
      const int n = nspecies * 7 + dir - 1;
      // or maybe TE fd = TE::F1 + (dir-1)?
      TE fd = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            const int ipl = i + (dir == 1);
            ql(n, ipl) = qc(b, fd, field::face::B(), k + (dir == 3), j + (dir == 2), ipl);
            qr(n, i) = qc(b, fd, field::face::B(), k, j, i);
          });
    }
  }
}

} // namespace ArtemisUtils

// Partial specializations
#include "pcm.hpp"
#include "plm.hpp"
#include "ppm.hpp"
#include "wenomz.hpp"
#include "wenoz.hpp"

#endif // ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_
