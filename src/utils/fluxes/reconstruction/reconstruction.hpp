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
  template <typename V>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const int b, const int k, const int j,
             const int il, const int iu, const V &q, parthenon::ScratchPad2D<Real> &ql,
             parthenon::ScratchPad2D<Real> &qr) const {
    PARTHENON_FAIL("No default implementation!");
  }
};

template <Fluid FLUID_TYPE, typename V>
KOKKOS_INLINE_FUNCTION void
correct_recon(parthenon::team_mbr_t const &member, const int dir, const int b,
              const int k, const int j, const int il, const int iu, const V &q,
              parthenon::ScratchPad2D<Real> &ql, parthenon::ScratchPad2D<Real> &qr) {

  if constexpr (is_grey<FLUID_TYPE>()) {
    constexpr int nvar = 5;
    const int nspecies = q.GetMaxNumberOfVars() / nvar;
    for (int n = 0; n < nspecies; ++n) {
      const int IER = n;
      const int IFX = nspecies + (n * 3) + ((dir - 1));
      const int IFY = nspecies + (n * 3) + ((dir - 1) + 1) % 3;
      const int IFZ = nspecies + (n * 3) + ((dir - 1) + 2) % 3;
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
                                 const int ipl = i + (dir == 1);
                                 Real &er_l = ql(IER, ipl);
                                 Real &fx_l = ql(IFX, ipl);
                                 Real &fy_l = ql(IFY, ipl);
                                 Real &fz_l = ql(IFZ, ipl);
                                 Real &er_r = qr(IER, i);
                                 Real &fx_r = qr(IFX, i);
                                 Real &fy_r = qr(IFY, i);
                                 Real &fz_r = qr(IFZ, i);

                                 // NOTE(AMD): This doesn't seem to work for rad fluid
                                 // do pcm?
                                 // const Real f2l = SQR(fx_l) + SQR(fy_l) + SQR(fz_l);
                                 // const Real f2r = SQR(fx_r) + SQR(fy_r) + SQR(fz_r);
                                 // const int pcm = (f2l > 1.0) | (er_l <= 0.0) | (f2r
                                 // > 1.0) | (er_r <= 0.0); er_l = (1 - pcm) * er_l + pcm
                                 // * q(b, IER, k, j, ipl); fx_l = (1 - pcm) * fx_l + pcm
                                 // * q(b, IFX, k, j, ipl); fy_l = (1 - pcm) * fy_l + pcm
                                 // * q(b, IFY, k, j, ipl); fz_l = (1 - pcm) * fz_l + pcm
                                 // * q(b, IFZ, k, j, ipl); er_r = (1 - pcm) * er_r + pcm
                                 // * q(b, IER, k, j, i); fx_r = (1 - pcm) * fx_r + pcm *
                                 // q(b, IFX, k, j, i); fy_r = (1 - pcm) * fy_r + pcm *
                                 // q(b, IFY, k, j, i); fz_r = (1 - pcm) * fz_r + pcm *
                                 // q(b, IFZ, k, j, i);

                                 // note f is dimensionless
                                 fx_l = (std::abs(fx_l) <= 1e-20) ? 0.0 : fx_l;
                                 fy_l = (std::abs(fy_l) <= 1e-20) ? 0.0 : fy_l;
                                 fz_l = (std::abs(fz_l) <= 1e-20) ? 0.0 : fz_l;
                                 fx_r = (std::abs(fx_r) <= 1e-20) ? 0.0 : fx_r;
                                 fy_r = (std::abs(fy_r) <= 1e-20) ? 0.0 : fy_r;
                                 fz_r = (std::abs(fz_r) <= 1e-20) ? 0.0 : fz_r;
                               });
    }
    // add barier inside the constexpr
    member.team_barrier();
  }
}

} // namespace ArtemisUtils

// Partial specializations
#include "pcm.hpp"
#include "plm.hpp"
#include "ppm.hpp"

#endif // ARTEMIS_UTILS_FLUXES_RECONSTRUCTION_RECONSTRUCTION_HPP_
