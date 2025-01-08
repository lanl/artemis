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
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf.hpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver, also known as Rusanov's
//! method, for hydrodynamics.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
#ifndef UTILS_FLUXES_RIEMANN_LLF_HPP_
#define UTILS_FLUXES_RIEMANN_LLF_HPP_

// NOTE(PDM): The following is taken directly from the open-source Athena++/AthenaK
// software, and adapted for Parthenon/Artemis by PDM on 10/08/23

// C++ headers
#include <algorithm>
#include <cmath>

// Artemis headers
#include "artemis.hpp"
#include "radiation/moment/radiation.hpp"
#include "utils/eos/eos.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::llf, ...>
//! \brief The LLF Riemann solver for ideal gas hydrodynamics
template <Fluid FLUID_TYPE>
class RiemannSolver<RSolver::llf, FLUID_TYPE> {
 public:
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void
  solve(const EOS &eos, const Real c, const Real chat,
        parthenon::team_mbr_t const &member, const int b, const int k, const int j,
        const int il, const int iu, const int dir,
        const parthenon::ScratchPad2D<Real> &wl, const parthenon::ScratchPad2D<Real> &wr,
        const V1 &p, const V2 &q, const V3 &vf) const {
    using TE = parthenon::TopologicalElement;
    // Check sensibility of flux direction
    PARTHENON_REQUIRE(dir > 0 && dir <= 3, "Invalid flux direction!");
    [[maybe_unused]] auto fdir = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);

    // TODO(BRR) temporary
    const Real gm1 = eos.GruneisenParamFromDensityTemperature(Null<Real>(), Null<Real>());

    // Obtain number of species
    int nvar = Null<int>();
    if constexpr (FLUID_TYPE == Fluid::gas) {
      nvar = 6;
    } else if constexpr (FLUID_TYPE == Fluid::dust) {
      nvar = 4;
    } else if constexpr (is_grey<FLUID_TYPE>()) {
      nvar = 5;
    }
    const int nspecies = p.GetMaxNumberOfVars() / nvar;

    for (int n = 0; n < nspecies; ++n) {
      const int IDN = n;
      const int ivx = nspecies + (n * 3) + ((dir - 1));
      const int ivy = nspecies + (n * 3) + ((dir - 1) + 1) % 3;
      const int ivz = nspecies + (n * 3) + ((dir - 1) + 2) % 3;
      // Unused indices for dust hydrodynamics
      const int IPR = nspecies * 4 + n;
      const int ISE = nspecies * 5 + n;
      [[maybe_unused]] const int IEN = IPR;
      [[maybe_unused]] const int IEG = ISE;

      [[maybe_unused]] Real igm1 = Null<Real>();
      [[maybe_unused]] Real gamma = Null<Real>();
      if constexpr (FLUID_TYPE == Fluid::gas) {
        igm1 = 1.0 / gm1;
        gamma = gm1 + 1.0;
      }

      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            // Create local references for L/R states (helps compiler vectorize)
            Real &wl_idn = wl(IDN, i);
            Real &wl_ivx = wl(ivx, i);
            Real &wl_ivy = wl(ivy, i);
            Real &wl_ivz = wl(ivz, i);

            Real &wr_idn = wr(IDN, i);
            Real &wr_ivx = wr(ivx, i);
            Real &wr_ivy = wr(ivy, i);
            Real &wr_ivz = wr(ivz, i);

            [[maybe_unused]] Real wl_ipr = Null<Real>();
            [[maybe_unused]] Real wr_ipr = Null<Real>();
            [[maybe_unused]] Real wl_ise = Null<Real>();
            [[maybe_unused]] Real wr_ise = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              wl_ipr = wl(IPR, i);
              wl_ise = wl(ISE, i);
              wr_ipr = wr(IPR, i);
              wr_ise = wr(ISE, i);
            }

            Real scalel = 1.0;
            Real scaler = 1.0;
            Real qscale = 1.0;
            [[maybe_unused]] Real pscalel = Null<Real>();
            [[maybe_unused]] Real pscaler = Null<Real>();
            [[maybe_unused]] Real sl = Null<Real>();
            [[maybe_unused]] Real sr = Null<Real>();
            if constexpr (is_grey<FLUID_TYPE>()) {
              Real fl = std::sqrt(SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
              Real fr = std::sqrt(SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
              const Real nlx = wl_ivx / (fl + Fuzz<Real>());
              const Real nrx = wr_ivx / (fr + Fuzz<Real>());
              fl = std::min(1.0, fl);
              fr = std::min(1.0, fr);
              const Real chil = Radiation::EddingtonFactor<FLUID_TYPE>(fl);
              const Real chir = Radiation::EddingtonFactor<FLUID_TYPE>(fr);
              qscale = chat;
              const auto [sla, slb] = Radiation::WaveSpeed(nlx, fl);
              const auto [sra, srb] = Radiation::WaveSpeed(nrx, fr);
              sl = std::min(sla, slb);
              sr = std::max(sra, srb);
              pscalel = chat * c * 0.5 * (1.0 - chil);
              pscaler = chat * c * 0.5 * (1.0 - chir);
              scalel = c * 0.5 * (3. * chil - 1.) / (fl * fl + Fuzz<Real>());
              scaler = c * 0.5 * (3. * chir - 1.) / (fr * fr + Fuzz<Real>());
            }

            // Compute sum of L/R fluxes
            Real qa = qscale * wl_idn * wl_ivx;
            Real qb = qscale * wr_idn * wr_ivx;
            Real fsum_d = qa + qb;
            Real fsum_mx = qa * scalel * wl_ivx + qb * scaler * wr_ivx;
            Real fsum_my = qa * scalel * wl_ivy + qb * scaler * wr_ivy;
            Real fsum_mz = qa * scalel * wl_ivz + qb * scaler * wr_ivz;

            [[maybe_unused]] Real el = Null<Real>();
            [[maybe_unused]] Real er = Null<Real>();
            [[maybe_unused]] Real fsum_e = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              el = wl_ipr * igm1 +
                   0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
              er = wr_ipr * igm1 +
                   0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
              fsum_e = (el + wl_ipr) * wl_ivx + (er + wr_ipr) * wr_ivx;
            }

            // Compute max wave speed in L/R states (see Toro eq. 10.43)
            Real a = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              qa = std::sqrt(gamma * wl_ipr / wl_idn);
              qb = std::sqrt(gamma * wr_ipr / wr_idn);
              a = std::max((std::abs(wl_ivx) + qa), (std::abs(wr_ivx) + qb));
            } else if constexpr (FLUID_TYPE == Fluid::dust) {
              a = std::max(std::abs(wl_ivx), std::abs(wr_ivx));
            } else if constexpr (is_grey<FLUID_TYPE>()) {
              a = qscale * std::max(sl, sr);
            }

            // Compute difference in L/R states dU, multiplied by max wave speed
            Real du_d = a * (wr_idn - wl_idn);
            Real du_mx = a * (wr_idn * scalel * wr_ivx - wl_idn * scaler * wl_ivx);
            Real du_my = a * (wr_idn * scalel * wr_ivy - wl_idn * scaler * wl_ivy);
            Real du_mz = a * (wr_idn * scalel * wr_ivz - wl_idn * scaler * wl_ivz);

            [[maybe_unused]] Real du_e = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              du_e = a * (er - el);
            }

            // Set an approximate interface pressure for coordinate source terms
            if constexpr (FLUID_TYPE == Fluid::gas) {
              p.flux(b, dir, IPR, k, j, i) = 0.5 * (wl_ipr + wr_ipr);
            } else if constexpr (is_grey<FLUID_TYPE>()) {
              p.flux(b, dir, IPR, k, j, i) = 0.5 * (wl_idn * pscalel + wr_idn * pscaler);
            }

            // Compute the LLF flux at interface
            const Real frho = 0.5 * (fsum_d - du_d);
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = 0.5 * (fsum_mx - du_mx);
            q.flux(b, dir, ivy, k, j, i) = 0.5 * (fsum_my - du_my);
            q.flux(b, dir, ivz, k, j, i) = 0.5 * (fsum_mz - du_mz);
            if constexpr (FLUID_TYPE == Fluid::gas) {
              q.flux(b, dir, IEN, k, j, i) = 0.5 * (fsum_e - du_e);

              // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
              q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
              vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
            }
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_LLF_HPP_
