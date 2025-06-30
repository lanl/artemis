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
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle.hpp
//! \brief Contains HLLE Riemann solver for hydrodynamics
//!
//! Computes fluxes using the Harten-Lax-vanLeer-Einfeldt (HLLE) Riemann solver.  This
//! flux is very diffusive, especially for contacts, and so it is not recommended for
//! applications. However it is better than LLF. Einfeldt et al.(1991) prove it is
//! positively conservative (cannot return negative densities or pressure), so it is a
//! useful option when other approximate solvers fail and/or when extra dissipation is
//! needed.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
//! - Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
//! - A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//!   schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).
#ifndef UTILS_FLUXES_RIEMANN_HLLE_HPP_
#define UTILS_FLUXES_RIEMANN_HLLE_HPP_

// NOTE(PDM): The following is taken directly from the open-source Athena++/AthenaK
// software, and adapted for Parthenon/Artemis by PDM on 10/08/23

// C++ headers
#include <algorithm>
#include <cmath>

// Artemis headers
#include "artemis.hpp"
#include "radiation/moments/moments.hpp"
#include "utils/eos/eos.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::hlle, ...>
//! \brief The HLLE Riemann solver for ideal gas/dust hydrodynamics
template <Fluid FLUID_TYPE, Closure CTYPE>
struct RiemannSolver<RSolver::hlle, FLUID_TYPE, CTYPE,
                     std::enable_if_t<FLUID_TYPE != Fluid::radiation>> {
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void operator()(const EOS &eos, const Real c, const Real chat,
                                         parthenon::team_mbr_t const &member, const int b,
                                         const int k, const int j, const int il,
                                         const int iu, const int dir,
                                         const parthenon::ScratchPad2D<Real> &wl,
                                         const parthenon::ScratchPad2D<Real> &wr,
                                         const V1 &p, const V2 &q, const V3 &vf) const {
    using TE = parthenon::TopologicalElement;
    // Check sensibility of flux direction
    PARTHENON_REQUIRE(dir > 0 && dir <= 3, "Invalid flux direction!");
    [[maybe_unused]] auto fdir = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);

    // TODO(BRR) temporary
    const Real gm1 = eos.GruneisenParamFromDensityTemperature(Null<Real>(), Null<Real>());

    // Obtain number of species
    int nspecies = Null<int>();
    if constexpr (FLUID_TYPE == Fluid::gas) {
      nspecies = q.GetSize(b, gas::cons::density());
    } else if constexpr (FLUID_TYPE == Fluid::dust) {
      nspecies = q.GetSize(b, dust::cons::density());
    }

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

            // Compute Roe-averaged state
            Real sqrtdl = sqrt(wl_idn);
            Real sqrtdr = sqrt(wr_idn);
            Real isdlpdr = 1.0 / (sqrtdl + sqrtdr);

            Real wroe_ivx = (sqrtdl * wl_ivx + sqrtdr * wr_ivx) * isdlpdr;
            Real wroe_ivy = (sqrtdl * wl_ivy + sqrtdr * wr_ivy) * isdlpdr;
            Real wroe_ivz = (sqrtdl * wl_ivz + sqrtdr * wr_ivz) * isdlpdr;

            [[maybe_unused]] Real el = Null<Real>();
            [[maybe_unused]] Real er = Null<Real>();
            [[maybe_unused]] Real hroe = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for ideal gas
              // EOS, rather than E or P directly. sqrtdl*hl = sqrtdl*(el+pl)/dl =
              // (el+pl)/sqrtdl
              el = wl_ipr * igm1 +
                   0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
              er = wr_ipr * igm1 +
                   0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
              hroe = ((el + wl_ipr) / sqrtdl + (er + wr_ipr) / sqrtdr) * isdlpdr;
            }

            // Compute the L/R wave speeds based on L/R and Roe-averaged values
            Real qa = Null<Real>(), qb = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              qa = std::sqrt(gamma * wl_ipr / wl_idn);
              qb = std::sqrt(gamma * wr_ipr / wr_idn);
            }

            [[maybe_unused]] Real sl = Null<Real>();
            [[maybe_unused]] Real sr = Null<Real>();
            [[maybe_unused]] Real al = Null<Real>();
            [[maybe_unused]] Real ar = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              Real a = hroe - 0.5 * (SQR(wroe_ivx) + SQR(wroe_ivy) + SQR(wroe_ivz));
              a = (a < 0.0) ? 0.0 : sqrt(gm1 * a);
              Real sla = wroe_ivx - a;
              Real slb = wl_ivx - qa;
              Real sra = wroe_ivx + a;
              Real srb = wr_ivx + qb;
              sl = std::min(sla, slb);
              sr = std::max(sra, srb);
              al = (sla < slb) ? a : qa;
              ar = (sra > srb) ? a : qb;
            } else if constexpr (FLUID_TYPE == Fluid::dust) {
              sl = std::min(wroe_ivx, wl_ivx);
              sr = std::max(wroe_ivx, wr_ivx);
            }

            // following min/max set to TINY_NUMBER to fix bug found in converging
            // supersonic flow
            Real bp = (sr > 0.0) ? sr : 1.0e-20;
            Real bm = (sl < 0.0) ? sl : -1.0e-20;

            // Compute L/R fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
            qa = wl_ivx - bm;
            qb = wr_ivx - bp;

            Real fl_d = wl_idn * qa;
            Real fr_d = wr_idn * qb;

            Real fl_mx = wl_idn * wl_ivx * qa;
            Real fr_mx = wr_idn * wr_ivx * qb;

            Real fl_my = wl_idn * wl_ivy * qa;
            Real fr_my = wr_idn * wr_ivy * qb;

            Real fl_mz = wl_idn * wl_ivz * qa;
            Real fr_mz = wr_idn * wr_ivz * qb;

            [[maybe_unused]] Real fl_e = Null<Real>();
            [[maybe_unused]] Real fr_e = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              fl_e = el * qa + wl_ipr * wl_ivx;
              fr_e = er * qb + wr_ipr * wr_ivx;
            }

            // Set an approximate interface pressure for coordinate source terms and
            // pressure contribution to flux.
            qa = 0.0;
            if (bp != bm) qa = 0.5 * (bp + bm) / (bp - bm);
            if constexpr (FLUID_TYPE == Fluid::gas) {
              p.flux(b, dir, IPR, k, j, i) =
                  0.5 * (wl_ipr + wr_ipr) + qa * (wl_ipr - wr_ipr);
            }

            // Compute the HLLE flux at interface. Formulae below equivalent to
            // Toro eq. 10.20, or Einfeldt et al. (1991) eq. 4.4b
            const Real frho = 0.5 * (fl_d + fr_d) + qa * (fl_d - fr_d);
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = 0.5 * (fl_mx + fr_mx) + qa * (fl_mx - fr_mx);
            q.flux(b, dir, ivy, k, j, i) = 0.5 * (fl_my + fr_my) + qa * (fl_my - fr_my);
            q.flux(b, dir, ivz, k, j, i) = 0.5 * (fl_mz + fr_mz) + qa * (fl_mz - fr_mz);
            if constexpr (FLUID_TYPE == Fluid::gas) {
              q.flux(b, dir, IEN, k, j, i) = 0.5 * (fl_e + fr_e) + qa * (fl_e - fr_e);

              // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
              q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
              vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
            }
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::hlle, ...>
//! \brief The HLLE Riemann solver for radiation
template <Fluid FLUID_TYPE, Closure CTYPE>
struct RiemannSolver<RSolver::hlle, FLUID_TYPE, CTYPE,
                     std::enable_if_t<FLUID_TYPE == Fluid::radiation>> {
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void operator()(const EOS &eos, const Real c, const Real chat,
                                         parthenon::team_mbr_t const &member, const int b,
                                         const int k, const int j, const int il,
                                         const int iu, const int dir,
                                         const parthenon::ScratchPad2D<Real> &wl,
                                         const parthenon::ScratchPad2D<Real> &wr,
                                         const V1 &p, const V2 &q, const V3 &vf) const {
    using TE = parthenon::TopologicalElement;
    // Check sensibility of flux direction
    PARTHENON_REQUIRE(dir > 0 && dir <= 3, "Invalid flux direction!");
    // Obtain number of species
    const int nspecies = q.GetSize(b, rad::cons::energy());

    for (int n = 0; n < nspecies; ++n) {
      const int IDN = n;
      const int ivx = nspecies + (n * 3) + ((dir - 1));
      const int ivy = nspecies + (n * 3) + ((dir - 1) + 1) % 3;
      const int ivz = nspecies + (n * 3) + ((dir - 1) + 2) % 3;
      const int IPR = nspecies * 4 + n;

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

            // Compute reduced flux magnitude and limit
            Real fl = std::sqrt(SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
            Real fr = std::sqrt(SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
            const Real nlx = wl_ivx / (fl + Fuzz<Real>());
            const Real nrx = wr_ivx / (fr + Fuzz<Real>());
            fl = std::min(1.0, fl);
            fr = std::min(1.0, fr);

            // Wave speeds
            const Real chil = Moments::EddingtonFactor<CTYPE>(fl);
            const Real chir = Moments::EddingtonFactor<CTYPE>(fr);
            const auto [sla, slb] = Moments::WaveSpeed<CTYPE>(nlx, fl);
            const auto [sra, srb] = Moments::WaveSpeed<CTYPE>(nrx, fr);
            const Real sl = std::min(sla, slb);
            const Real sr = std::max(sra, srb);

            // Scales
            const Real pscalel = chat * c * 0.5 * (1.0 - chil);
            const Real pscaler = chat * c * 0.5 * (1.0 - chir);
            const Real wl_ipr = pscalel * wl_idn;
            const Real wr_ipr = pscaler * wr_idn;
            const Real norml = fl * fl;
            const Real normr = fr * fr;
            const Real scalel = c * 0.5 * (3. * chil - 1.);
            const Real scaler = c * 0.5 * (3. * chir - 1.);

            // following min/max set to TINY_NUMBER to fix bug found in converging
            // supersonic flow
            const Real bp = (sr > 0.0) ? sr : 1.0e-20;
            const Real bm = (sl < 0.0) ? sl : -1.0e-20;

            // Compute L/R fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
            Real qa = chat * (wl_ivx - bm);
            Real qb = chat * (wr_ivx - bp);
            const Real fl_d = wl_idn * qa;
            const Real fr_d = wr_idn * qb;
            const Real fl_mx = scalel * wl_idn * qa * (wl_ivx / (norml + Fuzz<Real>()));
            const Real fr_mx = scaler * wr_idn * qb * (wr_ivx / (normr + Fuzz<Real>()));
            const Real fl_my = scalel * wl_idn * qa * (wl_ivy / (norml + Fuzz<Real>()));
            const Real fr_my = scaler * wr_idn * qb * (wr_ivy / (normr + Fuzz<Real>()));
            const Real fl_mz = scalel * wl_idn * qa * (wl_ivz / (norml + Fuzz<Real>()));
            const Real fr_mz = scaler * wr_idn * qb * (wr_ivz / (normr + Fuzz<Real>()));

            // Set an approximate interface pressure for coordinate source terms and
            // pressure contribution to flux.
            qa = 0.0;
            if (bp != bm) qa = 0.5 * (bp + bm) / (bp - bm);
            p.flux(b, dir, IPR, k, j, i) =
                0.5 * (wl_ipr + wr_ipr) + qa * (wl_ipr - wr_ipr);

            // Compute the HLLE flux at interface. Formulae below equivalent to
            // Toro eq. 10.20, or Einfeldt et al. (1991) eq. 4.4b
            const Real frho = 0.5 * (fl_d + fr_d) + qa * (fl_d - fr_d);
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = 0.5 * (fl_mx + fr_mx) + qa * (fl_mx - fr_mx);
            q.flux(b, dir, ivy, k, j, i) = 0.5 * (fl_my + fr_my) + qa * (fl_my - fr_my);
            q.flux(b, dir, ivz, k, j, i) = 0.5 * (fl_mz + fr_mz) + qa * (fl_mz - fr_mz);
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_HLLE_HPP_
