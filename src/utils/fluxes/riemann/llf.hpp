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
#include "radiation/moments/moments.hpp"
#include "utils/eos/eos.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::llf, ...>
//! \brief The LLF Riemann solver for ideal gas/dust hydrodynamics
template <Fluid FLUID_TYPE, Closure CTYPE>
struct RiemannSolver<RSolver::llf, FLUID_TYPE, CTYPE,
                     std::enable_if_t<FLUID_TYPE != Fluid::radiation>> {
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void
  operator()(const EOS &eos, const Real c, const Real chat, const Real mu0, bool do_mhd,
             parthenon::team_mbr_t const &member, const int b, const int k, const int j,
             const int il, const int iu, const int dir,
             const parthenon::ScratchPad2D<Real> &wl,
             const parthenon::ScratchPad2D<Real> &wr, const V1 &p, const V2 &q,
             const V3 &vf) const {

    using TE = parthenon::TopologicalElement;
    // Check sensibility of flux direction
    PARTHENON_REQUIRE(dir > 0 && dir <= 3, "Invalid flux direction!");
    [[maybe_unused]] auto fdir = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);

    // Obtain number of species
    int nspecies = Null<int>();
    if constexpr (FLUID_TYPE == Fluid::gas) {
      nspecies = q.GetSize(b, gas::cons::density());
    } else if constexpr (FLUID_TYPE == Fluid::dust) {
      nspecies = q.GetSize(b, dust::cons::density());
    }

    [[maybe_unused]] const int IBX = nspecies * 7 + (dir - 1);
    [[maybe_unused]] const int IBY = nspecies * 7 + ((dir - 1) + 1) % 3;
    [[maybe_unused]] const int IBZ = nspecies * 7 + ((dir - 1) + 2) % 3;
    [[maybe_unused]] const int IBM = nspecies * 7 + 3;
    [[maybe_unused]] const int IBXG = dir - 1;
    [[maybe_unused]] const int IBYG = dir % 3;
    [[maybe_unused]] const int IBZG = (dir + 1) % 3;
    for (int n = 0; n < nspecies; ++n) {
      do_mhd &= (n == 0);
      const int IDN = n;
      const int ivx = nspecies + (n * 3) + ((dir - 1));
      const int ivy = nspecies + (n * 3) + ((dir - 1) + 1) % 3;
      const int ivz = nspecies + (n * 3) + ((dir - 1) + 2) % 3;
      // Unused indices for dust hydrodynamics
      const int IPR = nspecies * 4 + n;
      const int ISE = nspecies * 5 + n;
      const int IBL = nspecies * 6 + n;
      [[maybe_unused]] const int IEN = IPR;
      [[maybe_unused]] const int IEG = ISE;

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
            [[maybe_unused]] Real wl_ibl = Null<Real>();
            [[maybe_unused]] Real wr_ibl = Null<Real>();
            // note these are intentionally 0 and not Null
            [[maybe_unused]] Real wl_ibx = 0.0;
            [[maybe_unused]] Real wl_iby = 0.0;
            [[maybe_unused]] Real wl_ibz = 0.0;
            [[maybe_unused]] Real wr_ibx = 0.0;
            [[maybe_unused]] Real wr_iby = 0.0;
            [[maybe_unused]] Real wr_ibz = 0.0;
            if constexpr (FLUID_TYPE == Fluid::gas) {
              wl_ipr = wl(IPR, i);
              wl_ise = wl(ISE, i);
              wl_ibl = wl(IBL, i);
              wr_ipr = wr(IPR, i);
              wr_ise = wr(ISE, i);
              wr_ibl = wr(IBL, i);
              if (do_mhd) {
                wl_ibx = wl(IBX, i);
                wl_iby = wl(IBY, i);
                wl_ibz = wl(IBZ, i);
                wr_ibx = wr(IBX, i);
                wr_iby = wr(IBY, i);
                wr_ibz = wr(IBZ, i);
              }
            }

            // Compute sum of L/R fluxes
            Real qa = wl_idn * wl_ivx;
            Real qb = wr_idn * wr_ivx;
            Real fsum_d = qa + qb;
            Real fsum_mx = qa * wl_ivx + qb * wr_ivx;
            Real fsum_my = qa * wl_ivy + qb * wr_ivy;
            Real fsum_mz = qa * wl_ivz + qb * wr_ivz;

            [[maybe_unused]] Real el = Null<Real>();
            [[maybe_unused]] Real er = Null<Real>();
            [[maybe_unused]] Real fsum_e = Null<Real>();
            [[maybe_unused]] Real pbl = 0.0;
            [[maybe_unused]] Real pbr = 0.0;
            [[maybe_unused]] Real vdBl = 0.0;
            [[maybe_unused]] Real vdBr = 0.0;
            [[maybe_unused]] Real fsum_by = 0.0;
            [[maybe_unused]] Real fsum_bz = 0.0;

            if constexpr (FLUID_TYPE == Fluid::gas) {
              el = wl_ise * wl_idn +
                   0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
              er = wr_ise * wr_idn +
                   0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
              fsum_e = (el + wl_ipr) * wl_ivx + (er + wr_ipr) * wr_ivx;
              if (do_mhd) {
                pbl = MHD::MagneticEnergyDensity(wl_ibx, wl_iby, wl_ibz, mu0);
                pbr = MHD::MagneticEnergyDensity(wr_ibx, wr_iby, wr_ibz, mu0);
                p.flux(b, dir, IBM, k, j, i) = 0.5 * (pbl + pbr);
                fsum_mx -= (SQR(wl_ibx) + SQR(wr_ibx)) / mu0;
                fsum_my -= (wl_ibx * wl_iby + wr_ibx * wr_iby) / mu0;
                fsum_mz -= (wl_ibx * wl_ibz + wr_ibx * wr_ibz) / mu0;
                vdBl = wl_ivx * wl_ibx + wl_ivy * wl_iby + wl_ivz * wl_ibz;
                vdBr = wr_ivx * wr_ibx + wr_ivy * wr_iby + wr_ivz * wr_ibz;
                el += pbl;
                er += pbr;
                // E includes magnetic energy and the total pressure contains an equal
                // magnetic-pressure contribution, so each state contributes 2 P_B v_x.
                fsum_e += 2.0 * (pbl * wl_ivx + pbr * wr_ivx) -
                           (wl_ibx * vdBl + wr_ibx * vdBr) / mu0;
                fsum_by = (wl_ivx * wl_iby - wl_ivy * wl_ibx) +
                          (wr_ivx * wr_iby - wr_ivy * wr_ibx);
                fsum_bz = (wl_ivx * wl_ibz - wl_ivz * wl_ibx) +
                          (wr_ivx * wr_ibz - wr_ivz * wr_ibx);
              }
            }

            // Compute max wave speed in L/R states (see Toro eq. 10.43)
            Real a = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              if (do_mhd) {
                qa = MHD::FastMagnetosonicSpeed(wl_ibl, wl_idn, wl_ibx, wl_iby, wl_ibz,
                                                mu0);
                qb = MHD::FastMagnetosonicSpeed(wr_ibl, wr_idn, wr_ibx, wr_iby, wr_ibz,
                                                mu0);
              } else {
                qa = std::sqrt(wl_ibl / wl_idn);
                qb = std::sqrt(wr_ibl / wr_idn);
              }
              a = std::max((std::abs(wl_ivx) + qa), (std::abs(wr_ivx) + qb));
            } else if constexpr (FLUID_TYPE == Fluid::dust) {
              a = std::max(std::abs(wl_ivx), std::abs(wr_ivx));
            }

            // Compute difference in L/R states dU, multiplied by max wave speed
            Real du_d = a * (wr_idn - wl_idn);
            Real du_mx = a * (wr_idn * wr_ivx - wl_idn * wl_ivx);
            Real du_my = a * (wr_idn * wr_ivy - wl_idn * wl_ivy);
            Real du_mz = a * (wr_idn * wr_ivz - wl_idn * wl_ivz);

            [[maybe_unused]] Real du_e = Null<Real>();
            [[maybe_unused]] Real du_by = 0.0;
            [[maybe_unused]] Real du_bz = 0.0;
            if constexpr (FLUID_TYPE == Fluid::gas) {
              du_e = a * (er - el);
              if (do_mhd) {
                du_by = a * (wr_iby - wl_iby);
                du_bz = a * (wr_ibz - wl_ibz);
              }
            }

            // Set an approximate interface pressure for coordinate source terms
            if constexpr (FLUID_TYPE == Fluid::gas) {
              p.flux(b, dir, IPR, k, j, i) = 0.5 * (wl_ipr + wr_ipr);
            }

            // Compute the LLF flux at interface
            const Real frho = 0.5 * (fsum_d - du_d);
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = 0.5 * (fsum_mx - du_mx);
            q.flux(b, dir, ivy, k, j, i) = 0.5 * (fsum_my - du_my);
            q.flux(b, dir, ivz, k, j, i) = 0.5 * (fsum_mz - du_mz);
            if constexpr (FLUID_TYPE == Fluid::gas) {
              if (do_mhd) {
                const Real fby = 0.5 * (fsum_by - du_by);
                const Real fbz = 0.5 * (fsum_bz - du_bz);
                p.flux(b, dir, field::cell::B(IBXG), k, j, i) = 0.0;
                p.flux(b, dir, field::cell::B(IBYG), k, j, i) = fby;
                p.flux(b, dir, field::cell::B(IBZG), k, j, i) = fbz;
              }
              q.flux(b, dir, IEN, k, j, i) = 0.5 * (fsum_e - du_e);

              // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
              q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
              vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
            }
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::llf, ...>
//! \brief The LLF Riemann solver for radiation
template <Fluid FLUID_TYPE, Closure CTYPE>
struct RiemannSolver<RSolver::llf, FLUID_TYPE, CTYPE,
                     std::enable_if_t<FLUID_TYPE == Fluid::radiation>> {
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void
  operator()(const EOS &eos, const Real c, const Real chat, const Real mu0,
             const bool do_mhd, parthenon::team_mbr_t const &member, const int b,
             const int k, const int j, const int il, const int iu, const int dir,
             const parthenon::ScratchPad2D<Real> &wl,
             const parthenon::ScratchPad2D<Real> &wr, const V1 &p, const V2 &q,
             const V3 &vf) const {
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
            const Real chil = Moments::ThriceEddingtonFactor<CTYPE>(fl);
            const Real chir = Moments::ThriceEddingtonFactor<CTYPE>(fr);
            const auto [sla, slb] = Moments::WaveSpeed<CTYPE>(nlx, fl);
            const auto [sra, srb] = Moments::WaveSpeed<CTYPE>(nrx, fr);
            const Real sl = std::min(sla, slb);
            const Real sr = std::max(sra, srb);

            // Scales
            const Real pscalel = chat * c * (3.0 - chil) / 6.0;
            const Real pscaler = chat * c * (3.0 - chir) / 6.0;
            const Real scalel = c * 0.5 * ((chil - 1.) / (fl * fl + Fuzz<Real>()));
            const Real scaler = c * 0.5 * ((chir - 1.) / (fr * fr + Fuzz<Real>()));

            // Compute sum of L/R fluxes
            const Real qa = chat * wl_idn * wl_ivx;
            const Real qb = chat * wr_idn * wr_ivx;
            const Real fsum_d = qa + qb;
            const Real fsum_mx = qa * scalel * wl_ivx + qb * scaler * wr_ivx;
            const Real fsum_my = qa * scalel * wl_ivy + qb * scaler * wr_ivy;
            const Real fsum_mz = qa * scalel * wl_ivz + qb * scaler * wr_ivz;

            // Compute max wave speed in L/R states (see Toro eq. 10.43)
            const Real a = chat * std::max(std::abs(sl), std::abs(sr));

            // Compute difference in L/R states dU, multiplied by max wave speed
            const Real du_d = a * (wr_idn - wl_idn);
            const Real du_mx = a * (wr_idn * scalel * wr_ivx - wl_idn * scaler * wl_ivx);
            const Real du_my = a * (wr_idn * scalel * wr_ivy - wl_idn * scaler * wl_ivy);
            const Real du_mz = a * (wr_idn * scalel * wr_ivz - wl_idn * scaler * wl_ivz);

            // Set an approximate interface pressure for coordinate source terms
            p.flux(b, dir, IPR, k, j, i) = 0.5 * (wl_idn * pscalel + wr_idn * pscaler);

            // Compute the LLF flux at interface
            const Real frho = 0.5 * (fsum_d - du_d);
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = 0.5 * (fsum_mx - du_mx);
            q.flux(b, dir, ivy, k, j, i) = 0.5 * (fsum_my - du_my);
            q.flux(b, dir, ivz, k, j, i) = 0.5 * (fsum_mz - du_mz);
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_LLF_HPP_
