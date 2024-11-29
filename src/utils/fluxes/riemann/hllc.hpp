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
//! \file hllc.hpp
//! \brief The HLLC Riemann solver for hydrodynamics, an extension of the HLLE fluxes to
//! include the contact wave.  Only works for ideal gas EOS in hydrodynamics.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
//!
//! - P. Batten, N. Clarke, C. Lambert, and D. M. Causon, "On the Choice of Wavespeeds
//!   for the HLLC Riemann Solver", SIAM J. Sci. & Stat. Comp. 18, 6, 1553-1570, (1997).
#ifndef UTILS_FLUXES_RIEMANN_HLLC_HPP_
#define UTILS_FLUXES_RIEMANN_HLLC_HPP_

// NOTE(PDM): The following is taken directly from the open-source Athena++/AthenaK
// software, and adapted for Parthenon/Artemis by PDM on 10/08/23

// C++ headers
#include <algorithm>
#include <cmath>

// Artemis headers
#include "artemis.hpp"
#include "utils/eos/eos.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::hllc, ...>
//! \brief The HLLC Riemann solver for ideal gas hydrodynamics
template <Fluid FLUID_TYPE>
class RiemannSolver<RSolver::hllc, FLUID_TYPE> {
 public:
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void
  solve(const EOS &eos, const Real chat, parthenon::team_mbr_t const &member, const int b, const int k,
        const int j, const int il, const int iu, const int dir,
        const parthenon::ScratchPad2D<Real> &wl, const parthenon::ScratchPad2D<Real> &wr,
        const V1 &p, const V2 &q, const V3 &vf) const {
    using TE = parthenon::TopologicalElement;
    // Check sensibility of flux direction
    PARTHENON_REQUIRE(dir > 0 && dir <= 3, "Invalid flux direction!");
    auto fdir = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);

    // TODO(BRR) temporary
    const Real gm1 = eos.GruneisenParamFromDensityTemperature(Null<Real>(), Null<Real>());

    // Obtain number of species (energy equation required for HLLC)
    const int nspecies = p.GetMaxNumberOfVars() / 6;

    for (int n = 0; n < nspecies; ++n) {
      const int IDN = n;
      const int ivx = nspecies + (n * 3) + ((dir - 1));
      const int ivy = nspecies + (n * 3) + ((dir - 1) + 1) % 3;
      const int ivz = nspecies + (n * 3) + ((dir - 1) + 2) % 3;
      const int IPR = nspecies * 4 + n;
      const int ISE = nspecies * 5 + n;
      const int IEN = IPR;
      const int IEG = ISE;

      Real igm1 = 1.0 / gm1;
      Real gamma = gm1 + 1.0;
      Real alpha = (gamma + 1.0) / (2.0 * gamma);

      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            // Create local references for L/R states (helps compiler vectorize)
            Real &wl_idn = wl(IDN, i);
            Real &wl_ivx = wl(ivx, i);
            Real &wl_ivy = wl(ivy, i);
            Real &wl_ivz = wl(ivz, i);
            Real &wl_ipr = wl(IPR, i);
            Real &wl_ise = wl(ISE, i);

            Real &wr_idn = wr(IDN, i);
            Real &wr_ivx = wr(ivx, i);
            Real &wr_ivy = wr(ivy, i);
            Real &wr_ivz = wr(ivz, i);
            Real &wr_ipr = wr(IPR, i);
            Real &wr_ise = wr(ISE, i);

            // Compute middle state estimates with PVRS (Toro 10.5.2)
            // define 6 registers used below
            Real qa, qb, qc, qd, qe, qf;
            qa = std::sqrt(gamma * wl_ipr / wl_idn);
            qb = std::sqrt(gamma * wr_ipr / wr_idn);
            Real el =
                wl_ipr * igm1 + 0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
            Real er =
                wr_ipr * igm1 + 0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
            qc = 0.25 * (wl_idn + wr_idn) *
                 (qa + qb); // average density * average sound speed
            qd = 0.5 * (wl_ipr + wr_ipr + (wl_ivx - wr_ivx) * qc); // P_mid

            // Compute sound speed in L,R
            qe = (qd <= wl_ipr) ? 1.0
                                : std::sqrt(1.0 + alpha * ((qd / wl_ipr) - 1.0)); // ql
            qf = (qd <= wr_ipr) ? 1.0
                                : std::sqrt(1.0 + alpha * ((qd / wr_ipr) - 1.0)); // qr

            // Compute the max/min wave speeds based on L/R
            Real sl = wl_ivx - qa * qe;
            Real sr = wr_ivx + qb * qf;

            // following min/max set to TINY_NUMBER to fix bug found in converging
            // supersonic flow
            qa = sr > 0.0 ? sr : 1.0e-20;  // bp
            qb = sl < 0.0 ? sl : -1.0e-20; // bm

            // Compute the contact wave speed and pressure
            qe = wl_ivx - sl; // vxl
            qf = wr_ivx - sr; // vxr

            qc = wl_ipr + qe * wl_idn * wl_ivx; // tl
            qd = wr_ipr + qf * wr_idn * wr_ivx; // tr

            Real ml = wl_idn * qe;
            Real mr = -(wr_idn * qf);

            // Determine the contact wave speed...
            Real am = (qc - qd) / (ml + mr);
            // ...and the pressure at the contact surface
            Real cp = (ml * qd + mr * qc) / (ml + mr);
            cp = cp > 0.0 ? cp : 0.0;

            // Compute L/R fluxes along the line bm (qb), bp (qa)
            qe = wl_idn * (wl_ivx - qb);
            qf = wr_idn * (wr_ivx - qa);

            Real fld = qe;
            Real frd = qf;
            Real flmx = qe * wl_ivx; // + wl_ipr;
            Real frmx = qf * wr_ivx; // + wr_ipr;
            Real flmy = qe * wl_ivy;
            Real frmy = qf * wr_ivy;
            Real flmz = qe * wl_ivz;
            Real frmz = qf * wr_ivz;
            Real fle = el * (wl_ivx - qb) + wl_ipr * wl_ivx;
            Real fre = er * (wr_ivx - qa) + wr_ipr * wr_ivx;

            // Compute flux weights or scales.  Set an approximate interface pressure for
            // coordinate source terms and pressure contribution to flux.
            if (am >= 0.0) {
              qc = am / (am - qb);
              qd = 0.0;
              qe = -qb / (am - qb);
            } else {
              qc = 0.0;
              qd = -am / (qa - am);
              qe = qa / (qa - am);
            }
            p.flux(b, dir, IPR, k, j, i) = qc * wl_ipr + qd * wr_ipr + qe * cp;

            // Compute the HLLC flux at interface, including weighted contribution of the
            // flux along the contact
            const Real frho = qc * fld + qd * frd;
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = qc * flmx + qd * frmx; // + qe * cp;
            q.flux(b, dir, ivy, k, j, i) = qc * flmy + qd * frmy;
            q.flux(b, dir, ivz, k, j, i) = qc * flmz + qd * frmz;
            q.flux(b, dir, IEN, k, j, i) = qc * fle + qd * fre + qe * cp * am;

            // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
            q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
            vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_HLLC_HPP_
