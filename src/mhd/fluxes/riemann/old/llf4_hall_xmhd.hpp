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
//! \file llf4_hall_xmhd.hpp
//! \brief Local Lax Friedrichs (LLF4_HALL_XMHD) Riemann solver, also known as Rusanov's
//! method, for hydrodynamics.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
#ifndef UTILS_FLUXES_RIEMANN_LLF4_HALL_XMHD_HPP_
#define UTILS_FLUXES_RIEMANN_LLF4_HALL_XMHD_HPP_

// NOTE(PDM): The following is taken directly from the open-source Athena++/AthenaK
// software, and adapted for Parthenon/Artemis by PDM on 10/08/23

// C++ headers
#include <algorithm>
#include <cmath>

// Artemis headers
#include "artemis.hpp"
#include "utils/eos/eos.hpp"
#include "hllc_hall_xmhd.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::llf4_hall_xmhd, ...>
//! \brief The LLF4_HALL_XMHD Riemann solver for ideal gas hydrodynamics
template <Fluid FLUID_TYPE>
class RiemannSolver<RSolver::llf4_hall_xmhd, FLUID_TYPE> {
 public:
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void
  solve(const EOS &eos, parthenon::team_mbr_t const &member, const int b, const int k,
        const int j, const int il, const int iu, const int dir,
        const parthenon::ScratchPad2D<Real> &wl, const parthenon::ScratchPad2D<Real> &wr,
        const V1 &p, const V2 &q, const V3 &vf,
	const bool do_mhd) const {
    using TE = parthenon::TopologicalElement;
    // Check sensibility of flux direction
    PARTHENON_REQUIRE(dir > 0 && dir <= 3, "Invalid flux direction!");
    [[maybe_unused]] auto fdir = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);

    // TODO(BRR) temporary
    const Real gm1 = eos.GruneisenParamFromDensityTemperature(Null<Real>(), Null<Real>());

    // Obtain number of species
    int nvar = Null<int>();
    if constexpr (FLUID_TYPE == Fluid::gas) {
      nvar = do_mhd ? 9 + 7 : 6;
    } else if constexpr (FLUID_TYPE == Fluid::dust) {
      nvar = 4;
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
      // YH: For mhd
      [[maybe_unused]] const int ibx = ivx + 5;
      [[maybe_unused]] const int iby = ivy + 5;
      [[maybe_unused]] const int ibz = ivz + 5;
      [[maybe_unused]] const int iEx = ibx + 3;
      [[maybe_unused]] const int iEy = iby + 3;
      [[maybe_unused]] const int iEz = ibz + 3;
      [[maybe_unused]] const int iJx = iEx + 3;
      [[maybe_unused]] const int iJy = iEy + 3;
      [[maybe_unused]] const int iJz = iEz + 3;
      [[maybe_unused]] const int iPe = std::max(iJx, std::max(iJy, iJz)) + 1;

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
	    // YH: added to account for mhd
            [[maybe_unused]] Real bxi = Null<Real>(); // YH: normal B field
            [[maybe_unused]] Real wl_iby = Null<Real>();
            [[maybe_unused]] Real wl_ibz = Null<Real>();
            [[maybe_unused]] Real wr_iby = Null<Real>();
            [[maybe_unused]] Real wr_ibz = Null<Real>();
	    // YH: For XMHD
            [[maybe_unused]] Real wl_iEx = Null<Real>();
            [[maybe_unused]] Real wl_iEy = Null<Real>();
            [[maybe_unused]] Real wl_iEz = Null<Real>();
            [[maybe_unused]] Real wr_iEx = Null<Real>();
            [[maybe_unused]] Real wr_iEy = Null<Real>();
            [[maybe_unused]] Real wr_iEz = Null<Real>();
            [[maybe_unused]] Real wl_iJx = Null<Real>();
            [[maybe_unused]] Real wl_iJy = Null<Real>();
            [[maybe_unused]] Real wl_iJz = Null<Real>();
            [[maybe_unused]] Real wr_iJx = Null<Real>();
            [[maybe_unused]] Real wr_iJy = Null<Real>();
            [[maybe_unused]] Real wr_iJz = Null<Real>();
            [[maybe_unused]] Real wl_iPe = Null<Real>();
            [[maybe_unused]] Real wr_iPe = Null<Real>();
            [[maybe_unused]] Real ne_l = Null<Real>();
            [[maybe_unused]] Real ne_r = Null<Real>();
            [[maybe_unused]] Real Se_l = Null<Real>();
            [[maybe_unused]] Real Se_r = Null<Real>();
            [[maybe_unused]] Real ue_l = Null<Real>();
            [[maybe_unused]] Real ue_r = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              wl_ipr = wl(IPR, i);
              wr_ipr = wr(IPR, i);
	      wl_ise = wl(ISE, i);
              wr_ise = wr(ISE, i);
	      if (do_mhd) {
                bxi = wl(ibx, i);		
                wl_iby = wl(iby, i);
                wl_ibz = wl(ibz, i);
                wr_iby = wr(iby, i);
                wr_ibz = wr(ibz, i);

		wl_iEx = wl(iEx, i);
                wr_iEx = wr(iEx, i);
                wl_iEy = wl(iEy, i);
                wr_iEy = wr(iEy, i);
                wl_iEz = wl(iEz, i);
                wr_iEz = wr(iEz, i);
                wl_iJx = wl(iJx, i);
                wr_iJx = wr(iJx, i);
                wl_iJy = wl(iJy, i);
                wr_iJy = wr(iJy, i);
                wl_iJz = wl(iJz, i);
                wr_iJz = wr(iJz, i);
                wl_iPe = wl(iPe, i);
                wr_iPe = wr(iPe, i);
              } 
            }

// ==========================================================================================
// From Athena++
// ==========================================================================================
	    Real spd[5];
	    constexpr Real SMALL_NUMBER = 1.0e-4;

	    // Compute L/R states for selected conserved variables
    	    Real bxsq = bxi*bxi;
  	    // (KGF): group transverse vector components for floating-point associativity symmetry
	    Real pbl = 0.5*(bxsq + (SQR(wl_iby) + SQR(wl_ibz)));  // magnetic pressure (l/r)
	    Real pbr = 0.5*(bxsq + (SQR(wr_iby) + SQR(wr_ibz)));
	    Real kel = 0.5*wl_idn*(SQR(wl_ivx) + (SQR(wl_ivy) + SQR(wl_ivz)));
	    Real ker = 0.5*wr_idn*(SQR(wr_ivx) + (SQR(wr_ivy) + SQR(wr_ivz)));

	    // Compute conserved variables
	    Real ul_d  = wl_idn;
	    Real ul_mx = wl_ivx*ul_d;
	    Real ul_my = wl_ivy*ul_d;
	    Real ul_mz = wl_ivz*ul_d;
	    Real ul_e = wl_ipr * igm1 +
                   0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz)) +
		   0.5 * (SQR(bxi) + SQR(wl_iby) + SQR(wl_ibz));
	    Real ul_by = wl_iby;
	    Real ul_bz = wl_ibz;

	    Real ur_d  = wr_idn;
	    Real ur_mx = wr_ivx*ur_d;
	    Real ur_my = wr_ivy*ur_d;
	    Real ur_mz = wr_ivz*ur_d;
	    Real ur_e = wr_ipr * igm1 +
                   0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz)) +
		   0.5 * (SQR(bxi) + SQR(wr_iby) + SQR(wr_ibz));
	    Real ur_by = wr_iby;
	    Real ur_bz = wr_ibz;

	    // Compute max wave speed in L/R states (see Toro eq. 10.43)
            Real a = Null<Real>();
            Real qa = std::sqrt(gamma * wl_ipr / wl_idn);
            Real qb = std::sqrt(gamma * wr_ipr / wr_idn);
            Real aL = std::sqrt(gamma * wl_ipr / wl_idn);
            Real aR = std::sqrt(gamma * wr_ipr / wr_idn);
            Real caL = std::sqrt((SQR(bxi) + SQR(wl_iby) + SQR(wl_ibz)) / wl_idn);
            Real caR = std::sqrt((SQR(bxi) + SQR(wr_iby) + SQR(wr_ibz)) / wr_idn);
            Real qaL = std::max(aL, caL);
            Real qbR = std::max(aR, caR);
            Real caL_x = std::sqrt(SQR(bxi) / wl_idn);
            Real caR_x = std::sqrt(SQR(bxi) / wr_idn);
            Real apcaL = SQR(aL) + SQR(caL);
            Real apcaxL = aL * caL_x;
            Real cfl = std::sqrt(0.5*((SQR(aL)+SQR(caL)) + std::sqrt(SQR(apcaL) - 4.*SQR(apcaxL))));
            Real apcaR = SQR(aR) + SQR(caR);
            Real apcaxR = aR * caR_x;
            Real cfr = std::sqrt(0.5*((SQR(aR)+SQR(caR)) + std::sqrt(SQR(apcaR) - 4.*SQR(apcaxR))));
	    qa = std::max(qaL, cfl);
            qb = std::max(qbR, cfr);
	    a = std::max((std::abs(wl_ivx) + qa), (std::abs(wr_ivx) + qb));

	    // YH: Modify here to account for hall wavespeed
            auto pco = p.GetCoordinates(b);
            geometry::Coords<Coordinates::cartesian> coords(pco, k, j, i);
            const auto &dx = coords.GetCellWidths();
            const Real kmax = pi/dx[dir-1];

	    Real eta_H_L = lambda_ion*sqrt(pbl)/(L0*ne_l);    
	    Real cA_L = std::sqrt(pbl / wl_idn);      
	    Real cw_L = 0.5*eta_H_L*kmax + sqrt(SQR(0.5*eta_H_L*kmax) + SQR(cA_L));
	    Real cL_Hall = std::min(cw_L, c_per_v);     

	    Real eta_H_R = lambda_ion*sqrt(pbr)/(L0*ne_r);    
	    Real cA_R = std::sqrt(pbr / wr_idn);   
	    Real cw_R = 0.5*eta_H_R*kmax + sqrt(SQR(0.5*eta_H_R*kmax) + SQR(cA_R));
	    Real cR_Hall = std::min(cw_R, c_per_v);
	    const Real cmax_L = std::max(cfl, cL_Hall);
            const Real cmax_R = std::max(cfr, cR_Hall);	
	    Real  a_cw = std::max(abs(wl_ivx) + cL_Hall, abs(wr_ivx) + cR_Hall);

	    spd[0] = std::min( wl_ivx-cmax_L, wr_ivx-cmax_R );
	    spd[4] = std::max( wl_ivx+cmax_L, wr_ivx+cmax_R );

	    //--- Step 3.  Compute L/R fluxes
	    Real ptl = wl_ipr + pbl; // total pressures L,R
	    Real ptr = wr_ipr + pbr;

	    Real fl_d  = ul_mx;
	    Real fl_mx = ul_mx*wl_ivx + ptl - bxsq; // YH: as artemis seems to separate pressure from momentum flux
	    Real fl_my = ul_my*wl_ivx - bxi*ul_by;
	    Real fl_mz = ul_mz*wl_ivx - bxi*ul_bz;
	    Real fl_e  = wl_ivx*(ul_e + ptl - bxsq) - bxi*(wl_ivy*ul_by + wl_ivz*ul_bz);
	    Real fl_by = ul_by*wl_ivx - bxi*wl_ivy;
	    Real fl_bz = ul_bz*wl_ivx - bxi*wl_ivz;

	    Real fr_d  = ur_mx;
	    Real fr_mx = ur_mx*wr_ivx + ptr - bxsq; // YH: as artemis seems to separate pressure from momentum flux
	    Real fr_my = ur_my*wr_ivx - bxi*ur_by;
	    Real fr_mz = ur_mz*wr_ivx - bxi*ur_bz;
	    Real fr_e  = wr_ivx*(ur_e + ptr - bxsq) - bxi*(wr_ivy*ur_by + wr_ivz*ur_bz);
	    Real fr_by = ur_by*wr_ivx - bxi*wr_ivy;
	    Real fr_bz = ur_bz*wr_ivx - bxi*wr_ivz;

	    //--- Step 4.  Compute middle and Alfven wave speeds
	    Real sdl = spd[0] - wl_ivx;  // S_i-u_i (i=L or R)
	    Real sdr = spd[4] - wr_ivx;

	    // S_M: eqn (38) of Miyoshi & Kusano
	    // (KGF): group ptl, ptr terms for floating-point associativity symmetry
	    spd[2] = (sdr*ur_mx - sdl*ul_mx + (ptl - ptr))/(sdr*ur_d - sdl*ul_d);

	    Real sdml   = spd[0] - spd[2];  // S_i-S_M (i=L or R)
	    Real sdmr   = spd[4] - spd[2];
	    Real sdml_inv = 1.0/sdml;
	    Real sdmr_inv = 1.0/sdmr;
	    // eqn (43) of Miyoshi & Kusano
	    Real ulst_d = ul_d * sdl * sdml_inv;
	    Real urst_d = ur_d * sdr * sdmr_inv;
	    Real ulst_d_inv = 1.0/ulst_d;
	    Real urst_d_inv = 1.0/urst_d;
	    Real sqrtdl = std::sqrt(ulst_d);
	    Real sqrtdr = std::sqrt(urst_d);

	    // eqn (51) of Miyoshi & Kusano
	    spd[1] = spd[2] - std::abs(bxi)/sqrtdl;
	    spd[3] = spd[2] + std::abs(bxi)/sqrtdr;

	    //--- Step 5.  Compute intermediate states
	    // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
	    // TODO(felker): place an assertion that ptstl==ptstr
	    Real ptstl = ptl + ul_d*sdl*(spd[2]-wl_ivx);
	    Real ptstr = ptr + ur_d*sdr*(spd[2]-wr_ivx);
	    Real ptst = 0.5*(ptstr + ptstl);  // total pressure (star state)

	    // ul* - eqn (39) of M&K
	    Real ulst_mx = ulst_d * spd[2];
	    Real ulst_my, ulst_mz, ulst_by, ulst_bz;
	    if (std::abs(ul_d*sdl*sdml-bxsq) < (SMALL_NUMBER)*ptst) {
	      // Degenerate case
	      ulst_my = ulst_d * wl_ivy;
	      ulst_mz = ulst_d * wl_ivz;

	      ulst_by = ul_by;
	      ulst_bz = ul_bz;
	    } else {
	      // eqns (44) and (46) of M&K
	      Real tmp = bxi*(sdl - sdml)/(ul_d*sdl*sdml - bxsq);
	      ulst_my = ulst_d * (wl_ivy - ul_by*tmp);
	      ulst_mz = ulst_d * (wl_ivz - ul_bz*tmp);

	      // eqns (45) and (47) of M&K
	      tmp = (ul_d*SQR(sdl) - bxsq)/(ul_d*sdl*sdml - bxsq);
	      ulst_by = ul_by * tmp;
	      ulst_bz = ul_bz * tmp;
	    }
	    // v_i* dot B_i*
	    // (KGF): group transverse momenta terms for floating-point associativity symmetry
	    Real vbstl = (ulst_mx*bxi+(ulst_my*ulst_by+ulst_mz*ulst_bz))*ulst_d_inv;
	    // eqn (48) of M&K
	    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
	    Real ulst_e = (sdl*ul_e - ptl*wl_ivx + ptst*spd[2] +
              	bxi*(wl_ivx*bxi + (wl_ivy*ul_by + wl_ivz*ul_bz) - vbstl))*sdml_inv;

	    // ur* - eqn (39) of M&K
	    Real urst_mx = urst_d * spd[2];
	    Real urst_my, urst_mz, urst_by, urst_bz;
	    if (std::abs(ur_d*sdr*sdmr - bxsq) < (SMALL_NUMBER)*ptst) {
   	      // Degenerate case
	      urst_my = urst_d * wr_ivy;
	      urst_mz = urst_d * wr_ivz;

	      urst_by = ur_by;
	      urst_bz = ur_bz;
	    } else {
	      // eqns (44) and (46) of M&K
	      Real tmp = bxi*(sdr - sdmr)/(ur_d*sdr*sdmr - bxsq);
	      urst_my = urst_d * (wr_ivy - ur_by*tmp);
	      urst_mz = urst_d * (wr_ivz - ur_bz*tmp);

	      // eqns (45) and (47) of M&K
	      tmp = (ur_d*SQR(sdr) - bxsq)/(ur_d*sdr*sdmr - bxsq);
	      urst_by = ur_by * tmp;
	      urst_bz = ur_bz * tmp;
	    }
	    // v_i* dot B_i*
	    // (KGF): group transverse momenta terms for floating-point associativity symmetry
	    Real vbstr = (urst_mx*bxi+(urst_my*urst_by+urst_mz*urst_bz))*urst_d_inv;
	    // eqn (48) of M&K
	    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
	    Real urst_e = (sdr*ur_e - ptr*wr_ivx + ptst*spd[2] +
	              bxi*(wr_ivx*bxi + (wr_ivy*ur_by + wr_ivz*ur_bz) - vbstr))*sdmr_inv;
	    // ul** and ur** - if Bx is near zero, same as *-states
	    Real invsumd = 1.0/(sqrtdl + sqrtdr);
	    Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

	    Real uldst_d = ulst_d; // Eq. 49
	    Real urdst_d = urst_d;

	    // YH: save rho** for below as LLF4 solver will update it as intermediate for flux
	    const Real rhol_dst = uldst_d;
	    const Real rhor_dst = urdst_d;
	
	    Real uldst_mx = ulst_mx;
	    Real urdst_mx = urst_mx;

	    // eqn (59) of M&K
	    Real tmp = invsumd*(sqrtdl*(ulst_my*ulst_d_inv) + sqrtdr*(urst_my*urst_d_inv) +
                        bxsig*(urst_by - ulst_by));
	    Real uldst_my = uldst_d * tmp;
	    Real urdst_my = urdst_d * tmp;

	    // eqn (60) of M&K
	    tmp = invsumd*(sqrtdl*(ulst_mz*ulst_d_inv) + sqrtdr*(urst_mz*urst_d_inv) +
                   bxsig*(urst_bz - ulst_bz));
	    Real uldst_mz = uldst_d * tmp;
	    Real urdst_mz = urdst_d * tmp;

	    // eqn (61) of M&K
	    tmp = invsumd*(sqrtdl*urst_by + sqrtdr*ulst_by +
                   bxsig*sqrtdl*sqrtdr*((urst_my*urst_d_inv) - (ulst_my*ulst_d_inv)));
	    Real uldst_by = tmp;
	    Real urdst_by = tmp;

	    // eqn (62) of M&K
	    tmp = invsumd*(sqrtdl*urst_bz + sqrtdr*ulst_bz +
                   bxsig*sqrtdl*sqrtdr*((urst_mz*urst_d_inv) - (ulst_mz*ulst_d_inv)));
	    Real uldst_bz = tmp;
	    Real urdst_bz = tmp;

	    // eqn (63) of M&K
	    tmp = spd[2]*bxi + (uldst_my*uldst_by + uldst_mz*uldst_bz)/uldst_d;
	    Real uldst_e = ulst_e - sqrtdl*bxsig*(vbstl - tmp);
	    Real urdst_e = urst_e + sqrtdr*bxsig*(vbstr - tmp);

	    //--- YH: save Bmag
	    Real ulst_Bmag = 0.5 * (bxsq + SQR(ulst_by) + SQR(ulst_bz));
	    Real urst_Bmag = 0.5 * (bxsq + SQR(urst_by) + SQR(urst_bz));
	    Real uldst_Bmag = 0.5 * (bxsq + SQR(uldst_by) + SQR(uldst_bz));
            Real urdst_Bmag = 0.5 * (bxsq + SQR(urdst_by) + SQR(urdst_bz));

	    //--- Step 6.  Compute flux
	    uldst_d = spd[1] * (uldst_d - ulst_d);
	    uldst_mx = spd[1] * (uldst_mx - ulst_mx);
	    uldst_my = spd[1] * (uldst_my - ulst_my);
	    uldst_mz = spd[1] * (uldst_mz - ulst_mz);
	    uldst_e = spd[1] * (uldst_e - ulst_e);
	    uldst_by = spd[1] * (uldst_by - ulst_by);
	    uldst_bz = spd[1] * (uldst_bz - ulst_bz);

	    ulst_d = spd[0] * (ulst_d - ul_d);
	    ulst_mx = spd[0] * (ulst_mx - ul_mx);
	    ulst_my = spd[0] * (ulst_my - ul_my);
	    ulst_mz = spd[0] * (ulst_mz - ul_mz);
	    ulst_e = spd[0] * (ulst_e - ul_e);
	    ulst_by = spd[0] * (ulst_by - ul_by);
	    ulst_bz = spd[0] * (ulst_bz - ul_bz);

	    urdst_d = spd[3] * (urdst_d - urst_d);
	    urdst_mx = spd[3] * (urdst_mx - urst_mx);
	    urdst_my = spd[3] * (urdst_my - urst_my);
	    urdst_mz = spd[3] * (urdst_mz - urst_mz);
	    urdst_e = spd[3] * (urdst_e - urst_e);
	    urdst_by = spd[3] * (urdst_by - urst_by);
	    urdst_bz = spd[3] * (urdst_bz - urst_bz);

	    urst_d = spd[4] * (urst_d  - ur_d);
	    urst_mx = spd[4] * (urst_mx - ur_mx);
	    urst_my = spd[4] * (urst_my - ur_my);
	    urst_mz = spd[4] * (urst_mz - ur_mz);
	    urst_e = spd[4] * (urst_e - ur_e);
	    urst_by = spd[4] * (urst_by - ur_by);
	    urst_bz = spd[4] * (urst_bz - ur_bz);

	    // LLF part
	    const Real el = wl_ipr * igm1 +
                   0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz)) +
		   0.5 * (SQR(bxi) + SQR(wl_iby) + SQR(wl_ibz));
	    const Real er = wr_ipr * igm1 +
                   0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz)) +
		   0.5 * (SQR(bxi) + SQR(wr_iby) + SQR(wr_ibz));
	    const Real sqrt_dl = std::sqrt(wl_idn);
  	    const Real sqrt_dr = std::sqrt(wr_idn);
	    const Real isdlpdr = 1.0/(sqrt_dl + sqrt_dr);
	    const Real wroe_idn = sqrt_dl*sqrt_dr;
	    const Real wroe_ivx = (sqrt_dl*wl_ivx + sqrt_dr*wr_ivx)*isdlpdr;
	    // Note Roe average of magnetic field is different (Athena++)
	    const Real wroe_ibx = (sqrt_dr*bxi + sqrt_dl*bxi)*isdlpdr;
	    const Real wroe_iby = (sqrt_dr*wl_iby + sqrt_dl*wr_iby)*isdlpdr;
	    const Real wroe_ibz = (sqrt_dr*wl_ibz + sqrt_dl*wr_ibz)*isdlpdr;
 	    const Real Bmag_roe = SQR(wroe_ibx) + SQR(wroe_iby) + SQR(wroe_ibz);
	    const Real eta_H_roe = lambda_ion*sqrt(Bmag_roe)/(L0*wroe_idn);
	    const Real cA_roe = std::sqrt(Bmag_roe / wroe_idn);
	    Real cw_roe = 0.5*eta_H_roe*kmax + sqrt(SQR(0.5*eta_H_roe*kmax) + SQR(cA_roe));
	    cw_roe = std::min(cw_roe, c_per_v);
	    a_cw = std::max(a_cw, abs(wroe_ivx) + cw_roe);

	    Real amax_2 = std::max(a, a_cw);
	    const Real du_d = amax_2 * (wr_idn - wl_idn);
	    const Real du_mx = amax_2 * (wr_idn * wr_ivx - wl_idn * wl_ivx);
	    const Real du_my = amax_2 * (wr_idn * wr_ivy - wl_idn * wl_ivy);
	    const Real du_mz = amax_2 * (wr_idn * wr_ivz - wl_idn * wl_ivz);
	    const Real du_e = amax_2 * (er - el);
	    const Real du_by = amax_2 * (wr_iby - wl_iby);
	    const Real du_bz = amax_2 * (wr_ibz - wl_ibz);
            q.flux(b, dir, IDN, k, j, i) = 0.5 * (fl_d + fr_d - du_d);
            q.flux(b, dir, ivx, k, j, i) = 0.5 * (fl_mx + fr_mx - du_mx);
            q.flux(b, dir, ivy, k, j, i) = 0.5 * (fl_my + fr_my - du_my);
            q.flux(b, dir, ivz, k, j, i) = 0.5 * (fl_mz + fr_mz - du_mz);
	    q.flux(b, dir, iby, k, j, i) = 0.5 * (fl_by + fr_by - du_by);
            q.flux(b, dir, ibz, k, j, i) = 0.5 * (fl_bz + fr_bz - du_bz);
	    q.flux(b, dir, IEN, k, j, i) = 0.5 * (fl_e + fr_e - du_e);
	    p.flux(b, dir, IPR, k, j, i) = 0.5 * (wl_ipr + wr_ipr);

	    if (spd[1] <= 0. <= spd[2]) {
	      // return Fl**
	      q.flux(b, dir, IDN, k, j, i) = fl_d  + ulst_d + uldst_d;
	      q.flux(b, dir, ivx, k, j, i) = fl_mx + ulst_mx + uldst_mx;
	      q.flux(b, dir, ivy, k, j, i) = fl_my + ulst_my + uldst_my;
	      q.flux(b, dir, ivz, k, j, i) = fl_mz + ulst_mz + uldst_mz;
	      q.flux(b, dir, IEN, k, j, i) = fl_e  + ulst_e + uldst_e;
	      q.flux(b, dir, iby, k, j, i) = fl_by + ulst_by + uldst_by;
	      q.flux(b, dir, ibz, k, j, i) = fl_bz + ulst_bz + uldst_bz;
	      p.flux(b, dir, IPR, k, j, i) = ptst - uldst_Bmag;
	    } else if (spd[2] <= 0. <= spd[3]) {
	      // return Fr**
	      q.flux(b, dir, IDN, k, j, i) = fr_d + urst_d + urdst_d;
	      q.flux(b, dir, ivx, k, j, i) = fr_mx + urst_mx + urdst_mx;
	      q.flux(b, dir, ivy, k, j, i) = fr_my + urst_my + urdst_my;
	      q.flux(b, dir, ivz, k, j, i) = fr_mz + urst_mz + urdst_mz;
	      q.flux(b, dir, IEN, k, j, i) = fr_e + urst_e + urdst_e;
	      q.flux(b, dir, iby, k, j, i) = fr_by + urst_by + urdst_by;
	      q.flux(b, dir, ibz, k, j, i) = fr_bz + urst_bz + urdst_bz;
	      p.flux(b, dir, IPR, k, j, i) = ptst - urdst_Bmag;
	    }
// ==========================================================================================
	    q.flux(b, dir, ibx, k, j, i) = 0.;
	    q.flux(b, dir, ivx, k, j, i) -= p.flux(b, dir, IPR, k, j, i);

            // LLF for XMHD components
	    [[maybe_unused]] Real fsum_Ex = Null<Real>();
            [[maybe_unused]] Real fsum_Ey = Null<Real>();
            [[maybe_unused]] Real fsum_Ez = Null<Real>();
            [[maybe_unused]] Real fsum_Jx = Null<Real>();
            [[maybe_unused]] Real fsum_Jy = Null<Real>();
            [[maybe_unused]] Real fsum_Jz = Null<Real>();
            [[maybe_unused]] Real fsum_Se = Null<Real>();
            // (a) for electron pressure in XMHD
            ne_l = Z_ion * wl_idn;
            Se_l = wl_iPe / pow(ne_l, gm1);
            ue_l = wl_ivx - (J0/(e_charge*n0*char_speed))*(wl_iJx/ne_l);
            ne_r = Z_ion * wr_idn;
            Se_r = wr_iPe / pow(ne_r, gm1);
            ue_r = wr_ivx - (J0/(e_charge*n0*char_speed))*(wr_iJx/ne_r);
	    const Real fl_Se = Se_l * ue_l;
            const Real fr_Se = Se_r * ue_r;
            fsum_Se = fl_Se + fr_Se;
            // (b) E-flux
            Real ExL = 0., ExR = 0.;
            Real EyL = 0., EyR = 0.;
            Real EzL = 0., EzR = 0.;
            EyL = -wl_ibz; EyR = -wr_ibz;
            EzL =  wl_iby; EzR =  wr_iby;
            fsum_Ex = SQR(c_per_v) * (ExL + ExR);
            fsum_Ey = SQR(c_per_v) * (EyL + EyR);
            fsum_Ez = SQR(c_per_v) * (EzL + EzR);
            // (c) J-flux
            const Real JJ_coef_l = lambda_ion/(L0*ne_l);
            const Real JJ_coef_r = lambda_ion/(L0*ne_r);
            const Real fl_Jx = 2.*wl_ivx*wl_iJx - JJ_coef_l*wl_iJx*wl_iJx - Pe_coef*wl_iPe;
            const Real fl_Jy = wl_ivx*wl_iJy + wl_ivy*wl_iJx - JJ_coef_l*wl_iJx*wl_iJy;
            const Real fl_Jz = wl_ivx*wl_iJz + wl_ivz*wl_iJx - JJ_coef_l*wl_iJx*wl_iJz;
            const Real fr_Jx = 2.*wr_ivx*wr_iJx - JJ_coef_r*wr_iJx*wr_iJx - Pe_coef*wr_iPe;
            const Real fr_Jy = wr_ivx*wr_iJy + wr_ivy*wr_iJx - JJ_coef_r*wr_iJx*wr_iJy;
            const Real fr_Jz = wr_ivx*wr_iJz + wr_ivz*wr_iJx - JJ_coef_r*wr_iJx*wr_iJz;
            fsum_Jx = fl_Jx + fr_Jx;
            fsum_Jy = fl_Jy + fr_Jy;
            fsum_Jz = fl_Jz + fr_Jz;
	    // (d) Diffusive stablization
	    [[maybe_unused]] Real du_Ex = Null<Real>();
            [[maybe_unused]] Real du_Ey = Null<Real>();
            [[maybe_unused]] Real du_Ez = Null<Real>();
            [[maybe_unused]] Real du_Jx = Null<Real>();
            [[maybe_unused]] Real du_Jy = Null<Real>();
            [[maybe_unused]] Real du_Jz = Null<Real>();
            [[maybe_unused]] Real du_Se = Null<Real>();
	    Real lamb12_l = wl_ivx - (lambda_ion*wl_iJx)/(L0*wl_idn);
            Real lamb12_r = wr_ivx - (lambda_ion*wr_iJx)/(L0*wr_idn);
            Real lamb12 = std::max(std::abs(lamb12_l),std::abs(lamb12_r));
            Real const1 = J0/(e_charge*n0*char_speed);
            Real const2 = (J0*m_ion*L0)/(e_charge*n0*char_speed*Z_ion*me*lambda_ion);
	    Real cm_l = 1.5*wl_ivx
                        -(lambda_ion*wl_iJx)/(L0*wl_idn)
                        -const1*wl_iJx/(2.*Z_ion*wl_idn) -
                        sqrt(pow(0.5*const1*wl_iJx/(Z_ion*wl_idn) - (lambda_ion*wl_iJx)/(L0*wl_idn),2)
                        + const2*wl_iPe/(wl_idn));
            Real cp_l = 1.5*wl_ivx
                        -(lambda_ion*wl_iJx)/(L0*wl_idn)
                        -const1*wl_iJx/(2.*Z_ion*wl_idn) +
                        sqrt(pow(0.5*const1*wl_iJx/(Z_ion*wl_idn) - (lambda_ion*wl_iJx)/(L0*wl_idn),2)
                        + const2*wl_iPe/(wl_idn));
            Real cm_r = 1.5*wr_ivx
                        -(lambda_ion*wr_iJx)/(L0*wr_idn)
                        -const1*wr_iJx/(2.*Z_ion*wr_idn) -
                        sqrt(pow(0.5*const1*wr_iJx/(Z_ion*wr_idn) - (lambda_ion*wr_iJx)/(L0*wr_idn),2)                        + const2*wr_iPe/(wr_idn));
            Real cp_r = 1.5*wr_ivx
                        -(lambda_ion*wr_iJx)/(L0*wr_idn)
                        -const1*wr_iJx/(2.*Z_ion*wr_idn) +
                        sqrt(pow(0.5*const1*wr_iJx/(Z_ion*wr_idn) - (lambda_ion*wr_iJx)/(L0*wr_idn),2)
                        + const2*wr_iPe/(wr_idn));
            Real a_m = std::max(std::abs(wl_ivx + cm_l),std::abs(wr_ivx + cm_r));
            Real a_p = std::max(std::abs(wl_ivx + cp_l),std::abs(wr_ivx + cp_r));
            Real a_EJ = std::max(a_m, a_p);
            a_EJ = std::max(a_EJ, lamb12);
            du_Se = a_EJ * (Se_r - Se_l);
            du_Ex = a_EJ * (wr_iEx - wl_iEx);
            du_Ey = a_EJ * (wr_iEy - wl_iEy);
            du_Ez = a_EJ * (wr_iEz - wl_iEz);
            du_Jx = a_EJ * (wr_iJx - wl_iJx);
            du_Jy = a_EJ * (wr_iJy - wl_iJy);
            du_Jz = a_EJ * (wr_iJz - wl_iJz);
	    
	    // YH: use HLLC for non-ideal components
	    const Real ne_L = wl_idn * Z_ion;
	    const Real ne_R = wr_idn * Z_ion;
	    const Real ne_starL = rhol_dst * Z_ion;
	    const Real ne_starR = rhor_dst * Z_ion;

	    const Real cm_min = std::min(wl_ivx + cm_l, wr_ivx + cm_r);
            const Real cp_min = std::min(wl_ivx + cp_l, wr_ivx + cp_r);
	    const Real lamb12_min = std::min(lamb12_l, lamb12_r);
            const Real sL_NI = std::min(lamb12_min, std::min(cm_min, cp_min));
	    const Real cm_max = std::max(wl_ivx + cm_l, wr_ivx + cm_r);
            const Real cp_max = std::max(wl_ivx + cp_l, wr_ivx + cp_r);
	    const Real lamb12_max = std::max(lamb12_l, lamb12_r);
            const Real sR_NI = std::max(lamb12_max, std::max(cm_max, cp_max));
	    // YH: Need HLL for J* to fulfill Toro's consistency condition
	    const Real Jx_HLL = (sR_NI*wr_iJx - sL_NI*wl_iJx - (fr_Jx - fl_Jx)) / (sR_NI - sL_NI);
	    const Real Jy_HLL = (sR_NI*wr_iJy - sL_NI*wl_iJy - (fr_Jy - fl_Jy)) / (sR_NI - sL_NI);
	    const Real Jz_HLL = (sR_NI*wr_iJz - sL_NI*wl_iJz - (fr_Jz - fl_Jz)) / (sR_NI - sL_NI);

	    /*q.flux(b, dir, iEx, k, j, i) = 0.5 * (fsum_Ex - du_Ex);
            q.flux(b, dir, iEy, k, j, i) = 0.5 * (fsum_Ey - du_Ey);
            q.flux(b, dir, iEz, k, j, i) = 0.5 * (fsum_Ez - du_Ez);
	    q.flux(b, dir, iJx, k, j, i) = 0.5 * (fsum_Jx - du_Jx) ;
            q.flux(b, dir, iJy, k, j, i) = 0.5 * (fsum_Jy - du_Jy) ;
            q.flux(b, dir, iJz, k, j, i) = 0.5 * (fsum_Jz - du_Jz) ;
            q.flux(b, dir, iPe, k, j, i) = 0.5 * (fsum_Se - du_Se) ;

	    if (sL_NI <= 0. <= spd[2]) {
	      const Real Se_starL = Se_l*(sL_NI-ue_l)/(sL_NI-spd[2]);
	      const Real Pe_starL = Se_starL*pow(ne_starL,gamma-1.);
	      const Real Jx_starL = Jx_HLL;
	      const Real Jy_starL = Jy_HLL;
	      const Real Jz_starL = Jz_HLL;
	      q.flux(b, dir, iJx, k, j, i) = fl_Jx + sL_NI*(Jx_starL-wl_iJx);
              q.flux(b, dir, iJy, k, j, i) = fl_Jy + sL_NI*(Jy_starL-wl_iJy); 
              q.flux(b, dir, iJz, k, j, i) = fl_Jz + sL_NI*(Jz_starL-wl_iJz);
              q.flux(b, dir, iPe, k, j, i) = fl_Se + sL_NI*(Se_starL-Se_l);
	    } else if (spd[2] <= 0. <= sR_NI) {
	      const Real Se_starR = Se_r*(sR_NI-ue_r)/(sR_NI-spd[2]);
              const Real Pe_starR = Se_starR*pow(ne_starR,gamma-1.);
              const Real Jx_starR = Jx_HLL;
              const Real Jy_starR = Jy_HLL;
              const Real Jz_starR = Jz_HLL;
              q.flux(b, dir, iJx, k, j, i) = fr_Jx + sR_NI*(Jx_starR-wr_iJx);
              q.flux(b, dir, iJy, k, j, i) = fr_Jy + sR_NI*(Jy_starR-wr_iJy);
              q.flux(b, dir, iJz, k, j, i) = fr_Jz + sR_NI*(Jz_starR-wr_iJz);
              q.flux(b, dir, iPe, k, j, i) = fr_Se + sR_NI*(Se_starR-Se_r);
	    }*/

	    if (sL_NI >= 0.) {
	      q.flux(b, dir, iJx, k, j, i) = fl_Jx;
              q.flux(b, dir, iJy, k, j, i) = fl_Jy;
              q.flux(b, dir, iJz, k, j, i) = fl_Jz;
              q.flux(b, dir, iPe, k, j, i) = fl_Se;
	    } else if (sL_NI <= 0. <= spd[2]) {
	      const Real Se_starL = Se_l*(sL_NI-ue_l)/(sL_NI-spd[2]);
	      const Real Pe_starL = Se_starL*pow(ne_starL,gamma-1.);
	      const Real Jx_starL = Jx_HLL;
	      const Real Jy_starL = Jy_HLL;
	      const Real Jz_starL = Jz_HLL;
	      q.flux(b, dir, iJx, k, j, i) = fl_Jx + sL_NI*(Jx_starL-wl_iJx);
              q.flux(b, dir, iJy, k, j, i) = fl_Jy + sL_NI*(Jy_starL-wl_iJy); 
              q.flux(b, dir, iJz, k, j, i) = fl_Jz + sL_NI*(Jz_starL-wl_iJz);
              q.flux(b, dir, iPe, k, j, i) = fl_Se + sL_NI*(Se_starL-Se_l);
	    } else if (spd[2] <= 0. <= sR_NI) {
	      const Real Se_starR = Se_r*(sR_NI-ue_r)/(sR_NI-spd[2]);
              const Real Pe_starR = Se_starR*pow(ne_starR,gamma-1.);
              const Real Jx_starR = Jx_HLL;
              const Real Jy_starR = Jy_HLL;
              const Real Jz_starR = Jz_HLL;
              q.flux(b, dir, iJx, k, j, i) = fr_Jx + sR_NI*(Jx_starR-wr_iJx);
              q.flux(b, dir, iJy, k, j, i) = fr_Jy + sR_NI*(Jy_starR-wr_iJy);
              q.flux(b, dir, iJz, k, j, i) = fr_Jz + sR_NI*(Jz_starR-wr_iJz);
              q.flux(b, dir, iPe, k, j, i) = fr_Se + sR_NI*(Se_starR-Se_r);
	    } else {
	      q.flux(b, dir, iJx, k, j, i) = fr_Jx;
              q.flux(b, dir, iJy, k, j, i) = fr_Jy;
              q.flux(b, dir, iJz, k, j, i) = fr_Jz;
              q.flux(b, dir, iPe, k, j, i) = fr_Se;
	    }

if (std::isnan(sL_NI) or std::isnan(sR_NI)) {
  std::cerr <<i<<","<<j<<","<<k<<": sL_NI="<<sL_NI<<", sR_NI="<<sR_NI<<std::endl;
  std::cerr <<"flux for ideal ="<<q.flux(b, dir, IDN, k, j, i)<<", "<<q.flux(b, dir, ivx, k, j, i)<<", "<<q.flux(b, dir, ivz, k, j, i)<<", "<<q.flux(b, dir, iby, k, j, i)<<", "<<q.flux(b, dir, ibz, k, j, i)<<", "<<q.flux(b, dir, IEN, k, j, i)<<std::endl;
  std::cerr <<"qflux="<<q.flux(b, dir, iJx, k, j, i)<<", "<<q.flux(b, dir, iJy, k, j, i)<<
	     ", "<< q.flux(b, dir, iJz, k, j, i)<<", "<<q.flux(b, dir, iPe, k, j, i)<<"\n";
  std::cerr <<"fsum = "<<fsum_Jx<<", "<<fsum_Jy<<", "<<fsum_Jz<<", "<<fsum_Se<<"\n";
  std::cerr <<"du_ ="<<du_Jx<<", "<<du_Jy<<", "<<du_Jz<<", "<<du_Se<<"\n";
  std::cerr << "================================\n";		    
  PARTHENON_FAIL("LLF4 failed!!!");
}

            // Set an approximate interface pressure for coordinate source terms
            if constexpr (FLUID_TYPE == Fluid::gas) {
            }
            // YH: this is flux for velocity at fcc orig from Artemis
            const Real frho = q.flux(b, dir, IDN, k, j, i);
            if constexpr (FLUID_TYPE == Fluid::gas) {
	      // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
              q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
              vf(b, fdir, n, k, j, i) =  frho / ((frho >= 0.0) ? wl_idn : wr_idn);
            }

          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_LLF4_HALL_XMHD_HPP_
