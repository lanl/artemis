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
//! \file llf_hall_xmhd.hpp
//! \brief Local Lax Friedrichs (LLF_HALL) Riemann solver, also known as Rusanov's
//! method, for hydrodynamics.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
#ifndef UTILS_FLUXES_RIEMANN_LLF_HALL_XMHD_HPP_
#define UTILS_FLUXES_RIEMANN_LLF_HALL_XMHD_HPP_

// NOTE(PDM): The following is taken directly from the open-source Athena++/AthenaK
// software, and adapted for Parthenon/Artemis by PDM on 10/08/23

// C++ headers
#include <algorithm>
#include <cmath>

// Artemis headers
#include "artemis.hpp"
#include "utils/eos/eos.hpp"

#include "mhd/extended/defs.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::llf_hall_xmhd, ...>
//! \brief The LLF_HALL_XMHD Riemann solver for ideal gas hydrodynamics
template <Fluid FLUID_TYPE>
class RiemannSolver<RSolver::llf_hall_xmhd, FLUID_TYPE> {
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
            [[maybe_unused]] Real wl_ibx = Null<Real>();
            [[maybe_unused]] Real wl_iby = Null<Real>();
            [[maybe_unused]] Real wl_ibz = Null<Real>();
            [[maybe_unused]] Real wr_ibx = Null<Real>();
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
	    [[maybe_unused]] Real Bmag_l = Null<Real>();
	    [[maybe_unused]] Real Bmag_r = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              wl_ipr = wl(IPR, i);
              wr_ipr = wr(IPR, i);
	      wl_ise = wl(ISE, i);
              wr_ise = wr(ISE, i);
	      if (do_mhd) {
                wl_ibx = wl(ibx, i);
                wl_iby = wl(iby, i);
                wl_ibz = wl(ibz, i);
                wr_ibx = wr(ibx, i);
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

            // Compute sum of L/R fluxes
            Real qa = wl_idn * wl_ivx;
            Real qb = wr_idn * wr_ivx;
            Real fsum_d = qa + qb;
            Real fsum_mx = qa * wl_ivx + qb * wr_ivx;
            Real fsum_my = qa * wl_ivy + qb * wr_ivy;
            Real fsum_mz = qa * wl_ivz + qb * wr_ivz;
	    if (do_mhd) {
	      Bmag_l = (SQR(wl_ibx) + SQR(wl_iby) + SQR(wl_ibz));
	      Bmag_r = (SQR(wr_ibx) + SQR(wr_iby) + SQR(wr_ibz));
              fsum_mx += (0.5*Bmag_l - SQR(wl_ibx)) + (0.5*Bmag_r - SQR(wr_ibx));
              fsum_my += - wl_ibx*wl_iby - wr_ibx*wr_iby;
              fsum_mz += - wl_ibx*wl_ibz - wr_ibx*wr_ibz;
            }

            [[maybe_unused]] Real el = Null<Real>();
            [[maybe_unused]] Real er = Null<Real>();
            [[maybe_unused]] Real fsum_e = Null<Real>();
	    [[maybe_unused]] Real fsum_bx = Null<Real>();
            [[maybe_unused]] Real fsum_by = Null<Real>();
            [[maybe_unused]] Real fsum_bz = Null<Real>();
	    [[maybe_unused]] Real fsum_Ex = Null<Real>();      
	    [[maybe_unused]] Real fsum_Ey = Null<Real>();         
	    [[maybe_unused]] Real fsum_Ez = Null<Real>();
	    [[maybe_unused]] Real fsum_Jx = Null<Real>();
            [[maybe_unused]] Real fsum_Jy = Null<Real>();
            [[maybe_unused]] Real fsum_Jz = Null<Real>();
	    [[maybe_unused]] Real fsum_Se = Null<Real>(); // YH: need u_e before I can construct flux for Se
            if constexpr (FLUID_TYPE == Fluid::gas) {
              el = wl_ipr * igm1 +
                   0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
              er = wr_ipr * igm1 +
                   0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
	      if (do_mhd) { // YH: account for magnetic field energy
                el += 0.5 * (SQR(wl_ibx) + SQR(wl_iby) + SQR(wl_ibz));
                er += 0.5 * (SQR(wr_ibx) + SQR(wr_iby) + SQR(wr_ibz));
              }
              fsum_e = (el + wl_ipr) * wl_ivx + (er + wr_ipr) * wr_ivx;
	      if (do_mhd) {
                  Real BL_e = 0.5 * (SQR(wl_ibx) + SQR(wl_iby) + SQR(wl_ibz));
                  Real BR_e = 0.5 * (SQR(wr_ibx) + SQR(wr_iby) + SQR(wr_ibz));
                  fsum_e = (el + wl_ipr + BL_e) * wl_ivx + (er + wr_ipr + BR_e) * wr_ivx;
                  fsum_e += - (wl_ibx*wl_ivx + wl_iby*wl_ivy + wl_ibz*wl_ivz) * wl_ibx
                          - (wr_ibx*wr_ivx + wr_iby*wr_ivy + wr_ibz*wr_ivz) * wr_ibx;

                  Real FyL = wl_iby * wl_ivx - wl_ibx * wl_ivy;
                  Real FyR = wr_iby * wr_ivx - wr_ibx * wr_ivy;
                  Real FzL = wl_ibz * wl_ivx - wl_ibx * wl_ivz;
                  Real FzR = wr_ibz * wr_ivx - wr_ibx * wr_ivz;
                  fsum_bx = 0.;
                  fsum_by = FyL + FyR;
                  fsum_bz = FzL + FzR;

		  // YH: for electron pressure in XMHD
		  ne_l = Z_ion * wl_idn; 
		  Se_l = wl_iPe / pow(ne_l, gm1);
		  ue_l = wl_ivx - (J0/(e_charge*n0*char_speed))*(wl_iJx/ne_l);
		  ne_r = Z_ion * wr_idn;
                  Se_r = wr_iPe / pow(ne_r, gm1);
		  ue_r = wr_ivx - (J0/(e_charge*n0*char_speed))*(wr_iJx/ne_r);
		  fsum_Se = Se_l * ue_l + Se_r * ue_r;

		  // For E-flux
		  Real ExL = 0., ExR = 0.;
		  Real EyL = 0., EyR = 0.;
		  Real EzL = 0., EzR = 0.;
		  EyL = -wl_ibz; EyR = -wr_ibz;
		  EzL =  wl_iby; EzR =  wr_iby;
		  fsum_Ex = SQR(c_per_v) * (ExL + ExR);
		  fsum_Ey = SQR(c_per_v) * (EyL + EyR);
		  fsum_Ez = SQR(c_per_v) * (EzL + EzR);

		  // For J-flux
		  Real JJ_coef_l = lambda_ion/(L0*ne_l);
		  Real JJ_coef_r = lambda_ion/(L0*ne_r);
		  Real JxL = 2.*wl_ivx*wl_iJx - JJ_coef_l*wl_iJx*wl_iJx - Pe_coef*wl_iPe;
		  Real JyL = wl_ivx*wl_iJy + wl_ivy*wl_iJx - JJ_coef_l*wl_iJx*wl_iJy;
		  Real JzL = wl_ivx*wl_iJz + wl_ivz*wl_iJx - JJ_coef_l*wl_iJx*wl_iJz;
		  Real JxR = 2.*wr_ivx*wr_iJx - JJ_coef_r*wr_iJx*wr_iJx - Pe_coef*wr_iPe;
                  Real JyR = wr_ivx*wr_iJy + wr_ivy*wr_iJx - JJ_coef_r*wr_iJx*wr_iJy;
                  Real JzR = wr_ivx*wr_iJz + wr_ivz*wr_iJx - JJ_coef_r*wr_iJx*wr_iJz;
		  fsum_Jx = JxL + JxR;
		  fsum_Jy = JyL + JyR;
		  fsum_Jz = JzL + JzR;
              }
            }

            // Compute max wave speed in L/R states (see Toro eq. 10.43)
            Real a = Null<Real>();
	    Real aL = Null<Real>();
	    Real aR = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              qa = std::sqrt(gamma * wl_ipr / wl_idn);
              qb = std::sqrt(gamma * wr_ipr / wr_idn);
	      if (do_mhd) {
                aL = std::sqrt(gamma * wl_ipr / wl_idn);
                aR = std::sqrt(gamma * wr_ipr / wr_idn);
                Real caL = std::sqrt((SQR(wl_ibx) + SQR(wl_iby) + SQR(wl_ibz)) / wl_idn);
                Real caR = std::sqrt((SQR(wr_ibx) + SQR(wr_iby) + SQR(wr_ibz)) / wr_idn);
                Real qaL = std::max(aL, caL);
                Real qbR = std::max(aR, caR);
                Real caL_x = std::sqrt(SQR(wl_ibx) / wl_idn);
                Real caR_x = std::sqrt(SQR(wr_ibx) / wr_idn);
                Real apcaL = SQR(aL) + SQR(caL);
                Real apcaxL = aL * caL_x;
                Real cfL = std::sqrt(0.5*((SQR(aL)+SQR(caL)) + std::sqrt(SQR(apcaL) - 4.*SQR(apcaxL))));
                Real apcaR = SQR(aR) + SQR(caR);
                Real apcaxR = aR * caR_x;
                Real cfR = std::sqrt(0.5*((SQR(aR)+SQR(caR)) + std::sqrt(SQR(apcaR) - 4.*SQR(apcaxR))));
                qa = std::max(qaL, cfL);
                qb = std::max(qbR, cfR);
              }
	      a = std::max((std::abs(wl_ivx) + qa), (std::abs(wr_ivx) + qb));
            } else if constexpr (FLUID_TYPE == Fluid::dust) {
              a = std::max(std::abs(wl_ivx), std::abs(wr_ivx));
            }

            // Compute difference in L/R states dU, multiplied by max wave speed
            Real du_d, du_mx, du_my, du_mz;

            [[maybe_unused]] Real du_e = Null<Real>();
	    [[maybe_unused]] Real du_bx = Null<Real>();
            [[maybe_unused]] Real du_by = Null<Real>();
            [[maybe_unused]] Real du_bz = Null<Real>();
	    [[maybe_unused]] Real du_Ex = Null<Real>(); 
	    [[maybe_unused]] Real du_Ey = Null<Real>();
	    [[maybe_unused]] Real du_Ez = Null<Real>();
	    [[maybe_unused]] Real du_Jx = Null<Real>();
            [[maybe_unused]] Real du_Jy = Null<Real>();
            [[maybe_unused]] Real du_Jz = Null<Real>();
	    [[maybe_unused]] Real du_Se = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              du_e = a * (er - el);
	      if (do_mhd) {
                du_bx = a * (wr_ibx - wl_ibx);
                du_by = a * (wr_iby - wl_iby);
                du_bz = a * (wr_ibz - wl_ibz);
		// YH: for XMHD
		Real aL_elec = sqrt(m_ion/me) * aL;
		Real aR_elec = sqrt(m_ion/me) * aR;
		Real a_elec = std::max((std::abs(wl_ivx) + aL_elec), 
			      (std::abs(wr_ivx) + aR_elec));
		// YH: correct wavespeed from Jacobian
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

		// Now try the most diffusive RS using largest speed
		Real amax = std::max(std::max(a, a_EJ), a_elec);
		
		du_Se = amax * (Se_r - Se_l);

		du_Ex = amax * (wr_iEx - wl_iEx);
                du_Ey = amax * (wr_iEy - wl_iEy);
                du_Ez = amax * (wr_iEz - wl_iEz);
		du_Jx = amax * (wr_iJx - wl_iJx);
		du_Jy = amax * (wr_iJy - wl_iJy);
		du_Jz = amax * (wr_iJz - wl_iJz);

		// From the ideal MHD components
		auto pco = p.GetCoordinates(b);
		geometry::Coords<Coordinates::cartesian> coords(pco, k, j, i);
		const auto &dx = coords.GetCellWidths();
		// Paper seem to have flexible def for eta_H so I can model this part
		Real kmax = 1./dx[dir-1];
		// Phase speed of whistler waves
		Real eta_H_L = lambda_ion*sqrt(Bmag_l)/(L0*ne_l);
                Real cA_L = std::sqrt(Bmag_l / wl_idn);
                Real cw_L = 0.5*eta_H_L*kmax + sqrt(SQR(0.5*eta_H_L*kmax) + SQR(cA_L));
		cw_L = std::min(cw_L, c_per_v);

                Real eta_H_R = lambda_ion*sqrt(Bmag_r)/(L0*ne_r);
                Real cA_R = std::sqrt(Bmag_r / wr_idn);
                Real cw_R = 0.5*eta_H_R*kmax + sqrt(SQR(0.5*eta_H_R*kmax) + SQR(cA_R));
		cw_R = std::min(cw_R, c_per_v);

		// (Type 1) Davis estimate
		Real a_cw = std::max(abs(wl_ivx) + cw_L, abs(wr_ivx) + cw_R);

		// (Type 2) Simple averaging
		/*Real Bmag = SQR(0.5*(wl_ibx + wr_ibx)) +
                            SQR(0.5*(wl_iby + wr_iby)) +
                            SQR(0.5*(wl_ibz + wr_ibz));
                Real eta_H = lambda_ion*sqrt(Bmag)/(L0*0.5*(ne_l+ne_r));
                Real cA = std::sqrt(Bmag / 0.5*(wl_idn+wr_idn));
                Real cw = 0.5*eta_H*kmax + sqrt(SQR(0.5*eta_H*kmax) + SQR(cA));
                cw = std::min(cw, c_per_v);
		a_cw = std::max(a_cw, abs(0.5*(wl_ivx+wr_ivx)) + cw);*/

		// (Type 3) Battern estimate using Roe averaging
		const Real sqrtdl = std::sqrt(wl_idn);
    		const Real sqrtdr = std::sqrt(wr_idn);
    		const Real isdlpdr = 1.0/(sqrtdl + sqrtdr);
		const Real wroe_idn = sqrtdl*sqrtdr;
		const Real wroe_ivx = (sqrtdl*wl_ivx + sqrtdr*wr_ivx)*isdlpdr;
		// Note Roe average of magnetic field is different (Athena++)
		const Real wroe_ibx = (sqrtdr*wl_ibx + sqrtdl*wr_ibx)*isdlpdr;
		const Real wroe_iby = (sqrtdr*wl_iby + sqrtdl*wr_iby)*isdlpdr;
		const Real wroe_ibz = (sqrtdr*wl_ibz + sqrtdl*wr_ibz)*isdlpdr;
		const Real Bmag_roe = SQR(wroe_ibx) + SQR(wroe_iby) + SQR(wroe_ibz);
		const Real eta_H_roe = lambda_ion*sqrt(Bmag_roe)/(L0*wroe_idn);
                const Real cA_roe = std::sqrt(Bmag_roe / wroe_idn);
                Real cw_roe = 0.5*eta_H_roe*kmax + sqrt(SQR(0.5*eta_H_roe*kmax) + SQR(cA_roe));
                cw_roe = std::min(cw_roe, c_per_v); 
		a_cw = std::max(a_cw, abs(wroe_ivx) + cw_roe);

		Real amax_2 = std::max(a, a_cw);
	        du_d = amax_2 * (wr_idn - wl_idn);
                du_mx = amax_2 * (wr_idn * wr_ivx - wl_idn * wl_ivx);
                du_my = amax_2 * (wr_idn * wr_ivy - wl_idn * wl_ivy);
                du_mz = amax_2 * (wr_idn * wr_ivz - wl_idn * wl_ivz);
	        du_e = amax_2 * (er - el);
                du_bx = amax_2 * (wr_ibx - wl_ibx);
                du_by = amax_2 * (wr_iby - wl_iby);
                du_bz = amax_2 * (wr_ibz - wl_ibz);

              }
            }

            // Set an approximate interface pressure for coordinate source terms
            if constexpr (FLUID_TYPE == Fluid::gas) {
              p.flux(b, dir, IPR, k, j, i) = 0.5 * (wl_ipr + wr_ipr);
            }

            // Compute the LLF_HALL_XMHD flux at interface
            const Real frho = 0.5 * (fsum_d - du_d);
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = 0.5 * (fsum_mx - du_mx);
            q.flux(b, dir, ivy, k, j, i) = 0.5 * (fsum_my - du_my);
            q.flux(b, dir, ivz, k, j, i) = 0.5 * (fsum_mz - du_mz);
            if constexpr (FLUID_TYPE == Fluid::gas) {
              q.flux(b, dir, IEN, k, j, i) = 0.5 * (fsum_e - du_e);

	      // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
	      // YH: i think from Eq. 5
              q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
              // YH: I think velocity face for +ve but I set to magnetic field for mhd so don't modify vf if is do_mhd
              vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
	      if (do_mhd) {
                // YH: flux for Bfield
                q.flux(b, dir, ibx, k, j, i) = 0.; //0.5 * (fsum_bx - du_bx);
                q.flux(b, dir, iby, k, j, i) = 0.5 * (fsum_by - du_by);
                q.flux(b, dir, ibz, k, j, i) = 0.5 * (fsum_bz - du_bz);
		q.flux(b, dir, iEx, k, j, i) = 0.5 * (fsum_Ex - du_Ex);
		q.flux(b, dir, iEy, k, j, i) = 0.5 * (fsum_Ey - du_Ey);
		q.flux(b, dir, iEz, k, j, i) = 0.5 * (fsum_Ez - du_Ez);
		q.flux(b, dir, iJx, k, j, i) = 0.5 * (fsum_Jx - du_Jx);
                q.flux(b, dir, iJy, k, j, i) = 0.5 * (fsum_Jy - du_Jy);
                q.flux(b, dir, iJz, k, j, i) = 0.5 * (fsum_Jz - du_Jz);
		// YH: flux for Pe (XMHD)
		q.flux(b, dir, iPe, k, j, i) = 0.5 * (fsum_Se - du_Se);
              } 
            }
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_LLF_HALL_XMHD_HPP_
