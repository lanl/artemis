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
//! \file hllc_hall_xmhd.hpp
//! \brief Harten-Lax-vanLeer (HLLC_HALL_XMHD) Riemann solver which accounts for
//!        whistler wave as fastest speed
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
#ifndef UTILS_FLUXES_RIEMANN_HLLC_HALL_XMHD_HPP_
#define UTILS_FLUXES_RIEMANN_HLLC_HALL_XMHD_HPP_

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

// Root solve for cubic equation
KOKKOS_INLINE_FUNCTION
Real solve_cubic(const std::array<Real, 4> coeffs, Real xguess, Real pT, 
		 const Real tol = 1.e-12, const int maxiter = 5000) {
  const Real a = coeffs[0];
  const Real b = coeffs[1];
  const Real c = coeffs[2];
  const Real d = coeffs[3];
  auto f = [a, b, c, d](Real x) {  // Cubic polynomial
    return ((a*x + b)*x + c)*x + d;
  };
  auto df = [a, b, c, d](Real x) { // Derivative of cubic polynomial
    return (3.*a*x + 2.*b)*x + c;
  };

  Real x = xguess;
    if (x < Real(0)) x = Real(0);
    if (x > pT)      x = pT;

    for (int i = 0; i < maxiter; ++i) {
        Real fx = f(x);
        if (Kokkos::abs(fx) < tol) return x;

        Real dfx = df(x);
        if (Kokkos::abs(dfx) < Real(1e-14)) break;

        Real xnew = x - fx / dfx;
        if (xnew > Real(0) && xnew < pT) {
            x = xnew;
        } else {
            break;
        }
    }
    return x;
}

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::hllc_hall_xmhd, ...>
//! \brief The HLLC_HALL_XMHD Riemann solver for ideal gas hydrodynamics
template <Fluid FLUID_TYPE>
class RiemannSolver<RSolver::hllc_hall_xmhd, FLUID_TYPE> {
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
            Real fl_d = wl_idn * wl_ivx;
            Real fr_d = wr_idn * wr_ivx;
	    Real fl_mx = fl_d * wl_ivx;
	    Real fr_mx = fr_d * wr_ivx;
	    Real fl_my = fl_d * wl_ivy;
            Real fr_my = fr_d * wr_ivy;
	    Real fl_mz = fl_d * wl_ivz;
            Real fr_mz = fr_d * wr_ivz;
	    if (do_mhd) {
	      Bmag_l = (SQR(wl_ibx) + SQR(wl_iby) + SQR(wl_ibz));
	      Bmag_r = (SQR(wr_ibx) + SQR(wr_iby) + SQR(wr_ibz));
	      fl_mx += (0.5*Bmag_l - SQR(wl_ibx));
	      fr_mx += (0.5*Bmag_r - SQR(wr_ibx));
	      fl_my += - wl_ibx*wl_iby;
	      fr_my += - wr_ibx*wr_iby;
	      fl_mz += - wl_ibx*wl_ibz;
	      fr_mz += - wr_ibx*wr_ibz;
            }
	    Real fsum_d = fl_d + fr_d;
            Real fsum_mx = fl_mx + fr_mx;
            Real fsum_my = fl_my + fr_my;
            Real fsum_mz = fl_mz + fr_mz;

            [[maybe_unused]] Real el = Null<Real>();
            [[maybe_unused]] Real er = Null<Real>();
	    [[maybe_unused]] Real fl_e = Null<Real>();
            [[maybe_unused]] Real fl_bx = Null<Real>();
            [[maybe_unused]] Real fl_by = Null<Real>();
            [[maybe_unused]] Real fl_bz = Null<Real>();
            [[maybe_unused]] Real fl_Ex = Null<Real>();
            [[maybe_unused]] Real fl_Ey = Null<Real>();
            [[maybe_unused]] Real fl_Ez = Null<Real>();
            [[maybe_unused]] Real fl_Jx = Null<Real>();
            [[maybe_unused]] Real fl_Jy = Null<Real>();
            [[maybe_unused]] Real fl_Jz = Null<Real>();
            [[maybe_unused]] Real fl_Se = Null<Real>();

	    [[maybe_unused]] Real fr_e = Null<Real>();
            [[maybe_unused]] Real fr_bx = Null<Real>();
            [[maybe_unused]] Real fr_by = Null<Real>();
            [[maybe_unused]] Real fr_bz = Null<Real>();
            [[maybe_unused]] Real fr_Ex = Null<Real>();
            [[maybe_unused]] Real fr_Ey = Null<Real>();
            [[maybe_unused]] Real fr_Ez = Null<Real>();
            [[maybe_unused]] Real fr_Jx = Null<Real>();
            [[maybe_unused]] Real fr_Jy = Null<Real>();
            [[maybe_unused]] Real fr_Jz = Null<Real>();
            [[maybe_unused]] Real fr_Se = Null<Real>();

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
		  fl_e = (el + wl_ipr + BL_e) * wl_ivx +
			  - (wl_ibx*wl_ivx + wl_iby*wl_ivy + wl_ibz*wl_ivz) * wl_ibx;
		  fr_e = (er + wr_ipr + BR_e) * wr_ivx +
			  - (wr_ibx*wr_ivx + wr_iby*wr_ivy + wr_ibz*wr_ivz) * wr_ibx;
                  fsum_e = fl_e + fr_e;

		  fl_bx = 0.; fr_bx = 0.;
                  fl_by = wl_iby * wl_ivx - wl_ibx * wl_ivy;
                  fr_by = wr_iby * wr_ivx - wr_ibx * wr_ivy;
                  fl_bz = wl_ibz * wl_ivx - wl_ibx * wl_ivz;
                  fr_bz = wr_ibz * wr_ivx - wr_ibx * wr_ivz;
                  fsum_bx = fl_bx + fr_bx;
                  fsum_by = fl_by + fr_by;
                  fsum_bz = fl_bz + fr_bz;

		  // YH: for electron pressure in XMHD
		  ne_l = Z_ion * wl_idn; 
		  Se_l = wl_iPe / pow(ne_l, gm1);
		  ue_l = wl_ivx - (J0/(e_charge*n0*char_speed))*(wl_iJx/ne_l);
		  ne_r = Z_ion * wr_idn;
                  Se_r = wr_iPe / pow(ne_r, gm1);
		  ue_r = wr_ivx - (J0/(e_charge*n0*char_speed))*(wr_iJx/ne_r);
		  fl_Se = Se_l * ue_l;
		  fr_Se = Se_r * ue_r;
		  fsum_Se = fl_Se + fr_Se;

		  // For E-flux
		  Real ExL = 0., ExR = 0.;
		  Real EyL = 0., EyR = 0.;
		  Real EzL = 0., EzR = 0.;
		  EyL = -wl_ibz; EyR = -wr_ibz;
		  EzL =  wl_iby; EzR =  wr_iby;
		  fl_Ex = SQR(c_per_v) * ExL;
		  fr_Ex = SQR(c_per_v) * ExR;
		  fl_Ey = SQR(c_per_v) * EyL;
                  fr_Ey = SQR(c_per_v) * EyR;
		  fl_Ez = SQR(c_per_v) * EzL;
                  fr_Ez = SQR(c_per_v) * EzR;
		  fsum_Ex = fl_Ex + fr_Ex;
		  fsum_Ey = fl_Ey + fr_Ey;
		  fsum_Ez = fl_Ez + fr_Ez;

		  // For J-flux
		  Real JJ_coef_l = lambda_ion/(L0*ne_l);
		  Real JJ_coef_r = lambda_ion/(L0*ne_r);
		  fl_Jx = 2.*wl_ivx*wl_iJx - JJ_coef_l*wl_iJx*wl_iJx - Pe_coef*wl_iPe;
		  fl_Jy = wl_ivx*wl_iJy + wl_ivy*wl_iJx - JJ_coef_l*wl_iJx*wl_iJy;
		  fl_Jz = wl_ivx*wl_iJz + wl_ivz*wl_iJx - JJ_coef_l*wl_iJx*wl_iJz;
		  fr_Jx = 2.*wr_ivx*wr_iJx - JJ_coef_r*wr_iJx*wr_iJx - Pe_coef*wr_iPe;
                  fr_Jy = wr_ivx*wr_iJy + wr_ivy*wr_iJx - JJ_coef_r*wr_iJx*wr_iJy;
                  fr_Jz = wr_ivx*wr_iJz + wr_ivz*wr_iJx - JJ_coef_r*wr_iJx*wr_iJz;
		  fsum_Jx = fl_Jx + fr_Jx;
		  fsum_Jy = fl_Jy + fr_Jy;
		  fsum_Jz = fl_Jz + fr_Jz;
              }
            }

            // Compute max wave speed in L/R states (see Toro eq. 10.43)
            Real a = Null<Real>();
	    Real aL = Null<Real>();
	    Real aR = Null<Real>();
	    Real cL_ideal = Null<Real>();
	    Real cR_ideal = Null<Real>();
	    Real cL_Hall = Null<Real>();
	    Real cR_Hall = Null<Real>();	    
	    Real cL_EJ = Null<Real>();
	    Real cR_EJ = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
              cL_ideal = std::sqrt(gamma * wl_ipr / wl_idn);
              cR_ideal = std::sqrt(gamma * wr_ipr / wr_idn);
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
                cL_ideal = std::max(qaL, cfL);
                cR_ideal = std::max(qbR, cfR);
              }
	      a = std::max((std::abs(wl_ivx) + cL_ideal), (std::abs(wr_ivx) + cR_ideal));
            } else if constexpr (FLUID_TYPE == Fluid::dust) {
              a = std::max(std::abs(wl_ivx), std::abs(wr_ivx));
            }

            // Compute difference in L/R states dU, multiplied by max wave speed
            Real du_d, du_mx, du_my, du_mz;

            auto pco = p.GetCoordinates(b);
            geometry::Coords<Coordinates::cartesian> coords(pco, k, j, i);
            const auto &dx = coords.GetCellWidths();
	    const Real kmax = 1./dx[dir-1];

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
	    [[maybe_unused]] Real sL_NI = Null<Real>();
	    [[maybe_unused]] Real sR_NI = Null<Real>();
            if constexpr (FLUID_TYPE == Fluid::gas) {
	      if (do_mhd) {
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
		cL_EJ = std::max(cm_l, cp_l);
		cR_EJ = std::max(cm_r, cp_r);

		// Extract min & max wavespeeds for non-ideal components
		Real cm_min = std::min(wl_ivx + cm_l, wr_ivx + cm_r);
		Real cp_min = std::min(wl_ivx + cp_l, wr_ivx + cp_r);
		Real lamb12_min = std::min(lamb12_l, lamb12_r);
		sL_NI = std::min(lamb12_min, std::min(cm_min, cp_min));
		Real cm_max = std::max(wl_ivx + cm_l, wr_ivx + cm_r);
                Real cp_max = std::max(wl_ivx + cp_l, wr_ivx + cp_r);
                Real lamb12_max = std::max(lamb12_l, lamb12_r);
                sR_NI = std::max(lamb12_max, std::max(cm_max, cp_max));

		// Now try the most diffusive RS using largest speed
		Real amax = std::max(std::max(a, a_EJ), a_elec);
		
		du_Se = amax * (Se_r - Se_l);

		du_Ex = amax * (wr_iEx - wl_iEx);
                du_Ey = amax * (wr_iEy - wl_iEy);
                du_Ez = amax * (wr_iEz - wl_iEz);
		du_Jx = amax * (wr_iJx - wl_iJx);
		du_Jy = amax * (wr_iJy - wl_iJy);
		du_Jz = amax * (wr_iJz - wl_iJz);

	        du_d  = (wr_idn - wl_idn);
                du_mx = (wr_idn * wr_ivx - wl_idn * wl_ivx);
                du_my = (wr_idn * wr_ivy - wl_idn * wl_ivy);
                du_mz = (wr_idn * wr_ivz - wl_idn * wl_ivz);
	        du_e  = (er - el);
                du_bx = (wr_ibx - wl_ibx);
                du_by = (wr_iby - wl_iby);
                du_bz = (wr_ibz - wl_ibz);

		Real eta_H_L = lambda_ion*sqrt(Bmag_l)/(L0*ne_l);   
		Real cA_L = std::sqrt(Bmag_l / wl_idn);    
		Real cw_L = 0.5*eta_H_L*kmax + sqrt(SQR(0.5*eta_H_L*kmax) + SQR(cA_L));
		cL_Hall = std::min(cw_L, c_per_v);
		
		Real eta_H_R = lambda_ion*sqrt(Bmag_r)/(L0*ne_r);  
		Real cA_R = std::sqrt(Bmag_r / wr_idn); 
		Real cw_R = 0.5*eta_H_R*kmax + sqrt(SQR(0.5*eta_H_R*kmax) + SQR(cA_R));
                cR_Hall = std::min(cw_R, c_per_v);
              }
            }

            // Set an approximate interface pressure for coordinate source terms
            if constexpr (FLUID_TYPE == Fluid::gas) {
              p.flux(b, dir, IPR, k, j, i) = 0.5 * (wl_ipr + wr_ipr);
            }

	    // Compute Roe-avaeraged state
            const Real sqrtdl = std::sqrt(wl_idn);
            const Real sqrtdr = std::sqrt(wr_idn);
            const Real isdlpdr = 1.0/(sqrtdl + sqrtdr);
            const Real wroe_idn = sqrtdl*sqrtdr;
            const Real wroe_ivx = (sqrtdl*wl_ivx + sqrtdr*wr_ivx)*isdlpdr;
	    const Real wroe_ivy = (sqrtdl*wl_ivy + sqrtdr*wr_ivy)*isdlpdr;
	    const Real wroe_ivz = (sqrtdl*wl_ivz + sqrtdr*wr_ivz)*isdlpdr;
            // Note Roe average of magnetic field is different (Athena++)
            const Real wroe_ibx = (sqrtdr*wl_ibx + sqrtdl*wr_ibx)*isdlpdr;
            const Real wroe_iby = (sqrtdr*wl_iby + sqrtdl*wr_iby)*isdlpdr;
            const Real wroe_ibz = (sqrtdr*wl_ibz + sqrtdl*wr_ibz)*isdlpdr;
            const Real x = 0.5*(SQR(wl_iby-wr_iby) + 
			    SQR(wl_ibz-wr_ibz))/(SQR(sqrtdl+sqrtdr));
    	    const Real y = 0.5*(wl_idn + wr_idn)/wroe_idn;
	    const Real hroe = ((el + wl_ipr + Bmag_l)/sqrtdl 
			    + (er + wr_ipr + Bmag_r)/sqrtdr)*isdlpdr;
	    // Compute fast-magnetosonic speed using eq. B18 (adiabatic) 
	    const Real btsq = SQR(wroe_iby) + SQR(wroe_ibz);
	    const Real bxi = 0.5*(wl_ibx + wr_ibx);
	    const Real vaxsq = bxi*bxi/wroe_idn;
	    const Real bt_starsq = (gm1 - (gm1 - 1.0)*y)*btsq;
	    const Real hp = hroe - (vaxsq + btsq/wroe_idn);
	    const Real vsq = SQR(wroe_ivx) + SQR(wroe_ivy) + SQR(wroe_ivz);
	    const Real twid_asq = std::max((gm1*(hp-0.5*vsq)-(gm1-1.0)*x), 0.0);
	    const Real ct2 = bt_starsq/wroe_idn;
    	    const Real tsum = vaxsq + ct2 + twid_asq;
    	    const Real tdif = vaxsq + ct2 - twid_asq;
    	    const Real cf2_cs2 = std::sqrt(tdif*tdif + 4.0*twid_asq*ct2);
    	    const Real cfsq = 0.5*(tsum + cf2_cs2);
            const Real a_Eind = std::sqrt(cfsq);

	    // Whistler wave speed with Roe-averaged
	    const Real Bmag_roe = SQR(wroe_ibx) + SQR(wroe_iby) + SQR(wroe_ibz);
            const Real eta_H_roe = lambda_ion*sqrt(Bmag_roe)/(L0*wroe_idn);
            const Real cA_roe = std::sqrt(Bmag_roe / wroe_idn);
            Real cw_roe = 0.5*eta_H_roe*kmax + sqrt(SQR(0.5*eta_H_roe*kmax) + SQR(cA_roe));
            cw_roe = std::min(cw_roe, c_per_v);

	    // Compute all the signal speeds
	    const Real cmax_L = cL_ideal;//std::max(cL_ideal, cL_Hall);
	    const Real cmax_R = cR_ideal;//std::max(cR_ideal, cR_Hall);
	    const Real cmax_roe = std::max(a_Eind, cw_roe);
	    const Real SL_Hall = std::min(wl_ivx-cmax_L, wr_ivx-cmax_R);
		    	//std::min(std::min(wl_ivx-cmax_L, wr_ivx-cmax_R), 
			//    		 wroe_ivx-cmax_roe);
	    const Real SR_Hall = std::max(wl_ivx+cmax_L, wr_ivx+cmax_R);
		    	//std::max(std::max(wl_ivx+cmax_L, wr_ivx+cmax_R),
			//   		 wroe_ivx+cmax_roe);
	    const Real SL_EJ = std::min(wl_ivx-cL_EJ, wr_ivx-cR_EJ);
            const Real SR_EJ = std::max(wl_ivx+cL_EJ, wr_ivx+cR_EJ);

	    const Real SRSL_Hall = SR_Hall * SL_Hall;
	    const Real diffS_Hall = SR_Hall - SL_Hall;
	    const Real SRSL_EJ = SR_EJ * SL_EJ;
            const Real diffS_EJ = SR_EJ - SL_EJ;

	    // HLLC stuffs (from Shengtai Li (2005))
            const Real ptl = wl_ipr + 0.5*(SQR(wl_ibx) + SQR(wl_iby) + SQR(wl_ibz));
            const Real ptr = wr_ipr + 0.5*(SQR(wr_ibx) + SQR(wr_iby) + SQR(wr_ibz));
	    const Real sM = (wr_idn*wr_ivx*(SR_Hall-wr_ivx) - wl_idn*wl_ivx*(SL_Hall-wl_ivx) + ptl - ptr - SQR(wl_ibx) + SQR(wr_ibx)) /
	    		    (wr_idn*(SR_Hall-wr_ivx) - wl_idn*(SL_Hall-wl_ivx));
	    //const Real rho_starL = wl_idn*(SL_Hall-wl_ivx)/(SL_Hall-sM);
	    //const Real rho_starR = wr_idn*(SR_Hall-wr_ivx)/(SR_Hall-sM);	   
	    // Need HLL for some variables to fulfill consistency condition
	    // YH: Wrong here as U_HLL is a piecewise function! thus, my code not working!
	    const Real rho_HLL = (SR_Hall*wr_idn - SL_Hall*wl_idn - (fr_d - fl_d)) /
                                 (SR_Hall - SL_Hall);
	    const Real rho_starL = rho_HLL; const Real rho_starR = rho_HLL;
	    const Real u_HLL = ((SR_Hall*wr_idn*wr_ivx - SL_Hall*wl_idn*wl_ivx - (fr_mx - fl_mx)) /
                               (SR_Hall - SL_Hall)) / rho_HLL; 
	    const Real v_HLL = ((SR_Hall*wr_idn*wr_ivy - SL_Hall*wl_idn*wl_ivy - (fr_my - fl_my)) /
                               (SR_Hall - SL_Hall)) / rho_HLL;
	    const Real w_HLL = ((SR_Hall*wr_idn*wr_ivz - SL_Hall*wl_idn*wl_ivz - (fr_mz - fl_mz)) /
                               (SR_Hall - SL_Hall)) / rho_HLL;
	    const Real Bx_HLL = (SR_Hall*wr_ibx - SL_Hall*wl_ibx - (fr_bx - fl_bx)) /
                                (SR_Hall - SL_Hall);
	    const Real By_HLL = (SR_Hall*wr_iby - SL_Hall*wl_iby - (fr_by - fl_by)) /
                                (SR_Hall - SL_Hall);
	    const Real Bz_HLL = (SR_Hall*wr_ibz - SL_Hall*wl_ibz - (fr_bz - fl_bz)) /
                                (SR_Hall - SL_Hall);

	    // Compute the HLLC_HALL for ideal MHD components of flux at interface
	    if (SL_Hall >= 0.) {
	      q.flux(b, dir, IDN, k, j, i) = fl_d;
              q.flux(b, dir, ivx, k, j, i) = fl_mx;
              q.flux(b, dir, ivy, k, j, i) = fl_my;
              q.flux(b, dir, ivz, k, j, i) = fl_mz;
 	      q.flux(b, dir, IEN, k, j, i) = fl_e;
	      q.flux(b, dir, ibx, k, j, i) = fl_bx;
	      q.flux(b, dir, iby, k, j, i) = fl_by;
	      q.flux(b, dir, ibz, k, j, i) = fl_bz;
	    } else if (SL_Hall <= sM) {
	      const Real Bx_starL = Bx_HLL;
	      const Real By_starL = By_HLL;
	      const Real Bz_starL = Bz_HLL;
	      const Real pstarL = wl_idn*(SL_Hall-wl_ivx)*(sM-wl_ivx) + wl_ipr 
		                - SQR(wl_ibx) + SQR(Bx_starL);
              const Real mx_starL = rho_starL*u_HLL;
              const Real my_starL = rho_starL*v_HLL;
              const Real mz_starL = rho_starL*w_HLL; 
	      const Real Bu_starL = Bx_starL*u_HLL + By_starL*v_HLL + Bz_starL*w_HLL;
	      const Real Bu_L = wl_ibx*wl_ivx + wl_iby*wl_ivy + wl_ibz*wl_ivz;
	      const Real e_starL = el*(SL_Hall-wl_ivx)/(SL_Hall-sM)
                      + ((pstarL*sM-wl_ipr*wl_ivx) - (Bx_starL*Bu_starL-wl_ibx*Bu_L)) / (SL_Hall-sM);

	      q.flux(b, dir, IDN, k, j, i) = fl_d + SL_Hall*(rho_starL-wl_idn);
              q.flux(b, dir, ivx, k, j, i) = fl_mx + SL_Hall*(mx_starL-wl_idn*wl_ivx);
              q.flux(b, dir, ivy, k, j, i) = fl_my + SL_Hall*(my_starL-wl_idn*wl_ivy);
	      q.flux(b, dir, ivz, k, j, i) = fl_mz + SL_Hall*(mz_starL-wl_idn*wl_ivz);
              q.flux(b, dir, IEN, k, j, i) = fl_e + SL_Hall*(e_starL-el);
              q.flux(b, dir, ibx, k, j, i) = fl_bx + SL_Hall*(wl_ibx-Bx_starL);
              q.flux(b, dir, iby, k, j, i) = fl_by + SL_Hall*(wl_iby-By_starL);
              q.flux(b, dir, ibz, k, j, i) = fl_bz + SL_Hall*(wl_ibz-Bz_starL);
	    } else if (sM <= SR_Hall) {
	      const Real Bx_starR = Bx_HLL;
              const Real By_starR = By_HLL;
              const Real Bz_starR = Bz_HLL;
              const Real pstarR = wr_idn*(SR_Hall-wr_ivx)*(sM-wr_ivx) + wr_ipr
                                - SQR(wr_ibx) + SQR(Bx_starR);
              const Real mx_starR = rho_starR*u_HLL;
              const Real my_starR = rho_starR*v_HLL; 
              const Real mz_starR = rho_starR*w_HLL;
              const Real Bu_starR = Bx_starR*u_HLL + By_starR*v_HLL + Bz_starR*w_HLL;
              const Real Bu_R = wr_ibx*wr_ivx + wr_iby*wr_ivy + wr_ibz*wr_ivz;
              const Real e_starR = er*(SR_Hall-wr_ivx)/(SR_Hall-sM)
                      + ((pstarR*sM-wr_ipr*wr_ivx) - (Bx_starR*Bu_starR-wr_ibx*Bu_R)) / (SR_Hall-sM);
		
	      q.flux(b, dir, IDN, k, j, i) = fr_d + SR_Hall*(rho_starR-wr_idn);
              q.flux(b, dir, ivx, k, j, i) = fr_mx + SR_Hall*(mx_starR-wr_idn*wr_ivx);
              q.flux(b, dir, ivy, k, j, i) = fr_my + SR_Hall*(my_starR-wr_idn*wr_ivy);
              q.flux(b, dir, ivz, k, j, i) = fr_mz + SR_Hall*(mz_starR-wr_idn*wr_ivz);
              q.flux(b, dir, IEN, k, j, i) = fr_e + SR_Hall*(e_starR-er);
              q.flux(b, dir, ibx, k, j, i) = fr_bx + SR_Hall*(wr_ibx-Bx_starR);
              q.flux(b, dir, iby, k, j, i) = fr_by + SR_Hall*(wr_iby-By_starR);
              q.flux(b, dir, ibz, k, j, i) = fr_bz + SR_Hall*(wr_ibz-Bz_starR);
	    } else {
	      q.flux(b, dir, IDN, k, j, i) = fr_d;
              q.flux(b, dir, ivx, k, j, i) = fr_mx;
              q.flux(b, dir, ivy, k, j, i) = fr_my;
              q.flux(b, dir, ivz, k, j, i) = fr_mz;
              q.flux(b, dir, IEN, k, j, i) = fr_e;
              q.flux(b, dir, ibx, k, j, i) = fr_bx;
              q.flux(b, dir, iby, k, j, i) = fr_by;
              q.flux(b, dir, ibz, k, j, i) = fr_bz;
	    }
	    q.flux(b, dir, ibx, k, j, i) = 0.; 

	    // YH: use HLLC for non-ideal components
	    const Real ne_L = wl_idn * Z_ion;
	    const Real ne_R = wr_idn * Z_ion;
	    const Real ne_starL = rho_starL * Z_ion;
	    const Real ne_starR = rho_starR * Z_ion;

	    // Obtained a constants
	    const Real a1_L = (n0*e_charge*char_speed/J0)*ne_starL*(sM-sL_NI);
	    const Real a2_L = (n0*e_charge*char_speed/J0)*ne_starL*(pow(ne_L/ne_starL,1.-gamma)) 
             		     *wl_iPe*(sL_NI-wl_ivx+(J0/(n0*e_charge*char_speed))*(wl_iJx/ne_L));
	    const Real a3_L = sL_NI - 2.*sM;
	    const Real a4_L = (lambda_ion/(L0*ne_starL));
	    const Real a5_L = m_ion*L0/(me*lambda_ion);
	    const Real a6_L = sL_NI*wl_iJx-2*wl_ivx*wl_iJx+(lambda_ion/(L0*ne_L))*wl_iJx*wl_iJx
		      	     +(m_ion*L0/(me*lambda_ion))*wl_iPe;

	    const Real a1_R = (n0*e_charge*char_speed/J0)*ne_starR*(sM-sR_NI);
            const Real a2_R = (n0*e_charge*char_speed/J0)*ne_starR*(pow(ne_R/ne_starR,1.-gamma))
                             *wr_iPe*(sR_NI-wr_ivx+(J0/(n0*e_charge*char_speed))*(wr_iJx/ne_R));
            const Real a3_R = sR_NI - 2.*sM;
            const Real a4_R = (lambda_ion/(L0*ne_starR));
            const Real a5_R = m_ion*L0/(me*lambda_ion);
            const Real a6_R = sR_NI*wr_iJx-2*wr_ivx*wr_iJx+(lambda_ion/(L0*ne_R))*wr_iJx*wr_iJx
                             +(m_ion*L0/(me*lambda_ion))*wr_iPe;

	    if (sL_NI >= 0.) {
	      q.flux(b, dir, iJx, k, j, i) = fl_Jx;
              q.flux(b, dir, iJy, k, j, i) = fl_Jy;
              q.flux(b, dir, iJz, k, j, i) = fl_Jz;
              q.flux(b, dir, iPe, k, j, i) = fl_Se;
	    } else if (sL_NI <= 0. <= sM) {
	      const std::array<Real, 4> coeffs_L = {a5_L, a3_L*a1_L+a4_L*a1_L*a1_L-a6_L, 
		      		a3_L*a2_L+2.*a1_L*a2_L*a4_L, a2_L*a2_L*a4_L};
	      const Real Pe_guess = a1_L-a1_R==0 ? wl_iPe : (a2_R-a2_L)/(a1_L-a1_R);
	      const Real Pe_starL = solve_cubic(coeffs_L,Pe_guess,wl_ipr);
	      const Real Jx_starL = a1_L + a2_L/Pe_starL;
	      const Real Jy_starL = (wl_ivy*Jx_starL+sL_NI*wl_iJy-wl_ivx*wl_iJy-wl_ivy*wl_iJx
			          + (lambda_ion/(L0*ne_L))*wl_iJx*wl_iJy) /
		      		    (sL_NI-sM+(lambda_ion/(L0*ne_starL))*Jx_starL);
	      const Real Jz_starL = (wl_ivz*Jx_starL+sL_NI*wl_iJz-wl_ivx*wl_iJz-wl_ivz*wl_iJx
			      	  + (lambda_ion/(L0*ne_L))*wl_iJx*wl_iJz) / 
                                    (sL_NI-sM+(lambda_ion/(L0*ne_starL))*Jx_starL);
	      const Real Se_starL = Pe_starL*pow(ne_starL,1.-gamma);
	      const Real ue_starL = sM - (J0/(n0*e_charge*char_speed))*(Jx_starL/ne_starL);
	      const Real JJ_coef_starL = lambda_ion/(L0*ne_starL);
	      q.flux(b, dir, iJx, k, j, i) = 2.*sM*Jx_starL - JJ_coef_starL*Jx_starL*Jx_starL 
		      			   - Pe_coef*Pe_starL;
              q.flux(b, dir, iJy, k, j, i) = sM*Jy_starL + wl_ivy*Jx_starL 
		      			   - JJ_coef_starL*Jx_starL*Jy_starL;
              q.flux(b, dir, iJz, k, j, i) = sM*Jz_starL + wl_ivz*Jx_starL 
		      			   - JJ_coef_starL*Jx_starL*Jz_starL;
              q.flux(b, dir, iPe, k, j, i) = ue_starL*Se_starL;
	    } else if (sM <= 0. <= sR_NI) {
              const std::array<Real, 4> coeffs_R = {a5_R, a3_R*a1_R+a4_R*a1_R*a1_R-a6_R, 
		      		a3_R*a2_R+2.*a1_R*a2_R*a4_R, a2_R*a2_R*a4_R};
	      const Real Pe_guess = a1_L-a1_R==0 ? wr_iPe : (a2_R-a2_L)/(a1_L-a1_R);
              const Real Pe_starR = solve_cubic(coeffs_R,Pe_guess,wr_ipr);
              const Real Jx_starR = a1_R + a2_R/Pe_starR;
              const Real Jy_starR = (wr_ivy*Jx_starR+sR_NI*wr_iJy-wr_ivx*wr_iJy-wr_ivy*wr_iJx
                                  + (lambda_ion/(L0*ne_R))*wr_iJx*wr_iJy) /
                                    (sR_NI-sM+(lambda_ion/(L0*ne_starR))*Jx_starR);
              const Real Jz_starR = (wr_ivz*Jx_starR+sR_NI*wr_iJz-wr_ivx*wr_iJz-wr_ivz*wr_iJx
                                  + (lambda_ion/(L0*ne_R))*wr_iJx*wr_iJz) /
                                    (sR_NI-sM+(lambda_ion/(L0*ne_starR))*Jx_starR);
              const Real Se_starR = Pe_starR*pow(ne_starR,1.-gamma);
              const Real ue_starR = sM - (J0/(n0*e_charge*char_speed))*(Jx_starR/ne_starR);
              const Real JJ_coef_starR = lambda_ion/(L0*ne_starR);
              q.flux(b, dir, iJx, k, j, i) = 2.*sM*Jx_starR - JJ_coef_starR*Jx_starR*Jx_starR
                                           - Pe_coef*Pe_starR;
              q.flux(b, dir, iJy, k, j, i) = sM*Jy_starR + wr_ivy*Jx_starR
                                           - JJ_coef_starR*Jx_starR*Jy_starR;
              q.flux(b, dir, iJz, k, j, i) = sM*Jz_starR + wr_ivz*Jx_starR
                                           - JJ_coef_starR*Jx_starR*Jz_starR;
              q.flux(b, dir, iPe, k, j, i) = ue_starR*Se_starR;
	    } else {
	      q.flux(b, dir, iJx, k, j, i) = fr_Jx;
              q.flux(b, dir, iJy, k, j, i) = fr_Jy;
              q.flux(b, dir, iJz, k, j, i) = fr_Jz;
              q.flux(b, dir, iPe, k, j, i) = fr_Se;
	    }
	    
	    // Flux for Efield based on LLF. Doesn't matter here as its Jacobian is zero and also, 
	    // flux update for Efield is directly done at the edge-located E using curl of B-fcc.
	    q.flux(b, dir, iEx, k, j, i) = 0.5 * (fsum_Ex - du_Ex);
            q.flux(b, dir, iEy, k, j, i) = 0.5 * (fsum_Ey - du_Ey);
            q.flux(b, dir, iEz, k, j, i) = 0.5 * (fsum_Ez - du_Ez);


	    // YH: set to zero to check if HLLC is ok or not
            q.flux(b, dir, iEx, k, j, i) = 0.;
            q.flux(b, dir, iEy, k, j, i) = 0.;
            q.flux(b, dir, iEz, k, j, i) = 0.;
            q.flux(b, dir, iJx, k, j, i) = 0.;
            q.flux(b, dir, iJy, k, j, i) = 0.;
            q.flux(b, dir, iJz, k, j, i) = 0.;
            q.flux(b, dir, iPe, k, j, i) = 0.;

	    const Real frho = 0.5 * (fsum_d - du_d);
	    // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
	    // YH: i think from Eq. 5
            q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
            // YH: I think velocity face for +ve but I set to magnetic field for mhd so don't modify vf if is do_mhd
            vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_HLLC_HALL_XMHD_HPP_
