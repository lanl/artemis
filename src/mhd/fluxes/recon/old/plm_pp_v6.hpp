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
#ifndef UTILS_FLUXES_RECONSTRUCTION_PLM_PP_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_PLM_PP_HPP_

// Artemis includes
#include "artemis.hpp"
#include "utils/fluxes/reconstruction/slope_limiter.hpp"
#include "mhd/extended/defs.hpp"

// NOTE(PDMM): The following is taken directly from the open-source AthenaK software, and
// adapted for Parthenon/Artemis by PDM

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn ArtemisUtils::PLM_slope()
//! \brief Compute linear slope in cell i for ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.
KOKKOS_INLINE_FUNCTION
Real PLM_slope(const Real &q_im1, const Real &q_i, const Real &q_ip1, const TVDType TVD_type) {
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  Real dqm = ApplyTVD(TVD_type, dql, dqr);
  return dqm;
}

KOKKOS_INLINE_FUNCTION Real pos_comp(const Real a) {return a>0. ? a : 0.;}
KOKKOS_INLINE_FUNCTION Real neg_comp(const Real a) {return a<0. ? a : 0.;}

template <typename V>
KOKKOS_INLINE_FUNCTION void pp_limiter(const int b, const int k, const int j, const int i, 
		                       const int dir, const V &q, 
				       parthenon::ScratchPad2D<Real> &dW,
		   		       const Real gamma, const Real dx,
				       const Real dt) {
  const int idn = 0;
  const int ivx = 1 + ((dir - 1));
  const int ivy = 1 + ((dir - 1) + 1) % 3;
  const int ivz = 1 + ((dir - 1) + 2) % 3;
  const int ipr = 1 * 4 + 0;
  const int ise = 1 * 5 + 0;
  const int ibx = ivx + 5;
  const int iby = ivy + 5;
  const int ibz = ivz + 5;
  const int iEx = ibx + 3;
  const int iEy = iby + 3;
  const int iEz = ibz + 3;
  const int iJx = iEx + 3;
  const int iJy = iEy + 3;
  const int iJz = iEz + 3;
  const int iPe = std::max(iJx, std::max(iJy, iJz)) + 1;
  const Real rho = q(b, idn, k, j, i);
  const Real ne = Z_ion * rho;
  const Real vx  = q(b, ivx, k, j, i);
  const Real vy  = q(b, ivy, k, j, i);
  const Real vz  = q(b, ivz, k, j, i);
  const Real p   = q(b, ipr, k, j, i);
  const Real ie  = q(b, ise, k, j, i);
  const Real Bx  = q(b, ibx, k, j, i);
  const Real By  = q(b, iby, k, j, i);
  const Real Bz  = q(b, ibz, k, j, i);
  const Real Jx  = q(b, iJx, k, j, i);
  const Real Jy  = q(b, iJy, k, j, i);
  const Real Jz  = q(b, iJz, k, j, i);
  const Real Pe  = q(b, iPe, k, j, i);
  const Real Dv  = dW(ivy, i);
  const Real Dw  = dW(ivz, i);
  Real Dp  = dW(ipr, i);
  const Real DBx = dW(ibx, i);
  const Real DBy = dW(iby, i);
  const Real DBz = dW(ibz, i);
  const Real DEx = dW(iEx, i);
  const Real DEy = dW(iEy, i);
  const Real DEz = dW(iEz, i);
  const Real DJx = dW(iJx, i);
  const Real DJy = dW(iJy, i);
  const Real DJz = dW(iJz, i);
  const Real DPe = dW(iPe, i);
  // (a) Limit Du
  /*Real Du = dx/(dt*(1.+gamma));
  if (abs(dW(ivx, i)) > Du and abs(dW(ivx, i)) > 0.) 
	  dW(ivx, i) *= (Du/abs(dW(ivx, i)));
  Du = dW(ivx, i);*/
  // (b) Limit D\rho    
  Real Drho = (1.-(dt/dx)*abs(dW(ivx, i)))*rho;     
  if (abs(dW(idn, i)) > Drho and abs(dW(idn, i)) > 0.)          
	  dW(idn, i) *= (Drho/abs(dW(idn, i)));         
  Drho = dW(idn, i);                 
  const Real Dne = Z_ion * Drho;
  // (a) Limit Du
  const Real eps = 1.e-16;
  const Real alpha = lambda_ion/(L0*ne);
  const Real slope = 1./rho;//1./(100.*rho);

  const Real Du_I = (vx*DEx + vx*Dv*Bz - vx*Dw*By + vy*(DEy + Dw*Bx) - vz*(-DEz + Dv*Bx))
	          / (Bz*vy - By*vz + eps);
  const Real betax = alpha*DJy*Bz - alpha*DJz*By - (alpha/ne)*Dne*(Jy*Bz - Jz*By);
  const Real betay = -DEz + Dv*Bx + alpha*DJx*By - alpha*DJy*Bx 
	  	   - (alpha/ne)*Dne*(Jx*By - Jy*Bx);
  const Real betaz = -DEy - Dw*Bx - alpha*DJx*Bz + alpha*DJz*Bx 
	  	   - (alpha/ne)*Dne*(-Jx*Bz + Jz*Bx);
  const Real tmpyx = (-vy + alpha*Jy) / (-vx + alpha*Jx + eps);
  const Real tmpzx = (vz - alpha*Jz) / (vx - alpha*Jx + eps);
  const Real Du_NI = (DEx + Dv*Bz - Dw*By - tmpyx*betaz - tmpzx*betay - betax)
	  	   / (tmpyx*Bz - tmpzx*By + eps);
  const Real fDu = Du_NI/(Du_I + eps);
  //const Real gDu = 1./(1.+slope*abs(fDu-1.));
  const Real gDu = (Dp!=0.) ? 1./(1.+slope*abs(Du_NI-Du_I)) : 1.;
  dW(ivx, i) *= gDu;
  const Real Du = dW(ivx, i);
  if (gDu<0. or gDu>1. or std::isnan(gDu)) 
    printf("i=%d: g=%.2e, Du_I=%.2e, Du_NI=%.2e, beta=(%.2e,%.2e,%.2e) \n",i,gDu,Du_I,Du_NI,betax,betay,betaz);

  /*const Real Du1_I = (DEy - vx*DBz + Dw*Bx) / (Bz+eps);
  const Real Du1_NI = Du1_I + alpha*(DJx*Bz + Jx*DBz - DJz*Bx + (Dne/ne)*(-Jx*Bz + Jz*Bx))/(Bz+eps);
  Real f1 = Du1_NI/(Du1_I+eps);
  Real g1 = 1./(1.+slope*abs(f1-1.));
  const Real Du2_I = (-DEz - vx*DBy + Dv*Bx) / (By+eps);
  const Real Du2_NI = Du2_I + alpha*(DJx*By + Jx*DBy - DJy*Bx - (Dne/ne)*(Jx*By - Jy*Bx))/(By+eps);
  Real f2 = Du2_NI/(Du2_I+eps);
  Real g2 = 1./(1.+slope*abs(f2-1.));  
  dW(ivx, i) *= std::min(g1, g2);
  const Real Du = dW(ivx, i);*/
  // (c) Limit Dp
  // YH: for some reason this absDu will affect stability of the code!!! Seems like I need to
  //     abs(du) instead of the original formulation. I need to consider applying this to
  //     D\rho too!
  const Real absDu = abs(Du); //originally is dx/(dt*(1.+gamma));
  const Real num1 = p*(1.-(dt/dx)*gamma*absDu);
  const Real maxJ = std::max(Jx, std::max(Jy, Jz));
  const Real minDJ = std::min(DJx, std::min(DJy, DJz));
  const Real minJ_DPe = std::min(Jx*DPe, std::min(Jy*DPe, Jz*DPe)); 
  const Real DeltaPe = (dt/(2.*dx))*(abs(vx*DPe) + gamma*Pe*absDu +
	(J0/(e_charge*n0*char_speed)) * (gamma*Pe*abs(maxJ*Dne)/SQR(ne) -
	 neg_comp(minJ_DPe)/ne - Pe*neg_comp(minDJ)/ne));
  const Real num2 = Pe + 2.*DeltaPe + 0.5*abs(DPe);
  const Real denom = 1.;
  Dp = (num1 - num2) / denom;
  Dp = std::max(Dp, 0.);
  if (abs(dW(ipr, i)) > Dp and abs(dW(ipr, i)) > 0.) 
	  dW(ipr, i) *= (Dp/abs(dW(ipr, i)));
  Dp = dW(ipr, i);
  // (c) ii. Artemis also reconstruct e so need account for that
  Real De = (1./(rho*(gamma-1)))*Dp - (ie/rho)*abs(Drho);
  De = std::max(0., De);
  if (abs(dW(ise, i)) > De and abs(dW(ise, i)) > 0.)
          dW(ise, i) *= (De/abs(dW(ise, i)));
  De = dW(ise, i);
  // (d) Compute f(dW) for low-beta plasma
  const Real Btan_dot = By*DBy + Bz*DBz;
  const Real Delta_u = (dt/(2.*dx))*(vx*Du + (Dp+Btan_dot)/rho);
  const Real Delta_v = (dt/(2.*dx))*(vx*Dv - (Bx*DBy)/rho);
  const Real Delta_w = (dt/(2.*dx))*(vx*Dw - (Bx*DBz)/rho);
  const Real DeltaBx = 0.;
  const Real DeltaBy = (dt/(2.*dx))*(vx*By + Du*By - Bx*Dv);
  const Real DeltaBz = (dt/(2.*dx))*(vx*Bz + Du*Bz - Bx*Dw);

  const Real rhoc = rho - (dt/(2.*dx))*(rho*neg_comp(Du) + neg_comp(vx*Dp));
  const Real rhostar = 3.*rho - 2.*rhoc;
  const Real Du_mag = SQR(Du) + SQR(Dv) + SQR(Dw);
  const Real DB_mag = SQR(DBx) + SQR(DBy) + SQR(DBz);
  const Real Deltau_mag = SQR(Delta_u) + SQR(Delta_v) + SQR(Delta_w);
  const Real DeltaB_mag = SQR(DeltaBx) + SQR(DeltaBy) + SQR(DeltaBz);
  const Real u_dot_Du = vx*Du + vy*Dv + vz*Dw;
  const Real Du_dot_Deltau = Du*Delta_u + Dv*Delta_v + Dw*Delta_w;
  const Real LdW = (rhoc*rho/rhostar)*Deltau_mag + DeltaB_mag +
	           0.125*((rhoc+SQR(Drho)/(4.*rhostar))*Du_mag + DB_mag) +
		   0.5*pos_comp(Drho*u_dot_Du) -
		   (rhoc/rhostar+1.)*neg_comp(Drho*Du_dot_Deltau) + 
		   (Pe + 2*DeltaPe)/(gamma-1.);
  const Real rhoe = (p + (dt/dx)*(neg_comp(vx*Dp) + gamma*p*neg_comp(Du))) 
	  	  / (gamma-1.);
  //const Real fdW = std::sqrt(rhoe/std::max(LdW, rhoe));
  const Real fdW = (rhoe==0.) ? 0. : std::exp(1.-std::max(LdW, rhoe)/rhoe);
  //const Real fdW = (rhoe==0.) ? 0. : std::pow(rhoe / std::max(LdW, rhoe), 2.0);
  if (std::isnan(fdW) or fdW<0. or fdW>1.) 
    printf("i=%d: fdW=%.2e, LdW=%.2e, rhoe=%.2e, DeltaPe=%.2e, Drho=%.2e, Dp=%.2e, DPe=%.2e, DJx=%.2e, ne=%.2e, Jx=%.2e, Pe=%.2e, vx=%.2e \n", i,fdW,LdW,rhoe,DeltaPe,Drho,Dp,DPe,DJx,ne, Jx, Pe, vx);

  // YH: Only apply to ideal MHD component! Rmb Pe shld have minimal diffusion! I not sure if this is causing my result to be worse...
  for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b)-7; ++n) {
    dW(n, i) *= fdW; // shld I *0.9 to give some allowance?
  }
}


//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm_pp, X1DIR, ...>
//! \brief The piecewise linear reconstruction method in the X1 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm_pp, X1DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply_pp(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql,
                                    parthenon::ScratchPad2D<Real> &qr,
				    parthenon::ScratchPad2D<Real> &dW,
				    const Real gamma, const Real dt, 
				    const TVDType TVD_type,
				    const Real dW_sw[16]) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          dW(n, i) = PLM_slope(q(b, n, k, j, i - 1), q(b, n, k, j, i),
                               q(b, n, k, j, i + 1), TVD_type);
        });
    }
    parthenon::par_for_inner(
    DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
      const int dir = 1;
      geometry::Coords<GEOM> coords(q.GetCoordinates(b), k, j, i);
      const auto &dx = coords.GetCellWidths();
      pp_limiter(b, k, j, i, dir, q, dW, gamma, dx[dir-1], dt);
    });
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          ql(n, i + 1) = q(b, n, k, j, i) + 0.5*dW(n, i)*dW_sw[n];
          qr(n, i)     = q(b, n, k, j, i) - 0.5*dW(n, i)*dW_sw[n];
        });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm_pp, X2DIR, ...>
//! \brief The piecewise linear reconstruction method in the X2 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm_pp, X2DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply_pp(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_jp1,
                                    parthenon::ScratchPad2D<Real> &qr_j,
				    parthenon::ScratchPad2D<Real> &dW,
				    const Real gamma, const Real dt,
				    const TVDType TVD_type,
				    const Real dW_sw[16]) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          dW(n, i) = PLM_slope(q(b, n, k, j - 1, i), q(b, n, k, j, i),
                               q(b, n, k, j + 1, i), TVD_type);
        });
    }
    parthenon::par_for_inner(
    DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
      const int dir = 2;
      geometry::Coords<GEOM> coords(q.GetCoordinates(b), k, j, i);
      const auto &dx = coords.GetCellWidths();
      pp_limiter(b, k, j, i, dir, q, dW, gamma, dx[dir-1], dt);
    });
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          ql_jp1(n, i) = q(b, n, k, j, i) + 0.5*dW(n, i)*dW_sw[n];
          qr_j(n, i)   = q(b, n, k, j, i) - 0.5*dW(n, i)*dW_sw[n];
        });
    }

  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm_pp, X3DIR, ...>
//! \brief The piecewise linear reconstruction method in the X3 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm_pp, X3DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply_pp(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_kp1,
                                    parthenon::ScratchPad2D<Real> &qr_k,
				    parthenon::ScratchPad2D<Real> &dW,
				    const Real gamma, const Real dt,
				    const TVDType TVD_type,
				    const Real dW_sw[16]) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          dW(n, i) = PLM_slope(q(b, n, k - 1, j, i), q(b, n, k, j, i),
                               q(b, n, k + 1, j, i), TVD_type);
        });
    }
    parthenon::par_for_inner(
    DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
      const int dir = 3;
      geometry::Coords<GEOM> coords(q.GetCoordinates(b), k, j, i);
      const auto &dx = coords.GetCellWidths();
      pp_limiter(b, k, j, i, dir, q, dW, gamma, dx[dir-1], dt);
    });
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          ql_kp1(n, i) = q(b, n, k, j, i) + 0.5*dW(n, i)*dW_sw[n];
          qr_k(n, i)   = q(b, n, k, j, i) - 0.5*dW(n, i)*dW_sw[n];
        });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_PLM_PP_HPP_
