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

KOKKOS_INLINE_FUNCTION Real pos_comp(const Real a) {return (a > 0.) ? a : 0.;}
KOKKOS_INLINE_FUNCTION Real neg_comp(const Real a) {return (a < 0.) ? a : 0.;}
KOKKOS_INLINE_FUNCTION Real sign(const Real a) {return (a > 0.) - (a < 0.);}

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
  const Real Bx  = q(b, ibx, k, j, i);
  const Real By  = q(b, iby, k, j, i);
  const Real Bz  = q(b, ibz, k, j, i);
  const Real Jx  = q(b, iJx, k, j, i);
  const Real Pe  = q(b, iPe, k, j, i);
  // (a) Limit D\rho
  /*Real Drho = 0.5*rho;//(gamma/(1.+gamma))*rho;
  if (abs(dW(idn, i)) > Drho and abs(dW(idn, i)) > 0.) 
	  dW(idn, i) = sign(dW(idn, i)) * Drho;
  Drho = dW(idn, i);
  const Real Dne = Z_ion * Drho;*/
  // (b) Limit Du
  Real Du = dx/(dt*(1.+gamma));
  if (abs(dW(ivx, i)) > Du and abs(dW(ivx, i)) > 0.) 
	  dW(ivx, i) = sign(dW(ivx, i)) * Du;
  Du = dW(ivx, i);
  // (a) Limit D\rho
  Real Drho = (1.-(dt/dx)*abs(Du))*rho;
  if (abs(dW(idn, i)) > Drho and abs(dW(idn, i)) > 0.)
          dW(idn, i) *= (Drho/abs(dW(idn, i)));
  Drho = dW(idn, i);
  const Real Dne = Z_ion * Drho;
  // (c) Limit Dp
  // YH: for some reason this absDu will affect stability of the code!!! Seems like I need to
  //     abs(Du) instead of the original formulation. I need to consider applying this to
  //     D\rho too!
  const Real absDu = abs(Du); //originally is dx/(dt*(1.+gamma));
  const Real num1 = p*(1.-(dt/dx)*gamma*absDu);
  const Real DJx = dW(iJx, i);
  const Real DPe = dW(iPe, i);
  const Real DeltaPe = (dt/(2.*dx))*(abs(vx*DPe) + gamma*Pe*absDu) +
	(J0/(e_charge*n0*char_speed)) * (gamma*Pe*abs(Jx*Dne)/SQR(ne)
	+ abs(Jx*DPe)/ne + Pe*abs(DJx)/ne);
	// - neg_comp(Jx*DPe)/ne - Pe*neg_comp(DJx)/ne));
  const Real num2 = Pe + 2.*DeltaPe + 0.5*pos_comp(DPe);
  const Real denom = 1.;
  Real Dp = (num1 - num2) / denom; // YH: issue with this is if num2>num1, Dp can be -ve!
  Dp = max(Dp, 0.); // YH: to avoid negative Dp which will ruin soln!
  if (abs(dW(ipr, i)) > Dp and abs(dW(ipr, i)) > 0.) 
	  dW(ipr, i) = sign(dW(ipr, i)) * Dp;
  Dp = dW(ipr, i);
  // (d) Compute f(dW) for low-beta plasma
  const Real Dv  = dW(ivy, i);
  const Real Dw  = dW(ivz, i);
  const Real DBx = dW(ibx, i);
  const Real DBy = dW(iby, i);
  const Real DBz = dW(ibz, i);

  const Real Btan_dot = By*DBy + Bz*DBz;
  const Real Delta_u = (dt/(2.*dx))*(vx*Du + (Dp+Btan_dot)/rho);
  const Real Delta_v = (dt/(2.*dx))*(vx*Dv - (Bx*DBy)/rho);
  const Real Delta_w = (dt/(2.*dx))*(vx*Dw - (Bx*DBz)/rho);
  const Real DeltaBx = 0.;
  const Real DeltaBy = (dt/(2.*dx))*(vx*By + Du*By - Bx*Dv);
  const Real DeltaBz = (dt/(2.*dx))*(vx*Bz + Du*Bz - Bx*Dw);

  const Real rhoc = rho + (dt/(2.*dx))*(rho*abs(Du) + abs(vx*Dp));
	  //rho - (dt/(2.*dx))*(rho*neg_comp(Du) + neg_comp(vx*Dp));
  const Real rhostar = 3.*rho - 2.*rhoc;
  const Real Du_mag = SQR(Du) + SQR(Dv) + SQR(Dw);
  const Real DB_mag = SQR(DBx) + SQR(DBy) + SQR(DBz);
  const Real Deltau_mag = SQR(Delta_u) + SQR(Delta_v) + SQR(Delta_w);
  const Real DeltaB_mag = SQR(DeltaBx) + SQR(DeltaBy) + SQR(DeltaBz);
  const Real u_dot_Du = vx*Du + vy*Dv + vz*Dw;
  const Real Du_dot_Deltau = Du*Delta_u + Dv*Delta_v + Dw*Delta_w;
  const Real LdW = (rhoc*rho/rhostar)*Deltau_mag + DeltaB_mag +
	           0.125*((rhoc+SQR(Drho)/(4.*rhostar))*Du_mag + DB_mag) +
		   0.5*abs(Drho*u_dot_Du)
                   + (rhoc/rhostar+1.)*abs(Drho*Du_dot_Deltau) +
                   (Pe + 2*DeltaPe)/(gamma-1.);
  		   //0.5*pos_comp(Drho*u_dot_Du)
		   //- (rhoc/rhostar+1.)*neg_comp(Drho*Du_dot_Deltau) + 
		   //(Pe + 2*DeltaPe)/(gamma-1.);
  const Real rhoe = (p - (dt/dx)*(abs(vx*Dp) + gamma*p*abs(Du))) / (gamma-1.);
	  	//(p + (dt/dx)*(neg_comp(vx*Dp) + gamma*p*neg_comp(Du))) 
	  	//   / (gamma-1.);
  const Real fdW = std::sqrt(rhoe/std::max(LdW, rhoe));
  if (fdW<0.8) {
Real f_pow2 = std::pow(rhoe / std::max(LdW, rhoe), 2.0);
// Smooth ratio
Real eps = 1e-12;
Real f_smooth = rhoe / (rhoe + LdW + eps);
// Power (tunable sensitivity)
Real n = 2.0;
Real f_power = std::pow(rhoe, n) / (std::pow(rhoe, n) + std::pow(LdW, n) + eps);
// Logistic form (same idea, often numerically nicer)
Real f_logistic = 1.0 / (1.0 + std::pow(LdW / (rhoe + eps), n));
// Exponential (very sensitive to LdW)
Real f_exp = std::exp(-LdW / (rhoe + eps));
// Tanh (soft saturation)
Real f_tanh = std::tanh(rhoe / (LdW + eps));
// ---- Debug print (compact) ----
std::cout << std::fixed << std::setprecision(6)
          << "YH: r=" << rhoe << " L=" << LdW
          << " | sqrt=" << fdW
          << " pow2=" << f_pow2
          << " smooth=" << f_smooth
          << " power=" << f_power
          << " logistic=" << f_logistic
          << " exp=" << f_exp
          << " tanh=" << f_tanh
          << std::endl;
  }
  //if (std::isnan(fdW) or fdW<0. or fdW>1.) 
    printf("i=%d: fdW=%.2e, LdW=%.2e, rhoe=%.2e, DeltaPe=%.2e, Drho=%.2e, Dp=%.2e, DPe=%.2e, DJx=%.2e, ne=%.2e, Jx=%.2e, Pe=%.2e, vx=%.2e \n", i,fdW,LdW,rhoe,DeltaPe,Drho,Dp,DPe,DJx,ne, Jx, Pe, vx);
  // YH: Only apply to ideal MHD component! Rmb Pe shld have minimal diffusion! I not sure if this is causing my result to be worse...
  if (iPe!=q.GetUpperBound(b)) printf("i=%d, iPe=%d, nl=%d, nu=%d \n",i,iPe,q.GetLowerBound(b),q.GetUpperBound(b));
  for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
    dW(n, i) *= fdW; // shld I *0.9 to give some allowance? Not here as dw affect ok region
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
				    const TVDType TVD_type) const {
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
      std::cout<<i<<": starting pp"<<std::endl;
      pp_limiter(b, k, j, i, dir, q, dW, gamma, dx[dir-1], dt);
      std::cout<<"end pp"<<std::endl;
    });
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          ql(n, i + 1) = q(b, n, k, j, i) + 0.5*dW(n, i);
          qr(n, i)     = q(b, n, k, j, i) - 0.5*dW(n, i);
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
				    const TVDType TVD_type) const {
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
          ql_jp1(n, i) = q(b, n, k, j, i) + 0.5*dW(n, i);
          qr_j(n, i)   = q(b, n, k, j, i) - 0.5*dW(n, i);
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
				    const TVDType TVD_type) const {
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
          ql_kp1(n, i) = q(b, n, k, j, i) + 0.5*dW(n, i);
          qr_k(n, i)   = q(b, n, k, j, i) - 0.5*dW(n, i);
        });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_PLM_PP_HPP_
