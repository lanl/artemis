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
#ifndef PGEN_LINEAR_WAVE_MHD_HPP_
#define PGEN_LINEAR_WAVE_MHD_HPP_
//! \file linear_wave_mhd.hpp
//! \brief Linear wave mhd problem generator for 1D/2D/3D problems. Direction of the w
//! wavevector is set to be along the x? axis by using the along_x? input flags, else it
//! is automatically set along the grid diagonal in 2D/3D.
//! This file also contains a function to compute L1 errors in solution.

// NOTE(PDM): The following is taken directly from the open-source Athena++/AthenaK
// software, and adapted for Parthenon/Artemis by PDM on 10/09/23

// C/C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

namespace {
const int NWAVE = 7;
//----------------------------------------------------------------------------------------
//! \struct LinWaveMHDVariables
//! \brief container for variables shared with linear wave mhd pgen and error functions
struct LinWaveMHDVariables {
  int wave_flag;
  Real amp, vflow, lambda;
  Real d0, p0, v1_0, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  Real lem[NWAVE][NWAVE], rem[NWAVE][NWAVE], ev[NWAVE];
  Real gamma, gm1;
  Real bx0, by0, bz0, dby, dbz;
  Real bscale, pscale, vscale;
  Real rho0;
};

} // end anonymous namespace

namespace linear_wave_mhd {

static LinWaveMHDVariables lwv;


//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//  \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//  Az are functions of x and y alone.
KOKKOS_INLINE_FUNCTION Real A1(const LinWaveMHDVariables& lwv, const Real x1, const Real x2, const Real x3) {
  Real x =  x1*lwv.cos_a2*lwv.cos_a3 + x2*lwv.cos_a2*lwv.sin_a3 + x3*lwv.sin_a2;
  Real y = -x1*lwv.sin_a3            + x2*lwv.cos_a3;
  Real Ay =  lwv.bz0*x - (lwv.dbz/lwv.k_par)*std::cos(lwv.k_par*(x));
  Real Az = -lwv.by0*x + (lwv.dby/lwv.k_par)*std::cos(lwv.k_par*(x)) + lwv.bx0*y;

  return -Ay*lwv.sin_a3 - Az*lwv.sin_a2*lwv.cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//  \brief A2: 2-component of vector potential
KOKKOS_INLINE_FUNCTION Real A2(const LinWaveMHDVariables& lwv, const Real x1, const Real x2, const Real x3) {
  Real x =  x1*lwv.cos_a2*lwv.cos_a3 + x2*lwv.cos_a2*lwv.sin_a3 + x3*lwv.sin_a2;
  Real y = -x1*lwv.sin_a3            + x2*lwv.cos_a3;
  Real Ay =  lwv.bz0*x - (lwv.dbz/lwv.k_par)*std::cos(lwv.k_par*(x));
  Real Az = -lwv.by0*x + (lwv.dby/lwv.k_par)*std::cos(lwv.k_par*(x)) + lwv.bx0*y;

  return Ay*lwv.cos_a3 - Az*lwv.sin_a2*lwv.sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential
KOKKOS_INLINE_FUNCTION Real A3(const LinWaveMHDVariables& lwv, const Real x1, const Real x2, const Real x3) {
  Real x =  x1*lwv.cos_a2*lwv.cos_a3 + x2*lwv.cos_a2*lwv.sin_a3 + x3*lwv.sin_a2;
  Real y = -x1*lwv.sin_a3            + x2*lwv.cos_a3;
  Real Az = -lwv.by0*x + (lwv.dby/lwv.k_par)*std::cos(lwv.k_par*(x)) + lwv.bx0*y;

  return Az*lwv.cos_a2;
}

//----------------------------------------------------------------------------------------
//! \fn void MagnetoHydroEigensystem()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal hydrodynamics
KOKKOS_INLINE_FUNCTION void MagnetoHydroEigensystem(const Real d, const Real v1, const Real v2,
                                             const Real v3, const Real p, // YH: need p to get h
                                             const Real gamma, Real eigenvalues[NWAVE],
                                             Real right_eigenmatrix[NWAVE][NWAVE],
					     const Real b1, const Real b2, const Real b3,
					     const Real x, const Real y, 
					     Real left_eigenmatrix[NWAVE][NWAVE]) {
  //--- Ideal Gas MagnetoHydrodynamics ---
  Real vsq = v1*v1 + v2*v2 + v3*v3;
  Real bsq = b1*b1 + b2*b2 + b3*b3;
  Real h = (p / (gamma - 1.0) + 0.5 * d * vsq + p + bsq) / d;
  Real a = std::sqrt(gamma * p / d);
  Real gm1 = gamma - 1.0;

  //================================From Athena++========================================
  Real btsq,bt_starsq,vaxsq,hp,twid_asq,cfsq,cf,cssq,cs;
  Real bt,bt_star,bet2,bet3,bet2_star,bet3_star,bet_starsq,vbet,alpha_f,alpha_s;      
  Real isqrtd,sqrtd,s,twid_a,qf,qs,af_prime,as_prime,afpbb,aspbb,vax;
  Real norm,cff,css,af,as,afpb,aspb,q2_star,q3_star,vqstr;
  Real ct2,tsum,tdif,cf2_cs2;
  Real qa,qb,qc,qd;
  btsq = b2*b2 + b3*b3;
  bt_starsq = (gm1 - (gm1 - 1.0)*y)*btsq;
  vaxsq = b1*b1/d;
  hp = h - (vaxsq + btsq/d);
  twid_asq = std::max((gm1*(hp-0.5*vsq)-(gm1-1.0)*x), TINY_NUMBER);
 
  // Compute fast- and slow-magnetosonic speeds (eq. B18)
  ct2 = bt_starsq/d;
  tsum = vaxsq + ct2 + twid_asq;
  tdif = vaxsq + ct2 - twid_asq;
  cf2_cs2 = std::sqrt(tdif*tdif + 4.0*twid_asq*ct2);

  cfsq = 0.5*(tsum + cf2_cs2);
  cf = std::sqrt(cfsq);

  cssq = twid_asq*vaxsq/cfsq;
  cs = std::sqrt(cssq);

  // Compute beta(s) (eqs. A17, B20, B28)
  bt = std::sqrt(btsq);
  bt_star = std::sqrt(bt_starsq);
  if (bt == 0.0) {
    bet2 = 1.0;
    bet3 = 0.0;
  } else {
    bet2 = b2/bt;
    bet3 = b3/bt;
  }
  bet2_star = bet2/std::sqrt(gm1 - (gm1-1.0)*y);
  bet3_star = bet3/std::sqrt(gm1 - (gm1-1.0)*y);
  bet_starsq = bet2_star*bet2_star + bet3_star*bet3_star;
  vbet = v2*bet2_star + v3*bet3_star;

  // Compute alpha(s) (eq. A16)
  if ((cfsq - cssq) == 0.0) {
    alpha_f = 1.0;
    alpha_s = 0.0;
  } else if ( (twid_asq - cssq) <= 0.0) {
    alpha_f = 0.0;
    alpha_s = 1.0;
  } else if ( (cfsq - twid_asq) <= 0.0) {
    alpha_f = 1.0;
    alpha_s = 0.0;
  } else {
    alpha_f = std::sqrt((twid_asq - cssq)/(cfsq - cssq));
    alpha_s = std::sqrt((cfsq - twid_asq)/(cfsq - cssq));
  }

  // Compute Q(s) and A(s) (eq. A14-15), etc.
  sqrtd = std::sqrt(d);
  isqrtd = 1.0/sqrtd;
  s = SIGN(b1);
  twid_a = std::sqrt(twid_asq);
  qf = cf*alpha_f*s;
  qs = cs*alpha_s*s;
  af_prime = twid_a*alpha_f*isqrtd;
  as_prime = twid_a*alpha_s*isqrtd;
  afpbb = af_prime*bt_star*bet_starsq;
  aspbb = as_prime*bt_star*bet_starsq;

  // Compute eigenvalues (eq. B17) - YH: these eqns from Stone2008 (Athena)
  vax = std::sqrt(vaxsq);
  eigenvalues[0] = v1 - cf;
  eigenvalues[1] = v1 - vax;
  eigenvalues[2] = v1 - cs;
  eigenvalues[3] = v1;
  eigenvalues[4] = v1 + cs;
  eigenvalues[5] = v1 + vax;
  eigenvalues[6] = v1 + cf;

  // Right-eigenvectors, stored as COLUMNS (eq. B21) */ YH: is row on Eq. B21
  right_eigenmatrix[0][0] = alpha_f;
  right_eigenmatrix[0][1] = 0.0;
  right_eigenmatrix[0][2] = alpha_s;
  right_eigenmatrix[0][3] = 1.0;
  right_eigenmatrix[0][4] = alpha_s;
  right_eigenmatrix[0][5] = 0.0;
  right_eigenmatrix[0][6] = alpha_f;

  right_eigenmatrix[1][0] = alpha_f*eigenvalues[0];
  right_eigenmatrix[1][1] = 0.0;
  right_eigenmatrix[1][2] = alpha_s*eigenvalues[2];
  right_eigenmatrix[1][3] = v1;
  right_eigenmatrix[1][4] = alpha_s*eigenvalues[4];
  right_eigenmatrix[1][5] = 0.0;
  right_eigenmatrix[1][6] = alpha_f*eigenvalues[6];

   qa = alpha_f*v2;
   qb = alpha_s*v2;
   qc = qs*bet2_star;
   qd = qf*bet2_star;
   right_eigenmatrix[2][0] = qa + qc;
   right_eigenmatrix[2][1] = -bet3;
   right_eigenmatrix[2][2] = qb - qd;
   right_eigenmatrix[2][3] = v2;
   right_eigenmatrix[2][4] = qb + qd;
   right_eigenmatrix[2][5] = bet3;
   right_eigenmatrix[2][6] = qa - qc;

   qa = alpha_f*v3;
   qb = alpha_s*v3;
   qc = qs*bet3_star;
   qd = qf*bet3_star;
   right_eigenmatrix[3][0] = qa + qc;
   right_eigenmatrix[3][1] = bet2;
   right_eigenmatrix[3][2] = qb - qd;
   right_eigenmatrix[3][3] = v3;
   right_eigenmatrix[3][4] = qb + qd;
   right_eigenmatrix[3][5] = -bet2;
   right_eigenmatrix[3][6] = qa - qc;

   right_eigenmatrix[4][0] = alpha_f*(hp - v1*cf) + qs*vbet + aspbb;
   right_eigenmatrix[4][1] = -(v2*bet3 - v3*bet2);
   right_eigenmatrix[4][2] = alpha_s*(hp - v1*cs) - qf*vbet - afpbb;
   right_eigenmatrix[4][3] = 0.5*vsq + (gm1-1.0)*x/gm1;
   right_eigenmatrix[4][4] = alpha_s*(hp + v1*cs) + qf*vbet - afpbb;
   right_eigenmatrix[4][5] = -right_eigenmatrix[4][1];
   right_eigenmatrix[4][6] = alpha_f*(hp + v1*cf) - qs*vbet + aspbb;

   right_eigenmatrix[5][0] = as_prime*bet2_star;
   right_eigenmatrix[5][1] = -bet3*s*isqrtd;
   right_eigenmatrix[5][2] = -af_prime*bet2_star;
   right_eigenmatrix[5][3] = 0.0;
   right_eigenmatrix[5][4] = right_eigenmatrix[5][2];
   right_eigenmatrix[5][5] = right_eigenmatrix[5][1];
   right_eigenmatrix[5][6] = right_eigenmatrix[5][0];

   right_eigenmatrix[6][0] = as_prime*bet3_star;
   right_eigenmatrix[6][1] = bet2*s*isqrtd;
   right_eigenmatrix[6][2] = -af_prime*bet3_star;
   right_eigenmatrix[6][3] = 0.0;
   right_eigenmatrix[6][4] = right_eigenmatrix[6][2];
   right_eigenmatrix[6][5] = right_eigenmatrix[6][1];
   right_eigenmatrix[6][6] = right_eigenmatrix[6][0];

   // Left-eigenvectors, stored as ROWS (eq. B29)
   // Normalize by 1/2a^{2}: quantities denoted by \hat{f}
   norm = 0.5/twid_asq;
   cff = norm*alpha_f*cf;
   css = norm*alpha_s*cs;
   qf *= norm;
   qs *= norm;
   af = norm*af_prime*d;
   as = norm*as_prime*d;
   afpb = norm*af_prime*bt_star;
   aspb = norm*as_prime*bt_star;

   // Normalize by (gamma-1)/2a^{2}: quantities denoted by \bar{f}
   norm *= gm1;
   alpha_f *= norm;
   alpha_s *= norm;
   q2_star = bet2_star/bet_starsq;
   q3_star = bet3_star/bet_starsq;
   vqstr = (v2*q2_star + v3*q3_star);
   norm *= 2.0;

   left_eigenmatrix[0][0] = alpha_f*(vsq-hp) + cff*(cf+v1) - qs*vqstr - aspb;
   left_eigenmatrix[0][1] = -alpha_f*v1 - cff;
   left_eigenmatrix[0][2] = -alpha_f*v2 + qs*q2_star;
   left_eigenmatrix[0][3] = -alpha_f*v3 + qs*q3_star;
   left_eigenmatrix[0][4] = alpha_f;
   left_eigenmatrix[0][5] = as*q2_star - alpha_f*b2;
   left_eigenmatrix[0][6] = as*q3_star - alpha_f*b3;

   left_eigenmatrix[1][0] = 0.5*(v2*bet3 - v3*bet2);
   left_eigenmatrix[1][1] = 0.0;
   left_eigenmatrix[1][2] = -0.5*bet3;
   left_eigenmatrix[1][3] = 0.5*bet2;
   left_eigenmatrix[1][4] = 0.0;
   left_eigenmatrix[1][5] = -0.5*sqrtd*bet3*s;
   left_eigenmatrix[1][6] = 0.5*sqrtd*bet2*s;

   left_eigenmatrix[2][0] = alpha_s*(vsq-hp) + css*(cs+v1) + qf*vqstr + afpb;
   left_eigenmatrix[2][1] = -alpha_s*v1 - css;
   left_eigenmatrix[2][2] = -alpha_s*v2 - qf*q2_star;
   left_eigenmatrix[2][3] = -alpha_s*v3 - qf*q3_star;
   left_eigenmatrix[2][4] = alpha_s;
   left_eigenmatrix[2][5] = -af*q2_star - alpha_s*b2;
   left_eigenmatrix[2][6] = -af*q3_star - alpha_s*b3;

   left_eigenmatrix[3][0] = 1.0 - norm*(0.5*vsq - (gm1-1.0)*x/gm1);
   left_eigenmatrix[3][1] = norm*v1;
   left_eigenmatrix[3][2] = norm*v2;
   left_eigenmatrix[3][3] = norm*v3;
   left_eigenmatrix[3][4] = -norm;
   left_eigenmatrix[3][5] = norm*b2;
   left_eigenmatrix[3][6] = norm*b3;

   left_eigenmatrix[4][0] = alpha_s*(vsq-hp) + css*(cs-v1) - qf*vqstr + afpb;
   left_eigenmatrix[4][1] = -alpha_s*v1 + css;
   left_eigenmatrix[4][2] = -alpha_s*v2 + qf*q2_star;
   left_eigenmatrix[4][3] = -alpha_s*v3 + qf*q3_star;
   left_eigenmatrix[4][4] = alpha_s;
   left_eigenmatrix[4][5] = left_eigenmatrix[2][5];
   left_eigenmatrix[4][6] = left_eigenmatrix[2][6];

   left_eigenmatrix[5][0] = -left_eigenmatrix[1][0];
   left_eigenmatrix[5][1] = 0.0;
   left_eigenmatrix[5][2] = -left_eigenmatrix[1][2];
   left_eigenmatrix[5][3] = -left_eigenmatrix[1][3];
   left_eigenmatrix[5][4] = 0.0;
   left_eigenmatrix[5][5] = left_eigenmatrix[1][5];
   left_eigenmatrix[5][6] = left_eigenmatrix[1][6];

   left_eigenmatrix[6][0] = alpha_f*(vsq-hp) + cff*(cf-v1) + qs*vqstr - aspb;
   left_eigenmatrix[6][1] = -alpha_f*v1 + cff;
   left_eigenmatrix[6][2] = -alpha_f*v2 - qs*q2_star;
   left_eigenmatrix[6][3] = -alpha_f*v3 - qs*q3_star;
   left_eigenmatrix[6][4] = alpha_f;
   left_eigenmatrix[6][5] = left_eigenmatrix[0][5];
   left_eigenmatrix[6][6] = left_eigenmatrix[0][6];
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWave_()
//! \brief Sets initial conditions for linear wave mhd tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  const Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;

  // read global parameters
  lwv.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  lwv.bscale = pin->GetOrAddReal("problem", "bscale", 1.0);
  lwv.pscale = pin->GetOrAddReal("problem", "pscale", 1.0);
  lwv.vscale = pin->GetOrAddReal("problem", "vscale", 1.0);
  lwv.wave_flag = pin->GetInteger("problem", "wave_flag");
  lwv.amp = pin->GetReal("problem", "amp");
  lwv.vflow = pin->GetOrAddReal("problem", "vflow", 0.0);
  bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", false);
  bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  const bool one_d = (ndim == 1);
  const bool two_d = (ndim == 2);
  const bool multi_d = (ndim > 1);
  const bool three_d = (ndim > 2);
  // error check input flags
  if ((along_x1 && (along_x2 || along_x3)) || (along_x2 && along_x3)) {
    PARTHENON_FAIL("Can only specify one of along_x1/2/3 to be true");
  }
  if ((along_x2 || along_x3) && one_d) {
    PARTHENON_FAIL("Cannot specify waves along x2 or x3 axis in 1D");
  }
  if (along_x3 && two_d) {
    PARTHENON_FAIL("Cannot specify waves along x3 axis in 2D");
  }
  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "linear_wave_mhd pgen requires Cartesian geometry!");

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR);
  Real x2size = pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR);
  Real x3size = pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR);

  // start with wavevector along x1 axis
  lwv.cos_a3 = 1.0;
  lwv.sin_a3 = 0.0;
  lwv.cos_a2 = 1.0;
  lwv.sin_a2 = 0.0;
  if (multi_d && !(along_x1)) {
    Real ang_3 = std::atan(x1size / x2size);
    lwv.sin_a3 = std::sin(ang_3);
    lwv.cos_a3 = std::cos(ang_3);
  }
  if (three_d && !(along_x1)) {
    Real ang_2 = std::atan(0.5 * (x1size * lwv.cos_a3 + x2size * lwv.sin_a3) / x3size);
    lwv.sin_a2 = std::sin(ang_2);
    lwv.cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 1.0;
    lwv.sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 0.0;
    lwv.sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  lwv.lambda = std::numeric_limits<float>::max();
  if (lwv.cos_a2 * lwv.cos_a3 > 0.0) {
    lwv.lambda = std::min(lwv.lambda, x1size * lwv.cos_a2 * lwv.cos_a3);
  }
  if (lwv.cos_a2 * lwv.sin_a3 > 0.0) {
    lwv.lambda = std::min(lwv.lambda, x2size * lwv.cos_a2 * lwv.sin_a3);
  }
  if (lwv.sin_a2 > 0.0) lwv.lambda = std::min(lwv.lambda, x3size * lwv.sin_a2);

  // Initialize k_parallel
  lwv.k_par = 2.0 * (M_PI) / lwv.lambda; // YH: Eq. 53 (GS08)

  // Set background state: v1_0 is parallel to wavevector.
  lwv.d0 = 1.0 * lwv.rho0;
  lwv.v1_0 = lwv.vflow * lwv.vscale;
  // TODO(PDM): Replace the below with a call to singularity-eos
  auto gas_pkg = pmb->packages.Get("gas");
  lwv.gamma = gas_pkg->Param<Real>("adiabatic_index");
  lwv.gm1 = lwv.gamma - 1.0;
  lwv.p0 = (1.0 * lwv.rho0 / lwv.gamma) * lwv.pscale;
  // YH: For MHD follow At * lwv.bscalehena++ instead of Gardiner 2008
  // YH: seems to be based on this https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html
  lwv.bx0 = 1.0 * sqrt(lwv.rho0) * lwv.bscale;
  lwv.by0 = std::sqrt(2.0) * sqrt(lwv.rho0) * lwv.bscale;
  lwv.bz0 = 0.5 * sqrt(lwv.rho0) * lwv.bscale;

  // Compute eigenvectors in hydrodynamics
  Real xfact = 0.0, yfact = 1.0; // YH: as no discontinuity so like athenak
  MagnetoHydroEigensystem(lwv.d0, lwv.v1_0, 0.0, 0.0, lwv.p0, lwv.gamma, lwv.ev, lwv.rem,
		          lwv.bx0, lwv.by0, lwv.bz0, xfact, yfact, lwv.lem);

  // set new time limit, interpreted as number of wave periods for evolution
  const Real nperiod = pin->GetOrAddReal("problem", "nperiod", 1.0);
  pin->SetReal("parthenon/time", "tlim",
               nperiod * (std::abs(lwv.lambda / lwv.ev[lwv.wave_flag])));

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
	  		 gas::prim::Pe>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  auto lin = lwv;

  pmb->par_for(
      "pgen_linwave1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xv = coords.GetCellCenter();
        const Real x1v = xv[0];
        const Real x2v = xv[1];
        const Real x3v = xv[2];
	// YH: x1 from Eq. 59 for Eq. 53 of GS08
        Real x = lin.cos_a2 * (x1v * lin.cos_a3 + x2v * lin.sin_a3) + x3v * lin.sin_a2;
        Real sn = std::sin(lin.k_par * x); // YH: but GS08 Eq. 53 uses cos instead of sin
        Real mx = lin.d0 * lin.vflow + lin.amp * sn * lin.rem[1][lin.wave_flag]; // YH: Eq. 53 (GS08)
        Real my = lin.amp * sn * lin.rem[2][lin.wave_flag]; // YH: no background for v2/v3
        Real mz = lin.amp * sn * lin.rem[3][lin.wave_flag];

        // compute cell-centered conserved variables
        const Real cd = lin.d0 + lin.amp * sn * lin.rem[0][lin.wave_flag]; // YH: Eq. 53 (GS08)
	// YH: Transform using Eq. 54 (GS08)
        const Real cm1 =
            mx * lin.cos_a2 * lin.cos_a3 - my * lin.sin_a3 - mz * lin.sin_a2 * lin.cos_a3;
        const Real cm2 =
            mx * lin.cos_a2 * lin.sin_a3 + my * lin.cos_a3 - mz * lin.sin_a2 * lin.sin_a3;
        const Real cm3 = mx * lin.sin_a2 + mz * lin.cos_a2;
	const Real bx = lin.bx0;
        const Real by = lin.by0 + lin.amp*sn*lin.rem[5][lin.wave_flag];
        const Real bz = lin.bz0 + lin.amp*sn*lin.rem[6][lin.wave_flag];
        const Real b1 = bx*lin.cos_a2*lin.cos_a3 - by*lin.sin_a3 - bz*lin.sin_a2*lin.cos_a3;
        const Real b2 = bx*lin.cos_a2*lin.sin_a3 + by*lin.cos_a3 - bz*lin.sin_a2*lin.sin_a3;
        const Real b3 = bx*lin.sin_a2                            + bz*lin.cos_a2;
        const Real ce = lin.p0 / lin.gm1 + 0.5 * lin.d0 * (lin.v1_0) * (lin.v1_0) +
                        lin.amp * sn * lin.rem[4][lin.wave_flag] +
			0.5 * (SQR(lin.bx0) + SQR(lin.by0) + SQR(lin.bz0));
        const Real cu = ce - 0.5 * (SQR(cm1) + SQR(cm2) + SQR(cm3)) / cd 
		           - 0.5 * (SQR(b1) + SQR(b2) + SQR(b3));
        v(0, gas::prim::density(), k, j, i) = cd;
        v(0, gas::prim::velocity(0), k, j, i) = cm1 / (cd);
        v(0, gas::prim::velocity(1), k, j, i) = cm2 / (cd);
        v(0, gas::prim::velocity(2), k, j, i) = cm3 / (cd);
        v(0, gas::prim::sie(), k, j, i) = cu / cd;
	v(0, gas::prim::Pe(), k, j, i) = 0.5 * cu * lwv.gm1; 
      });

  // YH: Add magnetic field here
  // 1) Initialize components of vector potential
  const int nghost = pin->GetInteger("parthenon/mesh", "nghost");
  const int nx1 = pin->GetInteger("parthenon/mesh", "nx1") + 2*nghost;
  const int nx2 = multi_d ? pin->GetInteger("parthenon/mesh", "nx2") + 2*nghost : 2;
  const int nx3 = three_d ? pin->GetInteger("parthenon/mesh", "nx3") + 2*nghost : 2;
  ParArrayND<Real> a1("a1", nx3, nx2, nx1);
  ParArrayND<Real> a2("a2", nx3, nx2, nx1);
  ParArrayND<Real> a3("a3", nx3, nx2, nx1);
  // wave amplitudes
  lin.dby = lin.amp * lin.rem[NWAVE-2][lin.wave_flag];
  lin.dbz = lin.amp * lin.rem[NWAVE-1][lin.wave_flag];
  // Initialize components of vector potential
  pmb->par_for(
      "pgen_linwave1", kb.s, kb.e+1, jb.s, jb.e+1, ib.s, ib.e+1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xv = coords.GetCellCenter();
        const Real x1v = xv[0];
        const Real x2v = xv[1];
        const Real x3v = xv[2];
	const auto &x1e = coords.GetEdgeCenterX1();
	const auto &x2e = coords.GetEdgeCenterX2();
	const auto &x3e = coords.GetEdgeCenterX3();
	if (i != ib.e+1)
	  a1(k, j, i) = A1(lin, x1e[0], x1e[1], x1e[2]);
	if (j != jb.e+1)  
	  a2(k, j, i) = A2(lin, x2e[0], x2e[1], x2e[2]);
	if (k != kb.e+1)
	  a3(k, j, i) = A3(lin, x3e[0], x3e[1], x3e[2]);
	});

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes},{parthenon::PDOpt::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "lw_mhdx", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto e2len = coords.GetEdgeLengthX2();
	const auto e3len = coords.GetEdgeLengthX3();
        vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = 
	    std::round((
		(a3(k, j+1, i) - a3(k, j, i))/e2len - 
		(a2(k+1, j, i) - a2(k, j, i))/e3len
	    ) * 1e10) / 1e10;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "lw_mhdy", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto e1len = coords.GetEdgeLengthX1();
        const auto e3len = coords.GetEdgeLengthX3();
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = 
	    std::round((
		(a1(k+1, j, i) - a1(k,j,i))/e3len -
                (a3(k, j, i+1) - a3(k,j,i))/e1len
	    ) * 1e10) / 1e10;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "lw_mhdz", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto e1len = coords.GetEdgeLengthX1();
        const auto e2len = coords.GetEdgeLengthX2();
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = 
	    std::round((
		(a2(k, j, i+1) - a2(k,j,i))/e1len -
                (a1(k, j+1, i) - a1(k,j,i))/e2len
	    ) * 1e10) / 1e10;
      });

  static auto desc_EJ =
      MakePackDescriptor<gas::edge::Efield,
      			 gas::edge::J>((pmb->resolved_packages).get());
  auto vEJ = desc_EJ.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E1);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E1);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E1, gas::edge::Efield(), k, j, i) = 0.;
	Real dBz_dy = -(vmag(0, TE::F3, 0, k, j - (ndim > 1 and j>0), i)
                        - vmag(0, TE::F3, 0, k, j, i)) / dx[1];
        Real dBy_dz = -(vmag(0, TE::F2, 0, k - (ndim > 2 and k>0), j, i)
                        - vmag(0, TE::F2, 0, k, j, i)) / dx[2];
        vEJ(0, TE::E1, gas::edge::J(), k, j, i) = dBz_dy - dBy_dz;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E2);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E2);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E2, gas::edge::Efield(), k, j, i) = 0.;
	Real dBz_dx = -(vmag(0, TE::F3, 0, k, j, i - (i>0))
                        - vmag(0, TE::F3, 0, k, j, i)) / dx[0];
        Real dBx_dz = -(vmag(0, TE::F1, 0, k - (ndim > 2 and k>0), j, i)
                        - vmag(0, TE::F1, 0, k, j, i)) / dx[2];
        vEJ(0, TE::E2, gas::edge::J(), k, j, i) = -dBz_dx + dBx_dz;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E3);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E3);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E3, gas::edge::Efield(), k, j, i) = 0.;
	Real dBy_dx = -(vmag(0, TE::F2, 0, k, j, i - (i>0))
                        - vmag(0, TE::F2, 0, k, j, i)) / dx[0];
        Real dBx_dy = -(vmag(0, TE::F1, 0, k, j - (ndim > 1 and j>0), i)
                        - vmag(0, TE::F1, 0, k, j, i)) / dx[1];
        vEJ(0, TE::E3, gas::edge::J(), k, j, i) = dBy_dx - dBx_dy;
      });

}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Computes errors in linear wave mhd solution by subtracting current solution from
//! ICs, and outputting errors to file. Problem must be run for an integer number of wave
//! periods.
template <Coordinates GEOM>
inline void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  using parthenon::MakePackDescriptor;
  const int nhydro = 5;
  const int nvars = nhydro + 3;

  // packing and capture variables for kernel
  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum,
                         gas::cons::total_energy,
			 gas::cons::Bfield>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto lin = lwv;

  ArtemisUtils::array_type<Real, nvars> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "LinearModesErrors", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    ArtemisUtils::array_type<Real, nvars> &lsum) {
        // Capture coordinates this Meshblock
        geometry::Coords<GEOM> coords(v.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        Real x1v = xv[0];
        Real x2v = xv[1];
        Real x3v = xv[2];
        Real vol = coords.Volume();

        Real x = lin.cos_a2 * (x1v * lin.cos_a3 + x2v * lin.sin_a3) + x3v * lin.sin_a2;
        Real sn = std::sin(lin.k_par * x);
        Real mx = lin.d0 * lin.vflow + lin.amp * sn * lin.rem[1][lin.wave_flag];
        Real my = lin.amp * sn * lin.rem[2][lin.wave_flag];
        Real mz = lin.amp * sn * lin.rem[3][lin.wave_flag];
        Real ca = lin.d0 + lin.amp * sn * lin.rem[0][lin.wave_flag];
        Real cm1 =
            mx * lin.cos_a2 * lin.cos_a3 - my * lin.sin_a3 - mz * lin.sin_a2 * lin.cos_a3;
        Real cm2 =
            mx * lin.cos_a2 * lin.sin_a3 + my * lin.cos_a3 - mz * lin.sin_a2 * lin.sin_a3;
        Real cm3 = mx * lin.sin_a2 + mz * lin.cos_a2;
        Real ce = lin.p0 / lin.gm1 + 0.5 * lin.d0 * (lin.v1_0) * (lin.v1_0) +
                  lin.amp * sn * lin.rem[4][lin.wave_flag] + 
		  0.5 * (SQR(lin.bx0) + SQR(lin.by0) + SQR(lin.bz0));

	Real bx = lin.bx0;
        Real by = lin.by0 + lin.amp*sn*lin.rem[5][lin.wave_flag];
        Real bz = lin.bz0 + lin.amp*sn*lin.rem[6][lin.wave_flag];
        Real b1 = bx*lin.cos_a2*lin.cos_a3 - by*lin.sin_a3 - bz*lin.sin_a2*lin.cos_a3;
        Real b2 = bx*lin.cos_a2*lin.sin_a3 + by*lin.cos_a3 - bz*lin.sin_a2*lin.sin_a3;
        Real b3 = bx*lin.sin_a2                            + bz*lin.cos_a2;

        // conserved variables:
        lsum.myArray[0] += vol * std::abs(v(b, gas::cons::density(0), k, j, i) - ca);
        lsum.myArray[1] += vol * std::abs(v(b, gas::cons::momentum(0), k, j, i) - cm1);
        lsum.myArray[2] += vol * std::abs(v(b, gas::cons::momentum(1), k, j, i) - cm2);
        lsum.myArray[3] += vol * std::abs(v(b, gas::cons::momentum(2), k, j, i) - cm3);
        lsum.myArray[4] += vol * std::abs(v(b, gas::cons::total_energy(0), k, j, i) - ce);
	lsum.myArray[5] += vol * std::abs(v(b, gas::cons::Bfield(0), k, j, i) - b1);
	lsum.myArray[6] += vol * std::abs(v(b, gas::cons::Bfield(1), k, j, i) - b2);
	lsum.myArray[7] += vol * std::abs(v(b, gas::cons::Bfield(2), k, j, i) - b3);
      },
      ArtemisUtils::SumMyArray<Real, Kokkos::HostSpace, nvars>(l1_err));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &(l1_err.myArray[0]), nvars, MPI_PARTHENON_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  // normalize errors by number of cells
  Real vol = (pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR)) *
             (pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR)) *
             (pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR));
  for (int i = 0; i < nvars; ++i)
    l1_err.myArray[i] = l1_err.myArray[i] / vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i = 0; i < nvars; ++i) {
    rms_err += SQR(l1_err.myArray[i]);
  }
  rms_err = std::sqrt(rms_err); // YH: Eq. 21 of Stone2020

  // root process opens output file and writes out errors
  if (parthenon::Globals::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("parthenon/job", "problem_id"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1         ");
      std::fprintf(pfile, "d_L1         M1_L1         M2_L1         M3_L1         E_L1");
      std::fprintf(pfile, "         B1c_L1         B2c_L1         B3c_L1");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmesh->mesh_size.nx(X1DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X3DIR));
    std::fprintf(pfile, "  %05d  %e ", tm.ncycle, rms_err);
    for (int i = 0; i < nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err.myArray[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicInnerX1()
//! \brief Sets BCs on -x boundary 
template <Coordinates GEOM>
inline void PeriodicInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
                                                 gas::source::mom,
                                                 gas::source::ener,
                                                 gas::source::Se>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int is = range.s;
  const int ie = range.e;
  const int shift = ie-is+1;
  pmb->par_for_bndry(
      "InflowInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i + shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
	// For source term
        for (int im=0; im<3; im++)
          v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, j, iref);
        v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, j, iref);
        v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, j, iref);
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int is_x = range_x.s;
  const int ie_x = range_x.e;
  const int shift_f1 = ie_x - is_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i + shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int is_y = range_y.s;
  const int ie_y = range_y.e;
  const int shift_f2 = ie_y - is_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = i + shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, j, iref);
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int is_z = range_z.s;
  const int ie_z = range_z.e;
  const int shift_f3 = ie_z - is_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i + shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, j, iref);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, j, iref);
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int isE_x = rangeE_x.s;
  const int ieE_x = rangeE_x.e;
  const int shift_e1 = ieE_x - isE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i + shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int isE_y = rangeE_y.s;
  const int ieE_y = rangeE_y.e;
  const int shift_e2 = ieE_y - isE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i + shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int isE_z = rangeE_z.s;
  const int ieE_z = rangeE_z.e;
  const int shift_e3 = ieE_z - isE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i + shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, j, iref);
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicOuterX1()
//! \brief Sets BCs on +x boundary 
template <Coordinates GEOM>
inline void PeriodicOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
                                                 gas::source::mom,
                                                 gas::source::ener,
                                                 gas::source::Se>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int is = range.s;
  const int ie = range.e;
  const int shift = ie-is+1;
  pmb->par_for_bndry(
      "InflowInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i - shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
	// For source term
        for (int im=0; im<3; im++)
          v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, j, iref);
        v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, j, iref);
        v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, j, iref);
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int is_x = range_x.s;
  const int ie_x = range_x.e;
  const int shift_f1 = ie_x - is_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i - shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int is_y = range_y.s;
  const int ie_y = range_y.e;
  const int shift_f2 = ie_y - is_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = i - shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, j, iref);
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int is_z = range_z.s;
  const int ie_z = range_z.e;
  const int shift_f3 = ie_z - is_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i - shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, j, iref);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, j, iref);
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int isE_x = rangeE_x.s;
  const int ieE_x = rangeE_x.e;
  const int shift_e1 = ieE_x - isE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i - shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int isE_y = rangeE_y.s;
  const int ieE_y = rangeE_y.e;
  const int shift_e2 = ieE_y - isE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i - shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int isE_z = rangeE_z.s;
  const int ieE_z = rangeE_z.e;
  const int shift_e3 = ieE_z - isE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i - shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, j, iref);
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicInnerX2()
//! \brief Sets BCs on -y boundary 
template <Coordinates GEOM>
inline void PeriodicInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
						 gas::source::mom,
						 gas::source::ener,
						 gas::source::Se>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;
  const int je = range.e;
  const int shift = je-js+1;
  pmb->par_for_bndry(
      "InflowInnerX2", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j + shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, jref, i);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, jref, i);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, jref, i);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, jref, i);
	// For source term
	for (int im=0; im<3; im++)
	  v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, jref, i);
	v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, jref, i);
	v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, jref, i);
      });
  
  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int js_x = range_x.s;
  const int je_x = range_x.e;
  const int shift_f1 = je_x - js_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j + shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, jref, i);
	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int js_y = range_y.s;
  const int je_y = range_y.e;
  const int shift_f2 = je_y - js_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x2, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = j + shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, jref, i);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, jref, i);
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int js_z = range_z.s;
  const int je_z = range_z.e;
  const int shift_f3 = je_z - js_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x2, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j + shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, jref, i);
  	});

  const auto &rangeE_x = bounds.GetBoundsJ(IndexDomain::interior, TE::E1);
  const int jsE_x = rangeE_x.s;
  const int jeE_x = rangeE_x.e;
  const int shift_e1 = jeE_x - jsE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int jref = j + shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_y = bounds.GetBoundsJ(IndexDomain::interior, TE::E2);
  const int jsE_y = rangeE_y.s;
  const int jeE_y = rangeE_y.e;
  const int shift_e2 = jeE_y - jsE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x2, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j + shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_z = bounds.GetBoundsJ(IndexDomain::interior, TE::E3);
  const int jsE_z = rangeE_z.s;
  const int jeE_z = rangeE_z.e;
  const int shift_e3 = jeE_z - jsE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x2, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j + shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, jref, i);
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicOuterX2()
//! \brief Sets BCs on +y boundary 
template <Coordinates GEOM>
inline void PeriodicOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
                                                 gas::source::mom,
                                                 gas::source::ener,
                                                 gas::source::Se>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;
  const int je = range.e;
  const int shift = je-js+1;
  pmb->par_for_bndry(
      "InflowInnerX2", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j - shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, jref, i);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, jref, i);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, jref, i);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, jref, i);
	// For source term
        for (int im=0; im<3; im++)
          v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, jref, i);
        v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, jref, i);
        v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, jref, i);
      });
  
  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int js_x = range_x.s;
  const int je_x = range_x.e;
  const int shift_f1 = je_x - js_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, jref, i);
	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int js_y = range_y.s;
  const int je_y = range_y.e;
  const int shift_f2 = je_y - js_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x2, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = j - shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, jref, i);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, jref, i);
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int js_z = range_z.s;
  const int je_z = range_z.e;
  const int shift_f3 = je_z - js_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x2, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j - shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, jref, i);
  	});

  const auto &rangeE_x = bounds.GetBoundsJ(IndexDomain::interior, TE::E1);
  const int jsE_x = rangeE_x.s;
  const int jeE_x = rangeE_x.e;
  const int shift_e1 = jeE_x - jsE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_y = bounds.GetBoundsJ(IndexDomain::interior, TE::E2);
  const int jsE_y = rangeE_y.s;
  const int jeE_y = rangeE_y.e;
  const int shift_e2 = jeE_y - jsE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x2, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_z = bounds.GetBoundsJ(IndexDomain::interior, TE::E3);
  const int jsE_z = rangeE_z.s;
  const int jeE_z = rangeE_z.e;
  const int shift_e3 = jeE_z - jsE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x2, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, jref, i);
        });
  return;
}


} // namespace linear_wave_mhd

#endif // PGEN_LINEAR_WAVE_MHD_HPP_
