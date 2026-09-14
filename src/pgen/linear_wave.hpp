//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef PGEN_LINEAR_WAVE_HPP_
#define PGEN_LINEAR_WAVE_HPP_
//! \file linear_wave.hpp
//! \brief Linear wave problem generator for 1D/2D/3D problems. Direction of the w
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
#include "mhd/mhd.hpp"
#include "utils/artemis_utils.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \struct LinWaveVariables
//! \brief container for variables shared with linear wave pgen and error functions
struct LinWaveVariables {
  int wave_flag;
  bool do_mhd;
  Real amp, vflow, lambda;
  Real d0, p0, v1_0, k_par;
  Real bx0, by0, bz0, dby, dbz, mu0;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  Real rem[7][7], ev[7];
  Real gamma, gm1;
};

} // end anonymous namespace

namespace linear_wave {

static LinWaveVariables lwv;

//----------------------------------------------------------------------------------------
//! \fn void HydroEigensystem()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal hydrodynamics
KOKKOS_INLINE_FUNCTION void HydroEigensystem(const Real d, const Real v1, const Real v2,
                                             const Real v3, const Real p,
                                             const Real gamma, Real eigenvalues[7],
                                             Real right_eigenmatrix[7][7]) {
  //--- Ideal Gas Hydrodynamics ---
  Real vsq = v1 * v1 + v2 * v2 + v3 * v3;
  Real h = (p / (gamma - 1.0) + 0.5 * d * vsq + p) / d;
  Real a = std::sqrt(gamma * p / d);

  // Compute eigenvalues (eq. B2)
  eigenvalues[0] = v1 - a;
  eigenvalues[1] = v1;
  eigenvalues[2] = v1;
  eigenvalues[3] = v1;
  eigenvalues[4] = v1 + a;

  // Right-eigenvectors, stored as COLUMNS (eq. B3)
  right_eigenmatrix[0][0] = 1.0;
  right_eigenmatrix[1][0] = v1 - a;
  right_eigenmatrix[2][0] = v2;
  right_eigenmatrix[3][0] = v3;
  right_eigenmatrix[4][0] = h - v1 * a;

  right_eigenmatrix[0][1] = 0.0;
  right_eigenmatrix[1][1] = 0.0;
  right_eigenmatrix[2][1] = 1.0;
  right_eigenmatrix[3][1] = 0.0;
  right_eigenmatrix[4][1] = v2;

  right_eigenmatrix[0][2] = 0.0;
  right_eigenmatrix[1][2] = 0.0;
  right_eigenmatrix[2][2] = 0.0;
  right_eigenmatrix[3][2] = 1.0;
  right_eigenmatrix[4][2] = v3;

  right_eigenmatrix[0][3] = 1.0;
  right_eigenmatrix[1][3] = v1;
  right_eigenmatrix[2][3] = v2;
  right_eigenmatrix[3][3] = v3;
  right_eigenmatrix[4][3] = 0.5 * vsq;

  right_eigenmatrix[0][4] = 1.0;
  right_eigenmatrix[1][4] = v1 + a;
  right_eigenmatrix[2][4] = v2;
  right_eigenmatrix[3][4] = v3;
  right_eigenmatrix[4][4] = h + v1 * a;
}

//----------------------------------------------------------------------------------------
//! \brief Compute the seven-wave right eigensystem of adiabatic ideal MHD.
//!
//! Magnetic fields are normalized by sqrt(mu0). Eigenvectors are stored as columns in
//! the conserved-variable ordering (rho, M1, M2, M3, E, B2, B3). This is adapted from
//! the open-source Athena++ linear-wave problem generator.
KOKKOS_INLINE_FUNCTION void MHDEigensystem(const Real d, const Real v1, const Real v2,
                                           const Real v3, const Real h, const Real b1,
                                           const Real b2, const Real b3, const Real gamma,
                                           Real eigenvalues[7],
                                           Real right_eigenmatrix[7][7]) {
  const Real gm1 = gamma - 1.0;
  const Real vsq = v1 * v1 + v2 * v2 + v3 * v3;
  const Real btsq = b2 * b2 + b3 * b3;
  const Real vaxsq = b1 * b1 / d;
  const Real hp = h - (vaxsq + btsq / d);
  const Real asq = std::max(gm1 * (hp - 0.5 * vsq), 1.0e-20);

  const Real ct2 = btsq / d;
  const Real tsum = vaxsq + ct2 + asq;
  const Real tdif = vaxsq + ct2 - asq;
  const Real cf2_cs2 = std::sqrt(tdif * tdif + 4.0 * asq * ct2);
  const Real cfsq = 0.5 * (tsum + cf2_cs2);
  const Real cf = std::sqrt(cfsq);
  const Real cssq = asq * vaxsq / cfsq;
  const Real cs = std::sqrt(cssq);

  const Real bt = std::sqrt(btsq);
  Real bet2, bet3;
  if (bt == 0.0) {
    bet2 = 1.0;
    bet3 = 0.0;
  } else {
    bet2 = b2 / bt;
    bet3 = b3 / bt;
  }

  Real alpha_f, alpha_s;
  if (cfsq == cssq) {
    alpha_f = 1.0;
    alpha_s = 0.0;
  } else if (asq <= cssq) {
    alpha_f = 0.0;
    alpha_s = 1.0;
  } else if (cfsq <= asq) {
    alpha_f = 1.0;
    alpha_s = 0.0;
  } else {
    alpha_f = std::sqrt((asq - cssq) / (cfsq - cssq));
    alpha_s = std::sqrt((cfsq - asq) / (cfsq - cssq));
  }

  const Real isqrtd = 1.0 / std::sqrt(d);
  const Real s = (b1 >= 0.0) ? 1.0 : -1.0;
  const Real a = std::sqrt(asq);
  const Real qf = cf * alpha_f * s;
  const Real qs = cs * alpha_s * s;
  const Real af_prime = a * alpha_f * isqrtd;
  const Real as_prime = a * alpha_s * isqrtd;
  const Real vbet = v2 * bet2 + v3 * bet3;

  eigenvalues[0] = v1 - cf;
  eigenvalues[1] = v1 - std::sqrt(vaxsq);
  eigenvalues[2] = v1 - cs;
  eigenvalues[3] = v1;
  eigenvalues[4] = v1 + cs;
  eigenvalues[5] = v1 + std::sqrt(vaxsq);
  eigenvalues[6] = v1 + cf;

  right_eigenmatrix[0][0] = alpha_f;
  right_eigenmatrix[0][1] = 0.0;
  right_eigenmatrix[0][2] = alpha_s;
  right_eigenmatrix[0][3] = 1.0;
  right_eigenmatrix[0][4] = alpha_s;
  right_eigenmatrix[0][5] = 0.0;
  right_eigenmatrix[0][6] = alpha_f;

  right_eigenmatrix[1][0] = alpha_f * eigenvalues[0];
  right_eigenmatrix[1][1] = 0.0;
  right_eigenmatrix[1][2] = alpha_s * eigenvalues[2];
  right_eigenmatrix[1][3] = v1;
  right_eigenmatrix[1][4] = alpha_s * eigenvalues[4];
  right_eigenmatrix[1][5] = 0.0;
  right_eigenmatrix[1][6] = alpha_f * eigenvalues[6];

  right_eigenmatrix[2][0] = alpha_f * v2 + qs * bet2;
  right_eigenmatrix[2][1] = -bet3;
  right_eigenmatrix[2][2] = alpha_s * v2 - qf * bet2;
  right_eigenmatrix[2][3] = v2;
  right_eigenmatrix[2][4] = alpha_s * v2 + qf * bet2;
  right_eigenmatrix[2][5] = bet3;
  right_eigenmatrix[2][6] = alpha_f * v2 - qs * bet2;

  right_eigenmatrix[3][0] = alpha_f * v3 + qs * bet3;
  right_eigenmatrix[3][1] = bet2;
  right_eigenmatrix[3][2] = alpha_s * v3 - qf * bet3;
  right_eigenmatrix[3][3] = v3;
  right_eigenmatrix[3][4] = alpha_s * v3 + qf * bet3;
  right_eigenmatrix[3][5] = -bet2;
  right_eigenmatrix[3][6] = alpha_f * v3 - qs * bet3;

  right_eigenmatrix[4][0] = alpha_f * (hp - v1 * cf) + qs * vbet + as_prime * bt;
  right_eigenmatrix[4][1] = -(v2 * bet3 - v3 * bet2);
  right_eigenmatrix[4][2] = alpha_s * (hp - v1 * cs) - qf * vbet - af_prime * bt;
  right_eigenmatrix[4][3] = 0.5 * vsq;
  right_eigenmatrix[4][4] = alpha_s * (hp + v1 * cs) + qf * vbet - af_prime * bt;
  right_eigenmatrix[4][5] = -right_eigenmatrix[4][1];
  right_eigenmatrix[4][6] = alpha_f * (hp + v1 * cf) - qs * vbet + as_prime * bt;

  right_eigenmatrix[5][0] = as_prime * bet2;
  right_eigenmatrix[5][1] = -bet3 * s * isqrtd;
  right_eigenmatrix[5][2] = -af_prime * bet2;
  right_eigenmatrix[5][3] = 0.0;
  right_eigenmatrix[5][4] = right_eigenmatrix[5][2];
  right_eigenmatrix[5][5] = right_eigenmatrix[5][1];
  right_eigenmatrix[5][6] = right_eigenmatrix[5][0];

  right_eigenmatrix[6][0] = as_prime * bet3;
  right_eigenmatrix[6][1] = bet2 * s * isqrtd;
  right_eigenmatrix[6][2] = -af_prime * bet3;
  right_eigenmatrix[6][3] = 0.0;
  right_eigenmatrix[6][4] = right_eigenmatrix[6][2];
  right_eigenmatrix[6][5] = right_eigenmatrix[6][1];
  right_eigenmatrix[6][6] = right_eigenmatrix[6][0];
}

KOKKOS_INLINE_FUNCTION Real A1(const LinWaveVariables &lin, const Real x1, const Real x2,
                               const Real x3) {
  const Real x = lin.cos_a2 * (x1 * lin.cos_a3 + x2 * lin.sin_a3) + x3 * lin.sin_a2;
  const Real y = -x1 * lin.sin_a3 + x2 * lin.cos_a3;
  const Real ay = lin.bz0 * x - (lin.dbz / lin.k_par) * std::cos(lin.k_par * x);
  const Real az =
      -lin.by0 * x + (lin.dby / lin.k_par) * std::cos(lin.k_par * x) + lin.bx0 * y;
  return -ay * lin.sin_a3 - az * lin.sin_a2 * lin.cos_a3;
}

KOKKOS_INLINE_FUNCTION Real A2(const LinWaveVariables &lin, const Real x1, const Real x2,
                               const Real x3) {
  const Real x = lin.cos_a2 * (x1 * lin.cos_a3 + x2 * lin.sin_a3) + x3 * lin.sin_a2;
  const Real y = -x1 * lin.sin_a3 + x2 * lin.cos_a3;
  const Real ay = lin.bz0 * x - (lin.dbz / lin.k_par) * std::cos(lin.k_par * x);
  const Real az =
      -lin.by0 * x + (lin.dby / lin.k_par) * std::cos(lin.k_par * x) + lin.bx0 * y;
  return ay * lin.cos_a3 - az * lin.sin_a2 * lin.sin_a3;
}

KOKKOS_INLINE_FUNCTION Real A3(const LinWaveVariables &lin, const Real x1, const Real x2,
                               const Real x3) {
  const Real x = lin.cos_a2 * (x1 * lin.cos_a3 + x2 * lin.sin_a3) + x3 * lin.sin_a2;
  const Real y = -x1 * lin.sin_a3 + x2 * lin.cos_a3;
  const Real az =
      -lin.by0 * x + (lin.dby / lin.k_par) * std::cos(lin.k_par * x) + lin.bx0 * y;
  return az * lin.cos_a2;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWave_()
//! \brief Sets initial conditions for linear wave tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  const Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;
  auto artemis_pkg = pmb->packages.Get("artemis");
  lwv.do_mhd = artemis_pkg->Param<bool>("do_mhd");

  // read global parameters
  lwv.wave_flag = pin->GetInteger("problem", "wave_flag");
  const int nwaves = lwv.do_mhd ? 7 : 5;
  PARTHENON_REQUIRE(lwv.wave_flag >= 0 && lwv.wave_flag < nwaves,
                    "problem/wave_flag is outside the range supported by linear_wave!");
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
                    "linear_wave pgen requires Cartesian geometry!");
  if (lwv.do_mhd) {
    PARTHENON_REQUIRE(!pmesh->multilevel,
                      "MHD linear waves currently require a uniform mesh!");
    PARTHENON_REQUIRE(artemis_pkg->Param<bool>("do_gas"),
                      "MHD linear waves require gas hydrodynamics!");
    PARTHENON_REQUIRE(!artemis_pkg->Param<bool>("do_dust"),
                      "MHD linear waves do not support dust!");
  }

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
  lwv.k_par = 2.0 * (M_PI) / lwv.lambda;

  // Set background state: v1_0 is parallel to wavevector.
  lwv.d0 = 1.0;
  lwv.v1_0 = lwv.vflow;
  // TODO(PDM): Replace the below with a call to singularity-eos
  auto gas_pkg = pmb->packages.Get("gas");
  PARTHENON_REQUIRE(gas_pkg->Param<std::string>("eos_type") == "ideal",
                    "linear_wave pgen requires an ideal gas");
  if (lwv.do_mhd) {
    PARTHENON_REQUIRE(gas_pkg->Param<int>("nspecies") == 1,
                      "MHD linear waves require one gas species!");
  }
  lwv.gamma = gas_pkg->Param<Real>("adiabatic_index");
  lwv.gm1 = lwv.gamma - 1.0;
  lwv.p0 = 1.0 / lwv.gamma;

  if (lwv.do_mhd) {
    const Real bx0_norm = 1.0;
    const Real by0_norm = std::sqrt(2.0);
    const Real bz0_norm = 0.5;
    const Real h0 = (lwv.p0 / lwv.gm1 + 0.5 * lwv.d0 * SQR(lwv.v1_0) + lwv.p0 +
                     SQR(bx0_norm) + SQR(by0_norm) + SQR(bz0_norm)) /
                    lwv.d0;
    MHDEigensystem(lwv.d0, lwv.v1_0, 0.0, 0.0, h0, bx0_norm, by0_norm, bz0_norm,
                   lwv.gamma, lwv.ev, lwv.rem);

    lwv.mu0 = pmb->packages.Get("mhd")->Param<Real>("mu0_code");
    const Real sqrt_mu0 = std::sqrt(lwv.mu0);
    lwv.bx0 = sqrt_mu0 * bx0_norm;
    lwv.by0 = sqrt_mu0 * by0_norm;
    lwv.bz0 = sqrt_mu0 * bz0_norm;
    lwv.dby = sqrt_mu0 * lwv.amp * lwv.rem[5][lwv.wave_flag];
    lwv.dbz = sqrt_mu0 * lwv.amp * lwv.rem[6][lwv.wave_flag];
  } else {
    HydroEigensystem(lwv.d0, lwv.v1_0, 0.0, 0.0, lwv.p0, lwv.gamma, lwv.ev, lwv.rem);
  }

  // set new time limit, interpreted as number of wave periods for evolution
  const Real nperiod = pin->GetOrAddReal("problem", "nperiod", 1.0);
  PARTHENON_REQUIRE(std::abs(lwv.ev[lwv.wave_flag]) >
                        std::numeric_limits<Real>::epsilon(),
                    "Selected linear wave has zero phase speed; set problem/vflow for "
                    "the entropy mode!");
  pin->SetReal("parthenon/time", "tlim",
               nperiod * (std::abs(lwv.lambda / lwv.ev[lwv.wave_flag])));

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         field::face::B>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v>((pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());
  auto &pco = pmb->coords;
  auto lin = lwv;
  const auto &cpars =
      pmb->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  if (lwv.do_mhd) {
    IndexRange ib1 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
    IndexRange jb1 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
    IndexRange kb1 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
    pmb->par_for(
        "pgen_linwave_b1", kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x1 = pco.template Xf<X1DIR>(i);
          const Real x2l = pco.template Xf<X2DIR>(j);
          const Real x2r = pco.template Xf<X2DIR>(j + 1);
          const Real x2c = pco.template Xc<X2DIR>(j);
          const Real x3l = pco.template Xf<X3DIR>(k);
          const Real x3r = pco.template Xf<X3DIR>(k + 1);
          const Real x3c = pco.template Xc<X3DIR>(k);
          v(0, TE::F1, field::face::B(), k, j, i) =
              (A3(lin, x1, x2r, x3c) - A3(lin, x1, x2l, x3c)) / (x2r - x2l) -
              (A2(lin, x1, x2c, x3r) - A2(lin, x1, x2c, x3l)) / (x3r - x3l);
        });

    IndexRange ib2 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2);
    IndexRange jb2 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
    IndexRange kb2 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
    pmb->par_for(
        "pgen_linwave_b2", kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x1l = pco.template Xf<X1DIR>(i);
          const Real x1r = pco.template Xf<X1DIR>(i + 1);
          const Real x1c = pco.template Xc<X1DIR>(i);
          const Real x2 = pco.template Xf<X2DIR>(j);
          const Real x3l = pco.template Xf<X3DIR>(k);
          const Real x3r = pco.template Xf<X3DIR>(k + 1);
          const Real x3c = pco.template Xc<X3DIR>(k);
          v(0, TE::F2, field::face::B(), k, j, i) =
              (A1(lin, x1c, x2, x3r) - A1(lin, x1c, x2, x3l)) / (x3r - x3l) -
              (A3(lin, x1r, x2, x3c) - A3(lin, x1l, x2, x3c)) / (x1r - x1l);
        });

    IndexRange ib3 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);
    IndexRange jb3 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
    IndexRange kb3 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
    pmb->par_for(
        "pgen_linwave_b3", kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x1l = pco.template Xf<X1DIR>(i);
          const Real x1r = pco.template Xf<X1DIR>(i + 1);
          const Real x1c = pco.template Xc<X1DIR>(i);
          const Real x2l = pco.template Xf<X2DIR>(j);
          const Real x2r = pco.template Xf<X2DIR>(j + 1);
          const Real x2c = pco.template Xc<X2DIR>(j);
          const Real x3 = pco.template Xf<X3DIR>(k);
          v(0, TE::F3, field::face::B(), k, j, i) =
              (A2(lin, x1r, x2c, x3) - A2(lin, x1l, x2c, x3)) / (x1r - x1l) -
              (A1(lin, x1c, x2r, x3) - A1(lin, x1c, x2l, x3)) / (x2r - x2l);
        });
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "pgen_linwave1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto &xv = coords.GetCellCenter(vg, 0, k, j, i);
        const Real x1v = xv[0];
        const Real x2v = xv[1];
        const Real x3v = xv[2];
        Real x = lin.cos_a2 * (x1v * lin.cos_a3 + x2v * lin.sin_a3) + x3v * lin.sin_a2;
        Real sn = std::sin(lin.k_par * x);
        Real mx = lin.d0 * lin.vflow + lin.amp * sn * lin.rem[1][lin.wave_flag];
        Real my = lin.amp * sn * lin.rem[2][lin.wave_flag];
        Real mz = lin.amp * sn * lin.rem[3][lin.wave_flag];

        // compute cell-centered conserved variables
        const Real cd = lin.d0 + lin.amp * sn * lin.rem[0][lin.wave_flag];
        const Real cm1 =
            mx * lin.cos_a2 * lin.cos_a3 - my * lin.sin_a3 - mz * lin.sin_a2 * lin.cos_a3;
        const Real cm2 =
            mx * lin.cos_a2 * lin.sin_a3 + my * lin.cos_a3 - mz * lin.sin_a2 * lin.sin_a3;
        const Real cm3 = mx * lin.sin_a2 + mz * lin.cos_a2;
        Real ce = lin.p0 / lin.gm1 + 0.5 * lin.d0 * (lin.v1_0) * (lin.v1_0) +
                  lin.amp * sn * lin.rem[4][lin.wave_flag];
        const Real ke = 0.5 * (SQR(cm1) + SQR(cm2) + SQR(cm3)) / cd;
        Real me = 0.0;
        if (lin.do_mhd) {
          const Real bx = 0.5 * (v(0, TE::F1, field::face::B(), k, j, i) +
                                 v(0, TE::F1, field::face::B(), k, j, i + 1));
          const Real by = 0.5 * (v(0, TE::F2, field::face::B(), k, j, i) +
                                 v(0, TE::F2, field::face::B(), k, j + multi_d, i));
          const Real bz = 0.5 * (v(0, TE::F3, field::face::B(), k, j, i) +
                                 v(0, TE::F3, field::face::B(), k + three_d, j, i));
          me = MHD::MagneticEnergyDensity(bx, by, bz, lin.mu0);
          ce += 0.5 * (SQR(lin.bx0) + SQR(lin.by0) + SQR(lin.bz0)) / lin.mu0;
        }
        v(0, gas::prim::density(), k, j, i) = cd;
        v(0, gas::prim::velocity(0), k, j, i) = cm1 / cd;
        v(0, gas::prim::velocity(1), k, j, i) = cm2 / cd;
        v(0, gas::prim::velocity(2), k, j, i) = cm3 / cd;
        v(0, gas::prim::sie(), k, j, i) = (ce - ke - me) / cd;
      });
}

//----------------------------------------------------------------------------------------
//! \brief Compute volume-weighted L1 errors for an MHD linear wave.
template <Coordinates GEOM>
inline void MHDUserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin,
                                 parthenon::SimTime &tm) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  constexpr int nvars = 8;

  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         field::cell::B, field::cell::divB>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  static auto desc_g = MakePackDescriptor<geom::vol, geom::x1v, geom::x2v, geom::x3v>(
      (pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto lin = lwv;
  const auto &cpars =
      pmesh->packages.Get("artemis")->Param<geometry::CoordParams>("coord_params");

  ArtemisUtils::array_type<Real, nvars> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MHDLinearModesErrors", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    ArtemisUtils::array_type<Real, nvars> &lsum) {
        geometry::Coords<GEOM> coords(cpars, v.GetCoordinates(b), k, j, i);
        const auto xv = coords.GetCellCenter(vg, b, k, j, i);
        const Real vol = coords.GetVolume(vg, b, k, j, i);
        const Real x =
            lin.cos_a2 * (xv[0] * lin.cos_a3 + xv[1] * lin.sin_a3) + xv[2] * lin.sin_a2;
        const Real sn = std::sin(lin.k_par * x);
        const Real ca = lin.d0 + lin.amp * sn * lin.rem[0][lin.wave_flag];
        const Real mx = lin.d0 * lin.vflow + lin.amp * sn * lin.rem[1][lin.wave_flag];
        const Real my = lin.amp * sn * lin.rem[2][lin.wave_flag];
        const Real mz = lin.amp * sn * lin.rem[3][lin.wave_flag];
        const Real cm1 =
            mx * lin.cos_a2 * lin.cos_a3 - my * lin.sin_a3 - mz * lin.sin_a2 * lin.cos_a3;
        const Real cm2 =
            mx * lin.cos_a2 * lin.sin_a3 + my * lin.cos_a3 - mz * lin.sin_a2 * lin.sin_a3;
        const Real cm3 = mx * lin.sin_a2 + mz * lin.cos_a2;
        const Real ce = lin.p0 / lin.gm1 + 0.5 * lin.d0 * SQR(lin.v1_0) +
                        0.5 * (SQR(lin.bx0) + SQR(lin.by0) + SQR(lin.bz0)) / lin.mu0 +
                        lin.amp * sn * lin.rem[4][lin.wave_flag];
        const Real by = lin.by0 + lin.dby * sn;
        const Real bz = lin.bz0 + lin.dbz * sn;
        const Real cb1 = lin.bx0 * lin.cos_a2 * lin.cos_a3 - by * lin.sin_a3 -
                         bz * lin.sin_a2 * lin.cos_a3;
        const Real cb2 = lin.bx0 * lin.cos_a2 * lin.sin_a3 + by * lin.cos_a3 -
                         bz * lin.sin_a2 * lin.sin_a3;
        const Real cb3 = lin.bx0 * lin.sin_a2 + bz * lin.cos_a2;

        lsum.myArray[0] += vol * std::abs(v(b, gas::cons::density(0), k, j, i) - ca);
        lsum.myArray[1] += vol * std::abs(v(b, gas::cons::momentum(0), k, j, i) - cm1);
        lsum.myArray[2] += vol * std::abs(v(b, gas::cons::momentum(1), k, j, i) - cm2);
        lsum.myArray[3] += vol * std::abs(v(b, gas::cons::momentum(2), k, j, i) - cm3);
        lsum.myArray[4] += vol * std::abs(v(b, gas::cons::total_energy(0), k, j, i) - ce);
        lsum.myArray[5] += vol * std::abs(v(b, field::cell::B(0), k, j, i) - cb1);
        lsum.myArray[6] += vol * std::abs(v(b, field::cell::B(1), k, j, i) - cb2);
        lsum.myArray[7] += vol * std::abs(v(b, field::cell::B(2), k, j, i) - cb3);
      },
      ArtemisUtils::SumMyArray<Real, Kokkos::HostSpace, nvars>(l1_err));

  Real max_divb = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "MHDLinearModesDivB", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        lmax = std::max(lmax, std::abs(v(b, field::cell::divB(), k, j, i)));
      },
      Kokkos::Max<Real>(max_divb));

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &(l1_err.myArray[0]), nvars, MPI_PARTHENON_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_divb, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  const Real volume = (pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR)) *
                      (pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR)) *
                      (pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR));
  Real rms_err = 0.0;
  for (int n = 0; n < nvars; ++n) {
    l1_err.myArray[n] /= volume;
    rms_err += SQR(l1_err.myArray[n]);
  }
  rms_err = std::sqrt(rms_err);

  if (parthenon::Globals::my_rank == 0) {
    std::string fname = pin->GetString("parthenon/job", "problem_id") + "-errs.dat";
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr)
        PARTHENON_FAIL("MHD linear-wave error output file could not be opened");
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr)
        PARTHENON_FAIL("MHD linear-wave error output file could not be opened");
      std::fprintf(pfile, "# Nx1 Nx2 Nx3 Ncycle RMS-L1 d_L1 M1_L1 M2_L1 M3_L1 ");
      std::fprintf(pfile, "E_L1 B1c_L1 B2c_L1 B3c_L1 max_divB\n");
    }
    std::fprintf(pfile, "%04d  %04d  %04d  %05d  %e", pmesh->mesh_size.nx(X1DIR),
                 pmesh->mesh_size.nx(X2DIR), pmesh->mesh_size.nx(X3DIR), tm.ncycle,
                 rms_err);
    for (int n = 0; n < nvars; ++n)
      std::fprintf(pfile, "  %e", l1_err.myArray[n]);
    std::fprintf(pfile, "  %e\n", max_divb);
    std::fclose(pfile);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Computes errors in linear wave solution by subtracting current solution from
//! ICs, and outputting errors to file. Problem must be run for an integer number of wave
//! periods.
template <Coordinates GEOM>
inline void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  PARTHENON_INSTRUMENT
  if (lwv.do_mhd) {
    MHDUserWorkAfterLoop<GEOM>(pmesh, pin, tm);
    return;
  }
  using parthenon::MakePackDescriptor;
  const int nhydro = 5;
  const int nvars = nhydro;

  // packing and capture variables for kernel
  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum,
                         gas::cons::total_energy>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  static auto desc_g = MakePackDescriptor<geom::vol, geom::x1v, geom::x2v, geom::x3v>(
      (pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto lin = lwv;
  const auto &cpars =
      pmesh->packages.Get("artemis")->template Param<geometry::CoordParams>(
          "coord_params");

  ArtemisUtils::array_type<Real, nvars> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "LinearModesErrors", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    ArtemisUtils::array_type<Real, nvars> &lsum) {
        // Capture coordinates this Meshblock
        geometry::Coords<GEOM> coords(cpars, v.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter(vg, b, k, j, i);
        Real x1v = xv[0];
        Real x2v = xv[1];
        Real x3v = xv[2];
        Real vol = coords.GetVolume(vg, b, k, j, i);

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
                  lin.amp * sn * lin.rem[4][lin.wave_flag];

        // conserved variables:
        lsum.myArray[0] += vol * std::abs(v(b, gas::cons::density(0), k, j, i) - ca);
        lsum.myArray[1] += vol * std::abs(v(b, gas::cons::momentum(0), k, j, i) - cm1);
        lsum.myArray[2] += vol * std::abs(v(b, gas::cons::momentum(1), k, j, i) - cm2);
        lsum.myArray[3] += vol * std::abs(v(b, gas::cons::momentum(2), k, j, i) - cm3);
        lsum.myArray[4] += vol * std::abs(v(b, gas::cons::total_energy(0), k, j, i) - ce);
      },
      ArtemisUtils::SumMyArray<Real, Kokkos::HostSpace, nvars>(l1_err));

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
  rms_err = std::sqrt(rms_err);

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
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1       ");
      std::fprintf(pfile, "d_L1         M1_L1         M2_L1         M3_L1         E_L1");
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

} // namespace linear_wave

#endif // PGEN_LINEAR_WAVE_HPP_
