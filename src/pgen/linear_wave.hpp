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
#include "utils/artemis_utils.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \struct LinWaveVariables
//! \brief container for variables shared with linear wave pgen and error functions
struct LinWaveVariables {
  int wave_flag;
  Real amp, vflow, lambda;
  Real d0, p0, v1_0, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  Real rem[5][5], ev[5];
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
                                             const Real gamma, Real eigenvalues[5],
                                             Real right_eigenmatrix[5][5]) {
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
//! \fn void ProblemGenerator::LinearWave_()
//! \brief Sets initial conditions for linear wave tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  const Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;

  // read global parameters
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
                    "linear_wave pgen requires Cartesian geometry!");

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
  lwv.gamma = gas_pkg->Param<Real>("adiabatic_index");
  lwv.gm1 = lwv.gamma - 1.0;
  lwv.p0 = 1.0 / lwv.gamma;

  // Compute eigenvectors in hydrodynamics
  HydroEigensystem(lwv.d0, lwv.v1_0, 0.0, 0.0, lwv.p0, lwv.gamma, lwv.ev, lwv.rem);

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
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
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
        const Real ce = lin.p0 / lin.gm1 + 0.5 * lin.d0 * (lin.v1_0) * (lin.v1_0) +
                        lin.amp * sn * lin.rem[4][lin.wave_flag];
        const Real cu = ce - 0.5 * (SQR(cm1) + SQR(cm2) + SQR(cm3)) / cd;
        v(0, gas::prim::density(), k, j, i) = cd;
        v(0, gas::prim::velocity(0), k, j, i) = cm1 / cd;
        v(0, gas::prim::velocity(1), k, j, i) = cm2 / cd;
        v(0, gas::prim::velocity(2), k, j, i) = cm3 / cd;
        v(0, gas::prim::sie(), k, j, i) = cu / cd;
      });
}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Computes errors in linear wave solution by subtracting current solution from
//! ICs, and outputting errors to file. Problem must be run for an integer number of wave
//! periods.
template <Coordinates GEOM>
inline void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
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
                  lin.amp * sn * lin.rem[4][lin.wave_flag];

        // conserved variables:
        lsum.myArray[0] += vol * std::abs(v(b, gas::cons::density(0), k, j, i) - ca);
        lsum.myArray[1] += vol * std::abs(v(b, gas::cons::momentum(0), k, j, i) - cm1);
        lsum.myArray[2] += vol * std::abs(v(b, gas::cons::momentum(1), k, j, i) - cm2);
        lsum.myArray[3] += vol * std::abs(v(b, gas::cons::momentum(2), k, j, i) - cm3);
        lsum.myArray[4] += vol * std::abs(v(b, gas::cons::total_energy(0), k, j, i) - ce);
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
