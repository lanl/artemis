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
#ifndef PGEN_ADVECTION_HPP_
#define PGEN_ADVECTION_HPP_
//! \file advection.hpp
//! \brief Advection problem generator for 1D/2D/3D problems. Direction of the wavevector
//! is set to be along the x? axis by using the along_x? input flags, else it is
//! automatically set along the grid diagonal in 2D/3D.
//! This file also contains a function to compute L1 errors in solution.

// NOTE(PDM): The following is adapted from the open-source Athena++/AthenaK
// linear wave test, adapted for pure advection and dust by PDM on 10/20/23

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
//! \struct AdvectionVariables
//! \brief container for variables shared with advection pgen and error functions
struct AdvectionVariables {
  Real amp, vflow, lambda;
  Real d0, p0, v1_0, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  Real gamma, gm1;
  int nspec;
};

} // end anonymous namespace

namespace advection {

static AdvectionVariables av;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Advection_()
//! \brief Sets initial conditions for advection tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  const Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;

  // read global parameters
  av.amp = pin->GetReal("problem", "amp");
  av.vflow = pin->GetOrAddReal("problem", "vflow", 0.0);
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
  av.cos_a3 = 1.0;
  av.sin_a3 = 0.0;
  av.cos_a2 = 1.0;
  av.sin_a2 = 0.0;
  if (multi_d && !(along_x1)) {
    Real ang_3 = std::atan(x1size / x2size);
    av.sin_a3 = std::sin(ang_3);
    av.cos_a3 = std::cos(ang_3);
  }
  if (three_d && !(along_x1)) {
    Real ang_2 = std::atan(0.5 * (x1size * av.cos_a3 + x2size * av.sin_a3) / x3size);
    av.sin_a2 = std::sin(ang_2);
    av.cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    av.cos_a3 = 0.0;
    av.sin_a3 = 1.0;
    av.cos_a2 = 1.0;
    av.sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    av.cos_a3 = 0.0;
    av.sin_a3 = 1.0;
    av.cos_a2 = 0.0;
    av.sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  av.lambda = std::numeric_limits<float>::max();
  if (av.cos_a2 * av.cos_a3 > 0.0) {
    av.lambda = std::min(av.lambda, x1size * av.cos_a2 * av.cos_a3);
  }
  if (av.cos_a2 * av.sin_a3 > 0.0) {
    av.lambda = std::min(av.lambda, x2size * av.cos_a2 * av.sin_a3);
  }
  if (av.sin_a2 > 0.0) av.lambda = std::min(av.lambda, x3size * av.sin_a2);

  // Initialize k_parallel
  av.k_par = 2.0 * (M_PI) / av.lambda;

  // Determine if gas and/or dust hydrodynamics are enabled
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  // Set background state: v1_0 is parallel to wavevector.
  // TODO(PDM): Replace the below with a call to singularity-eos
  av.d0 = 1.0;
  av.v1_0 = av.vflow;
  if (do_gas) {
    auto gas_pkg = pmb->packages.Get("gas");
    PARTHENON_REQUIRE((gas_pkg->Param<int>("nspecies") == 1),
                      "Advection pgen requires a single gas species.")
    av.gamma = gas_pkg->Param<Real>("adiabatic_index");
    av.gm1 = av.gamma - 1.0;
    av.p0 = 1.0 / av.gamma;
  }
  if (do_dust) {
    auto dust_pkg = pmb->packages.Get("dust");
    av.nspec = dust_pkg->Param<int>("nspecies");
    PARTHENON_REQUIRE((av.nspec == 2), "Advection pgen requires two dust species.")
  }

  // set new time limit, interpreted as number of wave periods for evolution
  const Real nperiod = pin->GetOrAddReal("problem", "nperiod", 1.0);
  pin->SetReal("parthenon/time", "tlim", nperiod * (std::abs(av.lambda / av.v1_0)));

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  auto adv = av;

  pmb->par_for(
      "pgen_linwave1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates

        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xv = coords.GetCellCenter();

        const Real x1v = xv[0];
        const Real x2v = xv[1];
        const Real x3v = xv[2];

        Real x = adv.cos_a2 * (x1v * adv.cos_a3 + x2v * adv.sin_a3) + x3v * adv.sin_a2;
        Real sn = std::sin(adv.k_par * x);
        Real mx = adv.d0 * adv.vflow + adv.amp * sn * adv.v1_0;
        const Real cd = adv.d0 + adv.amp * sn;
        const Real cm1 = mx * adv.cos_a2 * adv.cos_a3;
        const Real cm2 = mx * adv.cos_a2 * adv.sin_a3;
        const Real cm3 = mx * adv.sin_a2;
        const Real ce = adv.p0 / adv.gm1 + 0.5 * adv.d0 * SQR(adv.v1_0) +
                        0.5 * adv.d0 * adv.amp * sn * SQR(adv.v1_0);
        const Real cu = ce - 0.5 * (SQR(cm1) + SQR(cm2) + SQR(cm3)) / cd;

        // compute cell-centered conserved variables
        if (do_gas) {
          v(0, gas::prim::density(0), k, j, i) = cd;
          v(0, gas::prim::velocity(0), k, j, i) = cm1 / cd;
          v(0, gas::prim::velocity(1), k, j, i) = cm2 / cd;
          v(0, gas::prim::velocity(2), k, j, i) = cm3 / cd;
          v(0, gas::prim::sie(0), k, j, i) = cu / cd;
        }

        if (do_dust) {
          // dust species 1
          v(0, dust::prim::density(0), k, j, i) = cd;
          v(0, dust::prim::velocity(0), k, j, i) = cm1 / cd;
          v(0, dust::prim::velocity(1), k, j, i) = cm2 / cd;
          v(0, dust::prim::velocity(2), k, j, i) = cm3 / cd;
          // dust species 2
          v(0, dust::prim::density(1), k, j, i) = cd;
          v(0, dust::prim::velocity(1 * 3 + 0), k, j, i) = -cm1 / cd;
          v(0, dust::prim::velocity(1 * 3 + 1), k, j, i) = -cm2 / cd;
          v(0, dust::prim::velocity(1 * 3 + 2), k, j, i) = -cm3 / cd;
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Computes errors in advection solution by subtracting current solution from
//! ICs, and outputting errors to file. Problem must be run for an integer number of wave
//! periods.
template <Coordinates GEOM>
inline void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  using parthenon::MakePackDescriptor;
  const int nhyd_vars = 5;
  const int nspec_vars = 4;
  const int nspecies = 2;
  const int nvars = nhyd_vars + nspecies * nspec_vars;

  // packing and capture variables for kernel
  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         dust::cons::density, dust::cons::momentum>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto adv = av;

  // determine if gas and/or dust hydrodynamics are enabled
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  ArtemisUtils::array_type<Real, nvars> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "AdvectionErrors", DevExecSpace(), 0,
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

        Real x = adv.cos_a2 * (x1v * adv.cos_a3 + x2v * adv.sin_a3) + x3v * adv.sin_a2;
        Real sn = std::sin(adv.k_par * x);
        Real mx = adv.d0 * adv.vflow + adv.amp * sn * adv.v1_0;
        Real cd = adv.d0 + adv.amp * sn;
        Real cm1 = mx * adv.cos_a2 * adv.cos_a3;
        Real cm2 = mx * adv.cos_a2 * adv.sin_a3;
        Real cm3 = mx * adv.sin_a2;
        Real ce = adv.p0 / adv.gm1 + 0.5 * adv.d0 * SQR(adv.v1_0) +
                  0.5 * adv.d0 * adv.amp * sn * SQR(adv.v1_0);

        // conserved variables:
        if (do_gas) {
          lsum.myArray[0] += vol * std::abs(v(b, gas::cons::density(), k, j, i) - cd);
          lsum.myArray[1] += vol * std::abs(v(b, gas::cons::momentum(0), k, j, i) - cm1);
          lsum.myArray[2] += vol * std::abs(v(b, gas::cons::momentum(1), k, j, i) - cm2);
          lsum.myArray[3] += vol * std::abs(v(b, gas::cons::momentum(2), k, j, i) - cm3);
          lsum.myArray[4] +=
              vol * std::abs(v(b, gas::cons::total_energy(), k, j, i) - ce);
        }

        if (do_dust) {
          // dust species 2
          lsum.myArray[5] += vol * std::abs(v(b, dust::cons::density(0), k, j, i) - cd);
          lsum.myArray[6] += vol * std::abs(v(b, dust::cons::momentum(0), k, j, i) - cm1);
          lsum.myArray[7] += vol * std::abs(v(b, dust::cons::momentum(1), k, j, i) - cm2);
          lsum.myArray[8] += vol * std::abs(v(b, dust::cons::momentum(2), k, j, i) - cm3);
          // dust species 1
          lsum.myArray[9] += vol * std::abs(v(b, dust::cons::density(1), k, j, i) - cd);
          lsum.myArray[10] +=
              vol * std::abs(v(b, dust::cons::momentum(1 * 3 + 0), k, j, i) + cm1);
          lsum.myArray[11] +=
              vol * std::abs(v(b, dust::cons::momentum(1 * 3 + 1), k, j, i) + cm2);
          lsum.myArray[12] +=
              vol * std::abs(v(b, dust::cons::momentum(1 * 3 + 2), k, j, i) + cm3);
        }
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

  // compute rms error in gas
  Real rms_err_gas = 0.0;
  for (int i = 0; i < nhyd_vars; ++i) {
    rms_err_gas += SQR(l1_err.myArray[i]);
  }
  rms_err_gas = std::sqrt(rms_err_gas);

  // compute rms error in dust #1
  Real rms_err_dust1 = 0.0;
  for (int i = nhyd_vars; i < nhyd_vars + nspec_vars; ++i) {
    rms_err_dust1 += SQR(l1_err.myArray[i]);
  }
  rms_err_dust1 = std::sqrt(rms_err_dust1);

  // compute rms error in dust #2
  Real rms_err_dust2 = 0.0;
  for (int i = nhyd_vars + nspec_vars; i < nhyd_vars + 2 * nspec_vars; ++i) {
    rms_err_dust2 += SQR(l1_err.myArray[i]);
  }
  rms_err_dust2 = std::sqrt(rms_err_dust2);

  Real rms_err_dust[2];
  rms_err_dust[0] = 0.0;
  rms_err_dust[1] = 0.0;
  for (int n = 0; n < av.nspec; ++n) {
    for (int i = nhyd_vars + n * nspec_vars; i < nhyd_vars + (n + 1) * nspec_vars; ++i) {
      rms_err_dust[n] += SQR(l1_err.myArray[i]);
    }
  }

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
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1-gas       ");
      std::fprintf(pfile, "RMS-L1-dust1       RMS-L1-dust2       ");
      std::fprintf(pfile, "d_L1-gas         M1_L1-gas         M2_L1-gas         ");
      std::fprintf(pfile, "M3_L1-gas         E_L1-gas          ");
      std::fprintf(pfile, "RMS-L1-dust1       ");
      std::fprintf(pfile, "d_L1-dust1         M1_L1-dust1         M2_L1-dust1         ");
      std::fprintf(pfile, "M3_L1-dust1         E_L1-dust1          ");
      std::fprintf(pfile, "RMS-L1-dust2       ");
      std::fprintf(pfile, "d_L1-dust2         M1_L1-dust2         M2_L1-dust2         ");
      std::fprintf(pfile, "M3_L1-dust2         E_L1-dust2          ");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmesh->mesh_size.nx(X1DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X3DIR));
    std::fprintf(pfile, "  %05d  %e ", tm.ncycle, rms_err_gas);
    std::fprintf(pfile, "  %e ", rms_err_dust1);
    std::fprintf(pfile, "  %e ", rms_err_dust2);
    for (int i = 0; i < nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err.myArray[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

} // namespace advection
#endif // PGEN_ADVECTION_HPP_
