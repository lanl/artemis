//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef PGEN_GRAV_SLAB_HPP_
#define PGEN_GRAV_SLAB_HPP_
//! \file grav_slab.hpp
//! \brief Advection of self-gravitating slab problem based off of \S4.3 of Hanawa &
//! Mullen (2025).

// This file was created in part or in whole by generative AI

// C/C++ headers
#include <cmath>
#include <limits>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \struct GravSlabVariables
//! \brief container for variables shared with grav slab pgen and error functions
struct GravSlabVariables {
  Real rho0, p0;
  Real eps;
  Real four_pi_G;
  Real vx, vy, vz;
  Real kx, ky, kz;
  Real k2;
};

} // end anonymous namespace

namespace grav_slab {

static GravSlabVariables gsv;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::GravSlab_()
//! \brief Sets initial conditions for self-gravitating slab test
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;

  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "grav_slab pgen requires Cartesian geometry");

  // Hanawa & Mullen (2025) parameters
  gsv.rho0 = 1.0;
  gsv.p0 = 6.0;
  gsv.eps = 0.3;
  gsv.four_pi_G = 1.0;
  gsv.vx = 0.8;
  gsv.vy = 0.6;
  gsv.vz = 0.0;
  gsv.kx = 1.0 / 3.0;
  gsv.ky = 2.0 / 3.0;
  gsv.kz = 2.0 / 3.0;
  gsv.k2 = SQR(gsv.kx) + SQR(gsv.ky) + SQR(gsv.kz);

  // Check 4piG=1
  auto &grav_pkg = pmb->packages.Get("self_gravity");
  PARTHENON_REQUIRE(grav_pkg->Param<Real>("four_pi_G") == gsv.four_pi_G,
                    "grav_slab requires 4piG=1 via self_gravity/units_override=true");
  const Real four_pi_G = grav_pkg->Param<Real>("four_pi_G");

  // Extract adiabatic index
  auto gas_pkg = pmb->packages.Get("gas");

  // set new time limit, interpreted as number of wave periods for evolution
  const Real nperiod = pin->GetOrAddReal("problem", "nperiod", 1.0);
  pin->SetReal("parthenon/time", "tlim", nperiod * 3.0 * M_PI);

  const auto &eos_d = gas_pkg->template Param<ParArray1D<EOS>>("eos_d");

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
          (pmb->resolved_packages).get());
  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  auto vg = desc_g.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  auto slab = gsv;
  const auto &cpars =
      pmb->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  pmb->par_for(
      "pgen_grav_slab", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto &xv = coords.GetCellCenter(vg, 0, k, j, i);
        const Real x1v = xv[0];
        const Real x2v = xv[1];
        const Real x3v = xv[2];

        const Real kr = slab.kx * x1v + slab.ky * x2v + slab.kz * x3v;
        const Real c1 = std::cos(kr);
        const Real c2 = std::cos(2.0 * kr);
        const Real c3 = std::cos(3.0 * kr);
        const Real c4 = std::cos(4.0 * kr);

        const Real rho = slab.rho0 * (1.0 + slab.eps * c1 + (SQR(slab.eps) / 3.0) * c2);
        const Real pres =
            slab.p0 +
            (slab.four_pi_G * slab.eps * SQR(slab.rho0) / slab.k2) *
                ((1.0 - SQR(slab.eps) / 12.0) * c1 + (slab.eps / 3.0) * c2 +
                 (SQR(slab.eps) / 12.0) * c3 + (slab.eps * SQR(slab.eps) / 144.0) * c4);
        const Real sie = ArtemisUtils::EofPR(eos_d(0), pres, rho);

        v(0, gas::prim::density(), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = slab.vx;
        v(0, gas::prim::velocity(1), k, j, i) = slab.vy;
        v(0, gas::prim::velocity(2), k, j, i) = slab.vz;
        v(0, gas::prim::sie(), k, j, i) = sie;
      });
}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Computes errors in self-gravitating slab solution by subtracting current
//! solution from ICs, and outputting errors to file. Problem must be run for an integer
//! number of wave periods.
template <Coordinates GEOM>
inline void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  const int nvars = 5;

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
  auto slab = gsv;
  const auto &cpars =
      pmesh->packages.Get("artemis")->template Param<geometry::CoordParams>(
          "coord_params");
  const auto &eos_d =
      pmesh->packages.Get("gas")->template Param<ParArray1D<EOS>>("eos_d");

  ArtemisUtils::array_type<Real, nvars> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "SlabErrors", DevExecSpace(), 0,
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

        const Real kr = slab.kx * x1v + slab.ky * x2v + slab.kz * x3v;
        const Real c1 = std::cos(kr);
        const Real c2 = std::cos(2.0 * kr);
        const Real c3 = std::cos(3.0 * kr);
        const Real c4 = std::cos(4.0 * kr);

        // construct conserved state solution from primitives
        const Real rho = slab.rho0 * (1.0 + slab.eps * c1 + (SQR(slab.eps) / 3.0) * c2);
        const Real pres =
            slab.p0 +
            (slab.four_pi_G * slab.eps * SQR(slab.rho0) / slab.k2) *
                ((1.0 - SQR(slab.eps) / 12.0) * c1 + (slab.eps / 3.0) * c2 +
                 (SQR(slab.eps) / 12.0) * c3 + (slab.eps * SQR(slab.eps) / 144.0) * c4);
        const Real ca = rho;
        const Real cm1 = rho * slab.vx;
        const Real cm2 = rho * slab.vy;
        const Real cm3 = rho * slab.vz;
        const Real ce = rho * ArtemisUtils::EofPR(eos_d(0), pres, rho) +
                        0.5 * (SQR(cm1) + SQR(cm2) + SQR(cm3)) / rho;

        // accumulate L1 errors
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

} // namespace grav_slab

#endif // PGEN_GRAV_SLAB_HPP_
