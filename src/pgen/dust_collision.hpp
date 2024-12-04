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

// NOTE(@pdmullen): The following is taken directly from the open-source
// Athena++-dustfluid software, and adapted for Parthenon/Artemis by @Shengtai on 4/12/24

//! \file dust_collision.cpp
//! \brief dust collision problem generator for 1D problems.

#ifndef PGEN_DUST_COLLISION_HPP_
#define PGEN_DUST_COLLISION_HPP_

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
//#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace {

//----------------------------------------------------------------------------------------
//! \struct DustCollisionVariable
//! \brief container for variables shared with dust collision pgen
struct DustCollisionVariable {
  int prob_flag;
  int nDust;
  Real gamma, gm1;
  Real iso_cs;
};

} // end anonymous namespace

namespace dust_collision {

DustCollisionVariable dcv;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustCollision_()
//! \brief Sets initial conditions for dust collision tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto geom = artemis_pkg->Param<Coordinates>("coords");
  PARTHENON_REQUIRE(geom == Coordinates::cartesian,
                    "dust_collision pgen requires Cartesian geometry!");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  PARTHENON_REQUIRE(do_dust, "dust_collision pgen requires do_dust=true!");
  bool const_stopping_time = pin->GetOrAddBoolean("dust", "const_stopping_time", true);
  PARTHENON_REQUIRE(do_dust, "dust_collision pgen requires const_stopping_time=true!");

  // read global parameters
  dcv.prob_flag = pin->GetOrAddInteger("problem", "iprob", 1);
  dcv.nDust = pin->GetOrAddReal("dust", "nspecies", 2);
  PARTHENON_REQUIRE(dcv.nDust == 2, "dust_collision pgen requires dust nspecies == 2!");

  auto gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  dcv.gamma = gas_pkg->Param<Real>("adiabatic_index");
  dcv.gm1 = dcv.gamma - 1.0;
  dcv.iso_cs = pin->GetOrAddReal("gas", "iso_sound_speed", 1e-1);

  const Real gdens = 1.0;
  const Real gtemp = SQR(dcv.iso_cs);
  const Real vx_g = 1.0;
  const Real gsie = eos_d.InternalEnergyFromDensityTemperature(gdens, gtemp);

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
  auto &dcol = dcv;

  pmb->par_for(
      "pgen_dustCollision", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, gas::prim::density(), k, j, i) = gdens;
        v(0, gas::prim::velocity(0), k, j, i) = vx_g;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(), k, j, i) = gsie;

        // dust initial condition
        if (dcol.prob_flag == 1) { // stiff problem with smaller stopping time
          v(0, dust::prim::density(0), k, j, i) = 1.0;
          v(0, dust::prim::density(1), k, j, i) = 1.0;
        } else {
          v(0, dust::prim::density(0), k, j, i) = 10.0;
          v(0, dust::prim::density(1), k, j, i) = 100.0;
        }
        v(0, dust::prim::velocity(0 * 3 + 0), k, j, i) = 2.0;
        v(0, dust::prim::velocity(0 * 3 + 1), k, j, i) = 0.0;
        v(0, dust::prim::velocity(0 * 3 + 2), k, j, i) = 0.0;

        v(0, dust::prim::velocity(1 * 3 + 0), k, j, i) = 0.5;
        v(0, dust::prim::velocity(1 * 3 + 1), k, j, i) = 0.0;
        v(0, dust::prim::velocity(1 * 3 + 2), k, j, i) = 0.0;
      });
}

//----------------------------------------------------------------------------------------
//! \fn void PreStepUserworkInLoop
//! \brief output numerical solution and analytic solution before each step
//!  and writing to file.
inline void PreStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin,
                                  parthenon::SimTime &tm) {

  // packing and capture variables for kernel
  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  static auto desc = MakePackDescriptor<gas::prim::velocity, dust::prim::velocity>(
      (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // root process opens output file and writes out data
  if (parthenon::Globals::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("parthenon/job", "problem_id"));
    fname.append("-solu1.dat");
    static FILE *pfile = NULL;

    // The file exists -- reopen the file in append mode
    if (pfile == NULL) {
      if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
        if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
          PARTHENON_FAIL("Error output file could not be opened");
        }
        // The file does not exist -- open the file in write mode and add headers
      } else {
        if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
          PARTHENON_FAIL("Error output file could not be opened");
        }
        std::fprintf(pfile, "# time    vel-g vel-ge  vel-d1 vel-d1e vel-d2  vel-d2e \n");
      }
    }

    // analytic solution
    Real v_com, c1_g, c2_g, lam1, lam2, c1_d1, c2_d1, c1_d2, c2_d2;
    Real v_g, v_d1, v_d2;
    if (dcv.prob_flag == 1) {
      v_com = 7. / 6.;
      c1_g = -0.35610569612832;
      c2_g = 0.18943902946166;
      lam1 = -141.742430504416, lam2 = -1058.25756949558;
      c1_d1 = 0.85310244713865;
      c2_d1 = -0.01976911380532;
      c1_d2 = -0.49699675101033;
      c2_d2 = -0.16966991565634;
    } else {
      v_com = 0.63963963963963;
      c1_g = -0.06458203330249;
      c2_g = 0.42494239366285;
      lam1 = -0.52370200744224;
      lam2 = -105.976297992557;
      c1_d1 = 1.36237475791577;
      c2_d1 = -0.00201439755542;
      c1_d2 = -0.13559165545855;
      c2_d2 = -0.00404798418109;
    }

    v_g = v_com + c1_g * std::exp(lam1 * tm.time) + c2_g * std::exp(lam2 * tm.time);
    v_d1 = v_com + c1_d1 * std::exp(lam1 * tm.time) + c2_d1 * std::exp(lam2 * tm.time);
    v_d2 = v_com + c1_d2 * std::exp(lam1 * tm.time) + c2_d2 * std::exp(lam2 * tm.time);

    // write data

    ParArray1D<Real> vel("outdat", 3);

    Real vg0 = 0.0, vd10 = 0.0, vd20 = 0.0;
    pmb->par_for(
        "out_dustCollision", kb.s, kb.e, jb.s, jb.e, ib.s, ib.s,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          vel(0) = v(0, gas::prim::velocity(0), k, j, i);
          vel(1) = v(0, dust::prim::velocity(0), k, j, i);
          vel(2) = v(0, dust::prim::velocity(3), k, j, i);
        });

    auto dat = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), vel);
    vg0 = dat(0);
    vd10 = dat(1);
    vd20 = dat(2);

    std::fprintf(pfile, "  %e ", tm.time);
    std::fprintf(pfile, "  %e ", vg0);
    std::fprintf(pfile, "  %e ", v_g);
    std::fprintf(pfile, "  %e ", vd10);
    std::fprintf(pfile, "  %e ", v_d1);
    std::fprintf(pfile, "  %e ", vd20);
    std::fprintf(pfile, "  %e ", v_d2);

    std::fprintf(pfile, "\n");
    // std::fclose(pfile);
  }
}

} // namespace dust_collision

#endif // PGEN_DUST_COLLISION_HPP_
