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
// Athena++-dustfluid software, and adapted for Parthenon/Artemis by @Shengtai on 7/30/24

//! \file dust_coagulation.hpp
//! \brief dust collision problem generator for 1D problems.

#ifndef PGEN_DUST_COAGULATION_HPP_
#define PGEN_DUST_COAGULATION_HPP_

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
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace {

//----------------------------------------------------------------------------------------
//! \struct DustCoagulationVariable
//! \brief container for variables shared with dust_coagulation pgen
struct DustCoagulationVariable {
  int nDust;
  int nInit_dust;
  Real gamma, gm1;
  Real iso_cs;
  Real d2g;
  int nstep1Coag;
  Real rho_g0;
  Real mass0;
  Real time0;
  Real length0;
  Real vol0;
};

} // end anonymous namespace

namespace dust_coagulation {

DustCoagulationVariable dcv;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustCollision_()
//! \brief Sets initial conditions for dust coagulation tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto geom = artemis_pkg->Param<Coordinates>("coords");
  PARTHENON_REQUIRE(geom == Coordinates::cartesian,
                    "dust_coagulation pgen requires Cartesian geometry!");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  PARTHENON_REQUIRE(do_dust, "dust_coagulation pgen requires do_dust=true!");

  auto &dust_pkg = pmb->packages.Get("dust");

  const bool enable_coagulation = dust_pkg->Param<bool>("enable_coagulation");
  // PARTHENON_REQUIRE(enable_coagulation,
  //                  "dust_coagulation pgen requires enable_coagulation=true!");

  // read global parameters
  dcv.nDust = pin->GetOrAddReal("dust", "nspecies", 121);
  dcv.nInit_dust = pin->GetOrAddReal("problem", "nInit_dust", 1);
  dcv.d2g = pin->GetOrAddReal("problem", "dust_to_gas", 0.01);
  dcv.nstep1Coag = pin->GetOrAddReal("problem", "nstep1Coag", 50);

  if (Dust::cgsunit == NULL) {
    Dust::cgsunit = new Dust::CGSUnit();
  }
  if (!Dust::cgsunit->isSet()) {
    Dust::cgsunit->SetCGSUnit(pin);
  }

  dcv.length0 = Dust::cgsunit->length0;
  dcv.mass0 = Dust::cgsunit->mass0;
  dcv.time0 = Dust::cgsunit->time0;

  Real den_code2phy = 1.0;
  dcv.vol0 = Dust::cgsunit->vol0;

  den_code2phy = dcv.mass0 / dcv.vol0;

  dcv.rho_g0 = den_code2phy;

  // using MRN distribution for the initial dust setup
  ParArray1D<Real> dust_size = dust_pkg->template Param<ParArray1D<Real>>("sizes");

  auto s_p_prefh = Kokkos::create_mirror(dust_size);
  Kokkos::deep_copy(s_p_prefh, dust_size);

  Real sum1 = 0.0;
  for (int i = 0; i < dcv.nInit_dust; i++) {
    sum1 += std::sqrt(s_p_prefh(i));
  }

  for (int i = 0; i < dcv.nInit_dust; i++) {
    s_p_prefh(i) = std::sqrt(s_p_prefh(i)) / sum1;
  }

  for (int i = dcv.nInit_dust; i < dcv.nDust; i++) {
    s_p_prefh(i) = 0.0;
  }
  ParArray1D<Real> s_p_pref("init dust", dcv.nDust);
  Kokkos::deep_copy(s_p_pref, s_p_prefh);

  auto gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  dcv.gamma = gas_pkg->Param<Real>("adiabatic_index");
  dcv.gm1 = dcv.gamma - 1.0;
  dcv.iso_cs = pin->GetOrAddReal("gas", "iso_sound_speed", 1e-1);

  const Real gdens = 1.0;
  const Real gtemp = SQR(dcv.iso_cs);
  const Real vx_g = 0.0;
  const Real gsie = eos_d.InternalEnergyFromDensityTemperature(gdens, gtemp);
  std::cout << "gamma,cs,pre=" << dcv.gamma << " " << dcv.iso_cs << " " << gsie * dcv.gm1
            << std::endl;

  const Real vx_d = 0.0;
  const Real vy_d = 0.0;
  const Real vz_d = 0.0;

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
  auto &dcoag = dcv;

  pmb->par_for(
      "pgen_dustCoagulation", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, gas::prim::density(0), k, j, i) = gdens;
        v(0, gas::prim::velocity(0), k, j, i) = vx_g;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = gsie;

        // dust initial condition
        for (int n = 0; n < dcoag.nDust; ++n) {
          v(0, dust::prim::density(n), k, j, i) = dcoag.d2g * gdens * s_p_pref(n);
          v(0, dust::prim::velocity(n * 3 + 0), k, j, i) = vx_d;
          v(0, dust::prim::velocity(n * 3 + 1), k, j, i) = vy_d;
          v(0, dust::prim::velocity(n * 3 + 2), k, j, i) = vz_d;
        }
      });
}

} // namespace dust_coagulation

#endif // PGEN_DUST_COAGULATION_HPP_
