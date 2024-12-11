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
};

} // end anonymous namespace

namespace dust_coagulation {

struct CGSUnit {
  Real mass0 = 1.0;
  Real time0 = 1.0;
  Real length0 = 1.0;
  Real vol0 = 1.0;
  bool isurface_den = true;
  bool Code2PhysicalUnit_Set = false;

  bool isSet() const { return Code2PhysicalUnit_Set; }

  void SetCGSUnit(const Real &mass0_in, const Real &length0_in, const Real &time0_in,
                  const int isurf) {
    if (!Code2PhysicalUnit_Set) {
      mass0 = mass0_in;
      length0 = length0_in;
      vol0 = SQR(length0_in);
      if (isurf == 0) vol0 *= length0_in; // 3D volume
      time0 = time0_in;
      Code2PhysicalUnit_Set = true;
    }
  }

  void SetCGSUnit(ParameterInput *pin) {
    if (!Code2PhysicalUnit_Set) {
      const Real M_SUN = 1.988409870698051e33;   // gram (sun)
      const Real AU_LENGTH = 1.4959787070000e13; // cm
      const Real GRAV_CONST =
          6.674299999999999e-8; // gravitational const in cm^3 g^-1 s^-2

      Real mstar = pin->GetOrAddReal("problem", "mstar", 1.0) * M_SUN;
      length0 = pin->GetOrAddReal("problem", "r0_length", 1.0) * AU_LENGTH;
      const Real omega0 = std::sqrt(GRAV_CONST * mstar / (length0 * length0 * length0));
      time0 = 1. / omega0;

      const Real rho0 = pin->GetReal("problem", "rho0");
      mass0 = rho0 * mstar;

      isurface_den = pin->GetOrAddBoolean("dust", "surface_density_flag", true);
      vol0 = SQR(length0);
      if (isurface_den == 0) vol0 *= length0; // 3D volume
      Code2PhysicalUnit_Set = true;
    }
  }
};

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

  const bool do_coagulation = artemis_pkg->Param<bool>("do_coagulation");
  PARTHENON_REQUIRE(do_coagulation,
                    "dust_coagulation pgen requires physics coagulation=true!");

  // read global parameters
  dcv.nDust = pin->GetOrAddReal("dust", "nspecies", 121);
  dcv.nInit_dust = pin->GetOrAddReal("problem", "nInit_dust", 1);
  dcv.d2g = pin->GetOrAddReal("problem", "dust_to_gas", 0.01);

  // using MRN distribution for the initial dust setup
  ParArray1D<Real> dust_size = dust_pkg->template Param<ParArray1D<Real>>("sizes");

  // convert input "AU" and "mstar" to CGS unit
  CGSUnit *cgsunit = new CGSUnit();
  cgsunit->SetCGSUnit(pin);
  const Real den0 = cgsunit->mass0 / cgsunit->vol0; // density
  const Real time0 = cgsunit->time0;
  const Real vel0 = cgsunit->length0 / cgsunit->time0;

  const Real tlim_in = pin->GetReal("parthenon/time", "tlim");
  pin->SetReal("parthenon/time", "tlim", tlim_in * time0);

  if (pin->DoesBlockExist("parthenon/output0")) {
    const Real dt = pin->GetReal("parthenon/output0", "dt");
    pin->SetReal("parthenon/output0", "dt", dt * time0);
  }

  if (pin->DoesBlockExist("parthenon/output1")) {
    const Real dt = pin->GetReal("parthenon/output1", "dt");
    pin->SetReal("parthenon/output1", "dt", dt * time0);
  }

  if (pin->DoesBlockExist("parthenon/output2")) {
    const Real dt = pin->GetReal("parthenon/output2", "dt");
    pin->SetReal("parthenon/output2", "dt", dt * time0);
  }

  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  if (x1max < 1e10) {
    std::stringstream msg;
    msg << " reset x1min and x1max using cgs unit = " << cgsunit->length0 << std::endl;
    PARTHENON_FAIL(msg);
  }
  auto gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  dcv.gamma = gas_pkg->Param<Real>("adiabatic_index");
  dcv.gm1 = dcv.gamma - 1.0;
  dcv.iso_cs = pin->GetOrAddReal("gas", "iso_sound_speed", 1e-1);
  dcv.iso_cs *= vel0;

  const Real gdens = 1.0 * den0;
  const Real gtemp = SQR(dcv.iso_cs);
  const Real gsie = eos_d.InternalEnergyFromDensityTemperature(gdens, gtemp);
  if (pmb->gid == 0) {
    std::cout << "gamma,cs,temp=" << dcv.gamma << " " << dcv.iso_cs << " "
              << gsie * dcv.gm1 << ", time0,length0,mass0,den0,vel0=" << time0 << " "
              << cgsunit->length0 << " " << cgsunit->mass0 << " " << den0 << " " << vel0
              << std::endl;
  }

  const Real vx_g = 0.0 * vel0;
  const Real vx_d = 0.0 * vel0;

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

  Real sum1 = 0.0;
  pmb->par_reduce(
      "pgen_partialSum", 0, dcoag.nInit_dust - 1,
      KOKKOS_LAMBDA(const int n, Real &lsum) { lsum += std::sqrt(dust_size(n)); }, sum1);

  pmb->par_for(
      "pgen_dustCoagulation", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, gas::prim::density(0), k, j, i) = gdens;
        v(0, gas::prim::velocity(0), k, j, i) = vx_g;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = gsie;

        for (int n = 0; n < dcoag.nInit_dust; ++n) {
          const Real sratio = std::sqrt(dust_size(n)) / sum1;
          v(0, dust::prim::density(n), k, j, i) = dcoag.d2g * gdens * sratio;
          v(0, dust::prim::velocity(n * 3 + 0), k, j, i) = vx_d;
          v(0, dust::prim::velocity(n * 3 + 1), k, j, i) = 0.0;
          v(0, dust::prim::velocity(n * 3 + 2), k, j, i) = 0.0;
        }
        for (int n = dcoag.nInit_dust; n < dcoag.nDust; ++n) {
          v(0, dust::prim::density(n), k, j, i) = 0.0;
          v(0, dust::prim::velocity(n * 3 + 0), k, j, i) = vx_d;
          v(0, dust::prim::velocity(n * 3 + 1), k, j, i) = 0.0;
          v(0, dust::prim::velocity(n * 3 + 2), k, j, i) = 0.0;
        }
      });
}

} // namespace dust_coagulation

#endif // PGEN_DUST_COAGULATION_HPP_
