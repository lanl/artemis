//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

//! \file coag.hpp
//! \brief dust collision problem generator for 1D problems.

#ifndef PGEN_COAG_HPP_
#define PGEN_COAG_HPP_

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
//! \brief container for variables shared with coag pgen
struct DustCoagulationVariable {
  int ndust;
  int ninit_dust;
  Real h0;
  Real d2g;
  Real rho0;
  Real omk;
};

} // end anonymous namespace

namespace coag {

DustCoagulationVariable dcv;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DustCoagulation_()
//! \brief Sets initial conditions for dust coagulation tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Require Cartesian geometry
  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto geom = artemis_pkg->Param<Coordinates>("coords");
  PARTHENON_REQUIRE(geom == Coordinates::cartesian,
                    "coag pgen requires Cartesian geometry!");

  // Require dust physics
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  PARTHENON_REQUIRE(do_dust, "coag pgen requires do_dust=true!");

  // Require coagulation
  auto &dust_pkg = pmb->packages.Get("dust");
  const bool do_coagulation = artemis_pkg->Param<bool>("do_coagulation");
  PARTHENON_REQUIRE(do_coagulation, "coag pgen requires physics coagulation=true!");

  // Read global parameters
  dcv.ndust = dust_pkg->Param<int>("nspecies");
  dcv.ninit_dust = pin->GetOrAddReal("problem", "ninit_dust", 1);
  dcv.d2g = pin->GetOrAddReal("problem", "dust_to_gas", 0.01);
  dcv.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  dcv.omk = pin->GetOrAddReal("problem", "om0", 1.0);

  auto gas_pkg = pmb->packages.Get("gas");
  const auto &eos_h = gas_pkg->template Param<parthenon::HostArray1D<EOS>>("eos_h");

  // Extract adiabatic index and H0
  dcv.h0 = pin->GetOrAddReal("problem", "h0", 0.05);

  // Extract fluid state vector
  const Real gdens = dcv.rho0;
  const Real pres = gdens * SQR(dcv.h0 * dcv.omk);
  const Real gsie = ArtemisUtils::EofPR(eos_h(0), pres, gdens);

  // Using MRN distribution for the initial dust setup
  ParArray1D<Real> dust_size = dust_pkg->template Param<ParArray1D<Real>>("sizes");
  Real sum1 = 0.0;
  auto dcoag = dcv;
  pmb->par_reduce(
      "pgen_partialSum", 0, dcoag.ninit_dust - 1,
      KOKKOS_LAMBDA(const int n, Real &lsum) { lsum += std::sqrt(dust_size(n)); }, sum1);

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

  // Initialize state vectors
  const Real vx_g = 0.0;
  const Real vx_d = 0.0;
  pmb->par_for(
      "pgen_dustCoagulation", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Set gas state vector
        v(0, gas::prim::density(0), k, j, i) = gdens;
        v(0, gas::prim::velocity(0), k, j, i) = vx_g;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = gsie;

        // Set dust state vector
        for (int n = 0; n < dcoag.ninit_dust; ++n) {
          const Real sratio = std::sqrt(dust_size(n)) / sum1;
          v(0, dust::prim::density(n), k, j, i) = dcoag.d2g * gdens * sratio;
          v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx_d;
          v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = 0.0;
          v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = 0.0;
        }
        for (int n = dcoag.ninit_dust; n < dcoag.ndust; ++n) {
          v(0, dust::prim::density(n), k, j, i) = 0.0;
          v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx_d;
          v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = 0.0;
          v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = 0.0;
        }
      });
}

} // namespace coag

#endif // PGEN_COAG_HPP_
