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
#ifndef PGEN_ORSZAG_TANG_HPP_
#define PGEN_ORSZAG_TANG_HPP_
//! \file orszag_tang.hpp
//! \brief Orszag-Tang vortex problem generator.

// C++ headers
#include <cmath>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace {

struct OrszagTangParams {
  Real rho0;
  Real p0;
  Real v0;
  Real b0;
  Real x1min;
  Real x1max;
  Real x2min;
  Real x2max;
};

} // namespace

namespace orszag_tang {

static OrszagTangParams otv;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::OrszagTang()
//! \brief Sets initial conditions for the Orszag-Tang vortex.
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  const int ndim = pmb->pmy_mesh->ndim;

  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "orszag_tang pgen requires Cartesian geometry!");
  PARTHENON_REQUIRE(ndim >= 2,
                    "orszag_tang pgen requires at least 2 spatial dimensions!");
  PARTHENON_REQUIRE(do_gas, "orszag_tang pgen requires gas hydrodynamics!");
  PARTHENON_REQUIRE(!(do_dust), "orszag_tang pgen does not support dust!");
  PARTHENON_REQUIRE(do_mhd, "orszag_tang pgen requires MHD!");

  auto gas_pkg = pmb->packages.Get("gas");
  PARTHENON_REQUIRE(gas_pkg->Param<int>("nspecies") == 1,
                    "orszag_tang pgen requires a single gas species.");
  const auto &eos = gas_pkg->Param<EOS>("eos_h");

  otv.rho0 = pin->GetOrAddReal("problem", "rho0", 25.0 / (36.0 * M_PI));
  otv.p0 = pin->GetOrAddReal("problem", "p0", 5.0 / (12.0 * M_PI));
  otv.v0 = pin->GetOrAddReal("problem", "v0", 1.0);
  otv.b0 = pin->GetOrAddReal("problem", "b0", 1.0 / std::sqrt(4.0 * M_PI));
  otv.x1min = pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  otv.x1max = pmb->pmy_mesh->mesh_size.xmax(X1DIR);
  otv.x2min = pmb->pmy_mesh->mesh_size.xmin(X2DIR);
  otv.x2max = pmb->pmy_mesh->mesh_size.xmax(X2DIR);

  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         field::face::B>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib1 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
  IndexRange jb1 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  IndexRange kb1 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  IndexRange ib2 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2);
  IndexRange jb2 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  IndexRange kb2 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  IndexRange ib3 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);
  IndexRange jb3 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  IndexRange kb3 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  auto &pco = pmb->coords;
  auto pars = otv;

  const Real lx = pars.x1max - pars.x1min;
  const Real ly = pars.x2max - pars.x2min;
  const Real sie0 = ArtemisUtils::EofPR(eos, pars.p0, pars.rho0);
  const bool three_d = (ndim > 2);

  pmb->par_for(
      "orszag_tang_cc", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = (pco.template Xc<X1DIR>(i) - pars.x1min) / lx;
        const Real y = (pco.template Xc<X2DIR>(j) - pars.x2min) / ly;
        v(0, TE::CC, gas::prim::density(), k, j, i) = pars.rho0;
        v(0, TE::CC, gas::prim::velocity(0), k, j, i) =
            -pars.v0 * std::sin(2.0 * M_PI * y);
        v(0, TE::CC, gas::prim::velocity(1), k, j, i) =
            pars.v0 * std::sin(2.0 * M_PI * x);
        v(0, TE::CC, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, TE::CC, gas::prim::sie(), k, j, i) = sie0;
      });

  pmb->par_for(
      "orszag_tang_b1", kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real y = (pco.template Xc<X2DIR>(j) - pars.x2min) / ly;
        v(0, TE::F1, field::face::B(), k, j, i) = -pars.b0 * std::sin(2.0 * M_PI * y);
      });

  pmb->par_for(
      "orszag_tang_b2", kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = (pco.template Xc<X1DIR>(i) - pars.x1min) / lx;
        v(0, TE::F2, field::face::B(), k, j, i) = pars.b0 * std::sin(4.0 * M_PI * x);
      });

  if (three_d) {
    pmb->par_for(
        "orszag_tang_b3", kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v(0, TE::F3, field::face::B(), k, j, i) = 0.0;
        });
  }
}

} // namespace orszag_tang

#endif // PGEN_ORSZAG_TANG_HPP_
