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
#ifndef PGEN_FIELD_LOOP_HPP_
#define PGEN_FIELD_LOOP_HPP_
//! \file field_loop.hpp
//! \brief Advection of a weak, divergence-free magnetic field loop.

// C++ headers
#include <cmath>
#include <limits>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"

namespace {

struct FieldLoopVariables {
  bool tilted;
  Real radius, amp;
  Real rho0, p0, sie0;
  Real v1, v2, v3;
  Real x0, y0;
  Real cos_a, sin_a, lambda;
};

} // namespace

namespace field_loop {

static FieldLoopVariables flv;

KOKKOS_INLINE_FUNCTION Real WrapPeriodic(const Real x, const Real lambda) {
  Real wrapped = x;
  while (wrapped > 0.5 * lambda)
    wrapped -= lambda;
  while (wrapped < -0.5 * lambda)
    wrapped += lambda;
  return wrapped;
}

KOKKOS_INLINE_FUNCTION Real LoopPotential(const FieldLoopVariables &pars, const Real x1,
                                          const Real x2, const Real x3) {
  Real x = x1 - pars.x0;
  if (pars.tilted)
    x = WrapPeriodic(x1 * pars.cos_a + x3 * pars.sin_a - pars.x0, pars.lambda);
  const Real y = x2 - pars.y0;
  const Real r = std::sqrt(SQR(x) + SQR(y));
  return (r < pars.radius) ? pars.amp * (pars.radius - r) : 0.0;
}

KOKKOS_INLINE_FUNCTION Real A1(const FieldLoopVariables &pars, const Real x1,
                               const Real x2, const Real x3) {
  return pars.tilted ? -pars.sin_a * LoopPotential(pars, x1, x2, x3) : 0.0;
}

KOKKOS_INLINE_FUNCTION Real A3(const FieldLoopVariables &pars, const Real x1,
                               const Real x2, const Real x3) {
  return (pars.tilted ? pars.cos_a : 1.0) * LoopPotential(pars, x1, x2, x3);
}

//----------------------------------------------------------------------------------------
//! \brief Initialize the planar or tilted magnetic field-loop advection test.
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;

  const auto pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");

  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "field_loop pgen requires Cartesian geometry!");
  PARTHENON_REQUIRE(do_gas && do_mhd, "field_loop pgen requires gas and MHD!");

  const std::string loop_type = pin->GetOrAddString("problem", "loop_type", "planar");
  PARTHENON_REQUIRE(loop_type == "planar" || loop_type == "tilted",
                    "problem/loop_type must be either planar or tilted!");
  flv.tilted = loop_type == "tilted";
  PARTHENON_REQUIRE((flv.tilted && ndim == 3) || (!flv.tilted && ndim == 2),
                    "field_loop planar loops require 2D and tilted loops require 3D!");

  auto gas_pkg = pmb->packages.Get("gas");
  PARTHENON_REQUIRE(gas_pkg->Param<std::string>("eos_type") == "ideal",
                    "field_loop pgen requires an ideal gas!");
  PARTHENON_REQUIRE(gas_pkg->Param<int>("nspecies") == 1,
                    "field_loop pgen requires one gas species!");

  flv.radius = pin->GetOrAddReal("problem", "radius", 0.3);
  flv.amp = pin->GetOrAddReal("problem", "amp", 1.0e-3);
  flv.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  flv.p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  flv.x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  flv.y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  const Real vflow = pin->GetOrAddReal("problem", "vflow", 1.0);
  const Real nperiod = pin->GetOrAddReal("problem", "nperiod", 1.0);
  PARTHENON_REQUIRE(flv.radius > 0.0 && flv.amp > 0.0 && flv.rho0 > 0.0 && flv.p0 > 0.0,
                    "field_loop radius, amp, rho0, and p0 must be positive!");
  PARTHENON_REQUIRE(std::abs(vflow) > std::numeric_limits<Real>::epsilon(),
                    "field_loop requires a nonzero problem/vflow!");
  PARTHENON_REQUIRE(nperiod > 0.0, "field_loop requires a positive problem/nperiod!");

  const Real l1 = pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR);
  const Real l2 = pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR);
  const Real l3 = pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR);
  if (flv.tilted) {
    const Real diagonal = std::sqrt(SQR(l1) + SQR(l2) + SQR(l3));
    const Real x13_diagonal = std::sqrt(SQR(l1) + SQR(l3));
    flv.v1 = vflow * l1 / diagonal;
    flv.v2 = vflow * l2 / diagonal;
    flv.v3 = vflow * l3 / diagonal;
    flv.sin_a = l1 / x13_diagonal;
    flv.cos_a = l3 / x13_diagonal;
    flv.lambda = l1 * flv.cos_a;
    pin->SetReal("parthenon/time", "tlim", nperiod * diagonal / std::abs(vflow));
  } else {
    const Real diagonal = std::sqrt(SQR(l1) + SQR(l2));
    flv.v1 = vflow * l1 / diagonal;
    flv.v2 = vflow * l2 / diagonal;
    flv.v3 = pin->GetOrAddReal("problem", "v3", 0.0);
    flv.sin_a = 0.0;
    flv.cos_a = 1.0;
    flv.lambda = 1.0;
    pin->SetReal("parthenon/time", "tlim", nperiod * diagonal / std::abs(vflow));
  }
  flv.sie0 = flv.p0 / (flv.rho0 * (gas_pkg->Param<Real>("adiabatic_index") - 1.0));

  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         field::face::B>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  const auto &pco = pmb->coords;
  const auto pars = flv;

  const IndexRange ib1 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
  const IndexRange jb1 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  const IndexRange kb1 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "field_loop_b1", kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1 = pco.template Xf<X1DIR>(i);
        const Real x2l = pco.template Xf<X2DIR>(j);
        const Real x2r = pco.template Xf<X2DIR>(j + 1);
        const Real x3 = pco.template Xc<X3DIR>(k);
        v(0, TE::F1, field::face::B(), k, j, i) =
            (A3(pars, x1, x2r, x3) - A3(pars, x1, x2l, x3)) / (x2r - x2l);
      });

  const IndexRange ib2 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2);
  const IndexRange jb2 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  const IndexRange kb2 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "field_loop_b2", kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1l = pco.template Xf<X1DIR>(i);
        const Real x1r = pco.template Xf<X1DIR>(i + 1);
        const Real x1c = pco.template Xc<X1DIR>(i);
        const Real x2 = pco.template Xf<X2DIR>(j);
        const Real x3l = pco.template Xf<X3DIR>(k);
        const Real x3r = pco.template Xf<X3DIR>(k + 1);
        const Real x3c = pco.template Xc<X3DIR>(k);
        v(0, TE::F2, field::face::B(), k, j, i) =
            (A1(pars, x1c, x2, x3r) - A1(pars, x1c, x2, x3l)) / (x3r - x3l) -
            (A3(pars, x1r, x2, x3c) - A3(pars, x1l, x2, x3c)) / (x1r - x1l);
      });

  const IndexRange ib3 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);
  const IndexRange jb3 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  const IndexRange kb3 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "field_loop_b3", kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1 = pco.template Xc<X1DIR>(i);
        const Real x2l = pco.template Xf<X2DIR>(j);
        const Real x2r = pco.template Xf<X2DIR>(j + 1);
        const Real x3 = pco.template Xf<X3DIR>(k);
        v(0, TE::F3, field::face::B(), k, j, i) =
            -(A1(pars, x1, x2r, x3) - A1(pars, x1, x2l, x3)) / (x2r - x2l);
      });

  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "field_loop_gas", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, gas::prim::density(), k, j, i) = pars.rho0;
        v(0, gas::prim::velocity(0), k, j, i) = pars.v1;
        v(0, gas::prim::velocity(1), k, j, i) = pars.v2;
        v(0, gas::prim::velocity(2), k, j, i) = pars.v3;
        v(0, gas::prim::sie(), k, j, i) = pars.sie0;
      });
}

} // namespace field_loop

#endif // PGEN_FIELD_LOOP_HPP_
