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
#ifndef PGEN_RT_HPP_
#define PGEN_RT_HPP_

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace {
struct RTParams {
  Real rho0, rho1;
  Real pres0;
  Real y0;
  Real amp, freq;
  Real g;
};
} // end anonymous namespace

namespace rt {

static RTParams RT_params;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::RT
//! \brief
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  auto grav_pkg = pmb->packages.Get("gravity");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_grav = artemis_pkg->Param<bool>("do_gravity");

  RT_params.y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  RT_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  RT_params.pres0 = pin->GetOrAddReal("problem", "pres0", 1.0);
  RT_params.rho1 = pin->GetOrAddReal("problem", "rho1", 2.0);
  RT_params.freq = pin->GetOrAddReal("problem", "frequency", 6 * M_PI);
  RT_params.amp = pin->GetOrAddReal("problem", "amplitude", 0.01);

  const auto gm1 = gas_pkg->Param<Real>("adiabatic_index") - 1.0;

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
  auto pars = RT_params;

  const int ndim = ProblemDimension(pin);

  PARTHENON_REQUIRE(ndim > 1, "RT problem requires ndim >= 2");
  const bool three_d = (ndim == 3);
  Real gx = 0.0;
  Real zmin = Null<Real>(), zmax = Null<Real>();
  if (three_d) {
    zmin = pin->GetReal("parthenon/mesh", "x3min");
    zmax = pin->GetReal("parthenon/mesh", "x3max");
    if (do_grav) gx = grav_pkg->Param<Real>("gx3");
  } else {
    zmin = pin->GetReal("parthenon/mesh", "x2min");
    zmax = pin->GetReal("parthenon/mesh", "x2max");
    if (do_grav) gx = grav_pkg->Param<Real>("gx2");
  }

  // setup uniform ambient medium
  pmb->par_for(
      "RT", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto bbox = geometry::BBox(pco, k, j, i);
        const Real xc = 0.5 * (bbox.x1[0] + bbox.x1[1]);
        const Real zc =
            (three_d) ? 0.5 * (bbox.x3[0] + bbox.x3[1]) : 0.5 * (bbox.x2[0] + bbox.x2[1]);
        const int upper = (zc >= 0.0);
        const int ix = (three_d) ? 2 : 1;
        const Real dens = (upper) ? pars.rho1 : pars.rho0;
        const Real z0 = (upper) ? pars.y0 : zmin;

        // P = P0 + \int rho g dz
        //   = P0 + g*(z - zmin)*rho0                 z < zc
        //   = P0 + g*(zc-zmin) rho0 + g*(z-zc)*rho1  z>=zc
        const Real p0 = pars.pres0 + (upper)*gx * (pars.y0 - zmin) * pars.rho0;
        const Real pres = p0 + gx * (zc - z0) * dens;

        v(0, gas::prim::density(0), k, j, i) = dens;
        v(0, gas::prim::sie(0), k, j, i) = pres / (dens * gm1);
        v(0, gas::prim::velocity(0), k, j, i) = 0.0;
        v(0, gas::prim::velocity(1), k, j, i) = pars.amp * std::cos(pars.freq * xc);
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
      });
}

} // namespace rt
#endif // PGEN_RT_HPP_
