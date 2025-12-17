//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "gravity/gravity.hpp"
#include "utils/artemis_utils.hpp"

using namespace parthenon::package::prelude;
using ArtemisUtils::VI;

namespace Gravity {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::SelfGravity
//! \brief Applies accelerations due to a constant g
template <Coordinates GEOM>
TaskStatus SelfGravity(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  static auto desc =
      MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy,
                         dust::cons::momentum, gas::prim::density, dust::prim::density,
                         grav::phi>(resolved_pkgs.get());
  static auto descf = MakePackDescriptor<gas::cons::density>(
      resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto vmesh = desc.GetPack(md);
  auto vflux = descf.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int d1 = X1DIR;
  const int d2 = d1 + multi_d;
  const int d3 = d2 + three_d;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SelfGravity", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
        const Real hdtodx1 = (0.5 * dt / dx[0]);
        const Real hdtodx2 = multi_d * (0.5 * dt / dx[1]);
        const Real hdtodx3 = three_d * (0.5 * dt / dx[2]);

        // Potential differences
        const Real &phic = vmesh(b, grav::phi(), k, j, i);
        const Real dpl1 = -(phic - vmesh(b, grav::phi(), k, j, i - 1));
        const Real dpr1 = -(vmesh(b, grav::phi(), k, j, i + 1) - phic);
        const Real dpl2 = -(phic - vmesh(b, grav::phi(), k, j - multi_d, i));
        const Real dpr2 = -(vmesh(b, grav::phi(), k, j + multi_d, i) - phic);
        const Real dpl3 = -(phic - vmesh(b, grav::phi(), k - three_d, j, i));
        const Real dpr3 = -(vmesh(b, grav::phi(), k + three_d, j, i) - phic);

        // Shared b/w gas and dust
        const Real wdt1 = hdtodx1 * (dpl1 + dpr1);
        const Real wdt2 = hdtodx2 * (dpl2 + dpr2);
        const Real wdt3 = hdtodx3 * (dpl3 + dpr3);

        if (do_gas) {
          // Gravitational acceleration and energy release
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            const Real &rr = vmesh(b, gas::prim::density(n), k, j, i);
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += rr * wdt1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += rr * wdt2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += rr * wdt3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                hdtodx1 * (vflux.flux(b, d1, gas::cons::density(n), k, j, i) * dpl1 +
                           vflux.flux(b, d1, gas::cons::density(n), k, j, i + 1) * dpr1);
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                hdtodx2 *
                (vflux.flux(b, d2, gas::cons::density(n), k, j, i) * dpl2 +
                 vflux.flux(b, d2, gas::cons::density(n), k, j + multi_d, i) * dpr2);
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                hdtodx3 *
                (vflux.flux(b, d3, gas::cons::density(n), k, j, i) * dpl3 +
                 vflux.flux(b, d3, gas::cons::density(n), k + three_d, j, i) * dpr3);
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Gravitational acceleration
            const Real &rr = vmesh(b, dust::prim::density(n), k, j, i);
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) += rr * wdt1;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) += rr * wdt2;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) += rr * wdt3;
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef MeshData<Real> MD;
template TaskStatus SelfGravity<G::cartesian>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::cylindrical>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::spherical1D>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::spherical2D>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::spherical3D>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::axisymmetric>(MD *m, const Real t, const Real d);

} // namespace Gravity
