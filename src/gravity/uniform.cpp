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

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "gravity/gravity.hpp"
#include "utils/artemis_utils.hpp"

using namespace parthenon::package::prelude;
using ArtemisUtils::VI;

namespace Gravity {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::UniformGravity
//! \brief Applies accelerations due to a constant g
template <Coordinates GEOM>
TaskStatus UniformGravity(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  auto &gravity_pkg = pm->packages.Get("gravity");
  const Real gx1 = gravity_pkg->template Param<Real>("gx1");
  const Real gx2 = gravity_pkg->template Param<Real>("gx2");
  const Real gx3 = gravity_pkg->template Param<Real>("gx3");

  static auto desc =
      MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy,
                         dust::cons::momentum, gas::prim::density, gas::prim::velocity,
                         dust::prim::density>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UniformGravity", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &hx = coords.GetScaleFactors();

        if (do_gas) {
          // Gravitational acceleration and energy release
          // NOTE(PDM): for *self-gravity*, it is fairly well established that a good
          // choice for discretization for the energy source term invokes the cell
          // surface mass fluxes and the cell surface gravity.  Does this equally work
          // well with static gravity? Right now we adopt a simple cell-centered
          // approximation for the energy source term as does AthenaPK...
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            const Real rdt = dt * vmesh(b, gas::prim::density(n), k, j, i);
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += rdt * hx[0] * gx1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += rdt * hx[1] * gx2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += rdt * hx[2] * gx3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                rdt * (vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i) * gx1 +
                       vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i) * gx2 +
                       vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i) * gx3);
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Gravitational acceleration
            const Real rdt = dt * vmesh(b, dust::prim::density(n), k, j, i);
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) += rdt * hx[0] * gx1;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) += rdt * hx[1] * gx2;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) += rdt * hx[2] * gx3;
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> MD;
template TaskStatus UniformGravity<C::cartesian>(MD *m, const Real t, const Real d);
template TaskStatus UniformGravity<C::cylindrical>(MD *m, const Real t, const Real d);
template TaskStatus UniformGravity<C::spherical1D>(MD *m, const Real t, const Real d);
template TaskStatus UniformGravity<C::spherical2D>(MD *m, const Real t, const Real d);
template TaskStatus UniformGravity<C::spherical3D>(MD *m, const Real t, const Real d);
template TaskStatus UniformGravity<C::axisymmetric>(MD *m, const Real t, const Real d);

} // namespace Gravity
