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

using ArtemisUtils::VI;

namespace Gravity {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::PointMassGravity
//! \brief Applies accelerations due to a point mass gravitational potential
template <Coordinates GEOM>
TaskStatus PointMassGravity(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  auto &gravity_pkg = pm->packages.Get("gravity");
  const Real gm_ = gravity_pkg->template Param<Real>("gm");
  const Real sink_rate = dt * (gravity_pkg->template Param<Real>("sink_rate"));
  const Real sink_rad = gravity_pkg->template Param<Real>("sink");

  // The following assumes x1,x2,x3 are always in Cartesian coordinates.  Usually this is
  // just used for a central object at (0,0,0)...
  const Real pos_[3] = {gravity_pkg->template Param<Real>("x"),
                        gravity_pkg->template Param<Real>("y"),
                        gravity_pkg->template Param<Real>("z")};
  const Real rsft2_ = SQR(gravity_pkg->template Param<Real>("soft"));

  static auto desc =
      MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy, gas::cons::density,
                         dust::cons::momentum, dust::cons::density, gas::prim::density,
                         gas::prim::velocity, gas::prim::sie, dust::prim::density,
                         dust::prim::velocity>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;
  const bool multi_d_ = (ndim >= 2);
  const bool three_d_ = (ndim == 3);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PointMassGravity", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellCenter();
        const auto &hx = coords.GetScaleFactors();

        // Capture outside constexpr if
        const Real &gm = gm_;

        Real gx1 = 0.0, gx2 = 0.0, gx3 = 0.0;
        Real dr = Null<Real>();
        [[maybe_unused]] Real pos[3] = {pos_[0], pos_[1], pos_[2]};
        [[maybe_unused]] Real rsft2 = rsft2_;
        [[maybe_unused]] bool multi_d = multi_d_;
        [[maybe_unused]] bool three_d = three_d_;
        if constexpr (geometry::is_axisymmetric<GEOM>()) {
          if constexpr (geometry::is_spherical<GEOM>()) {
            const Real rad2 = SQR(dx[0]) + rsft2;
            gx1 = -gm / rad2;
            dr = std::sqrt(rad2);
          } else if constexpr (GEOM == Coordinates::axisymmetric) {
            const auto &[dxs, ex1, ex2, ex3] = coords.ConvertToSphWithVec(dx);
            dr = dxs[0];
            const Real rad2 = SQR(dr) + rsft2;
            const Real g = -gm / rad2;
            gx1 = g * ex1[0];
            gx2 = g * ex3[0];
          }
        } else {
          const auto &[dxc_, ex1, ex2, ex3] = coords.ConvertToCartWithVec(dx);

          // Calculate force in cartesian coordinates
          auto dxc = dxc_;
          for (int n = 0; n < 3; n++) {
            dxc[n] -= pos[n];
          }

          // Convert to spherical coordinates centered on the point mass
          geometry::Coords<Coordinates::cartesian> coords_c;
          const auto &dxs = coords_c.ConvertToSph(dxc);
          dr = dxs[0];

          // Compute acceleration
          const Real rad2 = SQR(dr) + rsft2;
          const Real idr3 = 1.0 / (std::sqrt(rad2) * rad2);
          Real g[3] = {-gm * dxc[0] * idr3, (multi_d) * (-gm * dxc[1] * idr3),
                       (three_d) * (-gm * dxc[2] * idr3)};

          // Convert back to problem coordinates
          // An alternative approach would be to convert to spherical system centered on
          // the point mass, then calculate -GM/r^2 \hat{r}, and convert back to problem
          // coords
          gx1 = g[0] * ex1[0] + g[1] * ex1[1] + g[2] * ex1[2];
          gx2 = g[0] * ex2[0] + g[1] * ex2[1] + g[2] * ex2[2];
          gx3 = g[0] * ex3[0] + g[1] * ex3[1] + g[2] * ex3[2];
        }

        // NOTE(ADM): if we have stability problems, we might want to uncomment this
        // Real com = 0.0;
        // if constexpr (GEOM != Coordinates::cartesian) {
        //   geometry::BBox bnds(coords,k,j,i);
        //   if constexpr((GEOM == Coordinates::spherical) ||
        //                (GEOM == Coordinates::axisymmetric)) {
        //     com = -gm*geometry::dh3dx1<GEOM>(bnds)/dx[0];
        //   } else if constexpr( (GEOM == Coordinates::cylindrical)) {
        //     com = -gm*geometry::dh2dx1<GEOM>(bnds)/dx[0];
        //   }
        // }
        // ... com + (gx1 - com)

        // Mass accretion
        //  d(rho)/dt = - f * rho
        //  d(rho v)/dt = - f * rho*v
        //  rho' = rho_0/ (1 + g*dt)
        //  (rho' - rho_0) = - f*dt/(1 + f*dt) * rho
        //
        // f = 0 for r > rsink
        // f = f*(x^2) where x = (r-rsink)/rsink for r>rsink
        //
        // Don't remove more than half the mass in one step
        // note that sink_rate has already been multiplied by dt
        const Real sramp = sink_rate * quad_ramp((dr - sink_rad) / sink_rad);
        Real fd = std::min(0.5, sramp / (1.0 + sramp));
        fd *= ((sink_rate > 0.0) && (dr <= sink_rad));

        if (do_gas) {
          // Gravitational acceleration and energy release
          // NOTE(PDM): See comments re: total energy source term in uniform.cpp
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            const Real &rho = vmesh(b, gas::prim::density(n), k, j, i);
            const Real &v1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            const Real &v2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            const Real &v3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            const Real &sie = vmesh(b, gas::prim::sie(n), k, j, i);
            const Real tote = rho * (sie + 0.5 * (SQR(v1) + SQR(v2) + SQR(v3)));

            // Add the gravitational acceleration and energy release
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += dt * rho * hx[0] * gx1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += dt * rho * hx[1] * gx2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += dt * rho * hx[2] * gx3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                dt * rho * (v1 * gx1 + v2 * gx2 + v3 * gx3);

            // Add mass accretion
            vmesh(b, gas::cons::density(n), k, j, i) -= fd * rho;
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) -= fd * hx[0] * rho * v1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) -= fd * hx[1] * rho * v2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) -= fd * hx[2] * rho * v3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) -= fd * tote;
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            const Real &rho = vmesh(b, dust::prim::density(n), k, j, i);
            const Real &v1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
            const Real &v2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
            const Real &v3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);

            // Add the gravitational acceleration
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) += dt * rho * hx[0] * gx1;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) += dt * rho * hx[1] * gx2;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) += dt * rho * hx[2] * gx3;

            // Add mass accretion
            vmesh(b, dust::cons::density(n), k, j, i) -= fd * rho;
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) -= fd * hx[0] * rho * v1;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) -= fd * hx[1] * rho * v2;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) -= fd * hx[2] * rho * v3;
          }
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> MD;
template TaskStatus PointMassGravity<C::cartesian>(MD *m, const Real t, const Real d);
template TaskStatus PointMassGravity<C::cylindrical>(MD *m, const Real t, const Real d);
template TaskStatus PointMassGravity<C::spherical1D>(MD *m, const Real t, const Real d);
template TaskStatus PointMassGravity<C::spherical2D>(MD *m, const Real t, const Real d);
template TaskStatus PointMassGravity<C::spherical3D>(MD *m, const Real t, const Real d);
template TaskStatus PointMassGravity<C::axisymmetric>(MD *m, const Real t, const Real d);

} // namespace Gravity
