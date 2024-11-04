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
//! \fn  TaskStatus Gravity::BinaryMassGravity
//! \brief Applies accelerations due to a binary
template <Coordinates GEOM>
TaskStatus BinaryMassGravity(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const bool do_rf = artemis_pkg->template Param<bool>("do_rotating_frame");
  if (!(do_gas || do_dust)) return TaskStatus::complete;

  auto &gravity_pkg = pm->packages.Get("gravity");
  const Real gm = gravity_pkg->template Param<Real>("gm");
  const Real qb = gravity_pkg->template Param<Real>("q");
  const Real mu1 = 1. / (1.0 + qb);
  const Real mu2 = qb / (1.0 + qb);
  const Real sink_rad1 = gravity_pkg->template Param<Real>("sink1");
  const Real sink_rad2 = gravity_pkg->template Param<Real>("sink2");
  const Real sink_rate1 = dt * (gravity_pkg->template Param<Real>("sink_rate1"));
  const Real sink_rate2 = dt * (gravity_pkg->template Param<Real>("sink_rate2"));
  const Real rsft1 = gravity_pkg->template Param<Real>("soft1");
  const Real rsft2 = gravity_pkg->template Param<Real>("soft2");

  const Real omf =
      (do_rf) ? pm->packages.Get("rotating_frame")->template Param<Real>("omega") : 0.0;

  // Calculate orbital parameters
  auto orb = gravity_pkg->template Param<Orbit>("orb");
  Real rb[3] = {Null<Real>()};
  Real vb[3] = {Null<Real>()};
  orb.solve(time, omf, rb, vb);

  // The following assumes x1,x2,x3 are always in Cartesian coordinates.  Usually this is
  // just used for a central object at (0,0,0)...
  const Real com[3] = {gravity_pkg->template Param<Real>("x"),
                       gravity_pkg->template Param<Real>("y"),
                       gravity_pkg->template Param<Real>("z")};
  Real pos1[3] = {Null<Real>()};
  Real pos2[3] = {Null<Real>()};
  for (int n = 0; n < 3; n++) {
    pos1[n] = com[n] - mu2 * rb[n];
    pos2[n] = com[n] + mu1 * rb[n];
  }

  // Packing and indexing
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
  const bool multi_d = (ndim >= 2);
  const bool three_d = (ndim == 3);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "BinaryMassGravity", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinate information
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellCenter();

        const auto &[dxc1_, ex1, ex2, ex3] = coords.ConvertToCartWithVec(dx);
        auto dxc1 = dxc1_;
        auto dxc2 = NewArray<Real, 3>();

        const auto &hx = coords.GetScaleFactors();

        // Calculate force in Cartesian coordinates
        for (int n = 0; n < 3; n++) {
          dxc2[n] = dxc1[n] - pos2[n];
          dxc1[n] -= pos1[n];
        }

        // Convert to spherical coordinates centered on the point mass
        geometry::Coords<Coordinates::cartesian> coords_cart;
        const auto &dxs1 = coords_cart.ConvertToSph(dxc1);
        const auto &dxs2 = coords_cart.ConvertToSph(dxc2);

        // Compute acceleration
        const Real rad2_1 = SQR(dxs1[0]) + SQR(rsft1);
        const Real rad2_2 = SQR(dxs2[0]) + SQR(rsft2);
        const Real idr3_1 = 1.0 / (std::sqrt(rad2_1) * rad2_1);
        const Real idr3_2 = 1.0 / (std::sqrt(rad2_2) * rad2_2);
        Real g[3] = {-gm * (mu1 * dxc1[0] * idr3_1 + mu2 * dxc2[0] * idr3_2),
                     multi_d * (-gm * (mu1 * dxc1[1] * idr3_1 + mu2 * dxc2[1] * idr3_2)),
                     three_d * (-gm * (mu1 * dxc1[2] * idr3_1 + mu2 * dxc2[2] * idr3_2))};

        // Convert back to problem coordinates
        // An alternative approach would be to convert to spherical system centered on the
        // point mass, then calculate -GM/r^2 \hat{r}, and convert back to problem coords
        const Real gx1 = g[0] * ex1[0] + g[1] * ex1[1] + g[2] * ex1[2];
        const Real gx2 = g[0] * ex2[0] + g[1] * ex2[1] + g[2] * ex2[2];
        const Real gx3 = g[0] * ex3[0] + g[1] * ex3[1] + g[2] * ex3[2];

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
        const Real sramp1 = sink_rate1 * quad_ramp((dxs1[0] - sink_rad1) / sink_rad1);
        const Real sramp2 = sink_rate2 * quad_ramp((dxs2[0] - sink_rad2) / sink_rad2);
        Real fd1 = std::min(0.25, sramp1 / (1.0 + sramp1));
        Real fd2 = std::min(0.25, sramp2 / (1.0 + sramp2));
        fd1 *= ((sink_rate1 > 0.0) && (sink_rad1 > 0.0) && (dxs1[0] <= sink_rad1));
        fd2 *= ((sink_rate2 > 0.0) && (sink_rad2 > 0.0) && (dxs2[0] <= sink_rad2));
        const Real fdsum = fd1 + fd2;

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
            vmesh(b, gas::cons::density(n), k, j, i) -= fdsum * rho;
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) -= fdsum * hx[0] * rho * v1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) -= fdsum * hx[1] * rho * v2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) -= fdsum * hx[2] * rho * v3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) -= fdsum * tote;
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
            vmesh(b, dust::cons::density(n), k, j, i) -= fdsum * rho;
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) -= fdsum * hx[0] * rho * v1;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) -= fdsum * hx[1] * rho * v2;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) -= fdsum * hx[2] * rho * v3;
          }
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> MD;
template TaskStatus BinaryMassGravity<C::cartesian>(MD *m, const Real t, const Real d);
template TaskStatus BinaryMassGravity<C::cylindrical>(MD *m, const Real t, const Real d);
template TaskStatus BinaryMassGravity<C::spherical3D>(MD *m, const Real t, const Real d);

} // namespace Gravity
