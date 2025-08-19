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
#ifndef ROTATING_FRAME_ROTATING_FRAME_IMPL_HPP_
#define ROTATING_FRAME_ROTATING_FRAME_IMPL_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

using ArtemisUtils::VI;

namespace RotatingFrame {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ShearingBoxImpl
//! \brief Calculate the shearing box frame body forces
TaskStatus ShearingBoxImpl(MeshData<Real> *md, const Real om0, const Real qshear,
                           const bool do_gas, const bool do_dust, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Packing
  static auto desc = parthenon::MakePackDescriptor<
      gas::cons::momentum, gas::cons::total_energy, dust::cons::momentum,
      gas::prim::density, gas::prim::velocity, dust::prim::density, dust::prim::velocity>(
      resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Indexing and dimensionality
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int three_d = (pm->ndim == 3);

  // Source term prefactors
  const Real qom = qshear * om0;
  const Real two_om = 2.0 * om0;
  const Real qm2_om = qom - two_om;
  const Real g3_over_x3 = (three_d) * (-SQR(om0));

  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ShearingBox", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Evaluate vertical gravity
        geometry::Coords<Coordinates::cartesian> coords(cpars, vmesh.GetCoordinates(b), k,
                                                        j, i);
        const Real g3 = g3_over_x3 * coords.x3v();

        if (do_gas) {
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            const Real &dd = vmesh(b, gas::prim::density(n), k, j, i);
            const Real &v1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            const Real &v2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            const Real &v3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            const Real rdt = dd * dt;
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += rdt * two_om * v2;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += rdt * qm2_om * v1;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += rdt * g3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                rdt * (qom * v1 * v2 + v3 * g3);
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            const Real &dd = vmesh(b, dust::prim::density(n), k, j, i);
            const Real &v1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
            const Real &v2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
            const Real rdt = dd * dt;
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) += rdt * two_om * v2;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) += rdt * qm2_om * v1;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) += rdt * g3;
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RotatingFrameImpl
//! \brief Calculate the rotating frame body forces
template <Coordinates GEOM>
TaskStatus RotatingFrameImpl(MeshData<Real> *md, const Real om0, const bool do_gas,
                             const bool do_dust, const Real dt) {
  // Adds the rotating frame terms to the azimuthal momentum equation and the energy
  // equation. Note that in comments in this function, R is always the cylindrical radius.

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc_flux =
      parthenon::MakePackDescriptor<gas::cons::density, gas::cons::momentum,
                                    gas::cons::total_energy, dust::cons::density,
                                    dust::cons::momentum>(resolved_pkgs.get(), {},
                                                          {parthenon::PDOpt::WithFluxes});
  auto vf = desc_flux.GetPack(md);
  static auto desc_g =
      parthenon::MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::ax1, geom::ax2,
                                    geom::ax3, geom::vol, geom::rfw1, geom::rfw2,
                                    geom::rfw3>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);
  const int multi_d = (pm->ndim >= 2);
  const int three_d = (pm->ndim == 3);

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  const Real omdt = om0 * dt;
  const Real om2dt = omdt * om0;
  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RotatingFrame", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vf.GetCoordinates(b), k, j, i);
        const std::array<Real, 3> xv{
            vg(0, geom::x1v(), coords.template index<geom::x1v>(k, j, i)),
            vg(0, geom::x2v(), coords.template index<geom::x2v>(k, j, i)),
            vg(0, geom::x3v(), coords.template index<geom::x3v>(k, j, i))};
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // The geometry dependent flux weighting
        // \pm <R^2>_\pm - <R^2>
        const std::array<Real, 2> ax1{
            vg(b, geom::ax1(), coords.template index<geom::ax1>(k, j, i)) *
                vg(b, geom::rfw1(), coords.template index<geom::rfw1>(k, j, i)),
            vg(b, geom::ax1(), coords.template index<geom::ax1>(k, j, i + 1)) *
                vg(b, geom::rfw1(), coords.template index<geom::rfw1>(k, j, i + 1))};
        const std::array<Real, 2> ax2{
            vg(b, geom::ax2(), coords.template index<geom::ax2>(k, j, i)) *
                vg(b, geom::rfw2(), coords.template index<geom::rfw2>(k, j, i)),
            vg(b, geom::ax2(), coords.template index<geom::ax2>(k, j + multi_d, i)) *
                vg(b, geom::rfw2(),
                   coords.template index<geom::rfw2>(k, j + multi_d, i))};
        const std::array<Real, 2> ax3{
            vg(b, geom::ax3(), coords.template index<geom::ax3>(k, j, i)) *
                vg(b, geom::rfw3(), coords.template index<geom::rfw3>(k, j, i)),
            vg(b, geom::ax3(), coords.template index<geom::ax3>(k + three_d, j, i)) *
                vg(b, geom::rfw3(),
                   coords.template index<geom::rfw3>(k + three_d, j, i))};

        const Real vol = vg(b, geom::vol(), coords.template index<geom::vol>(k, j, i));
        if (do_gas) {
          for (int n = 0; n < vf.GetSize(b, gas::cons::density()); ++n) {

            const Real divf =
                (vf.flux(b, X1DIR, gas::cons::density(n), k, j, i) * ax1[0] * bx1[0] +
                 vf.flux(b, X1DIR, gas::cons::density(n), k, j, i + 1) * ax1[1] *
                     bx1[1]) +
                multi_d *
                    (vf.flux(b, X2DIR, gas::cons::density(n), k, j, i) * ax2[0] * bx2[0] +
                     vf.flux(b, X2DIR, gas::cons::density(n), k, j + multi_d, i) *
                         ax2[1] * bx2[1]) +
                three_d *
                    (vf.flux(b, X3DIR, gas::cons::density(n), k, j, i) * ax3[0] * bx3[0] +
                     vf.flux(b, X3DIR, gas::cons::density(n), k + three_d, j, i) *
                         ax3[1] * bx3[1]);

            // dUphi/dt = - f . phi_hat where f = div.F phi_hat,cyl
            vf(b, gas::cons::momentum(VI(n, 0)), k, j, i) -= omdt * (divf / vol) * ex1[1];
            vf(b, gas::cons::momentum(VI(n, 1)), k, j, i) -= omdt * (divf / vol) * ex2[1];
            vf(b, gas::cons::momentum(VI(n, 2)), k, j, i) -= omdt * (divf / vol) * ex3[1];

            // average or area weighted? (Fm + Fp)/2 or (Ap*Fp + Am*Fm)/(Am + Ap)
            const Real fx[3] = {
                0.5 * (vf.flux(b, X1DIR, gas::cons::density(n), k, j, i) +
                       vf.flux(b, X1DIR, gas::cons::density(n), k, j, i + 1)),
                multi_d * 0.5 *
                    (vf.flux(b, X2DIR, gas::cons::density(n), k, j, i) +
                     vf.flux(b, X2DIR, gas::cons::density(n), k, j + multi_d, i)),
                three_d * 0.5 *
                    (vf.flux(b, X3DIR, gas::cons::density(n), k, j, i) +
                     vf.flux(b, X3DIR, gas::cons::density(n), k + three_d, j, i))};

            // + omega^2 R * F . R_hat * dt
            vf(b, gas::cons::total_energy(n), k, j, i) +=
                om2dt * xcyl[0] * (fx[0] * ex1[0] + fx[1] * ex2[0] + fx[2] * ex3[0]);
          }
        }
        if (do_dust) {
          for (int n = 0; n < vf.GetSize(b, dust::cons::density()); ++n) {
            const Real divf =
                (vf.flux(b, X1DIR, dust::cons::density(n), k, j, i) * ax1[0] * bx1[0] +
                 vf.flux(b, X1DIR, dust::cons::density(n), k, j, i + 1) * ax1[1] *
                     bx1[1]) +
                multi_d * (vf.flux(b, X2DIR, dust::cons::density(n), k, j, i) * ax2[0] *
                               bx2[0] +
                           vf.flux(b, X2DIR, dust::cons::density(n), k, j + multi_d, i) *
                               ax2[1] * bx2[1]) +
                three_d * (vf.flux(b, X3DIR, dust::cons::density(n), k, j, i) * ax3[0] *
                               bx3[0] +
                           vf.flux(b, X3DIR, dust::cons::density(n), k + three_d, j, i) *
                               ax3[1] * bx3[1]);

            vf(b, dust::cons::momentum(VI(n, 0)), k, j, i) -=
                omdt * (divf / vol) * ex1[1];
            vf(b, dust::cons::momentum(VI(n, 1)), k, j, i) -=
                omdt * (divf / vol) * ex2[1];
            vf(b, dust::cons::momentum(VI(n, 2)), k, j, i) -=
                omdt * (divf / vol) * ex3[1];
          }
        }
      });

  return TaskStatus::complete;
}

} // namespace RotatingFrame

#endif // ROTATING_FRAME_ROTATING_FRAME_IMPL_HPP_
