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
  const Real two_om = 2.0 * om0;
  const Real qm2_om = (qshear - 2.0) * om0;
  const Real homsq = (three_d) * (0.5 * SQR(om0));

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ShearingBox", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<Coordinates::cartesian> coords(vmesh.GetCoordinates(b), k, j, i);
        const Real &zp = coords.bnds.x3[1];
        const Real &zm = coords.bnds.x3[0];

        // Evaluate vertical gravity (at cell center)
        const Real g3 = homsq * (SQR(zm) - SQR(zp)) / (zp - zm);

        if (do_gas) {
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            const Real &dd = vmesh(b, gas::prim::density(n), k, j, i);
            const Real &v1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            const Real &v2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            const Real &v3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            const Real rdt = dd * dt;
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += rdt * two_om * v2;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += rdt * qm2_om * v1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += rdt * g3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) += rdt * v3 * g3;
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
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) += rdt * g3;
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
  const int multi_d = (pm->ndim >= 2);
  const int three_d = (pm->ndim == 3);

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  const Real omdt = om0 * dt;
  const Real om2dt = omdt * om0;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RotatingFrame", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vf.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // The geometry dependent flux weighting
        // \pm <R^2>_\pm - <R^2>
        const auto &[bx1, bx2, bx3] = coords.GetRFWeights();

        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = (multi_d) ? coords.GetFaceAreaX2() : NewArray<Real, 2>(0.0);
        const auto ax3 = (three_d) ? coords.GetFaceAreaX3() : NewArray<Real, 2>(0.0);

        const Real vol = coords.Volume();
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
