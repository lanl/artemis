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
  PARTHENON_INSTRUMENT
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
TaskStatus RotatingFrameImpl(MeshData<Real> *md, const Real om0, const bool do_oa,
                             const Real gm, const bool do_gas, const bool do_dust,
                             const Real dt) {
  PARTHENON_INSTRUMENT
  // Adds the rotating frame terms to the azimuthal momentum equation and the energy
  // equation. Note that in comments in this function, R is always the cylindrical radius.
  //
  // When orbital advection is active, the stored primitive velocity is
  //   v_stored = v_full - v_bg(R)
  // where v_bg = R*(OmegaKep(R) - omega_f) in the phi direction. The Riemann solver
  // transports the stored momentum, so we must correct for the missing background
  // angular momentum flux: d(rho*v_stored)/dt -= (1/V) * div(F_rho * v_bg).
  // The RF part (omega_f * R) uses the RFWeights formulation; the OA part uses
  // face-centered v_bg directly.

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
                                    geom::ax3, geom::vol, geom::rfw1m, geom::rfw1p,
                                    geom::rfw2m, geom::rfw2p, geom::rfw3m, geom::rfw3p>(
          resolved_pkgs.get());
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
        const auto &xv = coords.GetCellCenter(vg, b, k, j, i);
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // The geometry dependent flux weighting
        // \pm <R^2>_\pm - <R^2>
        const auto &[bx1, bx2, bx3] = coords.GetRFWeights(vg, b, k, j, i);
        const auto &ax1 = coords.GetFaceAreaX1(vg, b, k, j, i);
        const auto &ax2 = coords.GetFaceAreaX2(vg, b, k, j, i);
        const auto &ax3 = coords.GetFaceAreaX3(vg, b, k, j, i);
        const Real vol = coords.GetVolume(vg, b, k, j, i);

        // Compute face-centered OA background angular velocity (OmegaKep - omega_f)
        // for the orbital advection transport correction.
        // Face coordinates give us R at each face.
        Real oa_lbg_x1m = 0.0, oa_lbg_x1p = 0.0;
        Real oa_lbg_x2m = 0.0, oa_lbg_x2p = 0.0;
        Real oa_lbg_x3m = 0.0, oa_lbg_x3p = 0.0;
        if (do_oa) {
          const Real Rc = xcyl[0];
          const Real lbg_c = Rc * Rc * (OmegaKep(gm, xv[0]) - om0);

          // x1 faces
          const auto xf1m = coords.FaceCenX1(geometry::CellFace::lower);
          const auto xf1p = coords.FaceCenX1(geometry::CellFace::upper);
          const Real Rf1m = coords.ConvertToCyl(xf1m)[0];
          const Real Rf1p = coords.ConvertToCyl(xf1p)[0];
          oa_lbg_x1m = lbg_c - Rf1m * Rf1m * (OmegaKep(gm, xf1m[0]) - om0);
          oa_lbg_x1p = Rf1p * Rf1p * (OmegaKep(gm, xf1p[0]) - om0) - lbg_c;

          if (multi_d) {
            const auto xf2m = coords.FaceCenX2(geometry::CellFace::lower);
            const auto xf2p = coords.FaceCenX2(geometry::CellFace::upper);
            const Real Rf2m = coords.ConvertToCyl(xf2m)[0];
            const Real Rf2p = coords.ConvertToCyl(xf2p)[0];
            oa_lbg_x2m = lbg_c - Rf2m * Rf2m * (OmegaKep(gm, xf2m[0]) - om0);
            oa_lbg_x2p = Rf2p * Rf2p * (OmegaKep(gm, xf2p[0]) - om0) - lbg_c;
          }
          if (three_d) {
            const auto xf3m = coords.FaceCenX3(geometry::CellFace::lower);
            const auto xf3p = coords.FaceCenX3(geometry::CellFace::upper);
            const Real Rf3m = coords.ConvertToCyl(xf3m)[0];
            const Real Rf3p = coords.ConvertToCyl(xf3p)[0];
            oa_lbg_x3m = lbg_c - Rf3m * Rf3m * (OmegaKep(gm, xf3m[0]) - om0);
            oa_lbg_x3p = Rf3p * Rf3p * (OmegaKep(gm, xf3p[0]) - om0) - lbg_c;
          }
        }

        if (do_gas) {
          for (int n = 0; n < vf.GetSize(b, gas::cons::density()); ++n) {

            // Rotating frame angular momentum transport correction (omega * R^2 part)
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

            // OA angular momentum transport correction
            // div(F_rho * l_bg) using face-centered l_bg
            Real oa_divfl = 0.0;
            if (do_oa) {
              oa_divfl =
                  (vf.flux(b, X1DIR, gas::cons::density(n), k, j, i) * ax1[0] *
                       oa_lbg_x1m +
                   vf.flux(b, X1DIR, gas::cons::density(n), k, j, i + 1) * ax1[1] *
                       oa_lbg_x1p) +
                  multi_d * (vf.flux(b, X2DIR, gas::cons::density(n), k, j, i) * ax2[0] *
                                 oa_lbg_x2m +
                             vf.flux(b, X2DIR, gas::cons::density(n), k, j + multi_d, i) *
                                 ax2[1] * oa_lbg_x2p) +
                  three_d * (vf.flux(b, X3DIR, gas::cons::density(n), k, j, i) * ax3[0] *
                                 oa_lbg_x3m +
                             vf.flux(b, X3DIR, gas::cons::density(n), k + three_d, j, i) *
                                 ax3[1] * oa_lbg_x3p);
            }

            const Real mom_src = dt / vol * (om0 * divf + oa_divfl);
            vf(b, gas::cons::momentum(VI(n, 0)), k, j, i) -= mom_src * ex1[1];
            vf(b, gas::cons::momentum(VI(n, 1)), k, j, i) -= mom_src * ex2[1];
            vf(b, gas::cons::momentum(VI(n, 2)), k, j, i) -= mom_src * ex3[1];

            // Energy correction: only the rotating frame centrifugal work
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
            // Rotating frame correction
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

            // OA correction for dust
            Real oa_divfl = 0.0;
            if (do_oa) {
              oa_divfl =
                  (vf.flux(b, X1DIR, dust::cons::density(n), k, j, i) * ax1[0] *
                       oa_lbg_x1m +
                   vf.flux(b, X1DIR, dust::cons::density(n), k, j, i + 1) * ax1[1] *
                       oa_lbg_x1p) +
                  multi_d *
                      (vf.flux(b, X2DIR, dust::cons::density(n), k, j, i) * ax2[0] *
                           oa_lbg_x2m +
                       vf.flux(b, X2DIR, dust::cons::density(n), k, j + multi_d, i) *
                           ax2[1] * oa_lbg_x2p) +
                  three_d *
                      (vf.flux(b, X3DIR, dust::cons::density(n), k, j, i) * ax3[0] *
                           oa_lbg_x3m +
                       vf.flux(b, X3DIR, dust::cons::density(n), k + three_d, j, i) *
                           ax3[1] * oa_lbg_x3p);
            }

            const Real mom_src = dt / vol * (om0 * divf + oa_divfl);
            vf(b, dust::cons::momentum(VI(n, 0)), k, j, i) -= mom_src * ex1[1];
            vf(b, dust::cons::momentum(VI(n, 1)), k, j, i) -= mom_src * ex2[1];
            vf(b, dust::cons::momentum(VI(n, 2)), k, j, i) -= mom_src * ex3[1];
          }
        }
      });

  return TaskStatus::complete;
}

} // namespace RotatingFrame

#endif // ROTATING_FRAME_ROTATING_FRAME_IMPL_HPP_
