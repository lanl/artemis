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
#ifndef UTILS_DIFFUSION_MOMENTUM_DIFFUSION_HPP_
#define UTILS_DIFFUSION_MOMENTUM_DIFFUSION_HPP_

// Artemis includes
#include "artemis.hpp"
#include "diffusion_coeff.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace Diffusion {

template <Coordinates GEOM, Fluid FLUID_TYPE, parthenon::CoordinateDirection XDIR,
          typename SparsePack>
KOKKOS_INLINE_FUNCTION void
StrainTensorFace(parthenon::team_mbr_t const &member, const int b, const int n,
                 const int k, const int j, const int il, const int iu, const int multid,
                 const int threed, const SparsePack &vprim,
                 const parthenon::ScratchPad2D<Real> &flx) {
  // Fill the flx array with the strain tensor on the specified face
  //
  //
  // +---------------------+---------------------+
  // |                     |                     |
  // |                     |                     |
  // |                     |                     |
  // |        (i-1,j+1)    |       (i,j+1)       |
  // |          |          |          |          |
  // |          |          |          |          |
  // |          |          |          |          |
  // +----------|----------+----------|----------+
  // |          |          |          |          |
  // |          |          |          |          |
  // |          v          |          v          |
  // |        dvdxj-------> <-------dvdxj        |
  // |          ^        dv/dxi       ^          |
  // |          |          |          |          |
  // |          |          |          |          |
  // +----------|----------+----------|----------+
  // |          |          |          |          |
  // |          |          |          |          |
  // |          |          |          |          |
  // |        (i-1,j-1)    |       (i,j-1)       |
  // |                     |                     |
  // |                     |                     |
  // |                     |                     |
  // +---------------------+---------------------+

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Only gas fluids have momentum diffusion");
  auto pco = vprim.GetCoordinates(b);
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    geometry::Coords<GEOM> coords(pco, k, j, i);
    //  T_j^i = dv^i/dxj + hj^2/hi^2 dv^j/dxi  + v^k dhi/dxk / hi \delta_j^i
    Real xv[3] = {Null<Real>()};
    coords.GetCellCenter(xv);
    Real hx[3] = {Null<Real>()};
    coords.GetScaleFactors(hx);
    const Real v[3] = {vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i) / hx[0],
                       vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i) / hx[1],
                       vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i) / hx[2]};
    Real xf[3] = {Null<Real>()};

    if constexpr (XDIR == X1DIR) {
      coords.FaceCenX1(geometry::CellFace::lower, xf);
    } else if constexpr (XDIR == X2DIR) {
      coords.FaceCenX2(geometry::CellFace::lower, xf);
    } else if constexpr (XDIR == X3DIR) {
      coords.FaceCenX3(geometry::CellFace::lower, xf);
    }
    Real hxf[3] = {coords.hx1(xf[0], xf[1], xf[2]), coords.hx2(xf[0], xf[1], xf[2]),
                   coords.hx3(xf[0], xf[1], xf[2])};

    if constexpr (XDIR == X1DIR) {
      // T_*^1  flx = { T_1^1 , T_2^1 , T_3^1 }

      // need xm
      //      ym , yp, xmym, xmyp
      //      zm , zp, xmzm, xmzp

      geometry::Coords<GEOM> coords_xm(pco, k, j, i - 1);
      geometry::Coords<GEOM> coords_xmym(pco, k, j - multid, i - 1);
      geometry::Coords<GEOM> coords_xmyp(pco, k, j + multid, i - 1);
      geometry::Coords<GEOM> coords_ym(pco, k, j - multid, i);
      geometry::Coords<GEOM> coords_yp(pco, k, j + multid, i);

      geometry::Coords<GEOM> coords_xmzm(pco, k - threed, j, i - 1);
      geometry::Coords<GEOM> coords_xmzp(pco, k + threed, j, i - 1);
      geometry::Coords<GEOM> coords_zm(pco, k - threed, j, i);
      geometry::Coords<GEOM> coords_zp(pco, k + threed, j, i);

      Real xv_xm[3] = {Null<Real>()};
      Real xv_yp[3] = {Null<Real>()};
      Real xv_ym[3] = {Null<Real>()};
      Real xv_xmyp[3] = {Null<Real>()};
      Real xv_xmym[3] = {Null<Real>()};
      Real xv_zp[3] = {Null<Real>()};
      Real xv_zm[3] = {Null<Real>()};
      Real xv_xmzp[3] = {Null<Real>()};
      Real xv_xmzm[3] = {Null<Real>()};

      coords_xm.GetCellCenter(xv_xm);

      coords_ym.GetCellCenter(xv_ym);
      coords_yp.GetCellCenter(xv_yp);
      coords_xmym.GetCellCenter(xv_xmym);
      coords_xmyp.GetCellCenter(xv_xmyp);

      coords_zm.GetCellCenter(xv_zm);
      coords_zp.GetCellCenter(xv_zp);
      coords_xmzm.GetCellCenter(xv_xmzm);
      coords_xmzp.GetCellCenter(xv_xmzp);

      Real hx_xm[3] = {Null<Real>()};

      Real hx_ym[3] = {Null<Real>()};
      Real hx_yp[3] = {Null<Real>()};
      Real hx_xmym[3] = {Null<Real>()};
      Real hx_xmyp[3] = {Null<Real>()};

      Real hx_zm[3] = {Null<Real>()};
      Real hx_zp[3] = {Null<Real>()};
      Real hx_xmzm[3] = {Null<Real>()};
      Real hx_xmzp[3] = {Null<Real>()};

      coords_xm.GetScaleFactors(hx_xm);

      coords_ym.GetScaleFactors(hx_ym);
      coords_yp.GetScaleFactors(hx_yp);
      coords_xmym.GetScaleFactors(hx_xmym);
      coords_xmyp.GetScaleFactors(hx_xmyp);

      coords_zm.GetScaleFactors(hx_zm);
      coords_zp.GetScaleFactors(hx_zp);
      coords_xmzm.GetScaleFactors(hx_xmzm);
      coords_xmzp.GetScaleFactors(hx_xmzp);

      const Real dx1 = coords.Distance(xv, xv_xm);
      const Real dx2 = multid ? coords.Distance(xv_ym, xv_yp) : Fuzz<Real>();
      const Real dx2_xm = multid ? coords.Distance(xv_xmym, xv_xmyp) : Fuzz<Real>();
      const Real dx3 = threed ? coords.Distance(xv_zm, xv_zp) : Fuzz<Real>();
      const Real dx3_xm = threed ? coords.Distance(xv_xmzm, xv_xmzp) : Fuzz<Real>();

      // T_1^1  = 2 dv^1/dx1 +  v^k dh1/xk / h1
      const Real dv1 =
          v[0] - vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i - 1) / hx_xm[0];

      const Real src =
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i) / hx[0] * coords.dh1dx1() +
          vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i) / hx[1] * coords.dh1dx2() +
          vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i) / hx[2] * coords.dh1dx3();

      const Real src_xm = vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i - 1) /
                              hx_xm[0] * coords_xm.dh1dx1() +
                          vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i - 1) /
                              hx_xm[1] * coords_xm.dh1dx2() +
                          vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i - 1) /
                              hx_xm[2] * coords_xm.dh1dx3();

      flx(0, i) = 2 * dv1 / dx1 + 0.5 * (src + src_xm);

      // T_2^1  = dv^1/dx2 +  h2^2/h1^2 dv^2/dx1
      const Real dv2 =
          v[1] - vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i - 1) / hx_xm[1];

      const Real dv12 =
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j + multid, i) / hx_yp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j - multid, i) / hx_ym[0];

      const Real dv12_xm =
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j + multid, i - 1) / hx_xmyp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j - multid, i - 1) / hx_xmym[0];

      flx(1, i) = multid * 0.5 * (dv12 / dx2 + dv12_xm / dx2_xm) +
                  SQR(hxf[1] / hxf[0]) * dv2 / dx1;

      // T_3^1  = dv^1/dx3 +  h3^2/h1^2 dv^3/dx1
      const Real dv3 =
          v[2] - vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i - 1) / hx_xm[2];

      const Real dv13 =
          vprim(b, gas::prim::velocity(VI(n, 0)), k + threed, j, i) / hx_zp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k - threed, j, i) / hx_zm[0];

      const Real dv13_xm =
          vprim(b, gas::prim::velocity(VI(n, 0)), k + threed, j, i - 1) / hx_xmzp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k - threed, j, i - 1) / hx_xmzm[0];

      flx(2, i) = threed * 0.5 * (dv13 / dx3 + dv13_xm / dx3_xm) +
                  SQR(hxf[2] / hxf[0]) * dv3 / dx1;

    } else if constexpr (XDIR == X2DIR) {
      // T_*^2  flx = { T_1^2 , T_2^2 , T_3^2 }

      // need ym
      //      xm , xp, xpym, xmym
      //      zm , zp, ymzm, ymzp

      geometry::Coords<GEOM> coords_xp(pco, k, j, i + 1);
      geometry::Coords<GEOM> coords_xm(pco, k, j, i - 1);
      geometry::Coords<GEOM> coords_xpym(pco, k, j - 1, i + 1);
      geometry::Coords<GEOM> coords_xmym(pco, k, j - 1, i - 1);

      geometry::Coords<GEOM> coords_ym(pco, k, j - 1, i);

      geometry::Coords<GEOM> coords_zp(pco, k + threed, j, i);
      geometry::Coords<GEOM> coords_zm(pco, k - threed, j, i);
      geometry::Coords<GEOM> coords_ymzp(pco, k + threed, j - 1, i);
      geometry::Coords<GEOM> coords_ymzm(pco, k - threed, j - 1, i);

      Real xv_xm[3] = {Null<Real>()};
      Real xv_xp[3] = {Null<Real>()};
      Real xv_xmym[3] = {Null<Real>()};
      Real xv_xpym[3] = {Null<Real>()};

      Real xv_ym[3] = {Null<Real>()};

      Real xv_zm[3] = {Null<Real>()};
      Real xv_zp[3] = {Null<Real>()};
      Real xv_ymzm[3] = {Null<Real>()};
      Real xv_ymzp[3] = {Null<Real>()};

      coords_xm.GetCellCenter(xv_xm);
      coords_xp.GetCellCenter(xv_xp);
      coords_xmym.GetCellCenter(xv_xmym);
      coords_xpym.GetCellCenter(xv_xpym);

      coords_ym.GetCellCenter(xv_ym);

      coords_zm.GetCellCenter(xv_zm);
      coords_zp.GetCellCenter(xv_zp);
      coords_ymzm.GetCellCenter(xv_ymzm);
      coords_ymzp.GetCellCenter(xv_ymzp);

      Real hx_xm[3] = {Null<Real>()};
      Real hx_xp[3] = {Null<Real>()};
      Real hx_xmym[3] = {Null<Real>()};
      Real hx_xpym[3] = {Null<Real>()};

      Real hx_ym[3] = {Null<Real>()};

      Real hx_zm[3] = {Null<Real>()};
      Real hx_zp[3] = {Null<Real>()};
      Real hx_ymzm[3] = {Null<Real>()};
      Real hx_ymzp[3] = {Null<Real>()};

      coords_xm.GetScaleFactors(hx_xm);
      coords_xp.GetScaleFactors(hx_xp);
      coords_xmym.GetScaleFactors(hx_xmym);
      coords_xpym.GetScaleFactors(hx_xpym);

      coords_ym.GetScaleFactors(hx_ym);

      coords_zm.GetScaleFactors(hx_zm);
      coords_zp.GetScaleFactors(hx_zp);
      coords_ymzm.GetScaleFactors(hx_ymzm);
      coords_ymzp.GetScaleFactors(hx_ymzp);

      const Real dx1 = coords.Distance(xv_xm, xv_xp);
      const Real dx1_ym = coords.Distance(xv_xmym, xv_xpym);
      const Real dx2 = coords.Distance(xv, xv_ym);
      const Real dx3 = threed ? coords.Distance(xv_zm, xv_zp) : Fuzz<Real>();
      const Real dx3_ym = threed ? coords.Distance(xv_ymzm, xv_ymzp) : Fuzz<Real>();

      // T_1^2 = dv^2/dx1 + h1^2/h2^2 dv^1/dx2

      const Real dv1 =
          v[0] - vprim(b, gas::prim::velocity(VI(n, 0)), k, j - 1, i) / hx_ym[0];

      const Real dv21 = vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i + 1) / hx_xp[1] -
                        vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i - 1) / hx_xm[1];

      const Real dv21_ym =
          vprim(b, gas::prim::velocity(VI(n, 1)), k, j - 1, i + 1) / hx_xpym[1] -
          vprim(b, gas::prim::velocity(VI(n, 1)), k, j - 1, i - 1) / hx_xmym[1];

      flx(0, i) =
          0.5 * (dv21 / dx1 + dv21_ym / dx1_ym) +
          SQR(hxf[0] / hxf[1]) * dv1 / dx2; // fix hx to be the value on the face...

      // T_2^2 = 2 dv^2/dx2 +  v^k dh2/dxk / h2
      const Real dv2 =
          v[1] - vprim(b, gas::prim::velocity(VI(n, 1)), k, j - 1, i) / hx_ym[1];

      const Real src =
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i) / hx[0] * coords.dh2dx1() +
          vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i) / hx[1] * coords.dh2dx2() +
          vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i) / hx[2] * coords.dh2dx3();

      const Real src_ym = vprim(b, gas::prim::velocity(VI(n, 0)), k, j - 1, i) /
                              hx_ym[0] * coords_ym.dh2dx1() +
                          vprim(b, gas::prim::velocity(VI(n, 1)), k, j - 1, i) /
                              hx_ym[1] * coords_ym.dh2dx2() +
                          vprim(b, gas::prim::velocity(VI(n, 2)), k, j - 1, i) /
                              hx_ym[2] * coords_ym.dh2dx3();

      flx(1, i) = 2 * dv2 / dx2 + 0.5 * (src + src_ym);

      // T_3^2 = dv^2/dx3 + h1^3/h2^2 dv^3/dx2

      const Real dv3 =
          v[2] - vprim(b, gas::prim::velocity(VI(n, 2)), k, j - 1, i) / hx_ym[2];

      const Real dv23 =
          vprim(b, gas::prim::velocity(VI(n, 1)), k + threed, j, i) / hx_zp[1] -
          vprim(b, gas::prim::velocity(VI(n, 1)), k - threed, j, i) / hx_zm[1];

      const Real dv23_ym =
          vprim(b, gas::prim::velocity(VI(n, 1)), k + threed, j - 1, i) / hx_ymzp[1] -
          vprim(b, gas::prim::velocity(VI(n, 1)), k - threed, j - 1, i) / hx_ymzm[1];

      flx(2, i) = threed * 0.5 * (dv23 / dx3 + dv23_ym / dx3_ym) +
                  SQR(hxf[2] / hxf[1]) * dv3 / dx2;

    } else if constexpr (XDIR == X3DIR) {
      // T_*^3  flx = { T_1^3 , T_2^3 , T_3^3 }

      geometry::Coords<GEOM> coords_xp(pco, k, j, i + 1);
      geometry::Coords<GEOM> coords_xm(pco, k, j, i - 1);
      geometry::Coords<GEOM> coords_xpzm(pco, k - 1, j, i + 1);
      geometry::Coords<GEOM> coords_xmzm(pco, k - 1, j, i - 1);

      geometry::Coords<GEOM> coords_yp(pco, k, j + 1, i);
      geometry::Coords<GEOM> coords_ym(pco, k, j - 1, i);
      geometry::Coords<GEOM> coords_ypzm(pco, k - 1, j + 1, i);
      geometry::Coords<GEOM> coords_ymzm(pco, k - 1, j - 1, i);

      geometry::Coords<GEOM> coords_zm(pco, k - 1, j, i);

      Real xv_xm[3] = {Null<Real>()};
      Real xv_xp[3] = {Null<Real>()};
      Real xv_xmzm[3] = {Null<Real>()};
      Real xv_xpzm[3] = {Null<Real>()};

      Real xv_ym[3] = {Null<Real>()};
      Real xv_yp[3] = {Null<Real>()};
      Real xv_ymzm[3] = {Null<Real>()};
      Real xv_ypzm[3] = {Null<Real>()};

      Real xv_zm[3] = {Null<Real>()};

      coords_xm.GetCellCenter(xv_xm);
      coords_xp.GetCellCenter(xv_xp);
      coords_xmzm.GetCellCenter(xv_xmzm);
      coords_xpzm.GetCellCenter(xv_xpzm);

      coords_ym.GetCellCenter(xv_ym);
      coords_yp.GetCellCenter(xv_yp);
      coords_ymzm.GetCellCenter(xv_ymzm);
      coords_ypzm.GetCellCenter(xv_ypzm);

      coords_zm.GetCellCenter(xv_zm);

      Real hx_xm[3] = {Null<Real>()};
      Real hx_xp[3] = {Null<Real>()};
      Real hx_xmzm[3] = {Null<Real>()};
      Real hx_xpzm[3] = {Null<Real>()};

      Real hx_ym[3] = {Null<Real>()};
      Real hx_yp[3] = {Null<Real>()};
      Real hx_ymzm[3] = {Null<Real>()};
      Real hx_ypzm[3] = {Null<Real>()};

      Real hx_zm[3] = {Null<Real>()};

      coords_xm.GetScaleFactors(hx_xm);
      coords_xp.GetScaleFactors(hx_xp);
      coords_xmzm.GetScaleFactors(hx_xmzm);
      coords_xpzm.GetScaleFactors(hx_xpzm);

      coords_ym.GetScaleFactors(hx_ym);
      coords_yp.GetScaleFactors(hx_yp);
      coords_ymzm.GetScaleFactors(hx_ymzm);
      coords_ypzm.GetScaleFactors(hx_ypzm);

      coords_zm.GetScaleFactors(hx_zm);

      const Real dx1 = coords.Distance(xv_xm, xv_xp);
      const Real dx1_zm = coords.Distance(xv_xmzm, xv_xpzm);
      const Real dx2 = coords.Distance(xv_ym, xv_yp);
      const Real dx2_zm = coords.Distance(xv_ymzm, xv_ypzm);
      const Real dx3 = coords.Distance(xv, xv_zm);

      // T_1^3 = dv^3/dx1 + h1^2/h3^2 dv^1/dx3

      const Real dv1 =
          v[0] - vprim(b, gas::prim::velocity(VI(n, 0)), k - 1, j, i) / hx_zm[0];

      const Real dv31 = vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i + 1) / hx_xp[2] -
                        vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i - 1) / hx_xm[2];

      const Real dv31_zm =
          vprim(b, gas::prim::velocity(VI(n, 2)), k - 1, j, i + 1) / hx_xpzm[2] -
          vprim(b, gas::prim::velocity(VI(n, 2)), k - 1, j, i - 1) / hx_xmzm[2];

      flx(0, i) =
          0.5 * (dv31 / dx1 + dv31_zm / dx1_zm) + SQR(hxf[0] / hxf[2]) * dv1 / dx3;

      // T_2^3 = dv^3/dx2 + h2^2/h3^2 dv^2/dx3

      const Real dv2 =
          v[1] - vprim(b, gas::prim::velocity(VI(n, 1)), k - 1, j, i) / hx_zm[1];

      const Real dv32 = vprim(b, gas::prim::velocity(VI(n, 2)), k, j + 1, i) / hx_yp[2] -
                        vprim(b, gas::prim::velocity(VI(n, 2)), k, j - 1, i) / hx_ym[2];

      const Real dv32_zm =
          vprim(b, gas::prim::velocity(VI(n, 2)), k - 1, j + 1, i) / hx_ypzm[2] -
          vprim(b, gas::prim::velocity(VI(n, 2)), k - 1, j - 1, i) / hx_ymzm[2];

      flx(1, i) =
          0.5 * (dv32 / dx2 + dv32_zm / dx2_zm) + SQR(hxf[1] / hxf[2]) * dv2 / dx3;

      // T_3^3 = 2 dv^3/dx3 + v^k dh3/dxk /h3
      const Real dv3 =
          v[2] - vprim(b, gas::prim::velocity(VI(n, 2)), k - 1, j, i) / hx_zm[2];

      const Real src =
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i) / hx[0] * coords.dh3dx1() +
          vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i) / hx[1] * coords.dh3dx2() +
          vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i) / hx[2] * coords.dh3dx3();

      const Real src_zm = vprim(b, gas::prim::velocity(VI(n, 0)), k - 1, j, i) /
                              hx_zm[0] * coords_zm.dh3dx1() +
                          vprim(b, gas::prim::velocity(VI(n, 1)), k - 1, j, i) /
                              hx_zm[1] * coords_zm.dh3dx2() +
                          vprim(b, gas::prim::velocity(VI(n, 2)), k - 1, j, i) /
                              hx_zm[2] * coords_zm.dh3dx3();

      flx(2, i) = 2 * dv3 / dx3 + 0.5 * (src + src_zm);
    }
  });
}

template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim,
          typename SparsePackFlux>
KOKKOS_INLINE_FUNCTION void StressTensorFaceX1(
    DiffCoeffParams dp, parthenon::team_mbr_t const &member, const int b, const int n,
    const int k, const int j, const int il, const int iu, const int multid,
    const int threed, const int nspecies, const SparsePackPrim &p,
    const SparsePackFlux &qf, const parthenon::ScratchPad1D<Real> &divu,
    const parthenon::ScratchPad1D<Real> &mu, const parthenon::ScratchPad2D<Real> &flx) {
  // Fill the flx array with the stress tensor on the specified face

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Only gas fluids have momentum diffusion");
  auto pco = p.GetCoordinates(b);
  const bool avg = (dp.avg == DiffAvg::arithmetic);
  const bool havg = (dp.avg == DiffAvg::harmonic);
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    //  T_j^i = dv^i/dxj + hj^2/hi^2 dv^j/dxi  + v^k dhi/dxk / hi \delta_j^i
    geometry::Coords<GEOM> coords(pco, k, j, i);
    geometry::Coords<GEOM> coords_xm(pco, k, j, i - 1);
    Real hx[3] = {Null<Real>()};
    Real hx_xm[3] = {Null<Real>()};
    coords.GetScaleFactors(hx);
    coords_xm.GetScaleFactors(hx_xm);

    Real xf[3] = {Null<Real>()};
    coords.FaceCenX1(geometry::CellFace::lower, xf);
    const Real hx1f = coords.hx1(xf[0], xf[1], xf[2]);
    const Real mus = avg * FaceAverage<DiffAvg::arithmetic>(mu(i), mu(i - 1)) +
                     havg * FaceAverage<DiffAvg::harmonic>(mu(i), mu(i - 1));

    const int imx1 = VI(n, 0);
    const int imx2 = VI(n, 1);
    const int imx3 = VI(n, 2);
    const int ien = nspecies * 3 + n;

    const Real f1 =
        hx1f * mus * (flx(0, i) - 1. / 3 * (1. - dp.eta) * (divu(i) + divu(i - 1)));
    const Real f2 = hx1f * mus * flx(1, i);
    const Real f3 = hx1f * mus * flx(2, i);

    qf(b, TE::F1, imx1, k, j, i) += f1;
    qf(b, TE::F1, imx2, k, j, i) += f2;
    qf(b, TE::F1, imx3, k, j, i) += f3;
    // div(v.T) = 1/sqrt(g) d_i( sqrt(g) T_j^i v^j  )
    qf(b, TE::F1, ien, k, j, i) +=
        0.5 *
            (p(b, gas::prim::velocity(imx1), k, j, i) / hx[0] +
             p(b, gas::prim::velocity(imx1), k, j, i - 1) / hx_xm[0]) *
            f1 +
        0.5 *
            (p(b, gas::prim::velocity(imx2), k, j, i) / hx[1] +
             p(b, gas::prim::velocity(imx2), k, j, i - 1) / hx_xm[1]) *
            f2 +
        0.5 *
            (p(b, gas::prim::velocity(imx3), k, j, i) / hx[2] +
             p(b, gas::prim::velocity(imx3), k, j, i - 1) / hx_xm[2]) *
            f3;
  });
}

template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim,
          typename SparsePackFlux>
KOKKOS_INLINE_FUNCTION void StressTensorFaceX2(
    DiffCoeffParams dp, parthenon::team_mbr_t const &member, const int b, const int n,
    const int k, const int j, const int il, const int iu, const int multid,
    const int threed, const int nspecies, const SparsePackPrim &p,
    const SparsePackFlux &qf, const parthenon::ScratchPad1D<Real> &divu_jm1,
    const parthenon::ScratchPad1D<Real> &divu,
    const parthenon::ScratchPad1D<Real> &mu_jm1, const parthenon::ScratchPad1D<Real> &mu,
    const parthenon::ScratchPad2D<Real> &flx) {
  // Fill the flx array with the stress tensor on the specified face

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Only gas fluids have momentum diffusion");
  auto pco = p.GetCoordinates(b);
  const bool avg = dp.avg == DiffAvg::arithmetic;
  const bool havg = dp.avg == DiffAvg::harmonic;
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    geometry::Coords<GEOM> coords(p.GetCoordinates(b), k, j, i);
    geometry::Coords<GEOM> coords_ym(p.GetCoordinates(b), k, j - 1, i);
    Real hx[3] = {Null<Real>()};
    Real hx_ym[3] = {Null<Real>()};
    coords.GetScaleFactors(hx);
    coords_ym.GetScaleFactors(hx_ym);

    Real xf[3] = {Null<Real>()};
    coords.FaceCenX2(geometry::CellFace::lower, xf);
    const Real hx2f = coords.hx2(xf[0], xf[1], xf[2]);

    const Real mus = avg * FaceAverage<DiffAvg::arithmetic>(mu(i), mu_jm1(i)) +
                     havg * FaceAverage<DiffAvg::harmonic>(mu(i), mu_jm1(i));

    const Real f1 = hx2f * mus * flx(0, i);
    const Real f2 =
        hx2f * mus * (flx(1, i) - 1. / 3 * (1. - dp.eta) * (divu_jm1(i) + divu(i)));
    const Real f3 = hx2f * mus * flx(2, i);

    const int imx1 = VI(n, 0);
    const int imx2 = VI(n, 1);
    const int imx3 = VI(n, 2);
    const int ien = nspecies * 3 + n;

    qf(b, TE::F2, imx1, k, j, i) += f1;
    qf(b, TE::F2, imx2, k, j, i) += f2;
    qf(b, TE::F2, imx3, k, j, i) += f3;
    // v.T
    qf(b, TE::F2, ien, k, j, i) +=
        0.5 *
            (p(b, gas::prim::velocity(imx1), k, j, i) / hx[0] +
             p(b, gas::prim::velocity(imx1), k, j - 1, i) / hx_ym[0]) *
            f1 +
        0.5 *
            (p(b, gas::prim::velocity(imx2), k, j, i) / hx[1] +
             p(b, gas::prim::velocity(imx2), k, j - 1, i) / hx_ym[1]) *
            f2 +
        0.5 *
            (p(b, gas::prim::velocity(imx3), k, j, i) / hx[2] +
             p(b, gas::prim::velocity(imx3), k, j - 1, i) / hx_ym[2]) *
            f3;
  });
}

template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim,
          typename SparsePackFlux>
KOKKOS_INLINE_FUNCTION void StressTensorFaceX3(
    DiffCoeffParams dp, parthenon::team_mbr_t const &member, const int b, const int n,
    const int k, const int j, const int il, const int iu, const int multid,
    const int threed, const int nspecies, const SparsePackPrim &p,
    const SparsePackFlux &qf, const parthenon::ScratchPad1D<Real> &divu_km1,
    const parthenon::ScratchPad1D<Real> &divu,
    const parthenon::ScratchPad1D<Real> &mu_km1, const parthenon::ScratchPad1D<Real> &mu,
    const parthenon::ScratchPad2D<Real> &flx) {
  // Fill the flx array with the stress tensor on the specified face

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Only gas fluids have momentum diffusion");
  auto pco = p.GetCoordinates(b);
  const bool avg = dp.avg == DiffAvg::arithmetic;
  const bool havg = dp.avg == DiffAvg::harmonic;
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    geometry::Coords<GEOM> coords(p.GetCoordinates(b), k, j, i);
    geometry::Coords<GEOM> coords_zm(p.GetCoordinates(b), k - 1, j, i);
    Real hx[3] = {Null<Real>()};
    Real hx_zm[3] = {Null<Real>()};
    coords.GetScaleFactors(hx);
    coords_zm.GetScaleFactors(hx_zm);

    Real xf[3] = {Null<Real>()};
    coords.FaceCenX3(geometry::CellFace::lower, xf);
    const Real hx3f = coords.hx3(xf[0], xf[1], xf[2]);

    const Real mus = avg * FaceAverage<DiffAvg::arithmetic>(mu(i), mu_km1(i)) +
                     havg * FaceAverage<DiffAvg::harmonic>(mu(i), mu_km1(i));

    const Real f1 = hx3f * mus * flx(0, i);
    const Real f2 = hx3f * mus * flx(1, i);
    const Real f3 =
        hx3f * mus * (flx(2, i) - 1. / 3 * (1. - dp.eta) * (divu_km1(i) + divu(i)));

    const int imx1 = VI(n, 0);
    const int imx2 = VI(n, 1);
    const int imx3 = VI(n, 2);
    const int ien = nspecies * 3 + n;

    qf(b, TE::F3, imx1, k, j, i) += f1;
    qf(b, TE::F3, imx2, k, j, i) += f2;
    qf(b, TE::F3, imx3, k, j, i) += f3;
    // v.T
    qf(b, TE::F3, ien, k, j, i) +=
        0.5 *
            (p(b, gas::prim::velocity(imx1), k, j, i) / hx[0] +
             p(b, gas::prim::velocity(imx1), k - 1, j, i) / hx_zm[0]) *
            f1 +
        0.5 *
            (p(b, gas::prim::velocity(imx2), k, j, i) / hx[1] +
             p(b, gas::prim::velocity(imx2), k - 1, j, i) / hx_zm[1]) *
            f2 +
        0.5 *
            (p(b, gas::prim::velocity(imx3), k, j, i) / hx[2] +
             p(b, gas::prim::velocity(imx3), k - 1, j, i) / hx_zm[2]) *
            f3;
  });
}

template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim>
KOKKOS_INLINE_FUNCTION void
VelocityDivergence(parthenon::team_mbr_t const &member, const int b, const int n,
                   const int k, const int j, const int il, const int iu, const int multid,
                   const int threed, const SparsePackPrim &q,
                   const parthenon::ScratchPad1D<Real> &divu) {
  // Fill the flx array with the stress tensor on the specified face

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Only gas fluids have momentum diffusion");
  auto pco = q.GetCoordinates(b);
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    geometry::Coords<GEOM> coords(q.GetCoordinates(b), k, j, i);
    Real area_x1[2] = {Null<Real>()};
    Real area_x2[2] = {0.0};
    Real area_x3[2] = {0.0};
    const Real vol = coords.Volume();
    coords.GetFaceAreaX1(area_x1);
    coords.GetFaceAreaX2(area_x2);
    coords.GetFaceAreaX3(area_x3);
    const Real divv = area_x1[1] * (q(b, gas::prim::velocity(3 * n + 0), k, j, i) +
                                    q(b, gas::prim::velocity(3 * n + 0), k, j, i + 1)) -
                      area_x1[0] * (q(b, gas::prim::velocity(3 * n + 0), k, j, i) +
                                    q(b, gas::prim::velocity(3 * n + 0), k, j, i - 1)) +
                      multid * area_x2[1] *
                          (q(b, gas::prim::velocity(3 * n + 1), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 1), k, j + multid, i)) -
                      multid * area_x2[0] *
                          (q(b, gas::prim::velocity(3 * n + 1), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 1), k, j - multid, i)) +
                      threed * area_x3[1] *
                          (q(b, gas::prim::velocity(3 * n + 2), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 2), k + threed, j, i)) -
                      threed * area_x3[0] *
                          (q(b, gas::prim::velocity(3 * n + 2), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 2), k - threed, j, i));
    divu(i) = divv / (2.0 * vol);
  });
}

template <Coordinates GEOM, Fluid FLUID_TYPE, DiffType DIFF, typename PKG,
          typename SparsePackPrim, typename SparsePackFlux>
TaskStatus MomentumFluxImpl(MeshData<Real> *md, DiffCoeffParams dp, PKG &pkg,
                            SparsePackPrim vprim, SparsePackFlux vf) {

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Momentum diffusion only works with a gas fluid");

  auto pm = md->GetParentPointer();
  auto eos_d = pkg->template Param<EOS>("eos_d");

  const int scr_level = pkg->template Param<int>("scr_level");

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  const int ncells1 = (ib.e - ib.s + 1) + 2 * parthenon::Globals::nghost;

  const int multi_d = (pm->ndim >= 2);
  const int three_d = (pm->ndim == 3);

  int il = ib.s, iu = ib.e + 1;
  int jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  int scr_size = ScratchPad2D<Real>::shmem_size(3, ncells1) +
                 ScratchPad1D<Real>::shmem_size(ncells1) +
                 ScratchPad1D<Real>::shmem_size(ncells1);

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scr_size,
      scr_level, 0, md->NumBlocks() - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j) {
        ScratchPad2D<Real> flx(mbr.team_scratch(scr_level), 3, ncells1);
        ScratchPad1D<Real> mu(mbr.team_scratch(scr_level), ncells1);
        ScratchPad1D<Real> divu(mbr.team_scratch(scr_level), ncells1);
        const int nspecies = vprim.GetSize(b, gas::prim::density());
        for (int n = 0; n < nspecies; n++) {

          // 1. Compute the strain tensor at i-1/2
          StrainTensorFace<GEOM, FLUID_TYPE, X1DIR>(mbr, b, n, k, j, il, iu, multi_d,
                                                    three_d, vprim, flx);

          // 2. Compute div(u) on this pencil
          VelocityDivergence<GEOM, FLUID_TYPE>(mbr, b, n, k, j, il - 1, iu, multi_d,
                                               three_d, vprim, divu);
          // 3. Viscosity values. No barrier
          DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
          diffcoeff.evaluate(dp, mbr, b, n, k, j, il - 1, iu, vprim, eos_d, mu);

          mbr.team_barrier();

          // 4. Fill the stress tensor from the strain tensor and viscosity
          StressTensorFaceX1<GEOM, FLUID_TYPE>(dp, mbr, b, n, k, j, il, iu, multi_d,
                                               three_d, nspecies, vprim, vf, divu, mu,
                                               flx);
        }
      });

  // X2-Flux
  if (multi_d) {
    jl = jb.s - 1, ju = jb.e + 1;
    il = ib.s, iu = ib.e, kl = kb.s, ku = kb.e;
    scr_size = ScratchPad2D<Real>::shmem_size(3, ncells1) +
               ScratchPad1D<Real>::shmem_size(ncells1) * 2 +
               ScratchPad1D<Real>::shmem_size(ncells1) * 2;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scr_size,
        scr_level, 0, md->NumBlocks() - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k) {
          ScratchPad2D<Real> flx(mbr.team_scratch(scr_level), 3, ncells1);
          ScratchPad1D<Real> scr1(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr2(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr3(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr4(mbr.team_scratch(scr_level), ncells1);

          const int nspecies = vprim.GetSize(b, gas::prim::density());
          for (int n = 0; n < nspecies; n++) {
            for (int j = jl; j <= ju; ++j) {
              // permute scratch
              auto mu = scr1;
              auto mu_jm1 = scr2;
              auto divu = scr3;
              auto divu_jm1 = scr4;
              if ((j % 2) == 0) {
                mu = scr2;
                mu_jm1 = scr1;
                divu = scr4;
                divu_jm1 = scr3;
              }
              // 1. Compute the momentum fluxes at j+1/2
              StrainTensorFace<GEOM, FLUID_TYPE, X2DIR>(mbr, b, n, k, j, il, iu, multi_d,
                                                        three_d, vprim, flx);
              // 2. Compute div(u) on this pencil
              VelocityDivergence<GEOM, FLUID_TYPE>(mbr, b, n, k, j, il, iu, multi_d,
                                                   three_d, vprim, divu_jm1);

              // 2. Viscosity values. No barrier
              DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
              diffcoeff.evaluate(dp, mbr, b, n, k, j, il, iu, vprim, eos_d, mu_jm1);

              mbr.team_barrier();
              if (j > jl) {
                StressTensorFaceX2<GEOM, FLUID_TYPE>(dp, mbr, b, n, k, j, il, iu, multi_d,
                                                     three_d, nspecies, vprim, vf,
                                                     divu_jm1, divu, mu_jm1, mu, flx);
              }
            }
          }
        });
  }

  // X3-Flux
  if (three_d) {
    kl = kb.s - 1, ku = kb.e + 1;
    il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e;
    scr_size = ScratchPad2D<Real>::shmem_size(3, ncells1) +
               ScratchPad1D<Real>::shmem_size(ncells1) * 2 +
               ScratchPad1D<Real>::shmem_size(ncells1) * 2;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scr_size,
        scr_level, 0, md->NumBlocks() - 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int j) {
          ScratchPad2D<Real> flx(mbr.team_scratch(scr_level), 3, ncells1);
          ScratchPad1D<Real> scr1(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr2(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr3(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr4(mbr.team_scratch(scr_level), ncells1);

          const int nspecies = vprim.GetSize(b, gas::prim::density());
          for (int n = 0; n < nspecies; n++) {
            for (int k = kl; k <= ku; ++k) {
              // permute scratch
              auto mu = scr1;
              auto mu_km1 = scr2;
              auto divu = scr3;
              auto divu_km1 = scr4;
              if ((k % 2) == 0) {
                mu = scr2;
                mu_km1 = scr1;
                divu = scr4;
                divu_km1 = scr3;
              }
              // 1. Compute the momentum fluxes at k-1/2
              StrainTensorFace<GEOM, FLUID_TYPE, X3DIR>(mbr, b, n, k, j, il, iu, multi_d,
                                                        three_d, vprim, flx);
              // 2. Compute div(u) on this pencil
              VelocityDivergence<GEOM, FLUID_TYPE>(mbr, b, n, k, j, il, iu, multi_d,
                                                   three_d, vprim, divu_km1);

              // 2. Viscosity values. No barrier
              DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
              diffcoeff.evaluate(dp, mbr, b, n, k, j, il, iu, vprim, eos_d, mu_km1);

              mbr.team_barrier();
              if (k > kl) {
                StressTensorFaceX3<GEOM, FLUID_TYPE>(dp, mbr, b, n, k, j, il, iu, multi_d,
                                                     three_d, nspecies, vprim, vf,
                                                     divu_km1, divu, mu_km1, mu, flx);
              }
            }
          }
        });
  }
  return TaskStatus::complete;
}

} // namespace Diffusion

#endif // UTILS_DIFFUSION_MOMENTUM_DIFFUSION_HPP_
