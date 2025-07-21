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
#ifndef UTILS_DIFFUSION_MOMENTUM_DIFFUSION_HPP_
#define UTILS_DIFFUSION_MOMENTUM_DIFFUSION_HPP_

// Artemis includes
#include "artemis.hpp"
#include "diffusion_coeff.hpp"
#include "geometry/geometry.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace Diffusion {

//----------------------------------------------------------------------------------------
//! \fn void StrainTensorFace
//! \brief Computes strain rate tensor
template <Coordinates GEOM, Fluid FLUID_TYPE, parthenon::CoordinateDirection XDIR,
          typename SparsePack>
KOKKOS_INLINE_FUNCTION void
StrainTensorFace(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
                 const int b, const int n, const int k, const int j, const int il,
                 const int iu, const int multi_d, const int three_d, const Real qshear,
                 const Real om0, const SparsePack &vprim,
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
    geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
    //  T_j^i = dv^i/dxj + hj^2/hi^2 dv^j/dxi  + v^k dhi/dxk / hi \delta_j^i
    const auto &xv = coords.GetCellCenter();
    const auto &hx = coords.GetScaleFactors();

    const std::array<Real, 3> v{vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i) / hx[0],
                                vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i) / hx[1],
                                vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i) / hx[2]};
    auto xf = NewArray<Real, 3>();
    if constexpr (XDIR == X1DIR) {
      xf = coords.FaceCenX1(geometry::CellFace::lower);
    } else if constexpr (XDIR == X2DIR) {
      xf = coords.FaceCenX2(geometry::CellFace::lower);
    } else if constexpr (XDIR == X3DIR) {
      xf = coords.FaceCenX3(geometry::CellFace::lower);
    }
    const std::array<Real, 3> hxf{coords.hx1(xf[0], xf[1], xf[2]),
                                  coords.hx2(xf[0], xf[1], xf[2]),
                                  coords.hx3(xf[0], xf[1], xf[2])};

    if constexpr (XDIR == X1DIR) {
      // T_*^1  flx = { T_1^1 , T_2^1 , T_3^1 }

      // need xm
      //      ym , yp, xmym, xmyp
      //      zm , zp, xmzm, xmzp

      geometry::Coords<GEOM> coords_xm(cpars, pco, k, j, i - 1);
      geometry::Coords<GEOM> coords_xmym(cpars, pco, k, j - multi_d, i - 1);
      geometry::Coords<GEOM> coords_xmyp(cpars, pco, k, j + multi_d, i - 1);
      geometry::Coords<GEOM> coords_ym(cpars, pco, k, j - multi_d, i);
      geometry::Coords<GEOM> coords_yp(cpars, pco, k, j + multi_d, i);

      geometry::Coords<GEOM> coords_xmzm(cpars, pco, k - three_d, j, i - 1);
      geometry::Coords<GEOM> coords_xmzp(cpars, pco, k + three_d, j, i - 1);
      geometry::Coords<GEOM> coords_zm(cpars, pco, k - three_d, j, i);
      geometry::Coords<GEOM> coords_zp(cpars, pco, k + three_d, j, i);

      const auto &xv_xm = coords_xm.GetCellCenter();

      const auto &xv_ym = coords_ym.GetCellCenter();
      const auto &xv_yp = coords_yp.GetCellCenter();
      const auto &xv_xmym = coords_xmym.GetCellCenter();
      const auto &xv_xmyp = coords_xmyp.GetCellCenter();

      const auto &xv_zm = coords_zm.GetCellCenter();
      const auto &xv_zp = coords_zp.GetCellCenter();
      const auto &xv_xmzm = coords_xmzm.GetCellCenter();
      const auto &xv_xmzp = coords_xmzp.GetCellCenter();

      const auto &hx_xm = coords_xm.GetScaleFactors();

      const auto &hx_ym = coords_ym.GetScaleFactors();
      const auto &hx_yp = coords_yp.GetScaleFactors();
      const auto &hx_xmym = coords_xmym.GetScaleFactors();
      const auto &hx_xmyp = coords_xmyp.GetScaleFactors();

      const auto &hx_zm = coords_zm.GetScaleFactors();
      const auto &hx_zp = coords_zp.GetScaleFactors();
      const auto &hx_xmzm = coords_xmzm.GetScaleFactors();
      const auto &hx_xmzp = coords_xmzp.GetScaleFactors();

      const Real dx1 = xv[0] - xv_xm[0];
      const Real dx2 = multi_d ? (xv_yp[1] - xv_ym[1]) : Fuzz<Real>();
      const Real dx2_xm = multi_d ? (xv_xmyp[1] - xv_xmym[1]) : Fuzz<Real>();
      const Real dx3 = three_d ? (xv_zp[2] - xv_zm[2]) : Fuzz<Real>();
      const Real dx3_xm = three_d ? (xv_xmzp[2] - xv_xmzm[2]) : Fuzz<Real>();

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
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j + multi_d, i) / hx_yp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j - multi_d, i) / hx_ym[0];

      const Real dv12_xm =
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j + multi_d, i - 1) / hx_xmyp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k, j - multi_d, i - 1) / hx_xmym[0];

      flx(1, i) = multi_d * 0.5 * (dv12 / dx2 + dv12_xm / dx2_xm) +
                  SQR(hxf[1] / hxf[0]) * dv2 / dx1;

      // T_3^1  = dv^1/dx3 +  h3^2/h1^2 dv^3/dx1
      const Real dv3 =
          v[2] - vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i - 1) / hx_xm[2];

      const Real dv13 =
          vprim(b, gas::prim::velocity(VI(n, 0)), k + three_d, j, i) / hx_zp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k - three_d, j, i) / hx_zm[0];

      const Real dv13_xm =
          vprim(b, gas::prim::velocity(VI(n, 0)), k + three_d, j, i - 1) / hx_xmzp[0] -
          vprim(b, gas::prim::velocity(VI(n, 0)), k - three_d, j, i - 1) / hx_xmzm[0];

      flx(2, i) = three_d * 0.5 * (dv13 / dx3 + dv13_xm / dx3_xm) +
                  SQR(hxf[2] / hxf[0]) * dv3 / dx1;

    } else if constexpr (XDIR == X2DIR) {
      // T_*^2  flx = { T_1^2 , T_2^2 , T_3^2 }

      // need ym
      //      xm , xp, xpym, xmym
      //      zm , zp, ymzm, ymzp

      geometry::Coords<GEOM> coords_xp(cpars, pco, k, j, i + 1);
      geometry::Coords<GEOM> coords_xm(cpars, pco, k, j, i - 1);
      geometry::Coords<GEOM> coords_xpym(cpars, pco, k, j - 1, i + 1);
      geometry::Coords<GEOM> coords_xmym(cpars, pco, k, j - 1, i - 1);

      geometry::Coords<GEOM> coords_ym(cpars, pco, k, j - 1, i);

      geometry::Coords<GEOM> coords_zp(cpars, pco, k + three_d, j, i);
      geometry::Coords<GEOM> coords_zm(cpars, pco, k - three_d, j, i);
      geometry::Coords<GEOM> coords_ymzp(cpars, pco, k + three_d, j - 1, i);
      geometry::Coords<GEOM> coords_ymzm(cpars, pco, k - three_d, j - 1, i);

      const auto &xv_xm = coords_xm.GetCellCenter();
      const auto &xv_xp = coords_xp.GetCellCenter();
      const auto &xv_xmym = coords_xmym.GetCellCenter();
      const auto &xv_xpym = coords_xpym.GetCellCenter();

      const auto &xv_ym = coords_ym.GetCellCenter();

      const auto &xv_zm = coords_zm.GetCellCenter();
      const auto &xv_zp = coords_zp.GetCellCenter();
      const auto &xv_ymzm = coords_ymzm.GetCellCenter();
      const auto &xv_ymzp = coords_ymzp.GetCellCenter();

      const auto &hx_xm = coords_xm.GetScaleFactors();
      const auto &hx_xp = coords_xp.GetScaleFactors();
      const auto &hx_xmym = coords_xmym.GetScaleFactors();
      const auto &hx_xpym = coords_xpym.GetScaleFactors();

      const auto &hx_ym = coords_ym.GetScaleFactors();

      const auto &hx_zm = coords_zm.GetScaleFactors();
      const auto &hx_zp = coords_zp.GetScaleFactors();
      const auto &hx_ymzm = coords_ymzm.GetScaleFactors();
      const auto &hx_ymzp = coords_ymzp.GetScaleFactors();

      const Real dx1 = xv_xp[0] - xv_xm[0];
      const Real dx1_ym = xv_xpym[0] - xv_xmym[0];
      const Real dx2 = xv[1] - xv_ym[1];
      const Real dx3 = three_d ? (xv_zp[2] - xv_zm[2]) : Fuzz<Real>();
      const Real dx3_ym = three_d ? (xv_ymzp[2] - xv_ymzm[2]) : Fuzz<Real>();

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
          vprim(b, gas::prim::velocity(VI(n, 1)), k + three_d, j, i) / hx_zp[1] -
          vprim(b, gas::prim::velocity(VI(n, 1)), k - three_d, j, i) / hx_zm[1];

      const Real dv23_ym =
          vprim(b, gas::prim::velocity(VI(n, 1)), k + three_d, j - 1, i) / hx_ymzp[1] -
          vprim(b, gas::prim::velocity(VI(n, 1)), k - three_d, j - 1, i) / hx_ymzm[1];

      flx(2, i) = three_d * 0.5 * (dv23 / dx3 + dv23_ym / dx3_ym) +
                  SQR(hxf[2] / hxf[1]) * dv3 / dx2;

    } else if constexpr (XDIR == X3DIR) {
      // T_*^3  flx = { T_1^3 , T_2^3 , T_3^3 }

      geometry::Coords<GEOM> coords_xp(cpars, pco, k, j, i + 1);
      geometry::Coords<GEOM> coords_xm(cpars, pco, k, j, i - 1);
      geometry::Coords<GEOM> coords_xpzm(cpars, pco, k - 1, j, i + 1);
      geometry::Coords<GEOM> coords_xmzm(cpars, pco, k - 1, j, i - 1);

      geometry::Coords<GEOM> coords_yp(cpars, pco, k, j + 1, i);
      geometry::Coords<GEOM> coords_ym(cpars, pco, k, j - 1, i);
      geometry::Coords<GEOM> coords_ypzm(cpars, pco, k - 1, j + 1, i);
      geometry::Coords<GEOM> coords_ymzm(cpars, pco, k - 1, j - 1, i);

      geometry::Coords<GEOM> coords_zm(cpars, pco, k - 1, j, i);

      const auto &xv_xm = coords_xm.GetCellCenter();
      const auto &xv_xp = coords_xp.GetCellCenter();
      const auto &xv_xmzm = coords_xmzm.GetCellCenter();
      const auto &xv_xpzm = coords_xpzm.GetCellCenter();

      const auto &xv_ym = coords_ym.GetCellCenter();
      const auto &xv_yp = coords_yp.GetCellCenter();
      const auto &xv_ymzm = coords_ymzm.GetCellCenter();
      const auto &xv_ypzm = coords_ypzm.GetCellCenter();

      const auto &xv_zm = coords_zm.GetCellCenter();

      const auto &hx_xm = coords_xm.GetScaleFactors();
      const auto &hx_xp = coords_xp.GetScaleFactors();
      const auto &hx_xmzm = coords_xmzm.GetScaleFactors();
      const auto &hx_xpzm = coords_xpzm.GetScaleFactors();

      const auto &hx_ym = coords_ym.GetScaleFactors();
      const auto &hx_yp = coords_yp.GetScaleFactors();
      const auto &hx_ymzm = coords_ymzm.GetScaleFactors();
      const auto &hx_ypzm = coords_ypzm.GetScaleFactors();

      const auto &hx_zm = coords_zm.GetScaleFactors();

      const Real dx1 = xv_xp[0] - xv_xm[0];
      const Real dx1_zm = xv_xpzm[0] - xv_xmzm[0];
      const Real dx2 = xv_yp[1] - xv_ym[1];
      const Real dx2_zm = xv_ypzm[1] - xv_ymzm[1];
      const Real dx3 = xv[2] - xv_zm[2];

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

    // Add any strain rate due to the background shear velocity.
    // Uses the analytic expression at the face center
    const auto Eb = RotatingFrame::StrainRate<GEOM, XDIR>(qshear, om0, xf);
    flx(0, i) += Eb[0];
    flx(1, i) += Eb[1];
    flx(2, i) += Eb[2];
  });
}

//----------------------------------------------------------------------------------------
//! \fn void StressTensorFaceX1
//! \brief Stress Ttensor X1-Face
template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim,
          typename SparsePackFlux>
KOKKOS_INLINE_FUNCTION void StressTensorFaceX1(
    DiffCoeffParams dp, parthenon::team_mbr_t const &member,
    const geometry::CoordParams &cpars, const int b, const int n, const int k,
    const int j, const int il, const int iu, const int multi_d, const int three_d,
    const int nspecies, const SparsePackPrim &p, const SparsePackFlux &qf,
    const parthenon::ScratchPad1D<Real> &divu, const parthenon::ScratchPad1D<Real> &mu,
    const parthenon::ScratchPad2D<Real> &flx) {
  // Fill the flx array with the stress tensor on the specified face

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Only gas fluids have momentum diffusion");
  auto pco = p.GetCoordinates(b);
  const bool avg = (dp.avg == DiffAvg::arithmetic);
  const bool havg = (dp.avg == DiffAvg::harmonic);
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    //  T_j^i = dv^i/dxj + hj^2/hi^2 dv^j/dxi  + v^k dhi/dxk / hi \delta_j^i
    geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
    geometry::Coords<GEOM> coords_xm(cpars, pco, k, j, i - 1);
    const auto &hx = coords.GetScaleFactors();
    const auto &hx_xm = coords_xm.GetScaleFactors();

    const auto &xf = coords.FaceCenX1(geometry::CellFace::lower);
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

//----------------------------------------------------------------------------------------
//! \fn void StressTensorFaceX2
//! \brief Stress Ttensor X2-Face
template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim,
          typename SparsePackFlux>
KOKKOS_INLINE_FUNCTION void StressTensorFaceX2(
    DiffCoeffParams dp, parthenon::team_mbr_t const &member,
    const geometry::CoordParams &cpars, const int b, const int n, const int k,
    const int j, const int il, const int iu, const int multi_d, const int three_d,
    const int nspecies, const SparsePackPrim &p, const SparsePackFlux &qf,
    const parthenon::ScratchPad1D<Real> &divu_jm1,
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
    geometry::Coords<GEOM> coords(cpars, p.GetCoordinates(b), k, j, i);
    geometry::Coords<GEOM> coords_ym(cpars, p.GetCoordinates(b), k, j - 1, i);
    const auto &hx = coords.GetScaleFactors();
    const auto &hx_ym = coords_ym.GetScaleFactors();

    const auto &xf = coords.FaceCenX2(geometry::CellFace::lower);
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

//----------------------------------------------------------------------------------------
//! \fn void StressTensorFaceX3
//! \brief Stress Ttensor X3-Face
template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim,
          typename SparsePackFlux>
KOKKOS_INLINE_FUNCTION void StressTensorFaceX3(
    DiffCoeffParams dp, parthenon::team_mbr_t const &member,
    const geometry::CoordParams &cpars, const int b, const int n, const int k,
    const int j, const int il, const int iu, const int multi_d, const int three_d,
    const int nspecies, const SparsePackPrim &p, const SparsePackFlux &qf,
    const parthenon::ScratchPad1D<Real> &divu_km1,
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
    geometry::Coords<GEOM> coords(cpars, p.GetCoordinates(b), k, j, i);
    geometry::Coords<GEOM> coords_zm(cpars, p.GetCoordinates(b), k - 1, j, i);
    const auto &hx = coords.GetScaleFactors();
    const auto &hx_zm = coords_zm.GetScaleFactors();

    const auto &xf = coords.FaceCenX3(geometry::CellFace::lower);
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

//----------------------------------------------------------------------------------------
//! \fn void VelocityDivergence
//! \brief Computes velocity divergence
template <Coordinates GEOM, Fluid FLUID_TYPE, typename SparsePackPrim>
KOKKOS_INLINE_FUNCTION void
VelocityDivergence(parthenon::team_mbr_t const &member,
                   const geometry::CoordParams &cpars, const int b, const int n,
                   const int k, const int j, const int il, const int iu,
                   const int multi_d, const int three_d, const SparsePackPrim &q,
                   const parthenon::ScratchPad1D<Real> &divu) {
  // Fill the flx array with the stress tensor on the specified face

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Only gas fluids have momentum diffusion");
  auto pco = q.GetCoordinates(b);
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    geometry::Coords<GEOM> coords(cpars, q.GetCoordinates(b), k, j, i);

    const Real vol = coords.Volume();
    const auto area_x1 = coords.GetFaceAreaX1();
    const auto area_x2 = (multi_d) ? coords.GetFaceAreaX2() : NewArray<Real, 2>(0.0);
    const auto area_x3 = (three_d) ? coords.GetFaceAreaX3() : NewArray<Real, 2>(0.0);

    const Real divv = area_x1[1] * (q(b, gas::prim::velocity(3 * n + 0), k, j, i) +
                                    q(b, gas::prim::velocity(3 * n + 0), k, j, i + 1)) -
                      area_x1[0] * (q(b, gas::prim::velocity(3 * n + 0), k, j, i) +
                                    q(b, gas::prim::velocity(3 * n + 0), k, j, i - 1)) +
                      multi_d * area_x2[1] *
                          (q(b, gas::prim::velocity(3 * n + 1), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 1), k, j + multi_d, i)) -
                      multi_d * area_x2[0] *
                          (q(b, gas::prim::velocity(3 * n + 1), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 1), k, j - multi_d, i)) +
                      three_d * area_x3[1] *
                          (q(b, gas::prim::velocity(3 * n + 2), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 2), k + three_d, j, i)) -
                      three_d * area_x3[0] *
                          (q(b, gas::prim::velocity(3 * n + 2), k, j, i) +
                           q(b, gas::prim::velocity(3 * n + 2), k - three_d, j, i));
    divu(i) = divv / (2.0 * vol);
  });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MomentumFluxImpl
//! \brief Implementation for momentum flux calculation
template <Coordinates GEOM, Fluid FLUID_TYPE, DiffType DIFF, typename PKG,
          typename SparsePackPrim, typename SparsePackFlux>
TaskStatus MomentumFluxImpl(MeshData<Real> *md, DiffCoeffParams dp, PKG &pkg,
                            SparsePackPrim vprim, SparsePackFlux vf) {

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Momentum diffusion only works with a gas fluid");

  auto pm = md->GetParentPointer();
  auto eos_d = pkg->template Param<EOS>("eos_d");

  Real qshear = 0.0, om0 = 0.0;
  const bool do_shear = pm->packages.Get("artemis")->template Param<bool>("do_shear");
  if (do_shear) {
    auto &rframe_pkg = pm->packages.Get("rotating_frame");
    qshear = rframe_pkg->template Param<Real>("qshear");
    om0 = rframe_pkg->template Param<Real>("omega");
  }
  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

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
          StrainTensorFace<GEOM, FLUID_TYPE, X1DIR>(
              mbr, cpars, b, n, k, j, il, iu, multi_d, three_d, qshear, om0, vprim, flx);

          // 2. Compute div(u) on this pencil
          VelocityDivergence<GEOM, FLUID_TYPE>(mbr, cpars, b, n, k, j, il - 1, iu,
                                               multi_d, three_d, vprim, divu);
          // 3. Viscosity values. No barrier
          DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
          diffcoeff.evaluate(dp, mbr, b, n, k, j, il - 1, iu, vprim, eos_d, mu);

          mbr.team_barrier();

          // 4. Fill the stress tensor from the strain tensor and viscosity
          StressTensorFaceX1<GEOM, FLUID_TYPE>(dp, mbr, cpars, b, n, k, j, il, iu,
                                               multi_d, three_d, nspecies, vprim, vf,
                                               divu, mu, flx);
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
              StrainTensorFace<GEOM, FLUID_TYPE, X2DIR>(mbr, cpars, b, n, k, j, il, iu,
                                                        multi_d, three_d, qshear, om0,
                                                        vprim, flx);
              // 2. Compute div(u) on this pencil
              VelocityDivergence<GEOM, FLUID_TYPE>(mbr, cpars, b, n, k, j, il, iu,
                                                   multi_d, three_d, vprim, divu_jm1);

              // 2. Viscosity values. No barrier
              DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
              diffcoeff.evaluate(dp, mbr, b, n, k, j, il, iu, vprim, eos_d, mu_jm1);

              mbr.team_barrier();
              if (j > jl) {
                StressTensorFaceX2<GEOM, FLUID_TYPE>(dp, mbr, cpars, b, n, k, j, il, iu,
                                                     multi_d, three_d, nspecies, vprim,
                                                     vf, divu_jm1, divu, mu_jm1, mu, flx);
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
              StrainTensorFace<GEOM, FLUID_TYPE, X3DIR>(mbr, cpars, b, n, k, j, il, iu,
                                                        multi_d, three_d, qshear, om0,
                                                        vprim, flx);
              // 2. Compute div(u) on this pencil
              VelocityDivergence<GEOM, FLUID_TYPE>(mbr, cpars, b, n, k, j, il, iu,
                                                   multi_d, three_d, vprim, divu_km1);

              // 2. Viscosity values. No barrier
              DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
              diffcoeff.evaluate(dp, mbr, b, n, k, j, il, iu, vprim, eos_d, mu_km1);

              mbr.team_barrier();
              if (k > kl) {
                StressTensorFaceX3<GEOM, FLUID_TYPE>(dp, mbr, cpars, b, n, k, j, il, iu,
                                                     multi_d, three_d, nspecies, vprim,
                                                     vf, divu_km1, divu, mu_km1, mu, flx);
              }
            }
          }
        });
  }
  return TaskStatus::complete;
}

} // namespace Diffusion

#endif // UTILS_DIFFUSION_MOMENTUM_DIFFUSION_HPP_
