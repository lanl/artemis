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
#ifndef GEOMETRY_SPHERICAL_HPP_
#define GEOMETRY_SPHERICAL_HPP_

// Artemis includes
#include "geometry.hpp"

using namespace parthenon::package::prelude;

namespace geometry {
//----------------------------------------------------------------------------------------
//! The derived spherical specialization
//!
//!   Spherical coordinates are defined as
//!
//!          x1     x2     x3
//!     x = ( r,  theta,  phi)
//!     e_r     = sin(theta) cos(phi) e_x + sin(theta) sin(phi) e_y + cos(theta) e_z
//!     e_theta = cos(theta) cos(phi) e_x + cos(theta) sin(phi) e_y - sin(theta) e_z
//!     e_phi   = -sin(phi) e_x + cos(phi) e_y
template <>
class Coords<Coordinates::spherical> : public CoordsBase<Coords<Coordinates::spherical>> {
  // the derived  specialization

 public:
  KOKKOS_INLINE_FUNCTION
  Coords(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::spherical>>(pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::spherical>>() {}

  KOKKOS_INLINE_FUNCTION bool x1dep() { return true; }
  KOKKOS_INLINE_FUNCTION bool x2dep() { return true; }

  KOKKOS_INLINE_FUNCTION Real hx2(const Real x1, const Real x2, const Real x3) {
    return x1;
  }
  KOKKOS_INLINE_FUNCTION Real hx3(const Real x1, const Real x2, const Real x3) {
    return x1 * std::sin(x2);
  }

  KOKKOS_INLINE_FUNCTION Real x1v() {
    const Real dr2 = bnds.x1[0] * bnds.x1[0] + bnds.x1[1] * bnds.x1[1];
    return 0.75 * (bnds.x1[0] + bnds.x1[1]) * dr2 / (dr2 + bnds.x1[0] * bnds.x1[1]);
  }
  KOKKOS_INLINE_FUNCTION Real x2v() {
    // \int t * sin(t) *dt/ \int sin(t) * dt = d( sin(t) -
    // t*cos(t))/d(-cos(t))
    const Real ctm = std::cos(bnds.x2[0]);
    const Real ctp = std::cos(bnds.x2[1]);
    const Real dst = std::sin(bnds.x2[1]) - std::sin(bnds.x2[0]);
    return (dst - bnds.x2[1] * ctp + bnds.x2[0] * ctm) / std::abs(ctm - ctp);
  }

  KOKKOS_INLINE_FUNCTION Real hx2v() { return x1v(); }
  KOKKOS_INLINE_FUNCTION Real hx3v() {
    // \int r sin(t)  (r^2 sin(t) dr dt dp)
    // \int r^3 dr/ int r^2 dr  * \int sin^2 dt / \int sin dt
    // <r> <sin(t)>
    // \int sin^2(t) dt/\int sin(t) dt
    // \int sin^2(t) = 0.5( t - sin(t) cos(t)) + const
    const Real ctm = std::cos(bnds.x2[0]);
    const Real ctp = std::cos(bnds.x2[1]);
    const Real stm = std::sin(bnds.x2[0]);
    const Real stp = std::sin(bnds.x2[1]);

    const Real dsc = stp * ctp - stm * ctm;
    const Real dx2 = bnds.x2[1] - bnds.x2[0];

    return x1v() * 0.5 * (dx2 - dsc) / std::abs(ctm - ctp);
  }

  KOKKOS_INLINE_FUNCTION void FaceCenX2(const CellFace f, Real *xf) {
    // <r> = d(r^3/3) / d(r^2/2)
    xf[0] =
        2.0 / 3.0 *
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        (bnds.x1[0] + bnds.x1[1]);
    xf[1] = bnds.x2[static_cast<int>(f)];
    xf[2] = 0.5 * (bnds.x3[0] + bnds.x3[1]);
  }

  KOKKOS_INLINE_FUNCTION void FaceCenX3(const CellFace f, Real *xf) {
    // <r> = d(r^3/3) / d(r^2/2)
    xf[0] =
        2.0 / 3.0 *
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        (bnds.x1[0] + bnds.x1[1]);
    xf[1] = 0.5 * (bnds.x2[0] + bnds.x2[1]);
    xf[2] = bnds.x3[static_cast<int>(f)];
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) {
    // \int r^2 sin(t) dp*dt = d(-cos(t)) * r^2 *dp
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return x1f * x1f * std::abs(std::cos(bnds.x2[0]) - std::cos(bnds.x2[1])) * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX2(const Real x2f) {
    // \int r*sin(t)*dp*dr = d(r^2/2)*sin(t)*dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return 0.5 * (bnds.x1[1] + bnds.x1[0]) * std::sin(x2f) * dx1 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX3(const Real x3f) {
    // \int r*dt*dr = d(r^2/2)*dt
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    return 0.5 * (bnds.x1[0] + bnds.x1[1]) * dx1 * dx2;
  }

  KOKKOS_INLINE_FUNCTION Real Volume() {
    // \int r^2 sin(t) dr dt dp = d(r^3/3) d(-cos(t)) dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    const Real rfac =
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        3.0;
    const Real dx2 = std::abs(std::cos(bnds.x2[0]) - std::cos(bnds.x2[1]));
    return rfac * dx1 * dx2 * dx3;
  }

  KOKKOS_INLINE_FUNCTION Real dh2dx1() {
    return 3.0 / 2.0 * (bnds.x1[0] + bnds.x1[1]) /
           (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]);
  }
  KOKKOS_INLINE_FUNCTION Real dh3dx1() {
    return 3.0 / 2.0 * (bnds.x1[0] + bnds.x1[1]) /
           (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]);
  }
  KOKKOS_INLINE_FUNCTION Real dh3dx2() {
    return (std::sin(bnds.x2[1]) - std::sin(bnds.x2[0])) /
           std::abs(std::cos(bnds.x2[0]) - std::cos(bnds.x2[1]));
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCart(const Real xi[3], Real xo[3]) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    xo[0] = xi[0] * st * cp;
    xo[1] = xi[0] * st * sp;
    xo[2] = xi[0] * ct;
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToCart(const Real xi[3], Real ex1[3], Real ex2[3],
                                               Real ex3[3]) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    // clang-format off
    ex1[0] = st * cp;  ex1[1] = st * sp;  ex1[2] = ct;
    ex2[0] = ct * cp;  ex2[1] = ct * sp;  ex2[2] = -st;
    ex3[0] = -sp;      ex3[1] = cp;       ex3[2] = 0.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToSph(const Real xi[3], Real xo[3]) {
    xo[0] = xi[0];
    xo[1] = xi[1];
    xo[2] = xi[2];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToSph(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // clang-format off
    ex1[0] = 1.0;  ex1[1] = 0.0;  ex1[2] = 0.0;
    ex2[0] = 0.0;  ex2[1] = 1.0;  ex2[2] = 0.0;
    ex3[0] = 0.0;  ex3[1] = 0.0;  ex3[2] = 1.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCyl(const Real xi[3], Real xo[3]) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);

    xo[0] = xi[0] * st;
    xo[1] = xi[2];
    xo[2] = xi[0] * ct;
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToCyl(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    // clang-format off
    ex1[0] = st;   ex1[1] = 0.0;  ex1[2] = ct;
    ex2[0] = ct;   ex2[1] = 0.0;  ex2[2] = -st;
    ex3[0] = 0.0;  ex3[1] = 1.0;  ex3[2] = 0.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToAxi(const Real xi[3], Real xo[3]) {
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    xo[0] = xi[0] * st;
    xo[1] = xi[0] * ct;
    xo[2] = xi[2];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToAxi(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    // clang-format off
    ex1[0] = st;   ex1[1] = ct;   ex1[2] = 0.0;
    ex2[0] = ct;   ex2[1] = -st;  ex2[2] = 0.0;
    ex3[0] = 0.0;  ex3[1] = 0.0;  ex3[2] = 1.0;
    // clang-format on
  }
}; // Coords

} // namespace geometry

#endif // GEOMETRY_SPHERICAL_HPP_
