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
//!
//!  We have special handling for 1D and 2D that drops the angle dependence

template <>
class Coords<Coordinates::spherical3D>
    : public CoordsBase<Coords<Coordinates::spherical3D>> {
  // the derived  specialization

 public:
  KOKKOS_INLINE_FUNCTION
  Coords(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::spherical3D>>(pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::spherical3D>>() {}

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

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX2(const CellFace f) {
    // <r> = d(r^3/3) / d(r^2/2)
    return {2.0 / 3.0 *
                (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                 bnds.x1[1] * bnds.x1[1]) /
                (bnds.x1[0] + bnds.x1[1]),
            bnds.x2[static_cast<int>(f)], 0.5 * (bnds.x3[0] + bnds.x3[1])};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX3(const CellFace f) {
    // <r> = d(r^3/3) / d(r^2/2)
    return {2.0 / 3.0 *
                (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                 bnds.x1[1] * bnds.x1[1]) /
                (bnds.x1[0] + bnds.x1[1]),
            0.5 * (bnds.x2[0] + bnds.x2[1]), bnds.x3[static_cast<int>(f)]};
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

  KOKKOS_INLINE_FUNCTION Mat3x2 RFWeights() {
    // Set the flux averaging weights in the rotating frame for the angular momentum
    // \pm ( <R^2>_j^\pm - <R^2> ) where R is the cylindrical radius

    // volume average (r,theta)
    const Real rv = x1v();
    const Real stv = std::sin(x2v());

    // face averaged r on the X2 face
    const Real rf =
        2.0 / 3.0 *
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        (bnds.x1[0] + bnds.x1[1]);

    const Real r2cyl = SQR(rv * stv);

    std::array<Real, 2> bx1{r2cyl - SQR(bnds.x1[0] * stv), SQR(bnds.x1[1] * stv) - r2cyl};
    std::array<Real, 2> bx2{r2cyl - SQR(rf * std::sin(bnds.x2[0])),
                            SQR(rf * std::sin(bnds.x2[1])) - r2cyl};

    return {bx1, bx2, NewArray<Real, 2>(0.0)};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCart(const std::array<Real, 3> &xi) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    return {xi[0] * st * cp, xi[0] * st * sp, xi[0] * ct};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCart(const std::array<Real, 3> &xi) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    std::array<Real, 3> ex1{st * cp, st * sp, ct};
    std::array<Real, 3> ex2{ct * cp, ct * sp, -st};
    std::array<Real, 3> ex3{-sp, cp, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToSph(const std::array<Real, 3> &xi) {
    return xi;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToSph(const std::array<Real, 3> &xi) {
    std::array<Real, 3> ex1{1.0, 0.0, 0.0};
    std::array<Real, 3> ex2{0.0, 1.0, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCyl(const std::array<Real, 3> &xi) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);

    return {xi[0] * st, xi[2], xi[0] * ct};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCyl(const std::array<Real, 3> &xi) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    std::array<Real, 3> ex1{st, 0.0, ct};
    std::array<Real, 3> ex2{ct, 0.0, -st};
    std::array<Real, 3> ex3{0.0, 1.0, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToAxi(const std::array<Real, 3> &xi) {
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    return {xi[0] * st, xi[0] * ct, xi[2]};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToAxi(const std::array<Real, 3> &xi) {
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    std::array<Real, 3> ex1{st, ct, 0.0};
    std::array<Real, 3> ex2{ct, -st, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }
}; // Coords

template <>
class Coords<Coordinates::spherical2D>
    : public CoordsBase<Coords<Coordinates::spherical2D>> {
  // the derived  specialization for 2D spherical coordinates

 public:
  KOKKOS_INLINE_FUNCTION
  Coords(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::spherical2D>>(pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::spherical2D>>() {}

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

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX2(const CellFace f) {
    // <r> = d(r^3/3) / d(r^2/2)
    return {2.0 / 3.0 *
                (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                 bnds.x1[1] * bnds.x1[1]) /
                (bnds.x1[0] + bnds.x1[1]),
            bnds.x2[static_cast<int>(f)], 0.0};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX3(const CellFace f) {
    // <r> = d(r^3/3) / d(r^2/2)
    return {2.0 / 3.0 *
                (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                 bnds.x1[1] * bnds.x1[1]) /
                (bnds.x1[0] + bnds.x1[1]),
            0.5 * (bnds.x2[0] + bnds.x2[1]), 0.0};
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) {
    // \int r^2 sin(t) dp*dt = d(-cos(t)) * r^2 *dp
    return x1f * x1f * std::abs(std::cos(bnds.x2[0]) - std::cos(bnds.x2[1]));
  }
  KOKKOS_INLINE_FUNCTION Real AreaX2(const Real x2f) {
    // \int r*sin(t)*dp*dr = d(r^2/2)*sin(t)*dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    return 0.5 * (bnds.x1[1] + bnds.x1[0]) * std::sin(x2f) * dx1;
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
    const Real rfac =
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        3.0;
    const Real dx2 = std::abs(std::cos(bnds.x2[0]) - std::cos(bnds.x2[1]));
    return rfac * dx1 * dx2;
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

  KOKKOS_INLINE_FUNCTION Mat3x2 RFWeights() {
    // Set the flux averaging weights in the rotating frame for the angular momentum
    // \pm ( <R^2>_j^\pm - <R^2> ) where R is the cylindrical radius

    // volume average (r,theta)
    const Real rv = x1v();
    const Real stv = std::sin(x2v());

    // face averaged r on the X2 face
    const Real rf =
        2.0 / 3.0 *
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        (bnds.x1[0] + bnds.x1[1]);

    const Real r2cyl = SQR(rv * stv);

    std::array<Real, 2> bx1{r2cyl - SQR(bnds.x1[0] * stv), SQR(bnds.x1[1] * stv) - r2cyl};
    std::array<Real, 2> bx2{r2cyl - SQR(rf * std::sin(bnds.x2[0])),
                            SQR(rf * std::sin(bnds.x2[1])) - r2cyl};

    return {bx1, bx2, NewArray<Real, 2>(0.0)};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCart(const std::array<Real, 3> &xi) {
    const Real cp = 1.0;
    const Real sp = 0.0;
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    return {xi[0] * st * cp, xi[0] * st * sp, xi[0] * ct};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCart(const std::array<Real, 3> &xi) {
    const Real cp = 1.0;
    const Real sp = 0.0;
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    std::array<Real, 3> ex1{st * cp, st * sp, ct};
    std::array<Real, 3> ex2{ct * cp, ct * sp, -st};
    std::array<Real, 3> ex3{-sp, cp, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToSph(const std::array<Real, 3> &xi) {
    return xi;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToSph(const std::array<Real, 3> &xi) {
    std::array<Real, 3> ex1{1.0, 0.0, 0.0};
    std::array<Real, 3> ex2{0.0, 1.0, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCyl(const std::array<Real, 3> &xi) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);

    return {xi[0] * st, 0.0, xi[0] * ct};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCyl(const std::array<Real, 3> &xi) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    std::array<Real, 3> ex1{st, 0.0, ct};
    std::array<Real, 3> ex2{ct, 0.0, -st};
    std::array<Real, 3> ex3{0.0, 1.0, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToAxi(const std::array<Real, 3> &xi) {
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    return {xi[0] * st, xi[0] * ct, 0.0};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToAxi(const std::array<Real, 3> &xi) {
    const Real ct = std::cos(xi[1]);
    const Real st = std::sin(xi[1]);
    std::array<Real, 3> ex1{st, ct, 0.0};
    std::array<Real, 3> ex2{ct, -st, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }
}; // Coords

template <>
class Coords<Coordinates::spherical1D>
    : public CoordsBase<Coords<Coordinates::spherical1D>> {
  // the derived  specialization for 1D spherical coordinates

 public:
  KOKKOS_INLINE_FUNCTION
  Coords(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::spherical1D>>(pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::spherical1D>>() {}

  KOKKOS_INLINE_FUNCTION bool x1dep() { return true; }

  KOKKOS_INLINE_FUNCTION Real hx2(const Real x1, const Real x2, const Real x3) {
    return x1;
  }

  KOKKOS_INLINE_FUNCTION Real x1v() {
    const Real dr2 = bnds.x1[0] * bnds.x1[0] + bnds.x1[1] * bnds.x1[1];
    return 0.75 * (bnds.x1[0] + bnds.x1[1]) * dr2 / (dr2 + bnds.x1[0] * bnds.x1[1]);
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX2(const CellFace f) {
    // <r> = d(r^3/3) / d(r^2/2)
    return {2.0 / 3.0 *
                (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                 bnds.x1[1] * bnds.x1[1]) /
                (bnds.x1[0] + bnds.x1[1]),
            M_PI * 0.5, 0.0};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX3(const CellFace f) {
    // <r> = d(r^3/3) / d(r^2/2)
    return {2.0 / 3.0 *
                (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                 bnds.x1[1] * bnds.x1[1]) /
                (bnds.x1[0] + bnds.x1[1]),
            M_PI * 0.5, 0.0};
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) {
    // \int r^2 sin(t) dp*dt = d(-cos(t)) * r^2 *dp
    return x1f * x1f;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX2(const Real x2f) {
    // \int r*sin(t)*dp*dr = d(r^2/2)*sin(t)*dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    return 0.5 * (bnds.x1[1] + bnds.x1[0]) * dx1;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX3(const Real x3f) {
    // \int r*dt*dr = d(r^2/2)*dt
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    return 0.5 * (bnds.x1[0] + bnds.x1[1]) * dx1;
  }

  KOKKOS_INLINE_FUNCTION Real Volume() {
    // \int r^2 sin(t) dr dt dp = d(r^3/3) d(-cos(t)) dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real rfac =
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        3.0;
    return rfac * dx1;
  }

  KOKKOS_INLINE_FUNCTION Real dh2dx1() {
    return 3.0 / 2.0 * (bnds.x1[0] + bnds.x1[1]) /
           (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]);
  }
  KOKKOS_INLINE_FUNCTION Real dh3dx1() {
    return 3.0 / 2.0 * (bnds.x1[0] + bnds.x1[1]) /
           (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]);
  }

  KOKKOS_INLINE_FUNCTION Mat3x2 RFWeights() {
    // Set the flux averaging weights in the rotating frame for the angular momentum
    // \pm ( <R^2>_j^\pm - <R^2> ) where R is the cylindrical radius

    // volume average (r,theta)
    const Real rv = x1v();

    const Real r2cyl = SQR(rv);

    std::array<Real, 2> bx1{r2cyl - SQR(bnds.x1[0]), SQR(bnds.x1[1]) - r2cyl};

    return {bx1, NewArray<Real, 2>(0.0), NewArray<Real, 2>(0.0)};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCart(const std::array<Real, 3> &xi) {
    const Real cp = 1.0;
    const Real sp = 0.0;
    const Real ct = 0.0;
    const Real st = 1.0;
    return {xi[0] * st * cp, xi[0] * st * sp, xi[0] * ct};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCart(const std::array<Real, 3> &xi) {
    const Real cp = 1.0;
    const Real sp = 0.0;
    const Real ct = 0.0;
    const Real st = 1.0;
    std::array<Real, 3> ex1{st * cp, st * sp, ct};
    std::array<Real, 3> ex2{ct * cp, ct * sp, -st};
    std::array<Real, 3> ex3{-sp, cp, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToSph(const std::array<Real, 3> &xi) {
    return xi;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToSph(const std::array<Real, 3> &xi) {
    std::array<Real, 3> ex1{1.0, 0.0, 0.0};
    std::array<Real, 3> ex2{0.0, 1.0, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCyl(const std::array<Real, 3> &xi) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = 0.0;
    const Real st = 1.0;

    return {xi[0] * st, 0.0, xi[0] * ct};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCyl(const std::array<Real, 3> &xi) {
    // rhat = st * Rhat + ct * Zhat = (st, 0, ct)
    // that = ct * Rhat - st * Zhat = (ct, 0, -st)
    // phat = phat                  = (0, 1,0)
    const Real ct = 0.0;
    const Real st = 1.0;
    std::array<Real, 3> ex1{st, 0.0, ct};
    std::array<Real, 3> ex2{ct, 0.0, -st};
    std::array<Real, 3> ex3{0.0, 1.0, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToAxi(const std::array<Real, 3> &xi) {
    const Real ct = 0.0;
    const Real st = 1.0;
    return {xi[0] * st, xi[0] * ct, 0.0};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToAxi(const std::array<Real, 3> &xi) {
    const Real ct = 0.0;
    const Real st = 1.0;
    std::array<Real, 3> ex1{st, ct, 0.0};
    std::array<Real, 3> ex2{ct, -st, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }
}; // Coords

} // namespace geometry

#endif // GEOMETRY_SPHERICAL_HPP_
