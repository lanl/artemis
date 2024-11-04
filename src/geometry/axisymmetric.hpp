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
#ifndef GEOMETRY_AXISYMMETRIC_HPP_
#define GEOMETRY_AXISYMMETRIC_HPP_

// Artemis includes
#include "geometry.hpp"

using namespace parthenon::package::prelude;

namespace geometry {
//----------------------------------------------------------------------------------------
//! The derived axisymmetric specialization
//!
//!   Axisymmetric coordinates are defined as
//!
//!          x1  x2   x3
//!     x = ( R,  z,  phi)
//!     e_R   =  cos(phi) e_x + sin(phi) e_y + e_z
//!     e_phi = -sin(phi) e_x + cos(phi) e_y
//!     e_z   =                                e_z
//!
//! This is cylindrical coordinates with phi as x3
//! This is mainly a 1D/2D coordinate system.
template <>
class Coords<Coordinates::axisymmetric>
    : public CoordsBase<Coords<Coordinates::axisymmetric>> {
 public:
  KOKKOS_INLINE_FUNCTION
  Coords(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::axisymmetric>>(pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::axisymmetric>>() {}

  KOKKOS_INLINE_FUNCTION
  bool x1dep() { return true; }

  KOKKOS_INLINE_FUNCTION Real x1v() {
    return 2.0 / 3.0 *
           (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
           (bnds.x1[0] + bnds.x1[1]);
  }

  KOKKOS_INLINE_FUNCTION Real hx3(const Real x1, const Real x2, const Real x3) {
    return x1;
  }

  KOKKOS_INLINE_FUNCTION Real hx3v() { return x1v(); }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX2(const CellFace f) {
    // <r> = d(r^3/3) / d(r^2/2)
    std::array<Real, 3> xf{2.0 / 3.0 *
                               (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                                bnds.x1[1] * bnds.x1[1]) /
                               (bnds.x1[0] + bnds.x1[1]),
                           bnds.x2[static_cast<int>(f)], 0.5 * (bnds.x3[0] + bnds.x3[1])};
    return xf;
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) {
    // \int r*dz*dp = r*dz*dp
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return x1f * dx2 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX2(const Real x2f) {
    // \int r*dp*dr = d(r^2/2)*dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return (bnds.x1[0] + bnds.x1[1]) * 0.5 * dx1 * dx3;
  }

  KOKKOS_INLINE_FUNCTION Real Volume() {
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return (bnds.x1[0] + bnds.x1[1]) * 0.5 * dx1 * dx2 * dx3;
  }

  KOKKOS_INLINE_FUNCTION Real dh3dx1() { return 1.0 / (0.5 * (bnds.x1[0] + bnds.x1[1])); }

  KOKKOS_INLINE_FUNCTION Mat3x2 RFWeights() {
    // Set the flux averaging weights in the rotating frame for the angular momentum
    // \pm ( <R^2>_j^\pm - <R^2> )
    const Real ans = 0.5 * (bnds.x1[0] + bnds.x1[1]) * (bnds.x1[1] - bnds.x1[0]);
    return {NewArray<Real, 2>(ans), NewArray<Real, 2>(0.0), NewArray<Real, 2>(0.0)};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCart(const std::array<Real, 3> &xi) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    std::array<Real, 3> xo{xi[0] * cp, xi[0] * sp, xi[1]};
    return xo;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCart(const std::array<Real, 3> &xi) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    // clang-format off
    std::array<Real,3> ex1{  cp, 0.0,  sp};
    std::array<Real,3> ex2{ -sp, 0.0,  cp};
    std::array<Real,3> ex3{ 0.0, 1.0, 0.0};
    // clang-format on
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToSph(const std::array<Real, 3> &xi) {
    const Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real ct = xi[1] / (R + Fuzz<Real>());
    const Real st = xi[0] / (R + Fuzz<Real>());
    std::array<Real, 3> xo{R, std::acos(ct), xi[2]};
    return xo;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToSph(const std::array<Real, 3> &xi) {
    const Real rsph = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real ct = xi[1] / (rsph + Fuzz<Real>());
    const Real st = xi[0] / (rsph + Fuzz<Real>());
    // clang-format off
    std::array<Real,3> ex1{  st, 0.0,  ct};
    std::array<Real,3> ex2{ 0.0, 1.0, 0.0};
    std::array<Real,3> ex3{  ct, 0.0, -st};
    // clang-format on
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCyl(const std::array<Real, 3> &xi) {
    std::array<Real, 3> xo{xi[0], xi[2], xi[1]};
    return xo;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCyl(const std::array<Real, 3> &xi) {
    // clang-format off
    std::array<Real,3> ex1{ 1.0, 0.0, 0.0};
    std::array<Real,3> ex2{ 0.0, 0.0, 1.0};
    std::array<Real,3> ex3{ 0.0, 1.0, 0.0};
    // clang-format on
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToAxi(const std::array<Real, 3> &xi) {
    return xi;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToAxi(const std::array<Real, 3> &xi) {
    std::array<Real, 3> ex1{1.0, 0.0, 0.0};
    std::array<Real, 3> ex2{0.0, 1.0, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }
}; // Coords

} // namespace geometry

#endif // GEOMETRY_AXISYMMETRIC_HPP_
