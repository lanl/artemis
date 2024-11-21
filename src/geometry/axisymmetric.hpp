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

  KOKKOS_INLINE_FUNCTION void FaceCenX2(const CellFace f, Real *xf) {
    // <r> = d(r^3/3) / d(r^2/2)
    xf[0] =
        2.0 / 3.0 *
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        (bnds.x1[0] + bnds.x1[1]);
    xf[1] = bnds.x2[static_cast<int>(f)];
    xf[2] = 0.5 * (bnds.x3[0] + bnds.x3[1]);
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

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCart(const Real xi[3], Real xo[3]) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    xo[0] = xi[0] * cp;
    xo[1] = xi[0] * sp;
    xo[2] = xi[1];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToCart(const Real xi[3], Real ex1[3], Real ex2[3],
                                               Real ex3[3]) {
    const Real cp = std::cos(xi[2]);
    const Real sp = std::sin(xi[2]);
    // clang-format off
    ex1[0] = cp;   ex1[1] = 0.0;  ex1[2] = sp;
    ex2[0] = -sp;  ex2[1] = 0.0;  ex2[2] = cp;
    ex3[0] = 0.0;  ex3[1] = 1.0;  ex3[2] = 0.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToSph(const Real xi[3], Real xo[3]) {
    xo[0] = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real ct = xi[1] / (xo[0] == 0.0 ? Fuzz<Real>() : xo[0]);
    const Real st = xi[0] / (xo[0] == 0.0 ? Fuzz<Real>() : xo[0]);
    xo[1] = std::acos(ct);
    xo[2] = xi[2];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToSph(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    const Real rsph = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real ct = xi[1] / (rsph == 0.0 ? Fuzz<Real>() : rsph);
    const Real st = xi[0] / (rsph == 0.0 ? Fuzz<Real>() : rsph);
    // clang-format off
    ex1[0] = ct;   ex1[1] = 0.0;  ex1[2] = ct;
    ex2[0] = 0.0;  ex2[1] = 1.0;  ex2[2] = 0.0;
    ex3[0] = st;   ex3[1] = 0.0;  ex3[2] = -st;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCyl(const Real xi[3], Real xo[3]) {
    xo[0] = xi[0];
    xo[1] = xi[2];
    xo[2] = xi[1];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToCyl(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // clang-format off
    ex1[0] = 1.0;  ex1[1] = 0.0;  ex1[2] = 0.0;
    ex2[0] = 0.0;  ex2[1] = 0.0;  ex2[2] = 1.0;
    ex3[0] = 0.0;  ex3[1] = 1.0;  ex3[2] = 0.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToAxi(const Real xi[3], Real xo[3]) {
    xo[0] = xi[0];
    xo[1] = xi[1];
    xo[2] = xi[2];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToAxi(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // clang-format off
    ex1[0] = 1.0;  ex1[1] = 0.0;  ex1[2] = 0.0;
    ex2[0] = 0.0;  ex2[1] = 1.0;  ex2[2] = 0.0;
    ex3[0] = 0.0;  ex3[1] = 0.0;  ex3[2] = 1.0;
    // clang-format on
  }
}; // Coords

} // namespace geometry

#endif // GEOMETRY_AXISYMMETRIC_HPP_
