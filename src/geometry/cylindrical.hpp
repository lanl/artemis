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
#ifndef GEOMETRY_CYLINDRICAL_HPP_
#define GEOMETRY_CYLINDRICAL_HPP_

// Artemis includes
#include "geometry.hpp"

using namespace parthenon::package::prelude;

namespace geometry {
//----------------------------------------------------------------------------------------
//! The derived cylindrical specialization
//!
//!   Cylindrical coordinates are defined as
//!
//!          x1   x2   x3
//!     x = ( R,  phi,  z)
//!     e_R   =  cos(phi) e_x + sin(phi) e_y + e_z
//!     e_phi = -sin(phi) e_x + cos(phi) e_y
//!     e_z   =                                e_z
template <>
class Coords<Coordinates::cylindrical>
    : public CoordsBase<Coords<Coordinates::cylindrical>> {
 public:
  KOKKOS_INLINE_FUNCTION
  Coords(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::cylindrical>>(pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::cylindrical>>() {}

  KOKKOS_INLINE_FUNCTION
  bool x1dep() { return true; }

  KOKKOS_INLINE_FUNCTION Real x1v() {
    return 2.0 / 3.0 *
           (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
           (bnds.x1[0] + bnds.x1[1]);
  }

  KOKKOS_INLINE_FUNCTION Real hx2(const Real x1, const Real x2, const Real x3) {
    return x1;
  }

  KOKKOS_INLINE_FUNCTION Real hx2v() { return x1v(); }

  KOKKOS_INLINE_FUNCTION void FaceCenX3(const CellFace f, Real *xf) {
    //  <r> = d(r^3/3) / d(r^2/2)
    xf[0] =
        2.0 / 3.0 *
        (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
        (bnds.x1[0] + bnds.x1[1]);
    xf[1] = 0.5 * (bnds.x2[0] + bnds.x2[1]);
    xf[2] = bnds.x3[static_cast<int>(f)];
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) {
    // \int r*dz*dp   =  r*dz*dp
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return x1f * dx2 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX3(const Real x3f) {
    // \int r*dp*dr = d(r^2/2)*dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    return 0.5 * (bnds.x1[0] + bnds.x1[1]) * dx1 * dx2;
  }

  KOKKOS_INLINE_FUNCTION Real Volume() {
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return (bnds.x1[0] + bnds.x1[1]) * 0.5 * dx1 * dx2 * dx3;
  }

  KOKKOS_INLINE_FUNCTION Real dh2dx1() { return 1.0 / (0.5 * (bnds.x1[0] + bnds.x1[1])); }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCart(const Real xi[3], Real xo[3]) {
    const Real cp = std::cos(xi[1]);
    const Real sp = std::sin(xi[1]);
    xo[0] = xi[0] * cp;
    xo[1] = xi[0] * sp;
    xo[2] = xi[2];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToCart(const Real xi[3], Real ex1[3], Real ex2[3],
                                               Real ex3[3]) {
    const Real cp = std::cos(xi[1]);
    const Real sp = std::sin(xi[1]);
    // clang-format off
    ex1[0] = cp;   ex1[1] = sp;   ex1[2] = 0.0;
    ex2[0] = -sp;  ex2[1] = cp;   ex2[2] = 0.0;
    ex3[0] = 0.0;  ex3[1] = 0.0;  ex3[2] = 1.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToSph(const Real xi[3], Real xo[3]) {
    xo[0] = std::sqrt(xi[0] * xi[0] + xi[2] * xi[2]);
    const Real ct = xi[2] / (xo[0] == 0.0 ? Fuzz<Real>() : xo[0]);
    const Real st = xi[0] / (xo[0] == 0.0 ? Fuzz<Real>() : xo[0]);
    xo[1] = std::acos(ct);
    xo[2] = xi[1];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToSph(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    const Real rsph = std::sqrt(xi[0] * xi[0] + xi[2] * xi[2]);
    const Real ct = xi[2] / (rsph == 0 ? Fuzz<Real>() : rsph);
    const Real st = xi[0] / (rsph == 0 ? Fuzz<Real>() : rsph);
    // clang-format off
    ex1[0] = ct;   ex1[1] = ct;   ex1[2] = 0.0;
    ex2[0] = 0.0;  ex2[1] = 0.0;  ex2[2] = 1.0;
    ex3[0] = st;   ex3[1] = -st;  ex3[2] = 0.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCyl(const Real xi[3], Real xo[3]) {
    xo[0] = xi[0];
    xo[1] = xi[1];
    xo[2] = xi[2];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToCyl(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // clang-format off
    ex1[0] = 1.0;  ex1[1] = 0.0;  ex1[2] = 0.0;
    ex2[0] = 0.0;  ex2[1] = 1.0;  ex2[2] = 0.0;
    ex3[0] = 0.0;  ex3[1] = 0.0;  ex3[2] = 1.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToAxi(const Real xi[3], Real xo[3]) {
    xo[0] = xi[0];
    xo[1] = xi[2];
    xo[2] = xi[1];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToAxi(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // clang-format off
    ex1[0] = 1.0;  ex1[1] = 0.0;  ex1[2] = 0.0;
    ex2[0] = 0.0;  ex2[1] = 0.0;  ex2[2] = 1.0;
    ex3[0] = 0.0;  ex3[1] = 1.0;  ex3[2] = 0.0;
    // clang-format on
  }
}; // Coords

} // namespace geometry

#endif // GEOMETRY_CYLINDRICAL_HPP_
