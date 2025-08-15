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

// NOTE(@amd)
// This is a dirty trick because I am running into constexpr issues with actual member
// functions of the CRTP classes
namespace cyl {
template <class VAR>
constexpr bool is_x1dep() {
  return (std::is_same_v<VAR, geom::x1v> || std::is_same_v<VAR, geom::vol>);
}
template <class VAR>
constexpr bool is_x2dep() {
  return false;
}
template <class VAR>
constexpr bool is_x3dep() {
  return false;
}
} // namespace cyl

template <>
class Coords<Coordinates::cylindrical>
    : public CoordsBase<Coords<Coordinates::cylindrical>> {
 public:
  template <typename PAR>
  KOKKOS_INLINE_FUNCTION Coords(const PAR &cpars, const parthenon::Coordinates_t &pco,
                                const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::cylindrical>>(cpars, pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords(const bool log, const parthenon::Coordinates_t &pco, const int k, const int j,
         const int i)
      : CoordsBase<Coords<Coordinates::cylindrical>>(log, pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::cylindrical>>() {}
  template <typename PAR>
  Coords(const PAR &cpars) : CoordsBase<Coords<Coordinates::cylindrical>>(cpars) {}

  template <class VAR>
  KOKKOS_INLINE_FUNCTION int index_(const int k, const int j, const int i) const {
    if constexpr (cyl::is_x1dep<VAR>()) {
      return i;
    }
    return 0;
  }
  template <class VAR>
  KOKKOS_INLINE_FUNCTION std::array<int, 3> shape_() const {
    if constexpr (cyl::is_x1dep<VAR>()) {
      return {nx[0], 1, 1};
    }
    return {0, 0, 0};
  }

  KOKKOS_INLINE_FUNCTION bool x1dep() const { return true; }

  KOKKOS_INLINE_FUNCTION Real x1v() const {
    return 2.0 / 3.0 *
           (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] + bnds.x1[1] * bnds.x1[1]) /
           (bnds.x1[0] + bnds.x1[1]);
  }

  KOKKOS_INLINE_FUNCTION Real hx2(const Real x1, const Real x2, const Real x3) const {
    return x1;
  }

  KOKKOS_INLINE_FUNCTION Real hx2v() const { return x1v(); }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX3(const CellFace f) const {
    //  <r> = d(r^3/3) / d(r^2/2)
    return {2.0 / 3.0 *
                (bnds.x1[0] * bnds.x1[0] + bnds.x1[0] * bnds.x1[1] +
                 bnds.x1[1] * bnds.x1[1]) /
                (bnds.x1[0] + bnds.x1[1]),
            0.5 * (bnds.x2[0] + bnds.x2[1]), bnds.x3[static_cast<int>(f)]};
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) const {
    // \int r*dz*dp   =  r*dz*dp
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return x1f * dx2 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX3(const Real x3f) const {
    // \int r*dp*dr = d(r^2/2)*dp
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    return 0.5 * (bnds.x1[0] + bnds.x1[1]) * dx1 * dx2;
  }

  KOKKOS_INLINE_FUNCTION Real Volume() const {
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return (bnds.x1[0] + bnds.x1[1]) * 0.5 * dx1 * dx2 * dx3;
  }

  KOKKOS_INLINE_FUNCTION Real dh2dx1() const {
    return 1.0 / (0.5 * (bnds.x1[0] + bnds.x1[1]));
  }

  KOKKOS_INLINE_FUNCTION Mat3x2 RFWeights() const {
    // Set the flux averaging weights in the rotating frame for the angular momentum
    // \pm ( <R^2>_j^\pm - <R^2> )
    const Real ans = 0.5 * (bnds.x1[0] + bnds.x1[1]) * (bnds.x1[1] - bnds.x1[0]);
    return {NewArray<Real, 2>(ans), NewArray<Real, 2>(0.0), NewArray<Real, 2>(0.0)};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCart(const std::array<Real, 3> &xi) const {
    const Real cp = std::cos(xi[1]);
    const Real sp = std::sin(xi[1]);
    return {xi[0] * cp, xi[0] * sp, xi[2]};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCart(const std::array<Real, 3> &xi) const {
    const Real cp = std::cos(xi[1]);
    const Real sp = std::sin(xi[1]);
    std::array<Real, 3> ex1{cp, sp, 0.0};
    std::array<Real, 3> ex2{-sp, cp, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToSph(const std::array<Real, 3> &xi) const {
    const Real R = std::sqrt(xi[0] * xi[0] + xi[2] * xi[2]);
    const Real ct = xi[2] / (R + Fuzz<Real>());
    const Real st = xi[0] / (R + Fuzz<Real>());
    return {R, std::acos(ct), xi[1]};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToSph(const std::array<Real, 3> &xi) const {
    const Real rsph = std::sqrt(xi[0] * xi[0] + xi[2] * xi[2]);
    const Real ct = xi[2] / (rsph + Fuzz<Real>());
    const Real st = xi[0] / (rsph + Fuzz<Real>());
    std::array<Real, 3> ex1{st, ct, 0.0};
    std::array<Real, 3> ex2{0.0, 0.0, 1.0};
    std::array<Real, 3> ex3{ct, -st, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCyl(const std::array<Real, 3> &xi) const {
    return xi;
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCyl(const std::array<Real, 3> &xi) const {
    std::array<Real, 3> ex1{1.0, 0.0, 0.0};
    std::array<Real, 3> ex2{0.0, 1.0, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToAxi(const std::array<Real, 3> &xi) const {
    return {xi[0], xi[2], xi[1]};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToAxi(const std::array<Real, 3> &xi) const {
    std::array<Real, 3> ex1{1.0, 0.0, 0.0};
    std::array<Real, 3> ex2{0.0, 0.0, 1.0};
    std::array<Real, 3> ex3{0.0, 1.0, 0.0};
    return {ex1, ex2, ex3};
  }
}; // Coords

} // namespace geometry

#endif // GEOMETRY_CYLINDRICAL_HPP_
