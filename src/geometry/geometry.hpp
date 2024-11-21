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
#ifndef GEOMETRY_GEOMETRY_HPP_
#define GEOMETRY_GEOMETRY_HPP_
//! \file geometry.hpp
//! Geometry is handled via separate namespaces and a simple dispatch system to determine
//! which namespace to use. Throughout, if constexpr switches always have an ultimate
//! return statement outside of the if statement. This is to avoid spurious "missing
//! return statement" warnings when compiling with nvcc. See e.g.,
//! https://stackoverflow.com/a/64561686

// Artemis includes
#include "artemis.hpp"

using namespace parthenon::package::prelude;

namespace geometry {

// Face indexing
enum class CellFace { lower = 0, upper = 1 };

//----------------------------------------------------------------------------------------
//! \fn  Coordinates geometry::CoordSelect
//! \brief
inline Coordinates CoordSelect(std::string sys) {
  if (sys == "cartesian") {
    return Coordinates::cartesian;
  } else if (sys == "spherical") {
    return Coordinates::spherical;
  } else if (sys == "cylindrical") {
    return Coordinates::cylindrical;
  } else if (sys == "axisymmetric") {
    return Coordinates::axisymmetric;
  } else {
    return Coordinates::null;
  }
}

//----------------------------------------------------------------------------------------
//! \struct  geometry::BBox
//! \brief
struct BBox {
  // The bounding box of the zone
  // This is simply a container for the coords.Xf values
  KOKKOS_FUNCTION
  BBox(const parthenon::Coordinates_t &coords, const int k, const int j, const int i) {
    x1[0] = coords.Xf<X1DIR>(i);
    x1[1] = coords.Xf<X1DIR>(i + 1);
    x2[0] = coords.Xf<X2DIR>(j);
    x2[1] = coords.Xf<X2DIR>(j + 1);
    x3[0] = coords.Xf<X3DIR>(k);
    x3[1] = coords.Xf<X3DIR>(k + 1);
  }
  KOKKOS_FUNCTION
  BBox() {}

  Real x1[2] = {Null<Real>()};
  Real x2[2] = {Null<Real>()};
  Real x3[2] = {Null<Real>()};
};

//----------------------------------------------------------------------------------------
//! Special functions that we pull out of the class
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool x1dep() {
  return ((T == Coordinates::cylindrical) || (T == Coordinates::spherical) ||
          (T == Coordinates::axisymmetric));
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool x2dep() {
  return (T == Coordinates::spherical);
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool x3dep() {
  return false;
}

KOKKOS_INLINE_FUNCTION
bool x1dep(Coordinates T) {
  return ((T == Coordinates::cylindrical) || (T == Coordinates::spherical) ||
          (T == Coordinates::axisymmetric));
}
KOKKOS_INLINE_FUNCTION
bool x2dep(Coordinates T) { return (T == Coordinates::spherical); }
KOKKOS_INLINE_FUNCTION
bool x3dep(Coordinates T) { return false; }

//----------------------------------------------------------------------------------------
//! \class  geometry::CoordsBase
//! \brief  The base coordinates class that defines all methods and the default behavior
//! which is Cartesian.
template <class T>
class CoordsBase {
 public:
  BBox bnds;

  KOKKOS_INLINE_FUNCTION
  CoordsBase(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : bnds(pco, k, j, i) {}

  // This constructor allows us to easily access the coordinate conversion routines
  KOKKOS_INLINE_FUNCTION
  CoordsBase() : bnds() {}

  // Is the metric x1, x2, and/or x3 dependent?
  KOKKOS_INLINE_FUNCTION bool x1dep() { return false; }
  KOKKOS_INLINE_FUNCTION bool x2dep() { return false; }
  KOKKOS_INLINE_FUNCTION bool x3dep() { return false; }

  // Volume averaged coordinates
  // xiv = \int x_i dV/ \int dV
  KOKKOS_INLINE_FUNCTION Real x1v() { return 0.5 * (bnds.x1[0] + bnds.x1[1]); }
  KOKKOS_INLINE_FUNCTION Real x2v() { return 0.5 * (bnds.x2[0] + bnds.x2[1]); }
  KOKKOS_INLINE_FUNCTION Real x3v() { return 0.5 * (bnds.x3[0] + bnds.x3[1]); }

  // Scale factor functions and volume averaged scale factors
  // hxiv = \iht h_i dV/dV
  KOKKOS_INLINE_FUNCTION Real hx1(const Real x1, const Real x2, const Real x3) {
    return 1.0;
  }
  KOKKOS_INLINE_FUNCTION Real hx2(const Real x1, const Real x2, const Real x3) {
    return 1.0;
  }
  KOKKOS_INLINE_FUNCTION Real hx3(const Real x1, const Real x2, const Real x3) {
    return 1.0;
  }

  KOKKOS_INLINE_FUNCTION Real hx1v() { return 1.0; }
  KOKKOS_INLINE_FUNCTION Real hx2v() { return 1.0; }
  KOKKOS_INLINE_FUNCTION Real hx3v() { return 1.0; }

  KOKKOS_INLINE_FUNCTION void FaceCenX1(const CellFace f, Real *xf) {
    // The centroid value of the X1 face
    xf[0] = bnds.x1[static_cast<int>(f)];
    xf[1] = static_cast<T *>(this)->x2v();
    xf[2] = static_cast<T *>(this)->x3v();
  }
  KOKKOS_INLINE_FUNCTION void FaceCenX2(const CellFace f, Real *xf) {
    // The centroid value of the X2 face
    xf[0] = static_cast<T *>(this)->x1v();
    xf[1] = bnds.x2[static_cast<int>(f)];
    xf[2] = static_cast<T *>(this)->x3v();
  }
  KOKKOS_INLINE_FUNCTION void FaceCenX3(const CellFace f, Real *xf) {
    // The centroid value of the X3 face
    xf[0] = static_cast<T *>(this)->x1v();
    xf[1] = static_cast<T *>(this)->x2v();
    xf[2] = bnds.x3[static_cast<int>(f)];
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) {
    // The X1 face area
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return dx2 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX2(const Real x2f) {
    // The X2 face area
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return dx1 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX3(Real x3f) {
    // The X3 face area
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    return dx1 * dx2;
  }

  KOKKOS_INLINE_FUNCTION Real Volume() {
    // The zone volume
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return dx1 * dx2 * dx3;
  }

  // The relevant components of the connection for
  // orthogonal coordinates
  KOKKOS_INLINE_FUNCTION Real dh1dx1() { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh1dx2() { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh1dx3() { return 0.0; }

  KOKKOS_INLINE_FUNCTION Real dh2dx1() { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh2dx2() { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh2dx3() { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh3dx1() { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh3dx2() { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh3dx3() { return 0.0; }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCart(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to Cartesian
    xo[0] = xi[0];
    xo[1] = xi[1];
    xo[2] = xi[2];
  }

  KOKKOS_INLINE_FUNCTION void ConvertVecToCart(const Real xi[3], Real ex1[3], Real ex2[3],
                                               Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to Cartesian
    // clang-format off
    ex1[0] = 1.0;  ex1[1] = 0.0;  ex1[2] = 0.0;
    ex2[0] = 0.0;  ex2[1] = 1.0;  ex2[2] = 0.0;
    ex3[0] = 0.0;  ex3[1] = 0.0;  ex3[2] = 1.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToSph(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to spherical
    const Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real r = std::sqrt(R * R + xi[2] * xi[2]);
    const Real ct = xi[2] / ((r == 0) ? Fuzz<Real>() : r);
    const Real cp = xi[0] / ((R == 0) ? Fuzz<Real>() : R);
    const Real sp = xi[1] / ((R == 0) ? Fuzz<Real>() : R);
    xo[0] = r;
    xo[1] = std::acos(ct);
    xo[2] = std::atan2(sp, cp);
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToSph(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to spherical
    const Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real r = std::sqrt(R * R + xi[2] * xi[2]);
    const Real ct = xi[2] / ((r == 0) ? Fuzz<Real>() : r);
    const Real st = R / ((r == 0) ? Fuzz<Real>() : r);
    const Real cp = xi[0] / ((R == 0) ? Fuzz<Real>() : R);
    const Real sp = xi[1] / ((R == 0) ? Fuzz<Real>() : R);
    // clang-format off
    ex1[0] = st * cp;  ex1[1] = ct * cp;  ex1[2] = -sp;
    ex2[0] = st * sp;  ex2[1] = ct * sp;  ex2[2] = cp;
    ex3[0] = ct;       ex3[1] = -st;      ex3[2] = 0.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToCyl(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to cylindrical
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / ((R == 0) ? Fuzz<Real>() : R);
    const Real sp = xi[1] / ((R == 0) ? Fuzz<Real>() : R);
    xo[0] = R;
    xo[1] = std::atan2(sp, cp);
    xo[2] = xi[2];
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToCyl(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to cylindrical
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / ((R == 0) ? Fuzz<Real>() : R);
    const Real sp = xi[1] / ((R == 0) ? Fuzz<Real>() : R);
    // clang-format off
    ex1[0] = cp;   ex1[1] = -sp;  ex1[2] = 0.0;
    ex2[0] = sp;   ex2[1] = cp;   ex2[2] = 0.0;
    ex3[0] = 0.0;  ex3[1] = 0.0;  ex3[2] = 1.0;
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION void ConvertCoordsToAxi(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to axisymmetric
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / ((R == 0) ? Fuzz<Real>() : R);
    const Real sp = xi[1] / ((R == 0) ? Fuzz<Real>() : R);
    xo[0] = R;
    xo[1] = xi[2];
    xo[2] = std::atan2(sp, cp);
  }
  KOKKOS_INLINE_FUNCTION void ConvertVecToAxi(const Real xi[3], Real ex1[3], Real ex2[3],
                                              Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to axisymmetric
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / ((R == 0) ? Fuzz<Real>() : R);
    const Real sp = xi[1] / ((R == 0) ? Fuzz<Real>() : R);
    // clang-format off
    ex1[0] = cp;   ex1[1] = 0.0;  ex1[2] = -sp;
    ex2[0] = sp;   ex2[1] = 0.0;  ex2[2] = cp;
    ex3[0] = 0.0;  ex3[1] = 1.0;  ex3[2] = 0.0;
    // clang-format on
  }

  // NOTE(AMD):
  // The following methods are helper functions to return aggregate data
  // We use static_cast<T*>(this)-> to access the methods in order to call
  // the correct derived variants. This is the key to static polymorphism and CRTP.

  KOKKOS_INLINE_FUNCTION Real GetCellWidthX1() {
    // The cell width in the X1 direction
    const Real xv[3] = {static_cast<T *>(this)->x1v(), static_cast<T *>(this)->x2v(),
                        static_cast<T *>(this)->x3v()};
    return static_cast<T *>(this)->hx1(xv[0], xv[1], xv[2]) * (bnds.x1[1] - bnds.x1[0]);
  }
  KOKKOS_INLINE_FUNCTION Real GetCellWidthX2() {
    // The cell width in the X2 direction
    const Real xv[3] = {static_cast<T *>(this)->x1v(), static_cast<T *>(this)->x2v(),
                        static_cast<T *>(this)->x3v()};
    return static_cast<T *>(this)->hx2(xv[0], xv[1], xv[2]) * (bnds.x2[1] - bnds.x2[0]);
  }
  KOKKOS_INLINE_FUNCTION Real GetCellWidthX3() {
    // The cell width in the X3 direction
    const Real xv[3] = {static_cast<T *>(this)->x1v(), static_cast<T *>(this)->x2v(),
                        static_cast<T *>(this)->x3v()};
    return static_cast<T *>(this)->hx3(xv[0], xv[1], xv[2]) * (bnds.x3[1] - bnds.x3[0]);
  }

  KOKKOS_INLINE_FUNCTION void GetCellWidths(Real *dx) {
    // Return all cell widths
    const Real xv[3] = {static_cast<T *>(this)->x1v(), static_cast<T *>(this)->x2v(),
                        static_cast<T *>(this)->x3v()};
    dx[0] = static_cast<T *>(this)->hx1(xv[0], xv[1], xv[2]) * (bnds.x1[1] - bnds.x1[0]);
    dx[1] = static_cast<T *>(this)->hx2(xv[0], xv[1], xv[2]) * (bnds.x2[1] - bnds.x2[0]);
    dx[2] = static_cast<T *>(this)->hx3(xv[0], xv[1], xv[2]) * (bnds.x3[1] - bnds.x3[0]);
  }

  KOKKOS_INLINE_FUNCTION void GetCellCenter(Real *xv) {
    // Get the cell centroid
    xv[0] = static_cast<T *>(this)->x1v();
    xv[1] = static_cast<T *>(this)->x2v();
    xv[2] = static_cast<T *>(this)->x3v();
  }

  KOKKOS_INLINE_FUNCTION void GetScaleFactors(Real *hx) {
    // Get the volume averaged scale factors
    hx[0] = static_cast<T *>(this)->hx1v();
    hx[1] = static_cast<T *>(this)->hx2v();
    hx[2] = static_cast<T *>(this)->hx3v();
  }

  KOKKOS_INLINE_FUNCTION void GetFaceAreaX1(Real *areas) {
    // Get the lower and upper face areas in the X1 direction
    areas[0] = static_cast<T *>(this)->AreaX1(bnds.x1[0]);
    areas[1] = static_cast<T *>(this)->AreaX1(bnds.x1[1]);
  }
  KOKKOS_INLINE_FUNCTION void GetFaceAreaX2(Real *areas) {
    // Get the lower and upper face areas in the X2 direction
    areas[0] = static_cast<T *>(this)->AreaX2(bnds.x2[0]);
    areas[1] = static_cast<T *>(this)->AreaX2(bnds.x2[1]);
  }
  KOKKOS_INLINE_FUNCTION void GetFaceAreaX3(Real *areas) {
    // Get the lower and upper face areas in the X3 direction
    areas[0] = static_cast<T *>(this)->AreaX3(bnds.x3[0]);
    areas[1] = static_cast<T *>(this)->AreaX3(bnds.x3[1]);
  }

  KOKKOS_INLINE_FUNCTION Real GetFaceArea(const parthenon::CoordinateDirection &dir) {
    // Get the face area in the direction asked for
    PARTHENON_DEBUG_REQUIRE((dir >= X1DIR) && (dir <= X3DIR),
                            "GetFaceAreaDirection is bad!");
    return (dir == parthenon::CoordinateDirection::X1DIR)
               ? static_cast<T *>(this)->AreaX1(bnds.x1[0])
               : ((dir == parthenon::CoordinateDirection::X2DIR)
                      ? static_cast<T *>(this)->AreaX2(bnds.x2[0])
                      : static_cast<T *>(this)->AreaX3(bnds.x3[0]));
  }

  KOKKOS_INLINE_FUNCTION Real Distance(Real *x1, Real *x2) {
    // { dh1/dx1, dh2/dx1, dh3/dx1 }
    Real xc1[3], xc2[3];
    static_cast<T *>(this)->ConvertToCart(x1, xc1);
    static_cast<T *>(this)->ConvertToCart(x2, xc2);
    return std::sqrt(SQR(xc1[0] - xc2[0]) + SQR(xc1[1] - xc2[1]) + SQR(xc1[2] - xc2[2]));
  }

  KOKKOS_INLINE_FUNCTION void GetConnX1(Real *Gamma) {
    // { dh1/dx1, dh2/dx1, dh3/dx1 }
    Gamma[0] = static_cast<T *>(this)->dh1dx1();
    Gamma[1] = static_cast<T *>(this)->dh2dx1();
    Gamma[2] = static_cast<T *>(this)->dh3dx1();
  }
  KOKKOS_INLINE_FUNCTION void GetConnX2(Real *Gamma) {
    // { dh1/dx2, dh2/dx2, dh3/dx2 }
    Gamma[0] = static_cast<T *>(this)->dh1dx2();
    Gamma[1] = static_cast<T *>(this)->dh2dx2();
    Gamma[2] = static_cast<T *>(this)->dh3dx2();
  }
  KOKKOS_INLINE_FUNCTION void GetConnX3(Real *Gamma) {
    // { dh1/dx3, dh2/dx3, dh3/dx3 }
    Gamma[0] = static_cast<T *>(this)->dh1dx3();
    Gamma[1] = static_cast<T *>(this)->dh2dx3();
    Gamma[2] = static_cast<T *>(this)->dh3dx3();
  }

  KOKKOS_INLINE_FUNCTION void GetConns(Real *GammaX1, Real *GammaX2, Real *GammaX3) {
    // Return all of the connections
    GammaX1[0] = static_cast<T *>(this)->dh1dx1();
    GammaX1[1] = static_cast<T *>(this)->dh2dx1();
    GammaX1[2] = static_cast<T *>(this)->dh3dx1();

    GammaX2[0] = static_cast<T *>(this)->dh1dx2();
    GammaX2[1] = static_cast<T *>(this)->dh2dx2();
    GammaX2[2] = static_cast<T *>(this)->dh3dx2();

    GammaX3[0] = static_cast<T *>(this)->dh1dx3();
    GammaX3[1] = static_cast<T *>(this)->dh2dx3();
    GammaX3[2] = static_cast<T *>(this)->dh3dx3();
  }

  KOKKOS_INLINE_FUNCTION void ConvertToCart(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to Cartesian
    static_cast<T *>(this)->ConvertCoordsToCart(xi, xo);
  }

  KOKKOS_INLINE_FUNCTION void ConvertToCartWithVec(const Real xi[3], Real xo[3],
                                                   Real ex1[3], Real ex2[3],
                                                   Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to Cartesian

    static_cast<T *>(this)->ConvertVecToCart(xi, ex1, ex2, ex3);
    static_cast<T *>(this)->ConvertCoordsToCart(xi, xo);
  }

  KOKKOS_INLINE_FUNCTION void ConvertToAxi(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to Axisymmetric
    static_cast<T *>(this)->ConvertCoordsToAxi(xi, xo);
  }

  KOKKOS_INLINE_FUNCTION void ConvertToAxiWithVec(const Real xi[3], Real xo[3],
                                                  Real ex1[3], Real ex2[3], Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to Axisymmetric

    static_cast<T *>(this)->ConvertVecToAxi(xi, ex1, ex2, ex3);
    static_cast<T *>(this)->ConvertCoordsToAxi(xi, xo);
  }

  KOKKOS_INLINE_FUNCTION void ConvertToSph(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to spherical
    static_cast<T *>(this)->ConvertCoordsToSph(xi, xo);
  }

  KOKKOS_INLINE_FUNCTION void ConvertToSphWithVec(const Real xi[3], Real xo[3],
                                                  Real ex1[3], Real ex2[3], Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to spherical

    static_cast<T *>(this)->ConvertVecToSph(xi, ex1, ex2, ex3);
    static_cast<T *>(this)->ConvertCoordsToSph(xi, xo);
  }

  KOKKOS_INLINE_FUNCTION void ConvertToCyl(const Real xi[3], Real xo[3]) {
    // Convert the input vector from the problem coordinate system to cylindrical
    static_cast<T *>(this)->ConvertCoordsToCyl(xi, xo);
  }

  KOKKOS_INLINE_FUNCTION void ConvertToCylWithVec(const Real xi[3], Real xo[3],
                                                  Real ex1[3], Real ex2[3], Real ex3[3]) {
    // Convert the input vector from the problem coordinate system to cylindrical

    static_cast<T *>(this)->ConvertVecToCyl(xi, ex1, ex2, ex3);
    static_cast<T *>(this)->ConvertCoordsToCyl(xi, xo);
  }
};

//----------------------------------------------------------------------------------------
//! The derived base coordinates class
template <Coordinates GEOM>
class Coords : public CoordsBase<Coords<GEOM>> {};

//----------------------------------------------------------------------------------------
//! The derived cartesian specialization
template <>
class Coords<Coordinates::cartesian> : public CoordsBase<Coords<Coordinates::cartesian>> {
 public:
  KOKKOS_INLINE_FUNCTION
  Coords(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::cartesian>>(pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::cartesian>>() {}
};

} // namespace geometry

#include "axisymmetric.hpp"
#include "cylindrical.hpp"
#include "spherical.hpp"

#endif // GEOMETRY_GEOMETRY_HPP_
