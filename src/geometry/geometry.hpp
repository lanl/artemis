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

using Mat3x2 = std::tuple<std::array<Real, 2>, std::array<Real, 2>, std::array<Real, 2>>;
using Mat3x3 = std::tuple<std::array<Real, 3>, std::array<Real, 3>, std::array<Real, 3>>;
using Mat4x3 = std::tuple<std::array<Real, 3>, std::array<Real, 3>, std::array<Real, 3>,
                          std::array<Real, 3>>;

namespace geometry {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
template <Coordinates GEOM>
void InitBlockGeom(MeshBlock *pmb, ParameterInput *pin);
parthenon::TaskStatus UpdateGeom(parthenon::MeshBlockData<Real> *md);

// Face indexing
enum class CellFace { lower = 0, upper = 1 };

//----------------------------------------------------------------------------------------
//! \fn  Coordinates geometry::CoordSelect
//! \brief
inline Coordinates CoordSelect(std::string sys, const int ndim) {
  if (sys == "cartesian") {
    return Coordinates::cartesian;
  } else if (sys == "spherical") {
    if (ndim == 1) {
      return Coordinates::spherical1D;
    } else if (ndim == 2) {
      return Coordinates::spherical2D;
    } else {
      return Coordinates::spherical3D;
    }
  } else if (sys == "cylindrical") {
    return Coordinates::cylindrical;
  } else if (sys == "axisymmetric") {
    return Coordinates::axisymmetric;
  } else {
    return Coordinates::null;
  }
}

KOKKOS_INLINE_FUNCTION bool is_spherical(Coordinates T) {
  return (T == Coordinates::spherical3D) || (T == Coordinates::spherical1D) ||
         (T == Coordinates::spherical2D);
}
KOKKOS_INLINE_FUNCTION bool is_cylindrical(Coordinates T) {
  return (T == Coordinates::cylindrical);
}
KOKKOS_INLINE_FUNCTION bool is_axisymmetric(Coordinates T) {
  return (T == Coordinates::axisymmetric) || (T == Coordinates::spherical1D) ||
         (T == Coordinates::spherical2D);
}
KOKKOS_INLINE_FUNCTION bool is_cartesian(Coordinates T) {
  return (T == Coordinates::cartesian);
}

KOKKOS_INLINE_FUNCTION
bool x1dep(Coordinates T) {
  return is_spherical(T) || is_cylindrical(T) || (T == Coordinates::axisymmetric);
}
KOKKOS_INLINE_FUNCTION
bool x2dep(Coordinates T) {
  return (T == Coordinates::spherical3D) || (T == Coordinates::spherical2D);
}
KOKKOS_INLINE_FUNCTION
bool x3dep(Coordinates T) { return false; }

struct CoordParams {
  CoordParams(ParameterInput *pin) {
    log = (pin->GetOrAddString("artemis", "radial_spacing", "uniform") == "logarithmic");
    nx = {pin->GetInteger("parthenon/meshblock", "nx1"),
          pin->GetInteger("parthenon/meshblock", "nx2"),
          pin->GetInteger("parthenon/meshblock", "nx3")};
    const int ndim = (nx[2] > 1) ? 3 : ((nx[1] > 1) ? 2 : 1);
    sys = CoordSelect(pin->GetOrAddString("artemis", "coordinates", "cartesian"), ndim);
    dep = {x1dep(sys), x2dep(sys), x3dep(sys)};
  }
  KOKKOS_INLINE_FUNCTION
  CoordParams() = default;

  bool log = false;
  std::array<bool, 3> dep;
  std::array<int, 3> nx;
  Coordinates sys;
};

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

  Real x1[2] = {Null<Real>(), Null<Real>()};
  Real x2[2] = {Null<Real>(), Null<Real>()};
  Real x3[2] = {Null<Real>(), Null<Real>()};
};

//----------------------------------------------------------------------------------------
//! Special functions that we pull out of the class

template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool is_spherical() {
  return (T == Coordinates::spherical3D) || (T == Coordinates::spherical1D) ||
         (T == Coordinates::spherical2D);
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool is_cylindrical() {
  return (T == Coordinates::cylindrical);
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool is_axisymmetric() {
  return (T == Coordinates::axisymmetric) || (T == Coordinates::spherical1D) ||
         (T == Coordinates::spherical2D);
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool is_cartesian() {
  return (T == Coordinates::cartesian);
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool x1dep() {
  return is_spherical<T>() || is_cylindrical<T>() || (T == Coordinates::axisymmetric);
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool x2dep() {
  return (T == Coordinates::spherical3D) || (T == Coordinates::spherical2D);
}
template <Coordinates T>
KOKKOS_INLINE_FUNCTION constexpr bool x3dep() {
  return false;
}

//----------------------------------------------------------------------------------------
//! \class  geometry::CoordsBase
//! \brief  The base coordinates class that defines all methods and the default behavior
//! which is Cartesian.
template <class T>
class CoordsBase {
 public:
  BBox bnds;
  std::array<int, 3> nx;
  KOKKOS_INLINE_FUNCTION
  CoordsBase(const parthenon::Coordinates_t &pco, const int k, const int j, const int i)
      : bnds(pco, k, j, i), nx{1, 1, 1} {};

  template <typename PAR>
  KOKKOS_INLINE_FUNCTION CoordsBase(const PAR &cpars, const parthenon::Coordinates_t &pco,
                                    const int k, const int j, const int i)
      : bnds(pco, k, j, i) {
    nx = cpars.nx;
    if (cpars.log) {
      bnds.x1[0] = std::exp(bnds.x1[0]);
      bnds.x1[1] = std::exp(bnds.x1[1]);
    }
  }
  KOKKOS_INLINE_FUNCTION
  CoordsBase(const bool log, const parthenon::Coordinates_t &pco, const int k,
             const int j, const int i)
      : bnds(pco, k, j, i), nx{1, 1, 1} {
    if (log) {
      bnds.x1[0] = std::exp(bnds.x1[0]);
      bnds.x1[1] = std::exp(bnds.x1[1]);
    }
  }

  // This constructor allows us to easily access the coordinate conversion routines
  KOKKOS_INLINE_FUNCTION
  CoordsBase() : bnds(), nx{1, 1, 1} {}

  template <typename PAR>
  CoordsBase(const PAR &cpars) : bnds() {
    nx = cpars.nx;
  }

  template <class VAR>
  KOKKOS_INLINE_FUNCTION int index_(const int k, const int j, const int i) const {
    return 0;
  }
  template <class VAR>
  KOKKOS_INLINE_FUNCTION std::array<int, 3> shape_() const {
    return {1, 1, 1};
  }

  // Is the metric x1, x2, and/or x3 dependent?
  KOKKOS_INLINE_FUNCTION bool x1dep() const { return false; }
  KOKKOS_INLINE_FUNCTION bool x2dep() const { return false; }
  KOKKOS_INLINE_FUNCTION bool x3dep() const { return false; }

  // Volume averaged coordinates
  // xiv = \int x_i dV/ \int dV
  KOKKOS_INLINE_FUNCTION Real x1v() const { return 0.5 * (bnds.x1[0] + bnds.x1[1]); }
  KOKKOS_INLINE_FUNCTION Real x2v() const { return 0.5 * (bnds.x2[0] + bnds.x2[1]); }
  KOKKOS_INLINE_FUNCTION Real x3v() const { return 0.5 * (bnds.x3[0] + bnds.x3[1]); }

  // Scale factor functions and volume averaged scale factors
  // hxiv = \iht h_i dV/dV
  KOKKOS_INLINE_FUNCTION Real hx1(const Real x1, const Real x2, const Real x3) const {
    return 1.0;
  }
  KOKKOS_INLINE_FUNCTION Real hx2(const Real x1, const Real x2, const Real x3) const {
    return 1.0;
  }
  KOKKOS_INLINE_FUNCTION Real hx3(const Real x1, const Real x2, const Real x3) const {
    return 1.0;
  }

  KOKKOS_INLINE_FUNCTION Real hx1v() const { return 1.0; }
  KOKKOS_INLINE_FUNCTION Real hx2v() const { return 1.0; }
  KOKKOS_INLINE_FUNCTION Real hx3v() const { return 1.0; }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX1(const CellFace f) const {
    // The centroid value of the X1 face
    return {bnds.x1[static_cast<int>(f)], static_cast<const T *>(this)->x2v(),
            static_cast<const T *>(this)->x3v()};
  }
  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX2(const CellFace f) const {
    // The centroid value of the X2 face
    return {static_cast<const T *>(this)->x1v(), bnds.x2[static_cast<int>(f)],
            static_cast<const T *>(this)->x3v()};
  }
  KOKKOS_INLINE_FUNCTION std::array<Real, 3> FaceCenX3(const CellFace f) const {
    // The centroid value of the X3 face
    return {static_cast<const T *>(this)->x1v(), static_cast<const T *>(this)->x2v(),
            bnds.x3[static_cast<int>(f)]};
  }

  KOKKOS_INLINE_FUNCTION Real AreaX1(const Real x1f) const {
    // The X1 face area
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return dx2 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX2(const Real x2f) const {
    // The X2 face area
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return dx1 * dx3;
  }
  KOKKOS_INLINE_FUNCTION Real AreaX3(Real x3f) const {
    // The X3 face area
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    return dx1 * dx2;
  }

  KOKKOS_INLINE_FUNCTION Real Volume() const {
    // The zone volume
    const Real dx1 = bnds.x1[1] - bnds.x1[0];
    const Real dx2 = bnds.x2[1] - bnds.x2[0];
    const Real dx3 = bnds.x3[1] - bnds.x3[0];
    return dx1 * dx2 * dx3;
  }

  KOKKOS_INLINE_FUNCTION Mat3x2 RFWeights() const {
    // Set the flux averaging weights in the rotating frame for the angular momentum
    // \pm ( <R^2>_j^\pm - <R^2> )
    return {NewArray<Real, 2>(0.0), NewArray<Real, 2>(0.0), NewArray<Real, 2>(0.0)};
  }

  // The relevant components of the connection for
  // orthogonal coordinates
  KOKKOS_INLINE_FUNCTION Real dh1dx1() const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh1dx2() const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh1dx3() const { return 0.0; }

  KOKKOS_INLINE_FUNCTION Real dh2dx1() const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh2dx2() const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh2dx3() const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh3dx1() const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh3dx2() const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real dh3dx3() const { return 0.0; }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCart(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to Cartesian
    return xi;
  }

  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCart(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to Cartesian
    std::array<Real, 3> ex1{1.0, 0.0, 0.0};
    std::array<Real, 3> ex2{0.0, 1.0, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToSph(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to spherical
    const Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real r = std::sqrt(R * R + xi[2] * xi[2]);
    const Real ct = xi[2] / (r + Fuzz<Real>());
    const Real cp = xi[0] / (R + Fuzz<Real>());
    const Real sp = xi[1] / (R + Fuzz<Real>());
    return {r, std::acos(ct), std::atan2(sp, cp)};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToSph(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to spherical
    const Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real r = std::sqrt(R * R + xi[2] * xi[2]);
    const Real ct = xi[2] / (r + Fuzz<Real>());
    const Real st = R / (r + Fuzz<Real>());
    const Real cp = xi[0] / (R + Fuzz<Real>());
    const Real sp = xi[1] / (R + Fuzz<Real>());
    std::array<Real, 3> ex1{st * cp, ct * cp, -sp};
    std::array<Real, 3> ex2{st * sp, ct * sp, cp};
    std::array<Real, 3> ex3{ct, -st, 0.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToCyl(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to cylindrical
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / (R + Fuzz<Real>());
    const Real sp = xi[1] / (R + Fuzz<Real>());
    return {R, std::atan2(sp, cp), xi[2]};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToCyl(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to cylindrical
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / (R + Fuzz<Real>());
    const Real sp = xi[1] / (R + Fuzz<Real>());
    std::array<Real, 3> ex1{cp, -sp, 0.0};
    std::array<Real, 3> ex2{sp, cp, 0.0};
    std::array<Real, 3> ex3{0.0, 0.0, 1.0};
    return {ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertCoordsToAxi(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to axisymmetric
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / (R + Fuzz<Real>());
    const Real sp = xi[1] / (R + Fuzz<Real>());
    return {R, xi[2], std::atan2(sp, cp)};
  }
  KOKKOS_INLINE_FUNCTION Mat3x3 ConvertVecToAxi(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to axisymmetric
    Real R = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1]);
    const Real cp = xi[0] / (R + Fuzz<Real>());
    const Real sp = xi[1] / (R + Fuzz<Real>());
    std::array<Real, 3> ex1{cp, 0.0, -sp};
    std::array<Real, 3> ex2{sp, 0.0, cp};
    std::array<Real, 3> ex3{0.0, 1.0, 0.0};
    return {ex1, ex2, ex3};
  }

  // NOTE(AMD):
  // The following methods are helper functions to return aggregate data
  // We use static_cast<T*>(this)-> to access the methods in order to call
  // the correct derived variants. This is the key to static polymorphism and CRTP.
  //
  template <class VAR>
  KOKKOS_INLINE_FUNCTION int index(const int k, const int j, const int i) const {
    return static_cast<const T *>(this)->template index_<VAR>(k, j, i);
  }
  template <class VAR>
  std::array<int, 3> shape() const {
    return static_cast<const T *>(this)->template shape_<VAR>();
  }

  KOKKOS_INLINE_FUNCTION BBox &GetBounds() const {
    return static_cast<const T *>(this)->bnds;
  }
  KOKKOS_INLINE_FUNCTION Real GetCellWidthX1() const {
    // The cell width in the X1 direction
    const Real xv[3] = {static_cast<const T *>(this)->x1v(),
                        static_cast<const T *>(this)->x2v(),
                        static_cast<const T *>(this)->x3v()};
    return static_cast<const T *>(this)->hx1(xv[0], xv[1], xv[2]) *
           (bnds.x1[1] - bnds.x1[0]);
  }
  KOKKOS_INLINE_FUNCTION Real GetCellWidthX2() const {
    // The cell width in the X2 direction
    const Real xv[3] = {static_cast<const T *>(this)->x1v(),
                        static_cast<const T *>(this)->x2v(),
                        static_cast<const T *>(this)->x3v()};
    return static_cast<const T *>(this)->hx2(xv[0], xv[1], xv[2]) *
           (bnds.x2[1] - bnds.x2[0]);
  }
  KOKKOS_INLINE_FUNCTION Real GetCellWidthX3() const {
    // The cell width in the X3 direction
    const Real xv[3] = {static_cast<const T *>(this)->x1v(),
                        static_cast<const T *>(this)->x2v(),
                        static_cast<const T *>(this)->x3v()};
    return static_cast<const T *>(this)->hx3(xv[0], xv[1], xv[2]) *
           (bnds.x3[1] - bnds.x3[0]);
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> GetCellWidths() const {
    // Return all cell widths
    const Real xv[3] = {static_cast<const T *>(this)->x1v(),
                        static_cast<const T *>(this)->x2v(),
                        static_cast<const T *>(this)->x3v()};
    return {static_cast<const T *>(this)->hx1(xv[0], xv[1], xv[2]) *
                (bnds.x1[1] - bnds.x1[0]),
            static_cast<const T *>(this)->hx2(xv[0], xv[1], xv[2]) *
                (bnds.x2[1] - bnds.x2[0]),
            static_cast<const T *>(this)->hx3(xv[0], xv[1], xv[2]) *
                (bnds.x3[1] - bnds.x3[0])};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> GetCellCenter() const {
    // Get the cell centroid
    return {static_cast<const T *>(this)->x1v(), static_cast<const T *>(this)->x2v(),
            static_cast<const T *>(this)->x3v()};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> GetScaleFactors() const {
    // Get the volume averaged scale factors
    return {static_cast<const T *>(this)->hx1v(), static_cast<const T *>(this)->hx2v(),
            static_cast<const T *>(this)->hx3v()};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 2> GetFaceAreaX1() const {
    // Get the lower and upper face areas in the X1 direction
    return {static_cast<const T *>(this)->AreaX1(bnds.x1[0]),
            static_cast<const T *>(this)->AreaX1(bnds.x1[1])};
  }
  KOKKOS_INLINE_FUNCTION std::array<Real, 2> GetFaceAreaX2() const {
    // Get the lower and upper face areas in the X2 direction
    return {static_cast<const T *>(this)->AreaX2(bnds.x2[0]),
            static_cast<const T *>(this)->AreaX2(bnds.x2[1])};
  }
  KOKKOS_INLINE_FUNCTION std::array<Real, 2> GetFaceAreaX3() const {
    // Get the lower and upper face areas in the X3 direction
    return {static_cast<const T *>(this)->AreaX3(bnds.x3[0]),
            static_cast<const T *>(this)->AreaX3(bnds.x3[1])};
  }

  template <parthenon::CoordinateDirection XDIR>
  KOKKOS_INLINE_FUNCTION Real GetFaceArea() const {
    // Get the face area in the direction asked for
    if constexpr (XDIR == parthenon::CoordinateDirection::X1DIR) {
      return static_cast<const T *>(this)->AreaX1(bnds.x1[0]);
    } else if constexpr (XDIR == parthenon::CoordinateDirection::X2DIR) {
      return static_cast<const T *>(this)->AreaX2(bnds.x2[0]);
    } else if constexpr (XDIR == parthenon::CoordinateDirection::X3DIR) {
      return static_cast<const T *>(this)->AreaX3(bnds.x3[0]);
    }
    PARTHENON_FAIL("Bad GetFaceArea XDIR");
    return 0.0;
  }

  KOKKOS_INLINE_FUNCTION Real Distance(const std::array<Real, 3> &x1,
                                       const std::array<Real, 3> &x2) const {
    // { dh1/dx1, dh2/dx1, dh3/dx1 }
    const auto &xc1 = static_cast<const T *>(this)->ConvertToCart(x1);
    const auto &xc2 = static_cast<const T *>(this)->ConvertToCart(x2);
    return std::sqrt(SQR(xc1[0] - xc2[0]) + SQR(xc1[1] - xc2[1]) + SQR(xc1[2] - xc2[2]));
  }

  KOKKOS_INLINE_FUNCTION Mat3x2 GetRFWeights() const {
    return static_cast<const T *>(this)->RFWeights();
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> GetConnX1() const {
    // { dh1/dx1, dh2/dx1, dh3/dx1 }
    return {static_cast<const T *>(this)->dh1dx1(),
            static_cast<const T *>(this)->dh2dx1(),
            static_cast<const T *>(this)->dh3dx1()};
  }
  KOKKOS_INLINE_FUNCTION std::array<Real, 3> GetConnX2() const {
    // { dh1/dx2, dh2/dx2, dh3/dx2 }
    return {static_cast<const T *>(this)->dh1dx2(),
            static_cast<const T *>(this)->dh2dx2(),
            static_cast<const T *>(this)->dh3dx2()};
  }
  KOKKOS_INLINE_FUNCTION std::array<Real, 3> GetConnX3() const {
    // { dh1/dx3, dh2/dx3, dh3/dx3 }
    return {static_cast<const T *>(this)->dh1dx3(),
            static_cast<const T *>(this)->dh2dx3(),
            static_cast<const T *>(this)->dh3dx3()};
  }

  KOKKOS_INLINE_FUNCTION Mat3x3 GetConns() const {
    // Return all of the connections
    return {static_cast<const T *>(this)->GetConnX1(),
            static_cast<const T *>(this)->GetConnX2(),
            static_cast<const T *>(this)->GetConnX3()};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertToCart(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to Cartesian
    return static_cast<const T *>(this)->ConvertCoordsToCart(xi);
  }

  KOKKOS_INLINE_FUNCTION Mat4x3
  ConvertToCartWithVec(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to Cartesian

    const auto &[ex1, ex2, ex3] = static_cast<const T *>(this)->ConvertVecToCart(xi);
    const auto &xo = static_cast<const T *>(this)->ConvertCoordsToCart(xi);
    return {xo, ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertToAxi(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to Axisymmetric
    return static_cast<const T *>(this)->ConvertCoordsToAxi(xi);
  }

  KOKKOS_INLINE_FUNCTION Mat4x3 ConvertToAxiWithVec(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to Axisymmetric

    const auto &[ex1, ex2, ex3] = static_cast<const T *>(this)->ConvertVecToAxi(xi);
    const auto &xo = static_cast<const T *>(this)->ConvertCoordsToAxi(xi);
    return {xo, ex1, ex2, ex3};
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertToSph(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to spherical
    return static_cast<const T *>(this)->ConvertCoordsToSph(xi);
  }

  KOKKOS_INLINE_FUNCTION Mat4x3 ConvertToSphWithVec(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to spherical

    const auto &[ex1, ex2, ex3] = static_cast<const T *>(this)->ConvertVecToSph(xi);
    const auto &xo = static_cast<const T *>(this)->ConvertCoordsToSph(xi);
    return {xo, ex1, ex2, ex3};
  }
  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  ConvertToCyl(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to cylindrical
    return static_cast<const T *>(this)->ConvertCoordsToCyl(xi);
  }

  KOKKOS_INLINE_FUNCTION Mat4x3 ConvertToCylWithVec(const std::array<Real, 3> &xi) const {
    // Convert the input vector from the problem coordinate system to cylindrical

    const auto &[ex1, ex2, ex3] = static_cast<const T *>(this)->ConvertVecToCyl(xi);
    const auto &xo = static_cast<const T *>(this)->ConvertCoordsToCyl(xi);
    return {xo, ex1, ex2, ex3};
  }
};

//----------------------------------------------------------------------------------------
//! The derived base coordinates class
template <Coordinates GEOM>
class Coords : public CoordsBase<Coords<GEOM>> {
  static constexpr Coordinates Coordinates_type = GEOM;
};

//----------------------------------------------------------------------------------------
//! The derived cartesian specialization
template <>
class Coords<Coordinates::cartesian> : public CoordsBase<Coords<Coordinates::cartesian>> {
 public:
  template <typename PAR>
  KOKKOS_INLINE_FUNCTION Coords(const PAR &cpars, const parthenon::Coordinates_t &pco,
                                const int k, const int j, const int i)
      : CoordsBase<Coords<Coordinates::cartesian>>(cpars, pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords(const bool log, const parthenon::Coordinates_t &pco, const int k, const int j,
         const int i)
      : CoordsBase<Coords<Coordinates::cartesian>>(log, pco, k, j, i) {}
  KOKKOS_INLINE_FUNCTION
  Coords() : CoordsBase<Coords<Coordinates::cartesian>>() {}
  template <typename PAR>
  Coords(const PAR &cpars) : CoordsBase<Coords<Coordinates::cartesian>>(cpars) {}
};

} // namespace geometry

#include "axisymmetric.hpp"
#include "cylindrical.hpp"
#include "spherical.hpp"

#endif // GEOMETRY_GEOMETRY_HPP_
