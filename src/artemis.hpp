//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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

#ifndef ARTEMIS_ARTEMIS_HPP_
#define ARTEMIS_ARTEMIS_HPP_

// C++ includes
#include <limits>
#include <string>

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Singularity-eos includes
#include <singularity-eos/eos/eos.hpp>

using namespace parthenon;
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Create variable types to be used by Artemis

namespace gas {
namespace cons {
PAR_VAR(gas.cons, density);
PAR_VAR(gas.cons, total_energy);
PAR_VAR(gas.cons, internal_energy);
PAR_VAR(gas.cons, momentum);
} // namespace cons
namespace prim {
PAR_VAR(gas.prim, density);
PAR_VAR(gas.prim, pressure);
PAR_VAR(gas.prim, temperature);
PAR_VAR(gas.prim, velocity);
PAR_VAR(gas.prim, sie);
PAR_VAR(gas.prim, bmod);
} // namespace prim
namespace diff {
PAR_VAR(gas.diff, momentum);
PAR_VAR(gas.diff, energy);
} // namespace diff
namespace face {
PAR_VAR(gas.face, velocity);
} // namespace face
namespace src {
PAR_VAR(gas.src, energy);
}
} // namespace gas

namespace dust {
namespace cons {
PAR_VAR(dust.cons, density);
PAR_VAR(dust.cons, momentum);
} // namespace cons
namespace prim {
PAR_VAR(dust.prim, density);
PAR_VAR(dust.prim, velocity);
} // namespace prim
} // namespace dust

namespace rad {
namespace cons {
PAR_VAR(rad.cons, energy);
PAR_VAR(rad.cons, flux);
} // namespace cons
namespace prim {
PAR_VAR(rad.prim, energy);
PAR_VAR(rad.prim, pressure);
PAR_VAR(rad.prim, flux);
} // namespace prim
namespace opac {
PAR_VAR(rad.opac, absorption);
PAR_VAR(rad.opac, scattering);
} // namespace opac
namespace star {
PAR_VAR(rad.star, absorption);
PAR_SWARMVAR(Real, rad.star, flux);
PAR_SWARMVAR(Real, rad.star, v);
PAR_SWARMVAR(Real, rad.star, x);
PAR_SWARMVAR(int, rad.star, ijk);
} // namespace star
} // namespace rad

namespace grav {
PAR_VAR(grav, phi);
PAR_VAR(grav, rhs);
} // namespace grav

namespace geom {
PAR_VAR(geom, x1v);
PAR_VAR(geom, x2v);
PAR_VAR(geom, x3v);
PAR_VAR(geom, hx1v);
PAR_VAR(geom, hx2v);
PAR_VAR(geom, hx3v);
PAR_VAR(geom, hx1f1);
PAR_VAR(geom, hx2f1);
PAR_VAR(geom, hx3f1);
PAR_VAR(geom, hx1f2);
PAR_VAR(geom, hx2f2);
PAR_VAR(geom, hx3f2);
PAR_VAR(geom, hx1f3);
PAR_VAR(geom, hx2f3);
PAR_VAR(geom, hx3f3);
PAR_VAR(geom, dx1);
PAR_VAR(geom, dx2);
PAR_VAR(geom, dx3);
PAR_VAR(geom, vol);
PAR_VAR(geom, ax1);
PAR_VAR(geom, ax2);
PAR_VAR(geom, ax3);
PAR_VAR(geom, dh1dx1);
PAR_VAR(geom, dh2dx1);
PAR_VAR(geom, dh3dx1);
PAR_VAR(geom, dh1dx2);
PAR_VAR(geom, dh2dx2);
PAR_VAR(geom, dh3dx2);
PAR_VAR(geom, dh1dx3);
PAR_VAR(geom, dh2dx3);
PAR_VAR(geom, dh3dx3);
PAR_VAR(geom, rfw1m);
PAR_VAR(geom, rfw1p);
PAR_VAR(geom, rfw2m);
PAR_VAR(geom, rfw2p);
PAR_VAR(geom, rfw3m);
PAR_VAR(geom, rfw3p);
} // namespace geom

// Restart options (see Parthenon #1231)
#ifdef PORTABLE_RESTART
using BYTE = uint8_t;
#else
using BYTE = char;
#endif

// Constants that enumerate...
// ...Coordinate systems
enum class Coordinates {
  cartesian,
  cylindrical,
  spherical1D,
  spherical2D,
  spherical3D,
  axisymmetric,
  null
};

// ...Riemann solvers
enum class RSolver { hllc_general, hlle, llf, hllc_gamma, null };
// ... Upwinding (left vs right state)
enum class Upwind { l, r, null };
// ...Reconstruction algorithms
enum class ReconstructionMethod { pcm, plm, ppm, wenoz, wenomz, null };
// ...Fluid types
enum class Fluid { gas, dust, radiation, null };
// ...Closure types
enum class Closure { p1, m1, null };
// ...Boundary conditions
enum class ArtemisBC {
  reflect,
  outflow,
  extrap,
  inflow,
  conduct,
  ic,
  visc,
  user,
  periodic,
  none
};

// Tensor indexing (currently used in radiation moments)
enum TensIdx { X11 = 0, X22 = 1, X33 = 2, X23 = 3, X13 = 4, X12 = 5 };

// Floating point limits
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Big() {
  return std::numeric_limits<T>::max();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Eps() {
  return std::numeric_limits<T>::epsilon();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Tiny() {
  return std::numeric_limits<T>::lowest();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Fuzz() {
  if constexpr (std::is_same_v<T, float>) {
    return 1e-22;
  }
  return 1e-99;
}
// Round-off tolerance: `ulps` multiples of machine epsilon, scaled to the
// magnitude of the quantity being compared (default scale 1). Centralizes the
// `N * Eps() * scale` guards used to compare floating-point values that carry a
// few ULP of accumulated error. `ulps` is a small headroom factor, not a
// physical threshold.
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto RoundoffTol(const T ulps,
                                                       const T scale = T(1)) {
  return ulps * Eps<T>() * scale;
}

// Initialization nulls
static const std::string snull = "UNINITIALIZED STRING";
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Null() {
  return std::numeric_limits<T>::quiet_NaN();
}
template <>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Null<int>() {
  return Big<int>();
}

// Arrays
template <typename T, int N>
KOKKOS_FORCEINLINE_FUNCTION auto NewArray(T val = Null<T>()) {
  std::array<T, N> arr;
  // Note arr.fill() is not __device__
  for (int i = 0; i < N; i++)
    arr[i] = val;
  return arr;
}

// Problem dimensionality (determined from Parameter Input)
inline int ProblemDimension(parthenon::ParameterInput *pin) {
  const int nx[3] = {pin->GetInteger("parthenon/mesh", "nx1"),
                     pin->GetInteger("parthenon/mesh", "nx2"),
                     pin->GetInteger("parthenon/mesh", "nx3")};

  return (nx[0] > 1) + (nx[1] > 1) + (nx[2] > 1);
}

// Custom AMR criteria
namespace artemis {
extern std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock;

} // namespace artemis

#endif // ARTEMIS_ARTEMIS_HPP_
