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
#define ARTEMIS_VARIABLE(ns, varname)                                                    \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }
namespace gas {
namespace cons {
ARTEMIS_VARIABLE(gas.cons, density);
ARTEMIS_VARIABLE(gas.cons, total_energy);
ARTEMIS_VARIABLE(gas.cons, internal_energy);
ARTEMIS_VARIABLE(gas.cons, momentum);
} // namespace cons
namespace prim {
ARTEMIS_VARIABLE(gas.prim, density);
ARTEMIS_VARIABLE(gas.prim, pressure);
ARTEMIS_VARIABLE(gas.prim, temperature);
ARTEMIS_VARIABLE(gas.prim, velocity);
ARTEMIS_VARIABLE(gas.prim, sie);
ARTEMIS_VARIABLE(gas.prim, bmod);
} // namespace prim
namespace diff {
ARTEMIS_VARIABLE(gas.diff, momentum);
ARTEMIS_VARIABLE(gas.diff, energy);
} // namespace diff
namespace face {
ARTEMIS_VARIABLE(gas.face, velocity);
} // namespace face
namespace src {
ARTEMIS_VARIABLE(gas.src, energy);
}
} // namespace gas

namespace dust {
namespace cons {
ARTEMIS_VARIABLE(dust.cons, density);
ARTEMIS_VARIABLE(dust.cons, momentum);
} // namespace cons
namespace prim {
ARTEMIS_VARIABLE(dust.prim, density);
ARTEMIS_VARIABLE(dust.prim, velocity);
} // namespace prim
} // namespace dust

namespace rad {
namespace cons {
ARTEMIS_VARIABLE(rad.cons, energy);
ARTEMIS_VARIABLE(rad.cons, flux);
} // namespace cons
namespace prim {
ARTEMIS_VARIABLE(rad.prim, energy);
ARTEMIS_VARIABLE(rad.prim, pressure);
ARTEMIS_VARIABLE(rad.prim, flux);
} // namespace prim
namespace opac {
ARTEMIS_VARIABLE(rad.opac, absorption);
ARTEMIS_VARIABLE(rad.opac, scattering);
} // namespace opac
namespace star {
ARTEMIS_VARIABLE(rad.star, absorption);
SWARM_VARIABLE(Real, rad.star, flux);
SWARM_VARIABLE(Real, rad.star, v);
SWARM_VARIABLE(Real, rad.star, x);
SWARM_VARIABLE(int, rad.star, ijk);
} // namespace star
} // namespace rad

namespace grav {
ARTEMIS_VARIABLE(grav, phi);
ARTEMIS_VARIABLE(grav, rhs);
} // namespace grav

namespace geom {
ARTEMIS_VARIABLE(geom, x1v);
ARTEMIS_VARIABLE(geom, x2v);
ARTEMIS_VARIABLE(geom, x3v);
ARTEMIS_VARIABLE(geom, hx1v);
ARTEMIS_VARIABLE(geom, hx2v);
ARTEMIS_VARIABLE(geom, hx3v);
ARTEMIS_VARIABLE(geom, hx1f1);
ARTEMIS_VARIABLE(geom, hx2f1);
ARTEMIS_VARIABLE(geom, hx3f1);
ARTEMIS_VARIABLE(geom, hx1f2);
ARTEMIS_VARIABLE(geom, hx2f2);
ARTEMIS_VARIABLE(geom, hx3f2);
ARTEMIS_VARIABLE(geom, hx1f3);
ARTEMIS_VARIABLE(geom, hx2f3);
ARTEMIS_VARIABLE(geom, hx3f3);
ARTEMIS_VARIABLE(geom, dx1);
ARTEMIS_VARIABLE(geom, dx2);
ARTEMIS_VARIABLE(geom, dx3);
ARTEMIS_VARIABLE(geom, vol);
ARTEMIS_VARIABLE(geom, ax1);
ARTEMIS_VARIABLE(geom, ax2);
ARTEMIS_VARIABLE(geom, ax3);
ARTEMIS_VARIABLE(geom, dh1dx1);
ARTEMIS_VARIABLE(geom, dh2dx1);
ARTEMIS_VARIABLE(geom, dh3dx1);
ARTEMIS_VARIABLE(geom, dh1dx2);
ARTEMIS_VARIABLE(geom, dh2dx2);
ARTEMIS_VARIABLE(geom, dh3dx2);
ARTEMIS_VARIABLE(geom, dh1dx3);
ARTEMIS_VARIABLE(geom, dh2dx3);
ARTEMIS_VARIABLE(geom, dh3dx3);
ARTEMIS_VARIABLE(geom, rfw1m);
ARTEMIS_VARIABLE(geom, rfw1p);
ARTEMIS_VARIABLE(geom, rfw2m);
ARTEMIS_VARIABLE(geom, rfw2p);
ARTEMIS_VARIABLE(geom, rfw3m);
ARTEMIS_VARIABLE(geom, rfw3p);
} // namespace geom

#undef ARTEMIS_VARIABLE

// Restart options (see Parthenon #1231)
#ifdef PORTABLE_RESTART
using BYTE = uint8_t;
#else
using BYTE = char;
#endif

// TaskCollection function pointer for operator split tasks
using TaskCollectionFnPtr = TaskCollection (*)(Mesh *pm, const Real time, const Real dt);

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
