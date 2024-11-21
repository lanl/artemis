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
ARTEMIS_VARIABLE(gas.prim, velocity);
ARTEMIS_VARIABLE(gas.prim, sie);
} // namespace prim
namespace diff {
ARTEMIS_VARIABLE(gas.diff, momentum);
ARTEMIS_VARIABLE(gas.diff, energy);
} // namespace diff
namespace face {
ARTEMIS_VARIABLE(gas.face, velocity);
} // namespace face
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
ARTEMIS_VARIABLE(dust, stopping_time);
} // namespace dust
#undef ARTEMIS_VARIABLE

// TaskCollection function pointer for operator split tasks
using TaskCollectionFnPtr = TaskCollection (*)(Mesh *pm, parthenon::SimTime &tm,
                                               const Real dt);

// Constants that enumerate...
// ...Coordinate systems
enum class Coordinates { cartesian, cylindrical, spherical, axisymmetric, null };
// ...Riemann solvers
enum class RSolver { hllc, hlle, llf, null };
// ...Reconstrution algorithms
enum class ReconstructionMethod { pcm, plm, ppm, null };
// ...Fluid types
enum class Fluid { gas, dust, null };
// ...Boundary conditions
// constants that enumerate dust drag method
enum class DragMethod {
  explicitNoFeedback,
  explicitFeedback,
  implicitNoFeedback,
  implicitFeedback,
  null
};
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

// Floating point limits
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Big() {
  return std::numeric_limits<T>::max();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Tiny() {
  return std::numeric_limits<T>::min();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Fuzz() {
  return std::numeric_limits<T>::epsilon();
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

namespace artemis {
extern std::vector<TaskCollectionFnPtr> OperatorSplitTasks;
extern std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock;
} // namespace artemis

#endif // ARTEMIS_ARTEMIS_HPP_
