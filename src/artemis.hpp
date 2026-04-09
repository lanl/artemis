//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
ARTEMIS_VARIABLE(gas.cons, Bfield); // YH: add this variable for magnetic flux 
ARTEMIS_VARIABLE(gas.cons, Efield); // YH: for ease i start with directly interpolate to center
ARTEMIS_VARIABLE(gas.cons, J);
ARTEMIS_VARIABLE(gas.cons, Se); // YH: electron entropy density
ARTEMIS_VARIABLE(gas.cons, divB);
ARTEMIS_VARIABLE(gas.cons, divE);
} // namespace cons
namespace prim {
ARTEMIS_VARIABLE(gas.prim, density);
ARTEMIS_VARIABLE(gas.prim, pressure);
ARTEMIS_VARIABLE(gas.prim, velocity);
ARTEMIS_VARIABLE(gas.prim, sie);
ARTEMIS_VARIABLE(gas.prim, Bfield);
ARTEMIS_VARIABLE(gas.prim, Efield); // YH: for ease i start with directly interpolate to center
ARTEMIS_VARIABLE(gas.prim, J);
ARTEMIS_VARIABLE(gas.prim, Pe); // YH: electron pressure
ARTEMIS_VARIABLE(gas.prim, Te);
ARTEMIS_VARIABLE(gas.prim, Ti);
} // namespace prim
namespace diff {
ARTEMIS_VARIABLE(gas.diff, momentum);
ARTEMIS_VARIABLE(gas.diff, energy);
} // namespace diff
namespace face {
ARTEMIS_VARIABLE(gas.face, velocity);
ARTEMIS_VARIABLE(gas.face, bfield); // YH: add MHD
//ARTEMIS_VARIABLE(gas.face, lambdaL); // YH: for upw CT (Mignone 2020)
//ARTEMIS_VARIABLE(gas.face, lambdaR);
//ARTEMIS_VARIABLE(gas.face, velL); 
//ARTEMIS_VARIABLE(gas.face, velR);
}
namespace edge { // YH: cell vertices variables
ARTEMIS_VARIABLE(gas.edge, Efield);
ARTEMIS_VARIABLE(gas.edge, J);
ARTEMIS_VARIABLE(gas.edge, E_flux); // Eflux at edge not node as it is not divergence!
}
namespace node {
ARTEMIS_VARIABLE(gas.node, E_flux); // Try Eflux from RS to correct asymmetry
ARTEMIS_VARIABLE(gas.node, J_flux);
ARTEMIS_VARIABLE(gas.node, EJ_src); // Another variant of EJSource update at node instead
}
namespace source { 
ARTEMIS_VARIABLE(gas.source, expEJ_E);
ARTEMIS_VARIABLE(gas.source, expEJ_J);
// YH: intermediate variables to help with coding
ARTEMIS_VARIABLE(gas.source, nonideal_fcc);
ARTEMIS_VARIABLE(gas.source, nonideal_edge);
ARTEMIS_VARIABLE(gas.source, nonideal_fcc1);
ARTEMIS_VARIABLE(gas.source, nonideal_edge1);
ARTEMIS_VARIABLE(gas.source, relativis_edge);
ARTEMIS_VARIABLE(gas.source, relativis_fcc);

// Store source to update
ARTEMIS_VARIABLE(gas.source, mom);
ARTEMIS_VARIABLE(gas.source, ener);
ARTEMIS_VARIABLE(gas.source, Se);
ARTEMIS_VARIABLE(gas.source, Bfield);
}
// YH: for evolved boundary
namespace boundary {
ARTEMIS_VARIABLE(gas.boundary, density);
ARTEMIS_VARIABLE(gas.boundary, pressure);
ARTEMIS_VARIABLE(gas.boundary, velocity);
ARTEMIS_VARIABLE(gas.boundary, Bfield);
ARTEMIS_VARIABLE(gas.boundary, Bfield_fcc);
ARTEMIS_VARIABLE(gas.boundary, Efield);
ARTEMIS_VARIABLE(gas.boundary, J);
ARTEMIS_VARIABLE(gas.boundary, Pe);
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
#undef ARTEMIS_VARIABLE

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
// YH: add TVD type
enum class TVDType {
  Minmod,   // Most Diffusive
  VanAlbada,
  Original, // Van Leer
  MC,
  Superbee  // Least Diffusive
};
// YH: For resistivity model type
enum class EtaType {
  None,
  IdealMHD,
  Spitzer
};
// ...Riemann solvers
enum class RSolver { hllc, hlle, llf, hlld, llf_xmhd, hlld_xmhd, llf_hall_xmhd, hll_hall_xmhd, hlle_hall_xmhd, hlldc_llf_hall_xmhd, hlldc_hall_xmhd, hlldc_xmhd, hlldc_llf_xmhd, null };
// ... Upwinding (left vs right state)
enum class Upwind { l, r, null };
// ...Reconstrution algorithms
enum class ReconstructionMethod { pcm, plm, ppm, Bcorrection,  
	plm_rho,   // Density-dependent slope limiter
	plm_pp,    // Positive-preserving 
	plm_modPe, // Fixed Superbee to Pe reconstruction
	null };
// ...Fluid types
enum class Fluid { gas, dust, null };
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
  sym,
  conducting,
  none
};

// Floating point limits
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto Big() {
  return std::numeric_limits<T>::max();
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

inline int ProblemDimension(parthenon::ParameterInput *pin) {
  const int nx[3] = {pin->GetInteger("parthenon/mesh", "nx1"),
                     pin->GetInteger("parthenon/mesh", "nx2"),
                     pin->GetInteger("parthenon/mesh", "nx3")};

  return (nx[0] > 1) + (nx[1] > 1) + (nx[2] > 1);
}

namespace artemis {
extern std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock;
} // namespace artemis

#endif // ARTEMIS_ARTEMIS_HPP_
