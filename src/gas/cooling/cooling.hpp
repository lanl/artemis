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
#ifndef GAS_COOLING_COOLING_HPP_
#define GAS_COOLING_COOLING_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"

using namespace parthenon::package::prelude;

namespace Gas {
namespace Cooling {

enum class CoolingType { beta, null };
enum class TempRefType { powerlaw, nbody, null };

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM, TempRefType T>
TaskStatus BetaCooling(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
TaskStatus CoolingSource(MeshData<Real> *md, const Real time, const Real dt);

struct TempParams {
  Real tcyl;
  Real tsph;
  Real cyl_plaw;
  Real sph_plaw;
  Real tfloor;
};

template <Coordinates GEOM, TempRefType T>
KOKKOS_INLINE_FUNCTION Real TemperatureProfile(geometry::Coords<GEOM> &coords, Real t,
                                               Real xv[3], TempParams pars) {
  if constexpr (T == TempRefType::powerlaw) {
    Real xcyl[3] = {Null<Real>()};
    Real xsph[3] = {Null<Real>()};
    coords.ConvertToCyl(xv, xcyl);
    coords.ConvertToSph(xv, xsph);
    return pars.tfloor + pars.tcyl * std::pow(xcyl[0], pars.cyl_plaw) +
           pars.tsph * std::pow(xsph[0], pars.sph_plaw);
  }
  return 0.0;
}

} // namespace Cooling
} // namespace Gas

#endif // GAS_COOLING_COOLING_HPP_
