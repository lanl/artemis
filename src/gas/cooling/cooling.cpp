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

// Artemis includes
#include "gas/cooling/cooling.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"

using namespace parthenon::package::prelude;

namespace Gas {
namespace Cooling {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Cooling::Initialize
//! \brief Adds intialization function for cooling package
//!
//!   Beta Cooling:
//!     dT/dt = (T - Tref)/tc
//!     with t_c Omega(R,z) = beta(R,z)
//!     beta = beta_min + beta0 *exp(-b*z^2 / Tref)
//!
//!   Power-law reference temperature:
//!     Tref = Tcyl*R^a  + Tsph * r^b
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto cooling = std::make_shared<StateDescriptor>("cooling");
  Params &params = cooling->AllParams();

  std::string cool_type = pin->GetString("cooling", "type");
  CoolingType ctype = CoolingType::null;
  if (cool_type == "beta") {
    ctype = CoolingType::beta;
    const Real beta0 = pin->GetReal("cooling", "beta0");
    const Real beta_min = pin->GetOrAddReal("cooling", "beta_min", 1e-12);
    const Real escale = pin->GetOrAddReal("cooling", "exp_scale", 0.0);
    params.Add("beta0", beta0);
    params.Add("beta_min", beta_min);
    params.Add("escale", escale);
  } else {
    PARTHENON_FAIL("Unknown cooling type");
  }
  params.Add("type", ctype);

  std::string temp_type = pin->GetString("cooling", "tref");
  TempRefType ttype = TempRefType::null;
  TempParams tpars;
  tpars.tfloor = pin->GetOrAddReal("cooling", "tfloor", 0.0);
  if (temp_type == "powerlaw") {
    ttype = TempRefType::powerlaw;
    tpars.tcyl = pin->GetOrAddReal("cooling", "tcyl", 0.0);
    tpars.cyl_plaw = pin->GetOrAddReal("cooling", "cyl_plaw", 0.0);
    tpars.tsph = pin->GetOrAddReal("cooling", "tsph", 0.0);
    tpars.sph_plaw = pin->GetOrAddReal("cooling", "sph_plaw", 0.0);
    if (tpars.tcyl * tpars.tsph != 0.0) {
      PARTHENON_WARN("Warning: you have cooling/tcyl and cooling/tsph not equal to zero. "
                     "Was this intended?");
    }
  } else if (temp_type == "nbody") {
    ttype = TempRefType::nbody;
    tpars.tcyl = pin->GetOrAddReal("cooling", "tcyl", 0.0);
    tpars.cyl_plaw = pin->GetOrAddReal("cooling", "cyl_plaw", 0.0);
    tpars.tsph = pin->GetOrAddReal("cooling", "tsph", 0.0);
    tpars.sph_plaw = pin->GetOrAddReal("cooling", "sph_plaw", 0.0);
    if (tpars.tcyl * tpars.tsph != 0.0) {
      PARTHENON_WARN("Warning: you have cooling/tcyl and cooling/tsph not equal to zero. "
                     "Was this intended?");
    }
  } else {
    PARTHENON_FAIL("Unknown cooling reference temperature");
  }
  params.Add("ttype", ttype);
  params.Add("tpars", tpars);

  return cooling;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Cooling::CoolingSource
//! \brief Wrapper function for external cooling options
template <Coordinates GEOM>
TaskStatus CoolingSource(MeshData<Real> *md, const Real time, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &pkg = pm->packages.Get("cooling");
  CoolingType ctype = pkg->template Param<CoolingType>("type");
  TempRefType ttype = pkg->template Param<TempRefType>("ttype");
  if (ctype == CoolingType::beta) {
    if (ttype == TempRefType::powerlaw) {
      return BetaCooling<GEOM, TempRefType::powerlaw>(md, time, dt);
    } else if (ttype == TempRefType::nbody) {
      return BetaCooling<GEOM, TempRefType::nbody>(md, time, dt);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> MD;
template TaskStatus CoolingSource<C::cartesian>(MD *md, const Real t, const Real d);
template TaskStatus CoolingSource<C::cylindrical>(MD *md, const Real t, const Real d);
template TaskStatus CoolingSource<C::spherical3D>(MD *md, const Real t, const Real d);
template TaskStatus CoolingSource<C::spherical1D>(MD *md, const Real t, const Real d);
template TaskStatus CoolingSource<C::spherical2D>(MD *md, const Real t, const Real d);
template TaskStatus CoolingSource<C::axisymmetric>(MD *md, const Real t, const Real d);

} // namespace Cooling
} // namespace Gas
