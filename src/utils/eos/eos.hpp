//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_EOS_EOS_HPP_
#define UTILS_EOS_EOS_HPP_

#include "artemis.hpp"
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/base/root-finding-1d/root_finding.hpp>
#include <singularity-eos/base/spiner_table_utils.hpp>
#include <singularity-eos/eos/eos.hpp>

#include "ideal_h_he.hpp"

namespace ArtemisUtils {

// Maximum size of lambda array for optional extra EOS arguments.
// As it happens this must be >= 1 for device code.
static constexpr int lambda_max_vals = 1;

// Variant containing all EOSs to be used in Artemis.

using EOS =
    singularity::Variant<singularity::UnitSystem<singularity::IdealGas>,
#ifdef SPINER_USE_HDF
                         singularity::UnitSystem<ArtemisEOS::IdealHHe>,
                         singularity::UnitSystem<singularity::SpinerEOSDependsRhoT>,
                         singularity::UnitSystem<singularity::SpinerEOSDependsRhoSie>
#endif
                         >;

// Below are functions that invert the typical EOS calls
//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::TofPR
//! \brief Get temperature from pressure and density
KOKKOS_INLINE_FUNCTION
Real TofPR(const EOS &eos, const Real pres, const Real dens) {
  using RootFinding1D::regula_falsi;
  using RootFinding1D::Status;

  // use units probably
  const Real tmin = 1e-10;
  const Real tmax = 1e10;

  // Root find over temperature matching pressure
  Real temp;
  auto status = regula_falsi(
      [&eos, &dens](const Real t) { return eos.PressureFromDensityTemperature(dens, t); },
      pres, std::sqrt(tmin * tmax), tmin, tmax, 1e-12, 1e-12, temp);
  if (status != Status::SUCCESS) {
    PARTHENON_DEBUG_WARN("TofPR did not converge");
    return Tiny<Real>();
  }
  return temp;
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::EofPR
//! \brief Get specific internal energy from pressure and density
KOKKOS_INLINE_FUNCTION
Real EofPR(const EOS &eos, const Real pres, const Real dens) {
  // first get temperature
  const Real temp = TofPR(eos, pres, dens);
  return eos.InternalEnergyFromDensityTemperature(dens, temp);
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::RofPT
//! \brief Get density from pressure and temperature
KOKKOS_INLINE_FUNCTION
Real RofPT(const EOS &eos, const Real pres, const Real temp) {
  using RootFinding1D::regula_falsi;
  using RootFinding1D::Status;

  // use units probably
  const Real dmin = 1e-10;
  const Real dmax = 1e20;

  // Root find over density matching pressure
  Real dens;
  auto status = regula_falsi(
      [&eos, temp](const Real d) { return eos.PressureFromDensityTemperature(d, temp); },
      pres, std::sqrt(dmin * dmax), dmin, dmax, 1e-12, 1e-12, dens);
  if (status != Status::SUCCESS) {
    PARTHENON_DEBUG_WARN("RofPT did not converge");
    return Tiny<Real>();
  }
  return dens;
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::EofPT
//! \brief Get specific internal energy from pressure and temperature
KOKKOS_INLINE_FUNCTION
Real EofPT(const EOS &eos, const Real pres, const Real temp) {
  // first get density
  const Real dens = RofPT(eos, pres, temp);
  return eos.InternalEnergyFromDensityTemperature(dens, temp);
}

} // namespace ArtemisUtils

#endif // UTILS_EOS_EOS_HPP_
