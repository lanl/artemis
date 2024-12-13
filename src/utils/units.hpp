//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_UNITS_HPP_
#define UTILS_UNITS_HPP_

#include "artemis.hpp"

#include <cmath>

namespace ArtemisUtils {

class Units {
 public:
  // Host-only constructor from parameter input file
  Units(ParameterInput *pin, std::shared_ptr<StateDescriptor> pkg);

  // Copy constructor must be marked with KOKKOS_FUNCTION
  KOKKOS_FUNCTION
  Units(const Units &other) {}

  // Unit conversions
  KOKKOS_INLINE_FUNCTION
  Real GetLengthCodeToPhysical() const { return length_; }
  KOKKOS_INLINE_FUNCTION
  Real GetLengthPhysicalToCode() const { return 1. / length_; }

  KOKKOS_INLINE_FUNCTION
  Real GetTimeCodeToPhysical() const { return time_; }
  KOKKOS_INLINE_FUNCTION
  Real GetTimePhysicalToCode() const { return 1. / time_; }

  KOKKOS_INLINE_FUNCTION
  Real GetMassCodeToPhysical() const { return mass_; }
  KOKKOS_INLINE_FUNCTION
  Real GetMassPhysicalToCode() const { return 1. / mass_; }

  KOKKOS_INLINE_FUNCTION
  Real GetEnergyCodeToPhysical() const { return energy_; }
  KOKKOS_INLINE_FUNCTION
  Real GetEnergyPhysicalToCode() const { return 1. / energy_; }

  KOKKOS_INLINE_FUNCTION
  Real GetNumberDensityCodeToPhysical() const { return number_density_; }
  KOKKOS_INLINE_FUNCTION
  Real GetNumberDensityPhysicalToCode() const { return 1. / number_density_; }

  KOKKOS_INLINE_FUNCTION
  Real GetMassDensityCodeToPhysical() const { return mass_density_; }
  KOKKOS_INLINE_FUNCTION
  Real GetMassDensityPhysicalToCode() const { return 1. / mass_density_; }

  KOKKOS_INLINE_FUNCTION
  Real GetTemperatureCodeToPhysical() const { return temperature_; }
  KOKKOS_INLINE_FUNCTION
  Real GetTemperaturePhysicalToCode() const { return 1. / temperature_; }

  // Physical constants
  KOKKOS_INLINE_FUNCTION
  Real GetGPhysical() const { return G_; }
  KOKKOS_INLINE_FUNCTION
  Real GetGCode() const { return G_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetKBPhysical() const { return kb_; }
  KOKKOS_INLINE_FUNCTION
  Real GetKBCode() const { return kb_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetCPhysical() const { return c_; }
  KOKKOS_INLINE_FUNCTION
  Real GetCCode() const { return c_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetHPhysical() const { return h_; }
  KOKKOS_INLINE_FUNCTION
  Real GetHCode() const { return h_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetMsolarPhysical() const { return Msolar_; }
  KOKKOS_INLINE_FUNCTION
  Real GetMsolarCode() const { return Msolar_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetAUPhysical() const { return AU_; }
  KOKKOS_INLINE_FUNCTION
  Real GetAUCode() const { return AU_code_; }

 private:
  // Unit conversion factors from code to physical units
  // e.g. length_ has units of cm when using CGS as physical unit system
  Real length_;
  Real time_;
  Real mass_;

  Real energy_;
  Real number_density_;
  Real mass_density_;
  Real temperature_;

  // Physical constants in physical units
  Real G_;      // Gravitational constant
  Real kb_;     // Boltzmann constant
  Real c_;      // Speed of light
  Real h_;      // Planck constant
  Real Msolar_; // Solar mass
  Real AU_;     // Astronomical unit

  // Physical constants in code units
  Real G_code_;
  Real kb_code_;
  Real c_code_;
  Real h_code_;
  Real Msolar_code_;
  Real AU_code_;
};

} // namespace ArtemisUtils

#endif // UTILS_UNITS_HPP_
