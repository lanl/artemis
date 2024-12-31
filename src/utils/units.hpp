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

enum class PhysicalUnits { scalefree, cgs };

class Units {
 public:
  // No default constructor
  Units() = delete;
  // Host-only constructor from parameter input file
  Units(ParameterInput *pin, std::shared_ptr<StateDescriptor> pkg);

  // Copy constructor must be marked with KOKKOS_FUNCTION
  KOKKOS_FUNCTION
  Units(const Units &other) = default;

  // Return physical unit system
  KOKKOS_INLINE_FUNCTION
  PhysicalUnits GetPhysicalUnits() const { return physical_units_; }

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
  Real GetSpeedCodeToPhysical() const { return length_ / time_; }
  KOKKOS_INLINE_FUNCTION
  Real GetSpeedPhysicalToCode() const { return time_ / mass_; }

  KOKKOS_INLINE_FUNCTION
  Real GetEnergyCodeToPhysical() const { return energy_; }
  KOKKOS_INLINE_FUNCTION
  Real GetEnergyPhysicalToCode() const { return 1. / energy_; }

  KOKKOS_INLINE_FUNCTION
  Real GetNumberDensityCodeToPhysical() const { return number_density_; }
  KOKKOS_INLINE_FUNCTION
  Real GetNumberDensityPhysicalToCode() const { return 1. / number_density_; }

  KOKKOS_INLINE_FUNCTION
  Real GetEnergyDensityCodeToPhysical() const { return energy_ * number_density_; }
  KOKKOS_INLINE_FUNCTION
  Real GetEnergyDensityPhysicalToCode() const { return 1. / (energy_ * number_density_); }

  KOKKOS_INLINE_FUNCTION
  Real GetMassDensityCodeToPhysical() const { return mass_ * number_density_; }
  KOKKOS_INLINE_FUNCTION
  Real GetMassDensityPhysicalToCode() const { return 1. / (mass_ * number_density_); }

  KOKKOS_INLINE_FUNCTION
  Real GetOpacityCodeToPhysical() const { return length_ * length_ / mass_; }
  KOKKOS_INLINE_FUNCTION
  Real GetOpacityPhysicalToCode() const { return mass_ / (length_ * length_); }

  KOKKOS_INLINE_FUNCTION
  Real GetSpecificHeatCodeToPhysical() const { return energy_ / mass_; }
  KOKKOS_INLINE_FUNCTION
  Real GetSpecificHeatPhysicalToCode() const { return mass_ / energy_; }

 private:
  // Unit conversion factors from code to physical units
  // e.g. length_ has units of cm when using CGS as physical unit system
  // Temperature is always Kelvin
  Real length_;
  Real time_;
  Real mass_;

  Real energy_;
  Real number_density_;

  PhysicalUnits physical_units_;
};

class Constants {
 public:
  Constants() = delete;

  KOKKOS_FUNCTION
  Constants(Units &units);

  KOKKOS_FUNCTION
  Constants(const Constants &other) {}

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
  Real GetARPhysical() const { return ar_; }
  KOKKOS_INLINE_FUNCTION
  Real GetARCode() const { return ar_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetAMUPhysical() const { return amu_; }
  KOKKOS_INLINE_FUNCTION
  Real GetAMUCode() const { return amu_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetMsolarPhysical() const { return Msolar_; }
  KOKKOS_INLINE_FUNCTION
  Real GetMsolarCode() const { return Msolar_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetAUPhysical() const { return AU_; }
  KOKKOS_INLINE_FUNCTION
  Real GetAUCode() const { return AU_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetPcPhysical() const { return pc_; }
  KOKKOS_INLINE_FUNCTION
  Real GetPcCode() const { return pc_code_; }

  KOKKOS_INLINE_FUNCTION
  Real GetYearPhysical() const { return Year_; }
  KOKKOS_INLINE_FUNCTION
  Real GetYearCode() const { return Year_code_; }

 private:
  // Physical constants in physical units
  Real G_;      // Gravitational constant
  Real kb_;     // Boltzmann constant
  Real c_;      // Speed of light
  Real h_;      // Planck constant
  Real ar_;     // Radiation constant
  Real amu_;    // Atomic mass unit
  Real eV_;     // Electron-volt
  Real Msolar_; // Solar mass
  Real AU_;     // Astronomical unit
  Real Rjup_;   // Jupiter radius
  Real Mjup_;   // Jupiter mass
  Real pc_;     // Parsec
  Real Year_;   // Year

  // Physical constants in code units
  Real G_code_;
  Real kb_code_;
  Real c_code_;
  Real h_code_;
  Real ar_code_;
  Real amu_code_;
  Real eV_code_;
  Real Msolar_code_;
  Real AU_code_;
  Real Rjup_code_;
  Real Mjup_code_;
  Real pc_code_;
  Real Year_code_;
};

} // namespace ArtemisUtils

#endif // UTILS_UNITS_HPP_
