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

#include "units.hpp"

namespace ArtemisUtils {

Units::Units(ParameterInput *pin) {
  unit_system_ = pin->GetOrAddString("artemis/units", "type", "code");
  if (unit_system_ == "code") {
    length_ = 1.;
    time_ = 1.;
    mass_ = 1.;
    G_ = 1.;
    kb_ = 1.;
    c_ = 1.;
    h_ = 1.;
    Msolar_ = 1.;
    AU_ = 1.;
  } else {
    if (unit_system_ == "cgs") {
      parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;
      G_ = pc.gravitational_constant;
      kb_ = pc.kb;
      c_ = pc.c;
      h_ = pc.h;
      Msolar_ = 1.988416e33;
      AU_ = 1.495978707e13;
    } else {
      PARTHENON_FAIL("Unit system not recognized!");
    }
  }

  // Custom physical constants
  // Either for specific numerical values for fine-grained consistency, or non-physical
  // units for e.g. radiation shocktube test problems
  if (pin->DoesParameterExist("artemis/units/constants", "G")) {
    G_ = pin->GetReal("artemis/units/constants", "G");
  }
  if (pin->DoesParameterExist("artemis/units/constants", "kb")) {
    kb_ = pin->GetReal("artemis/units/constants", "kb");
  }
  if (pin->DoesParameterExist("artemis/units/constants", "c")) {
    c_ = pin->GetReal("artemis/units/constants", "c");
  }
  if (pin->DoesParameterExist("artemis/units/constants", "h")) {
    h_ = pin->GetReal("artemis/units/constants", "h");
  }
  if (pin->DoesParameterExist("artemis/units/constants", "Msolar")) {
    Msolar_ = pin->GetReal("artemis/units/constants", "Msolar");
  }
  if (pin->DoesParameterExist("artemis/units/constants", "AU")) {
    AU_ = pin->GetReal("artemis/units/constants", "AU");
  }

  // Convert constants to code units
  G_code_ = G_ * std::pow(length_, -3) / mass_ * std::pow(time_, 2);
  kb_code_ = std::pow(time_, 2) / mass_ * std::pow(length_, -2); // 1 K = 1 code unit temp
  c_code_ = c_ * time_ / length_;
  h_code_ = h_ * time_ / mass_ * std::pow(length_, -2);
  Msolar_code_ = Msolar_ / mass_;
  AU_code_ = AU_ / length_;

  if (unit_system_ != "code") {

    unit_specifier_ = pin->GetOrAddString("artemis/units", "specifier", "base");
    if (unit_specifier_ == "base") {
      length_ = pin->GetReal("artemis/units", "length");
      time_ = pin->GetReal("artemis/units", "time");
      mass_ = pin->GetReal("artemis/units", "mass");
    } else if (unit_specifier_ == "orbit") {
      const Real Mstar = pin->GetReal("artemis/units", "star_mass");     // Solar masses
      const Real Rorbit = pin->GetReal("artemis/units", "orbit_radius"); // AU
      mass_ = Mstar * Msolar_;
      length_ = Rorbit * AU_;
      time_ = std::sqrt(4. * M_PI * M_PI / (G_ * mass_) *
                        std::pow(length_, 3)); // Orbital period
    } else {
      PARTHENON_FAIL("Unit specifier not recognized!");
    }
  }

  // Remaining conversion factors
  energy_ = std::pow(length_, 2) * mass_ * std::pow(time_, -2);
  number_density_ = std::pow(length_, -3);
  mass_density_ = mass_ * number_density_;
  temperature_ = 1.;
}

} // namespace ArtemisUtils
