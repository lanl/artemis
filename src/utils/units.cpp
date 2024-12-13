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

Units::Units(ParameterInput *pin, std::shared_ptr<StateDescriptor> pkg) {
  std::string unit_system_str = pin->GetOrAddString("artemis/units", "type", "scalefree");
  if (unit_system_str == "scalefree") {
    unit_system = UnitSystem::scalefree;
  } else if (unit_system_str == "cgs") {
    unit_system = UnitSystem::cgs;
  } else {
    PARTHENON_FAIL("Unit system not recognized! Valid choices are [scalefree, cgs]");
  }

  if (unit_system == UnitSystem::scalefree) {
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
    if (unit_system == UnitSystem::cgs) {
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

  // Convert constants to code units
  G_code_ = G_ * std::pow(length_, -3) / mass_ * std::pow(time_, 2);
  kb_code_ =
      kb_ * std::pow(time_, 2) / mass_ * std::pow(length_, -2); // 1 K = 1 code unit temp
  c_code_ = c_ * time_ / length_;
  h_code_ = h_ * time_ / mass_ * std::pow(length_, -2);
  Msolar_code_ = Msolar_ / mass_;
  AU_code_ = AU_ / length_;

  if (unit_system != UnitSystem::scalefree) {
    std::string unit_specifier =
        pin->GetOrAddString("artemis/units", "specifier", "base");
    if (unit_specifier == "base") {
      length_ = pin->GetReal("artemis/units", "length");
      time_ = pin->GetReal("artemis/units", "time");
      mass_ = pin->GetReal("artemis/units", "mass");
    } else if (unit_specifier == "orbit") {
      const Real total_mass = pin->GetReal("artemis/units", "total_mass"); // Solar masses
      const Real reference_radius =
          pin->GetReal("artemis/units", "reference_radius"); // AU
      mass_ = total_mass * Msolar_;
      length_ = reference_radius * AU_;
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

  // Store everything necessary in params for usage in analysis
  pkg->AddParam("unit_system", unit_system);
  pkg->AddParam("length", length_);
  pkg->AddParam("time", time_);
  pkg->AddParam("mass", mass_);
  pkg->AddParam("temperature", temperature_);
  pkg->AddParam("G", G_);
  pkg->AddParam("kb", kb_);
  pkg->AddParam("c", c_);
  pkg->AddParam("h", h_);
  pkg->AddParam("Msolar", Msolar_);
  pkg->AddParam("AU", AU_);
}

Constants::Constants(Units &units) {
  if (units.GetUnitSystem() == UnitSystem::scalefree) {
    G_ = 1.;
    kb_ = 1.;
    c_ = 1.;
    h_ = 1.;
    Msolar_ = 1.;
    AU_ = 1.;
  } else if (units.GetUnitSystem() == UnitSystem::cgs) {
    parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;
    G_ = pc.gravitational_constant;
    kb_ = pc.kb;
    c_ = pc.c;
    h_ = pc.h;
    Msolar_ = 1.988416e33;
    AU_ = 1.495978707e13;
  } else {
    PARTHENON_FAIL("Unknown unit system");
  }

  if (units.GetUnitSystem() != UnitSystem::scalefree) {
    const Real length = units.GetLengthCodeToPhysical();
    const Real time = units.GetTimeCodeToPhysical();
    const Real mass = units.GetMassCodeToPhysical();

    // Convert constants to code units
    G_code_ = G_ * std::pow(length, -3) / mass * std::pow(time, 2);
    kb_code_ =
        kb_ * std::pow(time, 2) / mass * std::pow(length, -2); // 1 K = 1 code unit temp
    c_code_ = c_ * time / length;
    h_code_ = h_ * time / mass * std::pow(length, -2);
    Msolar_code_ = Msolar_ / mass;
    AU_code_ = AU_ / length;
  }
}

} // namespace ArtemisUtils
