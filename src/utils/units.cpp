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

constexpr Real Msolar = 1.988416e33;
constexpr Real AU = 1.495978707e13;
constexpr Real Year = 31536000;
constexpr Real parsec = 3.0857e18;
constexpr Real Rjup = 6.991100e6;
constexpr Real Mjup = 1.8982e30;

Units::Units(ParameterInput *pin, std::shared_ptr<StateDescriptor> pkg) {
  std::string unit_system_str = pin->GetOrAddString("artemis/units", "type", "scalefree");
  if (unit_system_str == "scalefree") {
    unit_system_ = UnitSystem::scalefree;
  } else if (unit_system_str == "cgs") {
    unit_system_ = UnitSystem::cgs;
  } else {
    PARTHENON_FAIL("Unit system not recognized! Valid choices are [scalefree, cgs]");
  }

  if (unit_system_ != UnitSystem::scalefree) {
    std::string unit_specifier = pin->GetOrAddString("artemis", "unit_system", "base");
    if (unit_specifier == "base") {
      length_ = pin->GetReal("artemis", "length");
      time_ = pin->GetReal("artemis", "time");
      mass_ = pin->GetReal("artemis", "mass");
    } else if (unit_specifier == "orbit") {
      const Real total_mass = pin->GetReal("artemis", "total_mass"); // Solar masses
      const Real reference_radius = pin->GetReal("artemis", "reference_radius"); // AU
      mass_ = total_mass * Msolar;
      length_ = reference_radius * AU;
      parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;
      time_ = std::sqrt(4. * M_PI * M_PI / (pc.gravitational_constant * mass_) *
                        std::pow(length_, 3)); // Orbital period
    } else {
      PARTHENON_FAIL("Unit specifier not recognized!");
    }
  }

  // Remaining conversion factors
  energy_ = std::pow(length_, 2) * mass_ * std::pow(time_, -2);
  number_density_ = std::pow(length_, -3);
  mass_density_ = mass_ * number_density_;

  // Store everything necessary in params for usage in analysis
  pkg->AddParam("unit_system", unit_system_);
  pkg->AddParam("length", length_);
  pkg->AddParam("time", time_);
  pkg->AddParam("mass", mass_);
}

Constants::Constants(Units &units) {
  if (units.GetUnitSystem() == UnitSystem::scalefree) {
    G_ = 1.;
    kb_ = 1.;
    c_ = 1.;
    h_ = 1.;
    amu_ = 1.;
    eV_ = 1.;
    Msolar_ = 1.;
    AU_ = 1.;
    pc_ = 1.;
    Year_ = 1.;
  } else if (units.GetUnitSystem() == UnitSystem::cgs) {
    parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;
    G_ = pc.gravitational_constant;
    kb_ = pc.kb;
    c_ = pc.c;
    h_ = pc.h;
    amu_ = pc.amu;
    eV_ = pc.eV;
    Msolar_ = Msolar;
    AU_ = AU;
    Rjup_ = Rjup;
    Mjup_ = Mjup;
    pc_ = parsec;
    Year_ = Year;
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
    amu_code_ = amu_ / mass;
    eV_code_ = ev_ * std::pow(time, 2) / mass * std::pot(length, -2);
    Msolar_code_ = Msolar_ / mass;
    AU_code_ = AU_ / length;
    Rjup_code_ = Rjup_ / length;
    Mjup_code_ = Mjup_ / mass;
    pc_code_ = pc_ / length;
    Year_code_ = Year_ / time;
  }
}

} // namespace ArtemisUtils
