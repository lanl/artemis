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
  std::string physical_units_str =
      pin->GetOrAddString("artemis", "physical_units", "scalefree");
  if (physical_units_str == "scalefree") {
    physical_units_ = PhysicalUnits::scalefree;
  } else if (physical_units_str == "cgs") {
    physical_units_ = PhysicalUnits::cgs;
  } else {
    PARTHENON_FAIL("Physical unit system not recognized! Choices are [scalefree, cgs]");
  }
  length_ = 1.;
  time_ = 1.;
  mass_ = 1.;
  temp_ = 1.;
  if (physical_units_ != PhysicalUnits::scalefree) {
    std::string unit_conversion =
        pin->GetOrAddString("artemis", "unit_conversion", "base");
    if (unit_conversion == "ppd") {
      length_ = AU;
      mass_ = Msolar;
      time_ = Year / (2. * M_PI);
    } else if (unit_conversion == "base") {
      // do nothing
    } else {
      PARTHENON_FAIL("Unit conversion not recognized! Choices are [base, ppd]");
    }
    // not that these multiplied by whatever values were previously set.
    // For example, if unit_conversion=ppd and mass = 10.0, then
    // that sets mass_ to 10 MSolar.
    length_ *= pin->GetOrAddReal("artemis", "length", 1.);
    time_ *= pin->GetOrAddReal("artemis", "time", 1.);
    mass_ *= pin->GetOrAddReal("artemis", "mass", 1.);
    temp_ *= pin->GetOrAddReal("artemis", "temperature", 1.);
  }

  // Remaining conversion factors
  energy_ = std::pow(length_, 2) * mass_ * std::pow(time_, -2);
  number_density_ = std::pow(length_, -3);

  // Store everything necessary in params for usage in analysis
  pkg->AddParam("physical_units", physical_units_);
  pkg->AddParam("length", length_);
  pkg->AddParam("time", time_);
  pkg->AddParam("mass", mass_);
}

Constants::Constants(Units &units) {
  if (units.GetPhysicalUnits() == PhysicalUnits::scalefree) {
    G_ = 1.;
    kb_ = 1.;
    c_ = 1.;
    h_ = 1.;
    ar_ = 1.;
    amu_ = 1.;
    eV_ = 1.;
    Msolar_ = 1.;
    AU_ = 1.;
    pc_ = 1.;
    Year_ = 1.;
  } else if (units.GetPhysicalUnits() == PhysicalUnits::cgs) {
    parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;
    G_ = pc.gravitational_constant;
    kb_ = pc.kb;
    c_ = pc.c;
    h_ = pc.h;
    ar_ = pc.ar;
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

  const Real length = units.GetLengthCodeToPhysical();
  const Real time = units.GetTimeCodeToPhysical();
  const Real mass = units.GetMassCodeToPhysical();
  const Real temp = units.GetTemperatureCodeToPhysical();
  const Real energy = mass * std::pow(length / time, 2);

  // Convert constants to code units
  G_code_ = G_ * std::pow(length, -3) / mass * std::pow(time, 2);
  kb_code_ = kb_ * temp / energy;
  c_code_ = c_ * time / length;
  h_code_ = h_ / (energy * time);
  ar_code_ = ar_ * std::pow(temp, 4) * std::pow(length, 3) / energy;
  amu_code_ = amu_ / mass;
  eV_code_ = eV_ / energy;
  Msolar_code_ = Msolar_ / mass;
  AU_code_ = AU_ / length;
  Rjup_code_ = Rjup_ / length;
  Mjup_code_ = Mjup_ / mass;
  pc_code_ = pc_ / length;
  Year_code_ = Year_ / time;
}

} // namespace ArtemisUtils
