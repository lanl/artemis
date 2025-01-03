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

  if (physical_units_ == PhysicalUnits::scalefree) {
    length_ = 1.;
    time_ = 1.;
    mass_ = 1.;
  } else {
    std::string unit_conversion =
        pin->GetOrAddString("artemis", "unit_conversion", "base");
    if (unit_conversion == "base") {
      length_ = pin->GetReal("artemis", "length");
      time_ = pin->GetReal("artemis", "time");
      mass_ = pin->GetReal("artemis", "mass");
    } else if (unit_conversion == "ppd") {
      const Real r0_length = pin->GetOrAddReal("artemis", "r0_length", 1.0); // in AU
      const Real mstar = pin->GetOrAddReal("artemis", "mstar", 1.0);         // M_sun
      length_ = r0_length * AU;
      mass_ = mstar * Msolar * pin->GetReal("artemis", "rho0");
      time_ = Year / (2. * M_PI) * std::sqrt(r0_length / mstar) * r0_length;
    } else {
      PARTHENON_FAIL("Unit conversion not recognized! Choices are [base, ppd]");
    }
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

  // Convert constants to code units
  G_code_ = G_ * std::pow(length, -3) / mass * std::pow(time, 2);
  kb_code_ =
      kb_ * std::pow(time, 2) / mass * std::pow(length, -2); // 1 K = 1 code unit temp
  c_code_ = c_ * time / length;
  h_code_ = h_ * time / mass * std::pow(length, -2);
  ar_code_ = ar_ * length * time * time / mass;
  amu_code_ = amu_ / mass;
  eV_code_ = eV_ * std::pow(time, 2) / mass * std::pow(length, -2);
  Msolar_code_ = Msolar_ / mass;
  AU_code_ = AU_ / length;
  Rjup_code_ = Rjup_ / length;
  Mjup_code_ = Mjup_ / mass;
  pc_code_ = pc_ / length;
  Year_code_ = Year_ / time;
}

} // namespace ArtemisUtils
