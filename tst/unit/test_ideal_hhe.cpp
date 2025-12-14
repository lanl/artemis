//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
//  license in this material to reproduce, prepare derivative works, distribute copies to
//  the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// This file was created in part or in whole by generative AI

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include "artemis.hpp"
#include "utils/eos/eos.hpp"

using Catch::Approx;
using singularity::IdealGas;
using ArtemisEOS::IdealHHe;

#ifdef SPINER_USE_HDF
TEST_CASE("IdealHHe EOS basic construction and properties", "[IdealHHe][EOS]") {
  // Use solar composition: X = 0.7, Y = 0.28 (Z = 0.02)
  const Real X = 0.7;
  const Real Y = 0.28;
  
  // Temperature range: 1 K to 1e6 K (ltmin=0, ltmax=6 in log10)
  const Real ltmin = 0.0;
  const Real ltmax = 6.0;
  const int nt = 50;
  
  // Density range: 1e-15 to 1e-3 g/cc (ldmin=-15, ldmax=-3 in log10)
  const Real ldmin = -15.0;
  const Real ldmax = -3.0;
  const int nd = 50;
  
  // Create EOS without table lookup for testing
  const bool use_table = false;
  const std::string save_to_file = "";
  
  IdealHHe eos(X, Y, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table);

  SECTION("Temperature recovery from density and internal energy") {
    const Real rho = 1.0e-10; // g/cc
    const Real T_input = 1000.0; // K
    
    // Get internal energy at this state
    const Real sie = eos.InternalEnergyFromDensityTemperature(rho, T_input);
    
    // Recover temperature from density and internal energy
    const Real T_recovered = eos.TemperatureFromDensityInternalEnergy(rho, sie);
    
    // Temperature should be recovered to within 1%
    REQUIRE(T_recovered == Approx(T_input).epsilon(0.01));
  }
  
  SECTION("Pressure calculation from density and temperature") {
    const Real rho = 1.0e-10; // g/cc
    const Real T = 1000.0; // K
    
    const Real P = eos.PressureFromDensityTemperature(rho, T);
    
    // Pressure should be positive
    REQUIRE(P > 0.0);
  }
  
  SECTION("Round-trip consistency: rho, T -> sie -> T") {
    const Real rho = 1.0e-12; // g/cc
    const Real T_original = 5000.0; // K
    
    // Convert T to sie
    const Real sie = eos.InternalEnergyFromDensityTemperature(rho, T_original);
    
    // Convert sie back to T
    const Real T_recovered = eos.TemperatureFromDensityInternalEnergy(rho, sie);
    
    // Should recover original temperature to within 1%
    REQUIRE(T_recovered == Approx(T_original).epsilon(0.01));
  }
  
  SECTION("Internal energy increases with temperature") {
    const Real rho = 1.0e-10; // g/cc
    const Real T1 = 1000.0; // K
    const Real T2 = 2000.0; // K
    
    const Real sie1 = eos.InternalEnergyFromDensityTemperature(rho, T1);
    const Real sie2 = eos.InternalEnergyFromDensityTemperature(rho, T2);
    
    // Higher temperature should give higher internal energy
    REQUIRE(sie2 > sie1);
  }
}

TEST_CASE("IdealHHe EOS with different compositions", "[IdealHHe][EOS]") {
  const Real ltmin = 0.0;
  const Real ltmax = 6.0;
  const int nt = 50;
  const Real ldmin = -15.0;
  const Real ldmax = -3.0;
  const int nd = 50;
  const bool use_table = false;
  const std::string save_to_file = "";
  
  SECTION("Hydrogen-rich composition (X=0.9, Y=0.1)") {
    IdealHHe eos_H_rich(0.9, 0.1, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table);
    
    const Real rho = 1.0e-10;
    const Real T = 1000.0;
    
    const Real sie = eos_H_rich.InternalEnergyFromDensityTemperature(rho, T);
    const Real P = eos_H_rich.PressureFromDensityTemperature(rho, T);
    
    REQUIRE(sie > 0.0);
    REQUIRE(P > 0.0);
  }
  
  SECTION("Helium-rich composition (X=0.3, Y=0.7)") {
    IdealHHe eos_He_rich(0.3, 0.7, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table);
    
    const Real rho = 1.0e-10;
    const Real T = 1000.0;
    
    const Real sie = eos_He_rich.InternalEnergyFromDensityTemperature(rho, T);
    const Real P = eos_He_rich.PressureFromDensityTemperature(rho, T);
    
    REQUIRE(sie > 0.0);
    REQUIRE(P > 0.0);
  }
}

TEST_CASE("IdealHHe EOS with UnitSystem wrapper", "[IdealHHe][EOS][UnitSystem]") {
  // Use solar composition
  const Real X = 0.7;
  const Real Y = 0.28;
  const Real ltmin = 0.0;
  const Real ltmax = 6.0;
  const int nt = 50;
  const Real ldmin = -15.0;
  const Real ldmax = -3.0;
  const int nd = 50;
  const bool use_table = false;
  const std::string save_to_file = "";
  
  IdealHHe eos_base(X, Y, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table);
  
  SECTION("Unit conversion with CGS units") {
    // Define unit system: time [s], mass [g], length [cm], temperature [K]
    const Real time_cgs = 1.0;
    const Real mass_cgs = 1.0;
    const Real length_cgs = 1.0;
    const Real temp_cgs = 1.0;
    
    // Create IdealHHe inline as temporary for UnitSystem
    auto eos = singularity::UnitSystem<IdealHHe>(
        IdealHHe(X, Y, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table),
        singularity::eos_units_init::LengthTimeUnitsInit(),
        time_cgs, mass_cgs, length_cgs, temp_cgs);
    
    const Real rho = 1.0e-10; // g/cc
    const Real T = 1000.0; // K
    
    const Real sie = eos.InternalEnergyFromDensityTemperature(rho, T);
    const Real P = eos.PressureFromDensityTemperature(rho, T);
    
    REQUIRE(sie > 0.0);
    REQUIRE(P > 0.0);
  }
  
  SECTION("Unit conversion with scaled units") {
    // Define unit system with scale factors
    // Time: 1 day = 86400 s
    // Mass: 1 solar mass = 1.989e33 g
    // Length: 1 AU = 1.496e13 cm
    // Temperature: 1 K
    const Real time_scale = 86400.0;
    const Real mass_scale = 1.989e33;
    const Real length_scale = 1.496e13;
    const Real temp_scale = 1.0;
    
    // Create IdealHHe inline as temporary for UnitSystem
    auto eos = singularity::UnitSystem<IdealHHe>(
        IdealHHe(X, Y, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table),
        singularity::eos_units_init::LengthTimeUnitsInit(),
        time_scale, mass_scale, length_scale, temp_scale);
    
    // Density in code units (Msun/AU^3)
    // Convert 1e-10 g/cc to Msun/AU^3: 1e-10 * (AU^3/Msun)
    const Real rho_cgs = 1.0e-10; // g/cc
    const Real rho_code = rho_cgs * std::pow(length_scale, 3) / mass_scale;
    
    // Temperature in code units (same as K in this case)
    const Real T_code = 1000.0;
    
    const Real sie_code = eos.InternalEnergyFromDensityTemperature(rho_code, T_code);
    const Real P_code = eos.PressureFromDensityTemperature(rho_code, T_code);
    
    // Convert back to CGS for comparison
    // sie: [cm^2/s^2] -> code units: [length^2/time^2]
    const Real sie_cgs = sie_code * std::pow(length_scale, 2) / std::pow(time_scale, 2);
    
    // Verify physical values are positive
    REQUIRE(sie_cgs > 0.0);
    REQUIRE(P_code > 0.0);
  }
  
  SECTION("Consistency between wrapped and unwrapped EOS in CGS units") {
    // Wrap with CGS units (scale factors = 1)
    auto eos_wrapped = singularity::UnitSystem<IdealHHe>(
        IdealHHe(X, Y, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table),
        singularity::eos_units_init::LengthTimeUnitsInit(),
        1.0, 1.0, 1.0, 1.0);
    
    const Real rho = 1.0e-10; // g/cc
    const Real T = 1000.0; // K
    
    // Get results from both
    const Real sie_unwrapped = eos_base.InternalEnergyFromDensityTemperature(rho, T);
    const Real sie_wrapped = eos_wrapped.InternalEnergyFromDensityTemperature(rho, T);
    
    const Real P_unwrapped = eos_base.PressureFromDensityTemperature(rho, T);
    const Real P_wrapped = eos_wrapped.PressureFromDensityTemperature(rho, T);
    
    // Should match exactly with unit scale factors = 1
    REQUIRE(sie_wrapped == Approx(sie_unwrapped).epsilon(1e-10));
    REQUIRE(P_wrapped == Approx(P_unwrapped).epsilon(1e-10));
  }
  
  SECTION("Round-trip with unit conversions") {
    // Use code units
    const Real time_scale = 1.0e3;
    const Real mass_scale = 1.0e30;
    const Real length_scale = 1.0e10;
    const Real temp_scale = 1.0;
    
    // Create IdealHHe inline as temporary for UnitSystem
    auto eos = singularity::UnitSystem<IdealHHe>(
        IdealHHe(X, Y, ltmin, ltmax, nt, ldmin, ldmax, nd, save_to_file, use_table),
        singularity::eos_units_init::LengthTimeUnitsInit(),
        time_scale, mass_scale, length_scale, temp_scale);
    
    // Density in code units
    const Real rho_cgs = 1.0e-10;
    const Real rho_code = rho_cgs * std::pow(length_scale, 3) / mass_scale;
    const Real T_code = 1000.0;
    
    // Round-trip: T -> sie -> T
    const Real sie = eos.InternalEnergyFromDensityTemperature(rho_code, T_code);
    const Real T_recovered = eos.TemperatureFromDensityInternalEnergy(rho_code, sie);
    
    // Should recover temperature to within 1%
    REQUIRE(T_recovered == Approx(T_code).epsilon(0.01));
  }
}
#endif // SPINER_USE_HDF
