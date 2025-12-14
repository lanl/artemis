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

#include "artemis.hpp"
#include "utils/eos/eos.hpp"

using Catch::Approx;

TEST_CASE("IdealGas EOS recovers input gamma", "[eos][ideal_gas]") {
  // Create an ideal gas EOS with known gamma
  constexpr Real gamma = 5.0 / 3.0;  // Monoatomic ideal gas
  constexpr Real cv = 1.0e7;         // Specific heat (arbitrary units)
  
  // IdealGas constructor takes (Gamma - 1, Cv)
  singularity::IdealGas eos(gamma - 1.0, cv);
  
  SECTION("Gamma is correctly recovered") {
    // The IdealGas EOS should return the gamma we set
    Real recovered_gamma = eos.GruneisenParamFromDensityInternalEnergy(1.0, 1.0);
    
    // Gruneisen parameter for ideal gas is (gamma - 1)
    REQUIRE(recovered_gamma == Approx(gamma - 1.0).epsilon(1e-12));
  }
  
  SECTION("Pressure-density-temperature relation is consistent") {
    // Test the ideal gas law: P = rho * (gamma - 1) * cv * T
    Real rho = 1.0e-3;
    Real temp = 1.0e4;
    
    Real pressure = eos.PressureFromDensityTemperature(rho, temp);
    Real expected_pressure = rho * (gamma - 1.0) * cv * temp;
    
    REQUIRE(pressure == Approx(expected_pressure).epsilon(1e-12));
  }
  
  SECTION("Internal energy-density-temperature relation is consistent") {
    // Test: sie = cv * T
    Real rho = 1.0e-3;
    Real temp = 1.0e4;
    
    Real sie = eos.InternalEnergyFromDensityTemperature(rho, temp);
    Real expected_sie = cv * temp;
    
    REQUIRE(sie == Approx(expected_sie).epsilon(1e-12));
  }
  
  SECTION("Pressure from density and internal energy") {
    // Test: P = rho * sie * (gamma - 1)
    Real rho = 1.0e-3;
    Real sie = 1.0e11;
    
    Real pressure = eos.PressureFromDensityInternalEnergy(rho, sie);
    Real expected_pressure = rho * sie * (gamma - 1.0);
    
    REQUIRE(pressure == Approx(expected_pressure).epsilon(1e-12));
  }
  
  SECTION("Temperature from density and internal energy") {
    // Test: T = sie / cv
    Real rho = 1.0e-3;
    Real sie = 1.0e11;
    
    Real temp = eos.TemperatureFromDensityInternalEnergy(rho, sie);
    Real expected_temp = sie / cv;
    
    REQUIRE(temp == Approx(expected_temp).epsilon(1e-12));
  }
}

TEST_CASE("IdealGas EOS with different gamma values", "[eos][ideal_gas]") {
  constexpr Real cv = 1.0e7;
  
  SECTION("Monoatomic gas (gamma = 5/3)") {
    constexpr Real gamma = 5.0 / 3.0;
    singularity::IdealGas eos(gamma - 1.0, cv);
    
    Real gruneisen = eos.GruneisenParamFromDensityInternalEnergy(1.0, 1.0);
    REQUIRE(gruneisen == Approx(gamma - 1.0).epsilon(1e-12));
  }
}

TEST_CASE("IdealGas EOS with UnitSystem wrapper", "[eos][ideal_gas][units]") {
  // Set up unit conversions (example values similar to gas.cpp)
  constexpr Real gamma = 5.0 / 3.0;
  constexpr Real cv_physical = 1.0e7;  // Physical units (e.g., cm^2/s^2/K)
  constexpr Real time_code_to_physical = 1.0;
  constexpr Real mass_code_to_physical = 1.0;
  constexpr Real length_code_to_physical = 1.0;
  constexpr Real temperature_code_to_physical = 1.0;
  
  // Create EOS with UnitSystem wrapper (as done in gas.cpp around line 98)
  ArtemisUtils::EOS eos_host = singularity::UnitSystem<singularity::IdealGas>(
      singularity::IdealGas(gamma - 1.0, cv_physical),
      singularity::eos_units_init::LengthTimeUnitsInit(),
      time_code_to_physical,
      mass_code_to_physical,
      length_code_to_physical,
      temperature_code_to_physical);
  
  SECTION("Gamma is correctly recovered through UnitSystem") {
    Real recovered_gamma = eos_host.GruneisenParamFromDensityInternalEnergy(1.0, 1.0);
    REQUIRE(recovered_gamma == Approx(gamma - 1.0).epsilon(1e-12));
  }
  
  SECTION("Pressure calculation with UnitSystem") {
    Real rho = 1.0e-3;
    Real temp = 1.0e4;
    
    Real pressure = eos_host.PressureFromDensityTemperature(rho, temp);
    Real expected_pressure = rho * (gamma - 1.0) * cv_physical * temp;
    
    REQUIRE(pressure == Approx(expected_pressure).epsilon(1e-12));
  }
  
  SECTION("Temperature recovery with UnitSystem") {
    Real rho = 1.0e-3;
    Real sie = 1.0e11;
    
    Real temp = eos_host.TemperatureFromDensityInternalEnergy(rho, sie);
    Real expected_temp = sie / cv_physical;
    
    REQUIRE(temp == Approx(expected_temp).epsilon(1e-12));
  }
  
  SECTION("Unit conversions with different scale factors") {
    // Test with non-trivial unit conversions
    constexpr Real time_scale = 2.0;
    constexpr Real mass_scale = 3.0;
    constexpr Real length_scale = 4.0;
    constexpr Real temp_scale = 5.0;
    
    ArtemisUtils::EOS eos_scaled = singularity::UnitSystem<singularity::IdealGas>(
        singularity::IdealGas(gamma - 1.0, cv_physical),
        singularity::eos_units_init::LengthTimeUnitsInit(),
        time_scale, mass_scale, length_scale, temp_scale);
    
    // The EOS should still work with scaled units
    Real rho_code = 1.0;
    Real temp_code = 1.0;
    
    Real pressure_code = eos_scaled.PressureFromDensityTemperature(rho_code, temp_code);
    
    // Verify pressure is positive and physically reasonable
    REQUIRE(pressure_code > 0.0);
    
    // Verify round-trip consistency in code units
    Real sie_code = eos_scaled.InternalEnergyFromDensityTemperature(rho_code, temp_code);
    Real temp_recovered = eos_scaled.TemperatureFromDensityInternalEnergy(rho_code, sie_code);
    
    REQUIRE(temp_recovered == Approx(temp_code).epsilon(1e-10));
  }
}
