//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was created in part by generative AI

#ifndef PGEN_FUNCTION_INIT_HPP_
#define PGEN_FUNCTION_INIT_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <pips/device/device_chunk.hpp>
#include <pips/device/device_status.hpp>
#include <pips/device/device_value.hpp>
#include <pips/device/device_vm.hpp>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

namespace Rummy {
class FullDeck;
}

namespace artemis {
namespace function_init {

struct DeviceCallable {
  pips::device::DeviceModule module{};
  std::uint32_t entry_id = 0;
  int function_id = -1;

  KOKKOS_INLINE_FUNCTION bool IsValid() const { return function_id >= 0; }
};

enum class ThermodynamicInput : int {
  none = 0,
  temperature = 1,
  pressure = 2,
  internal_energy = 3
};

struct GasConfig {
  bool enabled = false;
  Coordinates input_system = Coordinates::null;
  ThermodynamicInput thermodynamic_input = ThermodynamicInput::none;
  DeviceCallable density;
  DeviceCallable thermodynamic;
  std::array<DeviceCallable, 3> velocity;
};

struct DustConfig {
  bool enabled = false;
  Coordinates input_system = Coordinates::null;
  DeviceCallable density;
  std::array<DeviceCallable, 3> velocity;
};

struct MomentConfig {
  bool enabled = false;
  Coordinates input_system = Coordinates::null;
  DeviceCallable energy;
  std::array<DeviceCallable, 3> reduced_flux;
};

struct FunctionInitConfig {
  Real time = 0.0;
  GasConfig gas;
  DustConfig dust;
  MomentConfig moment;

  bool Enabled() const { return gas.enabled || dust.enabled || moment.enabled; }
};

void Configure(ParameterInput *pin, Rummy::FullDeck *deck, Packages_t &packages);

namespace impl {

enum class ErrorCode : int {
  none = 0,
  vm_error_base = 100,
  non_numeric = 200,
  non_finite = 201,
  non_positive = 202
};

template <typename ERROR>
KOKKOS_INLINE_FUNCTION void RecordError(const ERROR &error, const int function_id,
                                        const int error_code, const int k, const int j,
                                        const int i) {
  if (Kokkos::atomic_compare_exchange(&error(0), 0, 1) == 0) {
    error(1) = function_id;
    error(2) = error_code;
    error(3) = k;
    error(4) = j;
    error(5) = i;
  }
}

template <typename ERROR>
KOKKOS_INLINE_FUNCTION bool
Call(pips::device::DeviceVM &vm, const DeviceCallable &callable,
     const pips::device::DeviceValue *args, const std::uint32_t argc, Real &value,
     const ERROR &error, const int k, const int j, const int i) {
  pips::device::DeviceValue result{};
  const auto status = vm.run(callable.module, callable.entry_id, args, argc, &result);
  if (status != pips::device::DeviceStatus::OK) {
    RecordError(error, callable.function_id,
                static_cast<int>(ErrorCode::vm_error_base) + static_cast<int>(status), k,
                j, i);
    return false;
  }
  if (!pips::device::dv_is_number(result)) {
    RecordError(error, callable.function_id, static_cast<int>(ErrorCode::non_numeric), k,
                j, i);
    return false;
  }
  value = static_cast<Real>(pips::device::dv_as_number(result));
  if (!std::isfinite(value)) {
    RecordError(error, callable.function_id, static_cast<int>(ErrorCode::non_finite), k,
                j, i);
    return false;
  }
  return true;
}

template <typename ERROR>
KOKKOS_INLINE_FUNCTION bool
RequirePositive(const Real value, const DeviceCallable &callable, const ERROR &error,
                const int k, const int j, const int i) {
  if (value > 0.0) return true;
  RecordError(error, callable.function_id, static_cast<int>(ErrorCode::non_positive), k,
              j, i);
  return false;
}

template <typename COORDS, typename XV>
KOKKOS_INLINE_FUNCTION void
ConvertCoordinates(const COORDS &coords, const XV &xi, const Coordinates input_system,
                   std::array<Real, 3> &xo, std::array<Real, 3> &ex1,
                   std::array<Real, 3> &ex2, std::array<Real, 3> &ex3) {
  if (input_system == Coordinates::cartesian) {
    const auto converted = coords.ConvertToCartWithVec(xi);
    xo = std::get<0>(converted);
    ex1 = std::get<1>(converted);
    ex2 = std::get<2>(converted);
    ex3 = std::get<3>(converted);
  } else if (input_system == Coordinates::axisymmetric) {
    const auto converted = coords.ConvertToAxiWithVec(xi);
    xo = std::get<0>(converted);
    ex1 = std::get<1>(converted);
    ex2 = std::get<2>(converted);
    ex3 = std::get<3>(converted);
  } else if (geometry::is_spherical(input_system)) {
    const auto converted = coords.ConvertToSphWithVec(xi);
    xo = std::get<0>(converted);
    ex1 = std::get<1>(converted);
    ex2 = std::get<2>(converted);
    ex3 = std::get<3>(converted);
  } else {
    const auto converted = coords.ConvertToCylWithVec(xi);
    xo = std::get<0>(converted);
    ex1 = std::get<1>(converted);
    ex2 = std::get<2>(converted);
    ex3 = std::get<3>(converted);
  }
}

KOKKOS_INLINE_FUNCTION std::array<Real, 3> ProjectVector(const std::array<Real, 3> &vin,
                                                         const std::array<Real, 3> &ex1,
                                                         const std::array<Real, 3> &ex2,
                                                         const std::array<Real, 3> &ex3) {
  return {vin[0] * ex1[0] + vin[1] * ex1[1] + vin[2] * ex1[2],
          vin[0] * ex2[0] + vin[1] * ex2[1] + vin[2] * ex2[2],
          vin[0] * ex3[0] + vin[1] * ex3[1] + vin[2] * ex3[2]};
}

inline std::string ErrorDescription(const int code) {
  if (code >= static_cast<int>(ErrorCode::vm_error_base) &&
      code < static_cast<int>(ErrorCode::non_numeric)) {
    return "PIPS device VM status " +
           std::to_string(code - static_cast<int>(ErrorCode::vm_error_base));
  }
  if (code == static_cast<int>(ErrorCode::non_numeric)) {
    return "returned a non-numeric value";
  }
  if (code == static_cast<int>(ErrorCode::non_finite)) {
    return "returned a non-finite value";
  }
  if (code == static_cast<int>(ErrorCode::non_positive)) {
    return "returned a non-positive density, energy, pressure, or temperature";
  }
  return "failed with unknown error code " + std::to_string(code);
}

} // namespace impl

template <Coordinates GEOM>
void Initialize(MeshBlock *pmb, ParameterInput *pin);

} // namespace function_init
} // namespace artemis

#endif // PGEN_FUNCTION_INIT_HPP_
