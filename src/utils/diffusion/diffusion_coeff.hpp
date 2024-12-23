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
#ifndef UTILS_DIFFUSION_DIFFUSION_COEFF_HPP_
#define UTILS_DIFFUSION_DIFFUSION_COEFF_HPP_

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/eos/eos.hpp"
#include "utils/units.hpp"

using ArtemisUtils::EOS;
namespace Diffusion {

enum class DiffType {
  viscosity_const,
  viscosity_alpha,
  conductivity_const,
  thermaldiff_const,
  null
};
enum class DiffAvg { arithmetic, harmonic, null };

inline DiffType ChooseDiffusion(std::string dtype, std::string type) {
  if (dtype == "viscosity") {
    if (type == "constant")
      return DiffType::viscosity_const;
    else if (type == "alpha")
      return DiffType::viscosity_alpha;
  } else if (dtype == "conductivity") {
    if (type == "constant")
      return DiffType::conductivity_const;
    else if (type == "diffusivity_constant")
      return DiffType::thermaldiff_const;
  }

  return DiffType::null;
}

inline DiffAvg ChooseAveraging(std::string choice) {
  if (choice == "arithmetic")
    return DiffAvg::arithmetic;
  else if (choice == "harmonic")
    return DiffAvg::harmonic;
  return DiffAvg::null;
}

struct DiffCoeffParams {
  DiffType type;
  DiffAvg avg;

  // Viscosity
  // -----------------
  // constant viscosity
  // nu_s is the kinematic shear viscosity (cgs : cm^2/s)
  // eta is the ratio of the bulk to shear kinematic viscosity
  Real nu_s, eta;
  // alpha viscosity
  // nu_s = alpha c_s^2 / Omega
  Real alpha, R0, Omega0;

  // Conduction
  // Heat flux q = - K grad(T) with K the thermal conductivity
  // -----------------
  // constant thermal diffusivity
  // kappa = K / (rho c_p) (cgs : cm^2/s)
  Real kappa_0;
  // constant conductivity
  // K  (cgs : erg/(cm s K) )
  Real hcond_0;

  // Diffusivity
  // -----------------
  // ..TBD..

  DiffCoeffParams() = default;
  DiffCoeffParams(std::string block_name, std::string dtype,
                  parthenon::ParameterInput *pin,
                  const ArtemisUtils::Constants &constants) {
    // Read the parameter file
    std::string type_ = pin->GetString(block_name, "type");
    type = ChooseDiffusion(dtype, type_);
    if (type == DiffType::null) {
      std::stringstream msg;
      msg << type_ << " in " << block_name << " is not supported";
      PARTHENON_FAIL(msg);
    }
    std::string avg_ = pin->GetOrAddString(block_name, "averaging", "arithmetic");
    avg = ChooseAveraging(avg_);
    if (avg == DiffAvg::null) {
      std::stringstream msg;
      msg << avg_ << " in " << block_name << " is not supported";
      PARTHENON_FAIL(msg);
    }
    // Read the parameters
    if (type == DiffType::viscosity_const) {
      nu_s = pin->GetReal(block_name, "nu");
      eta = pin->GetOrAddReal(block_name, "eta_bulk", 0.0);
    } else if (type == DiffType::viscosity_alpha) {
      alpha = pin->GetReal(block_name, "alpha");
      eta = pin->GetOrAddReal(block_name, "eta_bulk", 0.0);

      R0 = pin->GetOrAddReal("problem", "r0", 1.0);
      const Real gm = constants.GetGCode() * pin->GetReal("gravity", "mass_tot") *
                      constants.GetMsolarCode();
      Omega0 = std::sqrt(gm / (R0 * R0 * R0));
    } else if (type == DiffType::thermaldiff_const) {
      kappa_0 = pin->GetReal(block_name, "kappa");
    } else if (type == DiffType::conductivity_const) {
      hcond_0 = pin->GetReal(block_name, "cond");
    } else {
      std::stringstream msg;
      msg << type_ << " in " << block_name << " is not supported";
      PARTHENON_FAIL(msg);
    }
  }
};

// Zone averaging function

template <DiffAvg DAVG>
KOKKOS_INLINE_FUNCTION Real FaceAverage(const Real mu1, const Real mu2) {
  if constexpr (DAVG == DiffAvg::arithmetic) {
    return 0.5 * (mu1 + mu2);
  } else if constexpr (DAVG == DiffAvg::harmonic) {
    return 2.0 * mu1 * mu2 / (mu1 + mu2);
  } else {
    PARTHENON_FAIL("Invalid diffusion coefficient averaging method");
    return 0.0;
  }
  return 0.0;
}

template <DiffType DTYP, Coordinates GEOM, Fluid FLUID_TYPE>
class DiffusionCoeff {
 public:
  template <typename V1>
  KOKKOS_INLINE_FUNCTION void
  evaluate(const DiffCoeffParams &dp, parthenon::team_mbr_t const &member, const int b,
           const int n, const int k, const int j, const int il, const int iu, const V1 &p,
           const EOS &eos, const parthenon::ScratchPad1D<Real> &mu) const {
    PARTHENON_FAIL("No default implementation for diffusion coefficient");
  }
  KOKKOS_INLINE_FUNCTION Real Get(const DiffCoeffParams &dp,
                                  geometry::Coords<GEOM> coords, const Real dens,
                                  const Real sie, const EOS &eos) const {
    PARTHENON_FAIL("No default implementation for diffusion coefficient");
  }
};

// null
template <Coordinates GEOM, Fluid FLUID_TYPE>
class DiffusionCoeff<DiffType::null, GEOM, FLUID_TYPE> {
  // Constant Kinematic Viscosity
  // These routines return the dynamic viscosity, rho*nu (cgs : g/(cm s))
 public:
  template <typename V1>
  KOKKOS_INLINE_FUNCTION void
  evaluate(const DiffCoeffParams &dp, parthenon::team_mbr_t const &member, const int b,
           const int n, const int k, const int j, const int il, const int iu, const V1 &p,
           const EOS &eos, const parthenon::ScratchPad1D<Real> &mu) const {
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                             [&](const int i) { mu(i) = 0.0; });
  }

  KOKKOS_INLINE_FUNCTION Real Get(const DiffCoeffParams &dp,
                                  geometry::Coords<GEOM> coords, const Real dens,
                                  const Real sie, const EOS &eos) const {
    return 0.0;
  }
};

// Viscosity
template <Coordinates GEOM, Fluid FLUID_TYPE>
class DiffusionCoeff<DiffType::viscosity_const, GEOM, FLUID_TYPE> {
  // Constant Kinematic Viscosity
  // These routines return the dynamic viscosity, rho*nu (cgs : g/(cm s))
 public:
  template <typename V1>
  KOKKOS_INLINE_FUNCTION void
  evaluate(const DiffCoeffParams &dp, parthenon::team_mbr_t const &member, const int b,
           const int n, const int k, const int j, const int il, const int iu, const V1 &p,
           const EOS &eos, const parthenon::ScratchPad1D<Real> &mu) const {

    using TE = parthenon::TopologicalElement;

    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::viscosity_const,
        "Mismatch between evaluated viscosity type and input viscosity type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");

    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                             [&](const int i) {
                               const Real &dens = p(b, gas::prim::density(n), k, j, i);
                               mu(i) = dp.nu_s * dens;
                             });
  }

  KOKKOS_INLINE_FUNCTION Real Get(const DiffCoeffParams &dp,
                                  geometry::Coords<GEOM> coords, const Real dens,
                                  const Real sie, const EOS &eos) const {
    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::viscosity_const,
        "Mismatch between evaluated viscosity type and input viscosity type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");

    return dp.nu_s * dens;
  }
};

template <Coordinates GEOM, Fluid FLUID_TYPE>
class DiffusionCoeff<DiffType::viscosity_alpha, GEOM, FLUID_TYPE> {
  // Constant Alpha Viscosity
  // These routines return the dynamic viscosity, rho*nu (cgs : g/(cm s))
 public:
  template <typename V1>
  KOKKOS_INLINE_FUNCTION void
  evaluate(const DiffCoeffParams &dp, parthenon::team_mbr_t const &member, const int b,
           const int n, const int k, const int j, const int il, const int iu, const V1 &p,
           const EOS &eos, const parthenon::ScratchPad1D<Real> &mu) const {

    using TE = parthenon::TopologicalElement;

    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::viscosity_alpha,
        "Mismatch between evaluated viscosity type and input viscosity type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");

    auto pco = p.GetCoordinates(b);
    parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          geometry::Coords<GEOM> coords(pco, k, j, i);
          const auto &xv = coords.GetCellCenter();
          const auto &xs = coords.ConvertToSph(xv);
          const Real Omk = dp.Omega0 * std::pow(xs[0] / dp.R0, -1.5);

          const Real &dens = p(b, gas::prim::density(n), k, j, i);
          const Real &sie = p(b, gas::prim::sie(n), k, j, i);
          const Real blk = eos.BulkModulusFromDensityInternalEnergy(dens, sie);
          mu(i) = dp.alpha * blk / Omk;
        });
  }
  KOKKOS_INLINE_FUNCTION Real Get(const DiffCoeffParams &dp,
                                  geometry::Coords<GEOM> coords, const Real dens,
                                  const Real sie, const EOS &eos) const {
    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::viscosity_alpha,
        "Mismatch between evaluated viscosity type and input viscosity type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");

    const auto &xv = coords.GetCellCenter();
    const auto &xs = coords.ConvertToSph(xv);
    const Real Omk = dp.Omega0 * std::pow(xs[0] / dp.R0, -1.5);

    const Real blk = eos.BulkModulusFromDensityInternalEnergy(dens, sie);

    return dp.alpha * blk / Omk;
  }
};

// Conduction
template <Coordinates GEOM, Fluid FLUID_TYPE>
class DiffusionCoeff<DiffType::conductivity_const, GEOM, FLUID_TYPE> {
  // Conductivity
  // These routines return the conductivity, K (cgs : erg/(cm s K))
 public:
  template <typename V1>
  KOKKOS_INLINE_FUNCTION void
  evaluate(const DiffCoeffParams &dp, parthenon::team_mbr_t const &member, const int b,
           const int n, const int k, const int j, const int il, const int iu, const V1 &p,
           const EOS &eos, const parthenon::ScratchPad1D<Real> &mu) const {

    using TE = parthenon::TopologicalElement;

    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::conductivity_const,
        "Mismatch between evaluated conductivity type and input conductivity  type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");

    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                             [&](const int i) { mu(i) = dp.hcond_0; });
  }
  KOKKOS_INLINE_FUNCTION Real Get(const DiffCoeffParams &dp,
                                  geometry::Coords<GEOM> coords, const Real dens,
                                  const Real sie, const EOS &eos) const {

    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::conductivity_const,
        "Mismatch between evaluated conductivity type and input conductivity  type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");

    return dp.hcond_0;
  }
};

template <Coordinates GEOM, Fluid FLUID_TYPE>
class DiffusionCoeff<DiffType::thermaldiff_const, GEOM, FLUID_TYPE> {
  // Thermal Diffusivity
  // These routines return the conductivity, K (cgs : erg/(cm s K))
 public:
  template <typename V1>
  KOKKOS_INLINE_FUNCTION void
  evaluate(const DiffCoeffParams &dp, parthenon::team_mbr_t const &member, const int b,
           const int n, const int k, const int j, const int il, const int iu, const V1 &p,
           const EOS &eos, const parthenon::ScratchPad1D<Real> &mu) const {

    using TE = parthenon::TopologicalElement;

    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::conductivity_const,
        "Mismatch between evaluated conductivity type and input conductivity  type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");

    parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          const Real &dens = p(b, gas::prim::density(n), k, j, i);
          const Real &sie = p(b, gas::prim::sie(n), k, j, i);
          const Real cv = eos.SpecificHeatFromDensityInternalEnergy(dens, sie);
          //       const Real T =
          //       eos.TemperatureFromDensityInternalEnergy(dens,sie);

          mu(i) = dp.kappa_0 * dens * cv;
        });
  }
  KOKKOS_INLINE_FUNCTION Real Get(const DiffCoeffParams &dp,
                                  geometry::Coords<GEOM> coords, const Real dens,
                                  const Real sie, const EOS &eos) const {

    PARTHENON_DEBUG_REQUIRE(
        dp.type == DiffType::thermaldiff_const,
        "Mismatch between evaluated conductivity type and input conductivity  type");
    PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                            "Viscosity only works with the gas fluid");
    const Real cv = eos.SpecificHeatFromDensityInternalEnergy(dens, sie);

    return dp.kappa_0 * dens * cv;
  }
};

} // namespace Diffusion

#endif // UTILS_DIFFUSION_DIFFUSION_COEFF_HPP_
