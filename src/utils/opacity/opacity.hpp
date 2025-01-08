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
#ifndef UTILS_OPACITY_OPACITY_HPP_
#define UTILS_OPACITY_OPACITY_HPP_

// Singularity-opac includes
#include <singularity-opac/photons/opac_photons.hpp>
#include <singularity-opac/photons/s_opac_photons.hpp>

namespace ArtemisUtils {

// Custom units/opacity model for ShocktubeA problem
// c=1732.05, arad=7.716e-4
struct BasePhysicalConstantsShocktubeA : singularity::BaseUnity {
  static constexpr Real speed_of_light = 1732.05;
  static constexpr Real planck = 0.0344;
};
using PhysicalConstantsShocktubeA =
    singularity::PhysicalConstants<BasePhysicalConstantsShocktubeA,
                                   singularity::UnitConversionDefault>;
using ShocktubeAOpacity =
    singularity::photons::PowerLawOpacity<PhysicalConstantsShocktubeA>;

// Custom units/opacity model for Thermalization problem
// c=1.0, arad=1.0
struct BasePhysicalConstantsThermalization : singularity::BaseUnity {
  static constexpr Real speed_of_light = 1.0;
  static constexpr Real planck = 5.46490601180566;
};
using PhysicalConstantsThermalization =
    singularity::PhysicalConstants<BasePhysicalConstantsThermalization,
                                   singularity::UnitConversionDefault>;
using ThermalizationOpacity =
    singularity::photons::GrayOpacity<PhysicalConstantsThermalization>;

// Reduced absorption variant for this codebase
using Opacity = singularity::photons::impl::Variant<
    singularity::photons::NonCGSUnits<singularity::photons::Gray>,
    singularity::photons::NonCGSUnits<singularity::photons::PowerLaw>,
    singularity::photons::NonCGSUnits<singularity::photons::EPBremss>, ShocktubeAOpacity,
    ThermalizationOpacity>;

// Reduced scattering variant for this codebase
using Scattering = singularity::photons::impl::S_Variant<
    singularity::photons::NonCGSUnitsS<singularity::photons::GrayS>,
    singularity::photons::NonCGSUnitsS<singularity::photons::ThomsonS>>;

} // namespace ArtemisUtils

#endif // UTILS_OPACITY_OPACITY_HPP_
