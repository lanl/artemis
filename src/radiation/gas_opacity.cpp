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
#include "gas_opacity.hpp"
#include "utils/opacity/opacity.hpp"

namespace Gas {
void InitGasOpacity(ParameterInput *pin, const ArtemisUtils::Units &units,
                    Params &params) {
  using namespace singularity::photons;

  // Opacity models
  const Real time = units.GetTimeCodeToPhysical();
  const Real mass = units.GetMassCodeToPhysical();
  const Real length = units.GetLengthCodeToPhysical();
  const Real temp = units.GetTemperatureCodeToPhysical();

  // Get frequency type (it should already be set in radiation Initialization)
  const auto frequency_type = params.Get<FrequencyType>("frequency_type");

  // Absorption opacity model
  std::string opacity_model_name =
      pin->GetOrAddString("gas/opacity/absorption", "opacity_model", "constant");

  // Mean absorption opacity (either read from table or uses model
  ArtemisUtils::Opacity mg_opacity;
  ArtemisUtils::MeanOpacity opacity;
  if (opacity_model_name == "table") {
    PARTHENON_REQUIRE(frequency_type == FrequencyType::gray,
                      "Only gray table opacity permitted, for now.");
    std::string table_filename =
        pin->GetString("gas/opacity/absorption", "opacity_table");
    opacity =
        singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
            singularity::photons::MeanOpacityBase(table_filename), time, mass, length,
            temp);
  } else {
    // Instantiate mean absorption opacity object (i.e., table)
    const Real lRhoMin_a = pin->GetOrAddReal("gas/opacity/absorption", "lRhoMin", -1.0);
    const Real lRhoMax_a = pin->GetOrAddReal("gas/opacity/absorption", "lRhoMax", 1.0);
    const int NRho_a = pin->GetOrAddInteger("gas/opacity/absorption", "NRho", 2);
    const Real lTMin_a = pin->GetOrAddReal("gas/opacity/absorption", "lTMin", -1.0);
    const Real lTMax_a = pin->GetOrAddReal("gas/opacity/absorption", "lTMax", 1.0);
    const int NT_a = pin->GetOrAddInteger("gas/opacity/absorption", "NT", 2);

    if (opacity_model_name == "none") {
      auto model = Gray(0.0);
      opacity =
          singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
              singularity::photons::MeanOpacityBase(model, lRhoMin_a, lRhoMax_a, NRho_a,
                                                    lTMin_a, lTMax_a, NT_a),
              time, mass, length, temp);
      if (frequency_type == FrequencyType::multigroup) {
        mg_opacity = singularity::photons::NonCGSUnits<singularity::photons::Gray>(
            std::move(model), time, mass, length, temp);
      }
    } else if (opacity_model_name == "constant") {
      const Real kappa_a = pin->GetOrAddReal("gas/opacity/absorption", "kappa_a", 0.0);
      auto model = Gray(kappa_a);
      opacity =
          singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
              singularity::photons::MeanOpacityBase(model, lRhoMin_a, lRhoMax_a, NRho_a,
                                                    lTMin_a, lTMax_a, NT_a),
              time, mass, length, temp);
      if (frequency_type == FrequencyType::multigroup) {
        mg_opacity = singularity::photons::NonCGSUnits<singularity::photons::Gray>(
            std::move(model), time, mass, length, temp);
      }
    } else if (opacity_model_name == "powerlaw") {
      const Real coef_kappa_a =
          pin->GetOrAddReal("gas/opacity/absorption", "coef_kappa_a", 0.0);
      const Real rho_exp = pin->GetOrAddReal("gas/opacity/absorption", "rho_exp", 0.0);
      const Real temp_exp = pin->GetOrAddReal("gas/opacity/absorption", "temp_exp", 0.0);
      auto model = PowerLaw(coef_kappa_a, rho_exp, temp_exp);
      opacity =
          singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
              singularity::photons::MeanOpacityBase(model, lRhoMin_a, lRhoMax_a, NRho_a,
                                                    lTMin_a, lTMax_a, NT_a),
              time, mass, length, temp);
      if (frequency_type == FrequencyType::multigroup) {
        mg_opacity = singularity::photons::NonCGSUnits<singularity::photons::PowerLaw>(
            std::move(model), time, mass, length, temp);
      }
    } else {
      PARTHENON_FAIL("Opacity model not recognized!");
    }
  }

  params.Add("opacity_h", opacity);
  params.Add("opacity_d", opacity.GetOnDevice());
  if (frequency_type == FrequencyType::multigroup) {
    params.Add("mg_opacity_h", mg_opacity);
    params.Add("mg_opacity_d", mg_opacity.GetOnDevice());
  }

  // Scattering opacity model
  std::string scattering_model_name =
      pin->GetOrAddString("gas/opacity/scattering", "scattering_model", "none");

  // Instantiate mean scattering opacity object (i.e., table)
  const Real lRhoMin_s = pin->GetOrAddReal("gas/opacity/scattering", "lRhoMin", -1.0);
  const Real lRhoMax_s = pin->GetOrAddReal("gas/opacity/scattering", "lRhoMax", 1.0);
  const int NRho_s = pin->GetOrAddInteger("gas/opacity/scattering", "NRho", 2);
  const Real lTMin_s = pin->GetOrAddReal("gas/opacity/scattering", "lTMin", -1.0);
  const Real lTMax_s = pin->GetOrAddReal("gas/opacity/scattering", "lTMax", 1.0);
  const int NT_s = pin->GetOrAddInteger("gas/opacity/scattering", "NT", 2);

  ArtemisUtils::Scattering mg_scattering;
  ArtemisUtils::MeanScattering scattering;
  if (scattering_model_name == "none") {
    auto smodel = GrayS(0.0, 1.0);
    scattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityCGS>(
            singularity::photons::MeanSOpacityCGS(smodel, lRhoMin_s, lRhoMax_s, NRho_s,
                                                  lTMin_s, lTMax_s, NT_s),
            time, mass, length, temp);
    if (frequency_type == FrequencyType::multigroup) {
      mg_scattering = singularity::photons::NonCGSUnitsS<singularity::photons::GrayS>(
          std::move(smodel), time, mass, length, temp);
    }
  } else if (scattering_model_name == "constant") {
    const Real kappa_s = pin->GetOrAddReal("gas/opacity/scattering", "kappa_s", 0.0);
    auto smodel = GrayS(kappa_s, 1.0);
    scattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityCGS>(
            singularity::photons::MeanSOpacityCGS(smodel, lRhoMin_s, lRhoMax_s, NRho_s,
                                                  lTMin_s, lTMax_s, NT_s),
            time, mass, length, temp);
    if (frequency_type == FrequencyType::multigroup) {
      mg_scattering = singularity::photons::NonCGSUnitsS<singularity::photons::GrayS>(
          std::move(smodel), time, mass, length, temp);
    }
  } else {
    PARTHENON_FAIL("Scattering model not recognized!");
  }

  params.Add("scattering_h", scattering);
  params.Add("scattering_d", scattering.GetOnDevice());
  if (frequency_type == FrequencyType::multigroup) {
    params.Add("mg_scattering_h", mg_scattering);
    params.Add("mg_scattering_d", mg_scattering.GetOnDevice());
  }
}

} // namespace Gas
