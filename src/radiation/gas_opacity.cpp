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
void InitGasOpacity(ParameterInput *pin, const ArtemisUtils::Units &units, Params &params,
                    const std::string &radblock_name) {
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

  // check if using groups from parsed-in opacity table
  const bool use_opac_grps =
      pin->GetOrAddBoolean(radblock_name, "use_opac_groups", false);
  PARTHENON_REQUIRE(
      use_opac_grps ? opacity_model_name == "table" : true,
      "Opacity group bounds can only be used with opacity_model_name=table!");

  // set opacity group bounds: gray goes from 0 to infty
  std::vector<Real> opac_grp_bnds = {0.0, std::numeric_limits<Real>::infinity()};
  // reset to parsed input (for non-tabular opacity models) for multigroup
  if (frequency_type == FrequencyType::multigroup && !use_opac_grps) {
    const Real numin = pin->GetReal(radblock_name, "numin"); // in Hz
    const Real numax = pin->GetReal(radblock_name, "numax"); // in Hz
    const int n_nubins = pin->GetInteger(radblock_name, "n_nubins");
    // reset opacity group bounds
    // NOTE: these are group edges, not interior points
    opac_grp_bnds.assign(n_nubins + 1, 0.0);
    // assume uniform log-spacing, grid is midpoints in log-space
    const Real dlnu = (std::log(numax) - std::log(numin)) / n_nubins;
    for (int n = 0; n < n_nubins + 1; ++n) {
      opac_grp_bnds[n] = numin * std::exp(n * dlnu);
    }
  }
  int NG = static_cast<int>(opac_grp_bnds.size()) - 1;

  // Mean absorption opacity (either read from table or uses model
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
                                                    lTMin_a, lTMax_a, NT_a, opac_grp_bnds,
                                                    NG),
              time, mass, length, temp);
    } else if (opacity_model_name == "constant") {
      const Real kappa_a = pin->GetOrAddReal("gas/opacity/absorption", "kappa_a", 0.0);
      auto model = Gray(kappa_a);
      opacity =
          singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
              singularity::photons::MeanOpacityBase(model, lRhoMin_a, lRhoMax_a, NRho_a,
                                                    lTMin_a, lTMax_a, NT_a, opac_grp_bnds,
                                                    NG),
              time, mass, length, temp);
    } else if (opacity_model_name == "powerlaw") {
      const Real coef_kappa_a =
          pin->GetOrAddReal("gas/opacity/absorption", "coef_kappa_a", 0.0);
      const Real rho_exp = pin->GetOrAddReal("gas/opacity/absorption", "rho_exp", 0.0);
      const Real temp_exp = pin->GetOrAddReal("gas/opacity/absorption", "temp_exp", 0.0);
      auto model = PowerLaw(coef_kappa_a, rho_exp, temp_exp);
      opacity =
          singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
              singularity::photons::MeanOpacityBase(model, lRhoMin_a, lRhoMax_a, NRho_a,
                                                    lTMin_a, lTMax_a, NT_a, opac_grp_bnds,
                                                    NG),
              time, mass, length, temp);
    } else {
      PARTHENON_FAIL("Opacity model not recognized!");
    }
  }

  params.Add("opacity_h", opacity);
  params.Add("opacity_d", opacity.GetOnDevice());

  // Scattering opacity model
  std::string scattering_model_name =
      pin->GetOrAddString("gas/opacity/scattering", "scattering_model", "none");

  // ensure analytic scattering uses tabular absorption group bounds
  // TODO: table scattering opacity
  if (use_opac_grps && opacity_model_name == "table") {
    opac_grp_bnds = opacity.GetGroupBounds();
    NG = opacity.ngroups();
  }

  // Instantiate mean scattering opacity object (i.e., table)
  const Real lRhoMin_s = pin->GetOrAddReal("gas/opacity/scattering", "lRhoMin", -1.0);
  const Real lRhoMax_s = pin->GetOrAddReal("gas/opacity/scattering", "lRhoMax", 1.0);
  const int NRho_s = pin->GetOrAddInteger("gas/opacity/scattering", "NRho", 2);
  const Real lTMin_s = pin->GetOrAddReal("gas/opacity/scattering", "lTMin", -1.0);
  const Real lTMax_s = pin->GetOrAddReal("gas/opacity/scattering", "lTMax", 1.0);
  const int NT_s = pin->GetOrAddInteger("gas/opacity/scattering", "NT", 2);

  ArtemisUtils::MeanScattering scattering;
  if (scattering_model_name == "none") {
    auto smodel = GrayS(0.0, 1.0);
    scattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityBase>(
            singularity::photons::MeanSOpacityBase(smodel, lRhoMin_s, lRhoMax_s, NRho_s,
                                                   lTMin_s, lTMax_s, NT_s, opac_grp_bnds,
                                                   NG),
            time, mass, length, temp);
  } else if (scattering_model_name == "constant") {
    const Real kappa_s = pin->GetOrAddReal("gas/opacity/scattering", "kappa_s", 0.0);
    auto smodel = GrayS(kappa_s, 1.0);
    scattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityBase>(
            singularity::photons::MeanSOpacityBase(smodel, lRhoMin_s, lRhoMax_s, NRho_s,
                                                   lTMin_s, lTMax_s, NT_s, opac_grp_bnds,
                                                   NG),
            time, mass, length, temp);
  } else {
    PARTHENON_FAIL("Scattering model not recognized!");
  }

  params.Add("scattering_h", scattering);
  params.Add("scattering_d", scattering.GetOnDevice());
}

} // namespace Gas
