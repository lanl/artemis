//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

// C++ headers
#include <limits>

// Artemis includes
#include "artemis.hpp"
#include "gas.hpp"
#include "geometry/geometry.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/diffusion/diffusion.hpp"
#include "utils/diffusion/diffusion_coeff.hpp"
#include "utils/diffusion/momentum_diffusion.hpp"
#include "utils/diffusion/thermal_diffusion.hpp"
#include "utils/eos/eos.hpp"
#include "utils/fluxes/fluid_fluxes.hpp"
#include "utils/history.hpp"
#include "utils/opacity/opacity.hpp"
#include "utils/refinement/amr_criteria.hpp"
#include "utils/units.hpp"
#include "mhd/extended/defs.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace Gas {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Gas::Initialize
//! \brief Adds intialization function for gas hydrodynamics package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants,
                                            Packages_t &packages) {
  using namespace singularity::photons;

  auto gas = std::make_shared<StateDescriptor>("gas");
  Params &params = gas->AllParams();

  // Fluid behavior for this package
  Fluid fluid_type = Fluid::gas;
  params.Add("fluid_type", fluid_type);

  // Coordinates
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);
  params.Add("coords", coords);

  // Reconstruction algorithm
  ReconstructionMethod recon_method = ReconstructionMethod::null;
  const std::string recon = pin->GetOrAddString("gas", "reconstruct", "plm");
  if (recon.compare("pcm") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 1,
                      "PCM requires at least 1 ghost cell.");
    recon_method = ReconstructionMethod::pcm;
  } else if (recon.compare("plm") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                      "PLM requires at least 2 ghost cells.");
    recon_method = ReconstructionMethod::plm;
  } else if (recon.compare("plm_rho") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                      "PLM_PP requires at least 2 ghost cells.");
    recon_method = ReconstructionMethod::plm_rho;
  } else if (recon.compare("plm_pp") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                      "PLM_PP requires at least 2 ghost cells.");
    recon_method = ReconstructionMethod::plm_pp;
  } else if (recon.compare("plm_modPe") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                      "PLM_MODPE requires at least 2 ghost cells.");
    recon_method = ReconstructionMethod::plm_modPe;
  } else if (recon.compare("ppm") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 3,
                      "PPM requires at least 3 ghost cells.");
    if (coords != Coordinates::cartesian) {
      PARTHENON_WARN("Artemis' PPM implementation does not contain geometric corrections "
                     "for curvilinear coordinates.");
    }
    recon_method = ReconstructionMethod::ppm;
  } else {
    PARTHENON_FAIL("Reconstruction method not recognized.");
  }
  params.Add("recon", recon_method);

  // Riemann solver
  RSolver riemann_solver = RSolver::null;
  const std::string riemann = pin->GetOrAddString("gas", "riemann", "hllc");
  if (riemann.compare("hllc") == 0) {
    riemann_solver = RSolver::hllc;
  } else if (riemann.compare("hlle") == 0) {
    riemann_solver = RSolver::hlle;
  } else if (riemann.compare("llf") == 0) {
    riemann_solver = RSolver::llf;
  } else if (riemann.compare("hlld") == 0) {
    riemann_solver = RSolver::hlld;
  } else if (riemann.compare("llf_xmhd") == 0) {
    riemann_solver = RSolver::llf_xmhd;
  } else if (riemann.compare("hlld_xmhd") == 0) {
    riemann_solver = RSolver::hlld_xmhd;
  } else if (riemann.compare("llf_hall_xmhd") == 0) {
    riemann_solver = RSolver::llf_hall_xmhd;
  } else if (riemann.compare("hll_hall_xmhd") ==0) {
    riemann_solver = RSolver::hll_hall_xmhd;
  } else if (riemann.compare("hlle_hall_xmhd") ==0) {
    riemann_solver = RSolver::hlle_hall_xmhd;
  } else if (riemann.compare("hlldc_llf_hall_xmhd") == 0) {
    riemann_solver = RSolver::hlldc_llf_hall_xmhd;
  } else if (riemann.compare("hlldc_hall_xmhd") == 0) {
    riemann_solver = RSolver::hlldc_hall_xmhd;
  } else if (riemann.compare("hlldc_xmhd") == 0) {
    riemann_solver = RSolver::hlldc_xmhd;
  } else if (riemann.compare("hlldc_llf_xmhd") == 0) {
    riemann_solver = RSolver::hlldc_llf_xmhd;
  } else {
    PARTHENON_FAIL("Riemann solver (gas) not recognized.");
  }
  params.Add("rsolver", riemann_solver);

  // Courant, Friedrichs, & Lewy (CFL) Number
  const Real cfl_number = pin->GetOrAddReal("gas", "cfl", 0.8);
  params.Add("cfl", cfl_number);

  // YH: Added these for BCs that have time dependence
  params.Add("dt", 0.0, Params::Mutability::Restart);
  params.Add("ncycle", 0, Params::Mutability::Restart);

  // YH: To test which variables caused the dip in pressure
  const Real dW_idn = pin->GetOrAddReal("gas", "dW_idn", 1.);
  params.Add("dW_idn", dW_idn);
  const Real dW_ipr = pin->GetOrAddReal("gas", "dW_ipr", 1.);
  params.Add("dW_ipr", dW_ipr);
  const Real dW_ise = pin->GetOrAddReal("gas", "dW_ise", 1.);
  params.Add("dW_ise", dW_ise);
  const Real dW_ivx = pin->GetOrAddReal("gas", "dW_ivx", 1.);
  params.Add("dW_ivx", dW_ivx);
  const Real dW_ivy = pin->GetOrAddReal("gas", "dW_ivy", 1.);
  params.Add("dW_ivy", dW_ivy);
  const Real dW_ivz = pin->GetOrAddReal("gas", "dW_ivz", 1.);
  params.Add("dW_ivz", dW_ivz);
  const Real dW_ibx = pin->GetOrAddReal("gas", "dW_ibx", 1.);
  params.Add("dW_ibx", dW_ibx);
  const Real dW_iby = pin->GetOrAddReal("gas", "dW_iby", 1.);
  params.Add("dW_iby", dW_iby);
  const Real dW_ibz = pin->GetOrAddReal("gas", "dW_ibz", 1.);
  params.Add("dW_ibz", dW_ibz);
  const Real dW_iJx = pin->GetOrAddReal("gas", "dW_iJx", 1.);
  params.Add("dW_iJx", dW_iJx);
  const Real dW_iJy = pin->GetOrAddReal("gas", "dW_iJy", 1.);
  params.Add("dW_iJy", dW_iJy);
  const Real dW_iJz = pin->GetOrAddReal("gas", "dW_iJz", 1.);
  params.Add("dW_iJz", dW_iJz);
  const Real dW_iPe = pin->GetOrAddReal("gas", "dW_iPe", 1.);
  params.Add("dW_iPe", dW_iPe);

  // Equation of state
  const std::string eos_name = pin->GetOrAddString("gas", "eos", "ideal");
  if (eos_name == "ideal") {
    const Real gamma = pin->GetOrAddReal("gas", "gamma", 1.66666666667);
    auto cv = Null<Real>();
    auto mu = Null<Real>();
    if (pin->DoesParameterExist("gas", "cv")) {
      PARTHENON_REQUIRE(!pin->DoesParameterExist("gas", "mu"),
                        "Cannot specify both cv and mu");
      cv = pin->GetReal("gas", "cv");
      PARTHENON_REQUIRE(cv > 0, "Only positive cv allowed!");
      mu = constants.GetKBCode() / ((gamma - 1.) * constants.GetAMUCode() * cv);
    } else {
      mu = pin->GetOrAddReal("gas", "mu", 1.);
      PARTHENON_REQUIRE(mu > 0, "Only positive mean molecular weight allowed!");
      cv = constants.GetKBCode() / ((gamma - 1.) * constants.GetAMUCode() * mu);
    }
    params.Add("mu", mu);
    params.Add("cv", cv);
    EOS eos_host = singularity::UnitSystem<singularity::IdealGas>(
        singularity::IdealGas(gamma - 1., cv * units.GetSpecificHeatCodeToPhysical()),
        singularity::eos_units_init::LengthTimeUnitsInit(), units.GetTimeCodeToPhysical(),
        units.GetMassCodeToPhysical(), units.GetLengthCodeToPhysical(),
        units.GetTemperatureCodeToPhysical());
    EOS eos_device = eos_host.GetOnDevice();
    params.Add("eos_h", eos_host);
    params.Add("eos_d", eos_device);
    // TODO This needs to be removed when we convert everything to EOS calls
    params.Add("adiabatic_index", gamma);
  }

  // Opacity models
  const Real time = units.GetTimeCodeToPhysical();
  const Real mass = units.GetMassCodeToPhysical();
  const Real length = units.GetLengthCodeToPhysical();
  const Real temp = units.GetTemperatureCodeToPhysical();

  // Absorption opacity model
  std::string opacity_model_name =
      pin->GetOrAddString("gas/opacity/absorption", "opacity_model", "constant");

  // Mean absorption opacity (either read from table or uses model
  ArtemisUtils::MeanOpacity opacity;
  if (opacity_model_name == "table") {
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
    } else if (opacity_model_name == "constant") {
      const Real kappa_a = pin->GetOrAddReal("gas/opacity/absorption", "kappa_a", 0.0);
      auto model = Gray(kappa_a);
      opacity =
          singularity::photons::MeanNonCGSUnits<singularity::photons::MeanOpacityBase>(
              singularity::photons::MeanOpacityBase(model, lRhoMin_a, lRhoMax_a, NRho_a,
                                                    lTMin_a, lTMax_a, NT_a),
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
                                                    lTMin_a, lTMax_a, NT_a),
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
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityCGS>(
            singularity::photons::MeanSOpacityCGS(smodel, lRhoMin_s, lRhoMax_s, NRho_s,
                                                  lTMin_s, lTMax_s, NT_s),
            time, mass, length, temp);
  } else if (scattering_model_name == "constant") {
    const Real kappa_s = pin->GetOrAddReal("gas/opacity/scattering", "kappa_s", 0.0);
    auto smodel = GrayS(kappa_s, 1.0);
    scattering =
        singularity::photons::MeanNonCGSUnitsS<singularity::photons::MeanSOpacityCGS>(
            singularity::photons::MeanSOpacityCGS(smodel, lRhoMin_s, lRhoMax_s, NRho_s,
                                                  lTMin_s, lTMax_s, NT_s),
            time, mass, length, temp);
  } else {
    PARTHENON_FAIL("Scattering model not recognized!");
  }

  params.Add("scattering_h", scattering);
  params.Add("scattering_d", scattering.GetOnDevice());

  // Floors
  const Real dfloor = pin->GetOrAddReal("gas", "dfloor", 1.0e-20);
  const Real siefloor = pin->GetOrAddReal("gas", "siefloor", 1.0e-20);
  const Real Pefloor = pin->GetOrAddReal("mhd", "Pefloor", 1.0e-20);
  params.Add("dfloor", dfloor);
  params.Add("siefloor", siefloor);
  params.Add("Pefloor", Pefloor);

  // Dual energy switch
  // When internal > de_switch * total we use the total
  // The default turns off the switch
  const Real de_switch = pin->GetOrAddReal("gas", "de_switch", 0.0);
  params.Add("de_switch", de_switch);

  // Diffusion
  const bool do_viscosity = pin->GetOrAddBoolean("physics", "viscosity", false);
  params.Add("do_viscosity", do_viscosity);
  const bool do_conduction = pin->GetOrAddBoolean("physics", "conduction", false);
  params.Add("do_conduction", do_conduction);

  const bool do_diffusion = do_viscosity || do_conduction;
  params.Add("do_diffusion", do_diffusion);

  if (do_viscosity) {
    Diffusion::DiffCoeffParams dp("gas/viscosity", "viscosity", pin, constants, packages);
    params.Add("visc_params", dp);
  }
  if (do_conduction) {
    Diffusion::DiffCoeffParams dp("gas/conductivity", "conductivity", pin, constants,
                                  packages);
    params.Add("cond_params", dp);
  }

  // Number of gas species
  const int nspecies = pin->GetOrAddInteger("gas", "nspecies", 1);
  params.Add("nspecies", nspecies);
  std::vector<int> fluidids;
  for (int n = 0; n < nspecies; ++n)
    fluidids.push_back(n);

  // Scratch for gas flux
  const int scr_level = pin->GetOrAddInteger("gas", "scr_level", 0);
  params.Add("scr_level", scr_level);

  // Control field for sparse gas fields
  std::string control_field = gas::cons::density::name();

  // YH: Add user-defined metadata for ease
  Metadata::AddUserFlag("Density");
  Metadata::AddUserFlag("Velfield");
  Metadata::AddUserFlag("Energy");

  // Conserved Gas Density
  Metadata m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
		  	 Metadata::GetUserFlag("Density"),
                         Metadata::WithFluxes, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::cons::density>(m, control_field, fluidids);

  // Conserved Momenta
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Conserved, Metadata::GetUserFlag("Velfield"),
                Metadata::Independent, Metadata::WithFluxes, Metadata::Sparse},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::cons::momentum>(m, control_field, fluidids);

  // Conserved Gas Total Energy
  m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::WithFluxes,
                Metadata::Sparse, Metadata::Restart, Metadata::GetUserFlag("Energy")});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::cons::total_energy>(m, control_field, fluidids);

  // Auxillary Conserved Gas Thermal Energy (Volumetric, *Not* Specific)
  // not actually "conserved"
  m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                Metadata::Sparse, Metadata::WithFluxes});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::cons::internal_energy>(m, control_field, fluidids);

  // Primitive Density
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse, Metadata::GetUserFlag("Density")});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::prim::density>(m, control_field, fluidids);

  // Primitive Pressure (and associated Riemann pressures)
  Metadata::AddUserFlag("Pressure");
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::WithFluxes, Metadata::Sparse, 
		Metadata::FillGhost, Metadata::GetUserFlag("Pressure")});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::prim::pressure>(m, control_field, fluidids);

  // Primitive Velocities
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse, Metadata::GetUserFlag("Velfield")},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::prim::velocity>(m, control_field, fluidids);

  // Gas specific internal energy
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::prim::sie>(m, control_field, fluidids);

  // Normal face Velocity for PdV evaluation of internal energy
  m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::Sparse});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::face::velocity>(m, control_field, fluidids);

  // YH: add params for tracking time required in time-dep. BCs
  params.Add("track_time", 0.0, Params::Mutability::Restart);
  params.Add("track_dt", 1.e-16, Params::Mutability::Restart); 

  // YH: add type of slope limiter
  auto tvd_type = pin->GetOrAddString("gas","tvd_type","original");
  TVDType tvd_type_enum;
  if (tvd_type == "original") {
    tvd_type_enum = TVDType::Original;
  } else if (tvd_type == "van_albada") {
    tvd_type_enum = TVDType::VanAlbada;
  } else if (tvd_type == "minmod") {
    tvd_type_enum = TVDType::Minmod;
  } else if (tvd_type == "mc") {
    tvd_type_enum = TVDType::MC;
  } else if (tvd_type == "superbee") {
    tvd_type_enum = TVDType::Superbee;
  }
  params.Add("tvd_type", tvd_type_enum);

  // YH: Add magnetic field for mhd - add cc Bfield for now as some part of code used nvars=vprim.GetMaxNumberOfVars() 
  const bool do_mhd = pin->GetOrAddBoolean("physics", "mhd", false);
  params.Add("do_mhd", do_mhd);
  if (do_mhd) {
    // YH: ensure dimensional constants are consistent
    const Real vA_0 = B0/std::sqrt(n0*m_ion*mu0);
    const Real v_vs_vA = abs(vA_0-char_speed)/char_speed*100.;
    if (v_vs_vA > 1.) PARTHENON_FAIL("Error in dimensional constants as v NOT equal to vA!!!");

    auto eta_type = Null<std::string>();
    PARTHENON_REQUIRE(pin->DoesParameterExist("mhd", "eta_type"),
                    "Missing Ohmic resistive type selection!");
    eta_type = pin->GetString("mhd", "eta_type");
    EtaType eta_type_enum; // Convert string once on host to ensure device safe
    if (eta_type == "None") {
      eta_type_enum = EtaType::None;
    } else if (eta_type == "idealMHD") {
      eta_type_enum = EtaType::IdealMHD;
    } else if (eta_type == "Spitzer") {
      eta_type_enum = EtaType::Spitzer;
    } else {
      PARTHENON_REQUIRE(false, "No such eta model!");
    }
    params.Add("eta_type", eta_type_enum);

    auto ideal_Efield = Null<bool>();
    PARTHENON_REQUIRE(pin->DoesParameterExist("mhd", "ideal_Efield"),
		    "Do you need to initialize Efield using ideal MHD?");
    ideal_Efield = pin->GetBoolean("mhd", "ideal_Efield");
    params.Add("ideal_Efield", ideal_Efield);

    Metadata::AddUserFlag("Bfield");
    Metadata::AddUserFlag("Efield"); 
    Metadata::AddUserFlag("Efield_flux");
    Metadata::AddUserFlag("Jcurrent");
    Metadata::AddUserFlag("electron");
    Metadata::AddUserFlag("expEJ_E"); Metadata::AddUserFlag("expEJ_J");
    Metadata::AddUserFlag("EJ_src");

    auto MetadataBfield = Metadata::GetUserFlag("Bfield");
    auto MetadataEfield = Metadata::GetUserFlag("Efield");
    auto MetadataJcurrent = Metadata::GetUserFlag("Jcurrent");
    m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, MetadataBfield, Metadata::FillGhost},
               std::vector<int>({3}));
    gas->AddField<gas::prim::Bfield>(m);
	
    m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Conserved,
                  Metadata::WithFluxes, MetadataBfield},
               std::vector<int>({3}));
    gas->AddField<gas::cons::Bfield>(m);
    
    // YH: face-centered magnetic field
    m = Metadata({Metadata::Face, Metadata::Conserved, MetadataBfield,
                Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
	    		    parthenon::refinement_ops::RestrictAverage,
			    parthenon::refinement_ops::ProlongateInternalTothAndRoe>();
    gas->AddField<gas::face::bfield>(m);

    // YH: lambda for upw-CT (Mignone 2020)
    /*Metadata::AddUserFlag("VelL");
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
                  Metadata::GetUserFlag("VelL")}, std::vector<int>({3}));
    gas->AddField<gas::face::velL>(m);
    Metadata::AddUserFlag("VelR");
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
                  Metadata::GetUserFlag("VelR")}, std::vector<int>({3}));
    gas->AddField<gas::face::velR>(m);
    Metadata::AddUserFlag("LambdaL"); 
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
		  Metadata::GetUserFlag("LambdaL")});
    gas->AddField<gas::face::lambdaL>(m);
    Metadata::AddUserFlag("LambdaR");
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
                  Metadata::GetUserFlag("LambdaR")});
    gas->AddField<gas::face::lambdaR>(m);*/

    // YH: magnetic field should remain divergence-free
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataBfield});
    gas->AddField<gas::cons::divB>(m);
    // YH: check divE as related to quasi-neutrality
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, MetadataEfield});
    gas->AddField<gas::cons::divE>(m);

    // YH: place E field or its components through Ohm's law at cell vertices
    m = Metadata({Metadata::Edge, Metadata::Conserved, MetadataEfield,
		    Metadata::Independent, Metadata::FillGhost});
    gas->AddField<gas::edge::Efield>(m);
    m = Metadata({Metadata::Edge, Metadata::GetUserFlag("Efield_flux"), 
		    Metadata::FillGhost});
    gas->AddField<gas::edge::E_flux>(m);
    m = Metadata({Metadata::Node, Metadata::GetUserFlag("Efield_flux"), 
		    Metadata::FillGhost}, std::vector<int>({9}));
    gas->AddField<gas::node::E_flux>(m); // Add this to try correct asymmetry

    m = Metadata({Metadata::Edge, Metadata::Conserved, MetadataJcurrent,
                    Metadata::Independent, Metadata::FillGhost});
    gas->AddField<gas::edge::J>(m);
    m = Metadata({Metadata::Node, MetadataJcurrent, Metadata::Vector, Metadata::FillGhost},
		    std::vector<int>({9})); // YH: instead of 2D matrix just use 1D vector
    gas->AddField<gas::node::J_flux>(m);

    m = Metadata({Metadata::Node, Metadata::GetUserFlag("EJ_src"), Metadata::Vector, 
		    Metadata::FillGhost}, std::vector<int>({6})); 
    gas->AddField<gas::node::EJ_src>(m);

    // YH: place Efield & J at cell center for ease of coding for now
    m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Vector, MetadataEfield,
                    Metadata::Derived, Metadata::WithFluxes},
		    std::vector<int>({3}));
    gas->AddField<gas::cons::Efield>(m);
    m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                MetadataEfield, Metadata::OneCopy},
               std::vector<int>({3}));
    gas->AddField<gas::prim::Efield>(m); 

    m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Vector, MetadataJcurrent,
                    Metadata::Derived, Metadata::WithFluxes},
                    std::vector<int>({3}));
    gas->AddField<gas::cons::J>(m);
    m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                MetadataJcurrent, Metadata::OneCopy},
               std::vector<int>({3}));
    gas->AddField<gas::prim::J>(m);

    Metadata::AddUserFlag("Temp_ion");
    Metadata::AddUserFlag("Temp_elec");
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                  Metadata::FillGhost, Metadata::GetUserFlag("Temp_ion")});
    gas->AddField<gas::prim::Ti>(m);

    // YH: Electron-related variables
    m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::WithFluxes, 
                Metadata::Independent, Metadata::FillGhost, Metadata::GetUserFlag("electron")});
    gas->AddField<gas::cons::Se>(m);

    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
		  Metadata::FillGhost, Metadata::GetUserFlag("electron")});
    gas->AddField<gas::prim::Pe>(m);

    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                  Metadata::FillGhost, Metadata::GetUserFlag("Temp_elec")});
    gas->AddField<gas::prim::Te>(m);

    m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("expEJ_E")});
    gas->AddField<gas::source::expEJ_E>(m);
    m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("expEJ_J")});
    gas->AddField<gas::source::expEJ_J>(m);
    //YH: intermediate variables for computing source
    Metadata::AddUserFlag("NonIdeal");
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy,                  
		    Metadata::FillGhost, Metadata::GetUserFlag("NonIdeal")});
    gas->AddField<gas::source::nonideal_fcc>(m);
    m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("NonIdeal")});
    gas->AddField<gas::source::nonideal_edge>(m);
    Metadata::AddUserFlag("NonIdeal1");
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("NonIdeal1")});
    gas->AddField<gas::source::nonideal_fcc1>(m);
    m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("NonIdeal1")});
    gas->AddField<gas::source::nonideal_edge1>(m);
    Metadata::AddUserFlag("Relativistic");
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("Relativistic")});
    gas->AddField<gas::source::relativis_fcc>(m);
    m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("Relativistic")});
    gas->AddField<gas::source::relativis_edge>(m);

    // Store source term to update
    Metadata::AddUserFlag("Source_mom");
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("Source_mom")}
		    , std::vector<int>({3}));
    gas->AddField<gas::source::mom>(m);
    Metadata::AddUserFlag("Source_ener");
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("Source_ener")});
    gas->AddField<gas::source::ener>(m);
    Metadata::AddUserFlag("Source_Se");
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("Source_Se")});
    gas->AddField<gas::source::Se>(m);
    Metadata::AddUserFlag("Source_Bfield");
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy,
                    Metadata::FillGhost, Metadata::GetUserFlag("Source_Bfield")});
    gas->AddField<gas::source::Bfield>(m);


    // Boundary stuffs 
    const std::string ox1_bctype = pin->GetString("parthenon/mesh", "ox1_bc");
    if (ox1_bctype=="nrbc" or ox1_bctype=="orlanski") {
      Metadata::AddUserFlag("Boundary");
      m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse, Metadata::GetUserFlag("Boundary")});
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::density>(m, control_field, fluidids);
      m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::WithFluxes, Metadata::Sparse,
                Metadata::FillGhost, Metadata::GetUserFlag("Boundary")});
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::pressure>(m, control_field, fluidids);
      m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse, 
		Metadata::GetUserFlag("Boundary")}, std::vector<int>({3}));
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::velocity>(m, control_field, fluidids);
      m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse,
		Metadata::GetUserFlag("Boundary")}, std::vector<int>({3}));
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::Bfield>(m, control_field, fluidids);
      m = Metadata({Metadata::Face, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse,
                Metadata::GetUserFlag("Boundary")});
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::Bfield_fcc>(m, control_field, fluidids);
      m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse,
                Metadata::GetUserFlag("Boundary")});
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::Efield>(m, control_field, fluidids);
      m = Metadata({Metadata::Edge, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse,
                Metadata::GetUserFlag("Boundary")});
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::J>(m, control_field, fluidids);
      m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,                Metadata::FillGhost, Metadata::Sparse, Metadata::GetUserFlag("Boundary")});
      m.SetSparseThresholds(0.0, 0.0, 0.0);
      gas->AddSparsePool<gas::boundary::Pe>(m, control_field, fluidids);
    }
  }

  if (do_diffusion) {
    m = Metadata({Metadata::Face, Metadata::Flux, Metadata::Sparse},
                 std::vector<int>({3}));
    ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
    gas->AddSparsePool<gas::diff::momentum>(m, control_field, fluidids);
    m = Metadata({Metadata::Face, Metadata::Flux, Metadata::Sparse});
    ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
    gas->AddSparsePool<gas::diff::energy>(m, control_field, fluidids);
  }

  // Gas hydrodynamics timestep
  if (coords == Coordinates::cartesian) {
    gas->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cartesian>;
  } else if (coords == Coordinates::spherical1D) {
    gas->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical1D>;
  } else if (coords == Coordinates::spherical2D) {
    gas->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical2D>;
  } else if (coords == Coordinates::spherical3D) {
    gas->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical3D>;
  } else if (coords == Coordinates::cylindrical) {
    gas->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cylindrical>;
  } else if (coords == Coordinates::axisymmetric) {
    gas->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::axisymmetric>;
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }

  // Gas refinement criterion
  const std::string refine_field = pin->GetOrAddString("gas", "refine_field", "none");
  if (refine_field != "none") {
    // Check which field controls the refinement
    const bool ref_dens = (refine_field == "density");
    const bool ref_pres = (refine_field == "pressure");
    PARTHENON_REQUIRE((ref_dens || ref_pres) && !(ref_dens && ref_pres),
                      "Only density or pressure based criterion currently supported!");

    // Check the type of refinement (e.g., gradient vs magnitude)
    const std::string refine_type = pin->GetString("gas", "refine_type");
    const bool ref_grad = (refine_type == "gradient");
    const bool ref_mag = (refine_type == "magnitude");
    PARTHENON_REQUIRE((ref_grad || ref_mag) && !(ref_grad && ref_mag),
                      "Only gradient or magnitude based criterion currently supported!");

    // Specify appropriate AMR criterion callback
    if (ref_grad) {
      using ArtemisUtils::ScalarFirstDerivative;
      // Refinement threshold
      const Real thr = pin->GetReal("gas", "refine_thr");
      params.Add("refine_thr", thr);
      // Geometry specific refinement criteria
      typedef Coordinates C;
      typedef gas::prim::density pdens;
      typedef gas::prim::pressure ppres;
      // Cartesian
      if (coords == C::cartesian) {
        if (ref_dens) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::cartesian>;
        } else if (ref_pres) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::cartesian>;
        }
        // Spherical
      } else if (coords == C::spherical1D) {
        if (ref_dens) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::spherical1D>;
        } else if (ref_pres) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::spherical1D>;
        }
      } else if (coords == C::spherical2D) {
        if (ref_dens) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::spherical2D>;
        } else if (ref_pres) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::spherical2D>;
        }
      } else if (coords == C::spherical3D) {
        if (ref_dens) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::spherical3D>;
        } else if (ref_pres) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::spherical3D>;
        }
        // Cylindrical
      } else if (coords == C::cylindrical) {
        if (ref_dens) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::cylindrical>;
        } else if (ref_pres) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::cylindrical>;
        }
        // Axisymmetric
      } else if (coords == C::axisymmetric) {
        if (ref_dens) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::axisymmetric>;
        } else if (ref_pres) {
          gas->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::axisymmetric>;
        }
      }
    } else if (ref_mag) {
      using ArtemisUtils::ScalarMagnitude;
      const Real rthr = pin->GetReal("gas", "refine_thr");
      const Real dthr = pin->GetReal("gas", "deref_thr");
      params.Add("refine_thr", rthr);
      params.Add("deref_thr", dthr);
      if (ref_dens) {
        gas->CheckRefinementBlock = ScalarMagnitude<gas::prim::density>;
      } else if (ref_pres) {
        gas->CheckRefinementBlock = ScalarMagnitude<gas::prim::pressure>;
      }
    }
  }

  return gas;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Gas::EstimateTimestepMesh
//! \brief Compute gas hydrodynamics timestep
template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using RotatingFrame::BackgroundVelocity;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->Param<Real>("adiabatic_index"); 
  auto &params = gas_pkg->AllParams();
  auto eos_d = params.template Get<EOS>("eos_d");

  // YH: add mhd
  auto do_mhd = pm->packages.Get("artemis")->Param<bool>("do_mhd");

  // NOTE(@pdmullen): Without FARGO, dt must be additionally limited by the linear
  // advection of the shear background flow (vy0 = -q Omega x)
  bool do_shear = false;
  Real qshear = 0.0, om0 = 0.0;
  if (pm->packages.Get("artemis")->Param<bool>("do_rotating_frame")) {
    auto &rframe_pkg = pm->packages.Get("rotating_frame");
    qshear = rframe_pkg->Param<Real>("qshear");
    om0 = rframe_pkg->Param<Real>("omega");
    do_shear = (qshear * om0 != 0.0);
  }

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie, 
	  gas::prim::Bfield,
	  gas::prim::J,gas::prim::Pe>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;

  Real min_dt = Big<Real>();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Gas::EstimateTimestepMesh", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

        for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
          const Real &dens = vmesh(b, gas::prim::density(n), k, j, i);
          const Real &sie = vmesh(b, gas::prim::sie(n), k, j, i);
          const Real bulk = eos_d.BulkModulusFromDensityInternalEnergy(dens, sie);
	  Real a = std::sqrt(bulk / dens);
	  Real ne = Null<Real>(); // YH: For XMHD
	  Real ca = Null<Real>(); // YH: For MHD
	  if (do_mhd) { // YH: account for fast magnetosonic speed at timestep
	    Real bx = vmesh(b, gas::prim::Bfield(0), k, j, i);
	    Real by = vmesh(b, gas::prim::Bfield(1), k, j, i);
	    Real bz = vmesh(b, gas::prim::Bfield(2), k, j, i);
	    ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
	    Real cax = std::sqrt((SQR(bx))/dens);
	    Real cay = std::sqrt((SQR(by))/dens);
	    Real caz = std::sqrt((SQR(bz))/dens);
	    Real can = std::min(cax,cay);
	    can = std::min(can,caz);
	    Real csa = SQR(a) + SQR(ca);	    
	    Real acan = a * can;
	    Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
	    a = std::max(a,cf);

	    // YH: account for electron speed
	    Real press = bulk / gamma;
	    Real a_elec = std::sqrt(bulk / dens) * sqrt(m_ion/me);
	    a = std::max(a, a_elec);//std::max(a_elec, reduced_c));
	  }
	  const Real cs = a;
          Real denom = 0.0;
	  Real mindx = 1.e9;
          for (int d = 0; d < ndim; d++) {
	    Real vel = vmesh(b, gas::prim::velocity(VI(n, d)), k, j, i);
	    if (do_mhd) { 
		    ne = Z_ion * vmesh(b, gas::prim::density(n), k, j, i);
		    Real ue = vel - (J0/(e_charge*n0*char_speed))*
			    (vmesh(b, gas::prim::J(d), k, j, i)/ne);
		    vel = std::max(abs(vel), abs(ue));
	    }
            denom +=
                (std::abs(vel) + cs) / dx[d];
	    mindx = std::min(mindx, dx[d]);
          }
          ldt = std::min(ldt, 1.0 / denom);
	  /*if (do_mhd) {
	    const Real Pe = vmesh(b, gas::prim::Pe(), k, j, i);
	    const Real eta = resistivity::Spitzer(ne,Pe);
	    ldt = std::min(ldt, (eta/(t0*mu0))*(1./SQR(ca)));
	    //ldt = std::min(ldt, mindx/reduced_c);
	  }*/
	}

        if (do_shear) {
          const auto ww = BackgroundVelocity<GEOM>(qshear, om0, coords.x1v());
          ldt = std::min(ldt, dx[1] / std::abs(ww[1]));
        }
      },
      Kokkos::Min<Real>(min_dt));

  Real visc_dt = Big<Real>();
  const auto do_viscosity = params.template Get<bool>("do_viscosity");
  if (do_viscosity) {
    auto dp = params.template Get<Diffusion::DiffCoeffParams>("visc_params");
    if (dp.type == Diffusion::DiffType::viscosity_plaw) {
      visc_dt = Diffusion::EstimateTimestep<GEOM, Fluid::gas,
                                            Diffusion::DiffType::viscosity_plaw>(
          md, dp, gas_pkg, eos_d, vmesh);
    } else if (dp.type == Diffusion::DiffType::viscosity_alpha) {
      visc_dt = Diffusion::EstimateTimestep<GEOM, Fluid::gas,
                                            Diffusion::DiffType::viscosity_alpha>(
          md, dp, gas_pkg, eos_d, vmesh);
    }
  }

  Real cond_dt = Big<Real>();
  const auto do_conduction = params.template Get<bool>("do_conduction");
  if (do_conduction) {
    auto dp = params.template Get<Diffusion::DiffCoeffParams>("cond_params");
    if (dp.type == Diffusion::DiffType::conductivity_plaw) {
      cond_dt = Diffusion::EstimateTimestep<GEOM, Fluid::gas,
                                            Diffusion::DiffType::conductivity_plaw>(
          md, dp, gas_pkg, eos_d, vmesh);
    } else if (dp.type == Diffusion::DiffType::thermaldiff_plaw) {
      cond_dt = Diffusion::EstimateTimestep<GEOM, Fluid::gas,
                                            Diffusion::DiffType::thermaldiff_plaw>(
          md, dp, gas_pkg, eos_d, vmesh);
    }
  }
  Real diff_dt = std::min(visc_dt, cond_dt);

  const auto cfl_number = params.template Get<Real>("cfl");
  return cfl_number * std::min(min_dt, diff_dt);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gas::CalculateFluxes
//! \brief Evaluates advective fluxes for gas evolution
TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &pkg = pm->packages.Get("gas");
  Params &params = pm->packages.Get("artemis")->AllParams();
  const bool do_mhd = params.Get<bool>("do_mhd");

  static auto desc_prim =
      parthenon::MakePackDescriptor<gas::prim::density, gas::prim::velocity,
                                    gas::prim::pressure, gas::prim::sie>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  static auto desc_flux =
      parthenon::MakePackDescriptor<gas::cons::density, gas::cons::momentum,
                                    gas::cons::total_energy, gas::cons::internal_energy>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes}); 
  static auto desc_face =
      parthenon::MakePackDescriptor<gas::face::velocity>(resolved_pkgs.get()); 
  auto vprim = desc_prim.GetPack(md);
  auto vflux = desc_flux.GetPack(md);
  auto vface = desc_face.GetPack(md);

  if (do_mhd) { // YH: add mhd
    static auto desc_prim_mhd =
      parthenon::MakePackDescriptor<gas::prim::density, gas::prim::velocity,
                                    gas::prim::pressure, gas::prim::sie,
				    gas::prim::Bfield,
				    gas::prim::Efield,
				    gas::prim::J,
				    gas::prim::Pe>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
    static auto desc_flux_mhd =
      parthenon::MakePackDescriptor<gas::cons::density, gas::cons::momentum,
                                    gas::cons::total_energy, gas::cons::internal_energy,
				    gas::cons::Bfield,
				    gas::cons::Efield,
				    gas::cons::J,
				    gas::cons::Se>( // YH: for mhd
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
    static auto desc_face_mhd =
      parthenon::MakePackDescriptor<gas::face::velocity,
      				    gas::face::bfield>(resolved_pkgs.get());
				    //gas::face::velL, // YH: for upw CT (Mignone 2020)
                                    //gas::face::velR,
				    //gas::face::lambdaL,
				    //gas::face::lambdaR>(resolved_pkgs.get()); 
    auto vprim_mhd = desc_prim_mhd.GetPack(md);
    auto vflux_mhd = desc_flux_mhd.GetPack(md);
    auto vface_mhd = desc_face_mhd.GetPack(md);
    return ArtemisUtils::CalculateFluxes<Fluid::gas>(md, pkg, vprim_mhd, vflux_mhd, vface_mhd, pcm);
  }

  return ArtemisUtils::CalculateFluxes<Fluid::gas>(md, pkg, vprim, vflux, vface, pcm); 
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gas::FluxSource
//! \brief Evaluates coordinate terms from advective fluxes for gas evolution
TaskStatus FluxSource(MeshData<Real> *md, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &pkg = pm->packages.Get("gas");

  static auto desc_prim =
      parthenon::MakePackDescriptor<gas::prim::density, gas::prim::velocity,
                                    gas::prim::pressure>(resolved_pkgs.get(), {},
                                                         {parthenon::PDOpt::WithFluxes});
  static auto desc_cons =
      parthenon::MakePackDescriptor<gas::cons::momentum, gas::cons::internal_energy>(
          resolved_pkgs.get());
  static auto desc_face =
      parthenon::MakePackDescriptor<gas::face::velocity>(resolved_pkgs.get());
  auto vprim = desc_prim.GetPack(md);
  auto vcons = desc_cons.GetPack(md);
  auto vface = desc_face.GetPack(md);

  return ArtemisUtils::FluxSource(md, pkg, vprim, vcons, vface, dt);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gas::ViscousFlux
//  \brief Evaluates viscous flux
template <Coordinates GEOM>
TaskStatus ViscousFlux(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &pkg = pm->packages.Get("gas");

  const auto dp = pkg->template Param<Diffusion::DiffCoeffParams>("visc_params");

  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc_prim =
      parthenon::MakePackDescriptor<gas::prim::density, gas::prim::velocity,
                                    gas::prim::sie>(resolved_pkgs.get());

  // Assumes this packing ordering
  static auto desc_flux =
      parthenon::MakePackDescriptor<gas::diff::momentum, gas::diff::energy>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});

  auto vprim = desc_prim.GetPack(md);
  auto vf = desc_flux.GetPack(md);

  if (dp.type == Diffusion::DiffType::null) {
    return TaskStatus::complete;
  } else if (dp.type == Diffusion::DiffType::viscosity_plaw) {
    return Diffusion::MomentumFluxImpl<GEOM, Fluid::gas,
                                       Diffusion::DiffType::viscosity_plaw>(md, dp, pkg,
                                                                            vprim, vf);
  } else if (dp.type == Diffusion::DiffType::viscosity_alpha) {
    return Diffusion::MomentumFluxImpl<GEOM, Fluid::gas,
                                       Diffusion::DiffType::viscosity_alpha>(md, dp, pkg,
                                                                             vprim, vf);
  } else {
    PARTHENON_FAIL("Invalid viscosity type");
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gas::ThermalFlux
//  \brief Evaluates thermal flux
template <Coordinates GEOM>
TaskStatus ThermalFlux(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &pkg = pm->packages.Get("gas");

  const auto dp = pkg->template Param<Diffusion::DiffCoeffParams>("cond_params");

  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc_prim =
      parthenon::MakePackDescriptor<gas::prim::density, gas::prim::sie>(
          resolved_pkgs.get());

  // Assumes this packing ordering
  static auto desc_flux = parthenon::MakePackDescriptor<gas::diff::energy>(
      resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});

  auto vprim = desc_prim.GetPack(md);
  auto vf = desc_flux.GetPack(md);

  if (dp.type == Diffusion::DiffType::null) {
    return TaskStatus::complete;
  } else if (dp.type == Diffusion::DiffType::conductivity_plaw) {
    return Diffusion::ThermalFluxImpl<GEOM, Fluid::gas,
                                      Diffusion::DiffType::conductivity_plaw>(md, dp, pkg,
                                                                              vprim, vf);
  } else if (dp.type == Diffusion::DiffType::thermaldiff_plaw) {
    return Diffusion::ThermalFluxImpl<GEOM, Fluid::gas,
                                      Diffusion::DiffType::thermaldiff_plaw>(md, dp, pkg,
                                                                             vprim, vf);
  } else {
    PARTHENON_FAIL("Invalid conductivity type");
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gas::ZeroDiffusionFlux
//  \brief Resets the diffusion flux
TaskStatus ZeroDiffusionFlux(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &pkg = pm->packages.Get("gas");

  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc_flux =
      parthenon::MakePackDescriptor<gas::diff::momentum, gas::diff::energy>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});

  auto vf = desc_flux.GetPack(md);
  return Diffusion::ZeroDiffusionImpl(md, vf);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gas::DiffusionUpdate
//  \brief Applies the diffusion fluxes to update the momenta and energy
template <Coordinates GEOM>
TaskStatus DiffusionUpdate(MeshData<Real> *md, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &pkg = pm->packages.Get("gas");

  const auto do_viscosity = pkg->template Param<bool>("do_viscosity");

  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc_cons =
      parthenon::MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy,
                                    gas::cons::internal_energy>(resolved_pkgs.get());
  static auto desc_prim =
      parthenon::MakePackDescriptor<gas::prim::velocity>(resolved_pkgs.get());

  // Assumes this packing ordering
  static auto desc_flux =
      parthenon::MakePackDescriptor<gas::diff::momentum, gas::diff::energy>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});

  auto vcons = desc_cons.GetPack(md);
  auto vprim = desc_prim.GetPack(md);
  auto vf = desc_flux.GetPack(md);

  return Diffusion::DiffusionUpdateImpl<GEOM, Fluid::gas>(md, pkg, vcons, vprim, vf,
                                                          do_viscosity, dt);
}

//----------------------------------------------------------------------------------------
//! \fn  void Gas::AddHistoryImpl
//! \brief Add history outputs for gas quantities for generic coordinate system
template <Coordinates GEOM>
void AddHistoryImpl(Params &params) {
  using namespace ArtemisUtils;
  auto HstSum = parthenon::UserHistoryOperation::sum;
  using parthenon::HistoryOutputVar;
  parthenon::HstVec_list hst_vecs = {};

  // Mass
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVolumeIntegral<GEOM, gas::cons::density>, "gas_mass"));

  // Momenta
  typedef gas::cons::momentum cmom;
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVectorVolumeIntegral<GEOM, X1DIR, cmom>, "gas_momentum_x1"));
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVectorVolumeIntegral<GEOM, X2DIR, cmom>, "gas_momentum_x2"));
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVectorVolumeIntegral<GEOM, X3DIR, cmom>, "gas_momentum_x3"));

  // Energy (total and internal)
  typedef gas::cons::total_energy cte;
  typedef gas::cons::internal_energy cie;
  hst_vecs.emplace_back(
      HistoryOutputVec(HstSum, ReduceSpeciesVolumeIntegral<GEOM, cte>, "gas_energy"));
  hst_vecs.emplace_back(HistoryOutputVec(HstSum, ReduceSpeciesVolumeIntegral<GEOM, cie>,
                                         "gas_internal_energy"));

  params.Add(parthenon::hist_vec_param_key, hst_vecs);
}

//----------------------------------------------------------------------------------------
//! \fn  void Gas::AddHistory
//! \brief Add history outputs for gas quantities
void AddHistory(Coordinates coords, Params &params) {
  if (coords == Coordinates::cartesian) {
    AddHistoryImpl<Coordinates::cartesian>(params);
  } else if (coords == Coordinates::cylindrical) {
    AddHistoryImpl<Coordinates::cylindrical>(params);
  } else if (coords == Coordinates::spherical1D) {
    AddHistoryImpl<Coordinates::spherical1D>(params);
  } else if (coords == Coordinates::spherical2D) {
    AddHistoryImpl<Coordinates::spherical2D>(params);
  } else if (coords == Coordinates::spherical3D) {
    AddHistoryImpl<Coordinates::spherical3D>(params);
  } else if (coords == Coordinates::axisymmetric) {
    AddHistoryImpl<Coordinates::axisymmetric>(params);
  }
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef MeshData<Real> MD;
template Real EstimateTimestepMesh<Coordinates::cartesian>(MD *md);
template Real EstimateTimestepMesh<Coordinates::cylindrical>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical1D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical2D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical3D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::axisymmetric>(MD *md);

template TaskStatus ViscousFlux<Coordinates::cartesian>(MD *md);
template TaskStatus ViscousFlux<Coordinates::spherical1D>(MD *md);
template TaskStatus ViscousFlux<Coordinates::spherical2D>(MD *md);
template TaskStatus ViscousFlux<Coordinates::spherical3D>(MD *md);
template TaskStatus ViscousFlux<Coordinates::cylindrical>(MD *md);
template TaskStatus ViscousFlux<Coordinates::axisymmetric>(MD *md);

template TaskStatus ThermalFlux<Coordinates::cartesian>(MD *md);
template TaskStatus ThermalFlux<Coordinates::spherical1D>(MD *md);
template TaskStatus ThermalFlux<Coordinates::spherical2D>(MD *md);
template TaskStatus ThermalFlux<Coordinates::spherical3D>(MD *md);
template TaskStatus ThermalFlux<Coordinates::cylindrical>(MD *md);
template TaskStatus ThermalFlux<Coordinates::axisymmetric>(MD *md);

template TaskStatus DiffusionUpdate<Coordinates::cartesian>(MD *md, const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::spherical1D>(MD *md, const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::spherical2D>(MD *md, const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::spherical3D>(MD *md, const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::cylindrical>(MD *md, const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::axisymmetric>(MD *md, const Real dt);

} // namespace Gas
