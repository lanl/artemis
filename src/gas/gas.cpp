//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
  } else {
    PARTHENON_FAIL("Riemann solver (gas) not recognized.");
  }
  params.Add("rsolver", riemann_solver);

  // Courant, Friedrichs, & Lewy (CFL) Number
  const Real cfl_number = pin->GetOrAddReal("gas", "cfl", 0.8);
  params.Add("cfl", cfl_number);

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
    auto zbar = pin->GetOrAddReal("gas", "zbar", 1.0);
    EOS eos_host = singularity::IdealGas(gamma - 1., cv, {mu, zbar});
    EOS eos_device = eos_host.GetOnDevice();
    params.Add("eos_h", eos_host);
    params.Add("eos_d", eos_device);
    // TODO This needs to be removed when we convert everything to EOS calls
    params.Add("adiabatic_index", gamma);
  }

  // Absorption opacity model
  // TODO(@pdmullen): This may not be the right place for this... how about dust opacity?
  ArtemisUtils::Opacity opacity;
  std::string opacity_model_name =
      pin->GetOrAddString("gas/opacity/absorption", "opacity_model", "constant");
  const Real length = units.GetLengthCodeToPhysical();
  const Real time = units.GetTimeCodeToPhysical();
  const Real mass = units.GetMassCodeToPhysical();
  if (opacity_model_name == "none") {
    opacity = NonCGSUnits<Gray>(Gray(0.0), time, mass, length, 1.);
  } else if (opacity_model_name == "constant") {
    const Real kappa_a = pin->GetOrAddReal("gas/opacity/absorption", "kappa_a", 0.0);
    opacity = NonCGSUnits<Gray>(Gray(kappa_a), time, mass, length, 1.);
  } else if (opacity_model_name == "shocktube_a") {
    const Real coef_kappa_a =
        pin->GetOrAddReal("gas/opacity/absorption", "coef_kappa_a", 0.0);
    const Real rho_exp = pin->GetOrAddReal("gas/opacity/absorption", "rho_exp", 0.0);
    const Real temp_exp = pin->GetOrAddReal("gas/opacity/absorption", "temp_exp", 0.0);
    opacity = ArtemisUtils::ShocktubeAOpacity(coef_kappa_a, rho_exp, temp_exp);
  } else if (opacity_model_name == "thermalization") {
    const Real kappa_a = pin->GetOrAddReal("gas/opacity/absorption", "kappa_a", 0.0);
    opacity = ArtemisUtils::ThermalizationOpacity(kappa_a);
  } else {
    PARTHENON_FAIL("Opacity model not recognized!");
  }
  params.Add("opacity_h", opacity);
  params.Add("opacity_d", opacity.GetOnDevice());

  // Scattering opacity model
  // TODO(@pdmullen): This may not be the right place for this... how about dust opacity?
  ArtemisUtils::Scattering scattering;
  std::string scattering_model_name =
      pin->GetOrAddString("gas/opacity/scattering", "scattering_model", "none");
  if (scattering_model_name == "none") {
    scattering = NonCGSUnitsS<GrayS>(GrayS(0.0, 1.0), time, mass, length, 1.);
  } else if (scattering_model_name == "constant") {
    const Real kappa_s = pin->GetOrAddReal("gas/opacity/scattering", "kappa_s", 0.0);
    scattering = NonCGSUnitsS<GrayS>(GrayS(kappa_s, 1.0), time, mass, length, 1.);
  } else {
    PARTHENON_FAIL("Scattering model not recognized!");
  }
  params.Add("scattering_h", scattering);
  params.Add("scattering_d", scattering.GetOnDevice());

  // Floors
  const Real dfloor = pin->GetOrAddReal("gas", "dfloor", 1.0e-20);
  const Real siefloor = pin->GetOrAddReal("gas", "siefloor", 1.0e-20);
  params.Add("dfloor", dfloor);
  params.Add("siefloor", siefloor);

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

  // Conserved Gas Density
  Metadata m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                         Metadata::WithFluxes, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::cons::density>(m, control_field, fluidids);

  // Conserved Momenta
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Conserved,
                Metadata::Independent, Metadata::WithFluxes, Metadata::Sparse},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::cons::momentum>(m, control_field, fluidids);

  // Conserved Gas Total Energy
  m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::WithFluxes,
                Metadata::Sparse, Metadata::Restart});
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
                Metadata::FillGhost, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::prim::density>(m, control_field, fluidids);

  // Primitive Pressure (and associated Riemann pressures)
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::WithFluxes, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  gas->AddSparsePool<gas::prim::pressure>(m, control_field, fluidids);

  // Primitive Velocities
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse},
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
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &gas_pkg = pm->packages.Get("gas");
  auto &params = gas_pkg->AllParams();
  auto eos_d = params.template Get<EOS>("eos_d");

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
          resolved_pkgs.get());
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
          const Real cs = std::sqrt(bulk / dens);
          Real denom = 0.0;
          for (int d = 0; d < ndim; d++) {
            const Real ss =
                std::abs(vmesh(b, gas::prim::velocity(VI(n, d)), k, j, i)) + cs;
            denom += ss / dx[d];
          }
          ldt = std::min(ldt, 1.0 / denom);
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
template Real EstimateTimestepMesh<Coordinates::cartesian>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::cylindrical>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::spherical1D>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::spherical2D>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::spherical3D>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::axisymmetric>(MeshData<Real> *md);

template TaskStatus ViscousFlux<Coordinates::cartesian>(MeshData<Real> *md);
template TaskStatus ViscousFlux<Coordinates::spherical1D>(MeshData<Real> *md);
template TaskStatus ViscousFlux<Coordinates::spherical2D>(MeshData<Real> *md);
template TaskStatus ViscousFlux<Coordinates::spherical3D>(MeshData<Real> *md);
template TaskStatus ViscousFlux<Coordinates::cylindrical>(MeshData<Real> *md);
template TaskStatus ViscousFlux<Coordinates::axisymmetric>(MeshData<Real> *md);

template TaskStatus ThermalFlux<Coordinates::cartesian>(MeshData<Real> *md);
template TaskStatus ThermalFlux<Coordinates::spherical1D>(MeshData<Real> *md);
template TaskStatus ThermalFlux<Coordinates::spherical2D>(MeshData<Real> *md);
template TaskStatus ThermalFlux<Coordinates::spherical3D>(MeshData<Real> *md);
template TaskStatus ThermalFlux<Coordinates::cylindrical>(MeshData<Real> *md);
template TaskStatus ThermalFlux<Coordinates::axisymmetric>(MeshData<Real> *md);

template TaskStatus DiffusionUpdate<Coordinates::cartesian>(MeshData<Real> *md,
                                                            const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::spherical1D>(MeshData<Real> *md,
                                                              const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::spherical2D>(MeshData<Real> *md,
                                                              const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::spherical3D>(MeshData<Real> *md,
                                                              const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::cylindrical>(MeshData<Real> *md,
                                                              const Real dt);
template TaskStatus DiffusionUpdate<Coordinates::axisymmetric>(MeshData<Real> *md,
                                                               const Real dt);

} // namespace Gas
