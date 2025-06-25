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
#include "geometry/geometry.hpp"
#include "matter_coupling.hpp"
#include "moments.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/fluxes/fluid_fluxes.hpp"
#include "utils/history.hpp"
#include "utils/opacity/opacity.hpp"
#include "utils/refinement/amr_criteria.hpp"
#include "utils/units.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::MeanOpacity;
using ArtemisUtils::MeanScattering;
using ArtemisUtils::VI;

namespace Radiation {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Radiation::Initialize
//! \brief Adds intialization function for radiation hydrodynamics package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Constants &constants) {
  auto radiation = std::make_shared<StateDescriptor>("moments");
  Params &params = radiation->AllParams();

  // Metadata flags
  auto MetadataMoments = radiation->GetMetadataFlag();
  auto MetadataOperatorSplit = Metadata::GetUserFlag("OperatorSplit");

  // Closure type
  auto closure = pin->GetOrAddString("radiation/moment", "closure", "m1");
  if (closure == "m1") {
    params.Add("closure_type", Closure::m1);
  } else if (closure == "p1") {
    params.Add("closure_type", Closure::p1);
  } else {
    PARTHENON_FAIL("Invalid radiation closure");
  }

  // Coordinates
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);
  params.Add("coords", coords);

  // Reconstruction algorithm
  ReconstructionMethod recon_method = ReconstructionMethod::null;
  const std::string recon = pin->GetOrAddString("radiation/moment", "reconstruct", "plm");
  recon_method = ArtemisUtils::ChooseReconMethod(recon);
  params.Add("recon", recon_method);

  // Riemann solver
  RSolver riemann_solver = RSolver::null;
  const std::string riemann = pin->GetOrAddString("radiation/moment", "riemann", "hlle");
  if (riemann.compare("hlle") == 0) {
    riemann_solver = RSolver::hlle;
  } else if (riemann.compare("llf") == 0) {
    riemann_solver = RSolver::llf;
  } else {
    PARTHENON_FAIL("Riemann solver (radiation) not recognized.");
  }
  params.Add("rsolver", riemann_solver);

  // Courant, Friedrichs, & Lewy (CFL) Number
  const Real cfl_number = pin->GetOrAddReal("radiation/moment", "cfl", 0.8);
  params.Add("cfl", cfl_number);

  // how to handle the matter coupling:
  // full_coupling = false only does a loop over energy couopling
  // full_coupling = true also does an outer loop over momentum coupling
  params.Add("full_coupling",
             pin->GetOrAddBoolean("radiation/moment", "full_coupling", true));

  // We stuff some constants into params so that can be used in post-processing
  const Real light = constants.GetCCode();
  params.Add("c", light);
  const Real creduc = pin->GetOrAddReal("radiation/moment", "creduc", 1.0);
  params.Add("chat", light / creduc);
  const Real arad = constants.GetARCode();
  params.Add("arad", arad);

  // Floors
  const Real efloor = pin->GetOrAddReal("radiation/moment", "efloor", 1.0e-20);
  params.Add("efloor", efloor);

  // Number of radiation species
  const int nspecies = pin->GetOrAddInteger("radiation/moment", "nspecies", 1);
  params.Add("nspecies", nspecies);
  PARTHENON_REQUIRE(nspecies == 1, "Radiation only works with nspecies=1!");

  // Iteration params
  params.Add("outer_iteration_max",
             pin->GetOrAddInteger("radiation/moment", "outer_iteration_max", 100));
  params.Add("inner_iteration_max",
             pin->GetOrAddInteger("radiation/moment", "inner_iteration_max", 400));
  params.Add("outer_iteration_tol",
             pin->GetOrAddReal("radiation/moment", "outer_iteration_tol", 1e-10));
  params.Add("inner_iteration_tol",
             pin->GetOrAddReal("radiation/moment", "inner_iteration_tol", 1e-10));

  // Number of radiation "species" (i.e., groups)
  std::vector<int> fluidids;
  for (int n = 0; n < nspecies; ++n)
    fluidids.push_back(n);

  // Scratch for radiation flux
  const int scr_level = pin->GetOrAddInteger("radiation/moment", "scr_level", 0);
  params.Add("scr_level", scr_level);

  // Control field for sparse radiation fields
  std::string control_field = rad::cons::energy::name();

  // Conserved Energy Density
  Metadata m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                         Metadata::WithFluxes, Metadata::Sparse, MetadataMoments,
                         MetadataOperatorSplit});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::cons::energy>(m, control_field, fluidids);

  // Conserved Flux
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Conserved,
                Metadata::Independent, Metadata::WithFluxes, Metadata::Sparse,
                MetadataMoments, MetadataOperatorSplit},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::cons::flux>(m, control_field, fluidids);

  // Primitive Energy Density
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse, MetadataMoments,
                MetadataOperatorSplit});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::prim::energy>(m, control_field, fluidids);

  // Primitive Pressure (and associated Riemann pressures)
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::WithFluxes, Metadata::Sparse, MetadataMoments,
                MetadataOperatorSplit});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::prim::pressure>(m, control_field, fluidids);

  // Primitive Reduced Flux
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse, MetadataMoments,
                MetadataOperatorSplit},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::prim::flux>(m, control_field, fluidids);

  // Radiation refinement criterion
  const std::string refine_field =
      pin->GetOrAddString("radiation/moment", "refine_field", "none");
  if (refine_field != "none") {
    // Check which field controls the refinement
    const bool ref_dens = (refine_field == "density");
    const bool ref_pres = (refine_field == "pressure");
    PARTHENON_REQUIRE((ref_dens || ref_pres) && !(ref_dens && ref_pres),
                      "Only density or pressure based criterion currently supported!");

    // Check the type of refinement (e.g., gradient vs magnitude)
    const std::string refine_type = pin->GetString("radiation/moment", "refine_type");
    const bool ref_grad = (refine_type == "gradient");
    const bool ref_mag = (refine_type == "magnitude");
    PARTHENON_REQUIRE((ref_grad || ref_mag) && !(ref_grad && ref_mag),
                      "Only gradient or magnitude based criterion currently supported!");

    // Specify appropriate AMR criterion callback
    if (ref_grad) {
      using ArtemisUtils::ScalarFirstDerivative;
      // Refinement threshold
      const Real thr = pin->GetReal("radiation/moment", "refine_thr");
      params.Add("refine_thr", thr);
      // Geometry specific refinement criteria
      typedef Coordinates G;
      typedef rad::prim::energy pdens;
      typedef rad::prim::pressure ppres;
      // Cartesian
      if (coords == G::cartesian) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::cartesian>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::cartesian>;
        }
        // Spherical
      } else if (coords == G::spherical1D) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::spherical1D>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::spherical1D>;
        }
      } else if (coords == G::spherical2D) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::spherical2D>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::spherical2D>;
        }
      } else if (coords == G::spherical3D) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::spherical3D>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::spherical3D>;
        }
        // Cylindrical
      } else if (coords == G::cylindrical) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::cylindrical>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::cylindrical>;
        }
        // Axisymmetric
      } else if (coords == G::axisymmetric) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::axisymmetric>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::axisymmetric>;
        }
      }
    } else if (ref_mag) {
      using ArtemisUtils::ScalarMagnitude;
      const Real rthr = pin->GetReal("radiation/moment", "refine_thr");
      const Real dthr = pin->GetReal("radiation/moment", "deref_thr");
      params.Add("refine_thr", rthr);
      params.Add("deref_thr", dthr);
      if (ref_dens) {
        radiation->CheckRefinementBlock = ScalarMagnitude<rad::prim::energy>;
      } else if (ref_pres) {
        radiation->CheckRefinementBlock = ScalarMagnitude<rad::prim::pressure>;
      }
    }
  }

  return radiation;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Radiation::CalculateFluxes
//! \brief Evaluates advective fluxes for radiation evolution
TaskStatus CalculateFluxes(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &pkg = pm->packages.Get("moments");

  // Packing
  static auto desc_prim =
      parthenon::MakePackDescriptor<rad::prim::energy, rad::prim::flux,
                                    rad::prim::pressure>(resolved_pkgs.get(), {},
                                                         {parthenon::PDOpt::WithFluxes});
  static auto desc_flux =
      parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto vprim = desc_prim.GetPack(md);
  auto vflux = desc_flux.GetPack(md);
  SparsePack vface;

  // Call CalculateFluxes with appropriate Fluid and Closure type
  auto closure_type = pkg->Param<Closure>("closure_type");
  if (closure_type == Closure::m1) {
    return ArtemisUtils::CalculateFluxes<Fluid::radiation, Closure::m1>(
        md, pkg, vprim, vflux, vface, false);
  } else if (closure_type == Closure::p1) {
    return ArtemisUtils::CalculateFluxes<Fluid::radiation, Closure::p1>(
        md, pkg, vprim, vflux, vface, false);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Radiation::FluxSource
//! \brief Evaluates coordinate terms from advective fluxes for radiation evolution
TaskStatus FluxSource(MeshData<Real> *md, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &pkg = pm->packages.Get("moments");

  // Packing
  static auto desc_prim =
      parthenon::MakePackDescriptor<rad::prim::energy, rad::prim::flux,
                                    rad::prim::pressure>(resolved_pkgs.get(), {},
                                                         {parthenon::PDOpt::WithFluxes});
  static auto desc_cons =
      parthenon::MakePackDescriptor<rad::cons::flux>(resolved_pkgs.get());
  auto vprim = desc_prim.GetPack(md);
  auto vcons = desc_cons.GetPack(md);
  SparsePack vface;

  // Call FluxSource with appropriate Fluid and Closure type
  auto closure_type = pkg->Param<Closure>("closure_type");
  if (closure_type == Closure::m1) {
    return ArtemisUtils::FluxSource<Fluid::radiation, Closure::m1>(md, pkg, vprim, vcons,
                                                                   vface, dt);
  } else if (closure_type == Closure::p1) {
    return ArtemisUtils::FluxSource<Fluid::radiation, Closure::p1>(md, pkg, vprim, vcons,
                                                                   vface, dt);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus MatterCoupling
//! \brief
template <Coordinates GEOM>
TaskStatus MatterCoupling(MeshData<Real> *u0, const Real dt) {
  auto pm = u0->GetParentPointer();
  auto &artemis_pkg = pm->packages.Get("artemis");

  // Immediately exit if not evolving gas
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  if (!(do_gas)) return TaskStatus::complete;

  // Extract moments package and params
  auto &radiation_pkg = pm->packages.Get("moments");
  auto closure_type = radiation_pkg->template Param<Closure>("closure_type");
  auto full_coupling = radiation_pkg->template Param<bool>("full_coupling");

  // Call MatterCoupling with appropriate GEOM, Fluid, and Closure type given coupling
  if (closure_type == Closure::m1) {
    if (full_coupling) {
      return MatterCouplingFullSingleImpl<GEOM, Closure::m1>(u0, dt);
    } else {
      return MatterCouplingSimpleImpl<GEOM, Closure::m1>(u0, dt);
    }
  } else if (closure_type == Closure::p1) {
    if (full_coupling) {
      return MatterCouplingFullSingleImpl<GEOM, Closure::p1>(u0, dt);
    } else {
      return MatterCouplingSimpleImpl<GEOM, Closure::p1>(u0, dt);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef MeshData<Real> MD;
template TaskStatus MatterCoupling<G::cartesian>(MD *u0, const Real dt);
template TaskStatus MatterCoupling<G::cylindrical>(MD *u0, const Real dt);
template TaskStatus MatterCoupling<G::axisymmetric>(MD *u0, const Real dt);
template TaskStatus MatterCoupling<G::spherical1D>(MD *u0, const Real dt);
template TaskStatus MatterCoupling<G::spherical2D>(MD *u0, const Real dt);
template TaskStatus MatterCoupling<G::spherical3D>(MD *u0, const Real dt);

} // namespace Radiation
