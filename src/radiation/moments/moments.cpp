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
#include "matter_coupling_simple.hpp"
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

namespace Moments {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Moments::Initialize
//! \brief Adds intialization function for moments package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants) {
  auto moments = std::make_shared<StateDescriptor>("moments");
  Params &params = moments->AllParams();

  // Metadata flags
  auto MetadataMoments = moments->GetMetadataFlag();
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

  params.Add("fatal_if_unconverged",
             pin->GetOrAddBoolean("radiation/moment", "fatal_if_unconverged", true));

  // how to handle the matter coupling:
  // full_coupling = false only does a loop over energy coupling
  // full_coupling = true also does an outer loop over momentum coupling
  params.Add("full_coupling",
             pin->GetOrAddBoolean("radiation/moment", "full_coupling", true));

  // Radiation constants (including chat for Moments)
  // NOTE(@pdmullen): These are also stored in top level radiation package...
  const Real light = constants.GetCCode();
  params.Add("c", light);
  const Real arad = constants.GetARCode();
  params.Add("arad", arad);
  const Real creduc = pin->GetOrAddReal("radiation/moment", "creduc", 1.0);
  params.Add("chat", light / creduc);

  // Floors
  const Real efloor = pin->GetOrAddReal("radiation/moment", "efloor", 1.0e-20);
  params.Add("efloor", efloor);

  const Real tfloor = pin->GetOrAddReal("radiation/moment", "tfloor_cgs", 10.); // K
  params.Add("tfloor", tfloor * units.GetTemperaturePhysicalToCode());

  params.Add("use_opac",
             pin->GetOrAddBoolean("radiation/moment", "init_with_opac", true));

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

  const bool substep = pin->GetOrAddReal("radiation/moment", "substep", true);
  if (!substep) {
    if (coords == Coordinates::cartesian) {
      moments->EstimateTimestepMesh = EstimateTimeStepMesh<Coordinates::cartesian>;
    } else if (coords == Coordinates::spherical1D) {
      moments->EstimateTimestepMesh = EstimateTimeStepMesh<Coordinates::spherical1D>;
    } else if (coords == Coordinates::spherical2D) {
      moments->EstimateTimestepMesh = EstimateTimeStepMesh<Coordinates::spherical2D>;
    } else if (coords == Coordinates::spherical3D) {
      moments->EstimateTimestepMesh = EstimateTimeStepMesh<Coordinates::spherical3D>;
    } else if (coords == Coordinates::cylindrical) {
      moments->EstimateTimestepMesh = EstimateTimeStepMesh<Coordinates::cylindrical>;
    } else if (coords == Coordinates::axisymmetric) {
      moments->EstimateTimestepMesh = EstimateTimeStepMesh<Coordinates::axisymmetric>;
    } else {
      PARTHENON_FAIL("Invalid artemis/coordinate system!");
    }
  }

  // Number of radiation "species" (i.e., groups)
  std::vector<int> fluidids;
  for (int n = 0; n < nspecies; ++n)
    fluidids.push_back(n);

  // Scratch for radiation flux
  const int scr_level = pin->GetOrAddInteger("radiation/moment", "scr_level", 0);
  params.Add("scr_level", scr_level);

  // Logarithmic gridding?
  const bool log =
      pin->GetOrAddString("artemis", "radial_spacing", "uniform") == "logarithmic";

  // Control field for sparse radiation fields
  std::string control_field = rad::cons::energy::name();

  // Conserved Energy Density
  Metadata m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                         Metadata::WithFluxes, Metadata::Sparse, MetadataMoments,
                         MetadataOperatorSplit});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords, log);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  moments->AddSparsePool<rad::cons::energy>(m, control_field, fluidids);

  // Conserved Flux
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Conserved,
                Metadata::Independent, Metadata::WithFluxes, Metadata::Sparse,
                MetadataMoments, MetadataOperatorSplit},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords, log);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  moments->AddSparsePool<rad::cons::flux>(m, control_field, fluidids);

  // Primitive Energy Density
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse, MetadataMoments,
                MetadataOperatorSplit});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords, log);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  moments->AddSparsePool<rad::prim::energy>(m, control_field, fluidids);

  // Primitive Pressure (and associated Riemann pressures)
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::WithFluxes, Metadata::Sparse, MetadataMoments,
                MetadataOperatorSplit});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords, log);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  moments->AddSparsePool<rad::prim::pressure>(m, control_field, fluidids);

  // Primitive Reduced Flux
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse, MetadataMoments,
                MetadataOperatorSplit},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords, log);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  moments->AddSparsePool<rad::prim::flux>(m, control_field, fluidids);

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
          moments->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::cartesian>;
        } else if (ref_pres) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::cartesian>;
        }
        // Spherical
      } else if (coords == G::spherical1D) {
        if (ref_dens) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::spherical1D>;
        } else if (ref_pres) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::spherical1D>;
        }
      } else if (coords == G::spherical2D) {
        if (ref_dens) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::spherical2D>;
        } else if (ref_pres) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::spherical2D>;
        }
      } else if (coords == G::spherical3D) {
        if (ref_dens) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::spherical3D>;
        } else if (ref_pres) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::spherical3D>;
        }
        // Cylindrical
      } else if (coords == G::cylindrical) {
        if (ref_dens) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::cylindrical>;
        } else if (ref_pres) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::cylindrical>;
        }
        // Axisymmetric
      } else if (coords == G::axisymmetric) {
        if (ref_dens) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<pdens, G::axisymmetric>;
        } else if (ref_pres) {
          moments->CheckRefinementBlock = ScalarFirstDerivative<ppres, G::axisymmetric>;
        }
      }
    } else if (ref_mag) {
      using ArtemisUtils::ScalarMagnitude;
      const Real rthr = pin->GetReal("radiation/moment", "refine_thr");
      const Real dthr = pin->GetReal("radiation/moment", "deref_thr");
      params.Add("refine_thr", rthr);
      params.Add("deref_thr", dthr);
      if (ref_dens) {
        moments->CheckRefinementBlock = ScalarMagnitude<rad::prim::energy>;
      } else if (ref_pres) {
        moments->CheckRefinementBlock = ScalarMagnitude<rad::prim::pressure>;
      }
    }
  }

  return moments;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Moments::CalculateFluxes
//! \brief Evaluates advective fluxes for moments evolution
TaskStatus CalculateFluxes(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
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
  static auto desc_g =
      parthenon::MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::dx1, geom::dx2,
                                    geom::dx3, geom::hx1f1, geom::hx2f1, geom::hx3f1,
                                    geom::hx1f2, geom::hx2f2, geom::hx3f2, geom::hx1f3,
                                    geom::hx2f3, geom::hx3f3>(resolved_pkgs.get());
  auto vprim = desc_prim.GetPack(md);
  auto vflux = desc_flux.GetPack(md);
  SparsePack vface;
  auto vg = desc_g.GetPack(md);

  // Call CalculateFluxes with appropriate Fluid and Closure type
  auto closure_type = pkg->Param<Closure>("closure_type");
  if (closure_type == Closure::m1) {
    return ArtemisUtils::CalculateFluxes<Fluid::radiation, Closure::m1>(
        md, pkg, vprim, vflux, vface, vg, false);
  } else if (closure_type == Closure::p1) {
    return ArtemisUtils::CalculateFluxes<Fluid::radiation, Closure::p1>(
        md, pkg, vprim, vflux, vface, vg, false);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Moments::FluxSource
//! \brief Evaluates coordinate terms from advective fluxes for moments evolution
TaskStatus FluxSource(MeshData<Real> *md, const Real dt) {
  PARTHENON_INSTRUMENT
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
  static auto desc_g =
      parthenon::MakePackDescriptor<geom::vol, geom::dh1dx1, geom::dh2dx1, geom::dh3dx1,
                                    geom::dh1dx2, geom::dh2dx2, geom::dh3dx2,
                                    geom::dh1dx3, geom::dh2dx3, geom::dh3dx3>(
          resolved_pkgs.get());
  auto vprim = desc_prim.GetPack(md);
  auto vcons = desc_cons.GetPack(md);
  auto vg = desc_g.GetPack(md);
  SparsePack vface;

  // Call FluxSource with appropriate Fluid and Closure type
  auto closure_type = pkg->Param<Closure>("closure_type");
  if (closure_type == Closure::m1) {
    return ArtemisUtils::FluxSource<Fluid::radiation, Closure::m1>(md, pkg, vprim, vcons,
                                                                   vface, vg, dt);
  } else if (closure_type == Closure::p1) {
    return ArtemisUtils::FluxSource<Fluid::radiation, Closure::p1>(md, pkg, vprim, vcons,
                                                                   vface, vg, dt);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus MatterCoupling
//! \brief
template <Coordinates GEOM>
TaskStatus MatterCoupling(MeshData<Real> *u0, const Real dt) {
  PARTHENON_INSTRUMENT
  auto pm = u0->GetParentPointer();
  auto &artemis_pkg = pm->packages.Get("artemis");

  // Immediately exit if not evolving gas
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  if (!(do_gas)) return TaskStatus::complete;

  // Extract moments package and params
  auto &moments_pkg = pm->packages.Get("moments");
  auto closure_type = moments_pkg->template Param<Closure>("closure_type");
  auto full_coupling = moments_pkg->template Param<bool>("full_coupling");

  // Call MatterCoupling with appropriate GEOM, Fluid, and Closure type given coupling
  if (closure_type == Closure::m1) {
    if (full_coupling) {
      return MatterCouplingFullSingleImpl<GEOM, Closure::m1>(u0, dt);
    } else {
      //      return MatterCouplingSimpleImpl<GEOM, Closure::m1>(u0, dt);
    }
  } else if (closure_type == Closure::p1) {
    if (full_coupling) {
      return MatterCouplingFullSingleImpl<GEOM, Closure::p1>(u0, dt);
    } else {
      //     return MatterCouplingSimpleImpl<GEOM, Closure::p1>(u0, dt);
    }
  }
  return TaskStatus::complete;
}

template <Coordinates GEOM>
void InitMesh(parthenon::Mesh *pmesh) {
  PARTHENON_INSTRUMENT
  auto &moments_pkg = pmesh->packages.Get("moments");
  auto &gas_pkg = pmesh->packages.Get("gas");

  const Real arad = moments_pkg->Param<Real>("arad");
  const bool use_opac = moments_pkg->Param<bool>("use_opac");
  const auto &eos_d = gas_pkg->Param<EOS>("eos_d");

  for (int partition = 0; partition < pmesh->DefaultNumPartitions(); partition++) {
    auto md = pmesh->mesh_data.GetOrAdd("u0c", partition).get();

    // Packing and Indexing
    static auto desc =
        MakePackDescriptor<gas::prim::density, gas::prim::sie, rad::cons::energy,
                           rad::prim::energy, rad::cons::flux, rad::prim::flux>(
            (pmesh->resolved_packages).get());
    auto vmesh = desc.GetPack(md);

    IndexRange ib = md->GetBoundsI(IndexDomain::entire);
    IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
    IndexRange kb = md->GetBoundsK(IndexDomain::entire);
    const auto ndim = pmesh->ndim;
    const auto &cpars =
        pmesh->packages.Get("artemis")->template Param<geometry::CoordParams>(
            "coord_params");

    if (use_opac) {
      const auto &opac_d = gas_pkg->Param<MeanOpacity>("opacity_d");
      const bool multi_d = pmesh->ndim >= 2;
      const bool three_d = pmesh->ndim == 3;
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "Moments::InitMesh", DevExecSpace(), 0,
          md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
            const auto &dx = coords.GetCellWidths();
            const Real &rho = vmesh(b, gas::prim::density(0), k, j, i);
            const Real &sie = vmesh(b, gas::prim::sie(0), k, j, i);
            const Real T = eos_d.TemperatureFromDensityInternalEnergy(rho, sie);
            Real dx_min = dx[0];
            if (multi_d) dx_min = std::min(dx_min, dx[1]);
            if (three_d) dx_min = std::min(dx_min, dx[2]);
            const Real tau =
                std::min(1.0, dx_min * opac_d.RosselandMeanAbsorptionCoefficient(rho, T));
            const Real Erad = tau * arad * SQR(SQR(T));
            vmesh(b, rad::cons::energy(0), k, j, i) = Erad;
            vmesh(b, rad::prim::energy(0), k, j, i) = Erad;
            for (int d = 0; d < 3; d++) {
              vmesh(b, rad::cons::flux(d), k, j, i) = 0.0;
              vmesh(b, rad::prim::flux(d), k, j, i) = 0.0;
            }
          });
    } else {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "Moments::InitMesh", DevExecSpace(), 0,
          md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const Real &rho = vmesh(b, gas::prim::density(0), k, j, i);
            const Real &sie = vmesh(b, gas::prim::sie(0), k, j, i);
            const Real T = eos_d.TemperatureFromDensityInternalEnergy(rho, sie);
            const Real Erad = arad * SQR(SQR(T));
            vmesh(b, rad::cons::energy(0), k, j, i) = Erad;
            vmesh(b, rad::prim::energy(0), k, j, i) = Erad;
            for (int d = 0; d < 3; d++) {
              vmesh(b, rad::cons::flux(d), k, j, i) = 0.0;
              vmesh(b, rad::prim::flux(d), k, j, i) = 0.0;
            }
          });
    }
  }
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

template void InitMesh<G::cartesian>(parthenon::Mesh *pmesh);
template void InitMesh<G::cylindrical>(parthenon::Mesh *pmesh);
template void InitMesh<G::axisymmetric>(parthenon::Mesh *pmesh);
template void InitMesh<G::spherical1D>(parthenon::Mesh *pmesh);
template void InitMesh<G::spherical2D>(parthenon::Mesh *pmesh);
template void InitMesh<G::spherical3D>(parthenon::Mesh *pmesh);

} // namespace Moments
