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
#include "geometry/geometry.hpp"
#include "radiation.hpp"
#include "matter_coupling.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/fluxes/fluid_fluxes.hpp"
#include "utils/history.hpp"
#include "utils/opacity/opacity.hpp"
#include "utils/refinement/amr_criteria.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::Opacity;
using ArtemisUtils::Scattering;
using ArtemisUtils::VI;

namespace Radiation {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Radiation::Initialize
//! \brief Adds intialization function for radiation hydrodynamics package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto radiation = std::make_shared<StateDescriptor>("radiation");
  Params &params = radiation->AllParams();

  // Fluid behavior for this package
  auto closure = pin->GetOrAddString("radiation", "closure", "m1");
  if (closure == "m1") {
    params.Add("fluid_type", Fluid::greyM1);
  } else if (closure == "p1") {
    params.Add("fluid_type", Fluid::greyP1);
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
  const std::string recon = pin->GetOrAddString("radiation", "reconstruct", "plm");
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
  const std::string riemann = pin->GetOrAddString("radiation", "riemann", "hlle");
  if (riemann.compare("hlle") == 0) {
    riemann_solver = RSolver::hlle;
  } else if (riemann.compare("llf") == 0) {
    riemann_solver = RSolver::llf;
  } else {
    PARTHENON_FAIL("Riemann solver (radiation) not recognized.");
  }
  params.Add("rsolver", riemann_solver);

  // Courant, Friedrichs, & Lewy (CFL) Number
  const Real cfl_number = pin->GetOrAddReal("radiation", "cfl", 0.8);
  params.Add("cfl", cfl_number);

  params.Add("nstages", pin->GetOrAddInteger("radiation", "nstages", 2));

  auto pc = parthenon::constants::PhysicalConstants<parthenon::constants::CGS>();
  const Real light = pin->GetOrAddReal("radiation", "c", pc.c);
  printf("Light speed %lg\n", light);
  params.Add("c", light);
  const Real creduc = pin->GetOrAddReal("radiation", "creduc", 1.0);
  params.Add("chat", light / creduc);
  const Real arad = pin->GetOrAddReal("radiation", "arad", pc.ar);
  params.Add("arad", arad);

  // Floors
  const Real efloor = pin->GetOrAddReal("radiation", "efloor", 1.0e-20);
  params.Add("efloor", efloor);

  // Number of radiation species
  const int nspecies = pin->GetOrAddInteger("radiation", "nspecies", 1);
  params.Add("nspecies", nspecies);
  PARTHENON_REQUIRE(nspecies == 1, "Radiation only works with nspecies=1!");

  // Iteration params
  params.Add("outer_iteration_max",
             pin->GetOrAddInteger("radiation", "outer_iteration_max", 100));
  params.Add("inner_iteration_max",
             pin->GetOrAddInteger("radiation", "inner_iteration_max", 100));
  params.Add("outer_iteration_tol",
             pin->GetOrAddReal("radiation", "outer_iteration_tol", 1e-8));
  params.Add("inner_iteration_tol",
             pin->GetOrAddReal("radiation", "inner_iteration_tol", 1e-8));

  std::vector<int> fluidids;
  for (int n = 0; n < nspecies; ++n)
    fluidids.push_back(n);

  // Scratch for radiation flux
  const int scr_level = pin->GetOrAddInteger("radiation", "scr_level", 0);
  params.Add("scr_level", scr_level);

  // Control field for sparse radiation fields
  std::string control_field = rad::cons::energy::name();

  // Conserved Radiation Density
  Metadata m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                         Metadata::WithFluxes, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::cons::energy>(m, control_field, fluidids);

  // Conserved Momenta
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Conserved,
                Metadata::Independent, Metadata::WithFluxes, Metadata::Sparse},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::cons::flux>(m, control_field, fluidids);

  // Primitive Density
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::prim::energy>(m, control_field, fluidids);

  // Primitive Pressure (and associated Riemann pressures)
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::WithFluxes, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::prim::pressure>(m, control_field, fluidids);

  // Primitive Velocities
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::prim::flux>(m, control_field, fluidids);

  // Eddington Tensor
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse},
               std::vector<int>({6}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  radiation->AddSparsePool<rad::prim::flux>(m, control_field, fluidids);

  // Radiation refinement criterion
  const std::string refine_field =
      pin->GetOrAddString("radiation", "refine_field", "none");
  if (refine_field != "none") {
    // Check which field controls the refinement
    const bool ref_dens = (refine_field == "density");
    const bool ref_pres = (refine_field == "pressure");
    PARTHENON_REQUIRE((ref_dens || ref_pres) && !(ref_dens && ref_pres),
                      "Only density or pressure based criterion currently supported!");

    // Check the type of refinement (e.g., gradient vs magnitude)
    const std::string refine_type = pin->GetString("radiation", "refine_type");
    const bool ref_grad = (refine_type == "gradient");
    const bool ref_mag = (refine_type == "magnitude");
    PARTHENON_REQUIRE((ref_grad || ref_mag) && !(ref_grad && ref_mag),
                      "Only gradient or magnitude based criterion currently supported!");

    // Specify appropriate AMR criterion callback
    if (ref_grad) {
      using ArtemisUtils::ScalarFirstDerivative;
      // Refinement threshold
      const Real thr = pin->GetReal("radiation", "refine_thr");
      params.Add("refine_thr", thr);
      // Geometry specific refinement criteria
      typedef Coordinates C;
      typedef rad::prim::energy pdens;
      typedef rad::prim::pressure ppres;
      // Cartesian
      if (coords == C::cartesian) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::cartesian>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::cartesian>;
        }
        // Spherical
      } else if (coords == C::spherical1D) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::spherical1D>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::spherical1D>;
        }
      } else if (coords == C::spherical2D) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::spherical2D>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::spherical2D>;
        }
      } else if (coords == C::spherical3D) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::spherical3D>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::spherical3D>;
        }
        // Cylindrical
      } else if (coords == C::cylindrical) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::cylindrical>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::cylindrical>;
        }
        // Axisymmetric
      } else if (coords == C::axisymmetric) {
        if (ref_dens) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<pdens, C::axisymmetric>;
        } else if (ref_pres) {
          radiation->CheckRefinementBlock = ScalarFirstDerivative<ppres, C::axisymmetric>;
        }
      }
    } else if (ref_mag) {
      using ArtemisUtils::ScalarMagnitude;
      const Real rthr = pin->GetReal("radiation", "refine_thr");
      const Real dthr = pin->GetReal("radiation", "deref_thr");
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
//! \fn  Real Radiation::EstimateTimestepMesh
//! \brief Compute radiation timestep
template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &radiation_pkg = pm->packages.Get("radiation");
  auto &params = radiation_pkg->AllParams();
  const auto chat = params.template Get<Real>("chat");

  static auto desc = MakePackDescriptor<rad::prim::energy>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;

  Real min_dt = Big<Real>();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Radiation::EstimateTimestepMesh",
      DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
        Real denom = 0.0;
        for (int d = 0; d < ndim; d++) {
          denom += chat / dx[d];
        }
        ldt = std::min(ldt, 1.0 / denom);
      },
      Kokkos::Min<Real>(min_dt));

  const auto cfl_number = params.template Get<Real>("cfl");
  return cfl_number * min_dt;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Radiation::CalculateFluxes
//! \brief Evaluates advective fluxes for radiation evolution
TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &pkg = pm->packages.Get("radiation");

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

  auto fluid_type = pkg->Param<Fluid>("fluid_type");
  if (fluid_type == Fluid::greyM1) {
    return ArtemisUtils::CalculateFluxes<Fluid::greyM1>(md, pkg, vprim, vflux, vface,
                                                         pcm);
  } else if (fluid_type == Fluid::greyP1) {
    return ArtemisUtils::CalculateFluxes<Fluid::greyP1>(md, pkg, vprim, vflux, vface,
                                                         pcm);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Radiation::FluxSource
//! \brief Evaluates coordinate terms from advective fluxes for radiation evolution
TaskStatus FluxSource(MeshData<Real> *md, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &pkg = pm->packages.Get("radiation");

  static auto desc_prim =
      parthenon::MakePackDescriptor<rad::prim::energy, rad::prim::flux,
                                    rad::prim::pressure>(resolved_pkgs.get(), {},
                                                         {parthenon::PDOpt::WithFluxes});
  static auto desc_cons =
      parthenon::MakePackDescriptor<rad::cons::flux>(resolved_pkgs.get());
  auto vprim = desc_prim.GetPack(md);
  auto vcons = desc_cons.GetPack(md);
  SparsePack vface;

  auto fluid_type = pkg->Param<Fluid>("fluid_type");
  if (fluid_type == Fluid::greyM1) {
    return ArtemisUtils::FluxSource<Fluid::greyM1>(md, pkg, vprim, vcons, vface, dt);
  } else if (fluid_type == Fluid::greyP1) {
    return ArtemisUtils::FluxSource<Fluid::greyP1>(md, pkg, vprim, vcons, vface, dt);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Radiation::ApplyUpdate
//! \brief
template <Coordinates GEOM>
TaskStatus ApplyUpdate(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       const Real gam0, const Real gam1, const Real beta_dt) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  // Packing and indexing
  static auto desc = parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux>(
      resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});

  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);

        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = (multi_d) ? coords.GetFaceAreaX2() : NewArray<Real, 2>(0.0);
        const auto ax3 = (three_d) ? coords.GetFaceAreaX3() : NewArray<Real, 2>(0.0);

        const Real vol = coords.Volume();

        for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
          // compute flux divergence
          Real divf = (ax1[0] * v0.flux(b, X1DIR, n, k, j, i) -
                       ax1[1] * v0.flux(b, X1DIR, n, k, j, i + 1));
          if (multi_d)
            divf += (ax2[0] * v0.flux(b, X2DIR, n, k, j, i) -
                     ax2[1] * v0.flux(b, X2DIR, n, k, j + 1, i));
          if (three_d)
            divf += (ax3[0] * v0.flux(b, X3DIR, n, k, j, i) -
                     ax3[1] * v0.flux(b, X3DIR, n, k + 1, j, i));

          // Apply update
          v0(b, n, k, j, i) =
              gam0 * v0(b, n, k, j, i) + gam1 * v1(b, n, k, j, i) + divf * beta_dt / vol;
        }
      });
  return TaskStatus::complete;
}

template <Coordinates GEOM>
TaskStatus MatterCoupling(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt) {
  auto pm = u0->GetParentPointer();
  auto &artemis_pkg = pm->packages.Get("artemis");

  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  if (not do_gas) return TaskStatus::complete;

  auto &radiation_pkg = pm->packages.Get("radiation");
  auto fluid_type = radiation_pkg->template Param<Fluid>("fluid_type");
  if (fluid_type == Fluid::greyM1) {
    return MatterCouplingImpl<GEOM,Fluid::greyM1>(u0,u1,dt);
  } else if (fluid_type == Fluid::greyP1) {
    return MatterCouplingImpl<GEOM,Fluid::greyP1>(u0,u1,dt);
  }
  return TaskStatus::complete;
}

template TaskStatus ApplyUpdate<Coordinates::cartesian>(MeshData<Real> *u0,
                                                        MeshData<Real> *u1,
                                                        const int stage, const Real gam0,
                                                        const Real gam1,
                                                        const Real beta_dt);
template TaskStatus
ApplyUpdate<Coordinates::cylindrical>(MeshData<Real> *u0, MeshData<Real> *u1,
                                      const int stage, const Real gam0, const Real gam1,
                                      const Real beta_dt);
template TaskStatus
ApplyUpdate<Coordinates::axisymmetric>(MeshData<Real> *u0, MeshData<Real> *u1,
                                       const int stage, const Real gam0, const Real gam1,
                                       const Real beta_dt);
template TaskStatus
ApplyUpdate<Coordinates::spherical1D>(MeshData<Real> *u0, MeshData<Real> *u1,
                                      const int stage, const Real gam0, const Real gam1,
                                      const Real beta_dt);
template TaskStatus
ApplyUpdate<Coordinates::spherical2D>(MeshData<Real> *u0, MeshData<Real> *u1,
                                      const int stage, const Real gam0, const Real gam1,
                                      const Real beta_dt);
template TaskStatus
ApplyUpdate<Coordinates::spherical3D>(MeshData<Real> *u0, MeshData<Real> *u1,
                                      const int stage, const Real gam0, const Real gam1,
                                      const Real beta_dt);

template TaskStatus MatterCoupling<Coordinates::cartesian>(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt);
template TaskStatus MatterCoupling<Coordinates::cylindrical>(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt);
template TaskStatus MatterCoupling<Coordinates::axisymmetric>(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt);
template TaskStatus MatterCoupling<Coordinates::spherical1D>(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt);
template TaskStatus MatterCoupling<Coordinates::spherical2D>(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt);
template TaskStatus MatterCoupling<Coordinates::spherical3D>(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt);

} // namespace Radiation
