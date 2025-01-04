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
  Fluid fluid_type = Fluid::radiation;
  params.Add("fluid_type", fluid_type);

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

  return ArtemisUtils::CalculateFluxes<Fluid::radiation>(md, pkg, vprim, vflux, vface,
                                                         pcm);
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

  return ArtemisUtils::FluxSource<Fluid::radiation>(md, pkg, vprim, vcons, vface, dt);
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
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &artemis_pkg = pm->packages.Get("artemis");
  auto &radiation_pkg = pm->packages.Get("radiation");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  if (not do_gas) return TaskStatus::complete;

  auto &gas_pkg = pm->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto opac_d = gas_pkg->template Param<Opacity>("opacity_d");
  auto scat_d = gas_pkg->template Param<Scattering>("scattering_d");

  auto &rad_pkg = pm->packages.Get("radiation");
  auto &params = radiation_pkg->AllParams();
  const auto chat = params.template Get<Real>("chat");
  const auto c = params.template Get<Real>("c");
  const auto arad = params.template Get<Real>("arad");
  const auto outer_max = params.template Get<int>("outer_iteration_max");
  const auto inner_max = params.template Get<int>("inner_iteration_max");
  const auto outer_tol = params.template Get<Real>("outer_iteration_tol");
  const auto inner_tol = params.template Get<Real>("inner_iteration_tol");

  // Packing and indexing
  static auto desc =
      parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux,
                                    gas::cons::momentum, gas::cons::internal_energy,
                                    gas::cons::total_energy>(resolved_pkgs.get());

  static auto desc_prim =
      parthenon::MakePackDescriptor<gas::prim::velocity, gas::prim::density,
                                    gas::prim::sie>(resolved_pkgs.get());
  static auto desc_guess =
      parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux,
                                    gas::cons::momentum, gas::cons::internal_energy>(
          resolved_pkgs.get());

  const auto v0 = desc.GetPack(u0);
  const auto vprim = desc_prim.GetPack(u0);
  const auto vg = desc_guess.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  // Prepare scratch pad memory
  const int ncells1 = ib.e - ib.s + 1 + 2 * parthenon::Globals::nghost;
  const int ngas = vprim.GetMaxNumberOfVars() / 5;
  int scr_size = ScratchPad2D<Real>::shmem_size(ngas, ncells1) * 12;
  const int scr_level = rad_pkg->template Param<int>("scr_level");
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "MatterCoupling", DevExecSpace(), scr_size, scr_level,
      0, u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j) {
        ScratchPad2D<Real> cv(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> B(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> vx1(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> vx2(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> vx3(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> chip(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> chir(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> e0(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> eg(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> v0x1(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> v0x2(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> v0x3(mbr.team_scratch(scr_level), ngas, ncells1);

        // Save initial data
        for (int n = 0; n < ngas; ++n) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, mbr, ib.s, ib.e, [&](const int i) {
                const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                const Real &sie = vprim(b, gas::prim::sie(n), k, j, i);
                cv(n, i) = dens * eos_d.SpecificHeatFromDensityInternalEnergy(dens, sie);
                const Real T = eos_d.TemperatureFromDensityInternalEnergy(dens, sie);
                e0(n, i) = v0(b, gas::cons::internal_energy(n), k, j, i);
                eg(n, i) = e0(n, i);
                vx1(n, i) = vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i);
                vx2(n, i) = vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i);
                vx3(n, i) = vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i);
                B(n, i) = arad * SQR(SQR(T));
                v0x1(n, i) = vx1(n, i);
                v0x2(n, i) = vx2(n, i);
                v0x3(n, i) = vx3(n, i);
              });
        }
        mbr.team_barrier();

        // Do the solve
        // Par reduce to get max iterations?
        parthenon::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, mbr, ib.s, ib.e, [&](const int i) {
              // Outer iteratrion
              Real outer_err = 0.0;
              bool outer_conv = false;
              int outer_iter = 0;
              geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
              const auto &hx = coords.GetScaleFactors();
              std::array<Real, 3> F0{v0(b, rad::cons::flux(0), k, j, i) / hx[0],
                                     v0(b, rad::cons::flux(1), k, j, i) / hx[1],
                                     v0(b, rad::cons::flux(2), k, j, i) / hx[2]};
              Real Er0 = v0(b, rad::cons::energy(), k, j, i);
              std::array<Real, 3> f0{F0[0] / (c * Er0 + Fuzz<Real>()),
                                     F0[1] / (c * Er0 + Fuzz<Real>()),
                                     F0[2] / (c * Er0 + Fuzz<Real>())};
              const auto fedd = EddingtonTensor<Closure::M1>(f0);

              auto F = F0;
              auto Er = Er0;
              while (not outer_conv) {
                outer_err = 0.0;

                Real inner_err = 0.0;
                bool inner_conv = false;
                int inner_iter = 0;
                Real Ek = Er;

                while (not inner_conv) {
                  inner_err = 0.;

                  // Solve for new Ek
                  Real Fr = Ek - Er0;
                  Real ca = 0.0;
                  Real cb = 0.0;
                  for (int n = 0; n < ngas; n++) {
                    const Real B_ = B(n, i);
                    const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                    const Real T = std::pow(B(n, i) / arad, 0.25);
                    const Real e =
                        dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
                    const Real cv_ = cv(n, i);
                    // Evaluate opacities once per iteration
                    chip(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                    chir(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                    Real sigp = chip(n, i) * dt;
                    Real sigr = chir(n, i) * dt;
                    const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
                    const auto f = FleckFactor(arad, T, cv(n, i));

                    const auto &[a, b, d] =
                        EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);

                    const auto &[Ri, Fi] =
                        EnergyRHS(cv_, a, b, d, e, e0(n, i), Ek, B(n, i), c / chat);
                    Fr -= Ri;
                    const Real scale = 1.0 / (1.0 + c / chat * f * a);
                    ca -= (b - a) * scale;
                    const Real cb_ = -Fi * f * (b - a) * scale;
                    cb -= Fi * f * (b - a) * scale;
                  }

                  // (1 + ca)*dEk = -Fr + cb

                  const Real dEk = (-Fr + cb) / (1.0 + ca);
                  Real dE = 0.0;

                  for (int n = 0; n < ngas; n++) {
                    const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                    const Real T = std::pow(B(n, i) / arad, 0.25);
                    const Real e =
                        dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
                    const Real cv_ = cv(n, i);

                    Real sigp = chip(n, i) * dt;
                    Real sigr = chir(n, i) * dt;
                    const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
                    const auto f = FleckFactor(arad, T, cv(n, i));

                    const auto &[a, b, d] =
                        EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);
                    const auto &[Ri, Fi] =
                        EnergyRHS(cv_, a, b, d, e, e0(n, i), Ek, B(n, i), c / chat);

                    const Real ca = c / chat * f * a;
                    const Real cb = -c / chat * f * (b - a);
                    const Real scale = 1.0 / (1.0 + ca);
                    const Real dB = -f * Fi * scale + cb * dEk * scale;
                    inner_err = std::max(inner_err, std::abs(dB / B(n, i)));
                    B(n, i) += dB;
                    // const Real Tnew = std::pow(B(n, i) / arad, 0.25);
                    // const Real enew =
                    //     dens * eos_d.InternalEnergyFromDensityTemperature(dens, Tnew);
                    // // Should I reevaluate B -> T -> e
                    // // Use Ri?
                    // const Real deg = enew - e;
                    // dE -= deg;
                  }
                  // dE *= c / chat;
                  inner_err = std::max(inner_err, std::abs(dEk / (Ek + Fuzz<Real>())));
                  Ek += dEk;
                  // printf("dE %lg\n", std::abs(c / chat * dE / (Er + Fuzz<Real>())));
                  // Cap dE
                  // dE = ((Ek + c / chat * dE) < 0.0) ? -0.1 * chat / c * Ek : dE;
                  // inner_err =
                  //     std::max(inner_err, c / chat * std::abs(dE) / (Er +
                  //     Fuzz<Real>()));
                  // Ek += c / chat * dE;

                  inner_iter++;
                  inner_conv = (inner_err < inner_tol) || (inner_iter > inner_max);
                }
                if (inner_iter > inner_max) {
                  std::stringstream msg;
                  msg << "No inner converge: " << inner_err << " " << Er - Ek;
                  PARTHENON_FAIL(msg);
                }

                // We have updated energies, now update the flux and velocity
                Er = Ek;
                Real dE = 0.0;
                for (int n = 0; n < ngas; n++) {
                  const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                  const Real T = std::pow(B(n, i) / arad, 0.25);
                  const Real e =
                      dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
                  const Real cv_ = cv(n, i);

                  Real sigp = chip(n, i) * dt;
                  Real sigr = chir(n, i) * dt;
                  const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
                  const auto f = FleckFactor(arad, T, cv(n, i));

                  const auto &[a, b, d] =
                      EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);
                  const auto &[dEg, Fi] =
                      EnergyRHS(cv_, a, b, d, e, e0(n, i), Er, B(n, i), c / chat);
                  eg(n, i) = e0(n, i) - c / chat * dEg;

                  dE += dEg;
                }

                Er = Er0 + dE;

                // Update flux fixing v/c , T and Er
                Real alpha = 0.0;
                Real exx = 0.0;
                Real eyy = 0.0;
                Real ezz = 0.0;
                Real exy = 0.0;
                Real exz = 0.0;
                Real eyz = 0.0;
                std::array<Real, 3> delta{0.0, 0.0, 0.0};
                for (int n = 0; n < ngas; n++) {
                  const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                  const Real T =
                      eos_d.TemperatureFromDensityInternalEnergy(dens, eg(n, i));
                  const Real B = arad * SQR(SQR(T));
                  // Evaluate opacities once per iteration
                  chip(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                  chir(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                  Real sigp = chip(n, i) * dt;
                  Real sigr = chir(n, i) * dt;
                  const std::array<Real, 3> beta{vx1(n, i) / c, vx2(n, i) / c,
                                                 vx3(n, i) / c};
                  const auto &[a, b, d] =
                      MomentumExchangeCoeffs(sigp, sigr, beta, B, fedd, Er, c, chat);
                  alpha += a;
                  exx += b * beta[0] * beta[0];
                  eyy += b * beta[1] * beta[1];
                  ezz += b * beta[2] * beta[2];
                  exy += b * beta[0] * beta[1];
                  exz += b * beta[0] * beta[2];
                  eyz += b * beta[1] * beta[2];
                  delta[0] += d[0];
                  delta[1] += d[1];
                  delta[2] += d[2];
                }
                delta[0] = F0[0] + c * chat * delta[0];
                delta[1] = F0[1] + c * chat * delta[1];
                delta[2] = F0[2] + c * chat * delta[2];

                alpha = 1.0 + chat * alpha;
                exx *= chat;
                eyy *= chat;
                ezz *= chat;
                exy *= chat;
                exz *= chat;
                eyz *= chat;

                const Real c11 = alpha * (alpha + eyy + ezz) + (eyy * ezz - eyz);
                const Real c22 = alpha * (alpha + exx + ezz) + (exx * ezz - exz);
                const Real c33 = alpha * (alpha + exx + eyy) + (exx * eyy - exy);
                const Real c12 = -alpha * exy + (exz * eyz - exy * ezz);
                const Real c13 = -alpha * exz + (exy * eyz - exz * eyy);
                const Real c23 = -alpha * eyz + (exy * exz - eyz * exx);

                const Real det =
                    alpha * alpha * (alpha + exx + eyy + ezz) +
                    alpha * ((ezz * exx - exz * exz) + (ezz * eyy - eyz * eyz)) +
                    eyz * (exy * exz - eyz * exx) + exz * (exy * eyz - exz * eyy);

                std::array<Real, 3> Fn{0.0, 0.0, 0.0};

                Fn[0] = c11 * delta[0] + c12 * delta[1] + c13 * delta[2];
                Fn[1] = c12 * delta[0] + c22 * delta[1] + c23 * delta[2];
                Fn[2] = c13 * delta[0] + c23 * delta[1] + c33 * delta[2];
                const Real dFx1_ = Fn[0] - F0[0];
                const Real dFx2_ = Fn[1] - F0[1];
                const Real dFx3_ = Fn[2] - F0[2];

                // Fn = NormalizeFlux(Fn[0] / (c * Er), Fn[1] / (c * Er), Fn[2] / (c *
                // Er));

                const Real dFx1 = Fn[0] * c * Er - F0[0];
                const Real dFx2 = Fn[1] * c * Er - F0[1];
                const Real dFx3 = Fn[2] * c * Er - F0[2];

                for (int d = 0; d < 3; d++) {
                  // Fn[d] *= c * Er;
                  const Real err = std::abs((Fn[d] - F[d]) / (F[d] + Fuzz<Real>()));
                  outer_err = std::max(outer_err, err);
                  F[d] = Fn[d];
                }

                // Update material momentum from momentum conservation

                for (int n = 0; n < ngas; n++) {
                  const Real dfx = F[0] - F0[0];
                  const Real dfy = F[1] - F0[1];
                  const Real dfz = F[2] - F0[2];
                  const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                  const Real dvnx1 = -(F[0] - F0[0]) / (c * chat * dens);
                  const Real dvnx2 = -(F[1] - F0[1]) / (c * chat * dens);
                  const Real dvnx3 = -(F[2] - F0[2]) / (c * chat * dens);
                  Real vx[3] = {vx1(n, i), vx2(n, i), vx3(n, i)};
                  Real vnx1 = v0x1(n, i) + dvnx1;
                  Real vnx2 = v0x2(n, i) + dvnx2;
                  Real vnx3 = v0x3(n, i) + dvnx3;
                  const Real derr = vnx1 - vx1(n, i);
                  const Real fuzz_ = Fuzz<Real>();
                  const Real final_ = derr / (vx1(n, i) + Fuzz<Real>());
                  const Real one_ = 1.0 / (vx1(n, i) + Fuzz<Real>());
                  const Real two_ = std::abs(final_);
                  const Real three_ = std::abs(derr / (vx1(n, i) + Fuzz<Real>()));
                  // if ((i == 2) && (j == 3)) {
                  //   printf("v=[%lg,%lg,%lg]\n", vx1(n, i), vx2(n, i), vx3(n, i));
                  //   printf("v0=[%lg,%lg,%lg]\n", v0x1(n, i), v0x2(n, i), v0x3(n, i));
                  //   printf("derr=%lg\nfuzz=%lg\nfinal=%lg\none=%lg\ntwo=%lg\nthree=%"
                  //          "lg\nouter_err=%lg\n",
                  //          derr, fuzz_, final_, one_, two_, three_, outer_err);
                  // }
                  Real err[3] = {
                      std::abs((vnx1 - vx1(n, i)) / (vx1(n, i) + Fuzz<Real>())),
                      std::abs((vnx2 - vx2(n, i)) / (vx2(n, i) + Fuzz<Real>())),
                      std::abs((vnx3 - vx3(n, i)) / (vx3(n, i) + Fuzz<Real>()))};
                  outer_err = std::max(outer_err, err[0]);
                  outer_err = std::max(outer_err, err[1]);
                  outer_err = std::max(outer_err, err[2]);
                  // if ((i == 2) && (j == 3))
                  //   printf("err=[%lg,%lg,%lg] %lg\n", err[0], err[1], err[2],
                  //   outer_err);
                  vx1(n, i) = vnx1;
                  vx2(n, i) = vnx2;
                  vx3(n, i) = vnx3;
                }

                outer_iter++;
                outer_conv = (outer_err < outer_tol) || (outer_iter > outer_max);
              }
              if (outer_iter > outer_max) {
                std::stringstream msg;
                msg << "No outer converge: " << outer_err;
                PARTHENON_FAIL(msg);
              }

              v0(b, rad::cons::energy(), k, j, i) += (Er - Er0);
              v0(b, rad::cons::flux(0), k, j, i) = F[0] * hx[0];
              v0(b, rad::cons::flux(1), k, j, i) = F[1] * hx[1];
              v0(b, rad::cons::flux(2), k, j, i) = F[2] * hx[2];

              for (int n = 0; n < ngas; n++) {
                const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                const Real v02 = SQR(v0x1(n, i)) + SQR(v0x2(n, i)) + SQR(v0x3(n, i));
                const Real v2 = SQR(vx1(n, i)) + SQR(vx2(n, i)) + SQR(vx3(n, i));
                const Real dEk = 0.5 * dens * (v2 - v02);
                const Real dEg = eg(n, i) - e0(n, i);

                const Real dvx1 = vx1(n, i) - v0x1(n, i);
                const Real dvx2 = vx2(n, i) - v0x2(n, i);
                const Real dvx3 = vx3(n, i) - v0x3(n, i);

                v0(b, gas::cons::momentum(VI(n, 0)), k, j, i) += dens * dvx1 * hx[0];
                v0(b, gas::cons::momentum(VI(n, 1)), k, j, i) += dens * dvx2 * hx[1];
                v0(b, gas::cons::momentum(VI(n, 2)), k, j, i) += dens * dvx3 * hx[2];
                v0(b, gas::cons::internal_energy(n), k, j, i) += dEg;
                v0(b, gas::cons::total_energy(n), k, j, i) += dEk + dEg;
                // if (std::abs(std::max({dvx1, dvx2, dvx3, dEk, dEg})) > 1e-15) {
                //   printf("%d %d: %lg %lg %lg %lg %lg\n", j, i, dvx1, dvx2, dvx3, dEk,
                //          dEg);
                // }
              }
            });
      });

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

template TaskStatus MatterCoupling<Coordinates::cartesian>(MeshData<Real> *u0,
                                                           MeshData<Real> *u1,
                                                           const Real dt);
template TaskStatus MatterCoupling<Coordinates::cylindrical>(MeshData<Real> *u0,
                                                             MeshData<Real> *u1,
                                                             const Real dt);
template TaskStatus MatterCoupling<Coordinates::axisymmetric>(MeshData<Real> *u0,
                                                              MeshData<Real> *u1,
                                                              const Real dt);
template TaskStatus MatterCoupling<Coordinates::spherical1D>(MeshData<Real> *u0,
                                                             MeshData<Real> *u1,
                                                             const Real dt);
template TaskStatus MatterCoupling<Coordinates::spherical2D>(MeshData<Real> *u0,
                                                             MeshData<Real> *u1,
                                                             const Real dt);
template TaskStatus MatterCoupling<Coordinates::spherical3D>(MeshData<Real> *u0,
                                                             MeshData<Real> *u1,
                                                             const Real dt);

} // namespace Radiation
