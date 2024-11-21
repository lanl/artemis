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
#include <vector>

// Parthenon includes
#include <Kokkos_Core.hpp>
#include <globals.hpp>

// Artemis includes
#include "artemis.hpp"
#include "dust/coagulation/coagulation.hpp"
#include "dust/dust.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/fluxes/fluid_fluxes.hpp"
#include "utils/history.hpp"

using ArtemisUtils::VI;

namespace Dust {

CGSUnit *cgsunit = NULL;

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Dust::Initialize
//! \brief Adds intialization function for dust hydrodynamics package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto dust = std::make_shared<StateDescriptor>("dust");
  Params &params = dust->AllParams();

  // Fluid behavior for this package
  Fluid fluid_type = Fluid::dust;
  params.Add("fluid_type", fluid_type);

  // Coordinates
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys);
  params.Add("coords", coords);

  // Reconstruction algorithm
  ReconstructionMethod recon_method = ReconstructionMethod::null;
  const std::string recon = pin->GetOrAddString("dust", "reconstruct", "plm");
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
  const std::string riemann = pin->GetOrAddString("dust", "riemann", "hlle");
  if (riemann.compare("hlle") == 0) {
    riemann_solver = RSolver::hlle;
  } else if (riemann.compare("llf") == 0) {
    riemann_solver = RSolver::llf;
  } else {
    PARTHENON_FAIL("Riemann solver (dust) not recognized.");
  }
  params.Add("rsolver", riemann_solver);

  // Courant, Friedrichs, & Lewy (CFL) Number
  const Real cfl_number = pin->GetOrAddReal("dust", "cfl", 0.8);
  params.Add("cfl", cfl_number);

  // possible user_dt
  const Real user_dt = pin->GetOrAddReal("dust", "user_dt", 1.0e10);
  params.Add("user_dt", user_dt);

  // Floors
  const Real dfloor = pin->GetOrAddReal("dust", "dfloor", 1.0e-20);
  params.Add("dfloor", dfloor);

  // Number of dust species
  const int nspecies = pin->GetOrAddInteger("dust", "nspecies", 1);
  params.Add("nspecies", nspecies);

  // dust stopping time flag
  bool const_stopping_time = pin->GetOrAddBoolean("dust", "const_stopping_time", true);
  params.Add("const_stopping_time", const_stopping_time);

  bool enable_dust_drag = pin->GetOrAddBoolean("dust", "enable_dust_drag", false);
  params.Add("enable_dust_drag", enable_dust_drag);

  bool hst_out_d2g = pin->GetOrAddBoolean("dust", "hst_out_d2g", false);
  params.Add("hst_out_d2g", hst_out_d2g);

  bool enable_dust_coagulation = false;
  DragMethod drag_method = DragMethod::null;
  if (enable_dust_drag) {
    bool enable_dust_feedback = pin->GetBoolean("dust", "enable_dust_feedback");
    std::string drag_method1 = pin->GetOrAddString("dust", "drag_method", "implicit");

    if (drag_method1 == "implicit") {
      if (enable_dust_feedback) {
        drag_method = DragMethod::implicitFeedback;
      } else {
        drag_method = DragMethod::implicitNoFeedback;
      }
    } else {
      if (enable_dust_feedback) {
        drag_method = DragMethod::explicitFeedback;
      } else {
        drag_method = DragMethod::explicitNoFeedback;
      }
    }

    params.Add("drag_method", drag_method);
  } // end if (enable_dust_drag)

  if (const_stopping_time) {
    if (enable_dust_drag) {
      ParArray1D<Real> stopping_time("dust_stopping_time", nspecies);
      auto stopping_time_host = Kokkos::create_mirror_view(stopping_time);
      for (int n = 0; n < nspecies; ++n) {
        stopping_time_host(n) =
            pin->GetReal("dust", "stopping_time_" + std::to_string(n + 1));
      }
      Kokkos::deep_copy(stopping_time, stopping_time_host);
      params.Add("stopping_time", stopping_time);
    }
  } else {
    ParArray1D<Real> dust_size("dust_size", nspecies);
    auto dust_size_host = Kokkos::create_mirror_view(dust_size);

    // dust particle internal density g/cc
    Real rho_p = pin->GetOrAddReal("dust", "rho_p", 1.25);

    Real rho_p_orig = rho_p;

    if (cgsunit == NULL) {
      cgsunit = new CGSUnit;
    }
    if (!cgsunit->isSet()) {
      cgsunit->SetCGSUnit(pin);
    }
    params.Add("cgs_unit", cgsunit);

    // density in physical unit correspinding to code unit 1.
    // const Real rho_g0 = pin->GetOrAddReal("dust", "rho_g0", 1.0);
    const Real rho_g0 = cgsunit->mass0 / cgsunit->vol0;
    // re-scale rho_p to account for rho_g0 for stopping_time calculation
    rho_p /= rho_g0;

    // re-scale rho_p by prefactor coefficient
    if (cgsunit->isurface_den) {
      rho_p *= 0.5 * M_PI;
    } else {
      rho_p *= std::sqrt(M_PI / 8.);
    }

    params.Add("rho_p", rho_p);

    // input dust size flag
    bool user_input_size = pin->GetOrAddBoolean("dust", "user_input_size", false);

    // coagulation flag
    enable_dust_coagulation = pin->GetOrAddBoolean("physics", "coagulation", false);
    if (enable_dust_coagulation) {
      user_input_size = false;
    }

    if (user_input_size) {
      for (int n = 0; n < nspecies; ++n) {
        dust_size_host(n) = pin->GetReal("dust", "Size_" + std::to_string(n + 1));
      }
    } else {
      const Real s_min = pin->GetReal("dust", "min_dust_size");
      const Real s_max = pin->GetReal("dust", "max_dust_size");

      if (nspecies == 1) {
        dust_size_host(0) = s_min;
      } else if (nspecies > 1) {
        // create dust size array
        const Real mmin = 4.0 * M_PI / 3.0 * rho_p_orig * std::pow(s_min, 3);
        const Real mmax = 4.0 * M_PI / 3.0 * rho_p_orig * std::pow(s_max, 3);

        const Real cond = 1.0 / (1.0 - nspecies) * std::log(mmin / mmax);
        const Real conc = std::log(mmin);
        for (int n = 0; n < nspecies; ++n) {
          Real mgrid = std::exp(conc + cond * n);
          dust_size_host(n) = std::pow(3.0 * mgrid / (4.0 * M_PI * rho_p_orig), 1. / 3.);
          if (parthenon::Globals::my_rank == 0) {
            std::cout << "dust_size(" << n << ")=" << dust_size_host(n) << std::endl;
          }
        }
      }
    }
    Kokkos::deep_copy(dust_size, dust_size_host);
    params.Add("dust_size", dust_size);
  } // end if (const_stopping_time)
  params.Add("enable_coagulation", enable_dust_coagulation);

  std::vector<int> dustids;
  for (int n = 0; n < nspecies; ++n)
    dustids.push_back(n);

  // Scratch for dust flux
  const int scr_level = pin->GetOrAddInteger("dust", "scr_level", 0);
  params.Add("scr_level", scr_level);

  // Control field for sparse dust fields
  std::string control_field = dust::cons::density::name();

  // Conserved Dust Density
  Metadata m = Metadata({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                         Metadata::WithFluxes, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  dust->AddSparsePool<dust::cons::density>(m, control_field, dustids);

  // Conserved Momenta
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Conserved,
                Metadata::Independent, Metadata::WithFluxes, Metadata::Sparse},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  dust->AddSparsePool<dust::cons::momentum>(m, control_field, dustids);

  // Primitive Density
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::Sparse});
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  dust->AddSparsePool<dust::prim::density>(m, control_field, dustids);

  // Primitive Velocities
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::Intensive,
                Metadata::OneCopy, Metadata::FillGhost, Metadata::Sparse},
               std::vector<int>({3}));
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  dust->AddSparsePool<dust::prim::velocity>(m, control_field, dustids);

  // dust stopping time
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::Sparse});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  dust->AddSparsePool<dust::stopping_time>(m, control_field, dustids);

  // Dust hydrodynamics timestep
  if (coords == Coordinates::cartesian) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cartesian>;
  } else if (coords == Coordinates::spherical) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical>;
  } else if (coords == Coordinates::cylindrical) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cylindrical>;
  } else if (coords == Coordinates::axisymmetric) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::axisymmetric>;
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }

  return dust;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DustDrag::UpdateDustStoppingTime
//  \brief Wrapper function for update dust stopping time
template <Coordinates GEOM>
TaskStatus UpdateDustStoppingTime(MeshData<Real> *md) {

  using parthenon::MakePackDescriptor;

  auto pm = md->GetParentPointer();

  auto &dust_pkg = pm->packages.Get("dust");
  if ((!dust_pkg->template Param<bool>("enable_dust_drag")) &&
      (!dust_pkg->template Param<bool>("enable_coagulation"))) {
    return TaskStatus::complete;
  }

  auto &resolved_pkgs = pm->resolved_packages;

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  const int nspecies = dust_pkg->template Param<int>("nspecies");

  const bool const_stopping_time = dust_pkg->template Param<bool>("const_stopping_time");

  if (const_stopping_time) {
    ParArray1D<Real> stoppingTime =
        dust_pkg->template Param<ParArray1D<Real>>("stopping_time");
    static auto desc = MakePackDescriptor<dust::stopping_time>(resolved_pkgs.get());

    auto vmesh = desc.GetPack(md);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Dust::DustStoppingTime", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          // usually the stopping time depends gas density, unless constant is used
          // constant stopping time
          for (int n = 0; n < nspecies; ++n) {
            vmesh(b, dust::stopping_time(n), k, j, i) = stoppingTime(n);
          }
        });
  } else {
    const bool isurf_den = cgsunit->isurface_den;
    ParArray1D<Real> dust_size = dust_pkg->template Param<ParArray1D<Real>>("dust_size");
    static auto desc =
        MakePackDescriptor<gas::prim::density, gas::prim::pressure, dust::stopping_time>(
            resolved_pkgs.get());

    auto vmesh = desc.GetPack(md);

    auto &gas_pkg = pm->packages.Get("gas");

    const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
    const Real rho_p = dust_pkg->template Param<Real>("rho_p");

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Dust::DustStoppingTime2", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real dens_g = vmesh(b, gas::prim::density(0), k, j, i);
          // usually the Stokes number depends gas density, unless constant is used
          // stopping_time = Stokes_number/Omega_k

          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          // if constexpr (GEOM == Coordinates::cylindrical) {
          if (isurf_den) {
            // for surface denstiy, Stokes number = Pi/2*rho_p*s_p/sigma_g
            // calculate the Keplerian Omega at mid-plane
            Real hx[3] = {Null<Real>()};
            coords.GetScaleFactors(hx);
            Real rad = hx[2];
            Real Omega_k = 1.0 / std::sqrt(rad) / rad;
            for (int n = 0; n < nspecies; ++n) {
              const Real St = rho_p * dust_size(n) / dens_g;
              vmesh(b, dust::stopping_time(n), k, j, i) = St / Omega_k;
            }

          } else { // if constexpr ((GEOM == Coordinates::spherical) ||
                   //            (GEOM == Coordinates::axisymmetric)) {

            // for density (g/cc), Stokes number = sqrt(Pi/8)*rho_p*s_p*Omega_k/rho_g/Cs
            const Real pres = vmesh(b, gas::prim::pressure(0), k, j, i);
            const Real cs = std::sqrt(gamma * pres / dens_g);
            for (int n = 0; n < nspecies; ++n) {
              const Real StOme = rho_p * dust_size(n) / dens_g;
              vmesh(b, dust::stopping_time(n), k, j, i) = StOme / cs;
            }
          }

          // for (int n = 0; n < nspecies; ++n) {
          //  vmesh(b, dust::stopping_time(n), k, j, i) =
          //      StoppingTime<GEOM>(dens_g, cs, Omega_k, rho_p, dust_size(n));
          //}
        });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::EstimateTimestepMesh
//! \brief Compute dust hydrodynamics timestep
template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &dust_pkg = pm->packages.Get("dust");
  auto &params = dust_pkg->AllParams();
  const int nspecies = params.template Get<int>("nspecies");

  Real dragDt = params.template Get<Real>("user_dt");

  bool dt_stoppingTime = false;
  if (params.template Get<bool>("enable_dust_drag")) {
    UpdateDustStoppingTime<GEOM>(md);
    if ((params.template Get<DragMethod>("drag_method") ==
         DragMethod::explicitFeedback) ||
        (params.template Get<DragMethod>("drag_method") ==
         DragMethod::explicitNoFeedback)) {
      dt_stoppingTime = true;
    }
  }

  static auto desc =
      MakePackDescriptor<dust::prim::density, dust::prim::velocity, dust::stopping_time>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;

  Real min_dt = std::numeric_limits<Real>::max();
  const auto cfl_number = params.template Get<Real>("cfl");
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Dust::EstimateTimestepMesh", DevExecSpace(),
      0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        Real dx[3] = {Null<Real>()};
        coords.GetCellWidths(dx);

        for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
          Real denom = 0.0;
          for (int d = 0; d < ndim; d++) {
            denom += std::abs(vmesh(b, dust::prim::velocity(VI(n, d)), k, j, i)) / dx[d];
          }
          ldt = std::min(ldt, 1.0 / denom * cfl_number);
        }
        if (dt_stoppingTime) {
          for (int n = 0; n < nspecies; ++n) {
            ldt = std::min(ldt, vmesh(b, dust::stopping_time(n), k, j, i));
          }
        }
      },
      Kokkos::Min<Real>(min_dt));

  return std::min(dragDt, min_dt);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Dust::CalculateFluxes
//! \brief Evaluates advective fluxes for dust evolution
TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &pkg = pm->packages.Get("dust");

  static auto desc_prim =
      parthenon::MakePackDescriptor<dust::prim::density, dust::prim::velocity>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  static auto desc_flux =
      parthenon::MakePackDescriptor<dust::cons::density, dust::cons::momentum>(
          resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto vprim = desc_prim.GetPack(md);
  auto vflux = desc_flux.GetPack(md);
  SparsePack vface;

  return ArtemisUtils::CalculateFluxes<Fluid::dust>(md, pkg, vprim, vflux, vface, pcm);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Dust::FluxSource
//! \brief Evaluates coordinate terms from advective fluxes for dust evolution
TaskStatus FluxSource(MeshData<Real> *md, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &pkg = pm->packages.Get("dust");
  auto sys = pkg->Param<Coordinates>("coords");

  // Because dust is pressureless, we can skip this routine for some coordinate systems
  if (geometry::x1dep(sys) || ((geometry::x2dep(sys)) && (pm->ndim >= 2)) ||
      ((geometry::x3dep(sys)) && (pm->ndim == 3))) {
    static auto desc_prim =
        parthenon::MakePackDescriptor<dust::prim::density, dust::prim::velocity>(
            resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
    static auto desc_cons =
        parthenon::MakePackDescriptor<dust::cons::momentum>(resolved_pkgs.get());
    auto vprim = desc_prim.GetPack(md);
    auto vcons = desc_cons.GetPack(md);
    SparsePack vface;

    return ArtemisUtils::FluxSource(md, pkg, vprim, vcons, vface, dt);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Dust::ApplyDragForceSelect
//  \brief Wrapper function for dust-drag froce based on different drag-method
template <Coordinates GEOM>
TaskStatus ApplyDragForceSelect(MeshData<Real> *md, const Real dt) {

  using parthenon::MakePackDescriptor;

  auto pm = md->GetParentPointer();
  auto &dust_pkg = pm->packages.Get("dust");
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  auto &resolved_pkgs = pm->resolved_packages;
  auto &DUST_DRAG = dust_pkg->template Param<DragMethod>("drag_method");

  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::prim::density, gas::prim::velocity, dust::cons::density,
                         dust::cons::momentum, dust::prim::density, dust::prim::velocity,
                         dust::stopping_time>(resolved_pkgs.get());

  auto vmesh = desc.GetPack(md);

  if ((DUST_DRAG == DragMethod::explicitFeedback) ||
      (DUST_DRAG == DragMethod::explicitNoFeedback)) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Dust::DustDrag", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real dens_g = vmesh(b, gas::prim::density(0), k, j, i);
          const int nspecies = vmesh.GetUpperBound(b, dust::prim::density()) -
                               vmesh.GetLowerBound(b, dust::prim::density()) + 1;
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          Real hx[3] = {Null<Real>()};
          coords.GetScaleFactors(hx);

          for (int n = 0; n < nspecies; ++n) {
            const Real dens_d = vmesh(b, dust::prim::density(n), k, j, i);
            const Real tst = vmesh(b, dust::stopping_time(n), k, j, i);
            const Real cj = dt * dens_d / tst;

            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) +=
                cj * hx[0] *
                (vmesh(b, gas::prim::velocity(0), k, j, i) -
                 vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i));

            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) +=
                cj * hx[1] *
                (vmesh(b, gas::prim::velocity(1), k, j, i) -
                 vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i));

            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) +=
                cj * hx[2] *
                (vmesh(b, gas::prim::velocity(2), k, j, i) -
                 vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i));
          }
          if (DUST_DRAG == DragMethod::explicitFeedback) {
            const Real dens_g = vmesh(b, gas::cons::density(0), k, j, i);
            Real delta_mom1 = 0.0;
            Real delta_mom2 = 0.0;
            Real delta_mom3 = 0.0;
            for (int n = 0; n < nspecies; ++n) {
              const Real dens_d = vmesh(b, dust::prim::density(n), k, j, i);
              const Real tst = vmesh(b, dust::stopping_time(n), k, j, i);
              Real cj = dt * dens_d / tst;
              delta_mom1 -= cj * (vmesh(b, gas::prim::velocity(0), k, j, i) -
                                  vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i));
              delta_mom2 -= cj * (vmesh(b, gas::prim::velocity(1), k, j, i) -
                                  vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i));
              delta_mom3 -= cj * (vmesh(b, gas::prim::velocity(2), k, j, i) -
                                  vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i));
            }
            vmesh(b, gas::cons::momentum(0), k, j, i) += delta_mom1 * hx[0];
            vmesh(b, gas::cons::momentum(1), k, j, i) += delta_mom2 * hx[1];
            vmesh(b, gas::cons::momentum(2), k, j, i) += delta_mom3 * hx[2];

            // gas energy update: delta_E = delta_mom*(M_new - 0.5*delta_mom)/rho_g
            //                    delta_E = 0.5*rhog*(V_new^2 - V_old^2)

            vmesh(b, gas::cons::total_energy(0), k, j, i) +=
                (delta_mom1 * (vmesh(b, gas::cons::momentum(0), k, j, i) / hx[0] -
                               delta_mom1 * 0.5) +
                 delta_mom2 * (vmesh(b, gas::cons::momentum(1), k, j, i) / hx[1] -
                               delta_mom2 * 0.5) +
                 delta_mom3 * (vmesh(b, gas::cons::momentum(2), k, j, i) / hx[2] -
                               delta_mom3 * 0.5)) /
                dens_g;
          }
        });

  } else if ((DUST_DRAG == DragMethod::implicitFeedback) ||
             (DUST_DRAG == DragMethod::implicitNoFeedback)) {
    auto &dfloor = dust_pkg->template Param<Real>("dfloor");
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Dust::DustDragImp", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          Real hx[3] = {Null<Real>()};
          coords.GetScaleFactors(hx);
          const int nspecies = vmesh.GetUpperBound(b, dust::prim::density()) -
                               vmesh.GetLowerBound(b, dust::prim::density()) + 1;

          const Real dens_g = vmesh(b, gas::cons::density(0), k, j, i);
          Real mom1_g = vmesh(b, gas::cons::momentum(0), k, j, i);
          Real mom2_g = vmesh(b, gas::cons::momentum(1), k, j, i);
          Real mom3_g = vmesh(b, gas::cons::momentum(2), k, j, i);

          Real vel1_g = mom1_g / (dens_g * hx[0]);
          Real vel2_g = mom2_g / (dens_g * hx[1]);
          Real vel3_g = mom3_g / (dens_g * hx[2]);
          const Real coef1 = dens_g / vmesh(b, gas::prim::density(0), k, j, i);

          if (DUST_DRAG == DragMethod::implicitFeedback) {

            Real tmp1 = dens_g;
            const Real vel1_go = vel1_g;
            const Real vel2_go = vel2_g;
            const Real vel3_go = vel3_g;
            for (int n = 0; n < nspecies; ++n) {
              const Real dens_d = vmesh(b, dust::cons::density(n), k, j, i);
              const Real tst = vmesh(b, dust::stopping_time(n), k, j, i);
              const Real cj = dt * dens_d / tst * coef1; // use updated dens_g
              const Real bj1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
              const Real bj2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
              const Real bj3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);
              const Real dj = dens_d + cj;

              const Real cjOdj = cj / dj;
              tmp1 += (cj - cj * cjOdj);
              mom1_g += bj1 * cjOdj;
              mom2_g += bj2 * cjOdj;
              mom3_g += bj3 * cjOdj;
            }

            vel1_g = mom1_g / (tmp1 * hx[0]);
            vel2_g = mom2_g / (tmp1 * hx[1]);
            vel3_g = mom3_g / (tmp1 * hx[2]);
            vmesh(b, gas::cons::momentum(0), k, j, i) = dens_g * vel1_g * hx[0];
            vmesh(b, gas::cons::momentum(1), k, j, i) = dens_g * vel2_g * hx[1];
            vmesh(b, gas::cons::momentum(2), k, j, i) = dens_g * vel3_g * hx[2];

            // gas energy update: delta_E = 0.5*(v_new^2 - v_old^2)*rho_g
            vmesh(b, gas::cons::total_energy(0), k, j, i) +=
                0.5 *
                (SQR(vel1_g) - SQR(vel1_go) + SQR(vel2_g) - SQR(vel2_go) + SQR(vel3_g) -
                 SQR(vel3_go)) *
                dens_g;
          } // end if (DUST_DRAG == DragMethod::implicitFeedback)

          // update the dust
          for (int n = 0; n < nspecies; ++n) {
            const Real dens_d =
                std::max(dfloor, vmesh(b, dust::cons::density(n), k, j, i));
            const Real tst = vmesh(b, dust::stopping_time(n), k, j, i);
            const Real cj = dt * dens_d / tst * coef1; // use updated dens_g
            const Real bj1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
            const Real bj2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
            const Real bj3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);
            const Real dj = dens_d + cj;
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) =
                dens_d * (bj1 + cj * vel1_g * hx[0]) / dj;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) =
                dens_d * (bj2 + cj * vel2_g * hx[1]) / dj;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) =
                dens_d * (bj3 + cj * vel3_g * hx[2]) / dj;
          }
        });
  }

  return TaskStatus::complete;
}

// //----------------------------------------------------------------------------------------
// //! \fn  TaskStatus Dust::ApplyDragForceDragSelect
// //  \brief Wrapper function for dust-drag froce
// template <Coordinates GEOM>
// TaskStatus ApplyDragForceDragSelect(MeshData<Real> *md, const Real dt) {

//   using parthenon::MakePackDescriptor;

//   auto pm = md->GetParentPointer();
//   auto &dust_pkg = pm->packages.Get("dust");
//   if (!dust_pkg->template Param<bool>("enable_dust_drag")) {
//     return TaskStatus::complete;
//   }

//   const DragMethod drag_method = dust_pkg->template Param<DragMethod>("drag_method");

//   if (drag_method == DragMethod::explicitFeedback) {
//     return ApplyDragForceSelect<GEOM, DragMethod::explicitFeedback>(md, dt);

//   } else if (drag_method == DragMethod::explicitNoFeedback) {
//     return ApplyDragForceSelect<GEOM, DragMethod::explicitNoFeedback>(md, dt);

//   } else if (drag_method == DragMethod::implicitFeedback) {
//     return ApplyDragForceSelect<GEOM, DragMethod::implicitFeedback>(md, dt);

//   } else if (drag_method == DragMethod::implicitNoFeedback) {
//     return ApplyDragForceSelect<GEOM, DragMethod::implicitNoFeedback>(md, dt);

//   } else {
//     PARTHENON_FAIL("Drag method type not recognized!");
//   }
// }

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Dust::ApplyDragForce
//  \brief Wrapper function for dust-drag froce
TaskStatus ApplyDragForce(MeshData<Real> *md, const Real dt) {

  using parthenon::MakePackDescriptor;

  auto pm = md->GetParentPointer();
  auto &dust_pkg = pm->packages.Get("dust");
  if (!dust_pkg->template Param<bool>("enable_dust_drag")) {
    return TaskStatus::complete;
  }

  const Coordinates GEOM = dust_pkg->template Param<Coordinates>("coords");

  if (GEOM == Coordinates::cartesian) {
    return ApplyDragForceSelect<Coordinates::cartesian>(md, dt);
  } else if (GEOM == Coordinates::spherical) {
    return ApplyDragForceSelect<Coordinates::spherical>(md, dt);
  } else if (GEOM == Coordinates::cylindrical) {
    return ApplyDragForceSelect<Coordinates::cylindrical>(md, dt);
  } else if (GEOM == Coordinates::axisymmetric) {
    return ApplyDragForceSelect<Coordinates::axisymmetric>(md, dt);
  } else {
    PARTHENON_FAIL("Coordinate type not recognized!");
  }
}

//#define COAG_DEBUG
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Dust::CoagulationOneStep
//  \brief Wrapper function for coagulation procedure in one time step
template <Coordinates GEOM>
TaskStatus CoagulationOneStep(MeshData<Real> *md, const Real time, const Real dt) {

  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &dust_pkg = pm->packages.Get("dust");
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  auto &resolved_pkgs = pm->resolved_packages;
  const int nspecies = dust_pkg->template Param<int>("nspecies");

  auto &coag_pkg = pm->packages.Get("coagulation");
  auto &coag = coag_pkg->template Param<Dust::Coagulation::CoagParams>("coag_pars");

  PARTHENON_REQUIRE(cgsunit->isSet(),
                    "coagulation requires setting code-to-physical unit in user routine");

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::pressure, dust::cons::density,
                         dust::cons::momentum, dust::prim::density, dust::prim::velocity,
                         dust::stopping_time>(resolved_pkgs.get());

  auto vmesh = desc.GetPack(md);

  const Real alpha = 1e-3;
  const int nvel = 3;

  const int scr_level = 0; // level = 1 allows more memory access
  int scr_size = ScratchPad1D<Real>::shmem_size((5 + nvel) * nspecies);
  const Real den0 = cgsunit->mass0 / cgsunit->vol0;
  const Real time0 = cgsunit->time0;
  const Real vol0 = cgsunit->vol0;
  const Real vel0 = cgsunit->length0 / cgsunit->time0;

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  ParArray3D<int> nCalls("nCalls", pmb->cellbounds.ncellsk(IndexDomain::entire),
                         pmb->cellbounds.ncellsj(IndexDomain::entire),
                         pmb->cellbounds.ncellsi(IndexDomain::entire));

  int maxCalls = 0, maxSize = 1;
  auto &dfloor = dust_pkg->template Param<Real>("dfloor");

  int maxSize0 = 1;
  Real massd = 0.0;
  Kokkos::parallel_reduce(
      "coag::maxSize0",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {md->NumBlocks(), kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum,
                    int &lmax) {
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        for (int n = 0; n < nspecies; ++n) {
          Real &dens_d = vmesh(b, dust::cons::density(n), k, j, i);
          lsum += dens_d * coords.Volume();
        }
        for (int n = nspecies - 1; n >= 0; --n) {
          Real &dens_d = vmesh(b, dust::cons::density(n), k, j, i);
          if (dens_d > dfloor) {
            lmax = std::max(lmax, n);
            break;
          }
        }
      },
      massd, Kokkos::Max<int>(maxSize0));
#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  MPI_Reduce(MPI_IN_PLACE, &maxSize0, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, &massd, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  Real sumd1 = 0.0;
#ifdef COAG_DEBUG
  ParArray3D<Real> floor_tmp("floor_tmp", 17, nspecies, 2);
#endif

  for (int b = 0; b < md->NumBlocks(); b++) {
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Dust::Coagulation", parthenon::DevExecSpace(),
        scr_size, scr_level, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int k, const int j, const int i) {
          // code-to-physical unit
          // one-cell coagulation
          const int nDust_live = (vmesh.GetUpperBound(b, dust::prim::density()) -
                                  vmesh.GetLowerBound(b, dust::prim::density()) + 1);

          const Real dens_g = vmesh(b, gas::prim::density(0), k, j, i) * den0;
          Real dt_sync = dt * time0;
          const Real time1 = time * time0;

          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          Real hx[3] = {Null<Real>()};
          coords.GetScaleFactors(hx);
          const Real rad = hx[2]; // cylindrical R_cyl
          const Real Omega_k = 1.0 / std::sqrt(rad) / rad;
          const Real vol1 = coords.Volume() * vol0;

          int nCall1 = 0;

          const Real pres = vmesh(b, gas::prim::pressure(0), k, j, i);
          const Real cs = std::sqrt(gamma * pres / dens_g) * vel0;
          const Real omega1 = Omega_k / time0;
          const int nm = nspecies;

          ScratchPad1D<Real> rhod(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> stime(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> vel(mbr.team_scratch(scr_level), nvel * nspecies);
          ScratchPad1D<Real> source(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> Q(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> nQs(mbr.team_scratch(scr_level), nspecies);

          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int n) {
                // for (int n = 0; n < nm; ++n) {
                stime(n) = vmesh(b, dust::stopping_time(n), k, j, i) * time0;
                if (vmesh(b, dust::prim::density(n), k, j, i) > dfloor) {
                  rhod(n) = vmesh(b, dust::prim::density(n), k, j, i) * den0;
                  vel(0 + n * 3) =
                      vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i) * vel0;
                  vel(1 + n * 3) =
                      vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i) * vel0;
                  vel(2 + n * 3) =
                      vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i) * vel0;
                } else {
                  rhod(n) = 0.0;
                  vel(0 + n * 3) = 0.0;
                  vel(1 + n * 3) = 0.0;
                  vel(2 + n * 3) = 0.0;
                }
              });

          Coagulation::CoagulationOneCell(mbr, i, time1, dt_sync, dens_g, rhod, stime,
                                          vel, nvel, Q, nQs, alpha, cs, omega1, vol1,
                                          coag, source, nCall1);

          nCalls(k, j, i) = nCall1;

          // update dust density and velocity after coagulation
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int n) {
                // for (int n = 0; n < nspecies; ++n) {
                if (rhod(n) > 0.0) {
                  const Real rhod1 = rhod(n) / den0;
                  vmesh(b, dust::cons::density(n), k, j, i) = rhod1;
                  vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) =
                      rhod1 * vel(0 + n * 3) * hx[0] / vel0;
                  vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) =
                      rhod1 * vel(1 + n * 3) * hx[1] / vel0;
                  vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) =
                      rhod1 * vel(2 + n * 3) * hx[2] / vel0;
                } else {
                  vmesh(b, dust::cons::density(n), k, j, i) = 0.0;
                  vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) = 0.0;
                  vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) = 0.0;
                  vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) = 0.0;
                }
              });
#ifdef COAG_DEBUG
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int n) {
                // for (int n = 0; n < nspecies; n++) {
                floor_tmp(i - ib.s, n, 0) = source(n);
                floor_tmp(i - ib.s, n, 1) = vmesh(b, dust::cons::density(n), k, j, i);
              });
#endif
        });

    Real sumd0 = 0.0;
    Kokkos::parallel_reduce(
        "coag::nCallsMaximum",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({kb.s, jb.s, ib.s},
                                               {kb.e + 1, jb.e + 1, ib.e + 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lsum, int &lmax1,
                      int &lmax2) {
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          const Real vol00 = coords.Volume();
          for (int n = 0; n < nspecies; ++n) {
            Real &dens_d = vmesh(b, dust::cons::density(n), k, j, i);
            lsum += dens_d * vol00;
          }
          lmax1 = std::max(lmax1, nCalls(k, j, i));
          for (int n = nspecies - 1; n >= 0; --n) {
            Real &dens_d = vmesh(b, dust::cons::density(n), k, j, i);
            if (dens_d > dfloor) {
              lmax2 = std::max(lmax2, n);
              break;
            }
          }
        },
        Kokkos::Sum<Real>(sumd0), Kokkos::Max<int>(maxCalls), Kokkos::Max<int>(maxSize));
    sumd1 += sumd0;
  } // loop of blocks

#ifdef COAG_DEBUG
  auto floor_h =
      Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), floor_tmp);
  for (int i = 0; i <= 11; i++) {
    std::cout << "i=" << i << " ";
    for (int n = 0; n < 11; n++) {
      std::cout << floor_h(i, n, 0) << " ";
    }
    std::cout << std::endl;

    std::cout << "i=" << i << " ";
    for (int n = 0; n < 11; n++) {
      std::cout << floor_h(i, n, 1) << " ";
    }
    std::cout << std::endl;
  }
#endif

#ifdef MPI_PARALLEL
  // over all processors
  MPI_Reduce(MPI_IN_PLACE, &maxCalls, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, &maxSize, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, &sumd1, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "maxcalls,size= " << time << " " << maxCalls << " " << maxSize << " "
              << maxSize0 << " " << time0 * dt << std::endl
              << "total massd =" << time << " " << massd << " " << sumd1 << std::endl;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Dust::OperatorSplitDust
//! \brief dust operator split task collection
template <Coordinates GEOM>
TaskCollection OperatorSplitDust(Mesh *pm, parthenon::SimTime &tm, const Real dt) {
  TaskCollection tc;
  auto &coag_pkg = pm->packages.Get("coagulation");
  auto *dtCoag = coag_pkg->MutableParam<Real>("dtCoag");
  int nstep1Coag = coag_pkg->template Param<int>("nstep1Coag");
  if (tm.ncycle == 0) {
    *dtCoag = 0.0;
  }

  *dtCoag += dt;
  if ((tm.ncycle + 1) % nstep1Coag != 0) {
    return tc;
  }

  const Real time_local = tm.time + dt - (*dtCoag);
  const Real dt_local = (*dtCoag);
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "coagulation at: time,dt,cycle=" << time_local << " " << dt_local << " "
              << tm.ncycle << std::endl;
  }

  TaskID none(0);

  // Assemble tasks
  using namespace ::parthenon::Update;
  const int num_partitions = pm->DefaultNumPartitions();
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &base = pm->mesh_data.GetOrAdd("base", i);
    auto coag_step =
        tl.AddTask(none, CoagulationOneStep<GEOM>, base.get(), time_local, dt_local);

    // Set (remaining) fields to be communicated
    auto pre_comm = tl.AddTask(coag_step, PreCommFillDerived<MeshData<Real>>, base.get());

    // Set boundary conditions (both physical and logical)
    auto bcs = parthenon::AddBoundaryExchangeTasks(pre_comm, tl, base, pm->multilevel);

    // Update primitive variables
    auto c2p = tl.AddTask(bcs, FillDerived<MeshData<Real>>, base.get());
  }
  *dtCoag = 0.0;

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::ReduceD2gMaximum
//  \brief calculate dust-to-gas ratio maximum
std::vector<Real> ReduceD2gMaximum(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  static auto desc =
      MakePackDescriptor<gas::prim::density, dust::prim::density>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const int nblocks = md->NumBlocks();
  std::vector<Real> max_d2g(1, 0);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ReduceD2gMaximum", parthenon::DevExecSpace(),
      0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        const Real &rhog = vmesh(b, gas::prim::density(0), k, j, i);
        const int nspecies = (vmesh.GetUpperBound(b, dust::prim::density()) -
                              vmesh.GetLowerBound(b, dust::prim::density())) +
                             1;
        Real rhod = 0.0;
        for (int n = 0; n < nspecies; ++n) {
          rhod += vmesh(b, dust::prim::density(n), k, j, i);
        }
        lmax = std::max(lmax, rhod / rhog);
      },
      Kokkos::Max<Real>(max_d2g[0]));
  return max_d2g;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::AddHistoryImpl
//! \brief Add history outputs for dust quantities for generic coordinate system
template <Coordinates GEOM>
void AddHistoryImpl(Params &params) {
  using namespace ArtemisUtils;
  auto HstSum = parthenon::UserHistoryOperation::sum;
  using parthenon::HistoryOutputVar;
  parthenon::HstVec_list hst_vecs = {};

  // Mass
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVolumeIntegral<GEOM, dust::cons::density>, "dust_mass"));

  // Momenta
  typedef dust::cons::momentum cmom;
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVectorVolumeIntegral<GEOM, X1DIR, cmom>, "dust_momentum_x1"));
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVectorVolumeIntegral<GEOM, X2DIR, cmom>, "dust_momentum_x2"));
  hst_vecs.emplace_back(HistoryOutputVec(
      HstSum, ReduceSpeciesVectorVolumeIntegral<GEOM, X3DIR, cmom>, "dust_momentum_x3"));
  if (params.template Get<bool>("hst_out_d2g")) {
    auto HstMax = parthenon::UserHistoryOperation::max;
    hst_vecs.emplace_back(HistoryOutputVec(HstMax, ReduceD2gMaximum, "dust-to-gas"));
  }

  params.Add(parthenon::hist_vec_param_key, hst_vecs);
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::AddHistory
//! \brief Add history outputs for dust quantities
void AddHistory(Coordinates coords, Params &params) {
  if (coords == Coordinates::cartesian) {
    AddHistoryImpl<Coordinates::cartesian>(params);
  } else if (coords == Coordinates::cylindrical) {
    AddHistoryImpl<Coordinates::cylindrical>(params);
  } else if (coords == Coordinates::spherical) {
    AddHistoryImpl<Coordinates::spherical>(params);
  } else if (coords == Coordinates::axisymmetric) {
    AddHistoryImpl<Coordinates::axisymmetric>(params);
  }
}

//----------------------------------------------------------------------------------------
//! template instantiations
template Real EstimateTimestepMesh<Coordinates::cartesian>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::cylindrical>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::spherical>(MeshData<Real> *md);
template Real EstimateTimestepMesh<Coordinates::axisymmetric>(MeshData<Real> *md);

template TaskStatus UpdateDustStoppingTime<Coordinates::cartesian>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::cylindrical>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::spherical>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::axisymmetric>(MeshData<Real> *md);

template TaskStatus CoagulationOneStep<Coordinates::cartesian>(MeshData<Real> *md,
                                                               const Real time,
                                                               const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::cylindrical>(MeshData<Real> *md,
                                                                 const Real time,
                                                                 const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::spherical>(MeshData<Real> *md,
                                                               const Real time,
                                                               const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::axisymmetric>(MeshData<Real> *md,
                                                                  const Real time,
                                                                  const Real dt);

template TaskCollection OperatorSplitDust<Coordinates::cartesian>(Mesh *pm,
                                                                  parthenon::SimTime &tm,
                                                                  const Real dt);
template TaskCollection OperatorSplitDust<Coordinates::spherical>(Mesh *pm,
                                                                  parthenon::SimTime &tm,
                                                                  const Real dt);
template TaskCollection
OperatorSplitDust<Coordinates::cylindrical>(Mesh *pm, parthenon::SimTime &tm,
                                            const Real dt);
template TaskCollection
OperatorSplitDust<Coordinates::axisymmetric>(Mesh *pm, parthenon::SimTime &tm,
                                             const Real dt);
} // namespace Dust
