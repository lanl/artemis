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
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);
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
  std::vector<int> dustids;
  for (int n = 0; n < nspecies; ++n)
    dustids.push_back(n);

  bool hst_out_d2g = pin->GetOrAddBoolean("dust", "hst_out_d2g", false);
  params.Add("hst_out_d2g", hst_out_d2g);

  // coagulation flag
  bool enable_dust_coagulation = pin->GetOrAddBoolean("physics", "coagulation", false);

  // Dust sizes
  auto size_dist = pin->GetOrAddString("dust", "size_input", "direct");
  if (enable_dust_coagulation) size_dist = "logspace";
  params.Add("enable_coagulation", enable_dust_coagulation);

  if (size_dist == "linspace") {
    // uniform
    auto min_size = pin->GetReal("dust", "min_size");
    auto max_size = pin->GetReal("dust", "max_size");

    ParArray1D<Real> sizes("sizes", nspecies);
    auto h_sizes = sizes.GetHostMirror();
    const Real ds =
        (nspecies == 1) ? 0.0 : (max_size - min_size) / (static_cast<Real>(nspecies - 1));
    for (int n = 0; n < nspecies; n++) {
      h_sizes(n) = min_size + n * ds;
    }
    sizes.DeepCopy(h_sizes);
    params.Add("sizes", sizes);
    params.Add("h_sizes", h_sizes);

  } else if (size_dist == "logspace") {
    // uniform in log-space
    const auto lmin = std::log10(pin->GetReal("dust", "min_size"));
    const auto lmax = std::log10(pin->GetReal("dust", "max_size"));

    ParArray1D<Real> sizes("sizes", nspecies);
    auto h_sizes = sizes.GetHostMirror();
    const Real ds =
        (nspecies == 1) ? 0.0 : (lmax - lmin) / (static_cast<Real>(nspecies - 1));
    for (int n = 0; n < nspecies; n++) {
      h_sizes(n) = std::pow(10.0, lmin + n * ds);
    }
    sizes.DeepCopy(h_sizes);
    params.Add("sizes", sizes);
    params.Add("h_sizes", h_sizes);

    if (parthenon::Globals::my_rank == 0) {
      for (int n = 0; n < nspecies; n++) {
        std::cout << "dust_size(" << n << ")=" << h_sizes(n) << std::endl;
      }
    }
  } else if (size_dist == "direct") {
    // specify them directly
    auto sizes_v = pin->GetVector<Real>("dust", "sizes");

    ParArray1D<Real> sizes("sizes", nspecies);
    auto h_sizes = sizes.GetHostMirror();
    for (int n = 0; n < nspecies; n++) {
      h_sizes(n) = sizes_v[n];
    }
    sizes.DeepCopy(h_sizes);
    params.Add("sizes", sizes);
    params.Add("h_sizes", h_sizes);

  } else if (size_dist == "file") {
    // Read from a file
    auto input_file = pin->GetString("dust", "size_file");
    std::vector<std::vector<Real>> data = ArtemisUtils::loadtxt(input_file);

    ParArray1D<Real> sizes("sizes", nspecies);
    auto h_sizes = sizes.GetHostMirror();
    if (data.size() == 1) {
      for (int n = 0; n < nspecies; n++) {
        h_sizes(n) = data[0][n];
      }
    } else {
      for (int n = 0; n < nspecies; n++) {
        h_sizes(n) = data[n][0];
      }
    }
    sizes.DeepCopy(h_sizes);
    params.Add("sizes", sizes);
    params.Add("h_sizes", h_sizes);
  } else {
    PARTHENON_FAIL("dust/size_input not recognized!");
  }

  // Dust density
  const Real rho_p_orig = pin->GetOrAddReal("dust", "grain_density", 1.0);
  params.Add("grain_density", rho_p_orig);

  if (enable_dust_coagulation) {
    if (cgsunit == NULL) {
      cgsunit = new CGSUnit;
    }
    if (!cgsunit->isSet()) {
      cgsunit->SetCGSUnit(pin);
    }
    params.Add("cgs_unit", cgsunit);

    Real rho_p = rho_p_orig;
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
      rho_p /= cgsunit->length0;
    }

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "rho_p, rho_g0=" << rho_p << " " << rho_g0 << std::endl;
    }
    params.Add("rho_p", rho_p);
  }
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

  // dust stopping-time
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::Sparse});
  m.SetSparseThresholds(0.0, 0.0, 0.0);
  dust->AddSparsePool<dust::stopping_time>(m, control_field, dustids);
  // std::vector<int> stopping_id = {0};
  // dust->AddSparsePool<dust::stopping_time>(m, control_field, stopping_id);

  // Dust hydrodynamics timestep
  if (coords == Coordinates::cartesian) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cartesian>;
  } else if (coords == Coordinates::spherical1D) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical1D>;
  } else if (coords == Coordinates::spherical2D) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical2D>;
  } else if (coords == Coordinates::spherical3D) {
    dust->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical3D>;
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
//! \fn  TaskStatus Dust::UpdateDustStoppingTime
//  \brief Wrapper function for update dust stopping time
template <Coordinates GEOM>
TaskStatus UpdateDustStoppingTime(MeshData<Real> *md) {

  using parthenon::MakePackDescriptor;

  auto pm = md->GetParentPointer();

  auto &dust_pkg = pm->packages.Get("dust");
  if ((!dust_pkg->template Param<bool>("enable_coagulation"))) {
    return TaskStatus::complete;
  }

  auto &resolved_pkgs = pm->resolved_packages;

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  const int nspecies = dust_pkg->template Param<int>("nspecies");

  const bool isurf_den = cgsunit->isurface_den;
  ParArray1D<Real> dust_size = dust_pkg->template Param<ParArray1D<Real>>("sizes");
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

        if (isurf_den) {
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          // for surface denstiy, Stokes number = Pi/2*rho_p*s_p/sigma_g
          // calculate the Keplerian Omega at mid-plane
          Real rad;
          if constexpr (GEOM == Coordinates::cylindrical) {
            rad = coords.hx2v(); // cylindrical-rad
          } else {
            rad = coords.hx3v();
          }
          Real Omega_k = 1.0 / std::sqrt(rad) / rad;
          for (int n = 0; n < nspecies; ++n) {
            const Real St = rho_p * dust_size(n) / dens_g;
            vmesh(b, dust::stopping_time(n), k, j, i) = St / Omega_k;
          }

        } else {

          // for density (g/cc), Stokes number = sqrt(Pi/8)*rho_p*s_p*Omega_k/rho_g/Cs
          const Real pres = vmesh(b, gas::prim::pressure(0), k, j, i);
          const Real cs = std::sqrt(gamma * pres / dens_g);
          for (int n = 0; n < nspecies; ++n) {
            const Real StOme = rho_p * dust_size(n) / dens_g;
            vmesh(b, dust::stopping_time(n), k, j, i) = StOme / cs;
          }
        }
      });

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

  auto userDt = params.template Get<Real>("user_dt");
  auto nspecies = params.template Get<int>("nspecies");

  const auto cfl_number = params.template Get<Real>("cfl");
  static auto desc =
      MakePackDescriptor<dust::prim::density, dust::prim::velocity, dust::stopping_time>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;

  Real min_dt = std::numeric_limits<Real>::max();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Dust::EstimateTimestepMesh", DevExecSpace(),
      0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

        for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
          Real denom = 0.0;
          for (int d = 0; d < ndim; d++) {
            denom += std::abs(vmesh(b, dust::prim::velocity(VI(n, d)), k, j, i)) / dx[d];
          }
          ldt = std::min(ldt, 1.0 / denom * cfl_number);
        }
      },
      Kokkos::Min<Real>(min_dt));

  return std::min(userDt, min_dt);
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

  const Real alpha = coag_pkg->template Param<Real>("coag_alpha"); // 1e-3
  const int nvel = 3;

  const int scr_level = coag_pkg->template Param<int>("coag_scr_level");
  size_t isize = (5 + nvel) * nspecies;
  if (coag.integrator == 3 && coag.mom_coag) isize += nspecies;
  size_t scr_size = ScratchPad1D<Real>::shmem_size(isize);
  const Real den0 = cgsunit->mass0 / cgsunit->vol0;
  const Real time0 = cgsunit->time0;
  // const Real vol0 = cgsunit->vol0;
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
          const auto &hx = coords.GetScaleFactors();
          Real rad = hx[2];
          if constexpr (GEOM == Coordinates::cylindrical) {
            rad = hx[1];
          }
          const Real Omega_k = 1.0 / std::sqrt(rad) / rad; // code unit
          // const Real vol1 = coords.Volume() * vol0;

          int nCall1 = 0;

          const Real pres = vmesh(b, gas::prim::pressure(0), k, j, i);
          const Real cs = std::sqrt(gamma * pres / dens_g * den0) * vel0;
          const Real omega1 = Omega_k / time0;
          const int nm = nspecies;

          ScratchPad1D<Real> rhod(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> stime(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> vel(mbr.team_scratch(scr_level), nvel * nspecies);
          ScratchPad1D<Real> source(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> Q(mbr.team_scratch(scr_level), nspecies);
          ScratchPad1D<Real> nQs(mbr.team_scratch(scr_level), nspecies);
          [[maybe_unused]] ScratchPad1D<Real> Q2;
          if (coag.integrator == 3 && coag.mom_coag) {
            Q2 = ScratchPad1D<Real>(mbr.team_scratch(scr_level), nspecies);
          }

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
                                          vel, nvel, Q, nQs, alpha, cs, omega1, coag,
                                          source, nCall1, Q2);

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
  for (int i = 0; i <= 0; i++) {
    std::cout << "i=" << i << " ";
    for (int n = 0; n < 20; n++) {
      std::cout << floor_h(i, n, 0) << " ";
    }
    std::cout << std::endl;

    std::cout << "i=" << i << " ";
    for (int n = 0; n < 20; n++) {
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
  auto &coag = coag_pkg->template Param<Dust::Coagulation::CoagParams>("coag_pars");
  const int nspecies = coag.nm;
  std::vector<std::string> dust_var_names;
  for (int n = 0; n < nspecies; n++) {
    dust_var_names.push_back(dust::prim::density::name() + '_' + std::to_string(n));
    dust_var_names.push_back(dust::prim::velocity::name() + '_' + std::to_string(n));
  }
  auto &dust_subset =
      pm->mesh_data.AddShallow("dust_subset", pm->mesh_data.Get(), dust_var_names);
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
    // auto bcs = parthenon::AddBoundaryExchangeTasks(pre_comm, tl, base, pm->multilevel);
    auto &md_coag = pm->mesh_data.GetOrAdd("dust_subset", i);
    auto bcs = parthenon::AddBoundaryExchangeTasks(pre_comm, tl, md_coag, pm->multilevel);

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

template TaskStatus UpdateDustStoppingTime<Coordinates::cartesian>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::cylindrical>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::spherical1D>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::spherical2D>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::spherical3D>(MeshData<Real> *md);
template TaskStatus UpdateDustStoppingTime<Coordinates::axisymmetric>(MeshData<Real> *md);

template TaskStatus CoagulationOneStep<Coordinates::cartesian>(MeshData<Real> *md,
                                                               const Real time,
                                                               const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::cylindrical>(MeshData<Real> *md,
                                                                 const Real time,
                                                                 const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::spherical1D>(MeshData<Real> *md,
                                                                 const Real time,
                                                                 const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::spherical2D>(MeshData<Real> *md,
                                                                 const Real time,
                                                                 const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::spherical3D>(MeshData<Real> *md,
                                                                 const Real time,
                                                                 const Real dt);
template TaskStatus CoagulationOneStep<Coordinates::axisymmetric>(MeshData<Real> *md,
                                                                  const Real time,
                                                                  const Real dt);

template TaskCollection OperatorSplitDust<Coordinates::cartesian>(Mesh *pm,
                                                                  parthenon::SimTime &tm,
                                                                  const Real dt);
template TaskCollection
OperatorSplitDust<Coordinates::spherical1D>(Mesh *pm, parthenon::SimTime &tm,
                                            const Real dt);
template TaskCollection
OperatorSplitDust<Coordinates::spherical2D>(Mesh *pm, parthenon::SimTime &tm,
                                            const Real dt);
template TaskCollection
OperatorSplitDust<Coordinates::spherical3D>(Mesh *pm, parthenon::SimTime &tm,
                                            const Real dt);
template TaskCollection
OperatorSplitDust<Coordinates::cylindrical>(Mesh *pm, parthenon::SimTime &tm,
                                            const Real dt);
template TaskCollection
OperatorSplitDust<Coordinates::axisymmetric>(Mesh *pm, parthenon::SimTime &tm,
                                             const Real dt);

} // namespace Dust
