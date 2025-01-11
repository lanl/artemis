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
#include "dust/dust.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/fluxes/fluid_fluxes.hpp"
#include "utils/history.hpp"
#include "utils/units.hpp"

using ArtemisUtils::VI;

namespace Dust {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Dust::Initialize
//! \brief Adds intialization function for dust hydrodynamics package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units) {
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

  // Floors
  const Real dfloor = pin->GetOrAddReal("dust", "dfloor", 1.0e-20);
  params.Add("dfloor", dfloor);

  // Number of dust species
  const int nspecies = pin->GetOrAddInteger("dust", "nspecies", 1);
  params.Add("nspecies", nspecies);
  std::vector<int> dustids;
  for (int n = 0; n < nspecies; ++n)
    dustids.push_back(n);

  // Dust sizes
  const auto size_dist = pin->GetOrAddString("dust", "size_input", "direct");
  const auto input_units = pin->GetOrAddString("dust", "input_units", "cgs");
  Real length_conv = Null<Real>();
  Real rho_conv = Null<Real>();
  if (input_units == "cgs") {
    length_conv = units.GetLengthPhysicalToCode();
    rho_conv = units.GetMassDensityPhysicalToCode();
  } else if (input_units == "problem") {
    length_conv = 1.;
    rho_conv = 1.;
  } else {
    PARTHENON_FAIL("dust/input_units can only be cgs or problem.");
  }

  if (size_dist == "linspace") {
    // uniform
    auto min_size = length_conv * pin->GetReal("dust", "min_size");
    auto max_size = length_conv * pin->GetReal("dust", "max_size");

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
    const auto lmin = std::log10(length_conv * pin->GetReal("dust", "min_size"));
    const auto lmax = std::log10(length_conv * pin->GetReal("dust", "max_size"));

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

  } else if (size_dist == "direct") {
    // specify them directly
    auto sizes_v = pin->GetVector<Real>("dust", "sizes");

    ParArray1D<Real> sizes("sizes", nspecies);
    auto h_sizes = sizes.GetHostMirror();
    for (int n = 0; n < nspecies; n++) {
      h_sizes(n) = length_conv * sizes_v[n];
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
        h_sizes(n) = length_conv * data[0][n];
      }
    } else {
      for (int n = 0; n < nspecies; n++) {
        h_sizes(n) = length_conv * data[n][0];
      }
    }
    sizes.DeepCopy(h_sizes);
    params.Add("sizes", sizes);
    params.Add("h_sizes", h_sizes);
  } else {
    PARTHENON_FAIL("dust/size_input not recognized!");
  }

  // Dust density
  params.Add("grain_density", rho_conv * pin->GetOrAddReal("dust", "grain_density", 1.0));

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
//! \fn  Real Dust::EstimateTimestepMesh
//! \brief Compute dust hydrodynamics timestep
template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &dust_pkg = pm->packages.Get("dust");
  auto &params = dust_pkg->AllParams();

  static auto desc =
      MakePackDescriptor<dust::prim::density, dust::prim::velocity>(resolved_pkgs.get());
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
          ldt = std::min(ldt, 1.0 / denom);
        }
      },
      Kokkos::Min<Real>(min_dt));

  const auto cfl_number = params.template Get<Real>("cfl");
  return cfl_number * min_dt;
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

} // namespace Dust
