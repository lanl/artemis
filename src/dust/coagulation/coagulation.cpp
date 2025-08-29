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
// NOTE(@sli):
// The dust coagulation code is modified from the publicly available DustPy package
//          https://github.com/stammler/dustpy
//   and from their paper (Stammler and Birnstiel (2022) ApJ 935:35)
//          "DustPy: A Python Package for Dust Evolution in Protoplanetary Disks"
//========================================================================================

// Artemis includes
#include "dust/coagulation/coagulation.hpp"
#include "artemis.hpp"
#include "dust/dust.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/units.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace Dust {
namespace Coagulation {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Coagulalation::Initialize
//! \brief Adds intialization function for coagulation package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Params &dust_params,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants) {
  auto coag = std::make_shared<StateDescriptor>("coagulation");
  Params &params = coag->AllParams();

  // Assign CoagParams
  CoagParams dcpars;

  // Units
  dcpars.rho0 = units.GetMassDensityCodeToPhysical();
  dcpars.length0 = units.GetLengthCodeToPhysical();
  PARTHENON_REQUIRE(units.GetPhysicalUnits() == ArtemisUtils::PhysicalUnits::cgs,
                    "Coagulation physics requires physical_units = cgs");

  // Species and properties
  dcpars.nm = dust_params.Get<int>("nspecies");
  dcpars.dfloor = dcpars.rho0 * dust_params.Get<Real>("dfloor");
  dcpars.rho_p = dcpars.rho0 * dust_params.Get<Real>("grain_density");
  dcpars.vfrag = pin->GetOrAddReal("dust/coagulation", "vfrag", 1.e3); // cm/s
  dcpars.integrator = pin->GetOrAddInteger("dust/coagulation", "coag_int", 3);
  dcpars.use_adaptive =
      pin->GetOrAddBoolean("dust/coagulation", "coag_use_adaptive_step", true);
  dcpars.mom_coag = pin->GetOrAddBoolean("dust/coagulation", "coag_mom_preserve", true);
  dcpars.ncall_max = pin->GetOrAddInteger("dust/coagulation", "coag_nsteps_max", 1000);
  dcpars.const_omega =
      pin->GetOrAddBoolean("dust/coagulation", "const_coag_omega", false);
  dcpars.ibounce = pin->GetOrAddBoolean("dust/coagulation", "coag_bounce", false);
  dcpars.err_eps = pin->GetOrAddReal("dust/coagulation", "err_eps", 0.1);
  dcpars.S = pin->GetOrAddReal("dust/coagulation", "S", 0.9);
  dcpars.cfl = pin->GetOrAddReal("dust/coagulation", "cfl_coag", 0.1);
  dcpars.chi = pin->GetOrAddReal("dust/coagulation", "chi", 1.0);

  // Coordinate type
  // NOTE(@pdmullen): Following @sli's earlier implementation, rho_p and dfloor use solely
  // the density unit in construction, not the one weighted by length unit
  dcpars.coord = pin->GetOrAddBoolean("dust/coagulation", "surface_density_flag", true);
  if (dcpars.coord) dcpars.rho0 *= dcpars.length0;

  // Adaptivity
  if (dcpars.use_adaptive) {
    if (dcpars.integrator == 3) {
      dcpars.pgrow = -0.5;
      dcpars.pshrink = -1.0;
    } else if (dcpars.integrator == 5) {
      dcpars.pgrow = -0.2;
      dcpars.pshrink = -0.25;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in dust coagulation initialization: " << std::endl
          << "###   You can not use this integrator with adaptive step sizing: "
          << dcpars.integrator << std::endl;
      PARTHENON_FAIL(msg);
    }
    dcpars.errcon = std::pow((5. / dcpars.S), (1. / dcpars.pgrow));
  }

  // Dust sizes
  ParArray1D<Real> dust_size("dsize", dcpars.nm);
  auto sizes = dust_params.Get<ParArray1D<Real>>("sizes");
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "code2phys", parthenon::DevExecSpace(), 0,
      dcpars.nm - 1,
      KOKKOS_LAMBDA(const int i) { dust_size(i) = sizes(i) * dcpars.length0; });

  // Cheeck if sizes are compatibile with coagulation model
  auto h_sizes = dust_size.GetHostMirrorAndCopy();
  if (std::exp(3.0 / (1.0 - dcpars.nm) * std::log(h_sizes(0) / h_sizes(dcpars.nm - 1))) >
      std::sqrt(2.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in dust with coagulation: using nspecies >"
        << 3.0 * std::log(h_sizes(dcpars.nm - 1) / h_sizes(0)) /
                   (std::log(std::sqrt(2.0))) +
               1.
        << " instead of " << dcpars.nm << std::endl;
    PARTHENON_FAIL(msg);
  }

  // Allocate CoagParam arrays
  const Real a = 3.0 * std::log10(h_sizes(0) / h_sizes(dcpars.nm - 1)) /
                 static_cast<Real>(1 - dcpars.nm);
  const int n2drv = coag2drv::last2;
  dcpars.klf = ParArray2D<int>("klf", dcpars.nm, dcpars.nm);
  dcpars.mass_grid = ParArray1D<Real>("mass_grid", dcpars.nm);
  dcpars.coagR3D = ParArray3D<Real>("coagReal3D", n2drv, dcpars.nm, dcpars.nm);
  dcpars.cpod_notzero = ParArray3D<int>("idx_nzcpod", dcpars.nm, dcpars.nm, 4);
  dcpars.cpod_short = ParArray3D<Real>("nzcpod", dcpars.nm, dcpars.nm, 4);
  InitializeArray(dcpars.nm, dcpars.pgrid, dcpars.rho_p, dcpars.chi, a, dust_size,
                  dcpars.klf, dcpars.mass_grid, dcpars.coagR3D, dcpars.cpod_notzero,
                  dcpars.cpod_short);

  // Stash CoagParams
  params.Add("coag_pars", dcpars);

  // Remaining parameters for coagulation package
  params.Add("nstep_coag", pin->GetOrAddInteger("dust/coagulation", "nstep_coag", 50));
  params.Add("dt_coag", 0.0, Params::Mutability::Restart);
  params.Add("coag_alpha", pin->GetOrAddReal("dust/coagulation", "coag_alpha", 1.e-3));
  params.Add("coag_scr_level",
             pin->GetOrAddInteger("dust/coagulation", "coag_scr_level", 0));

  // Fields for stashing solver diagnostics
  const bool info_out = pin->GetOrAddBoolean("dust/coagulation", "coag_info_out", false);
  params.Add("coag_info_out", info_out);
  if (info_out) {
    Metadata m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    coag->AddField<dust::coag::ncalls>(m);
  }

  return coag;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Dust::CoagulationDriver
//! \brief dust wrapper function for Coagulation
template <Coordinates GEOM>
TaskListStatus CoagulationDriver(Mesh *pm, parthenon::SimTime &tm) {
  auto &coag_pkg = pm->packages.Get("coagulation");
  auto *dt_coag = coag_pkg->MutableParam<Real>("dt_coag");
  int nstep_coag = coag_pkg->template Param<int>("nstep_coag");

  // Increment dt_coag
  *dt_coag += tm.dt;

  // Determine if executing coagulation this cycle...
  if ((tm.ncycle + 1) % nstep_coag != 0) return TaskListStatus::complete;

  // ...and if so, compute/report time, dt, and cycle for coagulation and reset dt_coag
  const Real ltime = tm.time + tm.dt - (*dt_coag);
  const Real ldt = (*dt_coag);
  *dt_coag = 0.0;
  if (Globals::my_rank == 0) {
    std::cout << "(Coagulation) "
              << "cycle=" << tm.ncycle << " time=" << ltime << " dt=" << ldt << std::endl;
  }

  // Create MeshData register subset for dust
  std::vector<std::string> coag_names = pm->GetVariableNames(
      std::vector<std::string>{dust::cons::density::name(), dust::cons::momentum::name(),
                               dust::prim::density::name(), dust::prim::velocity::name()},
      std::vector<int>{});
  auto &md_coag = pm->mesh_data.AddShallow("md_coag", pm->mesh_data.Get(), coag_names);

  // Assemble tasks
  TaskCollection tc;
  TaskID none(0);
  using namespace ::parthenon::Update;
  const int num_partitions = pm->DefaultNumPartitions();
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &base = pm->mesh_data.GetOrAdd("base", i);
    auto &md_coag = pm->mesh_data.GetOrAdd("md_coag", i);

    // Execute coagulation step
    auto coag_step = tl.AddTask(none, CoagulationStep<GEOM>, base.get(), ltime, ldt);

    // C2P (on dust species)
    auto pre_comm =
        tl.AddTask(coag_step, PreCommFillDerived<MeshData<Real>>, md_coag.get());

    // Set boundary conditions (both physical and logical)
    auto bcs = parthenon::AddBoundaryExchangeTasks(pre_comm, tl, md_coag, pm->multilevel);

    // P2C (on dust species)
    auto p2c = tl.AddTask(bcs, FillDerived<MeshData<Real>>, md_coag.get());
  }

  return tc.Execute();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Dust::CoagulationStep
//  \brief Wrapper function for coagulation procedure in one time step
template <Coordinates GEOM>
TaskStatus CoagulationStep(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();

  // Extract EOS
  auto &gas_pkg = pm->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  // Extract dust params
  auto &dust_pkg = pm->packages.Get("dust");
  const int &nspecies = dust_pkg->template Param<int>("nspecies");
  const auto &dust_size = dust_pkg->template Param<ParArray1D<Real>>("sizes");
  const Real &dfloor = dust_pkg->template Param<Real>("dfloor");

  // Extract coagulation params
  auto &coag_pkg = pm->packages.Get("coagulation");
  auto &coag = coag_pkg->template Param<Dust::Coagulation::CoagParams>("coag_pars");
  const Real alpha = coag_pkg->template Param<Real>("coag_alpha");
  const int nvel = (coag.coord) ? 2 : 3;
  const int scr_level = coag_pkg->template Param<int>("coag_scr_level");
  const bool info_out_flag = coag_pkg->template Param<bool>("coag_info_out");

  // Extract geometry params and units
  auto &artemis_pkg = pm->packages.Get("artemis");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");
  const auto &units = artemis_pkg->template Param<ArtemisUtils::Units>("units");
  const Real time0 = units.GetTimeCodeToPhysical();
  const Real length0 = units.GetLengthCodeToPhysical();
  const Real rho0 = coag.rho0;
  const Real vel0 = length0 / time0;

  // Indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Packing
  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::sie, dust::cons::density,
                         dust::cons::momentum, dust::prim::density, dust::prim::velocity,
                         dust::coag::ncalls>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Global reduction of sizes (max) and dust mass (sum) before coagulation
  Real mass_d0 = Null<Real>();
  int max_size0 = Null<int>();
  if (info_out_flag) {
    PreCoagulationDiagnostics<GEOM>(md, vmesh, cpars, dfloor, mass_d0, max_size0);
  }

  // Coagulation
  size_t isize = (5 + nvel + (coag.integrator == 3 && coag.mom_coag)) * nspecies;
  size_t scr_size = ScratchPad1D<Real>::shmem_size(isize);
  ArtemisUtils::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Dust::Coagulation", parthenon::DevExecSpace(),
      scr_size, scr_level, 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j,
                    const int i) {
        // Allocate scratch
        ScratchPad1D<Real> stime(mbr.team_scratch(scr_level), nspecies);
        ScratchPad1D<Real> rhod(mbr.team_scratch(scr_level), nspecies);
        ScratchPad1D<Real> vel(mbr.team_scratch(scr_level), nvel * nspecies);
        ScratchPad1D<Real> source(mbr.team_scratch(scr_level), nspecies);
        ScratchPad1D<Real> Q(mbr.team_scratch(scr_level), nspecies);
        ScratchPad1D<Real> nQs(mbr.team_scratch(scr_level), nspecies);
        ScratchPad1D<Real> Q2(mbr.team_scratch(scr_level),
                              (coag.integrator == 3 && coag.mom_coag) * nspecies);

        // Actual npecies this block reported by SparsePack
        const int nm = vmesh.GetSize(b, dust::prim::density());

        // Extract geometry
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const auto &hx = coords.GetScaleFactors();
        const auto &xv = coords.GetCellCenter();
        const auto &xcyl = coords.ConvertToCyl(xv);
        const Real irad = coag.const_omega ? 1.0 : 1.0 / xcyl[0]; // cylindrical
        const Real omega1 = irad * std::sqrt(irad) / time0;       // code-units

        // Extract gas state vector
        const Real &gdens = vmesh(b, gas::prim::density(0), k, j, i);
        const Real &gsie = vmesh(b, gas::prim::sie(0), k, j, i);
        const Real &gbulk = eos_d.BulkModulusFromDensityInternalEnergy(gdens, gsie);
        const Real cs1 = std::sqrt(gbulk / gdens) * vel0;
        const Real gdens1 = gdens * rho0;

        // Extract time(step)
        const Real time1 = time * time0;
        Real dt_sync = dt * time0;

        // Set stopping times, rhod, and veld in scratch memory
        const Real st0 = (coag.coord) ? 0.5 * M_PI * coag.rho_p / gdens1 / omega1
                                      : std::sqrt(M_PI / 8.0) * coag.rho_p / gdens1 / cs1;
        parthenon::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int n) {
              // Calculate the stopping time
              stime(n) = st0 * dust_size(n) * length0;

              // Calculate rhod, vel
              const bool gtf = vmesh(b, dust::prim::density(n), k, j, i) > dfloor;
              rhod(n) = gtf * vmesh(b, dust::prim::density(n), k, j, i) * rho0;
              for (int d = 0; d < nvel; d++) {
                const int vidx = VI(n, d);
                vel(vidx) = gtf * vmesh(b, dust::prim::velocity(vidx), k, j, i) * vel0;
              }
            });
        mbr.team_barrier();

        // Coagulation Kernel
        int ncall = 0;
        Coagulation::CoagulationOneCell(mbr, i, time1, dt_sync, gdens1, rhod, stime, vel,
                                        nvel, Q, nQs, alpha, cs1, omega1, coag, source,
                                        ncall, Q2);
        // NOTE(@pdmullen): mbr.team_barrier() included at end of CoagulationOneCell...

        // Update dust density and momentum after coagulation
        parthenon::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int n) {
              const bool gt0 = (rhod(n) > 0.0);
              vmesh(b, dust::cons::density(n), k, j, i) = gt0 * (rhod(n) / rho0);
              for (int d = 0; d < nvel; d++) {
                const int vidx = VI(n, d);
                vmesh(b, dust::cons::momentum(vidx), k, j, i) =
                    gt0 * rhod(n) * vel(vidx) * hx[d] / (rho0 * vel0);
              }
            });

        // Diagnostics
        if (info_out_flag) {
          Kokkos::single(Kokkos::PerTeam(mbr),
                         [&]() { vmesh(b, dust::coag::ncalls(), k, j, i) = ncall; });
        }
      });

  // Global reduction of sizes (max) and dust mass (sum) before coagulation. Report
  // diagnostics to file, including max_ncalls
  Real mass_d1 = Null<Real>();
  int max_size1 = Null<int>();
  int max_calls = Null<int>();
  if (info_out_flag) {
    PostCoagulationDiagnostics<GEOM>(md, vmesh, cpars, dfloor, mass_d1, max_size1,
                                     max_calls);
    WriteCoagulationDiagnostics(md, time, dt, max_calls, max_size1, max_size0, mass_d1,
                                mass_d0);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef Mesh M;
typedef MeshData<Real> MD;
typedef parthenon::SimTime ST;
template TaskListStatus CoagulationDriver<G::cartesian>(M *pm, ST &tm);
template TaskListStatus CoagulationDriver<G::cylindrical>(M *pm, ST &tm);
template TaskListStatus CoagulationDriver<G::spherical1D>(M *pm, ST &tm);
template TaskListStatus CoagulationDriver<G::spherical2D>(M *pm, ST &tm);
template TaskListStatus CoagulationDriver<G::spherical3D>(M *pm, ST &tm);
template TaskListStatus CoagulationDriver<G::axisymmetric>(M *pm, ST &tm);
template TaskStatus CoagulationStep<G::cartesian>(MD *md, const Real t, const Real dt);
template TaskStatus CoagulationStep<G::cylindrical>(MD *md, const Real t, const Real dt);
template TaskStatus CoagulationStep<G::spherical1D>(MD *md, const Real t, const Real dt);
template TaskStatus CoagulationStep<G::spherical2D>(MD *md, const Real t, const Real dt);
template TaskStatus CoagulationStep<G::spherical3D>(MD *md, const Real t, const Real dt);
template TaskStatus CoagulationStep<G::axisymmetric>(MD *md, const Real t, const Real dt);

} // namespace Coagulation
} // namespace Dust
