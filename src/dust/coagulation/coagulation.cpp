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
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Params &gas_params,
                                            Params &dust_params,
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

  // Gas properties
  dcpars.mmw = gas_params.Get<Real>("mu") * constants.GetAMUPhysical();
  dcpars.cross_section = pin->GetOrAddReal("dust/coagulation", "cross_section", 2.0e-15);

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
                         dust::cons::momentum, dust::prim::density, dust::prim::velocity>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Global reduction of sizes (max) and dust mass (sum) before coagulation
  Real mass_d0 = Null<Real>();
  int max_size0 = Null<int>();
  if (info_out_flag) {
    CoagulationDiagnostics<GEOM>(md, vmesh, cpars, dfloor, mass_d0, max_size0);
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
        // NOTE(@pdmullen): mbr.team_barrier() included at end of CoagulationOneCell
        // NOTE(@pdmullen): ncall could be stored or reduced (see 0a5d72b)
        int ncall = Null<int>();
        Coagulation::CoagulationOneCell(mbr, time1, dt_sync, gdens1, rhod, stime, vel,
                                        nvel, Q, nQs, alpha, cs1, omega1, coag, source,
                                        ncall, Q2);

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
      });

  // Global reduction of sizes (max) and dust mass (sum) after coagulation
  if (info_out_flag) {
    Real mass_d1 = Null<Real>();
    int max_size1 = Null<int>();
    CoagulationDiagnostics<GEOM>(md, vmesh, cpars, dfloor, mass_d1, max_size1);
    WriteCoagulationDiagnostics(md, time, dt, max_size1, max_size0, mass_d1, mass_d0);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::PreCoagulationDiagnostics
//  \brief Gather pre-coagulation diagnostics
template <Coordinates GEOM>
void CoagulationDiagnostics(MeshData<Real> *md, DiagPack_t &vmesh,
                            const geometry::CoordParams &cpars, const Real &dfloor,
                            Real &mass_d, int &max_size) {
  // Indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Reduction
  Real lmass_d = 0.0;
  int lmax_size = 1;
  Kokkos::parallel_reduce(
      "coag::diag",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {md->NumBlocks(), kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum,
                    int &lmax) {
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const Real &vol = coords.Volume();

        // Sum over nspecies
        for (int n = 0; n < vmesh.GetSize(b, dust::cons::density()); ++n) {
          lsum += vmesh(b, dust::cons::density(n), k, j, i) * vol;
        }

        // Max
        for (int n = vmesh.GetSize(b, dust::cons::density()) - 1; n >= 0; --n) {
          const Real &dens_d = vmesh(b, dust::cons::density(n), k, j, i);
          if (dens_d > dfloor) {
            lmax = std::max(lmax, n);
            break;
          }
        }
      },
      lmass_d, Kokkos::Max<int>(lmax_size));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  // Sum over all processors
  MPI_Allreduce(MPI_IN_PLACE, &lmax_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &lmass_d, 1, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  mass_d = lmass_d;
  max_size = lmax_size;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::WriteCoagulationDiagnostics
//  \brief Write coagulation diagnostics to file
void WriteCoagulationDiagnostics(MeshData<Real> *md, const Real time, const Real dt,
                                 const int max_size1, const int max_size0,
                                 const Real mass_d1, const Real mass_d0) {
  if (parthenon::Globals::my_rank == 0) {
    auto pm = md->GetParentPointer();
    auto &artemis_pkg = pm->packages.Get("artemis");
    std::string fname;
    fname.assign(artemis_pkg->template Param<std::string>("job_name"));
    fname.append("_info.dat");
    static FILE *pfile = NULL;

    // The file exists -- reopen the file in append mode
    if (pfile == NULL) {
      if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
        if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
          PARTHENON_FAIL("Error output file could not be opened");
        }
        // The file does not exist -- open the file in write mode and add headers
      } else {
        if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
          PARTHENON_FAIL("Error output file could not be opened");
        }
        std::string label = "# time dt max_size1 max_size0 mass_d1 mass_d0 delta \n";
        std::fprintf(pfile, "%s", label.c_str());
      }
    }
    std::fprintf(pfile, "  %24.16e ", time);
    std::fprintf(pfile, "  %24.16e ", dt);
    std::fprintf(pfile, "  %d  %d ", max_size1, max_size0);
    std::fprintf(pfile, "  %24.16e  %24.16e  %24.16e", mass_d1, mass_d0,
                 1.0 - mass_d0 / mass_d1);
    std::fprintf(pfile, "\n");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::InitializeArray
//  \brief Initialize static coagulation arrays
void InitializeArray(const int nm, int &pgrid, const Real &rho_p, const Real &chi,
                     const Real &a, const ParArray1D<Real> dsize, ParArray2D<int> klf,
                     ParArray1D<Real> mass_grid, ParArray3D<Real> coag3d,
                     ParArray3D<int> cpod_notzero, ParArray3D<Real> cpod_short) {
  // Initialization Part I
  const int ikdelta = coag2drv::kdelta;
  const int icoef_fett = coag2drv::coef_fett;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag1", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < nm; j++) {
          coag3d(ikdelta, i, j) = 0.0;
        }
        coag3d(ikdelta, i, i) = 1.0;
        mass_grid(i) = 4.0 * M_PI / 3.0 * rho_p * dsize(i) * dsize(i) * dsize(i);
        for (int j = 0; j < nm; j++) {
          Real tmp1 = (1.0 - 0.5 * coag3d(ikdelta, i, j));
          coag3d(icoef_fett, i, j) = M_PI * SQR(dsize(i) + dsize(j)) * tmp1;
        }
      });

  // Set fragmentation variables
  const Real ten_a = std::pow(10.0, a);
  const Real ten_ma = 1.0 / ten_a;
  const int ce = static_cast<int>(std::floor(-1.0 / a * std::log10(1.0 - ten_ma))) + 1;

  // Used in integration
  pgrid = static_cast<int>(std::floor(1.0 / a));

  // Initialization Part II
  const int iphifrag = coag2drv::phifrag;
  const int iepsfrag = coag2drv::epsfrag;
  const int iafrag = coag2drv::afrag;
  const Real frag_slope = 1.0 / 6.0; // = 2.0 - 11.0 / 6.0;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag2", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        Real sum_pF = 0.0;
        for (int j = 0; j <= i; j++) {
          coag3d(iphifrag, j, i) = std::pow(mass_grid(j), frag_slope);
          sum_pF += coag3d(iphifrag, j, i);
        }
        // normalization
        for (int j = 0; j <= i; j++) {
          coag3d(iphifrag, j, i) /= sum_pF; // switch (i,j) from fortran
        }

        // Cratering
        for (int j = 0; j <= i - pgrid - 1; j++) {
          // FRAGMENT DISTRIBUTION
          // The largest fragment has the mass of the smaller collision partner

          // Mass bin of largest fragment
          klf(i, j) = j;

          coag3d(iafrag, i, j) = (1.0 + chi) * mass_grid(j);
          //                      |_______|
          //                           |
          //                    Mass of fragments
          coag3d(iepsfrag, i, j) = chi * mass_grid(j) / (mass_grid(i) * (1.0 - ten_ma));
        }

        int i1 = std::max(0, i - pgrid);
        for (int j = i1; j <= i; j++) {
          // The largest fragment has the mass of the larger collison partner
          klf(i, j) = i;
          coag3d(iafrag, i, j) = (mass_grid(i) + mass_grid(j));
        }
      });

  // Initialization Part III
  // --> dalp array
  // --> D matrix
  // --> E matrix
  ParArray2D<Real> e("epod", nm, nm);
  int idalp = coag2drv::dalp, idpod = coag2drv::dpod;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag4", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int k) {
        for (int j = 0; j < nm; j++) {
          if (j <= k + 1 - ce) {
            coag3d(idalp, k, j) = 1.0;
            coag3d(idpod, k, j) = -mass_grid(j) / (mass_grid(k) * (ten_a - 1.0));
          } else {
            coag3d(idpod, k, j) = -1.0;
            coag3d(idalp, k, j) = 0.0;
          }
        }
        // for E matrix-------------
        const Real mkkme = mass_grid(k) * (1.0 - ten_ma);
        const Real mkpek = mass_grid(k) * (ten_a - 1.0);
        const Real mkpeme = mass_grid(k) * (ten_a - ten_ma);
        for (int j = 0; j < nm; ++j) {
          if (j <= k - ce) {
            e(k, j) = mass_grid(j) / mkkme;
          } else {
            Real theta1 = (mkpeme - mass_grid(j) < 0.0) ? 0.0 : 1.0;
            e(k, j) = (1.0 - (mass_grid(j) - mkkme) / mkpek) * theta1;
          }
        }
      });

  // Initialization Part IV
  // -->cpod array;
  ParArray3D<Real> cpod("cpod", nm, nm, nm);
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "initializeCoag6", parthenon::DevExecSpace(),
      0, nm - 1, 0, nm - 1, KOKKOS_LAMBDA(const int i, const int j) {
        // initialize to zero first
        for (int k = 0; k < nm; k++) {
          cpod(i, j, k) = 0.0;
        }
        Real mloc = mass_grid(i) + mass_grid(j);
        if (mloc < mass_grid(nm - 1)) {
          int gg = 0;
          for (int k = std::max(i, j); k < nm - 1; k++) {
            if (mloc >= mass_grid(k) && mloc < mass_grid(k + 1)) {
              gg = k;
              break;
            }
          }

          cpod(i, j, gg) =
              (mass_grid(gg + 1) - mloc) / (mass_grid(gg + 1) - mass_grid(gg));
          cpod(i, j, gg + 1) = 1.0 - cpod(i, j, gg);

          // modified cpod(*) array-------------------------
          Real dtheta_ji = (j - i - 0.5 < 0.0) ? 0.0 : 1.0; // theta(j - i - 0.5);
          for (int k = 0; k < nm; k++) {
            Real theta_kj = (k - j - 1.5 < 0.0) ? 0.0 : 1.0; // theta(k - j - 1.5)
            cpod(i, j, k) = (0.5 * coag3d(ikdelta, i, j) * cpod(i, j, k) +
                             cpod(i, j, k) * theta_kj * dtheta_ji);
          }
          cpod(i, j, j) += coag3d(idpod, j, i);
          cpod(i, j, j + 1) += e(j + 1, i) * dtheta_ji;

        } //  end if
      });

  // Initialization Part V
  // -->cpod_nonzero and cpod_short array
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "initializeCoag7", parthenon::DevExecSpace(),
      0, nm - 1, 0, nm - 1, KOKKOS_LAMBDA(const int i, const int j) {
        if (j <= i) {
          // initialize cpod_notzero(i, j, 4) and cpod_short(i, j, 4)
          for (int k = 0; k < 4; k++) {
            cpod_notzero(i, j, k) = 0;
            cpod_short(i, j, k) = 0.0;
          }
          int inc = 0;
          for (int k = 0; k < nm; ++k) {
            Real dum = cpod(i, j, k) + cpod(j, i, k);
            if (dum != 0.0) {
              cpod_notzero(i, j, inc) = k;
              cpod_short(i, j, inc) = dum;
              inc++;
            }
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef Mesh M;
typedef MeshData<Real> MD;
typedef parthenon::SimTime ST;
typedef geometry::CoordParams CP;
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
// clang-format off
template void CoagulationDiagnostics<G::cartesian>(
    MD *md, DiagPack_t &vmesh, const CP &c, const Real &d, Real &massd, int &maxsize);
template void CoagulationDiagnostics<G::cylindrical>(
    MD *md, DiagPack_t &vmesh, const CP &c, const Real &d, Real &massd, int &maxsize);
template void CoagulationDiagnostics<G::spherical1D>(
    MD *md, DiagPack_t &vmesh, const CP &c, const Real &d, Real &massd, int &maxsize);
template void CoagulationDiagnostics<G::spherical2D>(
    MD *md, DiagPack_t &vmesh, const CP &c, const Real &d, Real &massd, int &maxsize);
template void CoagulationDiagnostics<G::spherical3D>(
    MD *md, DiagPack_t &vmesh, const CP &c, const Real &d, Real &massd, int &maxsize);
template void CoagulationDiagnostics<G::axisymmetric>(
    MD *md, DiagPack_t &vmesh, const CP &c, const Real &d, Real &massd, int &maxsize);
// clang-format on

} // namespace Coagulation
} // namespace Dust
