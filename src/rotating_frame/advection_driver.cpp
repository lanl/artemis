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

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "derived/fill_derived.hpp"
#include "geometry/geometry.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/integrators/artemis_integrator.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace RotatingFrame {

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus RotatingFrame::Advect
//! \brief Executes linear advection term for orbital advection
TaskListStatus Advect(Mesh *pmesh, const SimTime &tm) {
  // Craft a series of **equal** subsetps that sum to the unsplit step
  const Real dtlimit = EstimateTimestep(pmesh, 1.0);
  const int nsteps = static_cast<int>(std::ceil(tm.dt / dtlimit));
  const Real scdt = tm.dt / nsteps;

  // Report number of substeps
  if (tm.ncycle % tm.ncycle_out == 0) {
    if (Globals::my_rank == 0) {
      std::cout << "(Linear Advection) "
                << "Executing " << nsteps << " substeps"
                << " with dt=" << scdt << std::endl;
    }
  }

  // Execute LinearAdvectionStep over substeps
  for (int step = 1; step <= nsteps; step++) {
    auto status = LinearAdvectionStep(pmesh, tm, scdt).Execute();
    if (status != TaskListStatus::complete) return status;
  }

  return TaskListStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection LinearAdvectionStep
TaskCollection LinearAdvectionStep(Mesh *pmesh, const SimTime &tm, const Real scdt) {
  TaskCollection tc;
  if (!(pmesh->ndim >= 2)) return tc;

  // Construct TaskCollection
  using namespace ::parthenon::Update;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // Operator split linear advection
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);

    auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
    auto update = tl.AddTask(start_recv, LagrangeRemap, u0.get(), scdt);
    auto set_aux = tl.AddTask(
        update, ArtemisDerived::SetAuxillaryFields<Coordinates::cartesian>, u0.get());
    auto c2p = tl.AddTask(set_aux, PreCommFillDerived<MeshData<Real>>, u0.get());
    auto bcs = parthenon::AddBoundaryExchangeTasks(c2p, tl, u0, pmesh->multilevel);
    auto p2c = tl.AddTask(bcs, FillDerived<MeshData<Real>>, u0.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RotatingFrame::LagrangeRemap
//! \brief
TaskStatus LagrangeRemap(MeshData<Real> *u0, const Real scdt) {
  using parthenon::MakePackDescriptor;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Artemis package and params
  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  // Rotating frame package and params
  auto &rframe_pkg = pm->packages.Get("rotating_frame");
  const Real qshear = rframe_pkg->template Param<Real>("qshear");
  const Real om0 = rframe_pkg->template Param<Real>("omega");
  const ReconstructionMethod recon =
      rframe_pkg->template Param<ReconstructionMethod>("reconstruction");

  // Extract integrator weights
  const Real dwdt = -qshear * om0 * scdt;

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy, dust::cons::density,
                         dust::cons::momentum>(resolved_pkgs.get());
  auto v0 = desc.GetPack(u0);

  // Call upwind advection routines with requested recon
  if (recon == ReconstructionMethod::pcm) {
    return LagrangeRemapImpl<ReconstructionMethod::pcm>(u0, v0, dwdt);
  } else if (recon == ReconstructionMethod::plm) {
    return LagrangeRemapImpl<ReconstructionMethod::plm>(u0, v0, dwdt);
  } else if (recon == ReconstructionMethod::ppm) {
    return LagrangeRemapImpl<ReconstructionMethod::ppm>(u0, v0, dwdt);
  } else {
    PARTHENON_FAIL("Unsupported reconstruction method in rotating_frame");
  }

  return TaskStatus::complete;
}

} // namespace RotatingFrame
