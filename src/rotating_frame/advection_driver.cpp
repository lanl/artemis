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

// Artemis includes
#include "rotating_frame.hpp"
#include "artemis.hpp"
#include "rotating_frame_impl.hpp"
#include "utils/artemis_utils.hpp"

namespace RotatingFrame {

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus RotatingFrame::Advect
//! \brief Executes linear advection term for orbital advection
TaskListStatus Advect(Mesh *pmesh, const SimTime &tm) {
  PARTHENON_INSTRUMENT
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
  PARTHENON_INSTRUMENT
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

  return rframe_pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RotatingFrame::RotatingFrameForce
//! \brief
TaskStatus LagrangeRemap(MeshData<Real> *u0, const Real scdt) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const auto coords = artemis_pkg->Param<Coordinates>("coords");

  auto &rframe_pkg = pm->packages.Get("rotating_frame");
  const Real qshear = rframe_pkg->template Param<Real>("qshear");
  const Real om0 = rframe_pkg->template Param<Real>("omega");
  const auto recon = rframe_pkg->template Param<ReconstructionMethod>("recon");

  // Extract integrator weights
  const Real dwdt = -qshear * om0 * scdt;

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy, dust::cons::density,
                         dust::cons::momentum>(resolved_pkgs.get());
  auto v0 = desc.GetPack(u0);
  static auto desc_g =
      MakePackDescriptor<geom::vol, geom::x1v, geom::x2v, geom::x3v, geom::dx1, geom::dx2,
                         geom::dx3>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(u0);

  // Call upwind advection routines with requested recon
  if (recon == ReconstructionMethod::pcm) {
    return LagrangeRemapImpl<ReconstructionMethod::pcm>(u0, v0, vg, dwdt);
  } else if (recon == ReconstructionMethod::plm) {
    return LagrangeRemapImpl<ReconstructionMethod::plm>(u0, v0, vg, dwdt);
  } else if (recon == ReconstructionMethod::ppm) {
    return LagrangeRemapImpl<ReconstructionMethod::ppm>(u0, v0, vg, dwdt);
  } else {
    PARTHENON_FAIL("Rotating frame is not consistent with this coordinate system");
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Real RotatingFrame::EstimateTimestep
//! \brief Compute multiple of linear advection timestep (if do_shear)
template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  auto pmesh = md->GetParentPointer();
  const bool do_shear = pmesh->packages.Get("artemis")->template Param<bool>("do_shear");
  if (!(do_shear)) return Big<Real>();

  auto &rframe_pkg = pmesh->packages.Get("rotating_frame");
  const Real &dt_ratio = rframe_pkg->template Param<Real>("dt_ratio");
  return EstimateTimestep(pmesh, dt_ratio);
}

//----------------------------------------------------------------------------------------
//! \fn Real RotatingFrame::EstimateTimeStep
//! \brief Not enrolled in parthenon's determination for global dt
Real EstimateTimestep(parthenon::Mesh *pmesh, const Real dt_ratio) {
  PARTHENON_INSTRUMENT
  // Extract rotating frame params
  auto &rframe_pkg = pmesh->packages.Get("rotating_frame");
  const Real &om0 = rframe_pkg->Param<Real>("omega");
  const Real &qshear = rframe_pkg->Param<Real>("qshear");
  const Real &cfl = rframe_pkg->template Param<Real>("cfl");

  // Compute linear advection timestep to sub-cycle
  Real min_dt = Big<Real>();
  for (auto const &pmb : pmesh->block_list) {
    const auto &reg = pmb->block_size;
    const auto wp = BackgroundVelocity<Coordinates::cartesian>(qshear, om0, reg.xmax_[0]);
    const auto wm = BackgroundVelocity<Coordinates::cartesian>(qshear, om0, reg.xmin_[0]);
    const Real dx2 = (reg.xmax_[1] - reg.xmin_[1]) / reg.nx_[1];
    min_dt = std::min(min_dt, dx2 / std::max(std::abs(wp[1]), std::abs(wm[1])));
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &min_dt, 1, MPI_PARTHENON_REAL, MPI_MIN,
                                    MPI_COMM_WORLD));
#endif
  return cfl * min_dt * dt_ratio;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef MeshData<Real> MD;
template Real EstimateTimestepMesh<Coordinates::cartesian>(MD *md);
template Real EstimateTimestepMesh<Coordinates::cylindrical>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical1D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical2D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical3D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::axisymmetric>(MD *md);

} // namespace RotatingFrame
