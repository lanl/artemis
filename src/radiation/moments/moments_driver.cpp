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
#include "artemis.hpp"
#include "derived/fill_derived.hpp"
#include "radiation/moments/moments.hpp"
#include "utils/integrators/artemis_integrator.hpp"
#include "utils/units.hpp"

namespace Moments {

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus MomentsDriver
//! \brief
template <Coordinates GEOM>
TaskListStatus MomentsDriver(Mesh *pmesh, const SimTime &tm,
                             parthenon::LowStorageIntegrator *integrator) {
  // Craft a series of **equal** substeps that sum to the unsplit step
  const Real dtlimit = Moments::EstimateTimeStep<GEOM>(pmesh);
  const int nsteps = static_cast<int>(std::ceil(integrator->dt / dtlimit));
  integrator->dt = integrator->dt / nsteps;

  // Report number of substeps
  if (tm.ncycle % tm.ncycle_out == 0) {
    if (Globals::my_rank == 0) {
      std::cout << "(Radiation Moments) "
                << "Executing " << nsteps << " substeps"
                << " with dt=" << integrator->dt << std::endl;
    }
  }

  // Execute MomentsTasks over substeps
  for (int step = 1; step <= nsteps; step++) {
    auto status = MomentsTasks<GEOM>(pmesh, tm, integrator).Execute();
    if (status != TaskListStatus::complete) return status;
  }

  return TaskListStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus ArtemisDriver::MomentsTasks
//! \brief
template <Coordinates GEOM>
TaskCollection MomentsTasks(Mesh *pmesh, const SimTime &tm,
                            parthenon::LowStorageIntegrator *integrator) {
  using TQ = TaskQualifier;
  TaskCollection tc;

  using namespace ::parthenon::Update;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // Deep copy u0c into u1c for integrator logic
  auto &init_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = init_region[i];
    auto &u0m = pmesh->mesh_data.GetOrAdd("u0m", i);
    auto &u1m = pmesh->mesh_data.GetOrAdd("u1m", i);
    tl.AddTask(none, ArtemisUtils::DeepCopyConservedData, u1m.get(), u0m.get());
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
    tl.AddTask(none, ArtemisUtils::DeepCopyConservedData, u1.get(), u0.get());
  }

  // Now do explicit subcycling of radiation moment physics
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    const Real g0 = integrator->gam0[stage - 1];
    const Real g1 = integrator->gam1[stage - 1];
    const Real bdt = integrator->beta[stage - 1] * integrator->dt;

    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
      auto &u0m = pmesh->mesh_data.GetOrAdd("u0m", i);
      auto &u1m = pmesh->mesh_data.GetOrAdd("u1m", i);
      auto &u0c = pmesh->mesh_data.GetOrAdd("u0c", i);

      // Start looking for incoming messages (including for flux correction)
      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0c);
      auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, u0m);

      // Compute radiation fluxes
      auto rad_flx = tl.AddTask(none, Moments::CalculateFluxes, u0m.get());

      // Communicate and set fluxes
      auto send_flx = tl.AddTask(
          rad_flx, parthenon::SendBoundBufs<parthenon::BoundaryType::flxcor_send>, u0m);
      auto recv_flx = tl.AddTask(start_flx_recv, parthenon::ReceiveFluxCorrections, u0m);
      auto set_flx = tl.AddTask(recv_flx, parthenon::SetFluxCorrections, u0m);

      // Apply RK logic (and potentially flux divergence) operator to fields
      auto rupdate = tl.AddTask(rad_flx | set_flx, ArtemisUtils::ApplyUpdate<GEOM>,
                                u0m.get(), u1m.get(), g0, g1, bdt);
      auto cupdate = tl.AddTask(none, ArtemisUtils::ApplyUpdate<GEOM, false>, u0.get(),
                                u1.get(), g0, g1, 0.0);

      // Apply "coordinate source terms"
      auto coord_src = tl.AddTask(rupdate | cupdate, Moments::FluxSource, u0m.get(), bdt);

      // Apply matter-coupling step
      auto coupling =
          tl.AddTask(coord_src, Moments::MatterCoupling<GEOM>, u0c.get(), bdt);

      // Set auxillary fields
      auto set_aux =
          tl.AddTask(coupling, ArtemisDerived::SetAuxillaryFields<GEOM>, u0c.get());

      // Set (remaining) fields to be communicated
      auto pre_comm = tl.AddTask(set_aux, PreCommFillDerived<MeshData<Real>>, u0c.get());

      // Set boundary conditions (both physical and logical)
      auto bcs =
          parthenon::AddBoundaryExchangeTasks(pre_comm, tl, u0c, pmesh->multilevel);

      // Update primitive variables
      auto c2p = tl.AddTask(bcs, FillDerived<MeshData<Real>>, u0c.get());
    }
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef Mesh M;
typedef SimTime ST;
typedef parthenon::LowStorageIntegrator LSI;
template TaskListStatus MomentsDriver<G::cartesian>(M *pm, const ST &t, LSI *ii);
template TaskListStatus MomentsDriver<G::cylindrical>(M *pm, const ST &t, LSI *ii);
template TaskListStatus MomentsDriver<G::spherical1D>(M *pm, const ST &t, LSI *ii);
template TaskListStatus MomentsDriver<G::spherical2D>(M *pm, const ST &t, LSI *ii);
template TaskListStatus MomentsDriver<G::spherical3D>(M *pm, const ST &t, LSI *ii);
template TaskListStatus MomentsDriver<G::axisymmetric>(M *pm, const ST &t, LSI *ii);
template TaskCollection MomentsTasks<G::cartesian>(M *pm, const ST &t, LSI *ii);
template TaskCollection MomentsTasks<G::cylindrical>(M *pm, const ST &t, LSI *ii);
template TaskCollection MomentsTasks<G::spherical1D>(M *pm, const ST &t, LSI *ii);
template TaskCollection MomentsTasks<G::spherical2D>(M *pm, const ST &t, LSI *ii);
template TaskCollection MomentsTasks<G::spherical3D>(M *pm, const ST &t, LSI *ii);
template TaskCollection MomentsTasks<G::axisymmetric>(M *pm, const ST &t, LSI *ii);

} // namespace Moments
