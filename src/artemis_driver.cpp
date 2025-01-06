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

// NOTE(PDM): The following is largely borrowed from the open-source LANL phoebus
// software, with additional extensions motivated by other downstream development.

// Parthenon includes
#include <amr_criteria/refinement_package.hpp>
#include <prolong_restrict/prolong_restrict.hpp>

// Artemis Includes
#include "artemis.hpp"
#include "artemis_driver.hpp"
#include "drag/drag.hpp"
#include "dust/dust.hpp"
#include "gas/cooling/cooling.hpp"
#include "gas/gas.hpp"
#include "gravity/gravity.hpp"
#include "nbody/nbody.hpp"
#include "radiation/imc/imc.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/integrators/artemis_integrator.hpp"
#include "sts/sts.hpp"

using namespace parthenon::driver::prelude;

namespace artemis {
//----------------------------------------------------------------------------------------
//! \fn ArtemisDriver::ArtemisDriver
//! \brief Constructor for ArtemisDriver
template <Coordinates GEOM>
ArtemisDriver<GEOM>::ArtemisDriver(ParameterInput *pin, ApplicationInput *app_in,
                                   Mesh *pm, const bool is_restart_in)
    : EvolutionDriver(pin, app_in, pm), integrator(std::make_unique<Integrator_t>(pin)),
      is_restart(is_restart_in) {

  // Fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");
  pin->CheckRequired("parthenon/mesh", "ix3_bc");
  pin->CheckRequired("parthenon/mesh", "ox3_bc");

  // Extract artemis package
  artemis_pkg = pm->packages.Get("artemis").get();

  // Fluids and/or physics requested
  do_gas = artemis_pkg->template Param<bool>("do_gas");
  do_dust = artemis_pkg->template Param<bool>("do_dust");
  do_gravity = artemis_pkg->template Param<bool>("do_gravity");
  do_rotating_frame = artemis_pkg->template Param<bool>("do_rotating_frame");
  do_cooling = artemis_pkg->template Param<bool>("do_cooling");
  do_drag = artemis_pkg->template Param<bool>("do_drag");
  do_viscosity = artemis_pkg->template Param<bool>("do_viscosity");
  do_conduction = artemis_pkg->template Param<bool>("do_conduction");
  do_nbody = artemis_pkg->template Param<bool>("do_nbody");
  do_diffusion = do_viscosity || do_conduction;
  do_sts = artemis_pkg->template Param<bool>("do_sts");
  do_radiation = artemis_pkg->template Param<bool>("do_radiation");

  // NBody initialization tasks
  if (do_nbody) {
    // NBody coupling integrator (not to be confused with the rebound integrator)
    nbody_integrator = std::make_unique<Integrator_t>(pin);
    nbody_integrator->beta[0] = integrator->beta[0];
    for (int stage = 2; stage <= nbody_integrator->nstages; stage++) {
      const Real gam0 = integrator->gam0[stage - 1];
      const Real beta = integrator->beta[stage - 1];
      nbody_integrator->beta[stage - 1] = gam0 * nbody_integrator->beta[stage - 2] + beta;
    }
    for (int stage = 1; stage <= nbody_integrator->nstages; stage++) {
      const Real gam0 = integrator->gam0[stage - 1];
      const Real beta = integrator->beta[stage - 1];
      const Real nbetam1 = nbody_integrator->beta[(stage > 1) * (stage - 2)];
      const Real nfac = beta / (gam0 * nbetam1 + beta);
      nbody_integrator->gam0[stage - 1] = (stage == 1) ? 0.0 : 1.0 - nfac;
      nbody_integrator->gam1[stage - 1] = (stage == 1) ? 1.0 : nfac;
    }

    // Restarts/initial outputs
    if (is_restart) {
      NBody::InitializeFromRestart(pm);
    } else {
      NBody::Outputs(pmesh, tm.time);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus ArtemisDriver::Step
//! \brief Assembles the tasks associated with a step for the ArtemisDriver
template <Coordinates GEOM>
TaskListStatus ArtemisDriver<GEOM>::Step() {
  // Prepare registers
  PreStepTasks();

  // Execute explicit, unsplit physics
  auto status = StepTasks().Execute();
  if (status != TaskListStatus::complete) return status;

  // Execute operator split physics
  if (do_radiation) status = IMC::JaybenneIMC<GEOM>(pmesh, tm.time, tm.dt);
  if (status != TaskListStatus::complete) return status;

  // Compute new dt, (de)refine, and handle sparse (if enabled)
  status = PostStepTasks().Execute();

  // Extra artemis outputs
  if (do_nbody) NBody::Outputs(pmesh, tm.time + tm.dt);

  return status;
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisDriver::PreStepTasks
//! \brief Defines the tasks executed prior to the main integrator in the ArtemisDriver
template <Coordinates GEOM>
void ArtemisDriver<GEOM>::PreStepTasks() {
  // set the integration timestep
  integrator->dt = tm.dt;
  if (do_nbody) nbody_integrator->dt = tm.dt;

  // assign registers with fields required in unsplit integration
  parthenon::Metadata::FlagCollection flags;
  flags.Exclude(parthenon::Metadata::GetUserFlag("OperatorSplit"));
  auto names = pmesh->GetVariableNames(flags);
  auto &base = pmesh->mesh_data.Get();
  auto &u0 = pmesh->mesh_data.AddShallow("u0", base, names);
  auto &u1 = pmesh->mesh_data.Add("u1", u0);

  // Assign sts registers and First stage of STS integration
  auto &gas_pkg = pmesh->packages.Get("gas");
  auto min_diff_dt = gas_pkg->template Param<Real>("diff_dt");
  if (do_sts) {
    // compute the number of stages needed for the STS integrator
    int s_sts =
        static_cast<int>(0.5 * (std::sqrt(9.0 + 16.0 * tau / min_diff_dt) - 1.0)) + 1;
    if (s_sts % 2 == 0) s_sts += 1;
    
    if (parthenon::Globals::my_rank == 0) {
      const auto ratio = 2.0 * tau / mindt_diff;
      std::cout << "STS ratio: " << ratio << ", Taking " << s_sts << " steps." << std::endl;
      if (ratio > 400.1) {
        std::cout << "WARNING: ratio is > 400. Proceed at own risk." << std::endl;
      }
    }

    if (STSInt::rkl1){
      // (TODO) RKL1 : Full timestep dt_sts
      //STSRKL1(pmesh, tm.time, tm.dt, s_sts);
    }else if (STSInt::rkl2){
      // (TODO) RKL2 : // eq (21) using half hyperbolic timestep 
      // due to Strang split
      //STSRKL2(pmesh, tm.time, 0.5*tm.dt, s_sts);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskCollection ArtemisDriver::PreStepTasks
//! \brief Defines the main integrator's TaskCollection for the ArtemisDriver
template <Coordinates GEOM>
TaskCollection ArtemisDriver<GEOM>::StepTasks() {
  using TQ = TaskQualifier;
  TaskCollection tc;
  // Return empty TaskCollection if all unsplit physics disabled
  if (!(do_gas) && !(do_dust)) return tc;

  using namespace ::parthenon::Update;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // Deep copy u0 into u1 for integrator logic
  auto &init_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = init_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
    tl.AddTask(none, ArtemisUtils::DeepCopyConservedData, u1.get(), u0.get());
  }

  // Now do explicit integration of unsplit physics
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    const Real time = tm.time;
    const Real bdt = integrator->beta[stage - 1] * integrator->dt;

    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);

      // Start looking for incoming messages (including for flux correction)
      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
      auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, u0);

      // Compute hydrodynamic fluxes
      // NOTE(AMD): 1st stage of VL2 uses piecewise constant reconstruction
      const bool do_pcm = ((stage == 1) && (integrator->GetName() == "vl2"));
      TaskID gas_flx = none, dust_flx = none;
      if (do_gas) gas_flx = tl.AddTask(none, Gas::CalculateFluxes, u0.get(), do_pcm);
      if (do_dust) dust_flx = tl.AddTask(none, Dust::CalculateFluxes, u0.get(), do_pcm);

      // Compute (gas) diffusive fluxes
      TaskID diff_flx = none;
      if (do_diffusion && do_gas && !do_sts) {
        auto zf = tl.AddTask(none, Gas::ZeroDiffusionFlux, u0.get());
        TaskID vflx = zf, tflx = zf;
        if (do_viscosity) vflx = tl.AddTask(zf, Gas::ViscousFlux<GEOM>, u0.get());
        if (do_conduction) tflx = tl.AddTask(zf | vflx, Gas::ThermalFlux<GEOM>, u0.get());
        diff_flx = vflx | tflx;
      }

      // Communicate and set fluxes
      auto send_flx =
          tl.AddTask(gas_flx | dust_flx | diff_flx,
                     parthenon::SendBoundBufs<parthenon::BoundaryType::flxcor_send>, u0);
      auto recv_flx = tl.AddTask(start_flx_recv, parthenon::ReceiveFluxCorrections, u0);
      auto set_flx = tl.AddTask(recv_flx, parthenon::SetFluxCorrections, u0);

      // Apply flux divergence
      auto update =
          tl.AddTask(gas_flx | dust_flx | set_flx, ArtemisUtils::ApplyUpdate<GEOM>,
                     u0.get(), u1.get(), stage, integrator.get());

      // Apply "coordinate source terms"
      TaskID gas_coord_src = update, dust_coord_src = update;
      if (do_gas) gas_coord_src = tl.AddTask(update, Gas::FluxSource, u0.get(), bdt);
      if (do_dust) dust_coord_src = tl.AddTask(update, Dust::FluxSource, u0.get(), bdt);

      // Apply (gas) diffusion sources
      // NOTE(@pdmullen): I believe set_flx dependency implicitly inside gas_coord_src,
      // but included below explicitly for posterity
      TaskID gas_diff_src = gas_coord_src | diff_flx | set_flx;
      if (do_diffusion && do_gas && !do_sts) {
        gas_diff_src = tl.AddTask(gas_coord_src | diff_flx | set_flx,
                                  Gas::DiffusionUpdate<GEOM>, u0.get(), bdt);
      }

      // Apply gravity source term
      TaskID gravity_src = gas_coord_src | dust_coord_src | gas_diff_src;
      if (do_gravity) {
        gravity_src = tl.AddTask(gas_coord_src | dust_coord_src | gas_diff_src,
                                 Gravity::ExternalGravity<GEOM>, u0.get(), time, bdt);
      }

      // Apply rotating frame source term
      TaskID rframe_src = gravity_src;
      if (do_rotating_frame) {
        rframe_src = tl.AddTask(gravity_src, RotatingFrame::RotatingFrameForce, u0.get(),
                                time, bdt);
      }

      // Apply drag source term
      TaskID drag_src = rframe_src;
      if (do_drag) {
        drag_src = tl.AddTask(rframe_src, Drag::DragSource<GEOM>, u0.get(), time, bdt);
      }

      // Apply cooling source term
      TaskID cooling_src = drag_src;
      if (do_cooling) {
        cooling_src =
            tl.AddTask(drag_src, Gas::Cooling::CoolingSource<GEOM>, u0.get(), time, bdt);
      }

      // Set auxillary fields
      auto set_aux =
          tl.AddTask(cooling_src, ArtemisDerived::SetAuxillaryFields<GEOM>, u0.get());

      // Set (remaining) fields to be communicated
      auto pre_comm = tl.AddTask(set_aux, PreCommFillDerived<MeshData<Real>>, u0.get());

      // Set boundary conditions (both physical and logical)
      auto bcs = parthenon::AddBoundaryExchangeTasks(pre_comm, tl, u0, pmesh->multilevel);

      // Update primitive variables
      auto c2p = tl.AddTask(TQ::local_sync, bcs, FillDerived<MeshData<Real>>, u0.get());

      // Advance nbody integrator
      TaskID nbadv = c2p;
      if (do_nbody) {
        nbadv = tl.AddTask(TQ::once_per_region, c2p, NBody::Advance, pmesh, time, stage,
                           nbody_integrator.get());
      }
    }
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn TaskCollection ArtemisDriver::PreStepTasks
//! \brief Defines the TaskCollection for post step tasks in the ArtemisDriver
template <Coordinates GEOM>
TaskCollection ArtemisDriver<GEOM>::PostStepTasks() {
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  // TODO: Implement RKL2 STS integration

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &post_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = post_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto new_dt = tl.AddTask(none, EstimateTimestep<MeshData<Real>>, u0.get());
    auto refine = new_dt;
    if (pmesh->adaptive) {
      refine = tl.AddTask(new_dt, parthenon::Refinement::Tag<MeshData<Real>>, u0.get());
    }
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef Mesh M;
typedef ParameterInput PI;
typedef ApplicationInput AI;
template ArtemisDriver<C::cartesian>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<C::cartesian>::Step();
template void ArtemisDriver<C::cartesian>::PreStepTasks();
template TaskCollection ArtemisDriver<C::cartesian>::StepTasks();
template TaskCollection ArtemisDriver<C::cartesian>::PostStepTasks();
template ArtemisDriver<C::cylindrical>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<C::cylindrical>::Step();
template void ArtemisDriver<C::cylindrical>::PreStepTasks();
template TaskCollection ArtemisDriver<C::cylindrical>::StepTasks();
template TaskCollection ArtemisDriver<C::cylindrical>::PostStepTasks();
template ArtemisDriver<C::spherical1D>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<C::spherical1D>::Step();
template void ArtemisDriver<C::spherical1D>::PreStepTasks();
template TaskCollection ArtemisDriver<C::spherical1D>::StepTasks();
template TaskCollection ArtemisDriver<C::spherical1D>::PostStepTasks();
template ArtemisDriver<C::spherical2D>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<C::spherical2D>::Step();
template void ArtemisDriver<C::spherical2D>::PreStepTasks();
template TaskCollection ArtemisDriver<C::spherical2D>::StepTasks();
template TaskCollection ArtemisDriver<C::spherical2D>::PostStepTasks();
template ArtemisDriver<C::spherical3D>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<C::spherical3D>::Step();
template void ArtemisDriver<C::spherical3D>::PreStepTasks();
template TaskCollection ArtemisDriver<C::spherical3D>::StepTasks();
template TaskCollection ArtemisDriver<C::spherical3D>::PostStepTasks();
template ArtemisDriver<C::axisymmetric>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<C::axisymmetric>::Step();
template void ArtemisDriver<C::axisymmetric>::PreStepTasks();
template TaskCollection ArtemisDriver<C::axisymmetric>::StepTasks();
template TaskCollection ArtemisDriver<C::axisymmetric>::PostStepTasks();

} // namespace artemis
