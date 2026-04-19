//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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
#include "dust/coagulation/coagulation.hpp"
#include "dust/dust.hpp"
#include "gas/cooling/cooling.hpp"
#include "gas/gas.hpp"
#include "gravity/gravity.hpp"
#include "nbody/nbody.hpp"
#include "radiation/imc/imc.hpp"
#include "radiation/moments/moments.hpp"
#include "radiation/raytrace/raytrace.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "self_gravity/self_gravity.hpp"
#include "utils/integrators/artemis_integrator.hpp"

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
  do_self_gravity = artemis_pkg->template Param<bool>("do_self_gravity");
  do_rotating_frame = artemis_pkg->template Param<bool>("do_rotating_frame");
  do_orbital_advection = artemis_pkg->template Param<bool>("do_orbital_advection");
  do_cooling = artemis_pkg->template Param<bool>("do_cooling");
  do_drag = artemis_pkg->template Param<bool>("do_drag");
  do_viscosity = artemis_pkg->template Param<bool>("do_viscosity");
  do_conduction = artemis_pkg->template Param<bool>("do_conduction");
  do_nbody = artemis_pkg->template Param<bool>("do_nbody");
  do_diffusion = do_viscosity || do_conduction;
  do_imc = artemis_pkg->template Param<bool>("do_imc");
  do_moment = artemis_pkg->template Param<bool>("do_moment");
  do_coagulation = artemis_pkg->template Param<bool>("do_coagulation");
  do_raytrace = artemis_pkg->template Param<bool>("do_raytrace");

  // Update fluxes option--gas fields are needed for radiation temperature updates but for
  // rad-only test problems turn off advection
  update_fluxes = artemis_pkg->template Param<bool>("update_fluxes");

  ndim = pm->ndim;

  // Moments integrator
  if (do_moment) {
    auto rad_int = pin->GetOrAddString("radiation/moment", "integrator", "rk2");
    PARTHENON_REQUIRE(((rad_int == "rk1") || (rad_int == "rk2") || (rad_int == "rk3")),
                      "radiation/integrator must be rk1,rk2, or rk3.")
    rad_integrator = std::make_unique<Integrator_t>(rad_int);
  }

  // NBody integrator and initialization
  if (do_nbody) {
    // NBody coupling integrator (not to be confused with the rebound integrator)
    // NOTE(AMD): I bet this can be done with the parthenon Butcher integrator
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
  PARTHENON_INSTRUMENT
  // Prepare registers
  PreStepTasks();
  TaskListStatus status = TaskListStatus::complete;
  // Execute explicit, unsplit physics
  if (do_raytrace) {
    status = RT::RaytraceDriver(pmesh);
    if (status != TaskListStatus::complete) return status;
  }

  status = StepTasks().Execute();
  if (status != TaskListStatus::complete) return status;

  // Operator split, background linear advection
  if (do_orbital_advection && (ndim > 1)) {
    // only if the advection direction is included
    if constexpr (!geometry::is_axisymmetric<GEOM>()) {
      status = RotatingFrame::Advect<GEOM>(pmesh, tm);
      if (status != TaskListStatus::complete) return status;
    }
  }

  // Operator split, IMC/DDMC radiation with Jaybenne
  if (do_imc) {
    status = IMC::JaybenneIMC<GEOM>(pmesh, tm, tm.dt);
    if (status != TaskListStatus::complete) return status;
  }

  // Operator split, moments subcyling (M1 or P1)
  if (do_moment) {
    status = Moments::MomentsDriver<GEOM>(pmesh, tm, rad_integrator.get());
    if (status != TaskListStatus::complete) return status;
  }

  // Operator split, dust coagulation
  if (do_coagulation) status = Dust::Coagulation::CoagulationDriver<GEOM>(pmesh, tm);
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
  PARTHENON_INSTRUMENT
  // set the integration timestep
  integrator->dt = tm.dt;
  if (do_nbody) nbody_integrator->dt = tm.dt;
  if (do_moment) rad_integrator->dt = tm.dt;

  // Extract Base MeshData Registers
  auto &base = pmesh->mesh_data.Get();

  // Assign registers with fields required in unsplit integration
  parthenon::Metadata::FlagCollection unsplit_flags;
  unsplit_flags.Exclude(parthenon::Metadata::GetUserFlag("OperatorSplit"));
  auto unsplit_names = pmesh->GetVariableNames(unsplit_flags);
  auto &u0 = pmesh->mesh_data.AddShallow("u0", base, unsplit_names);
  auto &u1 = pmesh->mesh_data.Add("u1", u0);

  // Assign registers with fields required for moments
  if (do_moment) {
    parthenon::Metadata::FlagCollection moments_flags, geom_flags;
    moments_flags.TakeUnion(pmesh->packages.Get("moments")->GetMetadataFlag());
    geom_flags.TakeUnion(pmesh->packages.Get("geometry")->GetMetadataFlag());
    auto moment_names = pmesh->GetVariableNames(moments_flags);
    auto geom_names = pmesh->GetVariableNames(geom_flags);
    auto coupling_names = unsplit_names;
    coupling_names.insert(coupling_names.end(), moment_names.begin(), moment_names.end());
    moment_names.insert(moment_names.end(), geom_names.begin(), geom_names.end());
    auto &u0c = pmesh->mesh_data.AddShallow("u0c", base, coupling_names);
    auto &u0m = pmesh->mesh_data.AddShallow("u0m", base, moment_names);
    auto &u1m = pmesh->mesh_data.Add("u1m", u0m);
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskCollection ArtemisDriver::PreStepTasks
//! \brief Defines the main integrator's TaskCollection for the ArtemisDriver
template <Coordinates GEOM>
TaskCollection ArtemisDriver<GEOM>::StepTasks() {
  PARTHENON_INSTRUMENT
  using TQ = TaskQualifier;
  using namespace ::parthenon::Update;
  TaskCollection tc;

  // Return empty TaskCollection if all unsplit physics disabled
  if (!(do_gas) && !(do_dust)) return tc;

  // Extract parameters to construct TaskCollection
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
    const Real g0 = integrator->gam0[stage - 1];
    const Real g1 = integrator->gam1[stage - 1];
    const Real bdt = integrator->beta[stage - 1] * integrator->dt;

    // Compute gravitational potential
    if (do_self_gravity) SelfGravity::SolvePoisson(tc, pmesh);

    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);

      // Start looking for incoming messages (including for flux correction)
      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
      auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, u0);

      // Compute hydrodynamic fluxes
      // NOTE(@adempsey): 1st stage of VL2 uses piecewise constant reconstruction
      const bool do_pcm = ((stage == 1) && (integrator->GetName() == "vl2"));
      TaskID gas_flx = none, dust_flx = none;
      if (do_gas && update_fluxes)
        gas_flx = tl.AddTask(none, Gas::CalculateFluxes, u0.get(), do_pcm);
      if (do_dust) dust_flx = tl.AddTask(none, Dust::CalculateFluxes, u0.get(), do_pcm);

      // Compute (gas) diffusive fluxes
      TaskID diff_flx = none;
      if (do_diffusion && do_gas) {
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
                     u0.get(), u1.get(), g0, g1, bdt);

      // Apply "coordinate source terms"
      TaskID gas_coord_src = update, dust_coord_src = update;
      if (do_gas) gas_coord_src = tl.AddTask(update, Gas::FluxSource, u0.get(), bdt);
      if (do_dust) dust_coord_src = tl.AddTask(update, Dust::FluxSource, u0.get(), bdt);

      // Apply (gas) diffusion sources
      TaskID gas_diff_src = gas_coord_src | diff_flx | set_flx;
      if (do_diffusion && do_gas) {
        gas_diff_src = tl.AddTask(gas_coord_src | diff_flx | set_flx,
                                  Gas::DiffusionUpdate<GEOM>, u0.get(), bdt);
      }

      // Apply gravity source term
      TaskID gravity_src = gas_coord_src | dust_coord_src | gas_diff_src;
      if (do_gravity) {
        gravity_src = tl.AddTask(gas_coord_src | dust_coord_src | gas_diff_src,
                                 Gravity::ExternalGravity<GEOM>, u0.get(), time, bdt);
      }

      // Apply self-gravity source term
      TaskID self_gravity_src = gravity_src;
      if (do_self_gravity) {
        self_gravity_src =
            tl.AddTask(gravity_src, SelfGravity::SelfGravity<GEOM>, u0.get(), time, bdt);
      }

      TaskID rt_src = self_gravity_src;
      // Note that radiation moments will handle this source term if active
      if (do_raytrace && !do_moment) {
        rt_src = tl.AddTask(self_gravity_src, Gas::DepositEnergy, u0.get(), bdt);
      }

      // Apply rotating frame source term
      TaskID rframe_src = rt_src;
      if (do_rotating_frame) {
        rframe_src =
            tl.AddTask(rt_src, RotatingFrame::RotatingFrameForce, u0.get(), time, bdt);
      }

      // Apply drag source term
      // NOTE(@pdmullen): RK integrated, operator split drag (RHS computed from U)
      TaskID drag_src = rframe_src;
      if (do_drag) {
        drag_src = tl.AddTask(rframe_src, Drag::DragSource<GEOM>, u0.get(), time, bdt);
      }

      // Apply cooling source term
      // NOTE(@pdmullen): RK integrated, operator split cooling (RHS computed from U)
      TaskID cooling_src = drag_src;
      if (do_cooling) {
        cooling_src =
            tl.AddTask(drag_src, Gas::Cooling::CoolingSource<GEOM>, u0.get(), time, bdt);
      }

      // Set auxillary fields
      auto set_aux =
          tl.AddTask(cooling_src, ArtemisDerived::SetAuxillaryFields<GEOM>, u0.get());

      // Set (remaining) fields to be communicated
      auto c2p = tl.AddTask(set_aux, PreCommFillDerived<MeshData<Real>>, u0.get());

      // Set boundary conditions (both physical and logical)
      auto bcs = parthenon::AddBoundaryExchangeTasks(c2p, tl, u0, pmesh->multilevel);

      // Sync fields
      auto p2c = tl.AddTask(TQ::local_sync, bcs, FillDerived<MeshData<Real>>, u0.get());

      // Advance nbody integrator
      TaskID nbadv = p2c;
      if (do_nbody) {
        nbadv = tl.AddTask(TQ::once_per_region, p2c, NBody::Advance, pmesh, time, stage,
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
  PARTHENON_INSTRUMENT
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

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
typedef Coordinates G;
typedef Mesh M;
typedef ParameterInput PI;
typedef ApplicationInput AI;
template ArtemisDriver<G::cartesian>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<G::cartesian>::Step();
template void ArtemisDriver<G::cartesian>::PreStepTasks();
template TaskCollection ArtemisDriver<G::cartesian>::StepTasks();
template TaskCollection ArtemisDriver<G::cartesian>::PostStepTasks();
template ArtemisDriver<G::cylindrical>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<G::cylindrical>::Step();
template void ArtemisDriver<G::cylindrical>::PreStepTasks();
template TaskCollection ArtemisDriver<G::cylindrical>::StepTasks();
template TaskCollection ArtemisDriver<G::cylindrical>::PostStepTasks();
template ArtemisDriver<G::spherical1D>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<G::spherical1D>::Step();
template void ArtemisDriver<G::spherical1D>::PreStepTasks();
template TaskCollection ArtemisDriver<G::spherical1D>::StepTasks();
template TaskCollection ArtemisDriver<G::spherical1D>::PostStepTasks();
template ArtemisDriver<G::spherical2D>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<G::spherical2D>::Step();
template void ArtemisDriver<G::spherical2D>::PreStepTasks();
template TaskCollection ArtemisDriver<G::spherical2D>::StepTasks();
template TaskCollection ArtemisDriver<G::spherical2D>::PostStepTasks();
template ArtemisDriver<G::spherical3D>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<G::spherical3D>::Step();
template void ArtemisDriver<G::spherical3D>::PreStepTasks();
template TaskCollection ArtemisDriver<G::spherical3D>::StepTasks();
template TaskCollection ArtemisDriver<G::spherical3D>::PostStepTasks();
template ArtemisDriver<G::axisymmetric>::ArtemisDriver(PI *p, AI *a, M *m, const bool r);
template TaskListStatus ArtemisDriver<G::axisymmetric>::Step();
template void ArtemisDriver<G::axisymmetric>::PreStepTasks();
template TaskCollection ArtemisDriver<G::axisymmetric>::StepTasks();
template TaskCollection ArtemisDriver<G::axisymmetric>::PostStepTasks();

} // namespace artemis
