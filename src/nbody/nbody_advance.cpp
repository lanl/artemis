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

// Parthenon includes
#include <globals.hpp>

// Artemis includes
#include "artemis.hpp"
#include "nbody/nbody.hpp"
#include "nbody/nbody_utils.hpp"

namespace NBody {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus NBody::Advance
//! \brief Advances NBody simulation and syncs with Artemis integration
//! The integration when coupled to gas is not trivial when
//! one considers the integration stages of the Artemis integrator.
//!
//! The crucial point is to respect that intermediate stages of the integrator
//! are used to provide finer estimates to the RHS gas force. We thus treat intermediate
//! stages as providing better estimates for the particle positions at the end of the
//! hydro time step.
//!
//! Each call to this function should thus advance the particle positions to
//! "approximate" values based on the current best estimate for the hydro forces. This
//! means we need to keep a second rebound simulation that starts from the same initial
//! state as the "real" rebound simulation. This function then advances the second
//! simulation.
//!
//! At the end of the final integration stage, the two simulations are synced.
//!
//! The second simulation is only required when the particles feel the gas. In that case
//! we approximate the force by applying a kick to the particles velocities based on the
//! current gas force. We then call the rebound integrator with the modified velocities
//! and current positions. Subsequent calls to this function provide better kick forces
//! which we add to an average kick force.
//!
//! If particles do not feel the gas, then the kick step is omitted and after the first
//! stage rebound has provided the positions of the particles at the end of the hydro
//! time step, and this function does not need to be called in subsequent stages (if they
//! exist)
//!
//!
//! An example of a full RK2 step that takes (Un,xn,vn)->(U',x',v')
//! where U is the gas state, and (x,v) are the particle positions & velocities
//!
//!   stage 1:
//!      hydro step
//!        U* = Un - dt*F(Un,xn)
//!      nbody step
//!        v* = vn + dt*F(Un,xn)
//!        x*,v^ = reb_integrate( xn, v*)
//!
//!   stage 2:
//!      hydro step
//!        U' = 0.5*( Un + U*) - 0.5 dt*( F(U*,x*))
//!           = Un - 0.5 dt*( F(Un,xn) + F(U*,x*))
//!      nbody step
//!        v** = 0.5*(vn + v*) + 0.5*dt*( F(U*,x*))
//!            = vn  + .5*dt*( F(Un,xn) + F(U*,x*))
//!        x',v'  = reb_integrate(xn, v**)
//!
//!
//!   RK1 = stage 1 integrates to end
//!
//!   RK2 = stage 1 integrates to end
//!         stage 2 doesn't integrate
//!
//!   RK3 = stage 1 integrates to end
//!         stage 2 integrates to half
//!         stage 3 integrates to end
//!
//!   VL2 = stage 1 integrates to half
//!         stage 2 integrates to end
TaskStatus Advance(Mesh *pm, const Real time, const int stage,
                   const parthenon::LowStorageIntegrator *nbody_integ) {
  auto &nbody_pkg = pm->packages.Get("nbody");
  auto particle_id = nbody_pkg->Param<std::vector<int>>("particle_id");
  auto particles = nbody_pkg->Param<ParArray1D<Particle>>("particles");
  auto npart = nbody_pkg->Param<int>("npart");
  auto mscale = nbody_pkg->Param<Real>("mscale");
  auto pforce = nbody_pkg->Param<ParArray2D<Real>>("particle_force");
  auto pforce_tot = nbody_pkg->Param<ParArray2D<Real>>("particle_force_tot");
  auto pforce_step = nbody_pkg->Param<ParArray2D<Real>>("particle_force_step");
  auto reb_sim = nbody_pkg->Param<RebSim>("reb_sim");

  // Extract integrators/weights
  // NOTE(PDM): reb_integ is the rebound integrator pushing particles.  nbody_integ is the
  // integrator coupling the NBody package to the remainder of our unsplit physics
  auto reb_integ = nbody_pkg->Param<std::string>("integrator");
  const int nstages = nbody_integ->nstages;
  const Real dt = nbody_integ->dt;
  const Real dt_stage = nbody_integ->beta[stage - 1] * dt;
  const Real gam0 = nbody_integ->gam0[stage - 1];
  const Real gam1 = nbody_integ->gam1[stage - 1];

  // Extract rotating frame parameters (if applicable)
  auto &artemis_pkg = pm->packages.Get("artemis");
  auto frame_corr = nbody_pkg->Param<bool>("frame_correction");
  Real omegaf = 0.0;
  if ((artemis_pkg->Param<bool>("do_rotating_frame")) && (frame_corr)) {
    auto &rframe_pkg = pm->packages.Get("rotating_frame");
    omegaf = rframe_pkg->Param<Real>("omega");
  }

  // Extract mirrors
  auto pforce_h = pforce.GetHostMirrorAndCopy();
  auto pforce_step_h = pforce_step.GetHostMirrorAndCopy();
  auto pforce_tot_h = pforce_tot.GetHostMirrorAndCopy();
  auto particles_h = particles.GetHostMirrorAndCopy();

  // Reduce the forces from this timestep
#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, pforce_h.data(), pforce_h.size(), MPI_PARTHENON_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(pforce_h.data(), pforce_h.data(), pforce_h.size(), MPI_PARTHENON_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  RebSim r_sim;
  if (parthenon::Globals::my_rank == 0) {
    // Advance the simulation.  If this is the final stage, advance the master simulation.
    // If not, advance a copy of the master.
    int sid = disable_stderr();
    if (stage < nstages) {
      r_sim.copy(reb_sim);
    } else {
      r_sim = reb_sim;
    }
    SetReboundPtrs(r_sim);
    enable_stderr(sid);

    // Kick the particles based on the gas forces
    for (int n = 0; n < npart; n++) {
      // Update the current estimate for this steps force
      for (int i = 0; i < 7; i++) {
        pforce_step_h(n, i) = gam0 * pforce_step_h(n, i) + gam1 * pforce_h(n, i);
      }

      const int id = particle_id[n];
      if (particles_h(id).alive && particles_h(id).live &&
          (time >= particles_h(id).live_after) && (reb_integ != "none")) {
        // Apply the gravitational force
        // NOTE(ADM): pforce already contains factor of dt
        struct reb_particle *pl = reb_simulation_particle_by_hash(r_sim, n + 1);
        if (pl != nullptr) {
          const Real mp = pl->m;
          pl->vx += mscale * dt_stage * pforce_step_h(n, 1) / mp;
          pl->vy += mscale * dt_stage * pforce_step_h(n, 2) / mp;
          pl->vz += mscale * dt_stage * pforce_step_h(n, 3) / mp;
        }
      }
    }

    // Integrate to the end of the stage. If the user set reb_integ = none, do nothing
    if (reb_integ != "none") {
      reb_simulation_integrate(r_sim, time + dt_stage);
    }

    // If we are in a rotating frame, correct the rebound simulation
    if (omegaf != 0.0) {
      struct reb_vec3d axis = {.x = 0.0, .y = 0.0, .z = 1.0};
      struct reb_rotation r1 = reb_rotation_init_angle_axis(-omegaf * dt_stage, axis);
      reb_simulation_irotate(r_sim, r1);
    }
  }

  // Update our copies of the particle positions and velocities
  SyncWithRebound(r_sim, particle_id, particles);

  // Reset pforce for next step
  for (int n = 0; n < npart; n++) {
    for (int i = 0; i < 7; i++) {
      pforce_h(n, i) = 0.0;
    }
  }
  pforce.DeepCopy(pforce_h);

  // On the final stage, add the final force (i.e., the one that actually updated the
  // particles) for this step to the running total and reset the force for the next step
  if (parthenon::Globals::my_rank == 0) {
    if (stage == nstages) {
      for (int n = 0; n < npart; n++) {
        for (int i = 0; i < 7; i++) {
          pforce_tot_h(n, i) += dt_stage * pforce_step_h(n, i);
          pforce_step_h(n, i) = 0.0;
        }
      }
      pforce_tot.DeepCopy(pforce_tot_h);
    }
    pforce_step.DeepCopy(pforce_step_h);
  }

  return TaskStatus::complete;
}

} // namespace NBody
