//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#ifndef NBODY_NBODY_UTILS_HPP_
#define NBODY_NBODY_UTILS_HPP_

// This file was created in part by one of OpenAI's generative AI models

// C++/C includes
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <unistd.h> // for dup and dup2 on Unix-like systems

// REBOUND includes
extern "C" {
#define restrict __restrict__
#include "rebound.h"
}

// Artemis includes
#include "artemis.hpp"
#include "nbody/nbody.hpp"

namespace NBody {

extern void reb_extra_forces(struct reb_simulation *rsim);
extern int collision_resolution(struct reb_simulation *const r, struct reb_collision c);

class RebSim {
 public:
  // Constructor to initialize the shared_ptr with a custom deleter
  // RebSim() : reb_sim(nullptr, reb_sim_deleter) {}
  RebSim() : reb_sim(reb_simulation_create(), reb_sim_deleter) {}

  // Constructor that accepts a preexisting pointer
  RebSim(const RebSim &existing_reb_sim)
      : reb_sim(reb_simulation_copy(existing_reb_sim.get()), reb_sim_deleter) {}

  // Constructor that accepts a rebound filename
  RebSim(std::string reb_filename)
      : reb_sim(reb_simulation_create_from_file(reb_filename.data(), -1)) {}

  // Function to get a reference to the shared_ptr
  struct reb_simulation *get() const {
    PARTHENON_REQUIRE(reb_sim, "Internal pointer is null!");
    return reb_sim.get();
  }

  void set(struct reb_simulation *ptr) {
    PARTHENON_REQUIRE(ptr != nullptr, "Passing a null pointer!");
    reb_sim.reset(ptr);
  }

 private:
  // Shared pointer which will use a custom deleter
  std::shared_ptr<struct reb_simulation> reb_sim;

  // Shared custom deleter
  static void reb_sim_deleter(struct reb_simulation *p) {
    if (p != nullptr) {
      reb_simulation_free(p);
    }
  }
};

//----------------------------------------------------------------------------------------
//! \fn  void NBody::SetReboundPtrs
//! \brief
static void SetReboundPtrs(RebSim &reb_sim) {
  reb_sim.get()->collision_resolve = collision_resolution;
  if (RebAttrs::extras) reb_sim.get()->additional_forces = reb_extra_forces;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::PrintSystem
//! \brief
static void PrintSystem(const int npart, ParArray1D<Particle> particles) {
  if (parthenon::Globals::my_rank == 0) {
    for (int n = 0; n < npart; n++) {
      auto &part = particles(n);
      std::cout << "===============\n";
      std::cout << ((part.alive) ? "Alive" : "Dead") << "\n";
      std::cout << "id: " << part.id << "\n"
                << "Mass: " << part.GM << "\n"
                << "Radius: " << part.radius << "\n"
                << "Soft: " << part.rs << "\n"
                << "Sink: " << part.racc << "\n"
                << "gamma: " << part.gamma << "\n"
                << "beta: " << part.beta << "\n"
                << "couple: " << part.couple << "\n"
                << "target_rad: " << part.target_rad << "\n"
                << "live: " << part.live << "\n"
                << "live_after: " << part.live_after << "\n"
                << "x=(" << part.pos[0] << "," << part.pos[1] << "," << part.pos[2]
                << ")\n"
                << "v=(" << part.vel[0] << "," << part.vel[1] << "," << part.vel[2]
                << ")\n"
                << "===============\n"
                << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::SyncWithRebound
//! \brief Copy the positions and velocities from the rebound sim to the Particles list
static void SyncWithRebound(RebSim &r_sim, std::vector<int> particle_id,
                            ParArray1D<Particle> particles) {
  const auto npart = particle_id.size();
  auto particles_h = particles.GetHostMirrorAndCopy();

  std::vector<Real> data(npart * 7);
  std::vector<int> alive(npart);
  if (parthenon::Globals::my_rank == 0) {
    for (int n = 0; n < npart; n++) {
      struct reb_particle *p = reb_simulation_particle_by_hash(r_sim.get(), n + 1);
      if (p == nullptr) {
        alive[n] = 0;
      } else {
        alive[n] = 1;
        data[n * 7 + 0] = p->x;
        data[n * 7 + 1] = p->y;
        data[n * 7 + 2] = p->z;
        data[n * 7 + 3] = p->vx;
        data[n * 7 + 4] = p->vy;
        data[n * 7 + 5] = p->vz;
        data[n * 7 + 6] = p->m;
      }
    }
  }

#ifdef MPI_PARALLEL
  // Scatter new positions to other procs
  MPI_Bcast(data.data(), data.size(), MPI_PARTHENON_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(alive.data(), alive.size(), MPI_INT, 0, MPI_COMM_WORLD);
#endif

  for (int n = 0; n < npart; n++) {
    const int id = particle_id[n];
    if (alive[n] == 0) {
      particles_h(id).alive = 0;
      particles_h(id).couple = 0;
      particles_h(id).live = 0;
    } else {
      particles_h(id).pos[0] = data[n * 7 + 0];
      particles_h(id).pos[1] = data[n * 7 + 1];
      particles_h(id).pos[2] = data[n * 7 + 2];
      particles_h(id).vel[0] = data[n * 7 + 3];
      particles_h(id).vel[1] = data[n * 7 + 4];
      particles_h(id).vel[2] = data[n * 7 + 5];
      particles_h(id).GM = data[n * 7 + 6];
    }
  }

  particles.DeepCopy(particles_h);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  int NBody::disable_stderr
//! \brief
static int disable_stderr(void) {
  // Flush the stderr stream before redirecting to avoid mixing output
  std::fflush(stderr);

  // Save the current file descriptor of stderr
  int stderr_save_fd = dup(fileno(stderr));

  // Open /dev/null for writing
  int dev_null_fd = open("/dev/null", O_WRONLY);

  // Redirect stderr to /dev/null
  dup2(dev_null_fd, fileno(stderr));

  // Close the original /dev/null file descriptor as it's no longer needed
  close(dev_null_fd);
  return stderr_save_fd;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::enable_stderr
//! \brief
static void enable_stderr(int stderr_save_fd) {
  if (stderr_save_fd != -1) {
    // Restore the original stderr file descriptor
    dup2(stderr_save_fd, fileno(stderr));

    // Close the saved file descriptor as it's no longer needed
    close(stderr_save_fd);
  }
}

} // namespace NBody

#endif
