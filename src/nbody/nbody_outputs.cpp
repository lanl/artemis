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

// Rebound includes
extern "C" {
#define restrict __restrict__
#include "rebound.h"
}

// Artemis includes
#include "artemis.hpp"
#include "nbody.hpp"
#include "nbody_utils.hpp"

namespace NBody {
//----------------------------------------------------------------------------------------
//! \fn  void NBody::Outputs
//! \brief
void Outputs(parthenon::Mesh *pm, const Real time) {
  // Return immediately if nbody outputs are disabled
  auto nbody = pm->packages.Get("nbody").get();
  if (nbody->Param<bool>("disable_outputs")) return;

  // Check if we are ready to output, else return
  auto *tnext = nbody->MutableParam<Real>("tnext");
  if ((time < *tnext) || (*tnext == Big<Real>())) return;
  *tnext += nbody->Param<Real>("dt_output");

  // Extract nbody parameters
  auto npart = nbody->Param<int>("npart");
  auto particle_id = nbody->Param<std::vector<int>>("particle_id");
  auto *output_count = nbody->MutableParam<int>("output_count");
  auto r_sim = nbody->Param<RebSim>("reb_sim");
  printf("%s:%i v: %i\n", __FILE__, __LINE__, r_sim.get()->simulationarchive_version);
  auto base = nbody->Param<std::string>("output_base");
  auto particles = nbody->Param<ParArray1D<Particle>>("particles");
  auto pforce_tot = nbody->Param<ParArray2D<Real>>("particle_force_tot");
  auto orbit_output_count_d = nbody->Param<ParArray2D<int>>("orbit_output_count");

  // Host Mirrors
  auto particles_h = particles.GetHostMirrorAndCopy();
  auto pforce_tot_h = pforce_tot.GetHostMirrorAndCopy();
  auto orbit_output_count = orbit_output_count_d.GetHostMirrorAndCopy();

  // base.reb
  if (parthenon::Globals::my_rank == 0) {
    std::string fhst;
    fhst.assign(base);
    fhst.append(".reb");

    FILE *pfile;
    if ((pfile = std::fopen(fhst.c_str(), (*output_count == 0) ? "w" : "a")) == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function NBody::Outputs" << std::endl
          << "Output file '" << fhst << "' could not be opened";
      PARTHENON_FAIL(msg);
    }

    if (*output_count == 0) {
      // print the header
      int iout = 1;
      std::fprintf(pfile, "# NBody data N = %d\n", npart);
      std::fprintf(pfile, "# [%d]=time    ", iout++);
      std::fprintf(pfile, "[%d]=hash    ", iout++);
      std::fprintf(pfile, "[%d]=active    ", iout++);
      std::fprintf(pfile, "[%d]=mass    ", iout++);
      std::fprintf(pfile, "[%d]=x    ", iout++);
      std::fprintf(pfile, "[%d]=y    ", iout++);
      std::fprintf(pfile, "[%d]=z    ", iout++);
      std::fprintf(pfile, "[%d]=vx    ", iout++);
      std::fprintf(pfile, "[%d]=vy    ", iout++);
      std::fprintf(pfile, "[%d]=vz    ", iout++);
      std::fprintf(pfile, "[%d]=dm    ", iout++);
      std::fprintf(pfile, "[%d]=dmx_g    ", iout++);
      std::fprintf(pfile, "[%d]=dmy_g    ", iout++);
      std::fprintf(pfile, "[%d]=dmz_g    ", iout++);
      std::fprintf(pfile, "[%d]=dmx_a    ", iout++);
      std::fprintf(pfile, "[%d]=dmy_a    ", iout++);
      std::fprintf(pfile, "[%d]=dmz_a    ", iout++);
      std::fprintf(pfile, "\n");
    }

    // print the data
    for (int i = 0; i < npart; i++) {
      // info
      std::fprintf(pfile, "%.8e\t", r_sim.get()->t);
      std::fprintf(pfile, "%d\t", i + 1);
      std::fprintf(pfile, "%d\t", particles_h(i).alive);
      // data
      std::fprintf(pfile, "%.8e\t", particles_h(i).GM);
      std::fprintf(pfile, "%.8e\t", particles_h(i).pos[0]);
      std::fprintf(pfile, "%.8e\t", particles_h(i).pos[1]);
      std::fprintf(pfile, "%.8e\t", particles_h(i).pos[2]);
      std::fprintf(pfile, "%.8e\t", particles_h(i).vel[0]);
      std::fprintf(pfile, "%.8e\t", particles_h(i).vel[1]);
      std::fprintf(pfile, "%.8e\t", particles_h(i).vel[2]);

      // forces
      for (int j = 0; j < 7; j++) {
        std::fprintf(pfile, "%.8e\t", pforce_tot_h(i, j));
      }
      std::fprintf(pfile, "\n");
    }

    std::fclose(pfile);
  }
  *output_count += 1;

  // base.orb
  for (int i = 0; i < npart; i++) {
    if (particles_h(i).alive) {
      for (int j = i + 1; j < npart; j++) {
        if (particles_h(j).alive) {
          // Check if i-j is a bound binary
          Real dx2 = 0.0;
          Real dv2 = 0.0;
          for (int d = 0; d < 3; d++) {
            dx2 += SQR(particles_h(i).pos[d] - particles_h(j).pos[d]);
            dv2 += SQR(particles_h(i).vel[d] - particles_h(j).vel[d]);
          }
          dx2 = std::sqrt(dx2);
          const Real m1 = particles_h(i).GM;
          const Real m2 = particles_h(j).GM;
          const Real mb = m1 + m2;
          if ((0.5 * dv2 - mb / (dx2 + Fuzz<Real>())) < 0.0) {
            // This pair is bound.
            if (parthenon::Globals::my_rank == 0) {
              std::string fname;
              fname.assign(base);
              fname.append(".");
              fname.append(std::to_string(i));
              fname.append("_");
              fname.append(std::to_string(j));
              fname.append(".orb");
              FILE *pfile;
              const bool first = (orbit_output_count(i, j) == 0);
              std::stringstream msg;
              if ((pfile = std::fopen(fname.c_str(), (first) ? "w" : "a")) == nullptr) {
                msg << "### FATAL ERROR in function NBody::Outputs" << std::endl
                    << "Output file '" << fname << "' could not be opened";
                PARTHENON_FAIL(msg);
              }

              // If this is the first output, write header
              if (first) {
                int iout = 1;
                std::fprintf(pfile, "# NBody Orbit data\n"); // descriptor is first line
                std::fprintf(pfile, "# [%d]=time     ", iout++);
                std::fprintf(pfile, "[%d]=mb     ", iout++);
                std::fprintf(pfile, "[%d]=xc     ", iout++);
                std::fprintf(pfile, "[%d]=yc     ", iout++);
                std::fprintf(pfile, "[%d]=zc     ", iout++);
                std::fprintf(pfile, "[%d]=xb     ", iout++);
                std::fprintf(pfile, "[%d]=yb     ", iout++);
                std::fprintf(pfile, "[%d]=zb     ", iout++);
                std::fprintf(pfile, "[%d]=vxc     ", iout++);
                std::fprintf(pfile, "[%d]=vyc     ", iout++);
                std::fprintf(pfile, "[%d]=vzc     ", iout++);
                std::fprintf(pfile, "[%d]=vxb     ", iout++);
                std::fprintf(pfile, "[%d]=vyb     ", iout++);
                std::fprintf(pfile, "[%d]=vzb     ", iout++);
                std::fprintf(pfile, "[%d]=qb    ", iout++);
                std::fprintf(pfile, "[%d]=nb   ", iout++);
                std::fprintf(pfile, "[%d]=ab   ", iout++);
                std::fprintf(pfile, "[%d]=eb     ", iout++);
                std::fprintf(pfile, "[%d]=Ib     ", iout++);
                std::fprintf(pfile, "[%d]=o     ", iout++);
                std::fprintf(pfile, "[%d]=O     ", iout++);
                std::fprintf(pfile, "[%d]=pomega    ", iout++);
                std::fprintf(pfile, "[%d]=f   ", iout++);
                std::fprintf(pfile, "[%d]=h    ", iout++);
                std::fprintf(pfile, "[%d]=ex    ", iout++);
                std::fprintf(pfile, "[%d]=ey    ", iout++);
                std::fprintf(pfile, "[%d]=ix   ", iout++);
                std::fprintf(pfile, "[%d]=iy    ", iout++);
                std::fprintf(pfile, "[%d]=dm    ", iout++);
                std::fprintf(pfile, "[%d]=Fx_grav_com    ", iout++);
                std::fprintf(pfile, "[%d]=Fy_grav_com    ", iout++);
                std::fprintf(pfile, "[%d]=Fz_grav_com    ", iout++);
                std::fprintf(pfile, "[%d]=Fx_acc_com    ", iout++);
                std::fprintf(pfile, "[%d]=Fy_acc_com    ", iout++);
                std::fprintf(pfile, "[%d]=Fz_acc_com    ", iout++);
                std::fprintf(pfile, "[%d]=Fx_grav_bin    ", iout++);
                std::fprintf(pfile, "[%d]=Fy_grav_bin    ", iout++);
                std::fprintf(pfile, "[%d]=Fz_grav_bin    ", iout++);
                std::fprintf(pfile, "[%d]=Fx_acc_bin    ", iout++);
                std::fprintf(pfile, "[%d]=Fy_acc_bin    ", iout++);
                std::fprintf(pfile, "[%d]=Fz_acc_bin    ", iout++);
                std::fprintf(pfile, "\n"); // terminate line
              }

              // compute the orbit
              const Real qb = (m1 >= m2) ? m2 / m1 : m1 / m2;
              const Real mu1 = m1 / mb;
              const Real mu2 = m2 / mb;

              const int ip = (m1 >= m2) ? i : j;
              const int is = (m1 >= m2) ? j : i;
              struct reb_particle *primary =
                  reb_simulation_particle_by_hash(r_sim.get(), ip + 1);
              struct reb_particle *secondary =
                  reb_simulation_particle_by_hash(r_sim.get(), is + 1);
              struct reb_orbit o =
                  reb_orbit_from_particle(r_sim.get()->G, *secondary, *primary);

              std::fprintf(pfile, "%.8e\t", r_sim.get()->t);
              std::fprintf(pfile, "%.8e\t", mb);
              std::fprintf(pfile, "%.8e\t", mu1 * primary->x + mu2 * secondary->x);
              std::fprintf(pfile, "%.8e\t", mu1 * primary->y + mu2 * secondary->y);
              std::fprintf(pfile, "%.8e\t", mu1 * primary->z + mu2 * secondary->z);
              std::fprintf(pfile, "%.8e\t", secondary->x + primary->x);
              std::fprintf(pfile, "%.8e\t", secondary->y + primary->y);
              std::fprintf(pfile, "%.8e\t", secondary->z + primary->z);
              std::fprintf(pfile, "%.8e\t", mu1 * primary->vx + mu2 * secondary->vx);
              std::fprintf(pfile, "%.8e\t", mu1 * primary->vy + mu2 * secondary->vy);
              std::fprintf(pfile, "%.8e\t", mu1 * primary->vz + mu2 * secondary->vz);
              std::fprintf(pfile, "%.8e\t", secondary->vx + primary->vx);
              std::fprintf(pfile, "%.8e\t", secondary->vy + primary->vy);
              std::fprintf(pfile, "%.8e\t", secondary->vz + primary->vz);
              std::fprintf(pfile, "%.8e\t", qb);
              std::fprintf(pfile, "%.8e\t", o.n);
              std::fprintf(pfile, "%.8e\t", o.a);
              std::fprintf(pfile, "%.8e\t", o.e);
              std::fprintf(pfile, "%.8e\t", o.inc);
              std::fprintf(pfile, "%.8e\t", o.omega);
              std::fprintf(pfile, "%.8e\t", o.Omega);
              std::fprintf(pfile, "%.8e\t", o.pomega);
              std::fprintf(pfile, "%.8e\t", o.f);
              std::fprintf(pfile, "%.8e\t", o.h);
              std::fprintf(pfile, "%.8e\t", o.pal_k);
              std::fprintf(pfile, "%.8e\t", o.pal_h);
              std::fprintf(pfile, "%.8e\t", o.pal_ix);
              std::fprintf(pfile, "%.8e\t", o.pal_iy);
              std::fprintf(pfile, "%.8e\t", pforce_tot_h(ip, 0) + pforce_tot_h(is, 0));
              for (int d = 0; d < 3; d++) {
                std::fprintf(pfile, "%.8e\t",
                             pforce_tot_h(ip, 1 + d) + pforce_tot_h(is, 1 + d));
              }
              for (int d = 0; d < 3; d++) {
                std::fprintf(pfile, "%.8e\t",
                             pforce_tot_h(ip, 4 + d) + pforce_tot_h(is, 4 + d));
              }
              for (int d = 0; d < 3; d++) {
                std::fprintf(pfile, "%.8e\t",
                             mu1 * pforce_tot_h(is, 1 + d) -
                                 mu2 * pforce_tot_h(ip, 1 + d));
              }
              for (int d = 0; d < 3; d++) {
                std::fprintf(pfile, "%.8e\t",
                             mu1 * pforce_tot_h(is, 4 + d) -
                                 mu2 * pforce_tot_h(ip, 4 + d));
              }
              std::fprintf(pfile, "\n"); // terminate line
              std::fclose(pfile);
            }
            orbit_output_count(i, j) += 1;
          }
        }
      }
    }
  }

  // Reset pforce_tot
  for (int n = 0; n < npart; n++) {
    for (int i = 0; i < 7; i++) {
      pforce_tot_h(n, i) = 0.;
    }
  }

  // Sync
  orbit_output_count_d.DeepCopy(orbit_output_count);
  pforce_tot.DeepCopy(pforce_tot_h);
}

} // namespace NBody
