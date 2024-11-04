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

//! \file nbody_extras.cpp
//! Additional functions to pass to REBOUND
//! NOTE(ADM): Rebound requires simple function pointers, we can't pass our normal params
//! params objects around, so we make use of namespace specific global variables here.

// REBOUND includes
extern "C" {
#define restrict __restrict__
#include "rebound.h"
}

// Artemis includes
#include "nbody/nbody.hpp"
#include "utils/artemis_utils.hpp"

namespace NBody {
//----------------------------------------------------------------------------------------
//! GR functions (PN expansions)
namespace PNForce {
Real get_alpha_1(const Real vi2, const Real vj2, const Real ndj, const Real vij,
                 const Real xi, const Real xj) {
  return -vi2 - 2.0 * vj2 + 4.0 * vij + 1.5 * SQR(ndj) + 5.0 * xi + 4.0 * xj;
}
Real get_beta_1(const Real ndi, const Real ndj) { return 4.0 * ndi - 3.0 * ndj; }
Real get_alpha_2(const Real vi2, const Real ndi, const Real vj2, const Real ndj,
                 const Real vij, const Real xi, const Real xj) {
  const Real ndj2 = SQR(ndj);
  const Real ndi2 = SQR(ndi);
  const Real f1 = -2.0 * SQR(vj2) + 4.0 * vj2 * vij - 2.0 * SQR(vij) +
                  ndj2 * (1.5 * vi2 + 4.5 * vj2 - 6.0 * vij - 15.0 / 8.0 * ndj2);
  const Real f2 = -15.0 / 4.0 * vi2 + 5.0 / 4.0 * vj2 - 2.5 * vij + 39.0 / 2.0 * ndi2 -
                  39.0 * ndi * ndj + 17.0 / 2.0 * ndj2;
  const Real f3 = 4.0 * vj2 - 8.0 * vij + 2.0 * ndi2 - 4.0 * ndi * ndj - 6.0 * ndj2;
  const Real f4 = -57.0 / 4.0 * SQR(xi) - 9.0 * SQR(xj) - 69.0 / 2.0 * xi * xj;
  return f1 + f2 * xi + f3 * xj + f4;
}
Real get_beta_2(const Real vi2, const Real ndi, const Real vj2, const Real ndj,
                const Real vij, const Real xi, const Real xj) {
  const Real ndj2 = SQR(ndj);
  const Real f1 = vi2 * ndj + 4.0 * vj2 * ndi - 5.0 * vj2 * ndj - 4.0 * vij * ndi +
                  4.0 * vij * ndj - 6.0 * ndi * ndj2 + 4.5 * ndj * ndj2;
  const Real f2 = -63.0 / 4.0 * ndi + 55.0 / 4.0 * ndj;
  const Real f3 = -2.0 * ndi - 2.0 * ndj;
  return f1 + f2 * xi + f3 * xj;
}
Real get_alpha_25(const Real ndv, const Real xi, const Real xj, const Real v2) {
  return 4.0 / 5.0 * xi * ndv * (3.0 * v2 - 6.0 * xi + 52.0 / 3.0 * xj);
}
Real get_beta_25(const Real ndv, const Real xi, const Real xj, const Real v2) {
  return 4.0 / 5.0 * xi * (-v2 + 2.0 * xi - 8.0 * xj);
}

//----------------------------------------------------------------------------------------
//! \fn  void PNForce::gr_force
//! \brief Supplies additional forcings to rebound particles due to GR forces (PN terms)
//! Follows https://arxiv.org/pdf/astro-ph/0602125.pdf
void gr_force(struct reb_simulation *const rsim) {
  using ArtemisUtils::VDot;
  struct reb_particle *const part = rsim->particles;
  const int N = rsim->N;
  for (int i = 0; i < N; i++) {
    const Real mi = part[i].m;
    const Real rgi = rsim->G * mi / SQR(RebAttrs::c);
    const Real vi[3] = {part[i].vx / RebAttrs::c, part[i].vy / RebAttrs::c,
                        part[i].vz / RebAttrs::c};
    const Real ri[3] = {part[i].x, part[i].y, part[i].z};
    const Real vi2 = VDot(vi, vi);
    for (int j = i + 1; j < N; j++) {
      // Force of particle j on particle i
      // F = Gmimj/r^3 {  \vec{r} alpha + \vec{v} beta }
      const Real mj = part[j].m;
      const Real rgj = rsim->G * mj / SQR(RebAttrs::c);
      const Real vj[3] = {part[j].vx / RebAttrs::c, part[j].vy / RebAttrs::c,
                          part[j].vz / RebAttrs::c};
      const Real rj[3] = {part[j].x, part[j].y, part[j].z};
      const Real vj2 = VDot(vj, vj);
      const Real dv[3] = {vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]};
      const Real dr[3] = {ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]};
      const Real dr2 = VDot(dr, dr);
      const Real ddr = std::sqrt(dr2);
      const Real dv2 = VDot(dv, dv);
      const Real xgi = rgi / ddr;
      const Real xgj = rgj / ddr;
      const Real mb = mi + mj;
      const Real xg = rsim->G * mb / SQR(RebAttrs::c) / ddr;

      // check that these 2 are bound
      const Real ebin = 0.5 * dv2 - xg;
      if (ebin < 0.0) {
        const Real n[3] = {dr[0] / ddr, dr[1] / ddr, dr[2] / ddr};
        const Real vij = VDot(vi, vj);
        const Real ndi = VDot(n, vi);
        const Real ndj = VDot(n, vj);
        const Real ndv = ndi - ndj;
        const Real ndv2 = SQR(ndv);

        Real alpha_i = 0.0;
        Real alpha_j = 0.0;
        Real beta_i = 0.0;
        Real beta_j = 0.0;
        // Force on i
        alpha_i += get_alpha_1(vi2, vj2, ndj, vij, xgi, xgj);
        alpha_j += get_alpha_1(vj2, vi2, ndi, vij, xgj, xgi);
        beta_i += get_beta_1(ndi, ndj);
        beta_j -= get_beta_1(ndj, ndi);
        if (RebAttrs::PN > 1) {
          if (RebAttrs::include_pn2) {
            alpha_i += get_alpha_2(vi2, ndi, vj2, ndj, vij, xgi, xgj);
            alpha_j += get_alpha_2(vj2, ndj, vi2, ndi, vij, xgj, xgi);
            beta_i += get_beta_2(vi2, ndi, vj2, ndj, vij, xgi, xgj);
            beta_j -= get_beta_2(vj2, ndj, vi2, ndi, vij, xgj, xgi);
          }
          if (RebAttrs::PN > 2) {
            alpha_i += get_alpha_25(ndv, xgi, xgj, dv2);
            alpha_j += get_alpha_25(ndv, xgj, xgi, dv2);
            beta_i += get_beta_25(ndv, xgi, xgj, dv2);
            beta_j += get_beta_25(ndv, xgj, xgi, dv2);
          }
        }

        part[i].ax += rsim->G * mj / dr2 * (n[0] * alpha_i + dv[0] * beta_i);
        part[i].ay += rsim->G * mj / dr2 * (n[1] * alpha_i + dv[1] * beta_i);
        part[i].az += rsim->G * mj / dr2 * (n[2] * alpha_i + dv[2] * beta_i);

        part[j].ax -= rsim->G * mi / dr2 * (n[0] * alpha_j + dv[0] * beta_j);
        part[j].ay -= rsim->G * mi / dr2 * (n[1] * alpha_j + dv[1] * beta_j);
        part[j].az -= rsim->G * mi / dr2 * (n[2] * alpha_j + dv[2] * beta_j);
      }
    }
  }
}
} // namespace PNForce

//----------------------------------------------------------------------------------------
//! \fn  void NBody::reb_extra_forces
//! \brief Supplies additional forcings to rebound particles
void reb_extra_forces(struct reb_simulation *r) {
  if (RebAttrs::PN > 0) PNForce::gr_force(r);
}

//----------------------------------------------------------------------------------------
//! \fn  int NBody::collision_resolution
//! \brief Collision kernel
int collision_resolution(struct reb_simulation *const r, struct reb_collision c) {
  struct reb_particle p1 = r->particles[c.p1];
  struct reb_particle p2 = r->particles[c.p2];

  const Real mb = p1.m + p2.m;
  const Real dr = std::sqrt(SQR(p1.x - p2.x) + SQR(p1.y - p2.y) + SQR(p1.z - p2.z));
  const Real dv2 = SQR(p1.vx - p2.vx) + SQR(p1.vy - p2.vy) + SQR(p1.vz - p2.vz);
  const Real eb = 0.5 * dv2 - mb / (dr + Fuzz<Real>());
  if ((RebAttrs::merge_on_collision) || (eb <= 0.0)) {
    // they are bound, so merge them using rebound's function
    return reb_collision_resolve_merge(r, c);
  }

  // They are within their "radii" but we do not want to merge them
  return 0;
}

} // namespace NBody
