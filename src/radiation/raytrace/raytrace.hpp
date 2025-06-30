//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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
#ifndef RADIATION_RAYTRACE_RAYTRACE_HPP_
#define RADIATION_RAYTRACE_RAYTRACE_HPP_

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/units.hpp"

namespace RT {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants);

TaskListStatus RaytraceDriver(Mesh *pmesh);

struct ParticleWeights {
  Real dx2 = 1.0;
  Real dx3 = 1.0;
  int np2 = 1;
  int np3 = 1;
  Real wght = 1.;
  Real x2min, x2max, x3min, x3max;
  ParticleWeights(const Real x2min, const Real x2max, const Real x3min, const Real x3max)
      : x2min(x2min), x2max(x2max), x3min(x3min), x3max(x3max) {};
};

template <Coordinates GEOM>
TaskStatus PushParticlesImpl(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &rt_pkg = pm->packages.Get("raytrace");
  const auto efloor = rt_pkg->template Param<Real>("efloor");
  // Create SparsePack
  static auto desc =
      MakePackDescriptor<rad::opac::cross_section, gas::src::energy>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  // Create SwarmPacks
  static auto pdesc_r =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              rad::part::flux>("star");
  static auto pdesc_i = MakeSwarmPackDescriptor<rad::part::ijk>("star");
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Indexing and dimensionality
  const int ndim = pm->ndim;
  const bool multi_d = (ndim >= 2);
  const bool three_d = (ndim == 3);
  const int &nblocks = vmesh.GetNBlocks();
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();
  const auto ib = md->GetBoundsI(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "TransportPhotons", DevExecSpace(), 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // Set ijk coorectly?
          int &i = ppack_i(b, rad::part::ijk(0), n);
          const int j = ppack_i(b, rad::part::ijk(1), n);
          const int k = ppack_i(b, rad::part::ijk(2), n);
          geometry::Coords<GEOM> coords0(vmesh.GetCoordinates(b), k, j, ib.s);
          Real &ee = ppack_r(b, rad::part::flux(), n);
          Real &xp = ppack_r(b, swarm_position::x(), n);
          i = ib.s +
              static_cast<int>(std::floor((xp - coords0.bnds.x1[0]) /
                                          (coords0.bnds.x1[1] - coords0.bnds.x1[0])));

          while ((i <= ib.e) && (ee > 0.0)) {
            geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);

            // Deposit energy for this cell and decrement the photon energy
            const auto dx = coords.GetCellWidths();
            const Real sigma = vmesh(b, rad::opac::cross_section(), k, j, i);
            const Real dtau = dx[0] * sigma;
            const Real efac = (dtau > 100.) ? 0.0 : std::exp(-dtau);
            const Real reduc = (dtau <= 1e-4) ? dtau - 0.5 * SQR(dtau) : (1. - efac);
            Real dE = ee * reduc;
            Real Enew = ee * efac;

            if (Enew < efloor) Enew = 0.0;

            Kokkos::atomic_add(&(vmesh(b, gas::src::energy(), k, j, i)),
                               dE / coords.Volume());
            ee = Enew;
            // move the particle to the next face;
            i += 1;
            xp = coords.bnds.x1[1];
          }
        }
      });
  return TaskStatus::complete;
}

template <Coordinates GEOM>
TaskStatus SourceParticlesImpl(MeshData<Real> *md, const ParticleWeights &pwght) {
  // Create SwarmPacks

  // Create pack
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  static auto desc = MakePackDescriptor<gas::src::energy>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();

  auto &rt_pkg = pm->packages.Get("raytrace");
  const auto x1min = rt_pkg->Param<Real>("x1min");

  // auto newParticlesContext = swarm->AddEmptyParticles(pwght.np2 * pwght.np3);

  // Reset energy exchange
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SourceParticles::Reset", parthenon::DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        for (int n = 0; n < vmesh.GetSize(b, gas::src::energy()); n++) {
          vmesh(b, gas::src::energy(n), k, j, i) = 0.0;
        }
      });

  ParArray1D<int> new_parts("New particles", nblocks);
  ParArray1D<int> new_part_per_cell("New particles per cell", nblocks);
  ParArray1D<NewParticlesContext> new_contexts("New contexts", nblocks);
  auto new_contexts_h = new_contexts.GetHostMirror();
  auto new_parts_h = new_parts.GetHostMirror();
  auto new_part_per_cell_h = new_part_per_cell.GetHostMirror();
  int nparticles = 0;
  for (int b = 0; b < nblocks; ++b) {
    // Is this block on the x1min boundary
    const auto &pmb = md->GetBlockData(b)->GetBlockPointer();
    const bool on_boundary = pmb->IsPhysicalBoundary(parthenon::BoundaryFace::inner_x1);
    if (on_boundary) {
      auto &particles = md->GetSwarmData(b)->Get("star");
      const int lvl_fac = 1 << pmb->loc.level();
      int nbx2 = pmb->block_size.nx_[1];
      if (nbx2 > 1) nbx2 *= lvl_fac;
      int nbx3 = pmb->block_size.nx_[2];
      if (nbx3 > 1) nbx3 *= lvl_fac;
      new_parts_h(b) = nbx2 * nbx3;

      //%%%%%%%%%%%%%%%%%%%%%
      //  FIX THIS
      //%%%%%%%%%%%%%%%%%%%%%
      new_part_per_cell_h(b) = lvl_fac;

      new_contexts_h(b) = particles->AddEmptyParticles(new_parts_h(b));
      nparticles += new_parts_h(b);
    }
  }
  new_contexts.DeepCopy(new_contexts_h);
  new_parts.DeepCopy(new_parts_h);
  new_part_per_cell.DeepCopy(new_part_per_cell_h);
  Kokkos::fence();

  static auto pdesc_r =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              rad::part::flux>("star");
  static auto pdesc_i = MakeSwarmPackDescriptor<rad::part::ijk>("star");
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Initialize particles
  //
  const int three_d = pm->ndim == 3;
  const int multi_d = pm->ndim >= 2;
  const auto luminosity = rt_pkg->Param<Real>("luminosity");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SourcePhotons2", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j) {
        const int tot = new_parts(b);
        if (tot > 0) {
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, ib.s);
          const int nbx2 = (jb.e - jb.s) + 1;
          const int nbx3 = (kb.e - kb.s) + 1;
          const int nper_cell = (multi_d) ? new_part_per_cell(b) : 1;
          const int ntot_cell = (three_d) ? SQR(nper_cell) : ((multi_d) ? nper_cell : 1);
          const int offset = (j - jb.s) * ntot_cell + (k - kb.s) * ntot_cell * nbx2;
          for (int kp = 0; kp < nper_cell; kp++) {
            for (int jp = 0; jp < nper_cell; jp++) {
              const int np = offset + jp + nper_cell * kp;
              const int &n = new_contexts(b).GetNewParticleIndex(np);
              ppack_i(b, rad::part::ijk(0), n) = ib.s;
              ppack_i(b, rad::part::ijk(1), n) = j;
              ppack_i(b, rad::part::ijk(2), n) = k;
              const Real x = coords.bnds.x1[0];
              const Real ym = coords.bnds.x2[0] + jp * pwght.dx2;
              const Real yp = ym + pwght.dx2;
              const Real dcos = (multi_d) ? (std::cos(ym) - std::cos(yp)) : 2.;
              const Real z = coords.bnds.x3[0] + (kp + 0.5) * pwght.dx3;
              const Real dphi = (three_d) ? pwght.dx3 : 2 * M_PI;

              const Real dOmega = dcos * dphi;
              const Real flux = luminosity * dOmega / (4 * M_PI);

              ppack_r(b, swarm_position::x(), n) = x;
              ppack_r(b, swarm_position::y(), n) = 0.5 * (ym + yp);
              ppack_r(b, swarm_position::z(), n) = z;
              ppack_r(b, rad::part::flux(), n) = flux;
            }
          }
        }
      });
  return TaskStatus::complete;
}

} // namespace RT

#endif // RADIATION_RAYTRACE_RAYTRACE_HPP_