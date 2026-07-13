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

// C++ headers
#include <limits>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/units.hpp"

namespace RT {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants);

TaskListStatus RaytraceDriver(Mesh *pmesh, const Real time);

struct ParticleWeights {
  Real dx2 = 1.0;
  Real dx3 = 1.0;
  int np2 = 1;
  int np3 = 1;
  int max_level = 0;
  Real mult = 1.;
  Real x2min, x2max, x3min, x3max;
  ParticleWeights(const Real x2min, const Real x2max, const Real x3min, const Real x3max)
      : x2min(x2min), x2max(x2max), x3min(x3min), x3max(x3max) {};
};

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RT::GetIndices
//! \brief Map x1,x2,x3 to i,j,k
KOKKOS_FORCEINLINE_FUNCTION std::array<int, 3>
GetIndices(const parthenon::Coordinates_t &pco, std::array<Real, 3> x) {
  return {
      static_cast<int>(std::floor((x[0] - pco.Xf<1>(0)) / pco.CellWidth<1>(0, 0, 0))),
      static_cast<int>(std::floor((x[1] - pco.Xf<2>(0)) / pco.CellWidth<2>(0, 0, 0))),
      static_cast<int>(std::floor((x[2] - pco.Xf<3>(0)) / pco.CellWidth<3>(0, 0, 0)))};
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RT::PushParticlesImpl
//! \brief Implementation for pushing particles
template <Coordinates GEOM, bool LOGR>
TaskStatus PushParticlesImpl(MeshData<Real> *md, const geometry::CoordParams &cpars) {
  PARTHENON_INSTRUMENT
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &rt_pkg = pm->packages.Get("raytrace");
  const auto efloor = rt_pkg->template Param<Real>("efloor");
  const auto x1max = rt_pkg->template Param<Real>("x1max");
  const auto x1min = rt_pkg->template Param<Real>("x1min");
  const auto zero_rad = rt_pkg->template Param<Real>("stellar_radius") *
                        rt_pkg->template Param<Real>("radius_factor");
  // Create SparsePack
  static auto desc =
      MakePackDescriptor<rad::star::absorption, gas::src::energy>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);

  static auto desc_g = MakePackDescriptor<geom::vol>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);

  // Create SwarmPacks
  static auto pdesc_r =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              rad::star::flux>("star");
  static auto pdesc_i = MakeSwarmPackDescriptor<rad::star::ijk>("star");
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Indexing and dimensionality
  const int ndim = pm->ndim;
  const bool multi_d = (ndim >= 2);
  const bool three_d = (ndim == 3);
  const int &nblocks = vmesh.GetNBlocks();
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  const int ngh = parthenon::Globals::nghost;

  const Real rmin = (LOGR) ? std::exp(x1min) : x1min;
  const Real x1tol = 32.0 * std::numeric_limits<Real>::epsilon() *
                     std::max(1.0, std::max(std::abs(x1min), std::abs(x1max)));

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "TransportPhotons", DevExecSpace(), 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          Real &ee = ppack_r(b, rad::star::flux(), n);
          Real &xp = ppack_r(b, swarm_position::x(), n);
          const Real &yp = ppack_r(b, swarm_position::y(), n);
          const Real &zp = ppack_r(b, swarm_position::z(), n);
          int &i = ppack_i(b, rad::star::ijk(0), n);
          int &j = ppack_i(b, rad::star::ijk(1), n);
          int &k = ppack_i(b, rad::star::ijk(2), n);
          const auto &pco = vmesh.GetCoordinates(b);
          const auto inds = GetIndices(pco, {xp, yp, zp});
          i = ib.s + inds[0] - ngh;
          // A ray arrives exactly on a block face. Roundoff in floor() can otherwise
          // select the last ghost cell and deposit outside the active mesh.
          if (i == ib.s - 1 && std::abs(xp - pco.template Xf<X1DIR>(ib.s)) <= x1tol) {
            i = ib.s;
          }
          if (multi_d) j = jb.s + inds[1] - ngh;
          if (three_d) k = kb.s + inds[2] - ngh;

          while ((i >= ib.s) && (i <= ib.e) && (ee > 0.0)) {
            geometry::Coords<GEOM> coords(cpars, pco, k, j, i);

            // Deposit energy for this cell and decrement the photon energy
            const auto dx = coords.bnds.x1[1] - coords.bnds.x1[0];
            Real dtau = std::max(0.0, vmesh(b, rad::star::absorption(), k, j, i));

            // Corrections for additional extinction inside the inner boundary
            if (xp <= x1min + x1tol) {
              const Real inner_path = std::max(0.0, rmin - zero_rad);
              Real dtau_i = dtau * inner_path;
              const Real efac = (dtau_i > 100.) ? 0.0 : std::exp(-dtau_i);
              ee *= efac;
              if (ee < efloor) ee = 0.0;
              if (ee == 0.0) {
                swarm_d.MarkParticleForRemoval(n);
                break;
              }
            }
            dtau *= std::max(0.0, dx);
            const Real efac = (dtau > 100.) ? 0.0 : std::exp(-dtau);
            const Real reduc = (dtau <= 1e-4)
                                   ? dtau - 0.5 * SQR(dtau) + dtau * SQR(dtau) / 6.0
                                   : (1. - efac);
            Real dE = ee * reduc;
            ee *= efac;

            if (ee < efloor) ee = 0.0;

            Kokkos::atomic_add(&(vmesh(b, gas::src::energy(), k, j, i)),
                               dE / coords.GetVolume(vg, b, k, j, i));

            // move the particle to the next face;
            i += 1;
            if constexpr (LOGR) {
              xp = pco.template Xf<X1DIR>(i);
            } else {
              xp = coords.bnds.x1[1];
            }
            if (std::abs(xp - x1max) <= x1tol) xp = x1max;
            if ((ee == 0.0) || (xp >= x1max)) {
              swarm_d.MarkParticleForRemoval(n);
              break;
            }
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RT::SourceParticlesImpl
//! \brief Implementation for sourcing particles
template <Coordinates GEOM, bool LOGR>
TaskStatus SourceParticlesImpl(MeshData<Real> *md, const geometry::CoordParams &cpars,
                               const ParticleWeights &pwght) {
  PARTHENON_INSTRUMENT
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
  const auto x1min = rt_pkg->template Param<Real>("x1min");

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
      const int mult = 1 << (pwght.max_level - pmb->loc.level());
      int nbx2 = pmb->block_size.nx_[1];
      int mult_fac = 1;
      if (nbx2 > 1) mult_fac *= mult;
      int nbx3 = pmb->block_size.nx_[2];
      if (nbx3 > 1) mult_fac *= mult;
      new_parts_h(b) = nbx2 * nbx3 * mult_fac;
      new_part_per_cell_h(b) = mult;

      new_contexts_h(b) = particles->AddEmptyParticles(new_parts_h(b));
      nparticles += new_parts_h(b);
    }
  }
  new_contexts.DeepCopy(new_contexts_h);
  new_parts.DeepCopy(new_parts_h);
  new_part_per_cell.DeepCopy(new_part_per_cell_h);

  static auto pdesc_r =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              rad::star::flux>("star");
  static auto pdesc_i = MakeSwarmPackDescriptor<rad::star::ijk>("star");
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  // Initialize particles
  //
  const int three_d = pm->ndim == 3;
  const int multi_d = pm->ndim >= 2;
  const auto luminosity = rt_pkg->template Param<Real>("luminosity");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SourceParticles::Source", parthenon::DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j) {
        const int tot = new_parts(b);
        if (tot > 0) {
          const auto &pco = vmesh.GetCoordinates(b);
          geometry::Coords<GEOM> coords(cpars, pco, k, j, ib.s);
          const int nbx2 = (jb.e - jb.s) + 1;
          const int nper_cell = (multi_d) ? new_part_per_cell(b) : 1;
          const int ntot_cell = (three_d) ? SQR(nper_cell) : nper_cell;
          const int offset = (j - jb.s) * ntot_cell + (k - kb.s) * ntot_cell * nbx2;
          const int kpe = (three_d) ? nper_cell : 1;
          for (int kp = 0; kp < kpe; kp++) {
            for (int jp = 0; jp < nper_cell; jp++) {
              const int np = offset + jp + nper_cell * kp;
              const int &n = new_contexts(b).GetNewParticleIndex(np);
              ppack_i(b, rad::star::ijk(0), n) = ib.s;
              ppack_i(b, rad::star::ijk(1), n) = j;
              ppack_i(b, rad::star::ijk(2), n) = k;
              Real x = Null<Real>();
              if constexpr (LOGR) {
                x = pco.template Xf<X1DIR>(ib.s);
              } else {
                x = coords.bnds.x1[0];
              }
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
              ppack_r(b, rad::star::flux(), n) = flux;
            }
          }
        }
      });
  return TaskStatus::complete;
}

} // namespace RT

#endif // RADIATION_RAYTRACE_RAYTRACE_HPP_
