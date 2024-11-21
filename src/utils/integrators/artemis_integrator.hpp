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
#ifndef UTILS_INTEGRATORS_ARTEMIS_INTEGRATOR_HPP_
#define UTILS_INTEGRATORS_ARTEMIS_INTEGRATOR_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::DeepCopyConservedData
//! \brief
inline TaskStatus DeepCopyConservedData(MeshData<Real> *to, MeshData<Real> *from) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;

  std::vector<MetadataFlag> flags({Metadata::Conserved});
  static auto desc = MakePackDescriptor<any>(to, flags);
  const auto vt = desc.GetPack(to);
  const auto vf = desc.GetPack(from);
  const auto ibe = to->GetBoundsI(IndexDomain::entire);
  const auto jbe = to->GetBoundsJ(IndexDomain::entire);
  const auto kbe = to->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DeepCopyConservedData", parthenon::DevExecSpace(), 0,
      to->NumBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        for (int n = vt.GetLowerBound(b); n <= vt.GetUpperBound(b); ++n) {
          vt(b, n, k, j, i) = vf(b, n, k, j, i);
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::ApplyUpdate
//! \brief
template <Coordinates GEOM>
TaskStatus ApplyUpdate(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();

  // Extract integrator weights
  const Real gam0 = integrator->gam0[stage - 1];
  const Real gam1 = integrator->gam1[stage - 1];
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;

  // Packing and indexing
  std::vector<MetadataFlag> flags({Metadata::Conserved, Metadata::WithFluxes});
  static auto desc = MakePackDescriptor<any>(u0, flags, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);

        Real ax1[2] = {Null<Real>()}, ax2[2] = {Null<Real>()}, ax3[2] = {Null<Real>()};
        coords.GetFaceAreaX1(ax1);
        if (multi_d) coords.GetFaceAreaX2(ax2);
        if (three_d) coords.GetFaceAreaX3(ax3);

        const Real vol = coords.Volume();

        for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
          // compute flux divergence
          Real divf = (ax1[0] * v0.flux(b, X1DIR, n, k, j, i) -
                       ax1[1] * v0.flux(b, X1DIR, n, k, j, i + 1));
          if (multi_d)
            divf += (ax2[0] * v0.flux(b, X2DIR, n, k, j, i) -
                     ax2[1] * v0.flux(b, X2DIR, n, k, j + 1, i));
          if (three_d)
            divf += (ax3[0] * v0.flux(b, X3DIR, n, k, j, i) -
                     ax3[1] * v0.flux(b, X3DIR, n, k + 1, j, i));

          // Apply update
          v0(b, n, k, j, i) =
              gam0 * v0(b, n, k, j, i) + gam1 * v1(b, n, k, j, i) + divf * beta_dt / vol;
        }
      });
  return TaskStatus::complete;
}

} // namespace ArtemisUtils

#endif // UTILS_INTEGRATORS_ARTEMIS_INTEGRATOR_HPP_
