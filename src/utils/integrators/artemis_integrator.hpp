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
template <Coordinates GEOM, bool include_divf = true>
TaskStatus ApplyUpdate(MeshData<Real> *u0, MeshData<Real> *u1, const Real g0,
                       const Real g1, const Real beta_dt) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();

  // Packing and indexing
  std::vector<MetadataFlag> flags({Metadata::Conserved});
  static auto desc = MakePackDescriptor<any>(u0, flags, {parthenon::PDOpt::WithFluxes});
  static auto desc_g = MakePackDescriptor<geom::vol, geom::ax1, geom::ax2, geom::ax3>(u0);
  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc.GetPack(u1);
  const auto vg = desc_g.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, v0.GetCoordinates(b), k, j, i);
        const int d1 = X1DIR;
        const int d2 = d1 + multi_d;
        const int d3 = d2 + three_d;
        const Real bdt_vol = beta_dt / coords.GetVolume(vg, b, k, j, i);
        [[maybe_unused]] std::array<Real, 2> ax1{0}, ax2{0}, ax3{0};
        if constexpr (include_divf) {
          ax1 = coords.GetFaceAreaX1(vg, b, k, j, i);
          ax2 = coords.GetFaceAreaX2(vg, b, k, j, i);
          ax3 = coords.GetFaceAreaX3(vg, b, k, j, i);
        }
        // Advance state vector with flux divergence
        for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
          Real &v0n = v0(b, n, k, j, i);
          Real &v1n = v1(b, n, k, j, i);
          v0n = g0 * v0n + g1 * v1n;
          if constexpr (include_divf) {
            v0n += bdt_vol * ((ax1[0] * v0.flux(b, d1, n, k, j, i) -
                               ax1[1] * v0.flux(b, d1, n, k, j, i + 1)) +
                              (ax2[0] * v0.flux(b, d2, n, k, j, i) -
                               ax2[1] * v0.flux(b, d2, n, k, j + multi_d, i)) +
                              (ax3[0] * v0.flux(b, d3, n, k, j, i) -
                               ax3[1] * v0.flux(b, d3, n, k + three_d, j, i)));
          }
        }
      });
  return TaskStatus::complete;
}

} // namespace ArtemisUtils

#endif // UTILS_INTEGRATORS_ARTEMIS_INTEGRATOR_HPP_
