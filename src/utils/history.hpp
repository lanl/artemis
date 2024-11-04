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
#ifndef UTILS_HISTORY_HPP_
#define UTILS_HISTORY_HPP_

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

using ArtemisUtils::VI;

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn  std::vector<Real> ArtemisUtils::ReduceSpeciesVolumeIntegral
//! \brief Returns the per-species vector of volume integrals of the volumetric vector
//!        variable specified by VAR
template <Coordinates GEOM, typename VAR>
std::vector<Real> ReduceSpeciesVolumeIntegral(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<VAR>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  const int nspecies_total = vmesh.GetMaxNumberOfVars();

  std::vector<Real> integrals(nspecies_total, 0.0);
  for (int n = 0; n < nspecies_total; n++) {
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ReduceSpeciesVolumeIntegral",
        parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      Real &lint) {
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          const Real vv = coords.Volume();
          for (int nn = 0; nn < vmesh.GetSize(b, VAR()); nn++) {
            if (vmesh(b, VAR(nn)).sparse_id == n) {
              lint += vmesh(b, VAR(nn), k, j, i) * vv;
            }
          }
        },
        Kokkos::Sum<Real>(integrals[n]));
  }

  return integrals;
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<Real> ArtemisUtils::ReduceSpeciesVectorVolumeIntegral
//! \brief Returns the per-species vector of  volume integrals of the volumetric vector
//!        variable specified by VAR
template <Coordinates GEOM, int DIR, typename VAR>
std::vector<Real> ReduceSpeciesVectorVolumeIntegral(MeshData<Real> *md) {
  PARTHENON_REQUIRE(DIR > 0 && DIR <= 3, "Direction must be X1DIR, X2DIR, or X3DIR!");

  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  static auto desc = MakePackDescriptor<VAR>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  const int nspecies_total = vmesh.GetMaxNumberOfVars() / 3;

  std::vector<Real> integrals(nspecies_total, 0.0);
  for (int n = 0; n < nspecies_total; n++) {
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ReduceSpeciesVectorVolumeIntegral",
        parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      Real &lint) {
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
          const Real vv = coords.Volume();
          for (int nn = 0; nn < vmesh.GetSize(b, VAR()) / 3; nn++) {
            if (vmesh(b, VAR(VI(nn, DIR - 1))).sparse_id == n) {
              lint += vmesh(b, VAR(VI(nn, DIR - 1)), k, j, i) * vv;
            }
          }
        },
        Kokkos::Sum<Real>(integrals[n]));
  }

  return integrals;
}

} // namespace ArtemisUtils

#endif // UTILS_HISTORY_HPP_
