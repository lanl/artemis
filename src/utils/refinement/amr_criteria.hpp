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
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved. Licensed under
// the BSD 3-Clause License (the "LICENSE").
//========================================================================================
#ifndef UTILS_REFINEMENT_AMR_CRITERIA_HPP_
#define UTILS_REFINEMENT_AMR_CRITERIA_HPP_

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn  AmrTag ArtemisUtils::ScalarFirstDerivative
//! \brief
template <typename FIELD, Coordinates GEOM>
AmrTag ScalarFirstDerivative(MeshBlockData<Real> *md) {
  auto pmb = md->GetBlockPointer();
  auto pm = pmb->pmy_mesh;
  auto &pco = pmb->coords;
  auto &resolved_pkgs = pm->resolved_packages;

  Real thr = Null<Real>();
  if constexpr (std::is_same<FIELD, gas::prim::density>::value ||
                std::is_same<FIELD, gas::prim::pressure>::value) {
    thr = pm->packages.Get("gas")->template Param<Real>("refine_thr");
  }

  static auto desc = MakePackDescriptor<FIELD>(resolved_pkgs.get());
  auto v = desc.GetPack(md);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pmb->pmy_mesh->ndim;

  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  Real maxeps = 0.0;
  if (ndim == 3) {
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
        kb.s - 1, kb.e + 1, jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxeps) {
          geometry::Coords<GEOM> coords(cpars, pco, k, j, i);

          // Get stencil widths
          const Real sdx1 =
              vg(0, geom::x1v(), coords.template index<geom::x1v>(k, j, i + 1)) -
              vg(0, geom::x1v(), coords.template index<geom::x1v>(k, j, i - 1));
          const Real sdx2 =
              vg(0, geom::x2v(), coords.template index<geom::x2v>(k, j + 1, i)) -
              vg(0, geom::x2v(), coords.template index<geom::x2v>(k, j - 1, i));
          const Real sdx3 =
              vg(0, geom::x3v(), coords.template index<geom::x2v>(k + 1, j, i)) -
              vg(0, geom::x3v(), coords.template index<geom::x2v>(k - 1, j, i));

          // Get scale factors
          const std::array<Real, 3> hx{
              vg(0, geom::hx1v(), coords.template index<geom::hx1v>(k, j, i)),
              vg(0, geom::hx2v(), coords.template index<geom::hx2v>(k, j, i)),
              vg(0, geom::hx3v(), coords.template index<geom::hx3v>(k, j, i))};
          // NOTE(PDM): here, if passed a SparsePool, we will only be accessing the first
          // entry in the SparsePool.  If more fine-tuned control required, create a
          // user-defined AMR criterion.
          Real eps = std::sqrt(
              SQR((v(0, 0, k, j, i + 1) - v(0, 0, k, j, i - 1)) / sdx1 / hx[0]) +
              SQR((v(0, 0, k, j + 1, i) - v(0, 0, k, j - 1, i)) / sdx2 / hx[1]) +
              SQR((v(0, 0, k + 1, j, i) - v(0, 0, k - 1, j, i)) / sdx3 / hx[2]));
          // NOTE(PDM): somebody please check me if this normalization makes sense
          eps /= (v(0, 0, k, j, i) /
                  std::sqrt(SQR(sdx1 * hx[0]) + SQR(sdx2 * hx[1]) + SQR(sdx3 * hx[2])));
          lmaxeps = std::max(lmaxeps, eps);
        },
        Kokkos::Max<Real>(maxeps));
  } else if (ndim == 2) {
    int k = kb.s;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
        jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int j, const int i, Real &lmaxeps) {
          geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
          // Get stencil widths
          const Real sdx1 =
              vg(0, geom::x1v(), coords.template index<geom::x1v>(k, j, i + 1)) -
              vg(0, geom::x1v(), coords.template index<geom::x1v>(k, j, i - 1));
          const Real sdx2 =
              vg(0, geom::x2v(), coords.template index<geom::x2v>(k, j + 1, i)) -
              vg(0, geom::x2v(), coords.template index<geom::x2v>(k, j - 1, i));
          // Get scale factors

          const std::array<Real, 2> hx{
              vg(0, geom::hx1v(), coords.template index<geom::hx1v>(k, j, i)),
              vg(0, geom::hx2v(), coords.template index<geom::hx2v>(k, j, i))};

          Real eps = std::sqrt(
              SQR((v(0, 0, k, j, i + 1) - v(0, 0, k, j, i - 1)) / sdx1 / hx[0]) +
              SQR((v(0, 0, k, j + 1, i) - v(0, 0, k, j - 1, i)) / sdx2 / hx[1]));
          // NOTE(PDM): again, please check me...
          eps /= (v(0, 0, k, j, i) / std::sqrt(SQR(sdx1 * hx[0]) + SQR(sdx2 * hx[1])));
          lmaxeps = std::max(lmaxeps, eps);
        },
        Kokkos::Max<Real>(maxeps));
  } else {
    return AmrTag::same;
  }

  if (maxeps > thr) {
    return AmrTag::refine;
  }
  if (maxeps < 0.25 * thr) return AmrTag::derefine;
  return AmrTag::same;
}

//----------------------------------------------------------------------------------------
//! \fn  AmrTag ArtemisUtils::ScalarMagnitude
//! \brief
template <typename FIELD>
AmrTag ScalarMagnitude(MeshBlockData<Real> *md) {
  auto pmb = md->GetBlockPointer();
  auto pm = pmb->pmy_mesh;
  auto &resolved_pkgs = pm->resolved_packages;

  Real refine_above = Null<Real>();
  Real deref_below = Null<Real>();
  // NOTE(PDM): this is really ugly, but I am not sure a way around it yet...
  if constexpr (std::is_same<FIELD, gas::prim::density>::value ||
                std::is_same<FIELD, gas::prim::pressure>::value) {
    refine_above = pm->packages.Get("gas")->template Param<Real>("refine_thr");
    deref_below = pm->packages.Get("gas")->template Param<Real>("deref_thr");
  }

  static auto desc = MakePackDescriptor<FIELD>(resolved_pkgs.get());
  auto v = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  Real maxvv = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxvv) {
        lmaxvv = std::max(lmaxvv, v(0, 0, k, j, i));
      },
      Kokkos::Max<Real>(maxvv));

  if (maxvv > refine_above) return parthenon::AmrTag::refine;
  if (maxvv < deref_below) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
}

} // namespace ArtemisUtils

#endif // UTILS_REFINEMENT_AMR_CRITERIA_HPP_
