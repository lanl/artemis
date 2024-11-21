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
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pmb->pmy_mesh->ndim;

  Real maxeps = 0.0;
  if (ndim == 3) {
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
        kb.s - 1, kb.e + 1, jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxeps) {
          // Get coordinate positions
          Real ip1[3] = {Null<Real>()}, im1[3] = {Null<Real>()};
          Real jp1[3] = {Null<Real>()}, jm1[3] = {Null<Real>()};
          Real kp1[3] = {Null<Real>()}, km1[3] = {Null<Real>()};

          geometry::Coords<GEOM> coords_ip1(pco, k, j, i + 1);
          geometry::Coords<GEOM> coords_im1(pco, k, j, i - 1);
          geometry::Coords<GEOM> coords_jp1(pco, k, j + 1, i);
          geometry::Coords<GEOM> coords_jm1(pco, k, j - 1, i);
          geometry::Coords<GEOM> coords_kp1(pco, k + 1, j, i);
          geometry::Coords<GEOM> coords_km1(pco, k - 1, j, i);
          coords_ip1.GetCellCenter(ip1);
          coords_im1.GetCellCenter(im1);
          coords_jp1.GetCellCenter(jp1);
          coords_jm1.GetCellCenter(jm1);
          coords_kp1.GetCellCenter(kp1);
          coords_km1.GetCellCenter(km1);

          // Get stencil widths
          const Real sdx1 = ip1[0] - im1[0];
          const Real sdx2 = jp1[1] - jm1[1];
          const Real sdx3 = kp1[2] - km1[2];
          // Get scale factors
          Real cc[3] = {Null<Real>()};
          Real hx[3] = {Null<Real>()};
          geometry::Coords<GEOM> coords(pco, k, j, i);
          coords.GetCellCenter(cc);
          coords.GetScaleFactors(hx);
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
          // Get coordinate positions
          Real ip1[3] = {Null<Real>()}, im1[3] = {Null<Real>()};
          Real jp1[3] = {Null<Real>()}, jm1[3] = {Null<Real>()};

          geometry::Coords<GEOM> coords_ip1(pco, k, j, i + 1);
          geometry::Coords<GEOM> coords_im1(pco, k, j, i - 1);
          geometry::Coords<GEOM> coords_jp1(pco, k, j + 1, i);
          geometry::Coords<GEOM> coords_jm1(pco, k, j - 1, i);
          coords_ip1.GetCellCenter(ip1);
          coords_im1.GetCellCenter(im1);
          coords_jp1.GetCellCenter(jp1);
          coords_jm1.GetCellCenter(jm1);

          // Get stencil widths
          const Real sdx1 = ip1[0] - im1[0];
          const Real sdx2 = jp1[1] - jm1[1];
          // Get scale factors
          Real cc[3] = {Null<Real>()};
          geometry::Coords<GEOM> coords(pco, k, j, i);
          coords.GetCellCenter(cc);
          const Real hx1 = coords.hx1(cc[0], cc[1], cc[2]);
          const Real hx2 = coords.hx2(cc[0], cc[1], cc[2]);
          Real eps =
              std::sqrt(SQR((v(0, 0, k, j, i + 1) - v(0, 0, k, j, i - 1)) / sdx1 / hx1) +
                        SQR((v(0, 0, k, j + 1, i) - v(0, 0, k, j - 1, i)) / sdx2 / hx2));
          // NOTE(PDM): again, please check me...
          eps /= (v(0, 0, k, j, i) / std::sqrt(SQR(sdx1 * hx1) + SQR(sdx2 * hx2)));
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
