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
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef UTILS_REFINEMENT_PROLONGATION_HPP_
#define UTILS_REFINEMENT_PROLONGATION_HPP_

// C++ includes
#include <algorithm>
#include <cstring>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <interface/variable_state.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/domain.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn  void ArtemisUtils::GetGridSpacings
//! \brief compute distances from cell center to the nearest center in the + or -
//!        coordinate direction. Do so for both coarse and fine grids.
template <Coordinates C, int DIM>
KOKKOS_FORCEINLINE_FUNCTION void
GetGridSpacings(const Coordinates_t &coords, const Coordinates_t &coarse_coords, int k,
                int j, int i, int fk, int fj, int fi, Real *dxm, Real *dxp, Real *dxfm,
                Real *dxfp) {
  namespace gg = geometry;
  Real xm = Null<Real>(), xc = Null<Real>(), xp = Null<Real>();
  Real fxm = Null<Real>(), fxp = Null<Real>();
  gg::Coords<C> cc(coarse_coords, k, j, i);
  gg::Coords<C> cm(coarse_coords, k - (DIM == 3), j - (DIM == 2), i - (DIM == 1));
  gg::Coords<C> cp(coarse_coords, k + (DIM == 3), j + (DIM == 2), i + (DIM == 1));
  gg::Coords<C> fm(coords, fk, fj, fi);
  gg::Coords<C> fp(coords, fk + (DIM == 3), fj + (DIM == 2), fi + (DIM == 1));
  if constexpr (DIM == 1) {
    xm = cm.x1v(), xc = cc.x1v(), xp = cp.x1v();
    fxm = fm.x1v(), fxp = fp.x1v();
  } else if constexpr (DIM == 2) {
    xm = cm.x2v(), xc = cc.x2v(), xp = cp.x2v();
    fxm = fm.x2v(), fxp = fp.x2v();
  } else if constexpr (DIM == 3) {
    xm = cm.x3v(), xc = cc.x3v(), xp = cp.x3v();
    fxm = fm.x3v(), fxp = fp.x3v();
  }
  *dxm = xc - xm;
  *dxp = xp - xc;
  *dxfm = xc - fxm;
  *dxfp = fxp - xc;
}

//----------------------------------------------------------------------------------------
//! \fn  Real ArtemisUtils::GradMinMod
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real GradMinMod(const Real fc, const Real fm, const Real fp, const Real dxm,
                const Real dxp, Real &gxm, Real &gxp) {
  gxm = (fc - fm) / dxm;
  gxp = (fp - fc) / dxp;
  return 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
}

//----------------------------------------------------------------------------------------
//! \struct  ArtemisUtils::ProlongateSharedMinMod
//! \brief
template <Coordinates GEOM>
struct ProlongateSharedMinMod {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    PARTHENON_REQUIRE((el == TE::CC), "Artemis AMR only supports cell-centered fields!");

    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    constexpr int element_idx = static_cast<int>(el) % 3;

    const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
    const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
    const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

    constexpr bool INCLUDE_X1 = (DIM > 0);
    constexpr bool INCLUDE_X2 = (DIM > 1);
    constexpr bool INCLUDE_X3 = (DIM > 2);

    const Real fc = coarse(element_idx, l, m, n, k, j, i);

    Real dx1fm = 0;
    [[maybe_unused]] Real dx1fp = 0;
    [[maybe_unused]] Real gx1m = 0, gx1p = 0;
    if constexpr (INCLUDE_X1) {
      Real dx1m, dx1p;
      ArtemisUtils::GetGridSpacings<GEOM, 1>(coords, coarse_coords, k, j, i, fk, fj, fi,
                                             &dx1m, &dx1p, &dx1fm, &dx1fp);

      Real gx1c = ArtemisUtils::GradMinMod(fc, coarse(element_idx, l, m, n, k, j, i - 1),
                                           coarse(element_idx, l, m, n, k, j, i + 1),
                                           dx1m, dx1p, gx1m, gx1p);
      gx1m = gx1c;
      gx1p = gx1c;
    }

    Real dx2fm = 0;
    [[maybe_unused]] Real dx2fp = 0;
    [[maybe_unused]] Real gx2m = 0, gx2p = 0;
    if constexpr (INCLUDE_X2) {
      Real dx2m, dx2p;
      ArtemisUtils::GetGridSpacings<GEOM, 2>(coords, coarse_coords, k, j, i, fk, fj, fi,
                                             &dx2m, &dx2p, &dx2fm, &dx2fp);
      Real gx2c = ArtemisUtils::GradMinMod(fc, coarse(element_idx, l, m, n, k, j - 1, i),
                                           coarse(element_idx, l, m, n, k, j + 1, i),
                                           dx2m, dx2p, gx2m, gx2p);
      gx2m = gx2c;
      gx2p = gx2c;
    }

    Real dx3fm = 0;
    [[maybe_unused]] Real dx3fp = 0;
    [[maybe_unused]] Real gx3m = 0, gx3p = 0;
    if constexpr (INCLUDE_X3) {
      Real dx3m, dx3p;
      ArtemisUtils::GetGridSpacings<GEOM, 3>(coords, coarse_coords, k, j, i, fk, fj, fi,
                                             &dx3m, &dx3p, &dx3fm, &dx3fp);
      Real gx3c = ArtemisUtils::GradMinMod(fc, coarse(element_idx, l, m, n, k - 1, j, i),
                                           coarse(element_idx, l, m, n, k + 1, j, i),
                                           dx3m, dx3p, gx3m, gx3p);
      gx3m = gx3c;
      gx3p = gx3c;
    }

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    fine(element_idx, l, m, n, fk, fj, fi) =
        fc - (gx1m * dx1fm + gx2m * dx2fm + gx3m * dx3fm);
    if constexpr (INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj, fi + 1) =
          fc + (gx1p * dx1fp - gx2m * dx2fm - gx3m * dx3fm);
    if constexpr (INCLUDE_X2)
      fine(element_idx, l, m, n, fk, fj + 1, fi) =
          fc - (gx1m * dx1fm - gx2p * dx2fp + gx3m * dx3fm);
    if constexpr (INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj + 1, fi + 1) =
          fc + (gx1p * dx1fp + gx2p * dx2fp - gx3m * dx3fm);
    if constexpr (INCLUDE_X3)
      fine(element_idx, l, m, n, fk + 1, fj, fi) =
          fc - (gx1m * dx1fm + gx2m * dx2fm - gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj, fi + 1) =
          fc + (gx1p * dx1fp - gx2m * dx2fm + gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi) =
          fc - (gx1m * dx1fm - gx2p * dx2fp - gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi + 1) =
          fc + (gx1p * dx1fp + gx2p * dx2fp + gx3p * dx3fp);
  }
};

} // namespace ArtemisUtils

#endif // UTILS_REFINEMENT_PROLONGATION_HPP_
