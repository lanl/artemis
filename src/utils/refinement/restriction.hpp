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
#ifndef UTILS_REFINEMENT_RESTRICTION_HPP_
#define UTILS_REFINEMENT_RESTRICTION_HPP_

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

using namespace parthenon;
using namespace parthenon::driver::prelude;

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \struct  ArtemisUtils::RestrictAverage
//! \brief
template <Coordinates GEOM>
struct RestrictAverage {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &pco, const Coordinates_t &coarse_pco,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    PARTHENON_REQUIRE(
        el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::F3,
        "Artemis restriction only supports cell-centered and face-centered fields!");
    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2);
    constexpr int element_idx = static_cast<int>(el) % 3;

    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    const int i = (DIM > 0) ? (ci - cib.s) * 2 + ib.s : ib.s;
    const int j = (DIM > 1) ? (cj - cjb.s) * 2 + jb.s : jb.s;
    const int k = (DIM > 2) ? (ck - ckb.s) * 2 + kb.s : kb.s;

    // JMM: If dimensionality is wrong, accesses are out of bounds. Only
    // access cells if dimensionality is correct.
    Real vol[2][2][2], terms[2][2][2]; // memset not available on all accelerators
    for (int ok = 0; ok < 2; ++ok) {
      for (int oj = 0; oj < 2; ++oj) {
        for (int oi = 0; oi < 2; ++oi) {
          vol[ok][oj][oi] = terms[ok][oj][oi] = 0;
        }
      }
    }

    for (int ok = 0; ok < 1 + INCLUDE_X3; ++ok) {
      for (int oj = 0; oj < 1 + INCLUDE_X2; ++oj) {
        for (int oi = 0; oi < 1 + INCLUDE_X1; ++oi) {
          geometry::Coords<GEOM> artemis_coords(pco, k + ok, j + oj, i + oi);
          if constexpr (el == TE::CC) {
            vol[ok][oj][oi] = artemis_coords.Volume();
          } else if constexpr (el == TE::F1) {
            vol[ok][oj][oi] = artemis_coords.GetFaceArea(X1DIR);
          } else if constexpr (el == TE::F2) {
            vol[ok][oj][oi] = artemis_coords.GetFaceArea(X2DIR);
          } else if constexpr (el == TE::F3) {
            vol[ok][oj][oi] = artemis_coords.GetFaceArea(X3DIR);
          }
          terms[ok][oj][oi] =
              vol[ok][oj][oi] * fine(element_idx, l, m, n, k + ok, j + oj, i + oi);
        }
      }
    }

    // KGF: add the off-centered quantities first to preserve FP
    // symmetry
    const Real tvol = ((vol[0][0][0] + vol[0][1][0]) + (vol[0][0][1] + vol[0][1][1])) +
                      ((vol[1][0][0] + vol[1][1][0]) + (vol[1][0][1] + vol[1][1][1]));
    coarse(element_idx, l, m, n, ck, cj, ci) =
        (((terms[0][0][0] + terms[0][1][0]) + (terms[0][0][1] + terms[0][1][1])) +
         ((terms[1][0][0] + terms[1][1][0]) + (terms[1][0][1] + terms[1][1][1]))) /
        tvol;
  }
};

} // namespace ArtemisUtils

#endif // UTILS_REFINEMENT_RESTRICTION_HPP_
