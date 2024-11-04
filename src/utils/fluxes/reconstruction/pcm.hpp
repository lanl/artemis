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
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef UTILS_FLUXES_RECONSTRUCTION_PCM_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_PCM_HPP_

// Artemis includes
#include "artemis.hpp"

// NOTE(PDM): The following is taken directly from the open-source AthenaK software, and
// adapted for Parthenon/Artemis by PDM

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::pcm, X1DIR, ...>
//! \brief The piecewise constant reconstruction method in the X1 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::pcm, X1DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql,
                                    parthenon::ScratchPad2D<Real> &qr) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
                                 ql(n, i + 1) = q(b, n, k, j, i);
                                 qr(n, i) = q(b, n, k, j, i);
                               });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::pcm, X2DIR, ...>
//! \brief The piecewise constant reconstruction method in the X2 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::pcm, X2DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_jp1,
                                    parthenon::ScratchPad2D<Real> &qr_j) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
                                 ql_jp1(n, i) = q(b, n, k, j, i);
                                 qr_j(n, i) = q(b, n, k, j, i);
                               });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::pcm, X3DIR, ...>
//! \brief The piecewise constant reconstruction method in the X3 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::pcm, X3DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_kp1,
                                    parthenon::ScratchPad2D<Real> &qr_k) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
                                 ql_kp1(n, i) = q(b, n, k, j, i);
                                 qr_k(n, i) = q(b, n, k, j, i);
                               });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_PCM_HPP_
