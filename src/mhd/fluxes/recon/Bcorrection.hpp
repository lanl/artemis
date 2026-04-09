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
#ifndef UTILS_FLUXES_RECONSTRUCTION_BCORRECTION_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_BCORRECTION_HPP_

// Artemis includes
#include "artemis.hpp"

// NOTE(PDM): The following is taken directly from the open-source AthenaK software, and
// adapted for Parthenon/Artemis by PDM

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::Bcorrection, X1DIR, ...>
//! \brief The piecewise constant reconstruction method in the X1 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::Bcorrection, X1DIR, GEOM> {
 public:
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void apply_fcc(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V1 &q, parthenon::ScratchPad2D<Real> &ql,
                                    parthenon::ScratchPad2D<Real> &qr, const V2 &qf) const {
    // YH: should be correct as Bx does not change and this input correctly
    int n = q.GetUpperBound(b)-2-7; // YH: don't think this is general (only 1D)
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
			         // YH: For Bx
                                 ql(n, i) = qf(b, TE::F1, 1, k, j, i); 
                                 qr(n, i) = qf(b, TE::F1, 1, k, j, i);
			       	 });
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::Bcorrection, X2DIR, ...>
//! \brief The piecewise constant reconstruction method in the X2 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::Bcorrection, X2DIR, GEOM> {
 public:
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void apply_fcc(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V1 &q, parthenon::ScratchPad2D<Real> &ql_j,
                                    parthenon::ScratchPad2D<Real> &qr_j, const V2 &qf) const {
    int n = q.GetUpperBound(b)-1-7;
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
                                 ql_j(n, i) = qf(b, TE::F2, 1, k, j, i);
                                 qr_j(n, i) = qf(b, TE::F2, 1, k, j, i);
                               });
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::Bcorrection, X3DIR, ...>
//! \brief The piecewise constant reconstruction method in the X3 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::Bcorrection, X3DIR, GEOM> {
 public:
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void apply_fcc(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V1 &q, parthenon::ScratchPad2D<Real> &ql_k,
                                    parthenon::ScratchPad2D<Real> &qr_k, const V2 &qf) const {
    int n = q.GetUpperBound(b)-7;
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu,
                               [&](const int i) {
                                 ql_k(n, i) = qf(b, TE::F3, 1, k, j, i);
                                 qr_k(n, i) = qf(b, TE::F3, 1, k, j, i);
                               });
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_BCORRECTION_HPP_
