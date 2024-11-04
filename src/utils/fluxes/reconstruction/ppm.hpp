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
#ifndef UTILS_FLUXES_RECONSTRUCTION_PPM_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_PPM_HPP_

// Artemis includes
#include "artemis.hpp"

// NOTE(PDM): The following is taken directly from the open-source AthenaK software, and
// adapted for Parthenon/Artemis by PDM

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn ArtemisUtils::PPM4()
//! \brief Original PPM (Colella & Woodward) parabolic reconstruction.  Returns
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.
KOKKOS_INLINE_FUNCTION
void PPM4(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
          const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  //---- Interpolate L/R values (CS eqn 16, PH 3.26 and 3.27) ----
  // qlv = q at left  side of cell-center = q[i-1/2] = a_{j,-} in CS
  // qrv = q at right side of cell-center = q[i+1/2] = a_{j,+} in CS
  Real qlv = (7. * (q_i + q_im1) - (q_im2 + q_ip1)) / 12.0;
  Real qrv = (7. * (q_i + q_ip1) - (q_im1 + q_ip2)) / 12.0;

  //---- limit qrv and qlv to neighboring cell-centered values (CS eqn 13) ----
  qlv = std::max(qlv, std::min(q_i, q_im1));
  qlv = std::min(qlv, std::max(q_i, q_im1));
  qrv = std::max(qrv, std::min(q_i, q_ip1));
  qrv = std::min(qrv, std::max(q_i, q_ip1));

  //--- monotonize interpolated L/R states (CS eqns 14, 15) ---
  Real qc = qrv - q_i;
  Real qd = qlv - q_i;
  if ((qc * qd) >= 0.0) {
    qlv = q_i;
    qrv = q_i;
  } else {
    if (fabs(qc) >= 2.0 * fabs(qd)) {
      qrv = q_i - 2.0 * qd;
    }
    if (fabs(qd) >= 2.0 * fabs(qc)) {
      qlv = q_i - 2.0 * qc;
    }
  }

  //---- set L/R states ----
  ql_ip1 = qrv;
  qr_i = qlv;
  return;
}

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::ppm, X1DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X1 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::ppm, X1DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql,
                                    parthenon::ScratchPad2D<Real> &qr) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            PPM4(q(b, n, k, j, i - 2), q(b, n, k, j, i - 1), q(b, n, k, j, i),
                 q(b, n, k, j, i + 1), q(b, n, k, j, i + 2), ql(n, i + 1), qr(n, i));
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::ppm, X2DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X2 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::ppm, X2DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_jp1,
                                    parthenon::ScratchPad2D<Real> &qr_j) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            PPM4(q(b, n, k, j - 2, i), q(b, n, k, j - 1, i), q(b, n, k, j, i),
                 q(b, n, k, j + 1, i), q(b, n, k, j + 2, i), ql_jp1(n, i), qr_j(n, i));
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::ppm, X3DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X3 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::ppm, X3DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_kp1,
                                    parthenon::ScratchPad2D<Real> &qr_k) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            PPM4(q(b, n, k - 2, j, i), q(b, n, k - 1, j, i), q(b, n, k, j, i),
                 q(b, n, k + 1, j, i), q(b, n, k + 2, j, i), ql_kp1(n, i), qr_k(n, i));
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_PPM_HPP_
