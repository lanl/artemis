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
#ifndef UTILS_FLUXES_RECONSTRUCTION_PLM_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_PLM_HPP_

// Artemis includes
#include "artemis.hpp"

// NOTE(PDMM): The following is taken directly from the open-source AthenaK software, and
// adapted for Parthenon/Artemis by PDMM

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn ArtemisUtils::PLM()
//! \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.
KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
         Real &qr_i) {
  // compute L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
  Real dq2 = dql * dqr;
  Real dqm = dq2 / (dql + dqr);
  if (dq2 <= 0.0) dqm = 0.0;

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + dqm;
  qr_i = q_i - dqm;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ArtemisUtils::PLM_G()
//! \brief General PLM routine for non-uniform or non-Cartesian geometries. See Mignone
//! (2013).
KOKKOS_INLINE_FUNCTION
void PLM_G(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
           Real &qr_i, const Real x_im1, const Real x_i, const Real x_ip1,
           const Real xf[2], const Real dx) {
  // compute L/R slopes
  const Real dql = (q_i - q_im1) * dx / (x_i - x_im1);
  const Real dqr = (q_ip1 - q_i) * dx / (x_ip1 - x_i);

  // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
  const Real dq2 = dql * dqr;
  const Real cr = (x_ip1 - x_i) / (xf[1] - x_i);
  const Real cl = (x_i - x_im1) / (x_i - xf[0]);
  const Real dqm = (dq2 <= 0.0) ? 0.0
                                : dq2 * (cr * dql + cl * dqr) /
                                      (dql * dql + dqr * dqr + dq2 * (cl + cr - 2.0));

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + dqm * (xf[1] - x_i) / dx;
  qr_i = q_i - dqm * (x_i - xf[0]) / dx;
  return;
}

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm, X1DIR, ...>
//! \brief The piecewise linear reconstruction method in the X1 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm, X1DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql,
                                    parthenon::ScratchPad2D<Real> &qr) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            if constexpr (GEOM == Coordinates::cartesian) {
              PLM(q(b, n, k, j, i - 1), q(b, n, k, j, i), q(b, n, k, j, i + 1),
                  ql(n, i + 1), qr(n, i));
            } else {
              geometry::Coords<GEOM> coords_m(pco, k, j, i - 1);
              geometry::Coords<GEOM> coords_c(pco, k, j, i);
              geometry::Coords<GEOM> coords_p(pco, k, j, i + 1);
              const Real xvm = coords_m.x1v();
              const Real xvc = coords_c.x1v();
              const Real xvp = coords_p.x1v();
              const Real dx = coords_c.GetCellWidthX1();
              PLM_G(q(b, n, k, j, i - 1), q(b, n, k, j, i), q(b, n, k, j, i + 1),
                    ql(n, i + 1), qr(n, i), xvm, xvc, xvp, coords_c.bnds.x1, dx);
            }
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm, X2DIR, ...>
//! \brief The piecewise linear reconstruction method in the X2 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm, X2DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_jp1,
                                    parthenon::ScratchPad2D<Real> &qr_j) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            if constexpr (GEOM == Coordinates::cartesian) {
              PLM(q(b, n, k, j - 1, i), q(b, n, k, j, i), q(b, n, k, j + 1, i),
                  ql_jp1(n, i), qr_j(n, i));
            } else {
              geometry::Coords<GEOM> coords_m(pco, k, j - 1, i);
              geometry::Coords<GEOM> coords_c(pco, k, j, i);
              geometry::Coords<GEOM> coords_p(pco, k, j + 1, i);
              const Real xvm = coords_m.x2v();
              const Real xvc = coords_c.x2v();
              const Real xvp = coords_p.x2v();
              const Real dx = coords_c.GetCellWidthX2();
              PLM_G(q(b, n, k, j - 1, i), q(b, n, k, j, i), q(b, n, k, j + 1, i),
                    ql_jp1(n, i), qr_j(n, i), xvm, xvc, xvp, coords_c.bnds.x2, dx);
            }
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm, X3DIR, ...>
//! \brief The piecewise linear reconstruction method in the X3 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm, X3DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_kp1,
                                    parthenon::ScratchPad2D<Real> &qr_k) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            if constexpr (GEOM == Coordinates::cartesian) {
              PLM(q(b, n, k - 1, j, i), q(b, n, k, j, i), q(b, n, k + 1, j, i),
                  ql_kp1(n, i), qr_k(n, i));
            } else {
              geometry::Coords<GEOM> coords_m(pco, k - 1, j, i);
              geometry::Coords<GEOM> coords_c(pco, k, j, i);
              geometry::Coords<GEOM> coords_p(pco, k + 1, j, i);
              const Real xvm = coords_m.x3v();
              const Real xvc = coords_c.x3v();
              const Real xvp = coords_p.x3v();
              const Real dx = coords_c.GetCellWidthX3();
              PLM_G(q(b, n, k - 1, j, i), q(b, n, k, j, i), q(b, n, k + 1, j, i),
                    ql_kp1(n, i), qr_k(n, i), xvm, xvc, xvp, coords_c.bnds.x3, dx);
            }
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_PLM_HPP_
