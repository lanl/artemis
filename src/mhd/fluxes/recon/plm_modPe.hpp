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
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef UTILS_FLUXES_RECONSTRUCTION_PLM_MODPE_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_PLM_MODPE_HPP_

// Artemis includes
#include "artemis.hpp"
#include "utils/fluxes/reconstruction/slope_limiter.hpp"
#include "mhd/fluxes/recon/plm_rho.hpp"

// NOTE(PDMM): The following is taken directly from the open-source AthenaK software, and
// adapted for Parthenon/Artemis by PDM

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn ArtemisUtils::PLM_MODPE()
//! \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.
KOKKOS_INLINE_FUNCTION
void PLM_MODPE(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
         Real &qr_i, const Real fr, const TVDType TVD_type = TVDType::Original) {
  // compute L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  // YH: apply TVD slope e.g., TVD
  Real dqm = 0.5 * ApplyTVD(TVD_type, dql, dqr);

  //dqm *= fr; YH: no density-dependent slope limiting first
  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + dqm;
  qr_i = q_i - dqm;
  return;
}

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm_modPe, X1DIR, ...>
//! \brief The piecewise linear reconstruction method in the X1 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm_modPe, X1DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql,
                                    parthenon::ScratchPad2D<Real> &qr,
				    const TVDType TVD_type) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      TVDType tvdtype = (n==q.GetUpperBound(b)) ? TVDType::Superbee : TVD_type;
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
	    const Real fr = fr_Scurve(q(b, 0, k, j, i));
            PLM_MODPE(q(b, n, k, j, i - 1), q(b, n, k, j, i), q(b, n, k, j, i + 1),
                  ql(n, i + 1), qr(n, i), fr, tvdtype);
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm_modPe, X2DIR, ...>
//! \brief The piecewise linear reconstruction method in the X2 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm_modPe, X2DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_jp1,
                                    parthenon::ScratchPad2D<Real> &qr_j,
				    const TVDType TVD_type) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      TVDType tvdtype = (n==q.GetUpperBound(b)) ? TVDType::Superbee : TVD_type;
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
	    const Real fr = fr_Scurve(q(b, 0, k, j, i));
            PLM_MODPE(q(b, n, k, j - 1, i), q(b, n, k, j, i), q(b, n, k, j + 1, i),
                ql_jp1(n, i), qr_j(n, i), fr, tvdtype);
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::plm_modPe, X3DIR, ...>
//! \brief The piecewise linear reconstruction method in the X3 direction
template <Coordinates GEOM>
class Reconstruction<ReconstructionMethod::plm_modPe, X3DIR, GEOM> {
 public:
  template <typename V>
  KOKKOS_INLINE_FUNCTION void apply(parthenon::team_mbr_t const &member, const int b,
                                    const int k, const int j, const int il, const int iu,
                                    const V &q, parthenon::ScratchPad2D<Real> &ql_kp1,
                                    parthenon::ScratchPad2D<Real> &qr_k,
				    const TVDType TVD_type) const {
    auto &pco = q.GetCoordinates(b);
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      TVDType tvdtype = (n==q.GetUpperBound(b)) ? TVDType::Superbee : TVD_type;
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
	    const Real fr = fr_Scurve(q(b, 0, k, j, i));
            PLM_MODPE(q(b, n, k - 1, j, i), q(b, n, k, j, i), q(b, n, k + 1, j, i),
                ql_kp1(n, i), qr_k(n, i), fr, tvdtype);
          });
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_PLM_MODPE_HPP_
