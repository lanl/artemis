//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
#ifndef UTILS_FLUXES_RECONSTRUCTION_WENOZ_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_WENOZ_HPP_


// Artemis includes
#include "artemis.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn ArtemisUtils::WENOZ5()
//! \brief Improved WENO reconstruction from Borges et al. 2008, Castro+ 2011.  Returns
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.
KOKKOS_INLINE_FUNCTION
void WENOZ5(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
            const Real &q_ip2, Real &ql_ip1, Real &qr_i) {

  // smoothness indicators for each trial stencil [Jiang & Shu 1996]
  constexpr Real weno_beta_coeff_0 = 13. / 12.;
  constexpr Real weno_beta_coeff_1  = 0.25;
  const std::array<Real,3> beta{
            weno_beta_coeff_0 * SQR(q_im2 - 2 * q_im1 + q_i) +
            weno_beta_coeff_1 * SQR(q_im2 - 4 * q_im1 + 3 * q_i),
            weno_beta_coeff_0 * SQR(q_im1 - 2 * q_i + q_ip1) +
            weno_beta_coeff_1 * SQR(q_im1 + q_ip1),
            weno_beta_coeff_0 * SQR(q_i - 2 * q_ip1 + q_ip2) +
            weno_beta_coeff_1 * SQR(3 * q_i - 4 * q_ip1 + q_ip2)};

  const Real tau5 = std::abs(beta[0] - beta[2]); // [Borges+ 2008]

  // [Castro, Costa, & Don 2011]
  const std::array<Real,3> indicator{
                         SQR(tau5 / (beta[0] + Fuzz<Real>())),
                         SQR(tau5 / (beta[1] + Fuzz<Real>())),
                         SQR(tau5 / (beta[2] + Fuzz<Real>()))};

  // compute qL_ip1
  std::array<Real,3> f{2.0 * q_im2 - 7.0 * q_im1 + 11.0 * q_i,
                                  -1.0 * q_im1 + 5.0 * q_i + 2.0 * q_ip1,
                                   2.0 * q_i + 5.0 * q_ip1 - q_ip2};

  std::array<Real,3> alpha{
   0.1 * (1.0 + indicator[0]),
   0.6 * (1.0 + indicator[1]),
   0.3 * (1.0 + indicator[2])};
  Real alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

  ql_ip1 = (f[0] * alpha[0] + f[1] * alpha[1] + f[2] * alpha[2]) / alpha_sum;

  // compute qR_i
  f[0] = 2.0 * q_ip2 - 7.0 * q_ip1 + 11.0 * q_i;
  f[1] = -1.0 * q_ip1 + 5.0 * q_i + 2.0 * q_im1;
  f[2] = 2.0 * q_i + 5.0 * q_im1 - q_im2;

  alpha[0] = 0.1 * (1.0 + indicator[2]);
  alpha[2] = 0.3 * (1.0 + indicator[0]);
  alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

  qr_i = (f[0] * alpha[0] + f[1] * alpha[1] + f[2] * alpha[2]) / alpha_sum;

  return;
}

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::wenoz, X1DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X1 direction
template <Coordinates GEOM>
struct Reconstruction<ReconstructionMethod::wenoz, X1DIR, GEOM> {
  template <typename V>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
             const int b, const int k, const int j, const int il, const int iu,
             const V &q, parthenon::ScratchPad2D<Real> &ql,
             parthenon::ScratchPad2D<Real> &qr) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            WENOZ5(q(b, n, k, j, i - 2), q(b, n, k, j, i - 1), q(b, n, k, j, i),
                   q(b, n, k, j, i + 1), q(b, n, k, j, i + 2), ql(n, i + 1), qr(n, i));
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::wenoz, X2DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X2 direction
template <Coordinates GEOM>
struct Reconstruction<ReconstructionMethod::wenoz, X2DIR, GEOM> {
  template <typename V>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
             const int b, const int k, const int j, const int il, const int iu,
             const V &q, parthenon::ScratchPad2D<Real> &ql_jp1,
             parthenon::ScratchPad2D<Real> &qr_j) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            WENOZ5(q(b, n, k, j - 2, i), q(b, n, k, j - 1, i), q(b, n, k, j, i),
                   q(b, n, k, j + 1, i), q(b, n, k, j + 2, i), ql_jp1(n, i), qr_j(n, i));
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::wenoz, X3DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X3 direction
template <Coordinates GEOM>
struct Reconstruction<ReconstructionMethod::wenoz, X3DIR, GEOM> {
  template <typename V>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
             const int b, const int k, const int j, const int il, const int iu,
             const V &q, parthenon::ScratchPad2D<Real> &ql_kp1,
             parthenon::ScratchPad2D<Real> &qr_k) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            WENOZ5(q(b, n, k - 2, j, i), q(b, n, k - 1, j, i), q(b, n, k, j, i),
                   q(b, n, k + 1, j, i), q(b, n, k + 2, j, i), ql_kp1(n, i), qr_k(n, i));
          });
    }
  }
};

template <Coordinates GEOM>
struct ReconGradient<GEOM, ReconstructionMethod::wenoz> {
  template <typename V>
  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  operator()(const geometry::CoordParams &cpars, const V &q,
             const std::array<Real, 3> &dx, const int multi_d, const int three_d,
             const int b, const int n, const int k, const int j, const int i) const {
    std::array<Real, 3> dqdx{0.0, 0.0, 0.0};
    Real wl = Null<Real>(), wr = Null<Real>();

    WENOZ5(q(b, n, k, j, i - 2), q(b, n, k, j, i - 1), q(b, n, k, j, i),
           q(b, n, k, j, i + 1), q(b, n, k, j, i + 2), wl, wr);
    dqdx[0] = (wr - wl) / (2.0 * dx[0]);

    wl = Null<Real>(), wr = Null<Real>();
    WENOZ5(q(b, n, k, j - 2 * multi_d, i), q(b, n, k, j - multi_d, i), q(b, n, k, j, i),
           q(b, n, k, j + multi_d, i), q(b, n, k, j + 2 * multi_d, i), wl, wr);
    dqdx[1] = (wr - wl) / (2.0 * dx[1]);

    wl = Null<Real>(), wr = Null<Real>();
    WENOZ5(q(b, n, k - three_d, j, i), q(b, n, k - 2 * three_d, j, i), q(b, n, k, j, i),
           q(b, n, k + three_d, j, i), q(b, n, k + 2 * three_d, j, i), wl, wr);
    dqdx[2] = (wr - wl) / (2.0 * dx[2]);

    return dqdx;
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_WENOZ_HPP_
