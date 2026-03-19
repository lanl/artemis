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
#ifndef UTILS_FLUXES_RECONSTRUCTION_WENOMZ_HPP_
#define UTILS_FLUXES_RECONSTRUCTION_WENOMZ_HPP_

// Artemis includes
#include "artemis.hpp"

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! WENO-MZ reconstruction
//!
//! REFERENCES:
//! Wang Y., Zhao K., Yuan L., "A modified fifth-order WENO-Z scheme based on the
//! weights of the reformulated adaptive order WENO scheme"
//! Int J Numer Meth Fluids. 2024;96:1631–1652
//!
//! \fn ArtemisUtils::WENOMZ5()
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.
KOKKOS_INLINE_FUNCTION
void WENOMZ5(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
            const Real &q_ip2, Real &ql_ip1, Real &qr_i) {

  // Smooth WENO weights: See Jiang & Shu 1996

  constexpr Real weno_beta_coeff_0 = 13. / 12.;
  constexpr Real weno_beta_coeff_1  = 0.25;
  constexpr Real weno_beta_coeff_4  = 1. / 12.;

  const std::array<Real,4> beta{
            weno_beta_coeff_0 * SQR(q_im2 - 2 * q_im1 + q_i) +
            weno_beta_coeff_1 * SQR(q_im2 - 4 * q_im1 + 3 * q_i),
            weno_beta_coeff_0 * SQR(q_im1 - 2 * q_i + q_ip1) +
            weno_beta_coeff_1 * SQR(q_im1 + q_ip1),
            weno_beta_coeff_0 * SQR(q_i - 2 * q_ip1 + q_ip2) +
            weno_beta_coeff_1 * SQR(3 * q_i - 4 * q_ip1 + q_ip2), 
            weno_beta_coeff_4 * SQR(q_im1 - 2 * q_i + q_ip1)};

  Real tau_5 = std::abs(beta[0] - beta[2]);
  Real r = (std::abs(beta[2] - beta[1]) + Fuzz<Real>()) / (std::abs(beta[0] - beta[1]) + Fuzz<Real>());
  Real t0 = 1.0 + r;
  Real t2 = 1.0 + 1.0 / r;
  Real eta = tau_5 * SQR(SQR(tau_5 / (std::max(beta[0], beta[2]) + Fuzz<Real>()) ));


  const std::array<Real,3> indicator{
                         eta/(beta[0] + Fuzz<Real>()) + (tau_5 - eta)/(t0 * beta[3] + Fuzz<Real>()),
                         eta/(beta[1] + Fuzz<Real>()) + (tau_5 - eta)/(2. * beta[3] + Fuzz<Real>()),
                         eta/(beta[2] + Fuzz<Real>()) + (tau_5 - eta)/(t2 * beta[3] + Fuzz<Real>())};

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
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] = 2.0 * q_ip2 - 7.0 * q_ip1 + 11.0 * q_i;
  f[1] = -1.0 * q_ip1 + 5.0 * q_i   + 2.0 * q_im1;
  f[2] = 2.0 * q_i   + 5.0 * q_im1 -      q_im2;

  alpha[0] = 0.1 + 0.1*indicator[2];
  alpha[2] = 0.3 + 0.3*indicator[0];

  alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

  qr_i = (f[0] * alpha[0] + f[1] * alpha[1] + f[2] * alpha[2]) / alpha_sum;

  return;
}
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::wenomz, X1DIR, ...>
//! \brief The WENO-MZ method in the X1 direction
template <Coordinates GEOM>
struct Reconstruction<ReconstructionMethod::wenomz, X1DIR, GEOM> {
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
             const int b, const int k, const int j, const int il, const int iu,
             const V1 &q, const V2 &vg, parthenon::ScratchPad2D<Real> &ql,
             parthenon::ScratchPad2D<Real> &qr) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            WENOMZ5(q(b, n, k, j, i - 2), q(b, n, k, j, i - 1), q(b, n, k, j, i),
                 q(b, n, k, j, i + 1), q(b, n, k, j, i + 2), ql(n, i + 1), qr(n, i));
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::wenomz, X2DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X2 direction
template <Coordinates GEOM>
struct Reconstruction<ReconstructionMethod::wenomz, X2DIR, GEOM> {
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
             const int b, const int k, const int j, const int il, const int iu,
             const V1 &q, const V2 &vg, parthenon::ScratchPad2D<Real> &ql_jp1,
             parthenon::ScratchPad2D<Real> &qr_j) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            WENOMZ5(q(b, n, k, j - 2, i), q(b, n, k, j - 1, i), q(b, n, k, j, i),
                 q(b, n, k, j + 1, i), q(b, n, k, j + 2, i), ql_jp1(n, i), qr_j(n, i));
          });
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::Reconstruction<RSolver::wenomz, X3DIR, ...>
//! \brief The piecewise parabolic reconstruction method in the X3 direction
template <Coordinates GEOM>
struct Reconstruction<ReconstructionMethod::wenomz, X3DIR, GEOM> {
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void
  operator()(parthenon::team_mbr_t const &member, const geometry::CoordParams &cpars,
             const int b, const int k, const int j, const int il, const int iu,
             const V1 &q, const V2 &vg, parthenon::ScratchPad2D<Real> &ql_kp1,
             parthenon::ScratchPad2D<Real> &qr_k) const {
    for (int n = q.GetLowerBound(b); n <= q.GetUpperBound(b); ++n) {
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            WENOMZ5(q(b, n, k - 2, j, i), q(b, n, k - 1, j, i), q(b, n, k, j, i),
                 q(b, n, k + 1, j, i), q(b, n, k + 2, j, i), ql_kp1(n, i), qr_k(n, i));
          });
    }
  }
};

template <Coordinates GEOM>
struct ReconGradient<GEOM, ReconstructionMethod::wenomz> {
  template <typename V>
  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  operator()(const geometry::CoordParams &cpars, const V &q,
             const std::array<Real, 3> &dx, const int multi_d, const int three_d,
             const int b, const int n, const int k, const int j, const int i) const {
    std::array<Real, 3> dqdx{0.0, 0.0, 0.0};
    Real wl = Null<Real>(), wr = Null<Real>();

    WENOMZ5(q(b, n, k, j, i - 2), q(b, n, k, j, i - 1), q(b, n, k, j, i),
         q(b, n, k, j, i + 1), q(b, n, k, j, i + 2), wl, wr);
    dqdx[0] = (wr - wl) / (2.0 * dx[0]);

    wl = Null<Real>(), wr = Null<Real>();
    WENOMZ5(q(b, n, k, j - 2 * multi_d, i), q(b, n, k, j - multi_d, i), q(b, n, k, j, i),
         q(b, n, k, j + multi_d, i), q(b, n, k, j + 2 * multi_d, i), wl, wr);
    dqdx[1] = (wr - wl) / (2.0 * dx[1]);

    wl = Null<Real>(), wr = Null<Real>();
    WENOMZ5(q(b, n, k - three_d, j, i), q(b, n, k - 2 * three_d, j, i), q(b, n, k, j, i),
         q(b, n, k + three_d, j, i), q(b, n, k + 2 * three_d, j, i), wl, wr);
    dqdx[2] = (wr - wl) / (2.0 * dx[2]);

    return dqdx;
  }
};

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RECONSTRUCTION_WENOMZ_HPP_
