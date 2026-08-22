//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights
// reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

// This file was modified in part with the assistance of generative AI.
#ifndef RADIATION_MOMENTS_MATTER_COUPLING_HPP_
#define RADIATION_MOMENTS_MATTER_COUPLING_HPP_

#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "moments.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/opacity/opacity.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::MeanOpacity;
using ArtemisUtils::MeanScattering;
using ArtemisUtils::VI;
using ArtemisUtils::VNorm;

namespace Moments {

KOKKOS_INLINE_FUNCTION void
ComputeCouplingEnergyCoefficients(const Real sigp, const Real sigs, const Real g,
                                  const Real g2, const Real beta2, const Real bdbdp,
                                  const Real bdf, Real &ca, Real &cb, Real &cd) {
  // Algebraically equivalent to
  //   ca = g * (sigp + sigs - g2 * sigs * (1 + bdbdp));
  //   cb = g * sigp;
  //   cd = -g * bdf * (sigp + sigs - 2 * g2 * sigs);
  // but avoids subtracting nearly equal O(sigs) terms when beta << 1.
  ca = g * (sigp - g2 * sigs * (beta2 + bdbdp));
  cb = g * sigp;
  cd = g * bdf * (g2 * sigs * (1.0 + beta2) - sigp);
}

KOKKOS_INLINE_FUNCTION bool ComputeCouplingEquilibriumEnergy(const Real E0, const Real B,
                                                             const Real ca, const Real cb,
                                                             const Real cd, Real &Eeq) {
  // The implicit radiation-energy equation is
  //   (E - E0) + ca E - cb B + cd = 0.
  // Evaluate its solution using coefficient ratios.  This remains accurate in
  // the optically thick limit, where ca and cb may be O(1e10) and directly
  // evaluating ca*E - cb*B loses all useful digits near LTE.
  const Real denom = 1.0 + ca;
  if (std::abs(denom) <= Fuzz()) return false;
  const Real inv_denom = 1.0 / denom;
  Eeq = E0 * inv_denom + (cb * inv_denom) * B - cd * inv_denom;
  return true;
}

KOKKOS_INLINE_FUNCTION std::array<Real, 3>
ProjectCouplingFluxToEnergy(const std::array<Real, 3> &Fin, const Real E) {
  const Real fmag = VNorm(Fin);
  if (E <= 0.0) return {0.0, 0.0, 0.0};
  if (fmag <= E || fmag <= Fuzz()) return Fin;
  const Real scale = E / fmag;
  return {scale * Fin[0], scale * Fin[1], scale * Fin[2]};
}

KOKKOS_INLINE_FUNCTION bool SolveCouplingDense3x3(Real A[3][3], Real rhs[3],
                                                  std::array<Real, 3> &x) {
  // Small partial-pivoting Gaussian elimination used by the local momentum
  // Newton solve.  The system is only 3x3, so forming and factorizing it is
  // cheaper and more robust than attempting an analytic inverse.
  for (int col = 0; col < 3; ++col) {
    int pivot = col;
    Real pivot_abs = std::abs(A[col][col]);
    for (int row = col + 1; row < 3; ++row) {
      const Real candidate = std::abs(A[row][col]);
      if (candidate > pivot_abs) {
        pivot = row;
        pivot_abs = candidate;
      }
    }
    if (pivot_abs <= Fuzz()) return false;
    if (pivot != col) {
      for (int j = col; j < 3; ++j) {
        const Real tmp = A[col][j];
        A[col][j] = A[pivot][j];
        A[pivot][j] = tmp;
      }
      const Real tmp = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = tmp;
    }

    const Real inv_pivot = 1.0 / A[col][col];
    for (int row = col + 1; row < 3; ++row) {
      const Real factor = A[row][col] * inv_pivot;
      A[row][col] = 0.0;
      for (int j = col + 1; j < 3; ++j)
        A[row][j] -= factor * A[col][j];
      rhs[row] -= factor * rhs[col];
    }
  }

  for (int row = 2; row >= 0; --row) {
    Real value = rhs[row];
    for (int j = row + 1; j < 3; ++j)
      value -= A[row][j] * x[j];
    if (std::abs(A[row][row]) <= Fuzz()) return false;
    x[row] = value / A[row][row];
  }
  return true;
}

template <Closure CLOSURE>
KOKKOS_INLINE_FUNCTION bool EvaluateCouplingMomentumEnergyResidualBeta(
    const std::array<Real, 3> &beta, const std::array<Real, 3> &beta0,
    const std::array<Real, 3> &Fr0, const Real E0, const Real efloor, const Real B,
    const Real dEg, const Real Q, const Real dens, const Real eref, const Real c,
    const Real chat, const Real sigp, const Real sigs, const Real energy_floor_scale,
    std::array<Real, 3> &F, Real &E, Real &dEk, std::array<Real, 3> &residual,
    Real &residual_norm) {
  // Enforce gas+radiation pseudo-momentum conservation algebraically:
  //   rho c (beta-beta0) + (eref/chat) (F-Fr0) = 0,
  // where F is normalized by c*eref.  This relation is the source of the
  // instability in a direct Picard update when eref/(rho*c*chat) is large.
  const Real mu = eref / (dens * c * chat);

  const Real beta2 = SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);
  if (beta2 >= 1.0) return false;

  std::array<Real, 3> delta_F{0.0, 0.0, 0.0};
  Real delta_F2 = 0.0;
  Real beta0_dot_delta_F = 0.0;
  for (int d = 0; d < 3; ++d) {
    F[d] = Fr0[d] + (beta0[d] - beta[d]) / mu;
    delta_F[d] = F[d] - Fr0[d];
    delta_F2 += SQR(delta_F[d]);
    beta0_dot_delta_F += beta0[d] * delta_F[d];
  }

  // From beta-beta0 = -mu*(F-Fr0), evaluate the kinetic-energy
  // change without subtracting two nearly equal beta^2 values:
  //   dEk = (c/chat)[-beta0.dF + 0.5*mu*|dF|^2].
  dEk = c / chat * (-beta0_dot_delta_F + 0.5 * mu * delta_F2);
  E = E0 + chat / c * (Q - dEg - dEk);
  const Real floor_slop =
      128.0 * Eps() * std::max(1.0, std::max(std::abs(E0), std::abs(E)));
  if (E < efloor - floor_slop || E <= 0.0) return false;
  E = std::max(E, efloor);

  const Real fmag = VNorm(F);
  const Real realizability_slop = 256.0 * Eps() * std::max(1.0, E);
  if (fmag > E + realizability_slop) return false;

  const Real g2 = 1.0 / (1.0 - beta2);
  const Real g = std::sqrt(g2);
  const auto fedd = EddingtonTensor<CLOSURE>({F[0] / E, F[1] / E, F[2] / E});
  const std::array<Real, 3> bdp{
      beta[0] * fedd[TensIdx::X11] + beta[1] * fedd[TensIdx::X12] +
          beta[2] * fedd[TensIdx::X13],
      beta[0] * fedd[TensIdx::X12] + beta[1] * fedd[TensIdx::X22] +
          beta[2] * fedd[TensIdx::X23],
      beta[0] * fedd[TensIdx::X13] + beta[1] * fedd[TensIdx::X23] +
          beta[2] * fedd[TensIdx::X33]};
  const Real bdbdp = beta[0] * bdp[0] + beta[1] * bdp[1] + beta[2] * bdp[2];

  const Real sigf = sigp + sigs;
  const Real a = g * sigf;
  const Real bcoef = 2.0 * g2 * g * sigs;
  const Real d1 = g * (sigp * B + g2 * sigs * (1.0 + bdbdp) * E);
  const Real d2 = g * sigf * E;
  const std::array<Real, 3> rhs{Fr0[0] + d1 * beta[0] + d2 * bdp[0],
                                Fr0[1] + d1 * beta[1] + d2 * bdp[1],
                                Fr0[2] + d1 * beta[2] + d2 * bdp[2]};
  const auto Fsolve = SolveRadFlux(1.0 + a, bcoef, beta, rhs);
  const auto Ftarget = ProjectCouplingFluxToEnergy(Fsolve, E);

  const Real scale =
      std::max(energy_floor_scale,
               std::max(E, std::max(VNorm(Fr0), std::max(fmag, VNorm(Ftarget)))));
  residual_norm = 0.0;
  for (int d = 0; d < 3; ++d) {
    residual[d] = F[d] - Ftarget[d];
    residual_norm = std::max(residual_norm, std::abs(residual[d]) / scale);
  }
  return true;
}

template <Closure CLOSURE>
KOKKOS_INLINE_FUNCTION bool SolveCouplingMomentumEnergyBeta(
    const std::array<Real, 3> &beta0, const std::array<Real, 3> &Fr0, const Real E0,
    const Real efloor, const Real B, const Real dEg, const Real Q, const Real dens,
    const Real eref, const Real c, const Real chat, const Real sigp, const Real sigs,
    const Real energy_floor_scale, const Real tolerance, std::array<Real, 3> &beta,
    std::array<Real, 3> &F, Real &E, Real &dEk, Real &momentum_error, int &iterations) {
  const Real mu = eref / (dens * c * chat);

  const Real solve_tol = std::max(tolerance, 64.0 * Eps());
  constexpr int max_iterations = 32;
  constexpr int max_line_search = 20;
  const Real fd_factor = std::pow(Eps(), 1.0 / 3.0);

  std::array<Real, 3> residual{0.0, 0.0, 0.0};
  std::array<Real, 3> best_beta = beta;
  std::array<Real, 3> best_F = F;
  Real best_E = E;
  Real best_dEk = dEk;
  Real best_error = Big();

  for (iterations = 1; iterations <= max_iterations; ++iterations) {
    Real current_error = Big();
    if (!EvaluateCouplingMomentumEnergyResidualBeta<CLOSURE>(
            beta, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
            energy_floor_scale, F, E, dEk, residual, current_error))
      break;

    if (current_error < best_error) {
      best_error = current_error;
      best_beta = beta;
      best_F = F;
      best_E = E;
      best_dEk = dEk;
    }
    if (current_error <= solve_tol) {
      momentum_error = current_error;
      return true;
    }

    Real J[3][3]{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    bool jacobian_valid = true;
    for (int col = 0; col < 3; ++col) {
      const Real h =
          std::max(32.0 * Eps(), fd_factor * std::max(1.0e-3, std::abs(beta[col])));
      auto beta_plus = beta;
      auto beta_minus = beta;
      beta_plus[col] += h;
      beta_minus[col] -= h;

      std::array<Real, 3> Fplus, Fminus;
      std::array<Real, 3> Rplus, Rminus;
      Real Eplus = E;
      Real Eminus = E;
      Real dEkplus = dEk;
      Real dEkminus = dEk;
      Real err_plus = 0.0;
      Real err_minus = 0.0;
      const bool plus_valid = EvaluateCouplingMomentumEnergyResidualBeta<CLOSURE>(
          beta_plus, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
          energy_floor_scale, Fplus, Eplus, dEkplus, Rplus, err_plus);
      const bool minus_valid = EvaluateCouplingMomentumEnergyResidualBeta<CLOSURE>(
          beta_minus, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
          energy_floor_scale, Fminus, Eminus, dEkminus, Rminus, err_minus);

      if (plus_valid && minus_valid) {
        for (int row = 0; row < 3; ++row)
          J[row][col] = (Rplus[row] - Rminus[row]) / (2.0 * h);
      } else if (plus_valid) {
        for (int row = 0; row < 3; ++row)
          J[row][col] = (Rplus[row] - residual[row]) / h;
      } else if (minus_valid) {
        for (int row = 0; row < 3; ++row)
          J[row][col] = (residual[row] - Rminus[row]) / h;
      } else {
        jacobian_valid = false;
        break;
      }
    }

    std::array<Real, 3> step{0.0, 0.0, 0.0};
    if (jacobian_valid) {
      Real rhs[3]{-residual[0], -residual[1], -residual[2]};
      jacobian_valid = SolveCouplingDense3x3(J, rhs, step);
    }

    // If the numerical Jacobian is singular, use the analytically motivated
    // damped Picard direction.  In the diffusion limit the unstable Picard
    // gain is approximately -(4/3) mu E; the denominator below removes that
    // stiffness while retaining the correct fixed point.
    if (!jacobian_valid) {
      const Real relax = 1.0 / (1.0 + (4.0 / 3.0) * mu * std::max(E, energy_floor_scale));
      for (int d = 0; d < 3; ++d)
        step[d] = relax * mu * residual[d];
    }

    const Real step_mag = std::sqrt(SQR(step[0]) + SQR(step[1]) + SQR(step[2]));
    if (step_mag <= Fuzz()) break;
    if (step_mag > 0.25) {
      const Real scale = 0.25 / step_mag;
      for (int d = 0; d < 3; ++d)
        step[d] *= scale;
    }

    bool accepted = false;
    Real alpha = 1.0;
    for (int ls = 0; ls < max_line_search; ++ls) {
      std::array<Real, 3> beta_trial{beta[0] + alpha * step[0], beta[1] + alpha * step[1],
                                     beta[2] + alpha * step[2]};
      std::array<Real, 3> Ftrial, Rtrial;
      Real Etrial = E;
      Real dEktrial = dEk;
      Real trial_error = Big();
      const bool trial_valid = EvaluateCouplingMomentumEnergyResidualBeta<CLOSURE>(
          beta_trial, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
          energy_floor_scale, Ftrial, Etrial, dEktrial, Rtrial, trial_error);
      if (trial_valid && trial_error < current_error * (1.0 - 1.0e-4 * alpha)) {
        beta = beta_trial;
        F = Ftrial;
        E = Etrial;
        dEk = dEktrial;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }

    if (!accepted) {
      // Retry with the stiffness-aware Picard direction even when the Newton
      // Jacobian existed but produced a poor globalization step.
      const Real relax = 1.0 / (1.0 + (4.0 / 3.0) * mu * std::max(E, energy_floor_scale));
      for (int d = 0; d < 3; ++d)
        step[d] = relax * mu * residual[d];
      alpha = 1.0;
      for (int ls = 0; ls < max_line_search; ++ls) {
        std::array<Real, 3> beta_trial{beta[0] + alpha * step[0],
                                       beta[1] + alpha * step[1],
                                       beta[2] + alpha * step[2]};
        std::array<Real, 3> Ftrial, Rtrial;
        Real Etrial = E;
        Real dEktrial = dEk;
        Real trial_error = Big();
        const bool trial_valid = EvaluateCouplingMomentumEnergyResidualBeta<CLOSURE>(
            beta_trial, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp,
            sigs, energy_floor_scale, Ftrial, Etrial, dEktrial, Rtrial, trial_error);
        if (trial_valid && trial_error < current_error) {
          beta = beta_trial;
          F = Ftrial;
          E = Etrial;
          dEk = dEktrial;
          accepted = true;
          break;
        }
        alpha *= 0.5;
      }
    }
    if (!accepted) break;
  }

  beta = best_beta;
  F = best_F;
  E = best_E;
  dEk = best_dEk;
  momentum_error = best_error;
  return best_error <= solve_tol;
}

// In the high-matter-inertia limit mu = eref/(rho*c*chat) << 1, the
// physically resolvable momentum unknown is the radiation-flux change, not
// beta. A finite flux correction corresponds to a beta correction of order
// mu*dF, which can be smaller than one floating-point ulp of beta. Solving in
// beta then cannot represent the root even though F remains well resolved.
template <Closure CLOSURE>
KOKKOS_INLINE_FUNCTION bool EvaluateCouplingMomentumEnergyResidualFlux(
    const std::array<Real, 3> &F, const std::array<Real, 3> &beta0,
    const std::array<Real, 3> &Fr0, const Real E0, const Real efloor, const Real B,
    const Real dEg, const Real Q, const Real dens, const Real eref, const Real c,
    const Real chat, const Real sigp, const Real sigs, const Real energy_floor_scale,
    std::array<Real, 3> &beta, Real &E, Real &dEk, std::array<Real, 3> &residual,
    Real &residual_norm) {
  const Real mu = eref / (dens * c * chat);
  std::array<Real, 3> delta_F{0.0, 0.0, 0.0};
  Real delta_F2 = 0.0;
  Real beta0_dot_delta_F = 0.0;
  for (int d = 0; d < 3; ++d) {
    delta_F[d] = F[d] - Fr0[d];
    beta[d] = beta0[d] - mu * delta_F[d];
    delta_F2 += SQR(delta_F[d]);
    beta0_dot_delta_F += beta0[d] * delta_F[d];
  }

  const Real beta2 = SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);
  if (beta2 >= 1.0) return false;

  // This form remains accurate even when beta-beta0 is below the resolution
  // of a stored velocity component.
  dEk = c / chat * (-beta0_dot_delta_F + 0.5 * mu * delta_F2);
  E = E0 + chat / c * (Q - dEg - dEk);
  const Real floor_slop =
      128.0 * Eps() * std::max(1.0, std::max(std::abs(E0), std::abs(E)));
  if (E < efloor - floor_slop || E <= 0.0) return false;
  E = std::max(E, efloor);

  const Real fmag = VNorm(F);
  const Real realizability_slop = 256.0 * Eps() * std::max(1.0, E);
  if (fmag > E + realizability_slop) return false;

  const Real g2 = 1.0 / (1.0 - beta2);
  const Real g = std::sqrt(g2);
  const auto fedd = EddingtonTensor<CLOSURE>({F[0] / E, F[1] / E, F[2] / E});
  const std::array<Real, 3> bdp{
      beta[0] * fedd[TensIdx::X11] + beta[1] * fedd[TensIdx::X12] +
          beta[2] * fedd[TensIdx::X13],
      beta[0] * fedd[TensIdx::X12] + beta[1] * fedd[TensIdx::X22] +
          beta[2] * fedd[TensIdx::X23],
      beta[0] * fedd[TensIdx::X13] + beta[1] * fedd[TensIdx::X23] +
          beta[2] * fedd[TensIdx::X33]};
  const Real bdbdp = beta[0] * bdp[0] + beta[1] * bdp[1] + beta[2] * bdp[2];

  const Real sigf = sigp + sigs;
  const Real a = g * sigf;
  const Real bcoef = 2.0 * g2 * g * sigs;
  const Real d1 = g * (sigp * B + g2 * sigs * (1.0 + bdbdp) * E);
  const Real d2 = g * sigf * E;
  const std::array<Real, 3> rhs{Fr0[0] + d1 * beta[0] + d2 * bdp[0],
                                Fr0[1] + d1 * beta[1] + d2 * bdp[1],
                                Fr0[2] + d1 * beta[2] + d2 * bdp[2]};
  const auto Fsolve = SolveRadFlux(1.0 + a, bcoef, beta, rhs);
  const auto Ftarget = ProjectCouplingFluxToEnergy(Fsolve, E);

  const Real scale =
      std::max(energy_floor_scale,
               std::max(E, std::max(VNorm(Fr0), std::max(fmag, VNorm(Ftarget)))));
  residual_norm = 0.0;
  for (int d = 0; d < 3; ++d) {
    residual[d] = F[d] - Ftarget[d];
    residual_norm = std::max(residual_norm, std::abs(residual[d]) / scale);
  }
  return true;
}

template <Closure CLOSURE>
KOKKOS_INLINE_FUNCTION bool SolveCouplingMomentumEnergyFlux(
    const std::array<Real, 3> &beta0, const std::array<Real, 3> &Fr0, const Real E0,
    const Real efloor, const Real B, const Real dEg, const Real Q, const Real dens,
    const Real eref, const Real c, const Real chat, const Real sigp, const Real sigs,
    const Real energy_floor_scale, const Real tolerance, std::array<Real, 3> &beta,
    std::array<Real, 3> &F, Real &E, Real &dEk, Real &momentum_error, int &iterations) {
  const Real mu = eref / (dens * c * chat);
  const Real solve_tol = std::max(tolerance, 64.0 * Eps());
  constexpr int max_iterations = 32;
  constexpr int max_line_search = 20;
  const Real fd_factor = std::pow(Eps(), 1.0 / 3.0);

  std::array<Real, 3> residual{0.0, 0.0, 0.0};
  std::array<Real, 3> best_beta = beta;
  std::array<Real, 3> best_F = F;
  Real best_E = E;
  Real best_dEk = dEk;
  Real best_error = Big();

  for (iterations = 1; iterations <= max_iterations; ++iterations) {
    Real current_error = Big();
    if (!EvaluateCouplingMomentumEnergyResidualFlux<CLOSURE>(
            F, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
            energy_floor_scale, beta, E, dEk, residual, current_error))
      break;

    if (current_error < best_error) {
      best_error = current_error;
      best_beta = beta;
      best_F = F;
      best_E = E;
      best_dEk = dEk;
    }
    if (current_error <= solve_tol) {
      momentum_error = current_error;
      return true;
    }

    Real J[3][3]{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    bool jacobian_valid = true;
    for (int col = 0; col < 3; ++col) {
      const Real variable_scale =
          std::max(energy_floor_scale,
                   std::max(std::abs(F[col]), std::max(std::abs(Fr0[col]), 1.0e-6 * E)));
      // F is normalized by the local energy reference and can be much smaller
      // than unity. An absolute O(epsilon) perturbation would then be larger
      // than the correction implied by the requested relative residual.
      const Real h = std::max(64.0 * Eps() * variable_scale, fd_factor * variable_scale);
      auto Fplus = F;
      auto Fminus = F;
      Fplus[col] += h;
      Fminus[col] -= h;

      std::array<Real, 3> beta_plus, beta_minus;
      std::array<Real, 3> Rplus, Rminus;
      Real Eplus = E;
      Real Eminus = E;
      Real dEkplus = dEk;
      Real dEkminus = dEk;
      Real err_plus = 0.0;
      Real err_minus = 0.0;
      const bool plus_valid = EvaluateCouplingMomentumEnergyResidualFlux<CLOSURE>(
          Fplus, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
          energy_floor_scale, beta_plus, Eplus, dEkplus, Rplus, err_plus);
      const bool minus_valid = EvaluateCouplingMomentumEnergyResidualFlux<CLOSURE>(
          Fminus, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
          energy_floor_scale, beta_minus, Eminus, dEkminus, Rminus, err_minus);

      if (plus_valid && minus_valid) {
        for (int row = 0; row < 3; ++row)
          J[row][col] = (Rplus[row] - Rminus[row]) / (2.0 * h);
      } else if (plus_valid) {
        for (int row = 0; row < 3; ++row)
          J[row][col] = (Rplus[row] - residual[row]) / h;
      } else if (minus_valid) {
        for (int row = 0; row < 3; ++row)
          J[row][col] = (residual[row] - Rminus[row]) / h;
      } else {
        jacobian_valid = false;
        break;
      }
    }

    std::array<Real, 3> step{0.0, 0.0, 0.0};
    if (jacobian_valid) {
      Real rhs[3]{-residual[0], -residual[1], -residual[2]};
      jacobian_valid = SolveCouplingDense3x3(J, rhs, step);
    }
    if (!jacobian_valid) {
      for (int d = 0; d < 3; ++d)
        step[d] = -residual[d];
    }

    const Real step_mag = std::sqrt(SQR(step[0]) + SQR(step[1]) + SQR(step[2]));
    const Real state_scale =
        std::max(energy_floor_scale, std::max(E, std::max(VNorm(F), VNorm(Fr0))));
    // state_scale includes the radiation floor. Do not impose a unit-scale
    // absolute cutoff here: for E << 1 it can reject a resolvable correction
    // before the normalized momentum residual reaches its tolerance.
    if (step_mag <= 64.0 * Eps() * state_scale) break;
    if (step_mag > 0.25 * state_scale) {
      const Real scale = 0.25 * state_scale / step_mag;
      for (int d = 0; d < 3; ++d)
        step[d] *= scale;
    }

    bool accepted = false;
    Real alpha = 1.0;
    for (int ls = 0; ls < max_line_search; ++ls) {
      std::array<Real, 3> Ftrial{F[0] + alpha * step[0], F[1] + alpha * step[1],
                                 F[2] + alpha * step[2]};
      std::array<Real, 3> beta_trial, Rtrial;
      Real Etrial = E;
      Real dEktrial = dEk;
      Real trial_error = Big();
      const bool trial_valid = EvaluateCouplingMomentumEnergyResidualFlux<CLOSURE>(
          Ftrial, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
          energy_floor_scale, beta_trial, Etrial, dEktrial, Rtrial, trial_error);
      if (trial_valid && trial_error < current_error * (1.0 - 1.0e-4 * alpha)) {
        F = Ftrial;
        beta = beta_trial;
        E = Etrial;
        dEk = dEktrial;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }

    if (!accepted) {
      // A fixed-point step in F is well conditioned in this branch.
      for (int d = 0; d < 3; ++d)
        step[d] = -residual[d];
      alpha = 1.0;
      for (int ls = 0; ls < max_line_search; ++ls) {
        std::array<Real, 3> Ftrial{F[0] + alpha * step[0], F[1] + alpha * step[1],
                                   F[2] + alpha * step[2]};
        std::array<Real, 3> beta_trial, Rtrial;
        Real Etrial = E;
        Real dEktrial = dEk;
        Real trial_error = Big();
        const bool trial_valid = EvaluateCouplingMomentumEnergyResidualFlux<CLOSURE>(
            Ftrial, beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
            energy_floor_scale, beta_trial, Etrial, dEktrial, Rtrial, trial_error);
        if (trial_valid && trial_error < current_error) {
          F = Ftrial;
          beta = beta_trial;
          E = Etrial;
          dEk = dEktrial;
          accepted = true;
          break;
        }
        alpha *= 0.5;
      }
    }
    if (!accepted) break;
  }

  beta = best_beta;
  F = best_F;
  E = best_E;
  dEk = best_dEk;
  momentum_error = best_error;
  return best_error <= solve_tol;
}

template <Closure CLOSURE>
KOKKOS_INLINE_FUNCTION bool SolveCouplingMomentumEnergy(
    const std::array<Real, 3> &beta0, const std::array<Real, 3> &Fr0, const Real E0,
    const Real efloor, const Real B, const Real dEg, const Real Q, const Real dens,
    const Real eref, const Real c, const Real chat, const Real sigp, const Real sigs,
    const Real energy_floor_scale, const Real tolerance, std::array<Real, 3> &beta,
    std::array<Real, 3> &F, Real &E, Real &dEk, Real &momentum_error, int &iterations) {
  // Choose the nonlinear variable from the relative pseudo-inertia. For
  // mu <= 1 the gas velocity change can be much less resolvable than the
  // radiation-flux change, so solve in F. For mu > 1 solve in beta to avoid
  // magnifying flux perturbations into large velocity perturbations.
  const Real mu = eref / (dens * c * chat);
  if (mu <= 1.0) {
    return SolveCouplingMomentumEnergyFlux<CLOSURE>(
        beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
        energy_floor_scale, tolerance, beta, F, E, dEk, momentum_error, iterations);
  }
  return SolveCouplingMomentumEnergyBeta<CLOSURE>(
      beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp, sigs,
      energy_floor_scale, tolerance, beta, F, E, dEk, momentum_error, iterations);
}

template <typename EOSType, typename OpacityType, typename ScatteringType>
KOKKOS_INLINE_FUNCTION bool EvaluateCouplingInnerScalarResidual(
    const Real eg, const Real eg0, const Real dEk, const Real Q, const Real E0,
    const Real efloor, const Real Bfloor, const Real dens, const Real eref,
    const Real arad, const Real tfloor, const Real dt, const Real chat, const Real c,
    const Real g, const Real g2, const Real beta2, const Real bdbdp, const Real bdf,
    const EOSType &eos, const OpacityType &opacity, const ScatteringType &scattering,
    Real &E, Real &B, Real &residual) {
  // Exact reduced-speed-of-light total-energy conservation at fixed velocity:
  //   dEg + dEk + (c/chat) dEr = Q.
  E = E0 + chat / c * (Q - dEk - (eg - eg0));
  const Real floor_slop =
      64.0 * Eps() * std::max(1.0, std::max(std::abs(E0), std::abs(E)));
  if (E < efloor - floor_slop) return false;
  E = std::max(E, efloor);

  const Real eint = eg * eref / dens;
  const Real T = std::max(tfloor, eos.TemperatureFromDensityInternalEnergy(dens, eint));
  B = std::max(Bfloor, arad * SQR(SQR(T)) / eref);
  const Real sigp = chat * dt * opacity.PlanckMeanAbsorptionCoefficient(dens, T);
  const Real sigs =
      chat * dt * scattering.RosselandMeanTotalScatteringCoefficient(dens, T);
  Real ca = 0.0;
  Real cb = 0.0;
  Real cd = 0.0;
  Real Eeq = E;
  ComputeCouplingEnergyCoefficients(sigp, sigs, g, g2, beta2, bdbdp, bdf, ca, cb, cd);
  // A degenerate equilibrium linearization (1 + ca <= 0) has no well-defined
  // radiation-energy root; report the state as inadmissible rather than letting
  // the seed Eeq = E read as a zero source residual (false convergence).
  if (!ComputeCouplingEquilibriumEnergy(E0, B, ca, cb, cd, Eeq)) return false;

  // Compare two independently constructed radiation energies:
  //   E:   exact total-energy conservation,
  //   Eeq: the implicit radiation source equation.
  residual = E - Eeq;
  return true;
}

KOKKOS_INLINE_FUNCTION Real CouplingInnerScalarError(const Real eg, const Real eg0,
                                                     const Real dEk, const Real Q,
                                                     const Real E, const Real E0,
                                                     const Real chat, const Real c,
                                                     const Real energy_floor_scale,
                                                     const Real residual) {
  const Real scale = std::max(
      energy_floor_scale,
      std::max(std::max(std::abs(eg0), std::abs(eg)),
               std::max(std::abs(Q), std::max(c / chat * (std::abs(E0) + std::abs(E)),
                                              std::abs(dEk)))));
  return c / chat * std::abs(residual) / scale;
}

KOKKOS_INLINE_FUNCTION bool CouplingResidualChangesSign(const Real a, const Real b) {
  return (a <= 0.0 && b >= 0.0) || (a >= 0.0 && b <= 0.0);
}

template <typename EOSType, typename OpacityType, typename ScatteringType>
KOKKOS_INLINE_FUNCTION bool SolveCouplingInnerScalar(
    const Real eg0, const Real dEk, const Real Q, const Real E0, const Real efloor,
    const Real Bfloor, const Real dens, const Real eref, const Real arad,
    const Real tfloor, const Real dt, const Real chat, const Real c, const Real g,
    const Real g2, const Real beta2, const Real bdbdp, const Real bdf,
    const Real energy_floor_scale, const Real tolerance, const EOSType &eos,
    const OpacityType &opacity, const ScatteringType &scattering, Real &E, Real &B,
    Real &inner_err, int &iterations) {
  const Real eg_floor =
      dens * eos.InternalEnergyFromDensityTemperature(dens, tfloor) / eref;
  const Real eg_ceiling = eg0 + Q - dEk + c / chat * (E0 - efloor);
  if (eg_ceiling < eg_floor) return false;

  const Real Tguess = std::max(tfloor, std::pow(std::max(eref * B / arad, 0.0), 0.25));
  Real eg_guess = dens * eos.InternalEnergyFromDensityTemperature(dens, Tguess) / eref;
  eg_guess = std::max(eg_floor, std::min(eg_ceiling, eg_guess));

  Real Eguess = E;
  Real Bguess = B;
  Real Rguess = 0.0;
  bool guess_valid = EvaluateCouplingInnerScalarResidual(
      eg_guess, eg0, dEk, Q, E0, efloor, Bfloor, dens, eref, arad, tfloor, dt, chat, c, g,
      g2, beta2, bdbdp, bdf, eos, opacity, scattering, Eguess, Bguess, Rguess);

  Real best_eg = eg_guess;
  Real best_E = Eguess;
  Real best_B = Bguess;
  Real best_error = guess_valid
                        ? CouplingInnerScalarError(eg_guess, eg0, dEk, Q, Eguess, E0,
                                                   chat, c, energy_floor_scale, Rguess)
                        : Big();
  if (guess_valid && best_error <= tolerance) {
    E = Eguess;
    B = Bguess;
    inner_err = best_error;
    iterations = 1;
    return true;
  }

  Real Elo = E;
  Real Blo = B;
  Real Rlo = 0.0;
  const bool lo_valid = EvaluateCouplingInnerScalarResidual(
      eg_floor, eg0, dEk, Q, E0, efloor, Bfloor, dens, eref, arad, tfloor, dt, chat, c, g,
      g2, beta2, bdbdp, bdf, eos, opacity, scattering, Elo, Blo, Rlo);
  if (lo_valid) {
    const Real err = CouplingInnerScalarError(eg_floor, eg0, dEk, Q, Elo, E0, chat, c,
                                              energy_floor_scale, Rlo);
    if (err < best_error) {
      best_eg = eg_floor;
      best_E = Elo;
      best_B = Blo;
      best_error = err;
    }
  }

  Real Ehi = E;
  Real Bhi = B;
  Real Rhi = 0.0;
  const bool hi_valid = EvaluateCouplingInnerScalarResidual(
      eg_ceiling, eg0, dEk, Q, E0, efloor, Bfloor, dens, eref, arad, tfloor, dt, chat, c,
      g, g2, beta2, bdbdp, bdf, eos, opacity, scattering, Ehi, Bhi, Rhi);
  if (hi_valid) {
    const Real err = CouplingInnerScalarError(eg_ceiling, eg0, dEk, Q, Ehi, E0, chat, c,
                                              energy_floor_scale, Rhi);
    if (err < best_error) {
      best_eg = eg_ceiling;
      best_E = Ehi;
      best_B = Bhi;
      best_error = err;
    }
  }

  const bool global_bracketed =
      lo_valid && hi_valid && CouplingResidualChangesSign(Rlo, Rhi);
  bool bracketed = false;
  Real bracket_lo = eg_floor;
  Real bracket_hi = eg_ceiling;
  Real bracket_Rlo = Rlo;
  Real bracket_Rhi = Rhi;

  // Search outward from the current state before using the full admissible
  // interval. Geometric spacing is important in low-density cells: a large
  // change in B = a_r T^4 can correspond to an extremely small change in gas
  // energy, while the upper energy bound may be many orders of magnitude away.
  if (guess_valid) {
    constexpr int scan_points = 64;
    Real left_eg = eg_guess;
    Real left_R = Rguess;
    Real right_eg = eg_guess;
    Real right_R = Rguess;
    Real weight = 5.42101086242752217004e-20; // 2^-64
    for (int n = 1; n <= scan_points && !bracketed; ++n) {
      weight = (n == scan_points) ? 1.0 : 2.0 * weight;

      const Real next_left = eg_guess - weight * (eg_guess - eg_floor);
      Real Eleft = E;
      Real Bleft = B;
      Real Rleft = 0.0;
      const bool left_valid = EvaluateCouplingInnerScalarResidual(
          next_left, eg0, dEk, Q, E0, efloor, Bfloor, dens, eref, arad, tfloor, dt, chat,
          c, g, g2, beta2, bdbdp, bdf, eos, opacity, scattering, Eleft, Bleft, Rleft);
      if (left_valid) {
        const Real err = CouplingInnerScalarError(next_left, eg0, dEk, Q, Eleft, E0, chat,
                                                  c, energy_floor_scale, Rleft);
        if (err < best_error) {
          best_eg = next_left;
          best_E = Eleft;
          best_B = Bleft;
          best_error = err;
        }
        if (CouplingResidualChangesSign(Rleft, left_R)) {
          bracket_lo = next_left;
          bracket_hi = left_eg;
          bracket_Rlo = Rleft;
          bracket_Rhi = left_R;
          bracketed = true;
          break;
        }
        left_eg = next_left;
        left_R = Rleft;
      }

      const Real next_right = eg_guess + weight * (eg_ceiling - eg_guess);
      Real Eright = E;
      Real Bright = B;
      Real Rright = 0.0;
      const bool right_valid = EvaluateCouplingInnerScalarResidual(
          next_right, eg0, dEk, Q, E0, efloor, Bfloor, dens, eref, arad, tfloor, dt, chat,
          c, g, g2, beta2, bdbdp, bdf, eos, opacity, scattering, Eright, Bright, Rright);
      if (right_valid) {
        const Real err = CouplingInnerScalarError(next_right, eg0, dEk, Q, Eright, E0,
                                                  chat, c, energy_floor_scale, Rright);
        if (err < best_error) {
          best_eg = next_right;
          best_E = Eright;
          best_B = Bright;
          best_error = err;
        }
        if (CouplingResidualChangesSign(right_R, Rright)) {
          bracket_lo = right_eg;
          bracket_hi = next_right;
          bracket_Rlo = right_R;
          bracket_Rhi = Rright;
          bracketed = true;
          break;
        }
        right_eg = next_right;
        right_R = Rright;
      }
    }
  }

  if (!bracketed && global_bracketed) {
    bracket_lo = eg_floor;
    bracket_hi = eg_ceiling;
    bracket_Rlo = Rlo;
    bracket_Rhi = Rhi;
    bracketed = true;
  }

  iterations = 0;
  if (bracketed) {
    for (iterations = 1; iterations <= 96; ++iterations) {
      // Regula falsi when it lies safely inside the bracket; otherwise use the
      // bisection midpoint. This retains bracketing while accelerating smooth
      // cases.
      Real eg_trial = 0.5 * (bracket_lo + bracket_hi);
      const Real denom = bracket_Rhi - bracket_Rlo;
      if (std::abs(denom) > Fuzz()) {
        const Real eg_secant =
            (bracket_lo * bracket_Rhi - bracket_hi * bracket_Rlo) / denom;
        const Real width = bracket_hi - bracket_lo;
        if (eg_secant > bracket_lo + 0.1 * width && eg_secant < bracket_hi - 0.1 * width)
          eg_trial = eg_secant;
      }

      Real Etrial = E;
      Real Btrial = B;
      Real Rtrial = 0.0;
      const bool trial_valid = EvaluateCouplingInnerScalarResidual(
          eg_trial, eg0, dEk, Q, E0, efloor, Bfloor, dens, eref, arad, tfloor, dt, chat,
          c, g, g2, beta2, bdbdp, bdf, eos, opacity, scattering, Etrial, Btrial, Rtrial);
      if (!trial_valid) break;

      const Real err = CouplingInnerScalarError(eg_trial, eg0, dEk, Q, Etrial, E0, chat,
                                                c, energy_floor_scale, Rtrial);
      if (err < best_error) {
        best_eg = eg_trial;
        best_E = Etrial;
        best_B = Btrial;
        best_error = err;
      }
      if (err <= tolerance) break;

      if (CouplingResidualChangesSign(bracket_Rlo, Rtrial)) {
        bracket_hi = eg_trial;
        bracket_Rhi = Rtrial;
      } else {
        bracket_lo = eg_trial;
        bracket_Rlo = Rtrial;
      }

      const Real width_scale =
          std::max(energy_floor_scale, std::abs(bracket_lo) + std::abs(bracket_hi));
      if ((bracket_hi - bracket_lo) / width_scale <= 8.0 * Eps()) break;
    }
  }

  E = best_E;
  B = best_B;
  inner_err = best_error;
  const Real attainable_tolerance = std::max(tolerance, 32.0 * Eps());
  return inner_err <= attainable_tolerance;
}

template <typename EOSType, typename OpacityType>
KOKKOS_INLINE_FUNCTION bool EvaluateCouplingEnergyOnlyResidual(
    const Real eg, const Real eg0, const Real Q, const Real E0, const Real Emin,
    const Real dens, const Real eref, const Real arad, const Real tfloor, const Real dt,
    const Real chat, const Real c, const EOSType &eos, const OpacityType &opacity,
    Real &E, Real &B, Real &residual) {
  E = E0 + chat / c * (Q - (eg - eg0));
  if (E < Emin) return false;

  const Real eint = eg * eref / dens;
  const Real T = std::max(tfloor, eos.TemperatureFromDensityInternalEnergy(dens, eint));
  B = arad * SQR(SQR(T)) / eref;
  const Real sigp = chat * dt * opacity.PlanckMeanAbsorptionCoefficient(dens, T);

  // This fallback intentionally retains only thermal absorption/emission.
  // Scattering energy exchange is work associated with the momentum source,
  // which is frozen in this path.  Solve the implicit radiation equation in
  // ratio form; sigp*(E-B) is not a resolvable residual when sigp >> 1 and
  // E and B agree to machine precision.
  const Real inv_denom = 1.0 / (1.0 + sigp);
  const Real Eeq = E0 * inv_denom + (sigp * inv_denom) * B;
  residual = E - Eeq;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Moments::MatterCouplingFullSingleImpl
//! \brief Implementation for "full" radiation-matter coupling source
template <Coordinates GEOM, Closure CLOSURE>
TaskStatus MatterCouplingFullSingleImpl(MeshData<Real> *u0, const Real dt) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract gas package and params
  auto &gas_pkg = pm->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto opac_d = gas_pkg->template Param<MeanOpacity>("opacity_d");
  auto scat_d = gas_pkg->template Param<MeanScattering>("scattering_d");
  auto dflr = gas_pkg->template Param<Real>("dfloor");

  // Extract radiation package and params
  auto &moments_pkg = pm->packages.Get("moments");
  const auto chat = moments_pkg->template Param<Real>("chat");
  const auto c = moments_pkg->template Param<Real>("c");
  const auto arad = moments_pkg->template Param<Real>("arad");
  const auto rad_efloor = moments_pkg->template Param<Real>("efloor");
  const auto tfloor = moments_pkg->template Param<Real>("tfloor");
  const auto Bfloor_phys = arad * SQR(SQR(tfloor));
  const auto outer_max = moments_pkg->template Param<int>("outer_iteration_max");
  const auto inner_max = moments_pkg->template Param<int>("inner_iteration_max");
  const auto outer_tol = moments_pkg->template Param<Real>("outer_iteration_tol");
  const auto inner_tol = moments_pkg->template Param<Real>("inner_iteration_tol");
  const Real nonlinear_roundoff_tol = 64.0 * Eps();
  const auto fatal_if_unconverged =
      moments_pkg->template Param<bool>("fatal_if_unconverged");

  // Extract rotating frame quantities
  Real om0 = 0.0;
  Real qshear = 0.0;
  Real gm_bg = 0.0;
  if (pm->packages.Get("artemis")->template Param<bool>("do_orbital_advection") ||
      pm->packages.Get("artemis")->template Param<bool>("do_rotating_frame")) {
    auto &rframe_pkg = pm->packages.Get("rotating_frame");
    qshear = rframe_pkg->template Param<Real>("qshear");
    om0 = rframe_pkg->template Param<Real>("omega");
    gm_bg = rframe_pkg->template Param<Real>("gm");
  }
  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  const bool do_raytrace =
      pm->packages.Get("artemis")->template Param<bool>("do_raytrace");

  // Packing and indexing
  static auto desc = parthenon::MakePackDescriptor<
      rad::cons::energy, rad::cons::flux, gas::cons::density, gas::cons::momentum,
      gas::cons::internal_energy, gas::cons::total_energy, gas::src::energy>(
      resolved_pkgs.get());

  const auto v0 = desc.GetPack(u0);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(u0);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);

  /*
   *  This is a multi-step solve with some fallbacks. Energies and fluxes are normalized
   * to keep numbers around unity.
   *
   *  If the cells are close to the density floor, we skip the full momentum solve and
   * only do energy coupling. This prevents large accelerations.
   *
   *  For the outer loop we:
   *    - Hold the current gas velocity and radiation flux estimate fixed.
   *    - It chooses flux or velocity (beta) as the Newton variable based on mu = eref /
   * (rho * c * chat)
   *    - Run an inner nonlinear solve for material emission B and radiation energy E.
   *      - It uses a damped quasi-Newton update with line search.
   *      - If that stalls, it switches to a bracketed scalar solve in gas internal
   * energy.
   *      - If needed, it can use a momentum-energy predictor to seed the next outer
   * iteration.
   *    - With the converged thermal state, solve the coupled momentum system:
   *
   *  If no full iterate is available, it runs the conservative energy-only fallback:
   *      - freeze gas momentum and radiation flux;
   *      - bracket a thermal solution in gas internal energy;
   *      - use it automatically for atmosphere cells and floor-constrained failures.
   *
   */
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MatterCoupling", DevExecSpace(), 0, u0->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, v0.GetCoordinates(b), k, j, i);
        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);
        // y = U^(0) + dt S(y)

        // U^(0) values
        const Real dens_raw = v0(b, gas::cons::density(), k, j, i);
        const Real dens = (dens_raw > dflr) ? dens_raw : dflr;
        Real Q = 0.0;
        if (do_raytrace) Q = dt * v0(b, gas::src::energy(), k, j, i);

        // In the numerical atmosphere, retain thermal emission/absorption but
        // suppress the momentum update. This avoids accelerating floor-density
        // gas while still allowing absorbed raytraced energy to reradiate into
        // the moment field through the conservative energy-only fallback.
        constexpr Real atmosphere_floor_factor = 100.0;
        const bool numerical_atmosphere = dens_raw <= atmosphere_floor_factor * dflr;

        // Use the EOS-reconstructed energy as the nonlinear-solve reference,
        // but retain the original conserved energy for the final commit. In
        // particular, eg0 includes the material-temperature floor, so applying
        // only the solver increment to the original state would silently omit
        // that floor correction. Preserve the previous increment-only behavior
        // when the floor is inactive so an EOS round trip cannot perturb a
        // zero-opacity problem.
        const Real eg_state = v0(b, gas::cons::internal_energy(), k, j, i);
        const Real eint0 = std::max(0.0, eg_state / dens);
        const Real T_state = eos_d.TemperatureFromDensityInternalEnergy(dens, eint0);
        const bool initial_material_floor_active = T_state < tfloor;
        Real T = std::max(tfloor, T_state);
        Real eg0 = dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
        const Real eg_floor_delta = initial_material_floor_active ? eg0 - eg_state : 0.0;
        Real B = std::max(Bfloor_phys, arad * SQR(SQR(T)));

        const auto vb = RotatingFrame::BackgroundVelocity<GEOM>(
            qshear, om0, gm_bg, coords.GetCellCenter(vg, b, k, j, i));
        const std::array<Real, 3> p0{
            vb[0] * dens + v0(b, gas::cons::momentum(0), k, j, i) / hx[0],
            vb[1] * dens + v0(b, gas::cons::momentum(1), k, j, i) / hx[1],
            vb[2] * dens + v0(b, gas::cons::momentum(2), k, j, i) / hx[2]};

        // The radiation-energy floor and material-temperature floor are
        // independent constraints. In particular, an optically thin cell may
        // have E_r << a_r T_floor^4. Promoting E_r to Bfloor_phys here creates
        // an artificial LTE radiation bath and can make the constrained
        // matter-coupling equations inconsistent at the first timestep.
        const Real Er_state = v0(b, rad::cons::energy(), k, j, i);
        Real E0 = std::max(rad_efloor, Er_state);
        // Use the arithmetic mean as the reference scale. Unlike the
        // geometric mean, this keeps both normalized energies bounded when the
        // gas and radiation temperatures are initially very different.
        const Real eref_max = std::max(E0, B);
        const Real eref_min = std::min(E0, B);
        Real eref = 0.5 * eref_max * (1.0 + eref_min / std::max(eref_max, Fuzz()));
        const Real fref = c * eref;
        // Keep the radiation floor independent of the material temperature
        // floor. Bfloor constrains T; efloor constrains E_r.
        const Real efloor = rad_efloor / eref;
        const Real Bfloor = Bfloor_phys / eref;

        Q /= eref;
        E0 /= eref;
        eg0 /= eref;
        B /= eref;

        const auto fred0 =
            NormalizeFlux(v0(b, rad::cons::flux(0), k, j, i) / (hx[0] * fref * E0),
                          v0(b, rad::cons::flux(1), k, j, i) / (hx[1] * fref * E0),
                          v0(b, rad::cons::flux(2), k, j, i) / (hx[2] * fref * E0));
        const std::array<Real, 3> Fr0{E0 * fred0[0], E0 * fred0[1], E0 * fred0[2]};

        std::array<Real, 3> v{p0[0] / dens, p0[1] / dens, p0[2] / dens};
        const std::array<Real, 3> beta0{v[0] / c, v[1] / c, v[2] / c};
        const Real beta20 = SQR(beta0[0]) + SQR(beta0[1]) + SQR(beta0[2]);
        if (beta20 >= 1.0) {
          const Real delta_eg_state = eg_floor_delta + Q * eref;
          v0(b, gas::cons::internal_energy(), k, j, i) += delta_eg_state;
          v0(b, gas::cons::total_energy(), k, j, i) += delta_eg_state;
          v0(b, rad::cons::energy(), k, j, i) = E0 * eref;
          v0(b, rad::cons::flux(0), k, j, i) = Fr0[0] * hx[0] * fref;
          v0(b, rad::cons::flux(1), k, j, i) = Fr0[1] * hx[1] * fref;
          v0(b, rad::cons::flux(2), k, j, i) = Fr0[2] * hx[2] * fref;
          return;
        }
        const Real ke0 = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2])) / eref;

        Real E = E0;
        auto F = Fr0;

        // Start outer iteration
        int outer_iter = 0;
        int inner_iter = 0;
        Real outer_err = Big();
        Real inner_err = Big();

        std::array<Real, 3> dv{0.0, 0.0, 0.0};
        Real dEk = 0.0;
        Real dEg = 0.0;
        Real dEr = 0.0;
        const Real energy_floor_scale = std::max(efloor + Bfloor, Fuzz());
        Real escale = std::max(
            energy_floor_scale,
            std::max(std::abs(eg0), std::max(std::abs(Q), c / chat * std::abs(E0))));
        bool solve_valid = !numerical_atmosphere && escale > 0.0;
        bool outer_converged = false;
        bool have_complete_iterate = false;
        Real best_outer_err = Big();
        std::array<Real, 3> best_F = Fr0;
        std::array<Real, 3> best_dv{0.0, 0.0, 0.0};
        Real best_dEg = 0.0;
        Real best_dEk = 0.0;
        Real best_dEr = 0.0;
        bool scalar_inner_used = false;
        int momentum_predictor_count = 0;
        int momentum_iteration_count = 0;
        Real momentum_error = Big();

        for (outer_iter = 1; outer_iter <= outer_max; ++outer_iter) {
          if (!solve_valid) break;

          const Real ke = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2])) / eref;
          std::array<Real, 3> beta{v[0] / c, v[1] / c, v[2] / c};
          const Real beta2 = SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);
          if (beta2 >= 1.0 || E <= 0.0) {
            solve_valid = false;
            break;
          }
          const Real g2 = 1.0 / (1.0 - beta2);
          const Real g = std::sqrt(g2);

          // F is normalized by c*eref, so F/E is the reduced flux.
          const auto fedd = EddingtonTensor<CLOSURE>({F[0] / E, F[1] / E, F[2] / E});
          const std::array<Real, 3> bdp{
              beta[0] * fedd[TensIdx::X11] + beta[1] * fedd[TensIdx::X12] +
                  beta[2] * fedd[TensIdx::X13],
              beta[0] * fedd[TensIdx::X12] + beta[1] * fedd[TensIdx::X22] +
                  beta[2] * fedd[TensIdx::X23],
              beta[0] * fedd[TensIdx::X13] + beta[1] * fedd[TensIdx::X23] +
                  beta[2] * fedd[TensIdx::X33]};
          const Real bdbdp = beta[0] * bdp[0] + beta[1] * bdp[1] + beta[2] * bdp[2];
          const Real bdf = beta[0] * F[0] + beta[1] * F[1] + beta[2] * F[2];

          // Damped quasi-Newton solve for (B,E).
          Real previous_inner_err = Big();
          int stalled_iterations = 0;
          bool inner_converged = false;
          for (inner_iter = 1; inner_iter <= inner_max; ++inner_iter) {
            T = std::max(tfloor, std::pow(std::max(eref * B / arad, 0.0), 0.25));
            const Real eint =
                dens * eos_d.InternalEnergyFromDensityTemperature(dens, T) / eref;
            const Real Cv = dens * eos_d.SpecificHeatFromDensityTemperature(dens, T);
            const Real fleck = FleckFactor(arad, T, Cv);

            const Real sigp = chat * dt * opac_d.PlanckMeanAbsorptionCoefficient(dens, T);
            const Real sigs =
                chat * dt * scat_d.RosselandMeanTotalScatteringCoefficient(dens, T);
            Real ca = 0.0;
            Real cb = 0.0;
            Real cd = 0.0;
            ComputeCouplingEnergyCoefficients(sigp, sigs, g, g2, beta2, bdbdp, bdf, ca,
                                              cb, cd);
            Real Eeq_inner = E;
            // A degenerate equilibrium linearization leaves the seed
            // Eeq_inner = E, which would read as a zero source residual. Abandon
            // the quasi-Newton sweep (with a large error so it is not accepted)
            // and fall through to the bracketed scalar inner solve below.
            if (!ComputeCouplingEquilibriumEnergy(E0, B, ca, cb, cd, Eeq_inner)) {
              inner_err = Big();
              break;
            }

            // Retain the original residuals only to form the quasi-Newton
            // search direction.  Use the conservation/equilibrium form below
            // for convergence and line-search acceptance.
            const Real G0 = ca * E - cb * B + cd;
            const Real Fi = (ke - ke0) + (eint - eg0) - c / chat * G0 - Q;
            const Real Fr = (E - E0) + G0;
            const Real conservation_inner =
                (ke - ke0) + (eint - eg0) + c / chat * (E - E0) - Q;
            const Real source_inner = E - Eeq_inner;

            const Real escale_inner = std::max(
                energy_floor_scale,
                std::max(std::max(std::abs(eg0), std::abs(eint)),
                         std::max(std::abs(Q),
                                  std::max(c / chat * (std::abs(E0) + std::abs(E)),
                                           std::abs(ke - ke0)))));
            inner_err = std::max(std::abs(conservation_inner) / escale_inner,
                                 c / chat * std::abs(source_inner) / escale_inner);
            solve_valid =
                solve_valid && Cv > 0.0 && fleck >= 0.0 && sigp >= 0.0 && sigs >= 0.0;
            if (!solve_valid) break;
            if (inner_err <= inner_tol) {
              inner_converged = true;
              break;
            }

            if (inner_err >= previous_inner_err * (1.0 - 1.0e-6))
              ++stalled_iterations;
            else
              stalled_iterations = 0;
            previous_inner_err = inner_err;
            if (stalled_iterations >= 8) break;

            const Real dfac = 1.0 + c / chat * fleck * cb;
            const Real denom = dfac + ca;
            if (std::abs(denom) <= Fuzz()) {
              solve_valid = false;
              break;
            }
            const Real dE = dfac / denom * (-Fr) + fleck / denom * (-Fi * cb);
            const Real dB = c / chat * fleck / denom * (-ca * Fr) +
                            (1.0 + ca) * fleck / denom * (-Fi);

            bool accepted = false;
            Real best_trial_err = inner_err;
            Real best_trial_E = E;
            Real best_trial_B = B;
            Real alpha = 1.0;
            for (int ls = 0; ls < 12; ++ls) {
              const Real Etrial = std::max(efloor, E + alpha * dE);
              const Real Btrial = std::max(Bfloor, B + alpha * dB);
              const Real Ttrial =
                  std::max(tfloor, std::pow(std::max(eref * Btrial / arad, 0.0), 0.25));
              const Real eint_trial =
                  dens * eos_d.InternalEnergyFromDensityTemperature(dens, Ttrial) / eref;
              const Real sigp_trial =
                  chat * dt * opac_d.PlanckMeanAbsorptionCoefficient(dens, Ttrial);
              const Real sigs_trial =
                  chat * dt *
                  scat_d.RosselandMeanTotalScatteringCoefficient(dens, Ttrial);
              Real ca_trial = 0.0;
              Real cb_trial = 0.0;
              Real cd_trial = 0.0;
              ComputeCouplingEnergyCoefficients(sigp_trial, sigs_trial, g, g2, beta2,
                                                bdbdp, bdf, ca_trial, cb_trial, cd_trial);
              Real Eeq_trial = Etrial;
              const bool eq_ok_trial = ComputeCouplingEquilibriumEnergy(
                  E0, Btrial, ca_trial, cb_trial, cd_trial, Eeq_trial);
              const Real conservation_trial =
                  (ke - ke0) + (eint_trial - eg0) + c / chat * (Etrial - E0) - Q;
              const Real source_trial = Etrial - Eeq_trial;
              const Real escale_trial = std::max(
                  energy_floor_scale,
                  std::max(std::max(std::abs(eg0), std::abs(eint_trial)),
                           std::max(std::abs(Q),
                                    std::max(c / chat * (std::abs(E0) + std::abs(Etrial)),
                                             std::abs(ke - ke0)))));
              // Reject a trial whose equilibrium linearization is degenerate:
              // its source residual is undefined, so it must never win the line
              // search.
              const Real trial_err =
                  eq_ok_trial ? std::max(std::abs(conservation_trial) / escale_trial,
                                         c / chat * std::abs(source_trial) / escale_trial)
                              : Big();
              if (trial_err < best_trial_err) {
                best_trial_err = trial_err;
                best_trial_E = Etrial;
                best_trial_B = Btrial;
              }
              if ((trial_err <= inner_tol ||
                   trial_err <= inner_err * (1.0 - 1.0e-4 * alpha))) {
                E = Etrial;
                B = Btrial;
                accepted = true;
                break;
              }
              alpha *= 0.5;
            }
            if (!accepted) {
              if (best_trial_err < inner_err) {
                E = best_trial_E;
                B = best_trial_B;
              } else {
                break;
              }
            }
          } // inner iteration

          if (!inner_converged && solve_valid &&
              (inner_iter > inner_max || stalled_iterations >= 8) &&
              inner_err <= std::max(inner_tol, nonlinear_roundoff_tol)) {
            inner_converged = true;
          }

          // The opacity-frozen quasi-Newton Jacobian can stall when opacity is
          // strongly temperature dependent or the solution lies on a floor.
          // In that case eliminate E using exact total-energy conservation and
          // solve the remaining full mixed-frame radiation equation as a
          // bracketed scalar problem in gas internal energy.  The scalar
          // residual compares the energy from exact conservation with the
          // energy from the implicit source equation, avoiding opacity-
          // amplified cancellation near LTE.
          if (!inner_converged && solve_valid) {
            scalar_inner_used = true;
            int scalar_iterations = 0;
            const bool scalar_converged = SolveCouplingInnerScalar(
                eg0, ke - ke0, Q, E0, efloor, Bfloor, dens, eref, arad, tfloor, dt, chat,
                c, g, g2, beta2, bdbdp, bdf, energy_floor_scale, inner_tol, eos_d, opac_d,
                scat_d, E, B, inner_err, scalar_iterations);
            inner_iter += scalar_iterations;
            inner_converged = scalar_converged;
          }
          if (!inner_converged && solve_valid && momentum_predictor_count < outer_max) {
            // Bootstrap the kinetic-work term with the same coupled momentum
            // solve used below.  Unlike the old full Picard predictor, this
            // cannot jump to a superluminal intermediate state when the
            // reduced-c radiation pseudo-inertia dominates the gas inertia.
            const Real Tpred =
                std::max(tfloor, std::pow(std::max(eref * B / arad, 0.0), 0.25));
            const Real eg_pred =
                dens * eos_d.InternalEnergyFromDensityTemperature(dens, Tpred) / eref;
            const Real dEg_pred = eg_pred - eg0;
            const Real sigp_pred =
                chat * dt * opac_d.RosselandMeanAbsorptionCoefficient(dens, Tpred);
            const Real sigs_pred =
                chat * dt * scat_d.RosselandMeanTotalScatteringCoefficient(dens, Tpred);
            std::array<Real, 3> beta_pred{v[0] / c, v[1] / c, v[2] / c};
            std::array<Real, 3> Fpred = F;
            Real Epred = E;
            Real dEkpred = dEk;
            Real predictor_error = Big();
            int predictor_iterations = 0;
            const bool predictor_converged = SolveCouplingMomentumEnergy<CLOSURE>(
                beta0, Fr0, E0, efloor, B, dEg_pred, Q, dens, eref, c, chat, sigp_pred,
                sigs_pred, energy_floor_scale,
                std::max(outer_tol, nonlinear_roundoff_tol), beta_pred, Fpred, Epred,
                dEkpred, predictor_error, predictor_iterations);
            momentum_iteration_count += predictor_iterations;
            if (predictor_converged) {
              const Real predictor_scale = std::max(
                  energy_floor_scale, std::max(E, std::max(VNorm(F), VNorm(Fpred))));
              Real predictor_change = std::abs(Epred - E) / predictor_scale;
              for (int d = 0; d < 3; ++d) {
                predictor_change = std::max(predictor_change,
                                            std::abs(Fpred[d] - F[d]) / predictor_scale);
                predictor_change =
                    std::max(predictor_change, std::abs(beta_pred[d] - v[d] / c));
              }
              if (predictor_change > 8.0 * Eps()) {
                E = Epred;
                F = Fpred;
                dEk = dEkpred;
                const Real dv_factor = eref / (dens * chat);
                for (int d = 0; d < 3; ++d) {
                  dv[d] = -dv_factor * (F[d] - Fr0[d]);
                  v[d] = p0[d] / dens + dv[d];
                }
                ++momentum_predictor_count;
                --outer_iter;
                continue;
              }
            }
          }
          if (!inner_converged) break;

          T = std::max(tfloor, std::pow(std::max(eref * B / arad, 0.0), 0.25));
          const Real eg =
              dens * eos_d.InternalEnergyFromDensityTemperature(dens, T) / eref;
          dEg = eg - eg0;

          const Real sigp_flux =
              chat * dt * opac_d.RosselandMeanAbsorptionCoefficient(dens, T);
          const Real sigs_flux =
              chat * dt * scat_d.RosselandMeanTotalScatteringCoefficient(dens, T);

          // Solve the radiation-flux source equation together with gas
          // pseudo-momentum conservation and reduced-c total-energy
          // conservation.  A direct Picard update is unstable when
          //   mu = eref/(rho*c*chat)
          // is large; that is exactly the regime encountered just above a
          // density floor.  The local 3D Newton solve removes this stiffness.
          std::array<Real, 3> beta_momentum{v[0] / c, v[1] / c, v[2] / c};
          int momentum_iterations = 0;
          const bool momentum_converged = SolveCouplingMomentumEnergy<CLOSURE>(
              beta0, Fr0, E0, efloor, B, dEg, Q, dens, eref, c, chat, sigp_flux,
              sigs_flux, energy_floor_scale, std::max(outer_tol, nonlinear_roundoff_tol),
              beta_momentum, F, E, dEk, momentum_error, momentum_iterations);
          momentum_iteration_count += momentum_iterations;
          if (!momentum_converged) {
            solve_valid = false;
            break;
          }
          const Real dv_factor = eref / (dens * chat);
          for (int d = 0; d < 3; ++d) {
            dv[d] = -dv_factor * (F[d] - Fr0[d]);
            v[d] = p0[d] / dens + dv[d];
          }
          dEr = E - E0;

          // Recompute the full coupled residual from the final state of this
          // outer iteration. This avoids declaring convergence merely because
          // the kinetic-energy correction changed little in a Keplerian flow.
          const std::array<Real, 3> beta_out{v[0] / c, v[1] / c, v[2] / c};
          const Real beta2_out = SQR(beta_out[0]) + SQR(beta_out[1]) + SQR(beta_out[2]);
          if (beta2_out >= 1.0 || E <= 0.0) {
            solve_valid = false;
            break;
          }
          const Real g2_out = 1.0 / (1.0 - beta2_out);
          const Real g_out = std::sqrt(g2_out);
          const auto fedd_out = EddingtonTensor<CLOSURE>({F[0] / E, F[1] / E, F[2] / E});
          const std::array<Real, 3> bdp_out{beta_out[0] * fedd_out[TensIdx::X11] +
                                                beta_out[1] * fedd_out[TensIdx::X12] +
                                                beta_out[2] * fedd_out[TensIdx::X13],
                                            beta_out[0] * fedd_out[TensIdx::X12] +
                                                beta_out[1] * fedd_out[TensIdx::X22] +
                                                beta_out[2] * fedd_out[TensIdx::X23],
                                            beta_out[0] * fedd_out[TensIdx::X13] +
                                                beta_out[1] * fedd_out[TensIdx::X23] +
                                                beta_out[2] * fedd_out[TensIdx::X33]};
          const Real bdbdp_out = beta_out[0] * bdp_out[0] + beta_out[1] * bdp_out[1] +
                                 beta_out[2] * bdp_out[2];
          const Real bdf_out =
              beta_out[0] * F[0] + beta_out[1] * F[1] + beta_out[2] * F[2];

          T = std::max(tfloor, std::pow(std::max(eref * B / arad, 0.0), 0.25));
          const Real eg_out =
              dens * eos_d.InternalEnergyFromDensityTemperature(dens, T) / eref;
          dEg = eg_out - eg0;
          const Real sigp_energy =
              chat * dt * opac_d.PlanckMeanAbsorptionCoefficient(dens, T);
          const Real sigs_out =
              chat * dt * scat_d.RosselandMeanTotalScatteringCoefficient(dens, T);
          Real ca_out = 0.0;
          Real cb_out = 0.0;
          Real cd_out = 0.0;
          Real Eeq_out = E;
          ComputeCouplingEnergyCoefficients(sigp_energy, sigs_out, g_out, g2_out,
                                            beta2_out, bdbdp_out, bdf_out, ca_out, cb_out,
                                            cd_out);
          const bool eq_ok_out =
              ComputeCouplingEquilibriumEnergy(E0, B, ca_out, cb_out, cd_out, Eeq_out);
          const Real conservation_residual = dEg + dEk + c / chat * dEr - Q;
          const Real source_residual = E - Eeq_out;

          const Real sigp_flux_out =
              chat * dt * opac_d.RosselandMeanAbsorptionCoefficient(dens, T);
          const Real sigf_flux_out = sigp_flux_out + sigs_out;
          const Real a_out = g_out * sigf_flux_out;
          const Real b_out = 2.0 * g2_out * g_out * sigs_out;
          const Real d1_out =
              g_out * (sigp_flux_out * B + g2_out * sigs_out * (1.0 + bdbdp_out) * E);
          const Real d2_out = g_out * sigf_flux_out * E;
          const std::array<Real, 3> rhs_out{
              Fr0[0] + d1_out * beta_out[0] + d2_out * bdp_out[0],
              Fr0[1] + d1_out * beta_out[1] + d2_out * bdp_out[1],
              Fr0[2] + d1_out * beta_out[2] + d2_out * bdp_out[2]};
          const auto Fsolve_out = SolveRadFlux(1.0 + a_out, b_out, beta_out, rhs_out);
          const auto Ftarget_out = ProjectCouplingFluxToEnergy(Fsolve_out, E);

          escale =
              std::max(energy_floor_scale,
                       std::max(std::max(std::abs(eg0), std::abs(eg_out)),
                                std::max(std::abs(Q),
                                         std::max(c / chat * (std::abs(E0) + std::abs(E)),
                                                  std::abs(dEk)))));
          const Real energy_residual =
              std::max(std::abs(conservation_residual) / escale,
                       c / chat * std::abs(source_residual) / escale);
          const Real flux_scale = std::max(
              energy_floor_scale,
              std::max(E, std::max(VNorm(Fr0), std::max(VNorm(F), VNorm(Ftarget_out)))));
          Real flux_residual = 0.0;
          for (int d = 0; d < 3; ++d) {
            flux_residual =
                std::max(flux_residual, std::abs(F[d] - Ftarget_out[d]) / flux_scale);
          }
          const Real realizability_residual =
              std::max(0.0, (VNorm(F) - E) / std::max(E, energy_floor_scale));
          // These are residuals of the coupled equations evaluated at the
          // current state. Iterate-to-iterate changes are deliberately not an
          // acceptance criterion: a converged nonlinear root need not move by
          // less than a user tolerance tighter than the local arithmetic can
          // resolve.
          outer_err =
              std::max(energy_residual, std::max(flux_residual, realizability_residual));
          // eq_ok_out guards the source residual: a degenerate equilibrium makes
          // source_residual (E - Eeq_out) spuriously zero, so this iterate must
          // not be recorded as complete/converged below.
          solve_valid = solve_valid && eq_ok_out && sigp_energy >= 0.0 &&
                        sigs_out >= 0.0 && sigp_flux_out >= 0.0;
          if (!solve_valid) break;

          have_complete_iterate = true;
          if (outer_err < best_outer_err) {
            best_outer_err = outer_err;
            best_F = F;
            best_dv = dv;
            best_dEg = dEg;
            best_dEk = dEk;
            best_dEr = dEr;
          }
          if (outer_err <= outer_tol) {
            outer_converged = true;
            break;
          }
        } // outer iteration

        // A hard floor turns the nonlinear equality problem into a
        // constrained active-set problem. If the unconstrained root lies below
        // either floor, a zero equality residual does not exist in the
        // admissible state space. In that case use the conservative
        // energy-only solve below with momentum frozen, rather than repeatedly
        // applying a radiation drag whose kinetic-energy work cannot be paid by
        // either the gas or radiation field.
        const Real floor_active_tol = 256.0 * Eps();
        const bool material_floor_active =
            B <= Bfloor * (1.0 + floor_active_tol) + Fuzz();
        const bool radiation_floor_active =
            E <= efloor * (1.0 + floor_active_tol) + Fuzz();
        const bool floor_constrained_failure =
            solve_valid && !have_complete_iterate &&
            (material_floor_active || radiation_floor_active);
        bool fallback_success = false;
        Real fallback_error = Big();

        if (have_complete_iterate) {
          F = best_F;
          dv = best_dv;
          for (int d = 0; d < 3; ++d)
            v[d] = p0[d] / dens + dv[d];
          dEg = best_dEg;
          dEk = best_dEk;
          dEr = best_dEr;
          E = E0 + dEr;
        } else {
          // Conservative energy-only fallback. Hold radiation flux and gas
          // momentum fixed, eliminate E using total-energy conservation, and
          // bracket the remaining scalar radiation-energy residual in gas
          // internal energy. This is also the intentional atmosphere path.
          F = Fr0;
          dv = {0.0, 0.0, 0.0};
          v = {p0[0] / dens, p0[1] / dens, p0[2] / dens};
          dEk = 0.0;
          const Real Emin = std::max(efloor, VNorm(Fr0));
          const Real eg_floor =
              dens * eos_d.InternalEnergyFromDensityTemperature(dens, tfloor) / eref;
          const Real eg_max = eg0 + Q + c / chat * (E0 - Emin);

          bool fallback_valid = eg_max >= eg_floor;
          bool fallback_have = false;
          bool bracketed = false;
          Real bracket_lo = eg_floor;
          Real bracket_hi = eg_max;
          Real residual_lo = 0.0;
          Real best_eg = eg0;
          Real best_E_fallback = E0;
          Real best_B_fallback = B;
          Real best_residual = Big();
          Real previous_eg = eg_floor;
          Real previous_residual = 0.0;
          bool have_previous = false;

          // bisection fallback #1
          if (fallback_valid) {
            constexpr int fallback_scan_points = 64;
            for (int n = 0; n <= fallback_scan_points; ++n) {
              const Real frac = static_cast<Real>(n) / fallback_scan_points;
              const Real eg_trial = eg_floor + frac * (eg_max - eg_floor);
              Real E_trial = E0;
              Real B_trial = B;
              Real residual = 0.0;
              const bool valid_trial = EvaluateCouplingEnergyOnlyResidual(
                  eg_trial, eg0, Q, E0, Emin, dens, eref, arad, tfloor, dt, chat, c,
                  eos_d, opac_d, E_trial, B_trial, residual);
              if (!valid_trial) continue;

              fallback_have = true;
              if (std::abs(residual) < best_residual) {
                best_residual = std::abs(residual);
                best_eg = eg_trial;
                best_E_fallback = E_trial;
                best_B_fallback = B_trial;
              }
              if (have_previous && !bracketed &&
                  ((previous_residual <= 0.0 && residual >= 0.0) ||
                   (previous_residual >= 0.0 && residual <= 0.0))) {
                bracketed = true;
                bracket_lo = previous_eg;
                bracket_hi = eg_trial;
                residual_lo = previous_residual;
              }
              previous_eg = eg_trial;
              previous_residual = residual;
              have_previous = true;
            }
          }

          if (bracketed) {
            for (int n = 0; n < 64; ++n) {
              const Real eg_mid = 0.5 * (bracket_lo + bracket_hi);
              Real E_mid = E0;
              Real B_mid = B;
              Real residual_mid = 0.0;
              const bool valid_mid = EvaluateCouplingEnergyOnlyResidual(
                  eg_mid, eg0, Q, E0, Emin, dens, eref, arad, tfloor, dt, chat, c, eos_d,
                  opac_d, E_mid, B_mid, residual_mid);
              if (!valid_mid) break;
              if (std::abs(residual_mid) < best_residual) {
                best_residual = std::abs(residual_mid);
                best_eg = eg_mid;
                best_E_fallback = E_mid;
                best_B_fallback = B_mid;
              }
              if ((residual_lo <= 0.0 && residual_mid >= 0.0) ||
                  (residual_lo >= 0.0 && residual_mid <= 0.0)) {
                bracket_hi = eg_mid;
              } else {
                bracket_lo = eg_mid;
                residual_lo = residual_mid;
              }
              if ((bracket_hi - bracket_lo) /
                      std::max(energy_floor_scale,
                               std::abs(bracket_hi) + std::abs(bracket_lo)) <=
                  nonlinear_roundoff_tol)
                break;
            }
          }

          if (fallback_have) {
            dEg = best_eg - eg0;
            E = best_E_fallback;
            B = best_B_fallback;
            dEr = E - E0;
            const Real fallback_scale = std::max(
                energy_floor_scale,
                std::max(std::abs(Q), std::max(std::abs(eg0) + std::abs(best_eg),
                                               c / chat * (std::abs(E0) + std::abs(E)))));
            fallback_error = c / chat * best_residual / std::max(fallback_scale, Fuzz());
            fallback_success = true;
          } else {
            // Last-resort finite update. This is intentionally limited to the
            // external ray source and leaves the moment state unchanged.
            dEg = Q;
            dEr = 0.0;
            E = E0;
            F = Fr0;
            fallback_success = false;
          }
        }

        const bool accepted_floor_fallback =
            floor_constrained_failure && fallback_success;
        if (!outer_converged && fatal_if_unconverged && !numerical_atmosphere &&
            !accepted_floor_fallback) {
#ifndef NDEBUG
          // printf with args is slow on device, so hide this at compile time
          const Real beta_fail = std::sqrt(SQR(v[0] / c) + SQR(v[1] / c) + SQR(v[2] / c));
          const Real fred_fail = VNorm(F) / std::max(E, energy_floor_scale);
          const Real mu_fail = eref / (dens * c * chat);
          const Real delta_f_fail = VNorm({F[0] - Fr0[0], F[1] - Fr0[1], F[2] - Fr0[2]});
          printf("MatterCoupling full fail (%d,%d,%d,%d): outer=%.17e "
                 "(best=%.17e, tol=%.17e), inner=%.17e (tol=%.17e), "
                 "fallback=%.17e, E=%.17e, E0=%.17e, efloor=%.17e, "
                 "B=%.17e, Bfloor=%.17e, Q=%.17e, dEk=%.17e, "
                 "rho=%.17e, rho/dfloor=%.17e, beta0=%.17e, "
                 "beta=%.17e, fred=%.17e, dF=%.17e, chat/c=%.17e, "
                 "mu=%.17e, flux_variable=%d, outer_iter=%d, "
                 "inner_iter=%d, momentum_iter=%d, "
                 "momentum_err=%.17e, valid=%d, scalar=%d, "
                 "predictors=%d, complete=%d, floor_constrained=%d\n",
                 b, k, j, i, outer_err, best_outer_err, outer_tol, inner_err, inner_tol,
                 fallback_error, E, E0, efloor, B, Bfloor, Q, dEk, dens, dens / dflr,
                 std::sqrt(beta20), beta_fail, fred_fail, delta_f_fail, chat / c, mu_fail,
                 static_cast<int>(mu_fail <= 1.0), outer_iter, inner_iter,
                 momentum_iteration_count, momentum_error, solve_valid, scalar_inner_used,
                 momentum_predictor_count, have_complete_iterate,
                 floor_constrained_failure);
#endif
          PARTHENON_FAIL("Outer not converged");
        }

        // Include the material-floor correction in both internal and total
        // energy. Away from the floor this reduces to the original dEg update.
        const Real delta_eg_state = eg_floor_delta + dEg * eref;
        v0(b, gas::cons::internal_energy(), k, j, i) += delta_eg_state;
        v0(b, gas::cons::total_energy(), k, j, i) += delta_eg_state + dEk * eref;
        v0(b, rad::cons::energy(), k, j, i) = (E0 + dEr) * eref;
        v0(b, gas::cons::momentum(0), k, j, i) += dv[0] * dens * hx[0];
        v0(b, gas::cons::momentum(1), k, j, i) += dv[1] * dens * hx[1];
        v0(b, gas::cons::momentum(2), k, j, i) += dv[2] * dens * hx[2];
        v0(b, rad::cons::flux(0), k, j, i) = F[0] * hx[0] * fref;
        v0(b, rad::cons::flux(1), k, j, i) = F[1] * hx[1] * fref;
        v0(b, rad::cons::flux(2), k, j, i) = F[2] * hx[2] * fref;
      });

  return TaskStatus::complete;
}

} // namespace Moments

#endif // RADIATION_MOMENTS_MATTER_COUPLING_HPP_
