//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef DRAG_COLLISION_INTEGRALS_HPP_
#define DRAG_COLLISION_INTEGRALS_HPP_

//! \file collision_integrals.hpp
//! \brief Chapman-Cowling collision-integral backend for multifluid momentum and
//!        energy coupling.
//!
//! ## Theory
//! The Chapman-Enskog (Chapman-Cowling) treatment of transport in a gas mixture gives
//! binary drag and thermal-relaxation coefficients that depend on the pair collision
//! integral Omega^{(l,s)}_{ij}.  For rigid hard spheres (HS) all Omega^{(l,s)} reduce
//! to simple geometric prefactors and the momentum-transfer rate coefficient is
//!
//!   K_{ij} = (4/3) n_i n_j mu_{ij} * sigma_{ij}^2 * sqrt(8 pi kT / mu_{ij})
//!           = n_i n_j * K_unit
//!
//! where mu_{ij} = m_i m_j / (m_i + m_j) is the reduced mass (in AMU * code-mass),
//! sigma_{ij} = (sigma_i + sigma_j)/2 is the mean collision diameter, and T is the
//! pair temperature.
//!
//! The Lennard-Jones (LJ) surrogate uses the reduced temperature T* = kT/eps_{ij} to
//! evaluate a polynomial fit to the collision integrals Omega^{(1,1)*} and Omega^{(2,2)*}
//! (from Neufeld et al. 1972, also used in CHEMKIN).
//!
//! ## Units
//! All quantities are in Artemis code units.  The caller is responsible for supplying
//! sigma (collision diameter) in code-length units, eps (LJ well depth) in Kelvin (T is
//! also in Kelvin via the EOS), mu in AMU, density in code-mass/code-length^3, and T in
//! Kelvin.  The returned K [mass/(vol*time)] = [code-mass / (code-length^3 * code-time)]
//! can be multiplied by dt and divided by rho to get a dimensionless coupling strength.
//!
//! ## Backends
//! - GasDragModel::hard_sphere  -- calibrated hard-sphere, no adjustable parameter
//! - GasDragModel::lj           -- Lennard-Jones surrogate with Neufeld fits

#include <cmath>

#include "artemis.hpp"
#include "drag.hpp"

namespace Drag {
namespace CollisionIntegrals {

// ---------------------------------------------------------------------------
// Neufeld et al. (1972) polynomial fit coefficients for Omega^{(1,1)*}
// Valid for 0.3 <= T* <= 400.
// ---------------------------------------------------------------------------
namespace detail {
KOKKOS_FORCEINLINE_FUNCTION
Real Omega11Star(const Real Tstar) {
  // Neufeld 1972 fit: A / T*^B  + C/exp(D*T*) + E/exp(F*T*) + G/exp(H*T*)
  constexpr Real A = 1.06036, B = 0.15610;
  constexpr Real C = 0.19300, D = 0.47635;
  constexpr Real E = 1.03587, F = 1.52996;
  constexpr Real G = 1.76474, H = 3.89411;
  return A / std::pow(Tstar, B) + C / std::exp(D * Tstar) + E / std::exp(F * Tstar) +
         G / std::exp(H * Tstar);
}

KOKKOS_FORCEINLINE_FUNCTION
Real Omega22Star(const Real Tstar) {
  // Neufeld 1972 fit for Omega^{(2,2)*}
  constexpr Real A = 1.16145, B = 0.14874;
  constexpr Real C = 0.52487, D = 0.77320;
  constexpr Real E = 2.16178, F = 2.43787;
  constexpr Real G = -6.435e-4, H = 7.27371;
  constexpr Real R = 18.0323, S = -0.76830, W = 7.27371;
  // Extended fit (7-term, Kim & Monroe 2014 correction)
  return A / std::pow(Tstar, B) + C / std::exp(D * Tstar) + E / std::exp(F * Tstar) +
         G * std::sin(R * Tstar - W) / std::pow(Tstar, S);
}

// Combined pair properties
KOKKOS_FORCEINLINE_FUNCTION
Real sigma_ij(const Real sigma_i, const Real sigma_j) {
  return 0.5 * (sigma_i + sigma_j);
}

// Lorentz-Berthelot combination rule for LJ epsilon [K]
KOKKOS_FORCEINLINE_FUNCTION
Real eps_ij(const Real eps_i, const Real eps_j) {
  return std::sqrt(std::max(eps_i * eps_j, 0.0));
}

// Reduced mass in same units as m_i, m_j (AMU)
KOKKOS_FORCEINLINE_FUNCTION
Real mu_ij(const Real m_i, const Real m_j) {
  return m_i * m_j / (m_i + m_j + Fuzz<Real>());
}
} // namespace detail

// ---------------------------------------------------------------------------
//! \fn DragCoeff
//! \brief Momentum-transfer rate coefficient K_{ij} [mass/(vol*time)]
//!
//! \param model   collision model selector
//! \param m_i     molecular mass species i [AMU]
//! \param m_j     molecular mass species j [AMU]
//! \param sig_i   collision diameter species i [code-length]
//! \param sig_j   collision diameter species j [code-length]
//! \param eps_i   LJ epsilon/kB species i [K]  (0 for hard_sphere)
//! \param eps_j   LJ epsilon/kB species j [K]  (0 for hard_sphere)
//! \param rho_i   mass density species i [code-mass/code-length^3]
//! \param rho_j   mass density species j [code-mass/code-length^3]
//! \param T       pair temperature [K]
//!
//! \return K_{ij} such that d(rho_i v_i)/dt = -K_{ij}*(v_i - v_j)
// ---------------------------------------------------------------------------
KOKKOS_FORCEINLINE_FUNCTION
Real DragCoeff(const GasDragModel model, const Real m_i, const Real m_j, const Real sig_i,
               const Real sig_j, const Real eps_i, const Real eps_j, const Real rho_i,
               const Real rho_j, const Real T) {
  using namespace detail;
  constexpr Real pi = M_PI;

  // Number densities (n = rho / m, m already in consistent mass units so
  // we leave them as proportional: K is returned per-volume so n_i * n_j carries the
  // right dimensions when m is treated as in code-mass units = AMU * AMU_to_code)
  // We compute K = n_i * n_j * sigma_ij^2 * sqrt(8*pi*kT/mu_ij) * Omega11* * (4/3)
  // where all lengths are code-length and the mass unit cancels once we note that
  // rho = n * m_code and K*dt/rho is dimensionless.  The caller provides rho and m in
  // consistent units, so n_i = rho_i / m_i_code.  Here we work with rho directly:
  //   K = (rho_i/m_i) * (rho_j/m_j) * sigma_ij^2 * sqrt(8*pi*kT_code/mu_ij_code) * O11

  const Real sij = sigma_ij(sig_i, sig_j);
  const Real mij = mu_ij(m_i, m_j); // [AMU]
  // kT in code energy = kB_code * T.  We absorb kB_code into T by requiring
  // the EOS temperatures to be in K and passing kB_code separately... but
  // the drag module does not have direct access to kB in code units.
  // Instead, use the EOS relationship: cs^2 ~ kT/mu so kT_code ~ cs^2 * mu_code.
  // For the purpose of computing the collision integral, T is in K and we write:
  //   sqrt(8*pi*kT/mu_ij) as kT_code = T [K] * (kB [erg/K] * t^2/(m*l^2)) in code.
  // To avoid importing the full unit system into a device kernel, we accept T in K
  // and the caller is responsible for converting back.
  // Practical approach: leave the sqrt factor in K-AMU units (proportional to v_th)
  // and return K in units where [K] = [density^2 * sigma^2 * sqrt(K*AMU)] which is
  // consistent as long as all kernels use the same convention.
  // For the surrogate backend this is a known-scaling model; the absolute magnitude
  // is controlled by sigma and mu which the user calibrates.
  //
  // Hard-sphere Omega^{(1,1)*} = 1.
  Real O11 = 1.0;
  if (model == GasDragModel::lj) {
    const Real eij = eps_ij(eps_i, eps_j);
    const Real Tstar = (eij > 0.0) ? T / eij : 1.0;
    O11 = Omega11Star(std::max(Tstar, 0.3));
  }

  // Relative thermal speed prefactor: sqrt(8 pi kT / mu_ij)
  // Here T carries the kB factor: v_th^2 ~ T [in code-energy-per-mass].
  // If T is in Kelvin, we need kB. Accept T in K and use m_i in AMU,
  // then the mean relative speed is sqrt(8 k_B T / (pi mu_ij)) in CGS and
  // the factor k_B/AMU has value 8.314e7 erg/(g*K) = 8.314e7 cm^2/s^2/K.
  // Since the user provides sigma in code-length and rho in code-mass/length^3,
  // the returned K scales as [1/time] * [density] which is correct for a drag rate.
  // We expose the raw formula and let the unit system flow through naturally.
  //
  // K_raw = (4/3) * (rho_i/m_i) * (rho_j/m_j) * pi * sij^2 * sqrt(8*T/(pi*mij)) * O11
  // in units where T is in K and m is in AMU.  The caller must multiply by
  // (k_B_code / AMU_code) before using K in momentum equations so:
  //   K_phys = K_raw * (kB/AMU)_in_code_units

  const Real v_rel = std::sqrt(8.0 * T / (pi * mij + Fuzz<Real>()));
  // n_i = rho_i / m_i (in AMU-inverse code volume)
  const Real n_i = rho_i / (m_i + Fuzz<Real>());
  const Real n_j = rho_j / (m_j + Fuzz<Real>());
  const Real K = (4.0 / 3.0) * n_i * n_j * pi * sij * sij * v_rel * O11;
  return std::max(K, 0.0);
}

// ---------------------------------------------------------------------------
//! \fn ThermalRelaxRate
//! \brief Thermal energy exchange rate nu_E [energy/(vol*time*K)]
//!
//! Energy exchange rate between species i and j:
//!   dE_i/dt = nu_E * (T_j - T_i)
//!
//! In the hard-sphere Chapman-Cowling theory:
//!   nu_E = 2 * (rho_i/m_i) * (rho_j/m_j) * mu_ij/(m_i+m_j)
//!          * (f_i + f_j) / (f_i * m_i + f_j * m_j) * kB * K_momentum_factor
//!
//! where f_n = dof_n/2 is the per-molecule heat capacity (in units of kB).
//! For simplicity we use nu_E = (2 mu_ij / (m_i + m_j)) * K_{ij} * cv_factor
//! following the monatomic approximation of Wang-Chang & Uhlenbeck.
// ---------------------------------------------------------------------------
KOKKOS_FORCEINLINE_FUNCTION
Real ThermalRelaxRate(const GasDragModel model, const Real m_i, const Real m_j,
                      const Real sig_i, const Real sig_j, const Real eps_i,
                      const Real eps_j, const Real dof_i, const Real dof_j,
                      const Real rho_i, const Real rho_j, const Real T) {
  using namespace detail;
  constexpr Real pi = M_PI;

  const Real sij = sigma_ij(sig_i, sig_j);
  const Real mij = mu_ij(m_i, m_j);
  const Real m_sum = m_i + m_j + Fuzz<Real>();

  Real O11 = 1.0;
  if (model == GasDragModel::lj) {
    const Real eij = eps_ij(eps_i, eps_j);
    const Real Tstar = (eij > 0.0) ? T / eij : 1.0;
    O11 = Omega11Star(std::max(Tstar, 0.3));
  }

  const Real v_rel = std::sqrt(8.0 * T / (pi * mij + Fuzz<Real>()));
  const Real n_i = rho_i / (m_i + Fuzz<Real>());
  const Real n_j = rho_j / (m_j + Fuzz<Real>());

  // Base collision rate (same as momentum drag coefficient)
  const Real K = (4.0 / 3.0) * n_i * n_j * pi * sij * sij * v_rel * O11;

  // Energy coupling factor: in the Wang-Chang & Uhlenbeck theory for
  // rigid molecules, the thermal relaxation is reduced relative to the
  // momentum transfer by a factor 2*mu_ij/m_sum * dof_correction.
  // dof_correction = (dof_i + dof_j) / (dof_i * m_j + dof_j * m_i) * m_i * m_j / (...)
  // Simplified: use the monatomic (dof=3) mixing rule.
  const Real f_i = dof_i / 2.0; // heat capacity in units of kB per molecule
  const Real f_j = dof_j / 2.0;
  const Real dof_fac =
      (f_i + f_j) / (f_i * m_j + f_j * m_i + Fuzz<Real>()) * m_i * m_j / m_sum;
  const Real nu_E = 2.0 * mij * K * dof_fac;
  return std::max(nu_E, 0.0);
}

} // namespace CollisionIntegrals
} // namespace Drag

#endif // DRAG_COLLISION_INTEGRALS_HPP_
