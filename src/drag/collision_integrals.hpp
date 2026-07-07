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
//!   K_{ij} = (4/3) n_i n_j mu_{ij} * pi * sigma_{ij}^2 *
//!            sqrt(8 k_B T / (pi mu_{ij}))
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
//! sigma (collision diameter) in code-length units, eps (LJ well depth divided by k_B) in
//! EOS temperature units, density in code-mass/code-length^3, and T in EOS temperature
//! units. The returned K [mass/(vol*time)] = [code-mass / (code-length^3 * code-time)]
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
//! \param kb_code Boltzmann constant in code units
//! \param m_i     molecular mass species i [code mass]
//! \param m_j     molecular mass species j [code mass]
//! \param sig_i   collision diameter species i [code-length]
//! \param sig_j   collision diameter species j [code-length]
//! \param eps_i   LJ epsilon/kB species i [EOS temperature units] (0 for hard_sphere)
//! \param eps_j   LJ epsilon/kB species j [EOS temperature units] (0 for hard_sphere)
//! \param rho_i   mass density species i [code-mass/code-length^3]
//! \param rho_j   mass density species j [code-mass/code-length^3]
//! \param T       pair temperature [EOS temperature units]
//!
//! \return K_{ij} such that d(rho_i v_i)/dt = -K_{ij}*(v_i - v_j)
// ---------------------------------------------------------------------------
KOKKOS_FORCEINLINE_FUNCTION
Real DragCoeff(const GasDragModel model, const Real kb_code, const Real m_i,
               const Real m_j, const Real sig_i, const Real sig_j, const Real eps_i,
               const Real eps_j, const Real rho_i, const Real rho_j, const Real T) {
  using namespace detail;
  constexpr Real pi = M_PI;

  const Real sij = sigma_ij(sig_i, sig_j);
  const Real mij = mu_ij(m_i, m_j);
  Real O11 = 1.0;
  if (model == GasDragModel::lj) {
    const Real eij = eps_ij(eps_i, eps_j);
    const Real Tstar = (eij > 0.0) ? T / eij : 1.0;
    O11 = Omega11Star(std::max(Tstar, 0.3));
  }

  const Real v_rel =
      std::sqrt(8.0 * kb_code * T / (pi * mij + Fuzz<Real>()));
  const Real n_i = rho_i / (m_i + Fuzz<Real>());
  const Real n_j = rho_j / (m_j + Fuzz<Real>());
  const Real K = (4.0 / 3.0) * n_i * n_j * mij * pi * sij * sij * v_rel * O11;
  return std::max(K, 0.0);
}

// ---------------------------------------------------------------------------
//! \fn ThermalRelaxRate
//! \brief Thermal energy exchange coefficient H_{ij} [energy/(vol*time*temperature)]
//!
//! Thermal exchange between species i and j:
//!   dE_i/dt = H_{ij} * (T_j - T_i)
//!
//! In the conservative v1 closure we scale the momentum-transfer coefficient by:
//!   omega_E = 1                      (hard spheres)
//!           = Omega22*(T*)/Omega11*(T*)  (LJ surrogate)
//!   chi_ij  = 2 mu_ij/(m_i+m_j) * 2 sqrt(dof_i dof_j)/(dof_i + dof_j)
//!   cv_pair = 2 cv_i cv_j / (cv_i + cv_j)
//! so H_{ij} = K_{ij} * omega_E * chi_ij * cv_pair.
// ---------------------------------------------------------------------------
KOKKOS_FORCEINLINE_FUNCTION
Real ThermalRelaxRate(const GasDragModel model, const Real kb_code, const Real m_i,
                      const Real m_j, const Real sig_i, const Real sig_j,
                      const Real eps_i, const Real eps_j, const Real dof_i,
                      const Real dof_j, const Real cv_i, const Real cv_j,
                      const Real rho_i, const Real rho_j, const Real T) {
  using namespace detail;

  const Real mij = mu_ij(m_i, m_j);
  const Real m_sum = m_i + m_j + Fuzz<Real>();
  const Real K =
      DragCoeff(model, kb_code, m_i, m_j, sig_i, sig_j, eps_i, eps_j, rho_i, rho_j, T);

  Real omega_E = 1.0;
  if (model == GasDragModel::lj) {
    const Real eij = eps_ij(eps_i, eps_j);
    const Real Tstar = (eij > 0.0) ? T / eij : 1.0;
    omega_E = Omega22Star(std::max(Tstar, 0.3)) /
              Omega11Star(std::max(Tstar, 0.3));
  }

  const Real chi =
      (2.0 * mij / m_sum) *
      (2.0 * std::sqrt(std::max(dof_i * dof_j, 0.0)) / (dof_i + dof_j + Fuzz<Real>()));
  const Real cv_pair = 2.0 * cv_i * cv_j / (cv_i + cv_j + Fuzz<Real>());
  const Real H = K * omega_E * chi * cv_pair;
  return std::max(H, 0.0);
}

} // namespace CollisionIntegrals
} // namespace Drag

#endif // DRAG_COLLISION_INTEGRALS_HPP_
