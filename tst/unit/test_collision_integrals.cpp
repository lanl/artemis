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

// This file was created in part or in whole by generative AI

//! \file test_collision_integrals.cpp
//! \brief Catch2 unit tests for the Chapman-Cowling gas-gas collision backends.

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "drag/collision_integrals.hpp"

using Catch::Approx;
using Drag::GasDragModel;
using Drag::CollisionIntegrals::DragCoeff;
using Drag::CollisionIntegrals::ThermalRelaxRate;

namespace {

double HardSphereCoeff(const double kb_code, const double m0, const double m1,
                       const double s0, const double s1, const double rho0,
                       const double rho1, const double T) {
  const double sij = 0.5 * (s0 + s1);
  const double mij = m0 * m1 / (m0 + m1);
  const double nij0 = rho0 / m0;
  const double nij1 = rho1 / m1;
  const double vrel = std::sqrt(8.0 * kb_code * T / (M_PI * mij));
  return (4.0 / 3.0) * nij0 * nij1 * mij * M_PI * sij * sij * vrel;
}

double KineticEnergy(const double rho, const double p) { return 0.5 * p * p / rho; }

void ApplyTwoFluidImpulse(const double K, const double dt, const double rho0,
                          const double rho1, double &p0, double &p1) {
  const double v0 = p0 / rho0;
  const double v1 = p1 / rho1;
  const double J = K * dt * (v1 - v0) / (1.0 + K * dt * (1.0 / rho0 + 1.0 / rho1));
  p0 += J;
  p1 -= J;
}

double ApplyThermalExchange(const double H, const double dt, const double C0,
                            const double C1, const double T0, const double T1) {
  return H * dt * (T1 - T0) / (1.0 + H * dt * (1.0 / C0 + 1.0 / C1));
}

} // namespace

TEST_CASE("DragCoeff hard_sphere matches Chapman-Cowling form",
          "[collision_integrals][hard_sphere]") {
  const double kb_code = 1.0;
  const double m0 = 1.0;
  const double m1 = 16.0;
  const double s0 = 1.0e-3;
  const double s1 = 2.0e-3;
  const double rho0 = 1.0e-4;
  const double rho1 = 3.0e-4;
  const double T = 2.0e4;

  const double K = DragCoeff(GasDragModel::hard_sphere, kb_code, m0, m1, s0, s1, 0.0, 0.0,
                             rho0, rho1, T);
  const double K_expected = HardSphereCoeff(kb_code, m0, m1, s0, s1, rho0, rho1, T);

  REQUIRE(K == Approx(K_expected).epsilon(1.0e-12));
}

TEST_CASE("DragCoeff remains symmetric and positive",
          "[collision_integrals][hard_sphere]") {
  const double kb_code = 1.0;
  const double m0 = 1.0;
  const double m1 = 4.0;
  const double s0 = 1.0e-3;
  const double s1 = 3.0e-3;
  const double rho0 = 1.0e-4;
  const double rho1 = 2.0e-4;
  const double T = 1.0e4;

  const double K01 = DragCoeff(GasDragModel::hard_sphere, kb_code, m0, m1, s0, s1, 0.0,
                               0.0, rho0, rho1, T);
  const double K10 = DragCoeff(GasDragModel::hard_sphere, kb_code, m1, m0, s1, s0, 0.0,
                               0.0, rho1, rho0, T);

  REQUIRE(K01 >= 0.0);
  REQUIRE(K01 == Approx(K10).epsilon(1.0e-12));
}

TEST_CASE("DragCoeff hard_sphere keeps expected density sigma and temperature scaling",
          "[collision_integrals][hard_sphere]") {
  const double kb_code = 1.0;
  const double m = 2.0;
  const double s = 1.0e-3;
  const double rho = 1.0e-4;
  const double T = 1.0e4;

  const double K_base =
      DragCoeff(GasDragModel::hard_sphere, kb_code, m, m, s, s, 0.0, 0.0, rho, rho, T);
  const double K_rho = DragCoeff(GasDragModel::hard_sphere, kb_code, m, m, s, s, 0.0, 0.0,
                                 2.0 * rho, 2.0 * rho, T);
  const double K_sigma = DragCoeff(GasDragModel::hard_sphere, kb_code, m, m, 2.0 * s,
                                   2.0 * s, 0.0, 0.0, rho, rho, T);
  const double K_temp = DragCoeff(GasDragModel::hard_sphere, kb_code, m, m, s, s, 0.0,
                                  0.0, rho, rho, 4.0 * T);

  REQUIRE(K_rho == Approx(4.0 * K_base).epsilon(1.0e-12));
  REQUIRE(K_sigma == Approx(4.0 * K_base).epsilon(1.0e-12));
  REQUIRE(K_temp == Approx(2.0 * K_base).epsilon(1.0e-12));
}

TEST_CASE("Two-fluid implicit drag conserves momentum for unequal densities",
          "[collision_integrals][momentum]") {
  const double K = 5.0;
  const double dt = 0.4;
  const double rho0 = 1.0;
  const double rho1 = 3.0;
  double p0 = rho0 * 7.0;
  double p1 = rho1 * -2.0;

  const double p_sum_old = p0 + p1;
  const double dv_old = std::abs(p1 / rho1 - p0 / rho0);
  ApplyTwoFluidImpulse(K, dt, rho0, rho1, p0, p1);

  REQUIRE(p0 + p1 == Approx(p_sum_old).epsilon(1.0e-12));
  REQUIRE(std::abs(p1 / rho1 - p0 / rho0) < dv_old);
}

TEST_CASE("Two-fluid implicit drag reaches center-of-mass velocity in strong coupling",
          "[collision_integrals][momentum]") {
  const double K = 1.0e12;
  const double dt = 1.0;
  const double rho0 = 1.0;
  const double rho1 = 4.0;
  double p0 = rho0 * 9.0;
  double p1 = rho1 * -1.0;

  ApplyTwoFluidImpulse(K, dt, rho0, rho1, p0, p1);

  const double vcom = (rho0 * 9.0 + rho1 * -1.0) / (rho0 + rho1);
  REQUIRE(p0 / rho0 == Approx(vcom).epsilon(1.0e-9));
  REQUIRE(p1 / rho1 == Approx(vcom).epsilon(1.0e-9));
}

TEST_CASE("Drag kinetic energy loss becomes positive drift heat",
          "[collision_integrals][energy]") {
  const double K = 10.0;
  const double dt = 0.5;
  const double rho0 = 1.0;
  const double rho1 = 2.0;
  const double C0 = 3.0;
  const double C1 = 5.0;
  double p0 = rho0 * 8.0;
  double p1 = rho1 * -1.0;
  double e0 = 20.0;
  double e1 = 30.0;
  double u0 = 12.0;
  double u1 = 18.0;

  const double ke0_old = KineticEnergy(rho0, p0);
  const double ke1_old = KineticEnergy(rho1, p1);
  const double total_old = e0 + e1;

  ApplyTwoFluidImpulse(K, dt, rho0, rho1, p0, p1);

  const double ke0_new = KineticEnergy(rho0, p0);
  const double ke1_new = KineticEnergy(rho1, p1);
  const double dke0 = ke0_new - ke0_old;
  const double dke1 = ke1_new - ke1_old;
  const double qdrag = -(dke0 + dke1);

  REQUIRE(qdrag > 0.0);

  e0 += dke0;
  e1 += dke1;
  const double q0 = qdrag * C0 / (C0 + C1);
  const double q1 = qdrag * C1 / (C0 + C1);
  e0 += q0;
  e1 += q1;
  u0 += q0;
  u1 += q1;

  REQUIRE(e0 + e1 == Approx(total_old).epsilon(1.0e-12));
  REQUIRE((u0 - 12.0) + (u1 - 18.0) == Approx(qdrag).epsilon(1.0e-12));
}

TEST_CASE("ThermalRelaxRate is positive and symmetric as a pair coefficient",
          "[collision_integrals][thermal]") {
  const double kb_code = 1.0;
  const double m0 = 1.0;
  const double m1 = 16.0;
  const double s0 = 1.0e-3;
  const double s1 = 2.0e-3;
  const double e0 = 0.0;
  const double e1 = 0.0;
  const double dof0 = 3.0;
  const double dof1 = 3.0;
  const double cv0 = 1.5;
  const double cv1 = 1.5 / 16.0;
  const double rho0 = 1.0e-4;
  const double rho1 = 2.0e-4;
  const double T = 1.0e4;

  const double H01 = ThermalRelaxRate(GasDragModel::hard_sphere, kb_code, m0, m1, s0, s1,
                                      e0, e1, dof0, dof1, cv0, cv1, rho0, rho1, T);
  const double H10 = ThermalRelaxRate(GasDragModel::hard_sphere, kb_code, m1, m0, s1, s0,
                                      e1, e0, dof1, dof0, cv1, cv0, rho1, rho0, T);

  REQUIRE(H01 >= 0.0);
  REQUIRE(H01 == Approx(H10).epsilon(1.0e-12));
}

TEST_CASE("Implicit thermal exchange conserves energy and reaches weighted equilibrium",
          "[collision_integrals][thermal]") {
  const double H = 1.0e12;
  const double dt = 1.0;
  const double C0 = 2.0;
  const double C1 = 5.0;
  const double T0 = 12.0;
  const double T1 = 4.0;

  const double q = ApplyThermalExchange(H, dt, C0, C1, T0, T1);
  const double T0_new = T0 + q / C0;
  const double T1_new = T1 - q / C1;
  const double Teq = (C0 * T0 + C1 * T1) / (C0 + C1);

  REQUIRE(C0 * T0_new + C1 * T1_new == Approx(C0 * T0 + C1 * T1).epsilon(1.0e-12));
  REQUIRE(T0_new == Approx(Teq).epsilon(1.0e-9));
  REQUIRE(T1_new == Approx(Teq).epsilon(1.0e-9));
}

TEST_CASE(
    "LJ collision integrals enhance cold coupling and approach hard-sphere at high T*",
    "[collision_integrals][lj]") {
  const double kb_code = 1.0;
  const double m = 1.0;
  const double s = 1.0e-3;
  const double eps = 1.0e4;
  const double dof = 3.0;
  const double cv = 1.5;
  const double rho = 1.0e-4;
  const double T_low = 5.0e3;
  const double T_high = 1.0e6;

  const double K_lj_low =
      DragCoeff(GasDragModel::lj, kb_code, m, m, s, s, eps, eps, rho, rho, T_low);
  const double K_hs_low = DragCoeff(GasDragModel::hard_sphere, kb_code, m, m, s, s, 0.0,
                                    0.0, rho, rho, T_low);
  const double K_lj_high =
      DragCoeff(GasDragModel::lj, kb_code, m, m, s, s, eps, eps, rho, rho, T_high);
  const double K_hs_high = DragCoeff(GasDragModel::hard_sphere, kb_code, m, m, s, s, 0.0,
                                     0.0, rho, rho, T_high);

  const double H_lj_low = ThermalRelaxRate(GasDragModel::lj, kb_code, m, m, s, s, eps,
                                           eps, dof, dof, cv, cv, rho, rho, T_low);
  const double H_lj_high = ThermalRelaxRate(GasDragModel::lj, kb_code, m, m, s, s, eps,
                                            eps, dof, dof, cv, cv, rho, rho, T_high);

  REQUIRE(K_lj_low / K_hs_low > 1.0);
  REQUIRE(K_lj_low / K_hs_low > K_lj_high / K_hs_high);
  REQUIRE(H_lj_low / K_lj_low > 0.0);
  REQUIRE(H_lj_high / K_lj_high > 0.0);
}
