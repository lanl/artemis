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
#ifndef NBODY_PARTICLE_BASE_HPP_
#define NBODY_PARTICLE_BASE_HPP_

// Artemis includes
#include "artemis.hpp"

namespace NBody {

enum class SoftType { plummer, spline, none };

//----------------------------------------------------------------------------------------
//! \struct ParticleParams
//! \brief
struct ParticleParams {
  Real m;
  Real radius;
  Real rs;
  SoftType stype;
  Real racc;
  Real gamma;
  Real beta;
  int id;
  int couple;
  Real target_rad;
  int init;
  int live;
  Real live_after;
  Real x;
  Real y;
  Real z;
  Real vx;
  Real vy;
  Real vz;
};

//----------------------------------------------------------------------------------------
//! \class Particle
//! \brief
class Particle {
 public:
  int id;
  Real pos[3], vel[3];
  Real xf[3], vf[3];
  Real force[3];
  Real GM;
  Real radius;
  int couple;
  int live;
  int alive;
  Real live_after;
  Real rs, racc, gamma, beta;
  Real target_rad;
  int spline;

  KOKKOS_DEFAULTED_FUNCTION Particle() = default;
  KOKKOS_FUNCTION Particle(ParticleParams pars, const Real G, const Real Rf[3],
                           const Real Vf[3]) {
    alive = 1;
    id = pars.id;
    spline = (pars.stype == SoftType::spline);
    couple = pars.couple;
    live = (pars.live) & (couple);
    live_after = pars.live_after;
    target_rad = pars.target_rad;
    GM = G * pars.m;
    radius = pars.radius;
    rs = pars.rs;
    racc = pars.racc;
    gamma = pars.gamma;
    beta = pars.beta;
    pos[0] = pars.x;
    pos[1] = pars.y;
    pos[2] = pars.z;
    vel[0] = pars.vx;
    vel[1] = pars.vy;
    vel[2] = pars.vz;
    for (int d = 0; d < 3; d++) {
      xf[d] = Rf[d];
      vf[d] = Vf[d];
    }
  }

  KOKKOS_FUNCTION
  void RelativePosition(const Real x[3], Real *dx) const {
    for (int d = 0; d < 3; d++) {
      dx[d] = x[d] - (pos[d] - xf[d]);
    }
  }

  KOKKOS_FUNCTION
  void RelativeVelocity(const Real v[3], Real *dv) const {
    for (int d = 0; d < 3; d++) {
      dv[d] = v[d] - (vel[d] - vf[d]);
    }
  }

  KOKKOS_FUNCTION
  Real idr3(const Real dr2) const {
    const Real rs2 = SQR(rs);

    // plummer
    const Real idr3_p = 1.0 / (std::sqrt(dr2 + rs2) * (dr2 + rs2));

    // spline
    const Real dr3 = dr2 * std::sqrt(dr2);
    const Real u2 = dr2 / rs2;
    const Real u = std::sqrt(u2);
    const Real u3 = u * u2;
    const Real h3inv = 1. / (rs2 * rs);
    const Real idr3_s =
        (dr2 >= rs2) ? 1.0 / dr3
                     : ((u < 0.5) ? h3inv * (32.0 / 3.0 - 192.0 / 5.0 * u2 + 32.0 * u3)
                                  : h3inv * (64.0 / 3.0 - 48.0 * u + 192.0 / 5.0 * u2 -
                                             32.0 / 3.0 * u3 - 1.0 / (15.0 * u3)));

    // switch between the two types with spline equal to 0 or 1
    return idr3_p * (1 - spline) + spline * idr3_s;
  }

  KOKKOS_FUNCTION
  Real refine_distance(const Real *x) const {
    Real dx[3] = {Null<Real>()};
    RelativePosition(x, dx);
    const Real dr = std::sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
    return dr / (target_rad + Fuzz<Real>());
  }

  KOKKOS_FUNCTION
  void grav_accel(const Real *x, Real *g) const {
    Real dx[3] = {Null<Real>()};
    RelativePosition(x, dx);
    const Real dr2 = SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]);
    const Real idr3_ = idr3(dr2);
    for (int d = 0; d < 3; d++) {
      g[d] += -GM * idr3_ * dx[d];
    }
  }

  KOKKOS_FUNCTION
  void accrete(const Real *x, const Real den, const Real *v, const Real dt, Real *dm,
               Real *dmom, Real *dEk, Real *dEi) const {
    // Relative position and velocity with respect to the particle
    Real dx[3] = {Null<Real>()};
    Real dv[3] = {Null<Real>()};
    RelativePosition(x, dx);
    RelativeVelocity(v, dv);
    const Real dv2 = SQR(dv[0]) + SQR(dv[1]) + SQR(dv[2]);

    // Convert the gas coordinates to a spherical system centered on the particle
    Real dr[3] = {Null<Real>()};
    Real er[3] = {Null<Real>()}, et[3] = {Null<Real>()}, ep[3] = {Null<Real>()};
    CartToSph(dx, dr, er, et, ep);

    // Pull out the tangential relative velociteis
    const Real dvt = dv[0] * et[0] + dv[1] * et[1] + dv[2] * et[2];
    const Real dvp = dv[0] * ep[0] + dv[1] * ep[1] + dv[2] * ep[2];

    // This cell must be inside the accretion radius and bound to the particle
    const bool acc = ((racc > 0.0) && (dr[0] <= racc) &&
                      (-GM / (dr[0] + Fuzz<Real>()) + 0.5 * dv2 <= 0.0));

    // Smoothly transition to accretion with a quadratic ramp
    const Real ramp = SQR((racc - dr[0]) / (racc + Fuzz<Real>()));

    // mass accretion factor
    // Cap at 10% change
    const Real gdt = acc * std::min(ramp * gamma * dt, 1.0 / 9.0);
    const Real bdt = acc * std::min(ramp * beta * dt, 1.0 / 9.0);

    // mass and momentum changes
    //   drho/dt =  dS/dt
    //   d(rho v)/dt = dmx/dt
    //   dE/dt = v. dmx/dt
    //   de/dt = e*dS/dt + rho*cv*dT/dt
    //
    // NOTE(ADM): what if de/dt is set to -P*div(v) where div(v)= 1/r^2 d/dr(r^2 vr) is
    // centered on the particle? We imagine the gas that was removed contracts onto the
    // particle, releasing adiabatic heat.

    // Accrete mass
    const Real fm = -gdt / (1.0 + gdt);
    *dm += den * fm;

    // Kick in the tangential direction (i.e., radial velocity is fixed)
    const Real fp = (gdt - bdt) / ((1.0 + gdt) * (1.0 + bdt));
    const Real denp = den * (1.0 + fm);
    for (int i = 0; i < 3; i++) {
      const Real dmv = den * (fm * v[i] + fp * (dvt * et[i] + dvp * ep[i]));
      dmom[i] += dmv;
      const Real vxp = (den * v[i] + dmv) / denp;
      *dEk += 0.5 * (v[i] + vxp) * den * (vxp - v[i]) + 0.5 * den * fm * vxp * vxp;
    }
  }

  KOKKOS_FUNCTION
  void CartToSph(const Real *dx, Real *xout, Real *ex1, Real *ex2, Real *ex3) const {
    // Convert the input vector from the problem coordinate system to spherical
    const Real R = std::sqrt(SQR(dx[0]) + SQR(dx[1]));
    const Real r = std::sqrt(SQR(R) + SQR(dx[2]));
    const Real ct = dx[2] / ((r == 0) ? Fuzz<Real>() : r);
    const Real st = R / ((r == 0) ? Fuzz<Real>() : r);
    const Real cp = dx[0] / ((R == 0) ? Fuzz<Real>() : R);
    const Real sp = dx[1] / ((R == 0) ? Fuzz<Real>() : R);
    // clang-format off
    ex1[0] = st * cp;  ex1[1] = ct * cp;  ex1[2] = -sp;
    ex2[0] = st * sp;  ex2[1] = ct * sp;  ex2[2] = cp;
    ex3[0] = ct;       ex3[1] = -st;      ex3[2] = 0.0;
    // clang-format on
    xout[0] = r;
    xout[1] = std::acos(ct);
    xout[2] = std::atan2(sp, cp);

    return;
  }
};

} // namespace NBody

#endif // NBODY_PARTICLE_BASE_HPP_
