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

using Mat4x3 = std::tuple<std::array<Real, 3>, std::array<Real, 3>, std::array<Real, 3>,
                          std::array<Real, 3>>;
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

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> RelativePosition(const std::array<Real, 3> &x) const {
    auto dx = NewArray<Real, 3>();
    for (int d = 0; d < 3; d++) {
      dx[d] = x[d] - (pos[d] - xf[d]);
    }
    return dx;
  }

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 3> RelativeVelocity(const std::array<Real, 3> &v) const {
    auto dv = NewArray<Real, 3>();
    for (int d = 0; d < 3; d++) {
      dv[d] = v[d] - (vel[d] - vf[d]);
    }
    return dv;
  }

  KOKKOS_INLINE_FUNCTION
  Real idr1(const Real dr2) const {
    const Real rs2 = SQR(rs);

    // plummer
    const Real idr1_p = 1.0 / std::sqrt(dr2 + rs2 + Fuzz<Real>());

    // spline
    const Real dr1 = std::sqrt(dr2);
    const Real hinv = 1. / (rs + Fuzz<Real>());
    const Real u2 = dr2 / (rs2 + Fuzz<Real>());
    const Real u = std::sqrt(u2);
    const Real u3 = u * u2;
    const Real u4 = u2 * u2;
    const Real u5 = u4 * u;

    const Real idr1_s = (dr2 >= rs2)
                            ? 1.0 / dr1
                            : ((u < 0.5) ? hinv * (14.0 / 5.0 - 16.0 / 3.0 * u2 +
                                                   48.0 / 5.0 * u4 - 32.0 / 5.0 * u5)
                                         : hinv * (16.0 / 5.0 - 32.0 / 3.0 * u2 +
                                                   16.0 * u3 - 48.0 / 5.0 * u4 +
                                                   32.0 / 15.0 * u5 - 1.0 / (15.0 * u)));

    // switch between the two types with spline equal to 0 or 1
    return idr1_p * (1 - spline) + spline * idr1_s;
  }

  KOKKOS_INLINE_FUNCTION
  Real idr3(const Real dr2) const {
    const Real rs2 = SQR(rs);

    // plummer
    const Real idr3_p = 1.0 / (Fuzz<Real>() + std::sqrt(dr2 + rs2) * (dr2 + rs2));

    // spline
    const Real dr3 = dr2 * std::sqrt(dr2);
    const Real u2 = dr2 / (rs2 + Fuzz<Real>());
    const Real u = std::sqrt(u2);
    const Real u3 = u * u2;
    const Real h3inv = 1. / (rs2 * rs + Fuzz<Real>());
    const Real idr3_s =
        (dr2 >= rs2) ? 1.0 / dr3
                     : ((u < 0.5) ? h3inv * (32.0 / 3.0 - 192.0 / 5.0 * u2 + 32.0 * u3)
                                  : h3inv * (64.0 / 3.0 - 48.0 * u + 192.0 / 5.0 * u2 -
                                             32.0 / 3.0 * u3 - 1.0 / (15.0 * u3)));

    // switch between the two types with spline equal to 0 or 1
    return idr3_p * (1 - spline) + spline * idr3_s;
  }

  KOKKOS_INLINE_FUNCTION
  Real refine_distance(const std::array<Real, 3> &x) const {
    const auto &dx = RelativePosition(x);
    const Real dr = std::sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
    return dr / (target_rad + Fuzz<Real>());
  }

  KOKKOS_INLINE_FUNCTION
  Real grav_pot(const std::array<Real, 3> &x) const {
    const auto &dx = RelativePosition(x);
    const Real dr2 = SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]);
    return -GM * idr1(dr2);
  }

  KOKKOS_INLINE_FUNCTION
  void grav_accel(const std::array<Real, 3> &x, Real *g) const {
    const auto &dx = RelativePosition(x);
    const Real dr2 = SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]);
    const Real idr3_ = idr3(dr2);
    for (int d = 0; d < 3; d++) {
      g[d] += -GM * idr3_ * dx[d];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void accrete(const std::array<Real, 3> &x, const Real den, const std::array<Real, 3> &v,
               const std::array<Real, 3> &vb, const Real dt, Real *dm, Real *dmom,
               Real *dEk, Real *dEi) const {
    // Relative position and velocity with respect to the particle
    const std::array<Real, 3> vrel{v[0] + vb[0], v[1] + vb[1], v[2] + vb[2]};
    const auto &dx = RelativePosition(x);
    const auto &dv = RelativeVelocity(vrel);
    const Real dv2 = SQR(dv[0]) + SQR(dv[1]) + SQR(dv[2]);

    // Convert the gas coordinates to a spherical system centered on the particle
    const auto &[dr, er, et, ep] = CartToSph(dx);

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
      // Note that for a non-zero background velocity
      // the first term is v and the last terms use the v+v_back
      const Real dmv = den * (fm * v[i] + fp * (dvt * et[i] + dvp * ep[i]));
      dmom[i] += dmv;
      const Real vxp = (den * v[i] + dmv) / denp;
      *dEk += 0.5 * (v[i] + vxp) * den * (vxp - v[i]) + 0.5 * den * fm * vxp * vxp;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Mat4x3 CartToSph(const std::array<Real, 3> &dx) const {
    // Convert the input vector from the problem coordinate system to spherical
    const Real R = std::sqrt(SQR(dx[0]) + SQR(dx[1]));
    const Real r = std::sqrt(SQR(R) + SQR(dx[2]));
    const Real ct = dx[2] / (r + Fuzz<Real>());
    const Real st = R / (r + Fuzz<Real>());
    const Real cp = dx[0] / (R + Fuzz<Real>());
    const Real sp = dx[1] / (R + Fuzz<Real>());
    const std::array<Real, 3> ex1{st * cp, ct * cp, -sp};
    const std::array<Real, 3> ex2{st * sp, ct * sp, cp};
    const std::array<Real, 3> ex3{ct, -st, 0.0};
    const std::array<Real, 3> xout{r, std::acos(ct), std::atan2(sp, cp)};

    return {xout, ex1, ex2, ex3};
  }
};

} // namespace NBody

#endif // NBODY_PARTICLE_BASE_HPP_
