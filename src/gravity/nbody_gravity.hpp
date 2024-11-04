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
#ifndef GRAVITY_NBODY_GRAVITY_HPP_
#define GRAVITY_NBODY_GRAVITY_HPP_

#include "nbody/nbody.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"

using ArtemisUtils::VI;

namespace Gravity {
//----------------------------------------------------------------------------------------
//! \fn  void Gravity::NBodyGravityImpl
//! \brief Process fluids, applying effects of gravitational accelerations and accretion
template <Coordinates GEOM, typename V1>
KOKKOS_INLINE_FUNCTION void
NBodyGravityImpl(V1 vmesh, const NBody::Particle &pl,
                 ArtemisUtils::array_type<Real, 7> &lforce, const int b, const int k,
                 const int j, const int i, const bool do_gas, const bool do_dust,
                 const bool do_rf, const Real omf, const Real time, const Real dt) {
  // Extract coordinates
  geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
  const auto &x = coords.GetCellCenter();

  const auto &[xcart, ex1, ex2, ex3] = coords.ConvertToCartWithVec(x);

  const auto &hx = coords.GetScaleFactors();

  const Real vol = coords.Volume();

  // Compute gravitational acceleration
  Real g[3] = {0.0};
  pl.grav_accel(xcart, g);

  // Convert back to problem coordinates
  const Real gx1 = g[0] * ex1[0] + g[1] * ex1[1] + g[2] * ex1[2];
  const Real gx2 = g[0] * ex2[0] + g[1] * ex2[1] + g[2] * ex2[2];
  const Real gx3 = g[0] * ex3[0] + g[1] * ex3[1] + g[2] * ex3[2];

  // Get the rotational velocity
  auto vf = NewArray<Real, 3>(0.0);
  if (do_rf) {
    const auto &vrot = RotatingFrame::RotationVelocity<GEOM>(x, omf);
    vf[0] = ex1[0] * vrot[0] + ex2[0] * vrot[1] + ex3[0] * vrot[2];
    vf[1] = ex1[1] * vrot[0] + ex2[1] * vrot[1] + ex3[1] * vrot[2];
    vf[2] = ex1[2] * vrot[0] + ex2[2] * vrot[1] + ex3[2] * vrot[2];
  }

  // Process the fluids
  if (do_gas) {
    for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
      const Real &dens = vmesh(b, gas::prim::density(n), k, j, i);
      const Real v[3] = {vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i),
                         vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i),
                         vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i)};
      // Transform velocity to Cartesian
      auto vcart = NewArray<Real, 3>();
      vcart[0] = ex1[0] * v[0] + ex2[0] * v[1] + ex3[0] * v[2];
      vcart[1] = ex1[1] * v[0] + ex2[1] * v[1] + ex3[1] * v[2];
      vcart[2] = ex1[2] * v[0] + ex2[2] * v[1] + ex3[2] * v[2];

      // Mass accretion
      Real dm = 0.0;
      Real dmom[3] = {0.0};
      Real dek = 0.0;
      Real dei = 0.0;
      pl.accrete(xcart, dens, vcart, vf, dt, &dm, dmom, &dek, &dei);
      const Real dmx1 = dmom[0] * ex1[0] + dmom[1] * ex1[1] + dmom[2] * ex1[2];
      const Real dmx2 = dmom[0] * ex2[0] + dmom[1] * ex2[1] + dmom[2] * ex2[2];
      const Real dmx3 = dmom[0] * ex3[0] + dmom[1] * ex3[1] + dmom[2] * ex3[2];

      Real &cdens = vmesh(b, gas::cons::density(n), k, j, i);
      Real &cmom1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i);
      Real &cmom2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i);
      Real &cmom3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i);
      Real &cet = vmesh(b, gas::cons::total_energy(n), k, j, i);
      Real &cei = vmesh(b, gas::cons::internal_energy(n), k, j, i);

      // Update conserved variables
      const Real rdt = dens * dt;
      Kokkos::atomic_add(&cdens, dm);
      Kokkos::atomic_add(&cmom1, hx[0] * (rdt * gx1 + dmx1));
      Kokkos::atomic_add(&cmom2, hx[1] * (rdt * gx2 + dmx2));
      Kokkos::atomic_add(&cmom3, hx[2] * (rdt * gx3 + dmx3));
      Kokkos::atomic_add(&cet, dek + dei + rdt * (v[0] * gx1 + v[1] * gx2 + v[2] * gx3));
      Kokkos::atomic_add(&cei, dei);

      // Track the back reaction onto the planet
      lforce.myArray[0] -= vol * dm / dt;     // Mass accreted
      lforce.myArray[1] -= g[0] * dens * vol; // X-Force due to gravity
      lforce.myArray[2] -= g[1] * dens * vol; // Y-Force...
      lforce.myArray[3] -= g[2] * dens * vol; // Z-Force...
      lforce.myArray[4] -= dmom[0] / dt;      // X-Force due to accretion
      lforce.myArray[5] -= dmom[1] / dt;      // Y-Force
      lforce.myArray[6] -= dmom[2] / dt;      // Z-Force
    }
  }

  if (do_dust) {
    for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
      const Real &dens = vmesh(b, dust::prim::density(n), k, j, i);
      const Real v[3] = {vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i),
                         vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i),
                         vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i)};
      // Transform velocity to Cartesian
      auto vcart = NewArray<Real, 3>();
      vcart[0] = ex1[0] * v[0] + ex2[0] * v[1] + ex3[0] * v[2];
      vcart[1] = ex1[1] * v[0] + ex2[1] * v[1] + ex3[1] * v[2];
      vcart[2] = ex1[2] * v[0] + ex2[2] * v[1] + ex3[2] * v[2];

      // Mass accretion
      Real dm = 0.0;
      Real dmom[3] = {0.0};
      Real dek = 0.0;
      Real dei = 0.0;
      pl.accrete(xcart, dens, vcart, vf, dt, &dm, dmom, &dek, &dei);
      const Real dmx1 = dmom[0] * ex1[0] + dmom[1] * ex1[1] + dmom[2] * ex1[2];
      const Real dmx2 = dmom[0] * ex2[0] + dmom[1] * ex2[1] + dmom[2] * ex2[2];
      const Real dmx3 = dmom[0] * ex3[0] + dmom[1] * ex3[1] + dmom[2] * ex3[2];

      Real &cdens = vmesh(b, dust::cons::density(n), k, j, i);
      Real &cmom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
      Real &cmom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
      Real &cmom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);

      const Real rdt = dens * dt;
      Kokkos::atomic_add(&cdens, dm);
      Kokkos::atomic_add(&cmom1, hx[0] * (rdt * gx1 + dmx1));
      Kokkos::atomic_add(&cmom2, hx[1] * (rdt * gx2 + dmx2));
      Kokkos::atomic_add(&cmom3, hx[2] * (rdt * gx3 + dmx3));

      // Track the back reaction onto the planet
      lforce.myArray[0] -= vol * dm / dt;     // Mass accreted
      lforce.myArray[1] -= g[0] * dens * vol; // X-Force due to gravity
      lforce.myArray[2] -= g[1] * dens * vol; // Y-Force...
      lforce.myArray[3] -= g[2] * dens * vol; // Z-Force...
      lforce.myArray[4] -= dmom[0] / dt;      // X-Force due to accretion
      lforce.myArray[5] -= dmom[1] / dt;      // Y-Force
      lforce.myArray[6] -= dmom[2] / dt;      // Z-Force
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::NBodyGravity
//! \brief Applies accelerations due to collection of point masses
template <Coordinates GEOM>
TaskStatus NBodyGravity(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  bool do_rf = artemis_pkg->template Param<bool>("do_rotating_frame");

  const Real omf =
      (do_rf) ? pm->packages.Get("rotating_frame")->template Param<Real>("omega") : 0.0;

  // Grab all the relevent nbody data
  auto &nbody_pkg = pm->packages.Get("nbody");
  auto particles = nbody_pkg->template Param<ParArray1D<NBody::Particle>>("particles");
  const auto npart = static_cast<int>(particles.size());
  auto pforce = nbody_pkg->template Param<ParArray2D<Real>>("particle_force");
  auto pforce_h = pforce.GetHostMirrorAndCopy();

  static auto desc =
      MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy, gas::cons::density,
                         dust::cons::momentum, gas::cons::internal_energy,
                         dust::cons::density, gas::prim::density, gas::prim::velocity,
                         gas::prim::sie, dust::prim::density, dust::prim::velocity>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  for (int n = 0; n < npart; n++) {
    ArtemisUtils::array_type<Real, 7> lforce;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      ArtemisUtils::array_type<Real, 7> &lsum) {
          if (particles(n).couple) {
            NBodyGravityImpl<GEOM>(vmesh, particles(n), lsum, b, k, j, i, do_gas, do_dust,
                                   do_rf, omf, time, dt);
          }
        },
        ArtemisUtils::SumMyArray<Real, Kokkos::HostSpace, 7>(lforce));

    for (int i = 0; i < 7; i++) {
      pforce_h(n, i) += lforce.myArray[i];
    }
  }

  pforce.DeepCopy(pforce_h);

  return TaskStatus::complete;
}

template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION Real NBodyPotential(geometry::Coords<GEOM> &coords,
                                           const std::array<Real, 3> &xv,
                                           ParArray1D<NBody::Particle> particles,
                                           const int npart) {
  const auto &xcart = coords.ConvertToCart(xv);
  Real pot = 0.0;
  for (auto n = 0; n < npart; ++n) {
    pot += particles(n).grav_pot(xcart);
  }
  return pot;
}

} // namespace Gravity

#endif // GRAVITY_NBODY_GRAVITY_HPP_
