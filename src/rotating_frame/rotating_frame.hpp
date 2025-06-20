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
#ifndef ROTATING_FRAME_ROTATING_FRAME_HPP_
#define ROTATING_FRAME_ROTATING_FRAME_HPP_

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "derived/fill_derived.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/fluxes/reconstruction/reconstruction.hpp"
#include "utils/integrators/artemis_integrator.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using ArtemisUtils::PLM;
using ArtemisUtils::VI;

namespace RotatingFrame {

//----------------------------------------------------------------------------------------
//! Declarations
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

TaskStatus RotatingFrameForce(MeshData<Real> *md, const Real time, const Real dt);

TaskListStatus Advect(Mesh *pmesh, const SimTime &tm,
                      parthenon::LowStorageIntegrator *integrator);

TaskCollection LinearAdvectionStep(Mesh *pmesh, const SimTime &tm,
                                   parthenon::LowStorageIntegrator *integrator);

TaskStatus UpwindAdvection(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                           parthenon::LowStorageIntegrator *integrator);

Real EstimateTimeStep(parthenon::Mesh *pmesh);

struct ReconInfo {
  std::array<Real, 3> grad;
  std::array<Real, 3> xc;
  std::array<Real, 3> dx;
  geometry::BBox bnds;
  Real q;

  KOKKOS_FUNCTION
  ReconInfo() = default;
  template <typename V1>
  KOKKOS_INLINE_FUNCTION ReconInfo(const V1 &v0, const int b, const int n, const int k,
                                   const int j, const int i) {
    fill(v0, b, n, k, j, i);
  }

  template <typename V1>
  KOKKOS_INLINE_FUNCTION void fill(const V1 &v0, const int b, const int n, const int k,
                                   const int j, const int i) {
    geometry::Coords<Coordinates::cartesian> coords(v0.GetCoordinates(b), k, j, i);
    dx = coords.GetCellWidths();
    xc = coords.GetCellCenter();
    bnds = coords.bnds;

    q = v0(b, n, k, j, i);
    grad = {0.};
  }
};

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> RotatingFrame::BackgroundVelocity
//! \brief Returns signed shear velocity
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION std::array<Real, 3>
BackgroundVelocity(const Real qshear, const Real omega, const Real x1v) {
  if constexpr (GEOM == Coordinates::cartesian) {
    return {0.0, -qshear * omega * x1v, 0.0};
  }
  return {0.0, 0.0, 0.0};
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> RotatingFrame::StrainRate
//! \brief Returns the strain rate in a given direction associated with the background
template <Coordinates GEOM, parthenon::CoordinateDirection XDIR>
KOKKOS_INLINE_FUNCTION std::array<Real, 3> StrainRate(const Real qshear, const Real omega,
                                                      const std::array<Real, 3> &xf) {
  // We are computing (grad(v) + grad(v)^T) (no div(v) term)
  if constexpr (GEOM == Coordinates::cartesian) {
    if constexpr (XDIR == X1DIR) {
      // { T_1^1 , T_2^1 , T_3^1 }
      return {0.0, -qshear * omega, 0.0};
    } else if constexpr (XDIR == X2DIR) {
      // { T_1^2 , T_2^2 , T_3^2 }
      return {-qshear * omega, 0.0, 0.0};
    }
  }
  return {0.0, 0.0, 0.0};
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> RotatingFrame::RotationVelocity
//! \brief Returns std::array of components of rotation velocity
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION std::array<Real, 3> RotationVelocity(const std::array<Real, 3> &xv,
                                                            const Real omf) {
  // TODO(AMD): implicitly multiplied by a length of 1. But that's a unit system
  // depedendent constant. Should pass an R0.
  if constexpr (GEOM == Coordinates::cartesian) return {0.0, omf, 0.0};

  // Empty constructor to get access to conversion routine
  geometry::Coords<GEOM> coords;
  const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);
  const Real vp = omf * xcyl[0];
  return {ex1[1] * vp, ex2[1] * vp, ex3[1] * vp};
}

//----------------------------------------------------------------------------------------
//! \fn  UpwindAdvance
//! \brief
template <Upwind UDIR, typename V1>
KOKKOS_INLINE_FUNCTION void
UpwindAdvance(const V1 &v0, const ReconInfo &rp, const ReconInfo &r, const Real dwdt,
              const int threed, const int b, const int n, const int k, const int j,
              const int jp, const int i) {
  Real fac = std::abs(dwdt);
  Real y0 = r.bnds.x2[(UDIR == Upwind::r)];

  const Real dz = (threed) ? r.dx[2] : 1.0;
  const Real I0 = dwdt * r.xc[0] * r.dx[0] * dz;
  const Real dxcub =
      r.dx[0] / 3. *
      (SQR(r.bnds.x1[0]) + r.bnds.x1[0] * r.bnds.x1[1] + SQR(r.bnds.x1[1]));

  const std::array<Real, 3> I1{fac * dxcub * dz,
                               0.5 * SQR(fac) * dxcub + y0 * fac * r.xc[0] * r.dx[0] * dz,
                               threed * dz * I0};

  Real dq =
      (rp.q - ArtemisUtils::VDot(rp.grad, rp.xc)) * I0 + ArtemisUtils::VDot(rp.grad, I1);

  // ?
  v0(b, n, k, j, i) += dq;
  v0(b, n, k, jp, i) -= dq;

  // v0(b,n,k,j,i) = ....;

  // I0 = qO*xc * dx*dz*dt
  // Ix = qO*dt*dz* d(x^3/3)
  // Iy = (y0*q0*dt)
  //

  // Ix = xc*dx*vp*dt + qO*dt*d(x^3/3)
  // Iy = (y0 + 0.5*vp*dt)*dx*vp*dt + y0 + 0.5*qO*dt* (qO*dt*d(x^3/3))

  // I

  // if constexpr (FLUID_TYPE == Fluid::gas) {
  //   Real &u0_dn = v0(b, gas::cons::density(n), k, j, i);
  //   Real &u0_m1 = v0(b, gas::cons::momentum(VI(n, 0)), k, j, i);
  //   Real &u0_m2 = v0(b, gas::cons::momentum(VI(n, 1)), k, j, i);
  //   Real &u0_m3 = v0(b, gas::cons::momentum(VI(n, 2)), k, j, i);
  //   Real &u0_et = v0(b, gas::cons::total_energy(n), k, j, i);
  //   Real &u0_ei = v0(b, gas::cons::internal_energy(n), k, j, i);
  //   const Real &u1_dn = v1(b, gas::cons::density(n), k, j, i);
  //   const Real &u1_m1 = v1(b, gas::cons::momentum(VI(n, 0)), k, j, i);
  //   const Real &u1_m2 = v1(b, gas::cons::momentum(VI(n, 1)), k, j, i);
  //   const Real &u1_m3 = v1(b, gas::cons::momentum(VI(n, 2)), k, j, i);
  //   const Real &u1_et = v1(b, gas::cons::total_energy(n), k, j, i);
  //   const Real &u1_ei = v1(b, gas::cons::internal_energy(n), k, j, i);

  //   u0_dn = g0 * u0_dn + g1 * u1_dn + fac * (qp[0] - q[0]);
  //   u0_m1 = g0 * u0_m1 + g1 * u1_m1 + fac * (qp[1] - q[1]);
  //   u0_m2 = g0 * u0_m2 + g1 * u1_m2 + fac * (qp[2] - q[2]);
  //   u0_m3 = g0 * u0_m3 + g1 * u1_m3 + fac * (qp[3] - q[3]);
  //   u0_et = g0 * u0_et + g1 * u1_et + fac * (qp[4] - q[4]);
  //   u0_ei = g0 * u0_ei + g1 * u1_ei + fac * (qp[5] - q[5]);
  // } else if constexpr (FLUID_TYPE == Fluid::dust) {
  //   Real &u0_dn = v0(b, dust::cons::density(n), k, j, i);
  //   Real &u0_m1 = v0(b, dust::cons::momentum(VI(n, 0)), k, j, i);
  //   Real &u0_m2 = v0(b, dust::cons::momentum(VI(n, 1)), k, j, i);
  //   Real &u0_m3 = v0(b, dust::cons::momentum(VI(n, 2)), k, j, i);
  //   const Real &u1_dn = v1(b, dust::cons::density(n), k, j, i);
  //   const Real &u1_m1 = v1(b, dust::cons::momentum(VI(n, 0)), k, j, i);
  //   const Real &u1_m2 = v1(b, dust::cons::momentum(VI(n, 1)), k, j, i);
  //   const Real &u1_m3 = v1(b, dust::cons::momentum(VI(n, 2)), k, j, i);

  //   u0_dn = g0 * u0_dn + g1 * u1_dn + fac * (qp[0] - q[0]);
  //   u0_m1 = g0 * u0_m1 + g1 * u1_m1 + fac * (qp[1] - q[1]);
  //   u0_m2 = g0 * u0_m2 + g1 * u1_m2 + fac * (qp[2] - q[2]);
  //   u0_m3 = g0 * u0_m3 + g1 * u1_m3 + fac * (qp[3] - q[3]);
  // }
}

//----------------------------------------------------------------------------------------
//! \fn  ReconSingle
//! \brief Invoke PLM reconstruction for linear advection
//! NOTE(@pdmullen): Donor cell reconstruction *could* nestle into the stencil of the
//! Upwind function, but higher order reconstruction (e.g., PPM) is incompatible... Do we
//! want to support all reconstruction options?
// template <Upwind UDIR, typename VA, typename V1, typename V2>
// KOKKOS_INLINE_FUNCTION void ReconSingle(const V1 &q, V2 &arr, const int idx, const int
// b,
//                                         const int n, const int k, const int j,
//                                         const int i) {
//   Real wl = Null<Real>(), wr = Null<Real>();
//   PLM(q(b, VA(n), k, j - 1, i), q(b, VA(n), k, j, i), q(b, VA(n), k, j + 1, i), wl,
//   wr); arr[idx] = (wr - wl) * 0.5;
//   // if constexpr (UDIR == Upwind::l) arr[idx] = wl;
//   // if constexpr (UDIR == Upwind::r) arr[idx] = wr;
// }

//----------------------------------------------------------------------------------------
//! \fn  UpwindReconstruct
//! \brief
template <typename V1>
KOKKOS_INLINE_FUNCTION std::array<Real, 3>
UpwindReconstruct(const V1 &q, const std::array<Real, 3> &dx, const int threed,
                  const int b, const int n, const int k, const int j, const int i) {

  // TODO: Replace with PPM
  std::array<Real, 3> dqdx{0.0, 0.0, 0.0};
  Real wl = Null<Real>(), wr = Null<Real>();

  PLM(q(b, n, k, j, i - 1), q(b, n, k, j, i), q(b, n, k, j, i + 1), wl, wr);
  dqdx[0] = (wr - wl) / (2.0 * dx[0]);

  PLM(q(b, n, k, j - 1, i), q(b, n, k, j, i), q(b, n, k, j + 1, i), wl, wr);
  dqdx[1] = (wr - wl) / (2.0 * dx[1]);

  PLM(q(b, n, k - threed, j, i), q(b, n, k, j, i), q(b, n, k + threed, j, i), wl, wr);
  dqdx[2] = (wr - wl) / (2.0 * dx[2]);

  return dqdx;
}

//----------------------------------------------------------------------------------------
//! \fn  Upwind
//! \brief
template <Upwind UDIR, typename V1>
KOKKOS_INLINE_FUNCTION void Upwind(const V1 &v0, const int threed, const Real dwdt,
                                   const int b, const int k, IndexRange jb, const int i) {
  // upwinding direction
  bool l_stencil = false, r_stencil = false;
  if constexpr (UDIR == Upwind::l) l_stencil = true;
  if constexpr (UDIR == Upwind::r) r_stencil = true;
  const int jstart = l_stencil * jb.s + r_stencil * jb.e;
  const int jend = l_stencil * jb.e + r_stencil * jb.s;
  const int joff = r_stencil - l_stencil;

  // reconstruct and advance
  // TODO:
  // This reconstructs the conservatives. An alternative is to hold the density in a
  // separate register and divide the conservatives.
  for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
    ReconInfo rp(v0, b, n, k, jstart + joff, i);
    ReconInfo rc(v0, b, n, k, jstart, i);
    ReconInfo rm;
    rp.grad = UpwindReconstruct(v0, rp.dx, threed, b, n, k, jstart + joff, i);
    const auto qp = v0(b, n, k, jstart + joff, i);
    rc.grad = UpwindReconstruct(v0, rc.dx, threed, b, n, k, jstart, i);
    const auto qc = v0(b, n, k, jstart, i);
    for (int j = jb.s; j < jb.e; ++j) {
      const int jswp = l_stencil * j + r_stencil * (jb.e - (j - jb.s));
      rm.fill(v0, b, n, k, jswp - joff, i);
      rm.grad = UpwindReconstruct(v0, rm.dx, threed, b, n, k, jswp - joff, i);

      UpwindAdvance<UDIR>(v0, rp, rc, dwdt, threed, b, n, k, jswp, jswp + joff, i);

      rp = rc;
      rc = rm;
    }
    UpwindAdvance<UDIR>(v0, rp, rc, dwdt, threed, b, n, k, jend, jend + joff, i);
  }
}

} // namespace RotatingFrame

#endif // ROTATING_FRAME_ROTATING_FRAME_HPP_
