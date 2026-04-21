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

#define CUB(x) ((x) * (x) * (x))

namespace RotatingFrame {

//----------------------------------------------------------------------------------------
//! Declarations
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md);

template <Coordinates GEOM>
Real EstimateTimestep(parthenon::Mesh *pmesh, const Real dt_ratio);

TaskStatus RotatingFrameForce(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
TaskListStatus Advect(Mesh *pmesh, const SimTime &tm);

template <Coordinates GEOM>
TaskCollection LinearAdvectionStep(Mesh *pmesh, const SimTime &tm, const Real scdt);

template <Coordinates GEOM>
TaskStatus LagrangeRemap(MeshData<Real> *u0, const Real scdt);

struct ReconInfo {
  std::array<Real, 3> grad;
  std::array<Real, 3> xc;
  std::array<Real, 3> dx;
  geometry::BBox bnds;
  Real q;
  Real vol;

  ReconInfo() = default;
  template <Coordinates GEOM, typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION ReconInfo(const geometry::CoordParams &cpar, const V1 &v0,
                                   const V2 &vg, const int b, const int n, const int k,
                                   const int j, const int i) {
    fill<GEOM>(cpar, v0, vg, b, n, k, j, i);
  }

  template <Coordinates GEOM, typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION void fill(const geometry::CoordParams &cpar, const V1 &v0,
                                   const V2 &vg, const int b, const int n, const int k,
                                   const int j, const int i) {
    geometry::Coords<GEOM> coords(cpar, v0.GetCoordinates(b), k, j, i);
    dx = coords.GetCellWidths(vg, b, k, j, i);
    xc = coords.GetCellCenter(vg, b, k, j, i);
    vol = coords.GetVolume(vg, b, k, j, i);
    bnds = coords.bnds;

    q = v0(b, n, k, j, i);
    grad = {0.};
  }
};

//----------------------------------------------------------------------------------------
//! \fn  Real RotatingFrame::OmegaKep
//! \brief Returns Keplerian angular velocity at spherical radius R
KOKKOS_FORCEINLINE_FUNCTION Real OmegaKep(const Real gm, const Real R) {
  return std::sqrt(gm / (R * R * R));
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> RotatingFrame::BackgroundVelocity
//! \brief Returns the background velocity in coordinate basis.
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION std::array<Real, 3>
BackgroundVelocity(const Real qshear, const Real omega, const Real gm,
                   const std::array<Real, 3> &xv) {
  if constexpr (GEOM == Coordinates::cartesian) {
    return {0.0, -qshear * omega * xv[0], 0.0};
  } else if constexpr (GEOM == Coordinates::cylindrical) {
    const Real R = xv[0];
    const Real vphi = R * (OmegaKep(gm, R) - omega);
    return {0.0, vphi, 0.0};
  } else if constexpr (GEOM == Coordinates::spherical3D ||
                       GEOM == Coordinates::spherical2D) {
    const Real R = xv[0] * std::sin(xv[1]);
    const Real vphi = R * (OmegaKep(gm, xv[0]) - omega);
    return {0.0, 0.0, vphi};
  } else if constexpr (GEOM == Coordinates::axisymmetric) {
    const Real R = xv[0];
    const Real vphi = R * (OmegaKep(gm, R) - omega);
    return {0.0, 0.0, vphi};
  }
  return {0.0, 0.0, 0.0};
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> RotatingFrame::StrainRate
//! \brief Returns the strain rate in a given direction associated with the background
//!        shear/orbital velocity. Computes (grad(v) + grad(v)^T) (no div(v) term).
template <Coordinates GEOM, parthenon::CoordinateDirection XDIR>
KOKKOS_INLINE_FUNCTION std::array<Real, 3> StrainRate(const Real qshear, const Real omega,
                                                      const Real gm,
                                                      const std::array<Real, 3> &xf) {
  if constexpr (GEOM == Coordinates::cartesian) {
    if constexpr (XDIR == X1DIR) {
      // { T_1^1 , T_2^1 , T_3^1 }
      return {0.0, -qshear * omega, 0.0};
    } else if constexpr (XDIR == X2DIR) {
      // { T_1^2 , T_2^2 , T_3^2 }
      return {-qshear * omega, 0.0, 0.0};
    }
  } else if constexpr (GEOM == Coordinates::cylindrical) {
    const Real R = xf[0];
    const Real rdOdR = -1.5 * OmegaKep(gm, R);
    if constexpr (XDIR == X1DIR) {
      return {0.0, rdOdR, 0.0}; // {T_R^R, T_φ^R, T_z^R}
    } else if constexpr (XDIR == X2DIR) {
      return {rdOdR, 0.0, 0.0}; // {T_R^φ, T_φ^φ, T_z^φ}
    }
  } else if constexpr (GEOM == Coordinates::axisymmetric) {
    const Real R = xf[0];
    const Real rdOdR = -1.5 * OmegaKep(gm, R);
    if constexpr (XDIR == X1DIR) {
      return {0.0, 0.0, rdOdR}; // {T_R^R, T_z^R, T_φ^R}
    } else if constexpr (XDIR == X3DIR) {
      return {rdOdR, 0.0, 0.0}; // {T_R^φ, T_z^φ, T_φ^φ}
    }
  } else if constexpr (GEOM == Coordinates::spherical3D ||
                       GEOM == Coordinates::spherical2D) {
    const Real r = xf[0];
    const Real rdOdr = -1.5 * OmegaKep(gm, r);
    if constexpr (XDIR == X1DIR) {
      return {0.0, 0.0, rdOdr}; // T_φ^r; φ is x3
    } else if constexpr (XDIR == X3DIR) {
      return {rdOdr, 0.0, 0.0}; // T_r^φ
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
//! \struct OrbitalAdvection
//! \brief Bundles orbital advection parameters and geometry-dispatched methods.
template <Coordinates GEOM>
struct OrbitalAdvection {
  Real gm, omega_f, qshear, scdt, dwdt;
  static constexpr bool phi_is_x2 =
      (GEOM == Coordinates::cartesian || GEOM == Coordinates::cylindrical);
  static constexpr int phi_idx = phi_is_x2 ? 1 : 2;

  KOKKOS_FORCEINLINE_FUNCTION Real OmegaKep(const Real R) const {
    return std::sqrt(gm / (R * R * R));
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3>
  BackgroundVelocity(const std::array<Real, 3> &xv) const {
    if constexpr (GEOM == Coordinates::cartesian) {
      return {0.0, -qshear * omega_f * xv[0], 0.0};
    } else if constexpr (GEOM == Coordinates::cylindrical) {
      const Real R = xv[0];
      const Real vphi = R * (OmegaKep(R) - omega_f);
      return {0.0, vphi, 0.0};
    } else if constexpr (GEOM == Coordinates::spherical3D ||
                         GEOM == Coordinates::spherical2D) {
      const Real R = xv[0] * std::sin(xv[1]);
      const Real vphi = R * (OmegaKep(xv[0]) - omega_f);
      return {0.0, 0.0, vphi};
    } else if constexpr (GEOM == Coordinates::axisymmetric) {
      const Real R = xv[0];
      const Real vphi = R * (OmegaKep(R) - omega_f);
      return {0.0, 0.0, vphi};
    }
    return {0.0, 0.0, 0.0};
  }

  KOKKOS_INLINE_FUNCTION Real DeltaPhi(const std::array<Real, 3> &xc) const {
    if constexpr (GEOM == Coordinates::cartesian) {
      return dwdt * xc[0];
    } else if constexpr (GEOM == Coordinates::cylindrical) {
      return (OmegaKep(xc[0]) - omega_f) * scdt;
    } else {
      return (OmegaKep(xc[0]) - omega_f) * scdt;
    }
  }

  KOKKOS_INLINE_FUNCTION void ApplySkew(ReconInfo &ri) const {
    ri.xc[phi_idx] += DeltaPhi(ri.xc);
  }

  KOKKOS_INLINE_FUNCTION void ApplyGradSkew(ReconInfo &ri) const {
    if constexpr (GEOM == Coordinates::cartesian) {
      ri.grad[phi_idx] += dwdt * ri.grad[0];
    } else if constexpr (GEOM == Coordinates::cylindrical) {
      const Real Rc = ri.xc[0];
      const Real dOdR = -1.5 * OmegaKep(Rc) / Rc;
      ri.grad[phi_idx] += dOdR * ri.grad[0] * scdt;
    } else {
      const Real rc = ri.xc[0];
      const Real dOdr = -1.5 * OmegaKep(rc) / rc;
      ri.grad[phi_idx] += dOdr * ri.grad[0] * scdt;
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION Real PowerInt(const Real lo, const Real hi,
                                            const Real n) const {
    return (std::pow(hi, n + 1.0) - std::pow(lo, n + 1.0)) / (n + 1.0);
  }
  KOKKOS_INLINE_FUNCTION Real ComputeI0(const ReconInfo &r, const Real flip) const {
    if constexpr (GEOM == Coordinates::cartesian) {
      return flip * dwdt * r.xc[0] * r.dx[0] * r.dx[2];
    } else if constexpr (GEOM == Coordinates::cylindrical) {
      const Real alpha = std::sqrt(gm) * scdt;
      const Real omf_dt = omega_f * scdt;
      const Real Rm = r.bnds.x1[0], Rp = r.bnds.x1[1];
      const Real Q0 = r.bnds.x3[1] - r.bnds.x3[0];
      return flip * Q0 *
             (alpha * PowerInt(Rm, Rp, -0.5) - omf_dt * PowerInt(Rm, Rp, 1.0));
    } else {
      const Real alpha = std::sqrt(gm) * scdt;
      const Real omf_dt = omega_f * scdt;
      const Real rm = r.bnds.x1[0], rp = r.bnds.x1[1];
      const Real thm = r.bnds.x2[0], thp = r.bnds.x2[1];
      const Real Sth = std::cos(thm) - std::cos(thp);
      return flip * Sth *
             (alpha * PowerInt(rm, rp, 0.5) - omf_dt * PowerInt(rm, rp, 2.0));
    }
  }

  KOKKOS_INLINE_FUNCTION std::array<Real, 3> ComputeI1(const ReconInfo &r,
                                                       const Real flip, const Real vb,
                                                       const Real I0,
                                                       const int three_d) const {
    if constexpr (GEOM == Coordinates::cartesian) {
      const Real dz = r.dx[2];
      const Real dxcub =
          r.dx[0] / 3.0 *
          (SQR(r.bnds.x1[0]) + r.bnds.x1[0] * r.bnds.x1[1] + SQR(r.bnds.x1[1]));
      const Real y0 = r.bnds.x2[(vb < 0.0)];
      return {flip * dwdt * dxcub * dz,
              flip * (0.5 * SQR(dwdt) * dxcub * dz + y0 * dwdt * r.xc[0] * r.dx[0] * dz),
              three_d * r.xc[2] * I0};
    } else if constexpr (GEOM == Coordinates::cylindrical) {
      const Real alpha = std::sqrt(gm) * scdt;
      const Real omf_dt = omega_f * scdt;
      const Real Rm = r.bnds.x1[0], Rp = r.bnds.x1[1];
      const Real zm = r.bnds.x3[0], zp = r.bnds.x3[1];
      const Real Q0 = zp - zm;
      const Real Q1 = 0.5 * (SQR(zp) - SQR(zm));

      const Real R05 = PowerInt(Rm, Rp, -0.5);
      const Real R10 = PowerInt(Rm, Rp, 1.0);

      const Real I1R =
          flip * Q0 * (alpha * PowerInt(Rm, Rp, 0.5) - omf_dt * PowerInt(Rm, Rp, 2.0));
      const Real phi0 = r.bnds.x2[(vb < 0.0)];
      const Real dphi2_int = 0.5 * flip * Q0 *
                             (SQR(alpha) * PowerInt(Rm, Rp, -2.0) -
                              2.0 * alpha * omf_dt * R05 + SQR(omf_dt) * R10);
      const Real I1phi = phi0 * I0 + dphi2_int;
      const Real I1z = flip * (alpha * R05 * Q1 - omf_dt * R10 * Q1);
      return {I1R, I1phi, I1z};
    } else {
      const Real alpha = std::sqrt(gm) * scdt;
      const Real omf_dt = omega_f * scdt;
      const Real rm = r.bnds.x1[0], rp = r.bnds.x1[1];
      const Real thm = r.bnds.x2[0], thp = r.bnds.x2[1];
      const Real cthm = std::cos(thm);
      const Real cthp = std::cos(thp);
      const Real sthm = std::sin(thm);
      const Real sthp = std::sin(thp);
      const Real Sth = cthm - cthp;
      const Real Tth = (sthp - thp * cthp) - (sthm - thm * cthm);

      const Real I1r =
          flip * Sth * (alpha * PowerInt(rm, rp, 1.5) - omf_dt * PowerInt(rm, rp, 3.0));
      const Real I1th = (Sth != 0.0) ? (Tth / Sth) * I0 : r.xc[1] * I0;
      const Real phi0 = r.bnds.x3[(vb < 0.0)];
      const Real dphi2_int =
          0.5 * flip * Sth *
          (SQR(alpha) * std::log(rp / rm) - 2.0 * alpha * omf_dt * PowerInt(rm, rp, 0.5) +
           SQR(omf_dt) * PowerInt(rm, rp, 2.0));
      const Real I1phi = phi0 * I0 + dphi2_int;
      return {I1r, I1th, I1phi};
    }
  }
};

//----------------------------------------------------------------------------------------
//! \fn  RemapUpdate
//! \brief Conservative intersection-remap update using OrbitalAdvection struct.
//! Only interior cells (js <= j/jp <= je in the sweep direction) are updated.
template <Coordinates GEOM, typename V1>
KOKKOS_INLINE_FUNCTION void
RemapUpdate(const OrbitalAdvection<GEOM> &oa, const V1 &v0, const ReconInfo &rp,
            const ReconInfo &r, const Real vb, const int three_d, const int b,
            const int n, const int k, const int j, const int jp, const int i,
            const int js, const int je) {
  const Real flip = (vb < 0.0) ? -1.0 : 1.0;
  const Real I0 = oa.ComputeI0(r, flip);
  const auto I1 = oa.ComputeI1(r, flip, vb, I0, three_d);
  const Real dq =
      (rp.q - ArtemisUtils::VDot(rp.grad, rp.xc)) * I0 + ArtemisUtils::VDot(rp.grad, I1);
  if (j >= js && j <= je) v0(b, n, k, j, i) += dq / r.vol;
  if (jp >= js && jp <= je) v0(b, n, k, jp, i) -= dq / rp.vol;
}

//----------------------------------------------------------------------------------------
//! \fn  RemapCons
//! \brief Reconstruction and remap sweep along the advection direction.
template <Coordinates GEOM, Upwind UDIR, ReconstructionMethod R, typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void
RemapCons(const OrbitalAdvection<GEOM> &oa, const geometry::CoordParams &cpars,
          const V1 &v0, const V2 &vg, const int multi_d, const int three_d, const int b,
          const int k, IndexRange jb, const int i) {

  constexpr bool phi_is_x2 = OrbitalAdvection<GEOM>::phi_is_x2;

  // Extract coordinates at the start of the sweep range for vb computation
  geometry::Coords<GEOM> coords(cpars, v0.GetCoordinates(b), k, jb.s, i);
  const auto xc0 = coords.GetCellCenter(vg, b, k, jb.s, i);

  // Compute the signed background velocity at this radial position for sweep direction
  const Real vb = oa.DeltaPhi(xc0);

  // Integer gymnastics for sweep direction
  int joff = 1;
  int jstart = jb.e;
  int jend = jb.s - joff;
  if constexpr (UDIR == Upwind::l) {
    joff = -1;
    jstart = jb.s;
    jend = jb.e - joff;
  }

  const auto compare = (UDIR == Upwind::r) ? [](int j, int end) { return j >= end; }
                                           : [](int j, int end) { return j <= end; };

  ArtemisUtils::ReconGradient<GEOM, R> recon;
  ReconInfo rd, rc, ru;

  // Helper to fill ReconInfo with the correct index permutation
  auto fill_ri = [&](ReconInfo &ri, int n, int jsweep) {
    if constexpr (phi_is_x2) {
      ri.template fill<GEOM>(cpars, v0, vg, b, n, k, jsweep, i);
    } else {
      ri.template fill<GEOM>(cpars, v0, vg, b, n, jsweep, k, i);
    }
  };

  auto get_recon = [&](const ReconInfo &ri, int n, int jsweep) -> std::array<Real, 3> {
    if constexpr (phi_is_x2) {
      return recon(cpars, v0, ri.dx, multi_d, three_d, b, n, k, jsweep, i);
    } else {
      return recon(cpars, v0, ri.dx, multi_d, three_d, b, n, jsweep, k, i);
    }
  };

  for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
    fill_ri(ru, n, jstart + joff);
    fill_ri(rc, n, jstart);
    oa.ApplySkew(ru);
    oa.ApplySkew(rc);
    ru.grad = get_recon(ru, n, jstart + joff);
    rc.grad = get_recon(rc, n, jstart);
    oa.ApplyGradSkew(ru);
    oa.ApplyGradSkew(rc);

    // Execute remapping "sweep"
    for (int j = jstart; compare(j, jend); j -= joff) {
      const int jd = j - joff;
      if (compare(jd, jend)) {
        fill_ri(rd, n, jd);
        oa.ApplySkew(rd);
        rd.grad = get_recon(rd, n, jd);
        oa.ApplyGradSkew(rd);
      }
      if constexpr (phi_is_x2) {
        RemapUpdate<GEOM>(oa, v0, ru, rc, vb, three_d, b, n, k, j, j + joff, i, jb.s,
                          jb.e);
      } else {
        RemapUpdate<GEOM>(oa, v0, ru, rc, vb, three_d, b, n, j, k, j + joff, i, jb.s,
                          jb.e);
      }
      ru = rc;
      rc = rd;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  LagrangeRemapImpl
//! \brief Main par_for kernel for the Lagrange remap step.
template <Coordinates GEOM, ReconstructionMethod R, typename V1, typename V2>
TaskStatus LagrangeRemapImpl(MeshData<Real> *u0, const V1 &v0, const V2 &vg,
                             const Real dwdt, const Real gm, const Real omega_f,
                             const Real scdt) {
  PARTHENON_INSTRUMENT
  const int multi_d = u0->GetNDim() >= 2;
  const int three_d = u0->GetNDim() == 3;

  const auto &cpars = u0->GetParentPointer()
                          ->packages.Get("artemis")
                          ->template Param<geometry::CoordParams>("coord_params");
  IndexRange ib = u0->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u0->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u0->GetBoundsK(IndexDomain::interior);

  const OrbitalAdvection<GEOM> oa{gm, omega_f, 0.0, scdt, dwdt};

  if constexpr ((GEOM == Coordinates::cartesian || GEOM == Coordinates::cylindrical)) {
    // Outer loop over (k, i); inner sweep over j
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "LagrangeRemap", parthenon::DevExecSpace(), 0,
        u0->NumBlocks() - 1, kb.s, kb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &i) {
          geometry::Coords<GEOM> coords(cpars, v0.GetCoordinates(b), k, jb.s, i);
          const auto xc0 = coords.GetCellCenter(vg, b, k, jb.s, i);
          const Real vb = oa.DeltaPhi(xc0);
          if (vb < 0.0) {
            RemapCons<GEOM, Upwind::r, R>(oa, cpars, v0, vg, multi_d, three_d, b, k, jb,
                                          i);
          } else if (vb > 0.0) {
            RemapCons<GEOM, Upwind::l, R>(oa, cpars, v0, vg, multi_d, three_d, b, k, jb,
                                          i);
          }
        });
  } else if constexpr (GEOM == Coordinates::spherical3D) {
    // Outer loop over (j, i); inner sweep over k
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "LagrangeRemap", parthenon::DevExecSpace(), 0,
        u0->NumBlocks() - 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &j, const int &i) {
          geometry::Coords<GEOM> coords(cpars, v0.GetCoordinates(b), kb.s, j, i);
          const auto xc0 = coords.GetCellCenter(vg, b, kb.s, j, i);
          const Real vb = oa.DeltaPhi(xc0);
          if (vb < 0.0) {
            RemapCons<GEOM, Upwind::r, R>(oa, cpars, v0, vg, multi_d, three_d, b, j, kb,
                                          i);
          } else if (vb > 0.0) {
            RemapCons<GEOM, Upwind::l, R>(oa, cpars, v0, vg, multi_d, three_d, b, j, kb,
                                          i);
          }
        });
  } else {
    PARTHENON_FAIL("Unsupported geometry in LagrangeRemapImpl");
  }
  return TaskStatus::complete;
}
} // namespace RotatingFrame

#endif // ROTATING_FRAME_ROTATING_FRAME_HPP_
