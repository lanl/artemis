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

template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md);

Real EstimateTimestep(parthenon::Mesh *pmesh, const Real dt_ratio);

TaskStatus RotatingFrameForce(MeshData<Real> *md, const Real time, const Real dt);

TaskListStatus Advect(Mesh *pmesh, const SimTime &tm);

TaskCollection LinearAdvectionStep(Mesh *pmesh, const SimTime &tm, const Real scdt);

TaskStatus LagrangeRemap(MeshData<Real> *u0, const Real scdt);

struct ReconInfo {
  std::array<Real, 3> grad;
  std::array<Real, 3> xc;
  std::array<Real, 3> dx;
  geometry::BBox bnds;
  Real q;
  Real vol;

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
    vol = coords.Volume();
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
//! \fn  RemapUpdate
//! \brief
template <typename V1>
KOKKOS_INLINE_FUNCTION void
RemapUpdate(const V1 &v0, const ReconInfo &rp, const ReconInfo &r, const Real vb,
            const Real dwdt, const int three_d, const int b, const int n, const int k,
            const int j, const int jp, const int i) {
  // Upwind::r is vb < 0
  const Real fac = dwdt;
  const Real flip = (vb < 0.0) ? -1 : 1;
  const Real y0 = r.bnds.x2[(vb < 0.0)];

  const Real dz = r.dx[2];
  const Real I0 = flip * dwdt * r.xc[0] * r.dx[0] * dz;
  const Real dxcub =
      r.dx[0] / 3. *
      (SQR(r.bnds.x1[0]) + r.bnds.x1[0] * r.bnds.x1[1] + SQR(r.bnds.x1[1]));

  const std::array<Real, 3> I1{
      flip * fac * dxcub * dz,
      flip * (0.5 * SQR(fac) * dxcub * dz + y0 * dwdt * r.xc[0] * r.dx[0] * dz),
      three_d * flip * dz * I0};

  const Real dq =
      (rp.q - ArtemisUtils::VDot(rp.grad, rp.xc)) * I0 + ArtemisUtils::VDot(rp.grad, I1);
  v0(b, n, k, j, i) += dq / r.vol;
  v0(b, n, k, jp, i) -= dq / rp.vol;
}

//----------------------------------------------------------------------------------------
//! \fn  RemapCons
//! \brief
template <Upwind UDIR, ReconstructionMethod R, typename V1>
KOKKOS_INLINE_FUNCTION void RemapCons(const V1 &v0, const int multi_d, const int three_d,
                                      const Real dwdt, const int b, const int k,
                                      IndexRange jb, const int i) {
  // Extract coordinates
  const geometry::Coords<Coordinates::cartesian> coords(v0.GetCoordinates(b), k, jb.s, i);
  const Real vb = dwdt * (coords.bnds.x1[0] + coords.bnds.x1[1]) * 0.5;

  // Integer gymnastics
  int joff = 1;
  int jstart = jb.e;
  int jend = jb.s - joff;
  if constexpr (UDIR == Upwind::l) {
    joff = -1;
    jstart = jb.s;
    jend = jb.e - joff;
  }

  // Reconstruct and advance
  // NOTE(@adempsey): This reconstructs the conservatives. An alternative is to hold the
  // density in a separate register and divide the mass-weighted conservatives.
  const auto compare = (UDIR == Upwind::r) ? [](int j, int end) { return j >= end; }
                                           : [](int j, int end) { return j <= end; };

  ArtemisUtils::ReconGradient<R> recon;
  ReconInfo rd, rc, ru;
  for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
    ru.fill(v0, b, n, k, jstart + joff, i);
    rc.fill(v0, b, n, k, jstart, i);
    ru.grad = recon(v0, ru.dx, multi_d, three_d, b, n, k, jstart + joff, i);
    const auto qu = v0(b, n, k, jstart + joff, i);
    rc.grad = recon(v0, rc.dx, multi_d, three_d, b, n, k, jstart, i);
    const auto qc = v0(b, n, k, jstart, i);

    // Execute remapping "sweep"
    for (int j = jstart; compare(j, jend); j -= joff) {
      const int jd = j - joff;
      if (compare(jd, jend)) { // TODO(ADM): ... + (UDIR == Upwind::r)? to match earlier?
        rd.fill(v0, b, n, k, jd, i);
        rd.grad = recon(v0, rd.dx, multi_d, three_d, b, n, k, jd, i);
      }
      RemapUpdate(v0, ru, rc, vb, dwdt, three_d, b, n, k, j, j + joff, i);
      ru = rc;
      rc = rd;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  LagrangeRemapImpl
//! \brief
template <ReconstructionMethod R, typename V1>
TaskStatus LagrangeRemapImpl(MeshData<Real> *u0, const V1 &v0, const Real dwdt) {
  const int multi_d = u0->GetNDim() >= 2;
  PARTHENON_REQUIRE(multi_d, "Upwind Advection does not work in 1D");
  const int three_d = u0->GetNDim() == 3;

  IndexRange ib = u0->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u0->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u0->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "LagrangeRemap", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &i) {
        geometry::Coords<Coordinates::cartesian> coords(v0.GetCoordinates(b), k, jb.s, i);
        const Real vb = dwdt * 0.5 * (coords.bnds.x1[0] + coords.bnds.x1[1]);
        if (vb < 0.0) {
          RemapCons<Upwind::r, R>(v0, multi_d, three_d, dwdt, b, k, jb, i);
        } else if (vb > 0.0) {
          RemapCons<Upwind::l, R>(v0, multi_d, three_d, dwdt, b, k, jb, i);
        }
      });
  return TaskStatus::complete;
}
} // namespace RotatingFrame

#endif // ROTATING_FRAME_ROTATING_FRAME_HPP_
