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
#ifndef RADIATION_MOMENTS_MOMENTS_HPP_
#define RADIATION_MOMENTS_MOMENTS_HPP_

#include "artemis.hpp"
#include "derived/fill_derived.hpp"
#include "utils/integrators/artemis_integrator.hpp"
#include "utils/units.hpp"

namespace Moments {

//----------------------------------------------------------------------------------------
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants);
TaskStatus CalculateFluxes(MeshData<Real> *md);
TaskStatus FluxSource(MeshData<Real> *md, const Real dt);

template <Coordinates GEOM>
TaskStatus MatterCoupling(MeshData<Real> *u0, const Real dt);

void AddHistory(Coordinates coords, Params &params);

template <Coordinates GEOM>
TaskListStatus MomentsDriver(Mesh *pmesh, const SimTime &tm,
                             parthenon::LowStorageIntegrator *integrator);

template <Coordinates GEOM>
TaskCollection MomentsTasks(Mesh *pmesh, const SimTime &tm,
                            parthenon::LowStorageIntegrator *integrator);

template <Coordinates GEOM>
void InitMesh(parthenon::Mesh *pmesh);
//----------------------------------------------------------------------------------------
//! \fn Real Moments::EstimateTimeStepMesh
//! \brief Not enrolled in parthenon's determination for global dt if doing operator split
//! radiation
template <Coordinates GEOM>
Real EstimateTimeStepMesh(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &moments_pkg = pm->packages.Get("moments");
  auto &params = moments_pkg->AllParams();

  Real dxmin = Big<Real>();

  // Packing and Indexing
  static auto desc_g =
      MakePackDescriptor<geom::dx1, geom::dx2, geom::dx3>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const auto ndim = pm->ndim;
  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  // Compute minimum dx
  Real min_dx = Big<Real>();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Moments::EstimateTimestepMesh",
      DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldx_m) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vg.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths(vg, b, k, j, i);
        for (int d = 0; d < ndim; d++) {
          ldx_m = std::min(ldx_m, dx[d]);
        }
      },
      Kokkos::Min<Real>(min_dx));

  dxmin = std::min(dxmin, min_dx);

  const auto chat = params.template Get<Real>("chat");
  const auto cfl = params.template Get<Real>("cfl");
  return cfl * dxmin / chat;
}
//----------------------------------------------------------------------------------------
//! \fn Real Moments::EstimateTimeStep
//! \brief Not enrolled in parthenon's determination for global dt
template <Coordinates GEOM>
Real EstimateTimeStep(parthenon::Mesh *pmesh) {
  PARTHENON_INSTRUMENT
  auto &moments_pkg = pmesh->packages.Get("moments");
  auto &params = moments_pkg->AllParams();

  Real dxmin = Big<Real>();
  if constexpr (geometry::is_cartesian<GEOM>()) {
    for (auto const &pmb : pmesh->block_list) {
      const auto &reg = pmb->block_size;
      for (int d = 0; d < pmesh->ndim; d++) {
        const Real dx = (reg.xmax_[d] - reg.xmin_[d]) / reg.nx_[d];
        dxmin = std::min(dxmin, dx);
      }
    }
  } else {
    for (int partition = 0; partition < pmesh->DefaultNumPartitions(); partition++) {
      auto md = pmesh->mesh_data.GetOrAdd("u0", partition).get();

      // Packing and Indexing
      auto desc = MakeDefaultPackDescriptor();
      auto vmesh = desc.GetPack(md);
      static auto desc_g = MakePackDescriptor<geom::dx1, geom::dx2, geom::dx3>(
          (pmesh->resolved_packages).get());
      auto vg = desc_g.GetPack(md);

      IndexRange ib = md->GetBoundsI(IndexDomain::interior);
      IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
      IndexRange kb = md->GetBoundsK(IndexDomain::interior);
      const auto ndim = pmesh->ndim;
      const auto &cpars =
          pmesh->packages.Get("artemis")->template Param<geometry::CoordParams>(
              "coord_params");

      // Compute minimum dx
      Real min_dx = Big<Real>();
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, "Moments::EstimateTimestepMesh",
          DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldx_m) {
            // Extract coordinates
            geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
            const auto &dx = coords.GetCellWidths(vg, b, k, j, i);
            for (int d = 0; d < ndim; d++) {
              ldx_m = std::min(ldx_m, dx[d]);
            }
          },
          Kokkos::Min<Real>(min_dx));

      dxmin = std::min(dxmin, min_dx);
    }
  }

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &dxmin, 1, MPI_PARTHENON_REAL, MPI_MIN,
                                    MPI_COMM_WORLD));
#endif
  const auto chat = params.template Get<Real>("chat");
  const auto cfl = params.template Get<Real>("cfl");
  return cfl * dxmin / chat;
}

//----------------------------------------------------------------------------------------
//! \fn Real Moments::ThriceEddingtonFactor
//! \brief Computes 3x the Eddington factor given closure model
template <Closure CTYP>
KOKKOS_INLINE_FUNCTION Real ThriceEddingtonFactor(const Real f) {
  if constexpr (CTYP == Closure::p1) {
    return 1.0;
  } else if (CTYP == Closure::m1) {
    const Real f2 = f * f;
    return 3. * (3. + 4. * f2) / (5. + 2. * std::sqrt(4. - 3. * f2));
  } else {
    PARTHENON_FAIL("Closure model not recognized!");
    return 0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 6> Moments::EddingtonTensor
//! \brief Computes entries of Eddington tensor given closure model
template <Closure CTYP>
KOKKOS_INLINE_FUNCTION std::array<Real, 6>
EddingtonTensor(const std::array<Real, 3> fred) {
  if constexpr (CTYP == Closure::p1) {
    return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0};
  } else if constexpr (CTYP == Closure::m1) {
    Real fmag = std::sqrt(SQR(fred[0]) + SQR(fred[1]) + SQR(fred[2]));
    std::array<Real, 3> n{fred[0] / (fmag + Fuzz<Real>()),
                          fred[1] / (fmag + Fuzz<Real>()),
                          fred[2] / (fmag + Fuzz<Real>())};
    fmag = std::min(1.0, fmag);
    const std::array<Real, 3> f{n[0] * fmag, n[1] * fmag, n[2] * fmag};
    const Real chi = ThriceEddingtonFactor<CTYP>(fmag);
    const Real ca = (3. - chi) / 6.0;
    const Real cb = 0.5 * (chi - 1.);
    return {ca + cb * n[0] * n[0], ca + cb * n[1] * n[1], ca + cb * n[2] * n[2],
            cb * n[1] * n[2],      cb * n[0] * n[2],      cb * n[0] * n[1]};
  } else {
    PARTHENON_FAIL("Closure model not recognized!");
    return {0, 0, 0, 0, 0, 0};
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::tuple<Real, Real> Moments::WaveSpeed
//! \brief Computes wavespeed given closure model
template <Closure CTYP>
KOKKOS_INLINE_FUNCTION std::tuple<Real, Real> WaveSpeed(const Real mu, const Real f) {
  if constexpr (CTYP == Closure::p1) {
    const Real val = std::sqrt(1. / 3);
    return {-val, val};
  } else if constexpr (CTYP == Closure::m1) {
    const Real f2 = f * f;
    const Real det = 4. - 3 * f2;
    const Real sdet = std::sqrt(det);
    const Real fac = std::sqrt(2. / 3 * (det - sdet) + 2 * mu * mu * (2. - f2 - sdet));
    const Real norm = 1. / (sdet + Fuzz<Real>());
    return {norm * (mu * f - fac), norm * (mu * f + fac)};
  } else {
    PARTHENON_FAIL("Closure model not recognized!");
    return {0, 0};
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> Moments::NormalizeFlux
//! \brief Normalize radiation flux
KOKKOS_INLINE_FUNCTION
std::array<Real, 3> NormalizeFlux(const Real fx1, const Real fx2, const Real fx3) {
  Real f = std::sqrt(SQR(fx1) + SQR(fx2) + SQR(fx3));
  const Real nx1 = fx1 / (f + Fuzz<Real>());
  const Real nx2 = fx2 / (f + Fuzz<Real>());
  const Real nx3 = fx3 / (f + Fuzz<Real>());
  f = std::min(1.0, f);

  return {nx1 * f, nx2 * f, nx3 * f};
}

//----------------------------------------------------------------------------------------
//! \fn Real Moments::FleckFactor
//! \brief Returns Fleck factor dB/dE
KOKKOS_INLINE_FUNCTION
Real FleckFactor(const Real ar, const Real T, const Real cv) {
  return 4.0 * ar * T * T * T / cv;
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> Moments::SolveRadFlux
//! \brief
//!
//!   Invert this matrix:
//!
//!      | a + b*bx*bz           b*bx*by           b*bx*bz |
//!      |     b*bx*by       a + b*by*by           b*by*bz |
//!      |     b*bx*bz           b*by*bz       a + b*bz*bz |
//!
KOKKOS_INLINE_FUNCTION
std::array<Real, 3> SolveRadFlux(const Real a, const Real b,
                                 const std::array<Real, 3> &beta,
                                 const std::array<Real, 3> &rhs) {
  const Real bx2 = SQR(beta[0]);
  const Real by2 = SQR(beta[1]);
  const Real bz2 = SQR(beta[2]);
  const Real idet = 1.0 / (a * (a + b * (bx2 + by2 + bz2)));

  return {((a + b * (by2 + bz2)) * rhs[0] - b * beta[0] * beta[1] * rhs[1] -
           b * beta[0] * beta[2] * rhs[2]) *
              idet,
          (-b * beta[0] * beta[1] * rhs[0] + (a + b * (bx2 + bz2)) * rhs[1] -
           b * beta[1] * beta[2] * rhs[2]) *
              idet,
          (-b * beta[0] * beta[2] * rhs[0] - b * beta[1] * beta[2] * rhs[1] +
           (a + b * (bx2 + by2)) * rhs[2]) *
              idet};
}

} // namespace Moments

#endif // RADIATION_MOMENTS_MOMENTS_HPP_
