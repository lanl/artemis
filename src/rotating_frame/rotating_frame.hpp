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

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> RotatingFrame::BackgroundVelocity
//! \brief Returns signed shear velocity
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION std::array<Real, 3>
BackgroundVelocity(const Real qshear, const Real omega, const Real x1v) {
  if constexpr (GEOM == Coordinates::cartesian) {
    return {0.0, -qshear * omega * x1v, 0.0};
  } else {
    PARTHENON_FAIL("Shearing box currently only supports Cartesian geometries");
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::array<Real, 3> RotatingFrame::RotationVelocity
//! \brief Returns std::array of components of rotation velocity
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION std::array<Real, 3> RotationVelocity(const std::array<Real, 3> &xv,
                                                            const Real omf) {
  // Empty constructor to get access to conversion routine
  if constexpr (GEOM == Coordinates::cartesian) return {0.0, omf, 0.0};

  geometry::Coords<GEOM> coords;
  const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);
  const Real vp = omf * xcyl[0];
  return {ex1[1] * vp, ex2[1] * vp, ex3[1] * vp};
}

//----------------------------------------------------------------------------------------
//! \fn  UpwindAdvance
//! \brief
template <Fluid FLUID_TYPE, Upwind UDIR, typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void UpwindAdvance(const V1 &v0, const V1 &v1, const Real g0,
                                          const Real g1, const V2 &qp, const V2 &q,
                                          const Real wdt, const int b, const int n,
                                          const int k, const int j, const int i) {
  Real fac = Null<Real>();
  if constexpr (UDIR == Upwind::l) fac = wdt;
  if constexpr (UDIR == Upwind::r) fac = -wdt;

  if constexpr (FLUID_TYPE == Fluid::gas) {
    Real &u0_dn = v0(b, gas::cons::density(n), k, j, i);
    Real &u0_m1 = v0(b, gas::cons::momentum(VI(n, 0)), k, j, i);
    Real &u0_m2 = v0(b, gas::cons::momentum(VI(n, 1)), k, j, i);
    Real &u0_m3 = v0(b, gas::cons::momentum(VI(n, 2)), k, j, i);
    Real &u0_et = v0(b, gas::cons::total_energy(n), k, j, i);
    Real &u0_ei = v0(b, gas::cons::internal_energy(n), k, j, i);
    const Real &u1_dn = v1(b, gas::cons::density(n), k, j, i);
    const Real &u1_m1 = v1(b, gas::cons::momentum(VI(n, 0)), k, j, i);
    const Real &u1_m2 = v1(b, gas::cons::momentum(VI(n, 1)), k, j, i);
    const Real &u1_m3 = v1(b, gas::cons::momentum(VI(n, 2)), k, j, i);
    const Real &u1_et = v1(b, gas::cons::total_energy(n), k, j, i);
    const Real &u1_ei = v1(b, gas::cons::internal_energy(n), k, j, i);

    u0_dn = g0 * u0_dn + g1 * u1_dn + fac * (qp[0] - q[0]);
    u0_m1 = g0 * u0_m1 + g1 * u1_m1 + fac * (qp[1] - q[1]);
    u0_m2 = g0 * u0_m2 + g1 * u1_m2 + fac * (qp[2] - q[2]);
    u0_m3 = g0 * u0_m3 + g1 * u1_m3 + fac * (qp[3] - q[3]);
    u0_et = g0 * u0_et + g1 * u1_et + fac * (qp[4] - q[4]);
    u0_ei = g0 * u0_ei + g1 * u1_ei + fac * (qp[5] - q[5]);
  } else if constexpr (FLUID_TYPE == Fluid::dust) {
    Real &u0_dn = v0(b, dust::cons::density(n), k, j, i);
    Real &u0_m1 = v0(b, dust::cons::momentum(VI(n, 0)), k, j, i);
    Real &u0_m2 = v0(b, dust::cons::momentum(VI(n, 1)), k, j, i);
    Real &u0_m3 = v0(b, dust::cons::momentum(VI(n, 2)), k, j, i);
    const Real &u1_dn = v1(b, dust::cons::density(n), k, j, i);
    const Real &u1_m1 = v1(b, dust::cons::momentum(VI(n, 0)), k, j, i);
    const Real &u1_m2 = v1(b, dust::cons::momentum(VI(n, 1)), k, j, i);
    const Real &u1_m3 = v1(b, dust::cons::momentum(VI(n, 2)), k, j, i);

    u0_dn = g0 * u0_dn + g1 * u1_dn + fac * (qp[0] - q[0]);
    u0_m1 = g0 * u0_m1 + g1 * u1_m1 + fac * (qp[1] - q[1]);
    u0_m2 = g0 * u0_m2 + g1 * u1_m2 + fac * (qp[2] - q[2]);
    u0_m3 = g0 * u0_m3 + g1 * u1_m3 + fac * (qp[3] - q[3]);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  ReconSingle
//! \brief Invoke PLM reconstruction for linear advection
//! NOTE(@pdmullen): Donor cell reconstruction *could* nestle into the stencil of the
//! Upwind function, but higher order reconstruction (e.g., PPM) is incompatible... Do we
//! want to support all reconstruction options?
template <Upwind UDIR, typename VA, typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void ReconSingle(const V1 &q, V2 &arr, const int idx, const int b,
                                        const int n, const int k, const int j,
                                        const int i) {
  Real wl = Null<Real>(), wr = Null<Real>();
  PLM(q(b, VA(n), k, j - 1, i), q(b, VA(n), k, j, i), q(b, VA(n), k, j + 1, i), wl, wr);
  if constexpr (UDIR == Upwind::l) arr[idx] = wl;
  if constexpr (UDIR == Upwind::r) arr[idx] = wr;
}

//----------------------------------------------------------------------------------------
//! \fn  UpwindReconstruct
//! \brief
template <Fluid FLUID_TYPE, Upwind UDIR, typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void UpwindReconstruct(const V1 &vmesh, V2 &arr, const int b,
                                              const int n, const int k, const int j,
                                              const int i) {
  if constexpr (FLUID_TYPE == Fluid::gas) {
    ReconSingle<UDIR, gas::cons::density>(vmesh, arr, 0, b, n, k, j, i);
    ReconSingle<UDIR, gas::cons::momentum>(vmesh, arr, 1, b, VI(n, 0), k, j, i);
    ReconSingle<UDIR, gas::cons::momentum>(vmesh, arr, 2, b, VI(n, 1), k, j, i);
    ReconSingle<UDIR, gas::cons::momentum>(vmesh, arr, 3, b, VI(n, 2), k, j, i);
    ReconSingle<UDIR, gas::cons::total_energy>(vmesh, arr, 4, b, n, k, j, i);
    ReconSingle<UDIR, gas::cons::internal_energy>(vmesh, arr, 5, b, n, k, j, i);
  } else if constexpr (FLUID_TYPE == Fluid::dust) {
    ReconSingle<UDIR, dust::cons::density>(vmesh, arr, 0, b, n, k, j, i);
    ReconSingle<UDIR, dust::cons::momentum>(vmesh, arr, 1, b, VI(n, 0), k, j, i);
    ReconSingle<UDIR, dust::cons::momentum>(vmesh, arr, 2, b, VI(n, 1), k, j, i);
    ReconSingle<UDIR, dust::cons::momentum>(vmesh, arr, 3, b, VI(n, 2), k, j, i);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Upwind
//! \brief
template <Fluid FLUID_TYPE, Upwind UDIR, typename V1>
KOKKOS_INLINE_FUNCTION void Upwind(const V1 &v0, const V1 &v1, const Real g0,
                                   const Real g1, const Real wdt, const int b,
                                   const int k, IndexRange jb, const int i) {
  // fluid indexing
  int nu = Null<int>();
  if constexpr (FLUID_TYPE == Fluid::gas) nu = v0.GetSize(b, gas::cons::density());
  if constexpr (FLUID_TYPE == Fluid::dust) nu = v0.GetSize(b, dust::cons::density());

  // upwinding direction
  bool l_stencil = false, r_stencil = false;
  if constexpr (UDIR == Upwind::l) l_stencil = true;
  if constexpr (UDIR == Upwind::r) r_stencil = true;
  const int jstart = l_stencil * jb.s + r_stencil * jb.e;
  const int jend = l_stencil * jb.e + r_stencil * jb.s;
  const int joff = r_stencil - l_stencil;

  // reconstruct and advance
  for (int n = 0; n < nu; ++n) {
    auto uup = NewArray<Real, 6>();
    UpwindReconstruct<FLUID_TYPE, UDIR>(v0, uup, b, n, k, jstart + joff, i);
    auto uu = NewArray<Real, 6>();
    UpwindReconstruct<FLUID_TYPE, UDIR>(v0, uu, b, n, k, jstart, i);
    auto uum = NewArray<Real, 6>();
    for (int j = jb.s; j < jb.e; ++j) {
      const int jswp = l_stencil * j + r_stencil * (jb.e - (j - jb.s));
      auto uum = NewArray<Real, 6>();
      UpwindReconstruct<FLUID_TYPE, UDIR>(v0, uum, b, n, k, jswp - joff, i);
      UpwindAdvance<FLUID_TYPE, UDIR>(v0, v1, g0, g1, uup, uu, wdt, b, n, k, jswp, i);
      uup = uu;
      uu = uum;
    }
    UpwindAdvance<FLUID_TYPE, UDIR>(v0, v1, g0, g1, uup, uu, wdt, b, n, k, jend, i);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RotatingFrame::UpwindAdvection
//! \brief
static TaskStatus UpwindAdvection(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                                  parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Artemis package and params
  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  // Rotating frame package and params
  auto &rframe_pkg = pm->packages.Get("rotating_frame");
  const Real qshear = rframe_pkg->template Param<Real>("qshear");
  const Real om0 = rframe_pkg->template Param<Real>("omega");

  // Extract integrator weights
  const Real g0 = integrator->gam0[stage - 1];
  const Real g1 = integrator->gam1[stage - 1];
  const Real bdt = integrator->beta[stage - 1] * integrator->dt;

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy, dust::cons::density,
                         dust::cons::momentum>(resolved_pkgs.get());
  auto v0 = desc.GetPack(u0);
  auto v1 = desc.GetPack(u1);
  const int nblocks = u0->NumBlocks();
  IndexRange ib = u0->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u0->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u0->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpwindAdvection", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &i) {
        geometry::Coords<Coordinates::cartesian> coords(v0.GetCoordinates(b), 0, 0, i);
        const Real idx2 = 1.0 / (coords.bnds.x2[1] - coords.bnds.x2[0]);
        const Real x1v = 0.5 * (coords.bnds.x1[1] + coords.bnds.x1[0]);
        const auto ww = BackgroundVelocity<Coordinates::cartesian>(qshear, om0, x1v);
        const Real wbdt = ww[1] * idx2 * bdt;

        if (ww[1] >= 0.0) {
          if (do_gas) Upwind<Fluid::gas, Upwind::l>(v0, v1, g0, g1, wbdt, b, k, jb, i);
          if (do_dust) Upwind<Fluid::dust, Upwind::l>(v0, v1, g0, g1, wbdt, b, k, jb, i);
        } else {
          if (do_gas) Upwind<Fluid::gas, Upwind::r>(v0, v1, g0, g1, wbdt, b, k, jb, i);
          if (do_dust) Upwind<Fluid::dust, Upwind::r>(v0, v1, g0, g1, wbdt, b, k, jb, i);
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection LinearAdvectionStep
static TaskCollection LinearAdvectionStep(Mesh *pmesh, const Real time, const Real dt,
                                          parthenon::LowStorageIntegrator *integrator) {
  TaskCollection tc;
  if (!(pmesh->ndim >= 2)) return tc;

  // Construct TaskCollection
  using namespace ::parthenon::Update;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // Deep copy u0 into u1 for integrator logic
  auto &init_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = init_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
    tl.AddTask(none, ArtemisUtils::DeepCopyConservedData, u1.get(), u0.get());
  }

  // Operator split linear advection
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);

      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
      auto update =
          tl.AddTask(start_recv, UpwindAdvection, u0.get(), u1.get(), stage, integrator);
      auto set_aux = tl.AddTask(
          update, ArtemisDerived::SetAuxillaryFields<Coordinates::cartesian>, u0.get());
      auto c2p = tl.AddTask(set_aux, PreCommFillDerived<MeshData<Real>>, u0.get());
      auto bcs = parthenon::AddBoundaryExchangeTasks(c2p, tl, u0, pmesh->multilevel);
      auto p2c = tl.AddTask(bcs, FillDerived<MeshData<Real>>, u0.get());
    }
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus RotatingFrame::Advect
//! \brief Executes linear advection term for orbital advection
static TaskListStatus Advect(Mesh *pmesh, const Real time, const Real dt,
                             parthenon::LowStorageIntegrator *integrator) {
  return LinearAdvectionStep(pmesh, time, dt, integrator).Execute();
}

} // namespace RotatingFrame

#endif // ROTATING_FRAME_ROTATING_FRAME_HPP_
