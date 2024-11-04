//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_DIFFUSION_THERMAL_DIFFUSION_HPP_
#define UTILS_DIFFUSION_THERMAL_DIFFUSION_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "diffusion_coeff.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace Diffusion {

template <Coordinates GEOM, Fluid FLUID_TYPE, DiffType DIFF, typename PKG,
          typename SparsePackPrim, typename SparsePackFlux>
TaskStatus ThermalFluxImpl(MeshData<Real> *md, DiffCoeffParams dp, PKG &pkg,
                           SparsePackPrim vprim, SparsePackFlux vf) {
  // Set heat flux
  // dE/dt = div(q)
  // q = - K . grad(T)
  // K = rho cp * \chi

  PARTHENON_REQUIRE(FLUID_TYPE == Fluid::gas,
                    "thermal diffusion only works with a gas fluid");

  auto pm = md->GetParentPointer();
  auto eos_d = pkg->template Param<EOS>("eos_d");

  const int scr_level = pkg->template Param<int>("scr_level");

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  const int ncells1 = (ib.e - ib.s + 1) + 2 * parthenon::Globals::nghost;

  const int multi_d = (pm->ndim >= 2);
  const int three_d = (pm->ndim == 3);

  int il = ib.s, iu = ib.e + 1;
  int jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  int scr_size = ScratchPad1D<Real>::shmem_size(ncells1);

  const bool avg = (dp.avg == DiffAvg::arithmetic);
  const bool havg = (dp.avg == DiffAvg::harmonic);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scr_size,
      scr_level, 0, md->NumBlocks() - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j) {
        ScratchPad1D<Real> kappa(mbr.team_scratch(scr_level), ncells1);
        const int nspecies = vprim.GetSize(b, gas::prim::density());
        auto pco = vprim.GetCoordinates(b);
        for (int n = 0; n < nspecies; n++) {

          // this returns conductivity not thermal diffusivity
          DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
          diffcoeff.evaluate(dp, mbr, b, n, k, j, il - 1, iu, vprim, eos_d, kappa);

          mbr.team_barrier();

          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, mbr, il, iu, [&](const int i) {
                // F = -K grad(T)
                geometry::Coords<GEOM> coords(pco, k, j, i);
                geometry::Coords<GEOM> coords_m(pco, k, j, i - 1);
                const auto &xv = coords.GetCellCenter();
                const auto &xv_m = coords_m.GetCellCenter();
                const Real dx1 = coords.Distance(xv, xv_m);

                const Real T = eos_d.TemperatureFromDensityInternalEnergy(
                    vprim(b, gas::prim::density(n), k, j, i),
                    vprim(b, gas::prim::sie(n), k, j, i));

                const Real Tm = eos_d.TemperatureFromDensityInternalEnergy(
                    vprim(b, gas::prim::density(n), k, j, i - 1),
                    vprim(b, gas::prim::sie(n), k, j, i - 1));

                const Real kcond =
                    avg * FaceAverage<DiffAvg::arithmetic>(kappa(i), kappa(i - 1)) +
                    havg * FaceAverage<DiffAvg::harmonic>(kappa(i), kappa(i - 1));

                vf(b, TE::F1, gas::diff::energy(n), k, j, i) += kcond * (T - Tm) / dx1;
              });
        }
      });

  // X2-Flux
  if (multi_d) {
    jl = jb.s - 1, ju = jb.e + 1;
    il = ib.s, iu = ib.e, kl = kb.s, ku = kb.e;
    scr_size = ScratchPad1D<Real>::shmem_size(ncells1) * 2;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scr_size,
        scr_level, 0, md->NumBlocks() - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k) {
          ScratchPad1D<Real> scr1(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr2(mbr.team_scratch(scr_level), ncells1);

          const int nspecies = vprim.GetSize(b, gas::prim::density());
          auto pco = vprim.GetCoordinates(b);
          for (int n = 0; n < nspecies; n++) {
            for (int j = jl; j <= ju; ++j) {
              // permute scratch
              auto kappa = scr1;
              auto kappa_jm1 = scr2;
              if ((j % 2) == 0) {
                kappa = scr2;
                kappa_jm1 = scr1;
              }

              DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
              diffcoeff.evaluate(dp, mbr, b, n, k, j, il, iu, vprim, eos_d, kappa);

              mbr.team_barrier();
              if (j > jl) {
                parthenon::par_for_inner(
                    DEFAULT_INNER_LOOP_PATTERN, mbr, il, iu, [&](const int i) {
                      // F = -kappa * cv grad(T)
                      geometry::Coords<GEOM> coords(pco, k, j, i);
                      geometry::Coords<GEOM> coords_m(pco, k, j - 1, i);
                      const auto &xv = coords.GetCellCenter();
                      const auto &xv_m = coords_m.GetCellCenter();
                      const Real dx2 = coords.Distance(xv, xv_m);

                      const Real T = eos_d.TemperatureFromDensityInternalEnergy(
                          vprim(b, gas::prim::density(n), k, j, i),
                          vprim(b, gas::prim::sie(n), k, j, i));

                      const Real Tm = eos_d.TemperatureFromDensityInternalEnergy(
                          vprim(b, gas::prim::density(n), k, j - 1, i),
                          vprim(b, gas::prim::sie(n), k, j - 1, i));

                      const Real kcond =
                          avg * FaceAverage<DiffAvg::arithmetic>(kappa(i), kappa_jm1(i)) +
                          havg * FaceAverage<DiffAvg::harmonic>(kappa(i), kappa_jm1(i));

                      vf(b, TE::F2, gas::diff::energy(n), k, j, i) +=
                          kcond * (T - Tm) / dx2;
                    });
              }
            }
          }
        });
  }

  // X3-Flux
  if (three_d) {
    kl = kb.s - 1, ku = kb.e + 1;
    il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e;
    scr_size = ScratchPad1D<Real>::shmem_size(ncells1) * 2;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), scr_size,
        scr_level, 0, md->NumBlocks() - 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int j) {
          ScratchPad1D<Real> scr1(mbr.team_scratch(scr_level), ncells1);
          ScratchPad1D<Real> scr2(mbr.team_scratch(scr_level), ncells1);

          const int nspecies = vprim.GetSize(b, gas::prim::density());
          auto pco = vprim.GetCoordinates(b);
          for (int n = 0; n < nspecies; n++) {
            for (int k = kl; k <= ku; ++k) {
              // permute scratch
              auto kappa = scr1;
              auto kappa_km1 = scr2;
              if ((k % 2) == 0) {
                kappa = scr2;
                kappa_km1 = scr1;
              }

              // 2. Viscosity values. No barrier
              DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
              diffcoeff.evaluate(dp, mbr, b, n, k, j, il, iu, vprim, eos_d, kappa_km1);

              mbr.team_barrier();
              if (k > kl) {
                parthenon::par_for_inner(
                    DEFAULT_INNER_LOOP_PATTERN, mbr, il, iu, [&](const int i) {
                      // F = -kappa * cv grad(T)
                      geometry::Coords<GEOM> coords(pco, k, j, i);
                      geometry::Coords<GEOM> coords_m(pco, k - 1, j, i);
                      const auto &xv = coords.GetCellCenter();
                      const auto &xv_m = coords_m.GetCellCenter();
                      const Real dx3 = coords.Distance(xv, xv_m);

                      const Real T = eos_d.TemperatureFromDensityInternalEnergy(
                          vprim(b, gas::prim::density(n), k, j, i),
                          vprim(b, gas::prim::sie(n), k, j, i));

                      const Real Tm = eos_d.TemperatureFromDensityInternalEnergy(
                          vprim(b, gas::prim::density(n), k - 1, j, i),
                          vprim(b, gas::prim::sie(n), k - 1, j, i));

                      const Real kcond =
                          avg * FaceAverage<DiffAvg::arithmetic>(kappa(i), kappa_km1(i)) +
                          havg * FaceAverage<DiffAvg::harmonic>(kappa(i), kappa_km1(i));

                      vf(b, TE::F3, gas::diff::energy(n), k, j, i) +=
                          kcond * (T - Tm) / dx3;
                    });
              }
            }
          }
        });
  }
  return TaskStatus::complete;
}

} // namespace Diffusion

#endif // UTILS_DIFFUSION_MOMENTUM_DIFFUSION_HPP_
