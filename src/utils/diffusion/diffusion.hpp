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
#ifndef UTILS_DIFFUSION_DIFFUSION_HPP_
#define UTILS_DIFFUSION_DIFFUSION_HPP_

// Artemis includes
#include "artemis.hpp"
#include "diffusion_coeff.hpp"
#include "geometry/geometry.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace Diffusion {

template <typename SparsePackFlux>
TaskStatus ZeroDiffusionImpl(MeshData<Real> *md, SparsePackFlux vf) {

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  auto pm = md->GetParentPointer();
  const auto multi_d = (pm->ndim > 1);
  const auto three_d = (pm->ndim > 2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        for (int n = vf.GetLowerBound(b); n <= vf.GetUpperBound(b); ++n) {
          vf(b, TE::F1, n, k, j, i) = 0.0;
          vf(b, TE::F2, n, k, j, i) = 0.0;
          vf(b, TE::F3, n, k, j, i) = 0.0;
        }
        if (i == ib.e) {
          for (int n = vf.GetLowerBound(b); n <= vf.GetUpperBound(b); ++n) {
            vf(b, TE::F1, n, k, j, ib.e + 1) = 0.0;
          }
        }
        if ((j == jb.e) && (multi_d)) {
          for (int n = vf.GetLowerBound(b); n <= vf.GetUpperBound(b); ++n) {
            vf(b, TE::F2, n, k, jb.e + 1, i) = 0.0;
          }
        }
        if ((k == kb.e) && (three_d)) {
          for (int n = vf.GetLowerBound(b); n <= vf.GetUpperBound(b); ++n) {
            vf(b, TE::F3, n, kb.e + 1, j, i) = 0.0;
          }
        }
      });

  return TaskStatus::complete;
}

template <Coordinates GEOM, Fluid FLUID_TYPE, DiffType DIFF, typename PKG,
          typename SparsePackPrim>
Real EstimateTimestep(MeshData<Real> *md, DiffCoeffParams &dp, PKG &pkg, const EOS &eos,
                      SparsePackPrim vprim) {

  auto pm = md->GetParentPointer();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;

  Real min_dt = Big<Real>();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        geometry::Coords<GEOM> coords(vprim.GetCoordinates(b), k, j, i);
        Real dx[3] = {Null<Real>()};
        coords.GetCellWidths(dx);
        Real min_dx = Big<Real>();
        for (int d = 0; d < ndim; d++) {
          min_dx = std::min(min_dx, dx[d]);
        }

        for (int n = 0; n < vprim.GetSize(b, gas::prim::density()); ++n) {
          const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
          const Real &sie = vprim(b, gas::prim::sie(n), k, j, i);

          // Get the maximum diffusion coefficient (if there's more than one)
          DiffusionCoeff<DIFF, GEOM, FLUID_TYPE> diffcoeff;
          Real mu = diffcoeff.Get(dp, coords, dens, sie, eos);
          if constexpr (DIFF == DiffType::conductivity_const) {
            mu /= (dens * eos.SpecificHeatFromDensityInternalEnergy(dens, sie));
          } else if constexpr ((DIFF == DiffType::viscosity_const) ||
                               (DIFF == DiffType::viscosity_alpha)) {
            mu *= (1.0 + (dp.eta > 1.0) * (dp.eta - 1.0)) / dens;
          }
          ldt = std::min(ldt, SQR(min_dx) / (mu + Fuzz<Real>()));
        }
      },
      Kokkos::Min<Real>(min_dt));

  return min_dt / (2.0 * ndim);
}

template <Coordinates GEOM, Fluid FLUID_TYPE, typename PKG, typename SparsePackCons,
          typename SparsePackPrim, typename SparsePackFlux>
TaskStatus DiffusionUpdateImpl(MeshData<Real> *md, PKG &pkg, SparsePackCons v0,
                               SparsePackPrim p, SparsePackFlux vf,
                               const bool do_viscosity, const Real dt) {
  using TE = parthenon::TopologicalElement;

  PARTHENON_DEBUG_REQUIRE(FLUID_TYPE == Fluid::gas,
                          "Momentum diffusion only works with a gas fluid");

  auto pm = md->GetParentPointer();

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int multi_d = (pm->ndim > 1);
  const int three_d = (pm->ndim > 2);

  const bool x1dep = geometry::x1dep<GEOM>();
  const bool x2dep = (geometry::x2dep<GEOM>()) && (multi_d);
  const bool x3dep = (geometry::x3dep<GEOM>()) && (three_d);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // extract coordinate information
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
        Real ax1[2] = {0.0};
        Real ax2[2] = {0.0};
        Real ax3[2] = {0.0};
        coords.GetFaceAreaX1(ax1);
        if (multi_d) coords.GetFaceAreaX2(ax2);
        if (three_d) coords.GetFaceAreaX3(ax3);

        Real dhdx1[3] = {0.0, 0.0, 0.0};
        Real dhdx2[3] = {0.0, 0.0, 0.0};
        Real dhdx3[3] = {0.0, 0.0, 0.0};
        if (x1dep) coords.GetConnX1(dhdx1);
        if (x2dep) coords.GetConnX2(dhdx2);
        if (x3dep) coords.GetConnX3(dhdx3);

        Real xv[3] = {Null<Real>()};
        coords.GetCellCenter(xv);
        Real hx[3] = {Null<Real>()};
        coords.GetScaleFactors(hx);

        const Real vol = coords.Volume();
        const int nspecies = v0.GetSize(b, gas::cons::total_energy());
        for (int n = 0; n < nspecies; ++n) {
          const int imx1 = VI(n, 0);
          const int imx2 = VI(n, 1);
          const int imx3 = VI(n, 2);
          const int ien = nspecies * 3 + n;
          const int ieg = nspecies * 4 + n;
          Real divfxm = 0., divfym = 0., divfzm = 0.;
          // compute flux divergence
          if (do_viscosity) {
            divfxm = (ax1[0] * vf(b, TE::F1, imx1, k, j, i) -
                      ax1[1] * vf(b, TE::F1, imx1, k, j, i + 1)) +
                     multi_d * (ax2[0] * vf(b, TE::F2, imx1, k, j, i) -
                                ax2[1] * vf(b, TE::F2, imx1, k, j + multi_d, i)) +
                     three_d * (ax3[0] * vf(b, TE::F3, imx1, k, j, i) -
                                ax3[1] * vf(b, TE::F3, imx1, k + three_d, j, i));
            divfxm /= vol;
            Real src =
                dhdx1[0] * 0.5 *
                    (vf(b, TE::F1, imx1, k, j, i) + vf(b, TE::F1, imx1, k, j, i + 1)) +
                multi_d * dhdx1[1] * 0.5 *
                    (vf(b, TE::F2, imx2, k, j, i) +
                     vf(b, TE::F2, imx2, k, j + multi_d, i)) +
                three_d * dhdx1[2] * 0.5 *
                    (vf(b, TE::F3, imx3, k, j, i) +
                     vf(b, TE::F3, imx3, k + three_d, j, i));
            divfxm += x1dep * src;

            // vx2
            divfym = (ax1[0] * vf(b, TE::F1, imx2, k, j, i) -
                      ax1[1] * vf(b, TE::F1, imx2, k, j, i + 1)) +
                     multi_d * (ax2[0] * vf(b, TE::F2, imx2, k, j, i) -
                                ax2[1] * vf(b, TE::F2, imx2, k, j + multi_d, i)) +
                     three_d * (ax3[0] * vf(b, TE::F3, imx2, k, j, i) -
                                ax3[1] * vf(b, TE::F3, imx2, k + three_d, j, i));
            divfym /= vol;
            src = dhdx2[0] * 0.5 *
                      (vf(b, TE::F1, imx1, k, j, i) + vf(b, TE::F1, imx1, k, j, i + 1)) +
                  multi_d * dhdx2[1] * 0.5 *
                      (vf(b, TE::F2, imx2, k, j, i) +
                       vf(b, TE::F2, imx2, k, j + multi_d, i)) +
                  three_d * dhdx2[2] * 0.5 *
                      (vf(b, TE::F3, imx3, k, j, i) +
                       vf(b, TE::F3, imx3, k + three_d, j, i));

            divfym += x2dep * src;

            // vx3
            divfzm = (ax1[0] * vf(b, TE::F1, imx3, k, j, i) -
                      ax1[1] * vf(b, TE::F1, imx3, k, j, i + 1)) +
                     multi_d * (ax2[0] * vf(b, TE::F2, imx3, k, j, i) -
                                ax2[1] * vf(b, TE::F2, imx3, k, j + multi_d, i)) +
                     three_d * (ax3[0] * vf(b, TE::F3, imx3, k, j, i) -
                                ax3[1] * vf(b, TE::F3, imx3, k + three_d, j, i));
            divfzm /= vol;
            src = dhdx3[0] * 0.5 *
                      (vf(b, TE::F1, imx1, k, j, i) + vf(b, TE::F1, imx1, k, j, i + 1)) +
                  multi_d * dhdx3[1] * 0.5 *
                      (vf(b, TE::F2, imx2, k, j, i) +
                       vf(b, TE::F2, imx2, k, j + multi_d, i)) +
                  three_d * dhdx3[2] * 0.5 *
                      (vf(b, TE::F3, imx3, k, j, i) +
                       vf(b, TE::F3, imx3, k + three_d, j, i));

            divfzm += x3dep * src;
          }

          // energy
          Real divfe = (ax1[0] * vf(b, TE::F1, ien, k, j, i) -
                        ax1[1] * vf(b, TE::F1, ien, k, j, i + 1)) +
                       multi_d * (ax2[0] * vf(b, TE::F2, ien, k, j, i) -
                                  ax2[1] * vf(b, TE::F2, ien, k, j + multi_d, i)) +
                       three_d * (ax3[0] * vf(b, TE::F3, ien, k, j, i) -
                                  ax3[1] * vf(b, TE::F3, ien, k + three_d, j, i));
          divfe /= vol;

          v0(b, imx1, k, j, i) -= dt * divfxm;
          v0(b, imx2, k, j, i) -= dt * divfym;
          v0(b, imx3, k, j, i) -= dt * divfzm;
          v0(b, ien, k, j, i) -= dt * divfe;

          // internal energy
          // v . \Delta (rho v) - \Delta E
          // with \Delta (rho v) = div(F) / hx
          v0(b, ieg, k, j, i) -= dt * divfe - dt * (divfxm * p(b, imx1, k, j, i) / hx[0] +
                                                    divfym * p(b, imx2, k, j, i) / hx[1] +
                                                    divfzm * p(b, imx3, k, j, i) / hx[2]);
        }
      });
  return TaskStatus::complete;
}

} // namespace Diffusion
#endif // UTILS_DIFFUSION_DIFFUSION_HPP_
