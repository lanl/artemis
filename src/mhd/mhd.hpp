//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_

#include "artemis.hpp"
#include "utils/units.hpp"

namespace MHD {

KOKKOS_FORCEINLINE_FUNCTION Real MagneticEnergyDensity(const Real bx, const Real by,
                                                       const Real bz, const Real mu0) {
  return 0.5 * (SQR(bx) + SQR(by) + SQR(bz)) / mu0;
}

KOKKOS_FORCEINLINE_FUNCTION Real FastMagnetosonicSpeed(const Real bulk,
                                                       const Real density, const Real b2,
                                                       const Real bx, const Real mu0) {
  const Real wave_sum = (bulk + b2 / mu0) / density;
  const Real discriminant =
      std::max(0.0, SQR(wave_sum) - 4.0 * bulk * SQR(bx) / (mu0 * SQR(density)));
  return std::sqrt(0.5 * (wave_sum + std::sqrt(discriminant)));
}

KOKKOS_FORCEINLINE_FUNCTION Real FastMagnetosonicSpeed(const Real bulk,
                                                       const Real density, const Real bx,
                                                       const Real by, const Real bz,
                                                       const Real mu0) {
  // NOTE(AMD): bx is the component normal to the interface
  return FastMagnetosonicSpeed(bulk, density, SQR(bx) + SQR(by) + SQR(bz), bx, mu0);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants,
                                            Packages_t &packages);
TaskStatus AssembleEdgeEMF(MeshData<Real> *md);

template <typename T, Coordinates GEOM>
void SetCellCenteredMagneticFields(T *md) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  if (!do_mhd) return;

  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  static auto desc =
      MakePackDescriptor<field::face::B, field::cell::B>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  if (vmesh.GetMaxNumberOfVars() == 0) return;

  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);

  const int ndim = md->GetMeshPointer()->ndim;
  const int multid = ndim >= 2;
  const int threed = ndim == 3;
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetCellCenteredMagneticFields", parthenon::DevExecSpace(), 0,
      vmesh.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);

        const auto xv = coords.GetCellCenter(vg, b, k, j, i);
        const auto &bnds = coords.GetBounds();

        vmesh(b, TE::CC, field::cell::B(0), k, j, i) =
            ((bnds.x1[1] - xv[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i) +
             (xv[0] - bnds.x1[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i + 1)) /
            (bnds.x1[1] - bnds.x1[0]);
        vmesh(b, TE::CC, field::cell::B(1), k, j, i) =
            multid
                ? (((bnds.x2[1] - xv[1]) * vmesh(b, TE::F2, field::face::B(), k, j, i) +
                    (xv[1] - bnds.x2[0]) *
                        vmesh(b, TE::F2, field::face::B(), k, j + multid, i)) /
                   (bnds.x2[1] - bnds.x2[0]))
                : vmesh(b, TE::F2, field::face::B(), k, j, i);
        vmesh(b, TE::CC, field::cell::B(2), k, j, i) =
            threed
                ? (((bnds.x3[1] - xv[2]) * vmesh(b, TE::F3, field::face::B(), k, j, i) +
                    (xv[2] - bnds.x3[0]) *
                        vmesh(b, TE::F3, field::face::B(), k + threed, j, i)) /
                   (bnds.x3[1] - bnds.x3[0]))
                : vmesh(b, TE::F3, field::face::B(), k, j, i);
      });
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::AssembleEdgeEMFImpl
//! \brief Assemble unique edge EMFs from raw face induction fluxes on field::cell::B.
template <Coordinates G, typename PACK, typename GEO>
inline TaskStatus AssembleEdgeEMFImpl(MeshData<Real> *md, PACK v, GEO vg,
                                      const geometry::CoordParams &cpars) {
  PARTHENON_INSTRUMENT
  auto pm = md->GetParentPointer();

  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  //
  //     Fx(By) = -Ez and Fy(Bx) = Ez
  //
  //     E_i-1j--Fy_i-1j+1--E_ij+1---Fy_ij+1--E_i+1j+1
  //        |                 |                 |
  //        |                 |                 |
  //      Fx_i-1j           Fx_ij             Fx_i+1j
  //        |                 |                 |
  //        |                 |                 |
  //      E_i-1j---Fy_i-1j---E_ij------Fy_ij---E_i+1j
  //        |                 |                 |
  //        |                 |                 |
  //      Fx_i-1j-1         Fx_ij-1           Fx_i+1j-1
  //        |                 |                 |
  //        |                 |                 |
  //     E_i-1j-1--Fy_i-1j--E_ij-1-----Fy_ij---E_i+1j-1
  //
  if (multi_d) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E3", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
          const auto hx1 = coords.template GetScaleFactorsFace<X1DIR>(vg, b, k, j, i);
          const auto hx1_jm =
              coords.template GetScaleFactorsFace<X1DIR>(vg, b, k, j - 1, i);
          const auto hx2 = coords.template GetScaleFactorsFace<X2DIR>(vg, b, k, j, i);
          const auto hx2_im =
              coords.template GetScaleFactorsFace<X2DIR>(vg, b, k, j, i - 1);
          const Real h3e = coords.template GetEdgeScaleFactor<X3DIR>(vg, b, k, j, i);
          Real &emf = v.flux(b, TE::E3, field::face::B(), k, j, i);
          emf = h3e * 0.25 *
                (-v.flux(b, X1DIR, field::cell::B(1), k, j, i) / hx1[1] -
                 v.flux(b, X1DIR, field::cell::B(1), k, j - 1, i) / hx1_jm[1] +
                 v.flux(b, X2DIR, field::cell::B(0), k, j, i) / hx2[0] +
                 v.flux(b, X2DIR, field::cell::B(0), k, j, i - 1) / hx2_im[0]);
        });
  } else {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E3", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
          const auto hx1 = coords.template GetScaleFactorsFace<X1DIR>(vg, b, k, j, i);
          const Real h3e = coords.template GetEdgeScaleFactor<X3DIR>(vg, b, k, j, i);
          Real &emf = v.flux(b, TE::E3, field::face::B(), k, j, i);
          emf = -h3e * v.flux(b, X1DIR, field::cell::B(1), k, j, i) / hx1[1];
        });
  }

  if (three_d) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E2", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
          const auto hx1 = coords.template GetScaleFactorsFace<X1DIR>(vg, b, k, j, i);
          const auto hx1_km =
              coords.template GetScaleFactorsFace<X1DIR>(vg, b, k - 1, j, i);
          const auto hx3 = coords.template GetScaleFactorsFace<X3DIR>(vg, b, k, j, i);
          const auto hx3_im =
              coords.template GetScaleFactorsFace<X3DIR>(vg, b, k, j, i - 1);
          const Real h2e = coords.template GetEdgeScaleFactor<X2DIR>(vg, b, k, j, i);
          Real &emf = v.flux(b, TE::E2, field::face::B(), k, j, i);
          emf = h2e * 0.25 *
                (v.flux(b, X1DIR, field::cell::B(2), k, j, i) / hx1[2] +
                 v.flux(b, X1DIR, field::cell::B(2), k - 1, j, i) / hx1_km[2] -
                 v.flux(b, X3DIR, field::cell::B(0), k, j, i) / hx3[0] -
                 v.flux(b, X3DIR, field::cell::B(0), k, j, i - 1) / hx3_im[0]);
        });

    if (multi_d) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E1", parthenon::DevExecSpace(), 0,
          md->NumBlocks() - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
            geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
            const auto hx2 = coords.template GetScaleFactorsFace<X2DIR>(vg, b, k, j, i);
            const auto hx2_km =
                coords.template GetScaleFactorsFace<X2DIR>(vg, b, k - 1, j, i);
            const auto hx3 = coords.template GetScaleFactorsFace<X3DIR>(vg, b, k, j, i);
            const auto hx3_jm =
                coords.template GetScaleFactorsFace<X3DIR>(vg, b, k, j - 1, i);
            const Real h1e = coords.template GetEdgeScaleFactor<X1DIR>(vg, b, k, j, i);
            Real &emf = v.flux(b, TE::E1, field::face::B(), k, j, i);
            emf = h1e * 0.25 *
                  (-v.flux(b, X2DIR, field::cell::B(2), k, j, i) / hx2[2] -
                   v.flux(b, X2DIR, field::cell::B(2), k - 1, j, i) / hx2_km[2] +
                   v.flux(b, X3DIR, field::cell::B(1), k, j, i) / hx3[1] +
                   v.flux(b, X3DIR, field::cell::B(1), k, j - 1, i) / hx3_jm[1]);
          });
    }
  } else {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E2", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
          const auto hx1 = coords.template GetScaleFactorsFace<X1DIR>(vg, b, k, j, i);
          const Real h2e = coords.template GetEdgeScaleFactor<X2DIR>(vg, b, k, j, i);
          Real &emf = v.flux(b, TE::E2, field::face::B(), k, j, i);
          emf = h2e * v.flux(b, X1DIR, field::cell::B(2), k, j, i) / hx1[2];
        });

    if (multi_d) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E1", parthenon::DevExecSpace(), 0,
          md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
            geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
            const auto hx2 = coords.template GetScaleFactorsFace<X2DIR>(vg, b, k, j, i);
            const Real h1e = coords.template GetEdgeScaleFactor<X1DIR>(vg, b, k, j, i);
            Real &emf = v.flux(b, TE::E1, field::face::B(), k, j, i);
            emf = -h1e * v.flux(b, X2DIR, field::cell::B(2), k, j, i) / hx2[2];
          });
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void ArtemisUtils::ScaleMHDFlux
//! \brief Scales raw induction fluxes by scale factors associated with relevant coord sys
template <Coordinates G, int DIR, typename V3, typename V4>
KOKKOS_INLINE_FUNCTION void ScaleMHDFlux(parthenon::team_mbr_t const &member,
                                         const geometry::CoordParams &cpars, const int b,
                                         const int k, const int j, const int il,
                                         const int iu, const V4 &vg, const V3 &p) {
  if constexpr (G == Coordinates::cartesian) return;
  PARTHENON_REQUIRE(DIR > 0 && DIR <= 3, "Invalid flux direction!");

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
    geometry::Coords<G> coords(cpars, p.GetCoordinates(b), k, j, i);
    const auto &hx = coords.template GetScaleFactorsFace<DIR>(vg, b, k, j, i);
    p.flux(b, DIR, field::cell::B(0), k, j, i) *= hx[0];
    p.flux(b, DIR, field::cell::B(1), k, j, i) *= hx[1];
    p.flux(b, DIR, field::cell::B(2), k, j, i) *= hx[2];
  });

  return;
}

} // namespace MHD

#endif // MHD_MHD_HPP_
