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

//----------------------------------------------------------------------------------------
//! \brief Interpolate physical face-normal magnetic fields to the cell centroid.
//!
//! Face fields are area-averaged physical components. This returns a second-order
//! cell-centered physical field by interpolating each component to the coordinate-space
//! cell centroid.
template <typename COORDS, typename V, typename VG>
KOKKOS_INLINE_FUNCTION std::array<Real, 3>
FaceToCellCenteredB(const COORDS &coords, const V &vmesh, const VG &vg, const int b,
                    const int k, const int j, const int i, const bool multid,
                    const bool threed) {
  using TE = parthenon::TopologicalElement;
  const auto xv = coords.GetCellCenter(vg, b, k, j, i);
  const auto &bnds = coords.GetBounds();
  const Real bx =
      ((bnds.x1[1] - xv[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i) +
       (xv[0] - bnds.x1[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i + 1)) /
      (bnds.x1[1] - bnds.x1[0]);
  const Real by =
      multid ? (((bnds.x2[1] - xv[1]) * vmesh(b, TE::F2, field::face::B(), k, j, i) +
                 (xv[1] - bnds.x2[0]) * vmesh(b, TE::F2, field::face::B(), k, j + 1, i)) /
                (bnds.x2[1] - bnds.x2[0]))
             : vmesh(b, TE::F2, field::face::B(), k, j, i);
  const Real bz =
      threed ? (((bnds.x3[1] - xv[2]) * vmesh(b, TE::F3, field::face::B(), k, j, i) +
                 (xv[2] - bnds.x3[0]) * vmesh(b, TE::F3, field::face::B(), k + 1, j, i)) /
                (bnds.x3[1] - bnds.x3[0]))
             : vmesh(b, TE::F3, field::face::B(), k, j, i);
  return {bx, by, bz};
}

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
        const auto bcell =
            FaceToCellCenteredB(coords, vmesh, vg, b, k, j, i, multid, threed);
        vmesh(b, TE::CC, field::cell::B(0), k, j, i) = bcell[0];
        vmesh(b, TE::CC, field::cell::B(1), k, j, i) = bcell[1];
        vmesh(b, TE::CC, field::cell::B(2), k, j, i) = bcell[2];
      });
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
