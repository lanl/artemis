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
#ifndef MHD_EMF_HPP_
#define MHD_EMF_HPP_

#include "artemis.hpp"

namespace MHD {

//----------------------------------------------------------------------------------------
//! \brief Return the physical ideal-MHD electric field from cell-centered primitives.
template <int EDGE, typename PACK>
KOKKOS_INLINE_FUNCTION Real CellEMF(const PACK &v, const int b, const int k, const int j,
                                    const int i) {
  const Real v1 = v(b, gas::prim::velocity(0), k, j, i);
  const Real v2 = v(b, gas::prim::velocity(1), k, j, i);
  const Real v3 = v(b, gas::prim::velocity(2), k, j, i);
  const Real b1 = v(b, field::cell::B(0), k, j, i);
  const Real b2 = v(b, field::cell::B(1), k, j, i);
  const Real b3 = v(b, field::cell::B(2), k, j, i);
  if constexpr (EDGE == X1DIR) return v3 * b2 - v2 * b3;
  if constexpr (EDGE == X2DIR) return v1 * b3 - v3 * b1;
  return v2 * b1 - v1 * b2;
}

//----------------------------------------------------------------------------------------
//! \brief Select the Gardiner--Stone donor transverse derivative from a mass flux.
KOKKOS_INLINE_FUNCTION Real UpwindEMFGradient(const Real mass_flux, const Real left,
                                              const Real right) {
  const int sign = (mass_flux > 0.0) - (mass_flux < 0.0);
  return 0.5 * ((1.0 + sign) * left + (1.0 - sign) * right);
}

//----------------------------------------------------------------------------------------
//! \brief Gardiner--Stone contact-upwind EMF at an edge.
//!
//! Face values are extrapolated to the edge with transverse gradients selected
//! by the corresponding Riemann mass-flux sign (GS05 Eq. 51), then averaged.
template <int E1, Coordinates G, typename PACK, typename GEO>
KOKKOS_INLINE_FUNCTION Real UpwindEMF(const PACK &v, const GEO &vg,
                                      const geometry::CoordParams &cpars, const int block,
                                      const int k, const int j, const int i) {
  constexpr int E2 = E1 == X1DIR ? X2DIR : X1DIR;
  constexpr int E3 = E1 == X3DIR ? X2DIR : X3DIR;
  constexpr int e2_bcomp = E3 - 1;
  constexpr int e3_bcomp = E2 - 1;
  constexpr int e2_sign = 1 - 2 * (E1 % 2);
  constexpr int e3_sign = -e2_sign;
  const std::array<int, 3> pp{k, j, i};
  auto pm = pp;
  auto qm = pp;
  auto mm = pp;
  pm[3 - E2] -= 1;
  qm[3 - E3] -= 1;
  mm[3 - E2] -= 1;
  mm[3 - E3] -= 1;

  geometry::Coords<G> cpp(cpars, vg.GetCoordinates(block), pp[0], pp[1], pp[2]);
  geometry::Coords<G> cpm(cpars, vg.GetCoordinates(block), pm[0], pm[1], pm[2]);
  geometry::Coords<G> cqm(cpars, vg.GetCoordinates(block), qm[0], qm[1], qm[2]);
  geometry::Coords<G> cmm(cpars, vg.GetCoordinates(block), mm[0], mm[1], mm[2]);

  const auto xpp = cpp.GetCellCenter();
  const auto xpm = cpm.GetCellCenter();
  const auto xqm = cqm.GetCellCenter();
  const auto xmm = cmm.GetCellCenter();
  const auto &bpp = cpp.GetBounds();
  const auto &bpm = cpm.GetBounds();
  const auto &bqm = cqm.GetBounds();
  const auto &bmm = cmm.GetBounds();
  const std::array<Real, 3> xlo_pp{bpp.x1[0], bpp.x2[0], bpp.x3[0]};
  const std::array<Real, 3> xlo_pm{bpm.x1[0], bpm.x2[0], bpm.x3[0]};
  const std::array<Real, 3> xhi_pm{bpm.x1[1], bpm.x2[1], bpm.x3[1]};
  const std::array<Real, 3> xlo_qm{bqm.x1[0], bqm.x2[0], bqm.x3[0]};
  const std::array<Real, 3> xhi_qm{bqm.x1[1], bqm.x2[1], bqm.x3[1]};
  const std::array<Real, 3> xhi_mm{bmm.x1[1], bmm.x2[1], bmm.x3[1]};

  const auto ha_pp = cpp.template GetScaleFactorsFace<E2>(vg, block, pp[0], pp[1], pp[2]);
  const auto ha_qm = cqm.template GetScaleFactorsFace<E2>(vg, block, qm[0], qm[1], qm[2]);
  const auto hb_pp = cpp.template GetScaleFactorsFace<E3>(vg, block, pp[0], pp[1], pp[2]);
  const auto hb_pm = cpm.template GetScaleFactorsFace<E3>(vg, block, pm[0], pm[1], pm[2]);
  const auto hb_qm = cqm.template GetScaleFactorsFace<E3>(vg, block, qm[0], qm[1], qm[2]);
  const auto hb_mm = cmm.template GetScaleFactorsFace<E3>(vg, block, mm[0], mm[1], mm[2]);

  const Real ea_lo = e2_sign *
                     v.flux(block, E2, field::cell::B(e2_bcomp), qm[0], qm[1], qm[2]) /
                     ha_qm[e2_bcomp];
  const Real ea_hi = e2_sign *
                     v.flux(block, E2, field::cell::B(e2_bcomp), pp[0], pp[1], pp[2]) /
                     ha_pp[e2_bcomp];
  const Real eb_lo = e3_sign *
                     v.flux(block, E3, field::cell::B(e3_bcomp), pm[0], pm[1], pm[2]) /
                     hb_pm[e3_bcomp];
  const Real eb_hi = e3_sign *
                     v.flux(block, E3, field::cell::B(e3_bcomp), pp[0], pp[1], pp[2]) /
                     hb_pp[e3_bcomp];

  const Real emm = CellEMF<E1>(v, block, mm[0], mm[1], mm[2]);
  const Real epm = CellEMF<E1>(v, block, pm[0], pm[1], pm[2]);
  const Real eqm = CellEMF<E1>(v, block, qm[0], qm[1], qm[2]);
  const Real epp = CellEMF<E1>(v, block, pp[0], pp[1], pp[2]);

  constexpr int ia = E2 - 1;
  constexpr int ib = E3 - 1;
  const Real ga_lo_left = (eb_lo - emm) / (hb_mm[ib] * (xhi_mm[ib] - xmm[ib]));
  const Real ga_lo_right = (eb_hi - eqm) / (hb_qm[ib] * (xhi_qm[ib] - xqm[ib]));
  const Real ga_hi_left = (epm - eb_lo) / (hb_pm[ib] * (xpm[ib] - xlo_pm[ib]));
  const Real ga_hi_right = (epp - eb_hi) / (hb_pp[ib] * (xpp[ib] - xlo_pp[ib]));
  const Real ga_lo =
      UpwindEMFGradient(v.flux(block, E2, gas::cons::density(), qm[0], qm[1], qm[2]),
                        ga_lo_left, ga_lo_right);
  const Real ga_hi =
      UpwindEMFGradient(v.flux(block, E2, gas::cons::density(), pp[0], pp[1], pp[2]),
                        ga_hi_left, ga_hi_right);

  const Real gb_lo_left = (ea_lo - emm) / (ha_qm[ia] * (xhi_mm[ia] - xmm[ia]));
  const Real gb_lo_right = (ea_hi - epm) / (ha_pp[ia] * (xhi_pm[ia] - xpm[ia]));
  const Real gb_hi_left = (eqm - ea_lo) / (ha_qm[ia] * (xqm[ia] - xlo_qm[ia]));
  const Real gb_hi_right = (epp - ea_hi) / (ha_pp[ia] * (xpp[ia] - xlo_pp[ia]));
  const Real gb_lo =
      UpwindEMFGradient(v.flux(block, E3, gas::cons::density(), pm[0], pm[1], pm[2]),
                        gb_lo_left, gb_lo_right);
  const Real gb_hi =
      UpwindEMFGradient(v.flux(block, E3, gas::cons::density(), pp[0], pp[1], pp[2]),
                        gb_hi_left, gb_hi_right);

  const Real da_lo = ha_qm[ib] * (xhi_qm[ib] - xqm[ib]);
  const Real da_hi = ha_pp[ib] * (xpp[ib] - xlo_pp[ib]);
  const Real db_lo = hb_pm[ia] * (xhi_pm[ia] - xpm[ia]);
  const Real db_hi = hb_pp[ia] * (xpp[ia] - xlo_pp[ia]);
  return 0.25 * ((ea_lo + da_lo * ga_lo) + (ea_hi - da_hi * ga_hi) +
                 (eb_lo + db_lo * gb_lo) + (eb_hi - db_hi * gb_hi));
}

//----------------------------------------------------------------------------------------
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

  if (multi_d) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E3", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
          const Real h3e = coords.template GetEdgeScaleFactor<X3DIR>(vg, b, k, j, i);
          Real &emf = v.flux(b, TE::E3, field::face::B(), k, j, i);
          emf = h3e * UpwindEMF<X3DIR, G>(v, vg, cpars, b, k, j, i);
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
          const Real h2e = coords.template GetEdgeScaleFactor<X2DIR>(vg, b, k, j, i);
          Real &emf = v.flux(b, TE::E2, field::face::B(), k, j, i);
          emf = h2e * UpwindEMF<X2DIR, G>(v, vg, cpars, b, k, j, i);
        });
    if (multi_d) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "AssembleEdgeEMF::E1", parthenon::DevExecSpace(), 0,
          md->NumBlocks() - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
            geometry::Coords<G> coords(cpars, vg.GetCoordinates(b), k, j, i);
            const Real h1e = coords.template GetEdgeScaleFactor<X1DIR>(vg, b, k, j, i);
            Real &emf = v.flux(b, TE::E1, field::face::B(), k, j, i);
            emf = h1e * UpwindEMF<X1DIR, G>(v, vg, cpars, b, k, j, i);
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

} // namespace MHD

#endif // MHD_EMF_HPP_
