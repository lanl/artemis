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
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef UTILS_REFINEMENT_PROLONGATION_HPP_
#define UTILS_REFINEMENT_PROLONGATION_HPP_

// C++ includes
#include <algorithm>
#include <cstring>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <interface/variable_state.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/domain.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"

namespace ArtemisUtils {
template <Coordinates C, TopologicalElement EL>
KOKKOS_FORCEINLINE_FUNCTION Real
GetElementAverageWeight(const geometry::Coords<C> &coords) {
  if constexpr (EL == TE::CC) {
    return coords.Volume();
  } else if constexpr (EL == TE::F1) {
    return coords.template GetFaceArea<X1DIR>();
  } else if constexpr (EL == TE::F2) {
    return coords.template GetFaceArea<X2DIR>();
  } else if constexpr (EL == TE::F3) {
    return coords.template GetFaceArea<X3DIR>();
  } else if constexpr (EL == TE::E1) {
    return coords.GetEdgeLengthX1();
  } else if constexpr (EL == TE::E2) {
    return coords.GetEdgeLengthX2();
  } else if constexpr (EL == TE::E3) {
    return coords.GetEdgeLengthX3();
  }
  PARTHENON_FAIL("Unsupported topological element for geometric averaging weight!");
  return 0.0;
}

template <Coordinates C>
KOKKOS_FORCEINLINE_FUNCTION Real GetFaceAverageWeight(const geometry::Coords<C> &coords,
                                                      const int face_idx) {
  if (face_idx == 0) {
    return coords.template GetFaceArea<X1DIR>();
  } else if (face_idx == 1) {
    return coords.template GetFaceArea<X2DIR>();
  }
  return coords.template GetFaceArea<X3DIR>();
}

template <Coordinates C, int DIM, TopologicalElement EL>
KOKKOS_FORCEINLINE_FUNCTION Real GetCoordinate(const geometry::Coords<C> &coords) {
  static_assert(DIM >= 1 && DIM <= 3, "Invalid dimension!");
  if constexpr (EL == TE::CC) {
    if constexpr (DIM == 1) {
      return coords.x1v();
    } else if constexpr (DIM == 2) {
      return coords.x2v();
    }
    return coords.x3v();
  } else if constexpr (EL == TE::F1) {
    const auto xf = coords.FaceCenX1(geometry::CellFace::lower);
    return xf[DIM - 1];
  } else if constexpr (EL == TE::F2) {
    const auto xf = coords.FaceCenX2(geometry::CellFace::lower);
    return xf[DIM - 1];
  } else if constexpr (EL == TE::F3) {
    const auto xf = coords.FaceCenX3(geometry::CellFace::lower);
    return xf[DIM - 1];
  }
  PARTHENON_FAIL(
      "Artemis prolongation only supports cell-centered and face-centered fields!");
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn  void ArtemisUtils::GetGridSpacings
//! \brief compute distances from cell center to the nearest center in the + or -
//!        coordinate direction. Do so for both coarse and fine grids.
template <Coordinates C, int DIM, TopologicalElement EL>
KOKKOS_FORCEINLINE_FUNCTION void
GetGridSpacings(const Coordinates_t &coords, const Coordinates_t &coarse_coords,
                const bool log, int k, int j, int i, int fk, int fj, int fi, Real *dxm,
                Real *dxp, Real *dxfm, Real *dxfp) {
  namespace gg = geometry;
  Real xm = Null<Real>(), xc = Null<Real>(), xp = Null<Real>();
  Real fxm = Null<Real>(), fxp = Null<Real>();
  gg::Coords<C> cc(log, coarse_coords, k, j, i);
  gg::Coords<C> cm(log, coarse_coords, k - (DIM == 3), j - (DIM == 2), i - (DIM == 1));
  gg::Coords<C> cp(log, coarse_coords, k + (DIM == 3), j + (DIM == 2), i + (DIM == 1));
  gg::Coords<C> fm(log, coords, fk, fj, fi);
  gg::Coords<C> fp(log, coords, fk + (DIM == 3), fj + (DIM == 2), fi + (DIM == 1));
  xm = ArtemisUtils::GetCoordinate<C, DIM, EL>(cm);
  xc = ArtemisUtils::GetCoordinate<C, DIM, EL>(cc);
  xp = ArtemisUtils::GetCoordinate<C, DIM, EL>(cp);
  fxm = ArtemisUtils::GetCoordinate<C, DIM, EL>(fm);
  fxp = ArtemisUtils::GetCoordinate<C, DIM, EL>(fp);
  *dxm = xc - xm;
  *dxp = xp - xc;
  *dxfm = xc - fxm;
  *dxfp = fxp - xc;
}

//----------------------------------------------------------------------------------------
//! \fn  Real ArtemisUtils::GradMinMod
//! \brief
KOKKOS_FORCEINLINE_FUNCTION
Real GradMinMod(const Real fc, const Real fm, const Real fp, const Real dxm,
                const Real dxp, Real &gxm, Real &gxp) {
  gxm = (fc - fm) / dxm;
  gxp = (fp - fc) / dxp;
  return 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
}

//----------------------------------------------------------------------------------------
//! \struct  ArtemisUtils::ProlongateShared
//! \brief
template <Coordinates GEOM, bool log, bool use_minmod_slope>
struct ProlongateShared {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    PARTHENON_REQUIRE(
        el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::F3,
        "Artemis AMR only supports cell-centered and face-centered fields!");

    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    constexpr int element_idx = static_cast<int>(el) % 3;

    const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
    const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
    const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2);

    const Real fc = coarse(element_idx, l, m, n, k, j, i);
    geometry::Coords<GEOM> cc(log, coarse_coords, k, j, i);
    const Real wc = 1.0;
    const Real qc = wc * fc;

    Real dx1fm = 0;
    [[maybe_unused]] Real dx1fp = 0;
    [[maybe_unused]] Real gx1m = 0, gx1p = 0;
    if constexpr (INCLUDE_X1) {
      Real dx1m, dx1p;
      ArtemisUtils::GetGridSpacings<GEOM, 1, el>(coords, coarse_coords, log, k, j, i, fk,
                                                 fj, fi, &dx1m, &dx1p, &dx1fm, &dx1fp);

      geometry::Coords<GEOM> cm(log, coarse_coords, k, j, i - 1);
      geometry::Coords<GEOM> cp(log, coarse_coords, k, j, i + 1);
      const Real fm = coarse(element_idx, l, m, n, k, j, i - 1);
      const Real fp = coarse(element_idx, l, m, n, k, j, i + 1);
      const Real qm = fm;
      const Real qp = fp;
      Real gx1c = ArtemisUtils::GradMinMod(qc, qm, qp, dx1m, dx1p, gx1m, gx1p);
      if constexpr (use_minmod_slope) {
        gx1m = gx1c;
        gx1p = gx1c;
      }
    }

    Real dx2fm = 0;
    [[maybe_unused]] Real dx2fp = 0;
    [[maybe_unused]] Real gx2m = 0, gx2p = 0;
    if constexpr (INCLUDE_X2) {
      Real dx2m, dx2p;
      ArtemisUtils::GetGridSpacings<GEOM, 2, el>(coords, coarse_coords, log, k, j, i, fk,
                                                 fj, fi, &dx2m, &dx2p, &dx2fm, &dx2fp);
      geometry::Coords<GEOM> cm(log, coarse_coords, k, j - 1, i);
      geometry::Coords<GEOM> cp(log, coarse_coords, k, j + 1, i);
      const Real fm = coarse(element_idx, l, m, n, k, j - 1, i);
      const Real fp = coarse(element_idx, l, m, n, k, j + 1, i);
      const Real qm = fm;
      const Real qp = fp;
      Real gx2c = ArtemisUtils::GradMinMod(qc, qm, qp, dx2m, dx2p, gx2m, gx2p);
      if constexpr (use_minmod_slope) {
        gx2m = gx2c;
        gx2p = gx2c;
      }
    }

    Real dx3fm = 0;
    [[maybe_unused]] Real dx3fp = 0;
    [[maybe_unused]] Real gx3m = 0, gx3p = 0;
    if constexpr (INCLUDE_X3) {
      Real dx3m, dx3p;
      ArtemisUtils::GetGridSpacings<GEOM, 3, el>(coords, coarse_coords, log, k, j, i, fk,
                                                 fj, fi, &dx3m, &dx3p, &dx3fm, &dx3fp);
      geometry::Coords<GEOM> cm(log, coarse_coords, k - 1, j, i);
      geometry::Coords<GEOM> cp(log, coarse_coords, k + 1, j, i);
      const Real fm = coarse(element_idx, l, m, n, k - 1, j, i);
      const Real fp = coarse(element_idx, l, m, n, k + 1, j, i);
      const Real qm = fm;
      const Real qp = fp;
      Real gx3c = ArtemisUtils::GradMinMod(qc, qm, qp, dx3m, dx3p, gx3m, gx3p);
      if constexpr (use_minmod_slope) {
        gx3m = gx3c;
        gx3p = gx3c;
      }
    }

    Real qfine[2][2][2] = {{{0.0}}};
    bool active[2][2][2] = {{{false}}};
    auto stage_fine_value = [&](const int ok, const int oj, const int oi, const Real qf) {
      qfine[ok][oj][oi] = qf;
      active[ok][oj][oi] = true;
    };

    stage_fine_value(0, 0, 0, qc - (gx1m * dx1fm + gx2m * dx2fm + gx3m * dx3fm));
    if constexpr (INCLUDE_X1)
      stage_fine_value(0, 0, 1, qc + (gx1p * dx1fp - gx2m * dx2fm - gx3m * dx3fm));
    if constexpr (INCLUDE_X2)
      stage_fine_value(0, 1, 0, qc - (gx1m * dx1fm - gx2p * dx2fp + gx3m * dx3fm));
    if constexpr (INCLUDE_X2 && INCLUDE_X1)
      stage_fine_value(0, 1, 1, qc + (gx1p * dx1fp + gx2p * dx2fp - gx3m * dx3fm));
    if constexpr (INCLUDE_X3)
      stage_fine_value(1, 0, 0, qc - (gx1m * dx1fm + gx2m * dx2fm - gx3p * dx3fp));
    if constexpr (INCLUDE_X3 && INCLUDE_X1)
      stage_fine_value(1, 0, 1, qc + (gx1p * dx1fp - gx2m * dx2fm + gx3p * dx3fp));
    if constexpr (INCLUDE_X3 && INCLUDE_X2)
      stage_fine_value(1, 1, 0, qc - (gx1m * dx1fm - gx2p * dx2fp - gx3p * dx3fp));
    if constexpr (INCLUDE_X3 && INCLUDE_X2 && INCLUDE_X1)
      stage_fine_value(1, 1, 1, qc + (gx1p * dx1fp + gx2p * dx2fp + gx3p * dx3fp));

    if constexpr (el == TE::F1 || el == TE::F2 || el == TE::F3) {
      const Real coarse_flux = GetElementAverageWeight<GEOM, el>(cc) * fc;
      Real fine_flux = 0.0;
      Real fine_area = 0.0;
      for (int ok = 0; ok < 2; ++ok) {
        for (int oj = 0; oj < 2; ++oj) {
          for (int oi = 0; oi < 2; ++oi) {
            if (!active[ok][oj][oi]) continue;
            geometry::Coords<GEOM> cf(log, coords, fk + ok, fj + oj, fi + oi);
            const Real area = GetElementAverageWeight<GEOM, el>(cf);
            fine_flux += area * qfine[ok][oj][oi];
            fine_area += area;
          }
        }
      }
      const Real delta = (coarse_flux - fine_flux) / (fine_area + Fuzz<Real>());
      for (int ok = 0; ok < 2; ++ok) {
        for (int oj = 0; oj < 2; ++oj) {
          for (int oi = 0; oi < 2; ++oi) {
            if (active[ok][oj][oi]) qfine[ok][oj][oi] += delta;
          }
        }
      }
    }

    for (int ok = 0; ok < 2; ++ok) {
      for (int oj = 0; oj < 2; ++oj) {
        for (int oi = 0; oi < 2; ++oi) {
          if (active[ok][oj][oi]) {
            fine(element_idx, l, m, n, fk + ok, fj + oj, fi + oi) = qfine[ok][oj][oi];
          }
        }
      }
    }
  }
};

template <Coordinates GEOM, bool log>
struct ProlongateTothAndRoe {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return (cel == TE::CC) && (GetTopologicalType(fel) == TopologicalType::Face);
  }
  // Here, fel is the topological element on which the field is defined and
  // cel is the topological element on which we are filling the internal values
  // of the field. So, for instance, we could fill the fine cell values of an
  // x-face field within the volume of a coarse cell. This is assumes that the
  // values of the fine cells on the elements corresponding with the coarse cell
  // have been filled.
  template <int DIM, TopologicalElement fel = TopologicalElement::CC,
            TopologicalElement cel = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *,
     const ParArrayND<Real, VariableState> *pfine) {
    if constexpr (!IsSubmanifold(fel, cel)) {
      return;
    } else {
      if constexpr (!(fel == TE::F1)) {
        return;
      } else {
        const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
        const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
        const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;
        auto &fine = *pfine;
        constexpr int g3 = (DIM > 2) ? 1 : 0;
        constexpr int g2 = (DIM > 1) ? 1 : 0;
        constexpr int ny = (DIM > 1) ? 2 : 1;
        constexpr int nz = (DIM > 2) ? 2 : 1;
        constexpr int ncell = (DIM == 1 ? 2 : (DIM == 2 ? 4 : 8));
        constexpr int neq = ncell - 1;
        constexpr int nf1 = ny * nz;
        constexpr int nf2 = (DIM > 1) ? (2 * nz) : 0;
        constexpr int nf3 = (DIM > 2) ? 4 : 0;
        constexpr int nunk = nf1 + nf2 + nf3;
        constexpr int max_eq = 7;
        constexpr int max_unk = 12;

        auto get_indices = [&](const int comp, const int sx, const int sy, const int sz) {
          return std::array<int, 3>{fk + sz * g3, fj + sy * g2, fi + sx};
        };

        auto get_face_value = [&](const int comp, const int sx, const int sy,
                                  const int sz) -> Real & {
          const auto idx = get_indices(comp, sx, sy, sz);
          return fine(comp, l, m, n, idx[0], idx[1], idx[2]);
        };

        auto get_face_area = [&](const int comp, const int sx, const int sy,
                                 const int sz) {
          const auto idx = get_indices(comp, sx, sy, sz);
          geometry::Coords<GEOM> cf(log, coords, idx[0], idx[1], idx[2]);
          return ArtemisUtils::GetFaceAverageWeight(cf, comp);
        };

        auto get_face_flux = [&](const int comp, const int sx, const int sy,
                                 const int sz) {
          return get_face_area(comp, sx, sy, sz) * get_face_value(comp, sx, sy, sz);
        };

        auto set_face_flux = [&](const int comp, const int sx, const int sy, const int sz,
                                 const Real qf) {
          const Real wf = get_face_area(comp, sx, sy, sz);
          get_face_value(comp, sx, sy, sz) = qf / (wf + Fuzz<Real>());
        };

        auto get_face_normal_pos = [&](const int comp, const int sx, const int sy,
                                       const int sz) {
          const auto idx = get_indices(comp, sx, sy, sz);
          geometry::Coords<GEOM> cf(log, coords, idx[0], idx[1], idx[2]);
          if (comp == 0) {
            return ArtemisUtils::GetCoordinate<GEOM, 1, TE::F1>(cf);
          } else if (comp == 1) {
            return ArtemisUtils::GetCoordinate<GEOM, 2, TE::F2>(cf);
          }
          return ArtemisUtils::GetCoordinate<GEOM, 3, TE::F3>(cf);
        };

        auto idx_f1 = [&](const int sy, const int sz) { return sz * ny + sy; };
        auto idx_f2 = [&](const int sx, const int sz) { return nf1 + sz * 2 + sx; };
        auto idx_f3 = [&](const int sx, const int sy) { return nf1 + nf2 + sy * 2 + sx; };

        if constexpr (DIM == 1) {
          const Real q0 = get_face_flux(0, 0, 0, 0);
          const Real q2 = get_face_flux(0, 2, 0, 0);
          const Real x0 = get_face_normal_pos(0, 0, 0, 0);
          const Real x1 = get_face_normal_pos(0, 1, 0, 0);
          const Real x2 = get_face_normal_pos(0, 2, 0, 0);
          const Real alpha = (x1 - x0) / (x2 - x0 + Fuzz<Real>());
          set_face_flux(0, 1, 0, 0, q0 + alpha * (q2 - q0));
          return;
        }

        Real u0[max_unk] = {0.0};
        Real w[max_unk] = {0.0};
        Real D[max_eq][max_unk] = {{0.0}};
        Real rhs[max_eq] = {0.0};

        for (int sz = 0; sz < nz; ++sz) {
          for (int sy = 0; sy < ny; ++sy) {
            const int idx = idx_f1(sy, sz);
            const Real q0 = get_face_flux(0, 0, sy, sz);
            const Real q2 = get_face_flux(0, 2, sy, sz);
            const Real x0 = get_face_normal_pos(0, 0, sy, sz);
            const Real x1 = get_face_normal_pos(0, 1, sy, sz);
            const Real x2 = get_face_normal_pos(0, 2, sy, sz);
            const Real alpha = (x1 - x0) / (x2 - x0 + Fuzz<Real>());
            u0[idx] = q0 + alpha * (q2 - q0);
            w[idx] = get_face_area(0, 1, sy, sz);
          }
        }
        if constexpr (DIM > 1) {
          for (int sz = 0; sz < nz; ++sz) {
            for (int sx = 0; sx < 2; ++sx) {
              const int idx = idx_f2(sx, sz);
              const Real q0 = get_face_flux(1, sx, 0, sz);
              const Real q2 = get_face_flux(1, sx, 2, sz);
              const Real x0 = get_face_normal_pos(1, sx, 0, sz);
              const Real x1 = get_face_normal_pos(1, sx, 1, sz);
              const Real x2 = get_face_normal_pos(1, sx, 2, sz);
              const Real alpha = (x1 - x0) / (x2 - x0 + Fuzz<Real>());
              u0[idx] = q0 + alpha * (q2 - q0);
              w[idx] = get_face_area(1, sx, 1, sz);
            }
          }
        }
        if constexpr (DIM > 2) {
          for (int sy = 0; sy < 2; ++sy) {
            for (int sx = 0; sx < 2; ++sx) {
              const int idx = idx_f3(sx, sy);
              const Real q0 = get_face_flux(2, sx, sy, 0);
              const Real q2 = get_face_flux(2, sx, sy, 2);
              const Real x0 = get_face_normal_pos(2, sx, sy, 0);
              const Real x1 = get_face_normal_pos(2, sx, sy, 1);
              const Real x2 = get_face_normal_pos(2, sx, sy, 2);
              const Real alpha = (x1 - x0) / (x2 - x0 + Fuzz<Real>());
              u0[idx] = q0 + alpha * (q2 - q0);
              w[idx] = get_face_area(2, sx, sy, 1);
            }
          }
        }

        int row = 0;
        for (int sz = 0; sz < nz; ++sz) {
          for (int sy = 0; sy < ny; ++sy) {
            for (int sx = 0; sx < 2; ++sx) {
              if (sx == 1 && sy == ny - 1 && sz == nz - 1) continue;

              if (sx == 0) {
                D[row][idx_f1(sy, sz)] += 1.0;
                rhs[row] += get_face_flux(0, 0, sy, sz);
              } else {
                D[row][idx_f1(sy, sz)] -= 1.0;
                rhs[row] -= get_face_flux(0, 2, sy, sz);
              }

              if constexpr (DIM > 1) {
                if (sy == 0) {
                  D[row][idx_f2(sx, sz)] += 1.0;
                  rhs[row] += get_face_flux(1, sx, 0, sz);
                } else {
                  D[row][idx_f2(sx, sz)] -= 1.0;
                  rhs[row] -= get_face_flux(1, sx, 2, sz);
                }
              }

              if constexpr (DIM > 2) {
                if (sz == 0) {
                  D[row][idx_f3(sx, sy)] += 1.0;
                  rhs[row] += get_face_flux(2, sx, sy, 0);
                } else {
                  D[row][idx_f3(sx, sy)] -= 1.0;
                  rhs[row] -= get_face_flux(2, sx, sy, 2);
                }
              }

              ++row;
            }
          }
        }

        Real M[max_eq][max_eq] = {{0.0}};
        Real residual[max_eq] = {0.0};
        for (int r = 0; r < neq; ++r) {
          Real du = 0.0;
          for (int u = 0; u < nunk; ++u) {
            du += D[r][u] * u0[u];
          }
          residual[r] = du - rhs[r];
          for (int c = 0; c < neq; ++c) {
            Real sum = 0.0;
            for (int u = 0; u < nunk; ++u) {
              sum += D[r][u] * w[u] * D[c][u];
            }
            M[r][c] = sum;
          }
        }

        Real A[max_eq][max_eq + 1] = {{0.0}};
        for (int r = 0; r < neq; ++r) {
          for (int c = 0; c < neq; ++c) {
            A[r][c] = M[r][c];
          }
          A[r][neq] = residual[r];
        }

        for (int p = 0; p < neq; ++p) {
          int piv = p;
          Real max_abs = std::abs(A[p][p]);
          for (int r = p + 1; r < neq; ++r) {
            const Real cand = std::abs(A[r][p]);
            if (cand > max_abs) {
              max_abs = cand;
              piv = r;
            }
          }
          if (piv != p) {
            for (int c = p; c <= neq; ++c) {
              const Real tmp = A[p][c];
              A[piv][c] = A[p][c];
              A[p][c] = tmp;
            }
          }
          const Real pivot = A[p][p];
          if (std::abs(pivot) <= Fuzz<Real>()) continue;
          for (int r = p + 1; r < neq; ++r) {
            const Real fac = A[r][p] / pivot;
            for (int c = p; c <= neq; ++c) {
              A[r][c] -= fac * A[p][c];
            }
          }
        }

        Real lambda[max_eq] = {0.0};
        for (int r = neq - 1; r >= 0; --r) {
          Real sum = A[r][neq];
          for (int c = r + 1; c < neq; ++c) {
            sum -= A[r][c] * lambda[c];
          }
          const Real pivot = A[r][r];
          lambda[r] = (std::abs(pivot) > Fuzz<Real>()) ? (sum / pivot) : 0.0;
        }

        Real u[max_unk] = {0.0};
        for (int idx = 0; idx < nunk; ++idx) {
          Real corr = 0.0;
          for (int r = 0; r < neq; ++r) {
            corr += D[r][idx] * lambda[r];
          }
          u[idx] = u0[idx] - w[idx] * corr;
        }

        for (int sz = 0; sz < nz; ++sz) {
          for (int sy = 0; sy < ny; ++sy) {
            set_face_flux(0, 1, sy, sz, u[idx_f1(sy, sz)]);
          }
        }
        if constexpr (DIM > 1) {
          for (int sz = 0; sz < nz; ++sz) {
            for (int sx = 0; sx < 2; ++sx) {
              set_face_flux(1, sx, 1, sz, u[idx_f2(sx, sz)]);
            }
          }
        }
        if constexpr (DIM > 2) {
          for (int sy = 0; sy < 2; ++sy) {
            for (int sx = 0; sx < 2; ++sx) {
              set_face_flux(2, sx, sy, 1, u[idx_f3(sx, sy)]);
            }
          }
        }
      }
    }
  }
};

} // namespace ArtemisUtils

#endif // UTILS_REFINEMENT_PROLONGATION_HPP_
