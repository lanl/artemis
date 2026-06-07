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
//! \file poisson_equation.hpp
//! \brief The code here is largely borrowed from the poisson_gmg example in Parthenon.

#ifndef SELF_GRAVITY_POISSON_EQUATION_HPP_
#define SELF_GRAVITY_POISSON_EQUATION_HPP_

// Artemis includes
#include "geometry/geometry.hpp"

// Parthenon includes
#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/solver_utils.hpp>
#include <solvers/tridiag_solver.hpp>

using namespace parthenon::package::prelude;

namespace SelfGravity {

constexpr parthenon::TopologicalElement te = parthenon::TopologicalElement::CC;

// This class implement methods for calculating A.x = y and returning the diagonal of A,
// where A is the the matrix representing the discretized Poisson equation on the grid.
template <Coordinates GEOM, class var_t>
class PoissonEquation {
 public:
  using IndependentVars = parthenon::TypeList<var_t>;

  PoissonEquation(parthenon::ParameterInput *pin, const std::string &label) {}

  // Add tasks to calculate the result of the matrix A (which is implicitly defined by
  // this class) being applied to x_t and store it in field out_t
  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    auto flux_res = tl.AddTask(depends_on, CalculateFluxes, md_mat, md_in);
    if (!(md_mat->grid.type() == parthenon::GridType::two_level_composite)) {
      auto start_flxcor =
          tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
      auto send_flxcor =
          tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
      auto recv_flxcor =
          tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md_in);
      flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
    }
    return tl.AddTask(flux_res, FluxMultiplyMatrix, md_in, md_out);
  }

  template <parthenon::CoordinateDirection dir, class coords_t>
  KOKKOS_INLINE_FUNCTION static auto
  GetEffectiveInverseDx2(const coords_t &coords, const coords_t &coords_p,
                         const coords_t &coords_m, const int k, const int j,
                         const int i) {
    Real xc = Null<Real>(), xp = Null<Real>(), xm = Null<Real>();
    auto AA = NewArray<Real, 2>();
    const Real Vol = coords.Volume();
    if constexpr (dir == X1DIR) {
      xc = coords.x1v();
      xp = coords_p.x1v();
      xm = coords_m.x1v();
      AA = coords.GetFaceAreaX1();
    } else if constexpr (dir == X2DIR) {
      xc = coords.x2v();
      xp = coords_p.x2v();
      xm = coords_m.x2v();
      AA = coords.GetFaceAreaX2();
    } else {
      xc = coords.x3v();
      xp = coords_p.x3v();
      xm = coords_m.x3v();
      AA = coords.GetFaceAreaX3();
    }
    const Real dxp = xp - xc;
    const Real dxm = xc - xm;
    return std::make_pair(AA[1] / (dxp * Vol), AA[0] / (dxm * Vol));
  }

  // Calculate an approximation to the diagonal of the matrix A and store it in diag_t.
  // For a uniform grid or when flux correction is ignored, this diagonal calculation
  // is exact. Exactness is (probably) not required since it is just used in Jacobi
  // iterations.
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    const int ndim = md_mat->GetMeshPointer()->ndim;
    IndexRange ib = md_mat->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_mat->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_mat->GetBoundsK(IndexDomain::interior, te);
    int nblocks = md_mat->NumBlocks();

    auto pm = md_mat->GetParentPointer();
    const auto &cpars =
        pm->packages.Get("artemis")->template Param<geometry::CoordParams>(
            "coord_params");

    auto desc_diag = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack = desc_diag.GetPack(md_diag.get());
    using TE = parthenon::TopologicalElement;
    parthenon::par_for(
        "StoreDiagonal", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(cpars, pack.GetCoordinates(b), k, j, i);
          // Build the unigrid diagonal of the matrix
          Real diag_elem = 0.0;
          {
            geometry::Coords<GEOM> coords_p(cpars, pack.GetCoordinates(b), k, j, i + 1);
            geometry::Coords<GEOM> coords_m(cpars, pack.GetCoordinates(b), k, j, i - 1);
            const auto &[idx2p, idx2m] =
                GetEffectiveInverseDx2<X1DIR>(coords, coords_p, coords_m, k, j, i);
            diag_elem -= (idx2m + idx2p);
          }
          if (ndim > 1) {
            geometry::Coords<GEOM> coords_p(cpars, pack.GetCoordinates(b), k, j + 1, i);
            geometry::Coords<GEOM> coords_m(cpars, pack.GetCoordinates(b), k, j - 1, i);
            const auto &[idx2p, idx2m] =
                GetEffectiveInverseDx2<X2DIR>(coords, coords_p, coords_m, k, j, i);
            diag_elem -= (idx2m + idx2p);
          }
          if (ndim > 2) {
            geometry::Coords<GEOM> coords_p(cpars, pack.GetCoordinates(b), k + 1, j, i);
            geometry::Coords<GEOM> coords_m(cpars, pack.GetCoordinates(b), k - 1, j, i);
            const auto &[idx2p, idx2m] =
                GetEffectiveInverseDx2<X3DIR>(coords, coords_p, coords_m, k, j, i);
            diag_elem -= (idx2m + idx2p);
          }
          pack(b, te, var_t(), k, j, i) = diag_elem;
        });
    return TaskStatus::complete;
  }

  static parthenon::TaskStatus
  CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                  std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
    int nblocks = md->NumBlocks();

    auto pm = md_mat->GetParentPointer();
    const auto &cpars =
        pm->packages.Get("artemis")->template Param<geometry::CoordParams>(
            "coord_params");

    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get());
    parthenon::par_for(
        "CaclulateFluxes", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(cpars, pack.GetCoordinates(b), k, j, i);
          pack.flux(b, X1DIR, var_t(), k, j, i) =
              (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i)) /
              coords.GetCellWidthX1();
          if (i == ib.e) {
            geometry::Coords<GEOM> coords_p(cpars, pack.GetCoordinates(b), k, j, i + 1);
            pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j, i + 1)) /
                coords_p.GetCellWidthX1();
          }

          if (ndim > 1) {
            pack.flux(b, X2DIR, var_t(), k, j, i) =
                (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) /
                coords.GetCellWidthX2();
            if (j == jb.e) {
              geometry::Coords<GEOM> coords_p(cpars, pack.GetCoordinates(b), k, j + 1, i);
              pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j + 1, i)) /
                  coords_p.GetCellWidthX2();
            }
          }

          if (ndim > 2) {
            pack.flux(b, X3DIR, var_t(), k, j, i) =
                (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) /
                coords.GetCellWidthX3();
            if (k == kb.e) {
              geometry::Coords<GEOM> coords_p(cpars, pack.GetCoordinates(b), k + 1, j, i);
              pack.flux(b, X3DIR, var_t(), k + 1, j, i) =
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k + 1, j, i)) /
                  coords_p.GetCellWidthX3();
            }
          }
        });
    return TaskStatus::complete;
  }

  // Calculate A in_t = out_t (in the region covered by md) for a given set of fluxes
  // calculated with in_t (which have possibly been corrected at coarse fine boundaries)
  static parthenon::TaskStatus
  FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
    int nblocks = md->NumBlocks();

    auto pm = md->GetParentPointer();
    const auto &cpars =
        pm->packages.Get("artemis")->template Param<geometry::CoordParams>(
            "coord_params");

    static auto desc =
        parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    static auto desc_out = parthenon::MakePackDescriptor<var_t>(md_out.get());
    auto pack = desc.GetPack(md.get());
    auto pack_out = desc_out.GetPack(md_out.get());
    parthenon::par_for(
        "FluxMultiplyMatrix", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(cpars, pack.GetCoordinates(b), k, j, i);

          Real div = 0.0;
          {
            const auto AA = coords.GetFaceAreaX1();
            div += (pack.flux(b, X1DIR, var_t(), k, j, i) * AA[0] -
                    pack.flux(b, X1DIR, var_t(), k, j, i + 1) * AA[1]);
          }
          if (ndim > 1) {
            const auto AA = coords.GetFaceAreaX2();
            div += (pack.flux(b, X2DIR, var_t(), k, j, i) * AA[0] -
                    pack.flux(b, X2DIR, var_t(), k, j + 1, i) * AA[1]);
          }
          if (ndim > 2) {
            const auto AA = coords.GetFaceAreaX3();
            div += (pack.flux(b, X3DIR, var_t(), k, j, i) * AA[0] -
                    pack.flux(b, X3DIR, var_t(), k + 1, j, i) * AA[1]);
          }
          pack_out(b, te, var_t(), k, j, i) = div / coords.Volume();
        });
    return TaskStatus::complete;
  }
};

} // namespace SelfGravity

#endif // SELF_GRAVITY_POISSON_EQUATION_HPP_
