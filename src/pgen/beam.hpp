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
#ifndef PGEN_BEAM_HPP_
#define PGEN_BEAM_HPP_

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

namespace beam {

struct BeamParams {
  Real erad;
  Real width;
  Real mu;
};

inline void InitBeamParams(MeshBlock *pmb, ParameterInput *pin) {
  Params &params = pmb->packages.Get("artemis")->AllParams();
  if (!(params.hasKey("beam_params"))) {
    BeamParams beam_params;

    beam_params.erad = pin->GetOrAddReal("problem", "erad", 1.0);
    beam_params.width = pin->GetOrAddReal("problem", "width", 0.05);
    beam_params.mu = pin->GetOrAddReal("problem", "mu", 0.5);
    params.Add("beam_params", beam_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BEAM_()
//! \brief Sets initial conditions for BEAM tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  const Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;

  // Determine if gas and/or dust hydrodynamics are enabled
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_rad = artemis_pkg->Param<bool>("do_moment");

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         rad::prim::energy, rad::prim::flux>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "pgen_beam", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates

        // compute cell-centered conserved variables
        if (do_gas) {
          v(0, gas::prim::density(0), k, j, i) = 1.0;
          v(0, gas::prim::velocity(0), k, j, i) = 0.0;
          v(0, gas::prim::velocity(1), k, j, i) = 0.0;
          v(0, gas::prim::velocity(2), k, j, i) = 0.0;
          v(0, gas::prim::sie(0), k, j, i) = 1.0;
        }
        if (do_rad) {
          v(0, rad::prim::energy(0), k, j, i) = 1.0;
          v(0, rad::prim::flux(0), k, j, i) = 0.0;
          v(0, rad::prim::flux(1), k, j, i) = 0.0;
          v(0, rad::prim::flux(2), k, j, i) = 0.0;
        }
      });
}

template <Coordinates GEOM>
inline void BeamInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_rad = artemis_pkg->Param<bool>("do_moment");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, rad::prim::energy,
                                                 rad::prim::flux>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;

  auto &pars = artemis_pkg->Param<BeamParams>("beam_params");
  pmb->par_for_bndry(
      "BeamInnerX2", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC, coarse,
      fine, KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const Real xf = coords.bnds.x1[0];

        if (do_gas) {
          for (int d = 0; d < 3; d++)
            v(0, gas::prim::velocity(d), k, j, i) =
                v(0, gas::prim::velocity(d), k, js, i);
          v(0, gas::prim::density(0), k, j, i) = v(0, gas::prim::density(0), k, js, i);
          v(0, gas::prim::sie(0), k, j, i) = v(0, gas::prim::sie(0), k, js, i);
        }
        if (do_rad) {
          const bool bc = (xf <= pars.width);
          const Real erad = pars.erad;
          const Real f = std::sqrt(pars.mu);
          v(0, rad::prim::flux(0), k, j, i) =
              (bc) ? f : v(0, rad::prim::flux(0), k, js, i);
          v(0, rad::prim::flux(1), k, j, i) =
              (bc) ? f : -v(0, rad::prim::flux(1), k, js, i);
          v(0, rad::prim::flux(2), k, j, i) =
              (bc) ? 0.0 : v(0, rad::prim::flux(2), k, js, i);
          v(0, rad::prim::energy(0), k, j, i) =
              (bc) ? pars.erad : v(0, rad::prim::energy(0), k, js, i);
        }
      });
}

template <Coordinates GEOM>
inline void BeamInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_rad = artemis_pkg->Param<bool>("do_moment");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, rad::prim::energy,
                                                 rad::prim::flux>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int is = range.s;

  auto &pars = artemis_pkg->Param<BeamParams>("beam_params");
  pmb->par_for_bndry(
      "BeamInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC, coarse,
      fine, KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const Real yf = coords.bnds.x2[0];

        if (do_gas) {
          for (int d = 0; d < 3; d++)
            v(0, gas::prim::velocity(d), k, j, i) =
                v(0, gas::prim::velocity(d), k, j, is);
          v(0, gas::prim::density(0), k, j, i) = v(0, gas::prim::density(0), k, j, is);
          v(0, gas::prim::sie(0), k, j, i) = v(0, gas::prim::sie(0), k, j, is);
        }
        if (do_rad) {
          const bool bc = (yf <= pars.width);
          const Real f = std::sqrt(pars.mu);
          v(0, rad::prim::flux(0), k, j, i) =
              (bc) ? f : -v(0, rad::prim::flux(0), k, j, is);
          v(0, rad::prim::flux(1), k, j, i) =
              (bc) ? f : v(0, rad::prim::flux(1), k, j, is);
          v(0, rad::prim::flux(2), k, j, i) =
              (bc) ? 0.0 : v(0, rad::prim::flux(2), k, j, is);
          v(0, rad::prim::energy(0), k, j, i) =
              (bc) ? pars.erad : v(0, rad::prim::energy(0), k, j, is);
        }
      });
}

} // namespace beam
#endif // PGEN_BEAM_HPP_
