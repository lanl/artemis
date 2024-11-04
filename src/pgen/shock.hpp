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
#ifndef PGEN_SHOCK_HPP_
#define PGEN_SHOCK_HPP_
//! \file shock.hpp
//! \brief

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

// jaybenne includes
#include "jaybenne.hpp"

using ArtemisUtils::EOS;

namespace shock {

struct ShockParams {
  Real rhol, vxl, tl;
  Real rhor, vxr, tr;
  Real cv;
  Real xdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void InitShockParams
//! \brief Extracts shock parameters from ParameterInput.
inline void InitShockParams(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("shock_params"))) {
    ShockParams shock_params;
    shock_params.rhol = pin->GetOrAddReal("problem", "rhol", 1.0);
    shock_params.vxl = pin->GetOrAddReal("problem", "vxl", 2.0);
    shock_params.tl = pin->GetOrAddReal("problem", "tl", 0.6);
    shock_params.rhor = pin->GetOrAddReal("problem", "rhor", 2.285714);
    shock_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.875000);
    shock_params.tr = pin->GetOrAddReal("problem", "tr", 1.246875);
    shock_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.0005);
    shock_params.cv = pin->GetOrAddReal("gas", "cv", 1.5);
    params.Add("shock_params", shock_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Shock()
//! \brief Sets initial conditions for shock problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_radiation = artemis_pkg->Param<bool>("do_radiation");
  PARTHENON_REQUIRE(do_gas, "The shock problem requires gas hydrodynamics!");
  PARTHENON_REQUIRE(!(do_dust), "The shock problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // Shock parameters
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");

  // Setup shock state
  pmb->par_for(
      "shock", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const bool upwind = (xi[0] <= shkp.xdisc);
        const Real rho = upwind ? shkp.rhol : shkp.rhor;
        const Real vx = upwind ? shkp.vxl : shkp.vxr;
        const Real sie = upwind ? shkp.cv * shkp.tl : shkp.cv * shkp.tr;
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = sie;
      });

  if (do_radiation) jaybenne::InitializeRadiation(md.get(), true);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ShockInnerX1()
template <Coordinates GEOM>
inline void ShockInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");
  const auto nb = IndexRange{0, 0};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() > 0) {
    pmb->par_for_bndry(
        "ShockInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          v(0, gas::prim::density(0), k, j, i) = shkp.rhol;
          v(0, gas::prim::velocity(0), k, j, i) = shkp.vxl;
          v(0, gas::prim::velocity(1), k, j, i) = 0.0;
          v(0, gas::prim::velocity(2), k, j, i) = 0.0;
          v(0, gas::prim::sie(0), k, j, i) = shkp.cv * shkp.tl;
        });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ShockOuterX1()
template <Coordinates GEOM>
inline void ShockOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");
  const auto nb = IndexRange{0, 0};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() > 0) {
    pmb->par_for_bndry(
        "ShockOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
        coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          v(0, gas::prim::density(0), k, j, i) = shkp.rhor;
          v(0, gas::prim::velocity(0), k, j, i) = shkp.vxr;
          v(0, gas::prim::velocity(1), k, j, i) = 0.0;
          v(0, gas::prim::velocity(2), k, j, i) = 0.0;
          v(0, gas::prim::sie(0), k, j, i) = shkp.cv * shkp.tr;
        });
  }

  return;
}

} // namespace shock
#endif // PGEN_SHOCK_HPP_
