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
//!
//!  mu = mH, gamma = 5/3, rho*kappa = 577 /cm
//!  c/chat = 43.3526011561
//!  left state:         |  right state:
//!      T = 2.18e6 K    |   T = 7.98e6 K
//!    rho = 5.69 g/cc   | rho = 17.1 g/cc
//!     vx = 5.19e7 cm/s |  vx = 1.73e7 cm/s

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
    shock_params.rhol = pin->GetOrAddReal("problem", "rhol", 5.69);
    shock_params.vxl = pin->GetOrAddReal("problem", "vxl", 5.19e7);
    shock_params.tl = pin->GetOrAddReal("problem", "tl", 2.18e6);
    shock_params.rhor = pin->GetOrAddReal("problem", "rhor", 17.1);
    shock_params.vxr = pin->GetOrAddReal("problem", "vxr", 1.73e7);
    shock_params.tr = pin->GetOrAddReal("problem", "tr", 7.98e6);
    shock_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.0005);
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
  const bool do_imc = artemis_pkg->Param<bool>("do_imc");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  PARTHENON_REQUIRE(do_gas, "The shock problem requires gas hydrodynamics!");
  PARTHENON_REQUIRE(!(do_dust), "The shock problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");

  Real ar = Null<Real>();
  if (do_moment) {
    ar = pmb->packages.Get("radiation")->Param<Real>("arad");
  }

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
        const Real T = upwind ? shkp.tl : shkp.tr;
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) =
            eos_d.InternalEnergyFromDensityTemperature(rho, T);
        if (do_moment) {
          v(0, rad::prim::energy(0), k, j, i) = ar * SQR(SQR(T));
          v(0, rad::prim::flux(0), k, j, i) = 0.0;
          v(0, rad::prim::flux(1), k, j, i) = 0.0;
          v(0, rad::prim::flux(2), k, j, i) = 0.0;
        }
      });

  if (do_imc) jaybenne::InitializeRadiation(md.get(), true);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ShockInnerX1()
template <Coordinates GEOM>
inline void ShockInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  Real ar = Null<Real>();
  if (do_moment) {
    ar = pmb->packages.Get("radiation")->Param<Real>("arad");
  }
  const auto nb = IndexRange{0, 0};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, rad::prim::energy,
                                                 rad::prim::flux>(mbd);
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
          v(0, gas::prim::sie(0), k, j, i) =
              eos_d.InternalEnergyFromDensityTemperature(shkp.rhol, shkp.tl);
          if (do_moment) {
            v(0, rad::prim::energy(0), k, j, i) = ar * SQR(SQR(shkp.tl));
            v(0, rad::prim::flux(0), k, j, i) = 0.0;
            v(0, rad::prim::flux(1), k, j, i) = 0.0;
            v(0, rad::prim::flux(2), k, j, i) = 0.0;
          }
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
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  Real ar = Null<Real>();
  if (do_moment) {
    ar = pmb->packages.Get("radiation")->Param<Real>("arad");
  }
  const auto nb = IndexRange{0, 0};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, rad::prim::energy,
                                                 rad::prim::flux>(mbd);
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
          v(0, gas::prim::sie(0), k, j, i) =
              eos_d.InternalEnergyFromDensityTemperature(shkp.rhor, shkp.tr);
          if (do_moment) {
            v(0, rad::prim::energy(0), k, j, i) = ar * SQR(SQR(shkp.tr));
            v(0, rad::prim::flux(0), k, j, i) = 0.0;
            v(0, rad::prim::flux(1), k, j, i) = 0.0;
            v(0, rad::prim::flux(2), k, j, i) = 0.0;
          }
        });
  }

  return;
}

} // namespace shock
#endif // PGEN_SHOCK_HPP_
