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
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef PGEN_STRAT_HPP_
#define PGEN_STRAT_HPP_
//! \file strat.hpp
//! \brief Initializes a stratified shearing box. Initial conditions are in
//! vertical hydrostatic equilibrium.

// C/C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace strat {

struct StratParams {
  Real rho0, pres0, dens_min, pres_min;
  Real h;
  Real q;
  Real Om0;
  Real gm1;
  Real d2g;
  Real temp0;
};

//----------------------------------------------------------------------------------------
//! \fn void InitStratParams
//! \brief Extracts strat parameters from ParameterInput.
//! NOTE(PDM): In order for our user-defined BCs to be compatible with restarts, we must
//! reset the StratParams struct upon initialization.
inline void InitStratParams(MeshBlock *pmb, ParameterInput *pin) {
  Params &params = pmb->packages.Get("artemis")->AllParams();
  if (!(params.hasKey("strat_params"))) {
    StratParams strat_params;
    strat_params.q = pmb->packages.Get("rotating_frame")->Param<Real>("qshear");
    strat_params.Om0 = pmb->packages.Get("rotating_frame")->Param<Real>("omega");
    strat_params.h = pin->GetOrAddReal("problem", "h", 1.0);
    strat_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    strat_params.dens_min = pin->GetOrAddReal("problem", "dens_min", 1.0e-5);
    strat_params.pres_min = pin->GetOrAddReal("problem", "pres_min", 1.0e-8);
    strat_params.d2g = pin->GetOrAddReal("problem", "dust_to_gas", 0.01);
    strat_params.temp0 = SQR(strat_params.h * strat_params.Om0);
    strat_params.pres0 = strat_params.rho0 * strat_params.temp0;
    params.Add("strat_params", strat_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::strat()
//! \brief Sets initial conditions for shearing box problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "problem = strat only works for Cartesian Coordinates!");

  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  int nspec = Null<int>();
  if (do_dust) {
    auto dust_pkg = pmb->packages.Get("dust");
    nspec = dust_pkg->Param<int>("nspecies");
  }

  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto strat_params = artemis_pkg->Param<StratParams>("strat_params");

  // Dimensionality
  const int ndim = pmb->pmy_mesh->ndim;
  const bool three_d = (ndim == 3);

  // Packing
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &pco = pmb->coords;
  auto &pars = strat_params;

  pmb->par_for(
      "strat", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const Real x = coords.x1v();
        const Real z = coords.x3v();

        const Real vx1 = 0.0;
        const Real vx2 = -pars.q * pars.Om0 * x;
        const Real vx3 = 0.0;
        const Real temp = pars.temp0;
        const Real efac = (three_d) ? std::exp(-SQR(z) / (2.0 * SQR(pars.h))) : 1.0;
        const Real dens = std::max(pars.dens_min, efac * pars.rho0);
        const Real sie = eos_d.InternalEnergyFromDensityTemperature(dens, temp);

        v(0, gas::prim::density(0), k, j, i) = dens;
        v(0, gas::prim::velocity(0), k, j, i) = vx1;
        v(0, gas::prim::velocity(1), k, j, i) = vx2;
        v(0, gas::prim::velocity(2), k, j, i) = vx3;
        v(0, gas::prim::sie(0), k, j, i) = sie;
        if (do_dust) {
          const Real ddens = dens * pars.d2g;
          for (int n = 0; n < nspec; ++n) {
            v(0, dust::prim::density(n), k, j, i) = ddens;
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx1;
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vx2;
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vx3;
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratInnerX1()
//! \brief Sets BCs on -x boundary for shearing box
//!        Extrapolation bc + Outflow no inflow
template <Coordinates GEOM>
inline void ExtrapInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int is = range.s;

  pmb->par_for_bndry(
      "StratInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        geometry::Coords<GEOM> coords_s(pco, k, j, is);
        geometry::Coords<GEOM> coords_s1(pco, k, j, is + 1);
        const Real x0 = coords_s.x1v();
        const Real x1 = coords_s1.x1v();
        const Real dx = x1 - x0;
        const Real x = coords.x1v();

        const Real gv1 = v(0, gas::prim::velocity(0), k, j, is);
        const Real gv2 = v(0, gas::prim::velocity(1), k, j, is);
        const Real gv3 = v(0, gas::prim::velocity(2), k, j, is);
        const Real gv2p1 = v(0, gas::prim::velocity(1), k, j, is + 1);
        const Real vx1g = (gv1 > 0.0) ? 0.0 : gv1;
        const Real vx2g = gv2 + (gv2p1 - gv2) * (x - x0) / dx;
        const Real vx3g = gv3;
        const Real densg = v(0, gas::prim::density(0), k, j, is);
        const Real sieg = v(0, gas::prim::sie(0), k, j, is);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), k, j, is);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), k, j, is);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), k, j, is);
            const Real dv2p1 = v(0, dust::prim::velocity(VI(n, 1)), k, j, is + 1);
            const Real vx1d = (dv1 > 0.0) ? 0.0 : dv1;
            const Real vx2d = dv2 + (dv2p1 - dv2) * (x - x0) / dx;
            const Real vx3d = dv3;
            const Real densd = v(0, dust::prim::density(n), k, j, is);
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx1d;
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vx2d;
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vx3d;
            v(0, dust::prim::density(n), k, j, i) = densd;
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratOuterX1()
//! \brief Sets BCs on +y boundary in shearing box
//!        Extrapolation bc + Outflow no inflow
template <Coordinates GEOM>
inline void ExtrapOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int ie = range.e;

  pmb->par_for_bndry(
      "StratOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        geometry::Coords<GEOM> coords_e(pco, k, j, ie);
        geometry::Coords<GEOM> coords_e1(pco, k, j, ie - 1);
        const Real x0 = coords_e.x1v();
        const Real x1 = coords_e1.x1v();
        const Real dx = x0 - x1;
        const Real x = coords.x1v();

        const Real gv1 = v(0, gas::prim::velocity(0), k, j, ie);
        const Real gv2 = v(0, gas::prim::velocity(1), k, j, ie);
        const Real gv3 = v(0, gas::prim::velocity(2), k, j, ie);
        const Real gv2m1 = v(0, gas::prim::velocity(1), k, j, ie - 1);
        const Real vx1g = (gv1 < 0.0) ? 0.0 : gv1;
        const Real vx2g = gv2 + (gv2 - gv2m1) * (x - x0) / dx;
        const Real vx3g = gv3;
        const Real densg = v(0, gas::prim::density(0), k, j, ie);
        const Real sieg = v(0, gas::prim::sie(0), k, j, ie);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), k, j, ie);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), k, j, ie);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), k, j, ie);
            const Real dv2m1 = v(0, dust::prim::velocity(VI(n, 1)), k, j, ie - 1);
            const Real vx1d = (dv1 < 0.0) ? 0.0 : dv1;
            const Real vx2d = dv2 + (dv2 - dv2m1) * (x - x0) / dx;
            const Real vx3d = dv3;
            const Real ddens = v(0, dust::prim::density(n), k, j, ie);
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx1d;
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vx2d;
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vx3d;
            v(0, dust::prim::density(n), k, j, i) = ddens;
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratInnerX2()
//! \brief Sets BCs on -y boundary for shearing box
//!        The boundaries are a mix of inflow-outflow following the steady-state
//!        solution  vy = - q Omega x
//!
//!                    /\
//!                    ||
//!          y=ymax +--------+----------+
//!                 |        |   ||     |
//!                 |        |   \/     |
//!                 |        |          |
//!                 |  /\    |          |
//!                 |  ||    |          |
//!          y=ymin +--------+----------+
//!                         x=0   ||
//!                               \/
//!
template <Coordinates GEOM>
inline void ShearInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  auto &pars = artemis_pkg->Param<StratParams>("strat_params");
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;

  pmb->par_for_bndry(
      "StratInnerX2", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const Real z = coords.x3v();
        const Real x = coords.x1v();
        const Real xf = coords.bnds.x1[0];

        const Real vy0 = -pars.q * pars.Om0 * x;

        const Real gv1 = v(0, gas::prim::velocity(0), k, js, i);
        const Real gv2 = v(0, gas::prim::velocity(1), k, js, i);
        const Real gv3 = v(0, gas::prim::velocity(2), k, js, i);
        const Real vx1g = gv1;
        const Real vx2g = (xf >= 0) ? ((gv2 > 0.) ? 0.0 : gv2) : vy0;
        const Real vx3g = gv3;
        const Real densg = v(0, gas::prim::density(0), k, js, i);
        const Real sieg = v(0, gas::prim::sie(0), k, js, i);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), k, js, i);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), k, js, i);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), k, js, i);
            const Real vx1d = dv1;
            const Real vx2d = (xf >= 0) ? ((dv2 > 0.) ? 0.0 : dv2) : vy0;
            const Real vx3d = dv3;
            const Real densd = v(0, dust::prim::density(n), k, js, i);
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx1d;
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vx2d;
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vx3d;
            v(0, dust::prim::density(n), k, j, i) = densd;
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratOuterX2()
//! \brief Sets BCs on +y boundary in shearing box
//!        The boundaries are a mix of inflow-outflow following the steady-state
//!        solution  vy = - q Omega x
//!
//!                    /\
//!                    ||
//!          y=ymax +--------+----------+
//!                 |        |   ||     |
//!                 |        |   \/     |
//!                 |        |          |
//!                 |  /\    |          |
//!                 |  ||    |          |
//!          y=ymin +--------+----------+
//!                         x=0   ||
//!                               \/
//!
template <Coordinates GEOM>
inline void ShearOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  auto &pars = artemis_pkg->Param<StratParams>("strat_params");
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int je = range.e;

  pmb->par_for_bndry(
      "StratOuterX2", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const Real z = coords.x3v();
        const Real x = coords.x1v();
        const Real xf = coords.bnds.x1[0];

        const Real vy0 = -pars.q * pars.Om0 * x;

        const Real gv1 = v(0, gas::prim::velocity(0), k, je, i);
        const Real gv2 = v(0, gas::prim::velocity(1), k, je, i);
        const Real gv3 = v(0, gas::prim::velocity(2), k, je, i);
        const Real vx1g = gv1;
        const Real vx2g = (xf < 0) ? ((gv2 < 0.0) ? 0.0 : gv2) : vy0;
        const Real vx3g = gv3;
        const Real densg = v(0, gas::prim::density(0), k, je, i);
        const Real sieg = v(0, gas::prim::sie(0), k, je, i);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), k, je, i);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), k, je, i);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), k, je, i);
            const Real vx1d = dv1;
            const Real vx2d = (xf < 0) ? ((dv2 < 0.0) ? 0.0 : dv2) : vy0;
            const Real vx3d = dv3;
            const Real densd = v(0, dust::prim::density(n), k, je, i);
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx1d;
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vx2d;
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vx3d;
            v(0, dust::prim::density(n), k, j, i) = densd;
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratInnerX3()
//! \brief Sets BCs on -z boundary in shearing box
//! This is an outflow condition assuming rho = exp(-z^2/(2 H^2)) where H = cs/Omega
//! Extrapolation bc + Outflow no inflow
template <Coordinates GEOM>
inline void ExtrapInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsK(IndexDomain::interior, TE::CC);
  const int ks = range.s;

  pmb->par_for_bndry(
      "StratInnerX3", nb, IndexDomain::inner_x3, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        geometry::Coords<GEOM> coords_s(pco, ks, j, i);
        geometry::Coords<GEOM> coords_s1(pco, ks + 1, j, i);

        const Real z = coords.x3v();
        const Real z0 = coords_s.x3v();
        const Real z1 = coords_s1.x3v();
        const Real dz = z1 - z0;

        const Real gv1 = v(0, gas::prim::velocity(0), ks, j, i);
        const Real gv2 = v(0, gas::prim::velocity(1), ks, j, i);
        const Real gv3 = v(0, gas::prim::velocity(2), ks, j, i);
        const Real vx1g = gv1;
        const Real vx2g = gv2;
        const Real vx3g = (gv3 > 0.0) ? 0.0 : gv3;
        const Real gd = v(0, gas::prim::density(0), ks, j, i);
        const Real gdp1 = v(0, gas::prim::density(0), ks + 1, j, i);
        const Real drhog = gdp1 / gd;
        const Real densg = gd * std::pow(drhog, (z - z0) / dz);
        const Real sieg = v(0, gas::prim::sie(0), ks, j, i);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), ks, j, i);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), ks, j, i);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), ks, j, i);
            const Real vx1d = dv1;
            const Real vx2d = dv2;
            const Real vx3d = (dv3 > 0.0) ? 0.0 : dv3;
            const Real dd = v(0, dust::prim::density(n), ks, j, i);
            const Real ddp1 = v(0, dust::prim::density(n), ks + 1, j, i);
            const Real drhod = ddp1 / dd;
            const Real densd = dd * std::pow(drhod, (z - z0) / dz);
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx1d;
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vx2d;
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vx3d;
            v(0, dust::prim::density(n), k, j, i) = densd;
          }
        }
      });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratOuterX3()
//! \brief Sets BCs on +z boundary in shearing box
template <Coordinates GEOM>
inline void ExtrapOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  //  Extrapolation bc + Outflow no inflow
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsK(IndexDomain::interior, TE::CC);
  const int ke = range.e;

  pmb->par_for_bndry(
      "StratOuterX3", nb, IndexDomain::outer_x3, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        geometry::Coords<GEOM> coords_e(pco, ke, j, i);
        geometry::Coords<GEOM> coords_e1(pco, ke - 1, j, i);

        const Real z = coords.x3v();
        const Real z0 = coords_e.x3v();
        const Real z1 = coords_e1.x3v();
        const Real dz = z0 - z1;

        const Real gv1 = v(0, gas::prim::velocity(0), ke, j, i);
        const Real gv2 = v(0, gas::prim::velocity(1), ke, j, i);
        const Real gv3 = v(0, gas::prim::velocity(2), ke, j, i);
        const Real vx1g = gv1;
        const Real vx2g = gv2;
        const Real vx3g = (gv3 < 0.0) ? 0.0 : gv3;
        const Real gd = v(0, gas::prim::density(0), ke, j, i);
        const Real gdm1 = v(0, gas::prim::density(0), ke - 1, j, i);
        const Real drhog = gd / gdm1;
        const Real densg = gd * std::pow(drhog, (z - z0) / dz);
        const Real sieg = v(0, gas::prim::sie(0), ke, j, i);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), ke, j, i);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), ke, j, i);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), ke, j, i);
            const Real vx1d = dv1;
            const Real vx2d = dv2;
            const Real vx3d = (dv3 < 0.0) ? 0.0 : dv3;
            const Real dd = v(0, dust::prim::density(n), ke, j, i);
            const Real ddm1 = v(0, dust::prim::density(n), ke - 1, j, i);
            const Real drhod = dd / ddm1;
            const Real densd = dd * std::pow(drhod, (z - z0) / dz);
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vx1d;
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vx2d;
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vx3d;
            v(0, dust::prim::density(n), k, j, i) = densd;
          }
        }
      });
  return;
}

} // namespace strat
#endif // PGEN_STRAT_HPP_
