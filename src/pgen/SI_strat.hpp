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

//! \file streaming_stratified.hpp
//! \brief Initializes a stratified shearing box. Initial conditions are in
//! vertical hydrostatic equilibrium.
#ifndef PGEN_SI_STRAT_HPP_
#define PGEN_SI_STRAT_HPP_

// C/C++ headers
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace SI_strat {

struct SI_StratParams {
  Real rho0, pres0, dens_min, pres_min;
  Real h;
  Real q;
  Real Om0;
  int ShBoxCoord;
  bool StratFlag;
  Real gm1;
  Real d2g;
  Real iso_cs;
  Real temp0;
  Real etaVk;
  Real Kai0;
  Real amp;
  Real qomL;
  ParArray1D<Real> Hdust;
};

//----------------------------------------------------------------------------------------
//! \fn void InitStratParams
//! \brief Extracts strat parameters from ParameterInput.
//! NOTE(@pdmullen): In order for our user-defined BCs to be compatible with restarts,
//! we must reset the StratParams struct upon initialization.
inline void InitStratParams(MeshBlock *pmb, ParameterInput *pin) {
  Params &params = pmb->packages.Get("artemis")->AllParams();
  if (!(params.hasKey("SI_strat_params"))) {
    SI_StratParams strat_params;
    strat_params.q = pin->GetOrAddReal("rotating_frame", "qshear", 0.0);
    strat_params.Om0 = pin->GetOrAddReal("rotating_frame", "omega", 0.0);
    strat_params.h = pin->GetOrAddReal("problem", "h", 1.0);
    strat_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    strat_params.dens_min = pin->GetOrAddReal("problem", "dens_min", 1.0e-5);
    strat_params.pres_min = pin->GetOrAddReal("problem", "pres_min", 1.0e-8);
    strat_params.d2g = pin->GetOrAddReal("problem", "dust_to_gas", 0.01);
    strat_params.etaVk = pin->GetOrAddReal("problem", "etaVk", 0.05);
    strat_params.amp = pin->GetOrAddReal("problem", "amp", 0.05);
    strat_params.iso_cs = strat_params.h * strat_params.Om0;
    strat_params.Kai0 = 2.0 * strat_params.etaVk * strat_params.iso_cs;
    strat_params.temp0 = SQR(strat_params.iso_cs);
    strat_params.pres0 = strat_params.rho0 * strat_params.temp0;
    strat_params.ShBoxCoord = pin->GetOrAddInteger("rotating_frame", "shboxcoord", 1);
    strat_params.StratFlag =
        pin->GetOrAddBoolean("rotating_frame", "stratified_flag", true);
    const Real xmin = pin->GetReal("parthenon/mesh", "x1min");
    const Real xmax = pin->GetReal("parthenon/mesh", "x1max");
    strat_params.qomL = strat_params.q * strat_params.Om0 * (xmax - xmin);

    if (strat_params.StratFlag) {
      auto artemis_pkg = pmb->packages.Get("artemis");
      const bool do_dust = artemis_pkg->Param<bool>("do_dust");
      if (do_dust) {
        auto dust_pkg = pmb->packages.Get("dust");
        const int nDust = dust_pkg->Param<int>("nspecies");
        strat_params.Hdust = ParArray1D<Real>("hdust", nDust);
        auto Hdust_h = Kokkos::create_mirror_view(strat_params.Hdust);
        for (int n = 0; n < nDust; n++) {
          Hdust_h(n) =
              strat_params.h * pin->GetReal("dust", "Hratio_" + std::to_string(n + 1));
        }
        Kokkos::deep_copy(strat_params.Hdust, Hdust_h);
      }
    }

    params.Add("SI_strat_params", strat_params);

    // using BF = parthenon::BoundaryFace;
    // artemis_pkg->UserBoundaryFunctions[BF::inner_x1].push_back(ExtrapInnerX1<Coordinates::cartesian>());
  }
}

// std::uniform_real_distribution<Real> ran(-0.5, 0.5);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::strat()
//! \brief Sets initial conditions for shearing box problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // static std::mt19937 iseed(Globals::my_rank);
  // std::mt19937 iseed(pmb->gid);
  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "problem = strat only works for Cartesian Coordinates!");

  // Strat parameters
  auto strat_params = artemis_pkg->Param<SI_StratParams>("SI_strat_params");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  bool const_stopping_time = pin->GetOrAddBoolean("dust", "const_stopping_time", true);
  PARTHENON_REQUIRE(do_dust, "dust_collision pgen requires const_stopping_time=true!");

  int nDust = Null<int>();
  ParArray1D<Real> Stokes_number;
  ParArray1D<Real> Hdust;
  if (do_dust) {
    auto dust_pkg = pmb->packages.Get("dust");
    nDust = dust_pkg->Param<int>("nspecies");
    Stokes_number = dust_pkg->template Param<ParArray1D<Real>>("stopping_time");
    if (strat_params.StratFlag) {
      Hdust = strat_params.Hdust;
    }
  }

  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  Real AN(0.0), BN(0.0), Psi(0.0), kappap, kappap2;
  kappap = 2.0 * (2.0 - strat_params.q);
  kappap2 = SQR(kappap);
  auto &pars = strat_params;
  if (do_dust) {
    // for (int n = 0; n < nDust; ++n) {
    Kokkos::parallel_reduce(
        "SI_strat::AN", nDust,
        KOKKOS_LAMBDA(const int n, Real &lsum1, Real &lsum2) {
          lsum1 +=
              (pars.d2g * Stokes_number(n)) / (1.0 + kappap2 * SQR(Stokes_number(n)));
          lsum2 += (pars.d2g) / (1.0 + kappap2 * SQR(Stokes_number(n)));
        },
        AN, BN);
    AN *= kappap2;
    BN += 1.0;
    Psi = 1.0 / (SQR(AN) + kappap2 * SQR(BN));
  }

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

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/pmb->gid);

  if (pars.ShBoxCoord == 1) { // x-y plane
    pmb->par_for(
        "SIstrat", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(pco, k, j, i);
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          const Real x = coords.x1v();
          const Real z = coords.x3v();

          const Real K_vel = pars.q * pars.Om0 * x;
          const Real vx1 = AN * pars.Kai0 * Psi;
          const Real vx2 = -K_vel - 0.5 * kappap2 * BN * pars.Kai0 * Psi;
          const Real vx3 = 0.0;

          const Real del_vx1 =
              pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
          const Real del_vx2 =
              pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
          const Real del_vx3 =
              pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);

          const Real temp = pars.temp0;
          const Real efac = (three_d && pars.StratFlag)
                                ? std::exp(-z * z / (2. * pars.h * pars.h))
                                : 1.;
          const Real dens = std::max(pars.dens_min, efac * pars.rho0);

          const Real sie = eos_d.InternalEnergyFromDensityTemperature(dens, temp);

          v(0, gas::prim::density(0), k, j, i) = dens;
          v(0, gas::prim::velocity(0), k, j, i) = vx1 + del_vx1;
          v(0, gas::prim::velocity(1), k, j, i) = vx2 + del_vx2;
          v(0, gas::prim::velocity(2), k, j, i) = vx3 + del_vx3;
          v(0, gas::prim::sie(0), k, j, i) = sie;
          if (do_dust) {
            const Real ddens0 = pars.rho0 * pars.d2g;
            for (int n = 0; n < nDust; ++n) {
              const Real efac_d =
                  (three_d && pars.StratFlag)
                      ? std::exp(-z * z / (2. * SQR(Hdust(n)))) * pars.h / Hdust(n)
                      : 1.;
              const Real ddens = ddens0 * efac_d;
              const Real vxd1 = ((vx1 + 2.0 * Stokes_number(n) * (vx2 + K_vel)) /
                                 (1.0 + kappap2 * SQR(Stokes_number(n))));
              const Real vxd2 =
                  -K_vel + (((vx2 + K_vel) - (2.0 - pars.q) * Stokes_number(n) * vx1) /
                            (1.0 + kappap2 * SQR(Stokes_number(n))));
              const Real vxd3 = 0.0;
              const Real del_vxd1 =
                  pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
              const Real del_vxd2 =
                  pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
              const Real del_vxd3 =
                  pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);

              v(0, dust::prim::density(n), k, j, i) = ddens;
              v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vxd1 + del_vxd1;
              v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vxd2 + del_vxd2;
              v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vxd3 + del_vxd3;
            }
          }
          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });
  } else { // ShBoxCoord == 2, x-z plane
    pmb->par_for(
        "SIstrat", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(pco, k, j, i);
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          const Real x = coords.x1v();
          const Real z = coords.x2v();

          const Real K_vel = pars.q * pars.Om0 * x;
          const Real vx1 = AN * pars.Kai0 * Psi;
          const Real vx2 = 0.0;
          const Real vx3 = -K_vel - 0.5 * kappap2 * BN * pars.Kai0 * Psi;

          const Real del_vx1 =
              pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
          const Real del_vx2 =
              pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
          const Real del_vx3 =
              pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);

          const Real temp = pars.temp0;
          const Real efac =
              (pars.StratFlag) ? std::exp(-z * z / (2. * pars.h * pars.h)) : 1.;
          const Real dens = std::max(pars.dens_min, efac * pars.rho0);

          const Real sie = eos_d.InternalEnergyFromDensityTemperature(dens, temp);

          v(0, gas::prim::density(0), k, j, i) = dens;
          v(0, gas::prim::velocity(0), k, j, i) = vx1 + del_vx1;
          v(0, gas::prim::velocity(1), k, j, i) = vx2 + del_vx2;
          v(0, gas::prim::velocity(2), k, j, i) = vx3 + del_vx3;
          v(0, gas::prim::sie(0), k, j, i) = sie;
          if (do_dust) {
            const Real ddens0 = pars.rho0 * pars.d2g;
            for (int n = 0; n < nDust; ++n) {
              const Real efac_d =
                  pars.StratFlag
                      ? std::exp(-z * z / (2. * SQR(Hdust(n)))) * pars.h / Hdust(n)
                      : 1.;
              const Real ddens = ddens0 * efac_d;
              const Real vxd1 = ((vx1 + 2.0 * Stokes_number(n) * (vx3 + K_vel)) /
                                 (1.0 + kappap2 * SQR(Stokes_number(n))));
              const Real vxd2 = 0.0;
              const Real vxd3 =
                  -K_vel + (((vx3 + K_vel) - (2.0 - pars.q) * Stokes_number(n) * vx1) /
                            (1.0 + kappap2 * SQR(Stokes_number(n))));
              const Real del_vxd1 =
                  pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
              const Real del_vxd2 =
                  pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);
              const Real del_vxd3 =
                  pars.amp * pars.iso_cs * generator.drand(-0.5, 0.5); // ran(iseed);

              v(0, dust::prim::density(n), k, j, i) = ddens;
              v(0, dust::prim::velocity(VI(n, 0)), k, j, i) = vxd1 + del_vxd1;
              v(0, dust::prim::velocity(VI(n, 1)), k, j, i) = vxd2 + del_vxd2;
              v(0, dust::prim::velocity(VI(n, 2)), k, j, i) = vxd3 + del_vxd3;
            }
          }
          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratInnerX1()
//! \brief Sets BCs on -x boundary for shearing box
template <Coordinates GEOM>
inline void ExtrapInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  //  Extrapolation bc + Outflow no inflow
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;

  auto pmb = mbd->GetBlockPointer();
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int is = range.s;

  auto strat_params = artemis_pkg->Param<SI_StratParams>("SI_strat_params");
  const int ndim = pmb->pmy_mesh->ndim;
  if (strat_params.ShBoxCoord == 2 && ndim == 2) {
    auto &pars = strat_params;
    pmb->par_for_bndry(
        "StratInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
        coarse, fine,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          v(0, gas::prim::velocity(2), k, j, i) += pars.qomL;

          if (do_dust) {
            for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
              v(0, dust::prim::velocity(VI(n, 2)), k, j, i) += pars.qomL;
            }
          }
        });
  } else {
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
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::StratOuterX1()
//! \brief Sets BCs on +y boundary in shearing box
template <Coordinates GEOM>
inline void ExtrapOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  //  Extrapolation bc + Outflow no inflow
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;

  auto pmb = mbd->GetBlockPointer();
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto strat_params = artemis_pkg->Param<SI_StratParams>("SI_strat_params");
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int ie = range.e;

  const int ndim = pmb->pmy_mesh->ndim;
  if (strat_params.ShBoxCoord == 2 && ndim == 2) {
    auto &pars = strat_params;
    pmb->par_for_bndry(
        "StratOuterX1_2", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
        coarse, fine,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          v(0, gas::prim::velocity(2), k, j, i) -= pars.qomL;

          if (do_dust) {
            for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
              v(0, dust::prim::velocity(VI(n, 2)), k, j, i) -= pars.qomL;
            }
          }
        });
  } else {
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

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
  auto &pars = artemis_pkg->Param<SI_StratParams>("SI_strat_params");
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

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
  auto &pars = artemis_pkg->Param<SI_StratParams>("SI_strat_params");
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
//! \fn void ProblemGenerator::StratOuterX2()
//! \brief Sets BCs on +y boundary in shearing box
template <Coordinates GEOM>
inline void ExtrapInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;

  auto strat_params = artemis_pkg->Param<SI_StratParams>("SI_strat_params");
  auto &pars = strat_params;
  ParArray1D<Real> Hdust = pars.Hdust;

  pmb->par_for_bndry(
      "StratInnerX3", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        geometry::Coords<GEOM> coords_s(pco, k, js, i);
        geometry::Coords<GEOM> coords_s1(pco, k, js + 1, i);

        const Real z = coords.x2v();
        const Real z0 = coords_s.x2v();
        const Real z1 = coords_s1.x2v();
        const Real dz = z1 - z0;

        const Real gv1 = v(0, gas::prim::velocity(0), k, js, i);
        const Real gv2 = v(0, gas::prim::velocity(1), k, js, i);
        const Real gv3 = v(0, gas::prim::velocity(2), k, js, i);
        const Real vx1g = gv1;
        const Real vx2g = (gv2 > 0.0) ? 0.0 : gv2;
        const Real vx3g = gv3;
        // const Real gd = v(0, gas::prim::density(0), k, js, i);
        // const Real gdp1 = v(0, gas::prim::density(0), k, js + 1, i);
        // const Real drhog = gdp1 / gd;
        // const Real densg = gd * std::pow(drhog, (z - z0) / dz);
        const Real efac =
            (pars.StratFlag) ? std::exp(-z * z / (2. * pars.h * pars.h)) : 1.;
        const Real densg = std::max(pars.dens_min, efac * pars.rho0);
        const Real sieg = v(0, gas::prim::sie(0), k, js, i);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          const Real ddens0 = pars.rho0 * pars.d2g;
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), k, js, i);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), k, js, i);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), k, js, i);
            const Real vx1d = dv1;
            const Real vx2d = (dv2 > 0.0) ? 0.0 : dv2;
            const Real vx3d = dv3;
            const Real efac_d = pars.StratFlag ? std::exp(-z * z / (2. * SQR(Hdust(n)))) *
                                                     pars.h / Hdust(n)
                                               : 1.;
            const Real densd = ddens0 * efac_d;
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
//! \fn void ProblemGenerator::StratOuterX2()
//! \brief Sets BCs on +z boundary in shearing box
template <Coordinates GEOM>
inline void ExtrapOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  //  Extrapolation bc + Outflow no inflow
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int je = range.e;

  auto strat_params = artemis_pkg->Param<SI_StratParams>("SI_strat_params");
  auto &pars = strat_params;
  ParArray1D<Real> Hdust = pars.Hdust;

  pmb->par_for_bndry(
      "StratOuterX3", nb, IndexDomain::outer_x3, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        geometry::Coords<GEOM> coords_e(pco, k, je, i);
        geometry::Coords<GEOM> coords_e1(pco, k, je - 1, i);

        const Real z = coords.x2v();
        const Real z0 = coords_e.x2v();
        const Real z1 = coords_e1.x2v();
        const Real dz = z0 - z1;

        const Real gv1 = v(0, gas::prim::velocity(0), k, je, i);
        const Real gv2 = v(0, gas::prim::velocity(1), k, je, i);
        const Real gv3 = v(0, gas::prim::velocity(2), k, je, i);
        const Real vx1g = gv1;
        const Real vx2g = (gv2 < 0.0) ? 0.0 : gv2;
        const Real vx3g = gv3;
        // const Real gd = v(0, gas::prim::density(0), k, je, i);
        // const Real gdm1 = v(0, gas::prim::density(0), k, je - 1, i);
        // const Real drhog = gd / gdm1;
        // const Real densg = gd * std::pow(drhog, (z - z0) / dz);
        const Real efac =
            (pars.StratFlag) ? std::exp(-z * z / (2. * pars.h * pars.h)) : 1.;
        const Real densg = std::max(pars.dens_min, efac * pars.rho0);
        const Real sieg = v(0, gas::prim::sie(0), k, je, i);
        v(0, gas::prim::velocity(0), k, j, i) = vx1g;
        v(0, gas::prim::velocity(1), k, j, i) = vx2g;
        v(0, gas::prim::velocity(2), k, j, i) = vx3g;
        v(0, gas::prim::density(0), k, j, i) = densg;
        v(0, gas::prim::sie(0), k, j, i) = sieg;

        if (do_dust) {
          const Real ddens0 = pars.rho0 * pars.d2g;
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            const Real dv1 = v(0, dust::prim::velocity(VI(n, 0)), k, je, i);
            const Real dv2 = v(0, dust::prim::velocity(VI(n, 1)), k, je, i);
            const Real dv3 = v(0, dust::prim::velocity(VI(n, 2)), k, je, i);
            const Real vx1d = dv1;
            const Real vx2d = (dv2 < 0.0) ? 0.0 : dv2;
            const Real vx3d = dv3;
            // const Real dd = v(0, dust::prim::density(n), k, je, i);
            // const Real ddm1 = v(0, dust::prim::density(n), k, je - 1, i);
            // const Real drhod = dd / ddm1;
            // const Real densd = dd * std::pow(drhod, (z - z0) / dz);
            const Real efac_d = pars.StratFlag ? std::exp(-z * z / (2. * SQR(Hdust(n)))) *
                                                     pars.h / Hdust(n)
                                               : 1.;
            const Real densd = ddens0 * efac_d;
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
//! \fn void ProblemGenerator::StratInnerX3()
//! \brief Sets BCs on -z boundary in shearing box
template <Coordinates GEOM>
inline void ExtrapInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
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

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());
  auto &pco = pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = bounds.GetBoundsK(IndexDomain::interior, TE::CC);
  const int ke = range.e;

  pmb->par_for_bndry(
      "StratOuterX3", nb, IndexDomain::outer_x3, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
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

template <Coordinates GEOM, IndexDomain BDY>
inline void StratBoundary(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  // Options for the shearing box are

  // x1 -> extrapolation or outflow
  // x2 -> inflow
  // x3 -> extrapolation

  auto pmb = mbd->GetBlockPointer();
  auto artemis_pkg = pmb->packages.Get("artemis");
  ArtemisBC bc = ArtemisBC::none;

  if constexpr (BDY == IndexDomain::inner_x1) {
    bc = artemis_pkg->Param<ArtemisBC>("ix1_bc");
    if ((bc == ArtemisBC::extrap) || (bc == ArtemisBC::outflow)) {
      return ExtrapInnerX1<GEOM>(mbd, coarse);
    }
  } else if constexpr (BDY == IndexDomain::outer_x1) {
    bc = artemis_pkg->Param<ArtemisBC>("ox1_bc");
    if ((bc == ArtemisBC::extrap) || (bc == ArtemisBC::outflow)) {
      return ExtrapOuterX1<GEOM>(mbd, coarse);
    }
  } else if constexpr (BDY == IndexDomain::inner_x2) {
    bc = artemis_pkg->Param<ArtemisBC>("ix2_bc");
    if (bc == ArtemisBC::inflow) {
      return ShearInnerX2<GEOM>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return ExtrapInnerX2<GEOM>(mbd, coarse);
    }
  } else if constexpr (BDY == IndexDomain::outer_x2) {
    bc = artemis_pkg->Param<ArtemisBC>("ox2_bc");
    if (bc == ArtemisBC::inflow) {
      return ShearOuterX2<GEOM>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return ExtrapOuterX2<GEOM>(mbd, coarse);
    }
  } else if constexpr (BDY == IndexDomain::inner_x3) {
    bc = artemis_pkg->Param<ArtemisBC>("ix3_bc");
    if (bc == ArtemisBC::extrap) {
      return ExtrapInnerX3<GEOM>(mbd, coarse);
    }
  } else if constexpr (BDY == IndexDomain::outer_x3) {
    bc = artemis_pkg->Param<ArtemisBC>("ox3_bc");
    if (bc == ArtemisBC::extrap) {
      return ExtrapOuterX3<GEOM>(mbd, coarse);
    }
  }
}

} // namespace SI_strat
#endif // PGEN_SI_STRAT_HPP_
