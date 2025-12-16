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

#ifndef PGEN_ATMOSPHERE_HPP_
#define PGEN_ATMOSPHERE_HPP_
//! \file atmosphere.hpp
//! \brief Atmosphere problem generator for 1D/2D/3D problems. Direction of the wavevector
//! is set to be along the x? axis by using the along_x? input flags, else it is
//! automatically set along the grid diagonal in 2D/3D.
//! This file also contains a function to compute L1 errors in solution.

// NOTE(PDM): The following is adapted from the open-source Athena++/AthenaK
// linear wave test, adapted for pure atmosphere and dust by PDM on 10/20/23

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
#include "utils/artemis_utils.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace atmosphere {

struct AtmosphereParams {
  Real rho0, pres0, dfloor, siefloor;
  Real Om0;
  Real temp0;
  Real kbmu;
  bool three_d;
  bool do_dust;
  int npoints;
};

inline void InitAtmosphereParams(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("atmosphere_params"))) {
    AtmosphereParams atmosphere_params;

    atmosphere_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    atmosphere_params.temp0 = pin->GetReal("problem", "T0");

    if (pin->DoesBlockExist("gravity/linear")) {
      atmosphere_params.Om0 = std::abs(pin->GetOrAddReal("gravity/linear", "dgdx1", 0.0));
    } else {
      atmosphere_params.Om0 = std::abs(pin->GetOrAddReal("problem", "Om0", 0.0));
    }

    auto &gas_pkg = pmb->packages.Get("gas");
    const auto mu = gas_pkg->Param<Real>("mu");
    const auto eos = gas_pkg->Param<EOS>("eos_h");
    auto &constants = artemis_pkg->Param<ArtemisUtils::Constants>("constants");
    atmosphere_params.kbmu = constants.GetKBCode() / (mu * constants.GetAMUCode());
    atmosphere_params.dfloor = gas_pkg->Param<Real>("dfloor");
    atmosphere_params.siefloor = gas_pkg->Param<Real>("siefloor");
    atmosphere_params.pres0 = eos.PressureFromDensityTemperature(
        atmosphere_params.rho0,
        atmosphere_params.temp0); // atmosphere_params.rho0 * atmosphere_params.temp0;

    params.Add("atmosphere_params", atmosphere_params);
  }
}

KOKKOS_INLINE_FUNCTION
Real InitialDensity(const EOS &eos, const AtmosphereParams &pars, const Real x) {
  // dP/dz = -rho * g
  // dP = (dP/drho)_T drho
  // drho/dz = -rho * g / (dP/drho)_T
  Real dens = pars.rho0;
  if (std::abs(x) > 1e-16) {
    const Real dz = std::abs(x) / static_cast<Real>(pars.npoints);
    const Real dlnr = 1e-6;
    const Real ldmin = std::log(pars.dfloor / pars.rho0);
    Real pres = eos.PressureFromDensityTemperature(dens, pars.temp0);
    Real zj = 0.0;
    Real ld = 0.0;
    dens = std::exp(ld);
    for (int j = 0; j < pars.npoints; j++) {
      // ln(d/d0) = \int_0^z - Omega^2 z dz
      Real pp = eos.PressureFromDensityTemperature(dens * (1. + dlnr), pars.temp0);
      Real pm = eos.PressureFromDensityTemperature(dens * (1. - dlnr), pars.temp0);
      Real dPdrho = (pp - pm) / (dlnr * dens);
      ld -= 0.5 * (2 * j + 1) * SQR(pars.Om0 * dz) / (dPdrho + Fuzz<Real>());
      dens = std::exp(ld);
      if (dens <= pars.dfloor) return pars.dfloor;
    }
  }
  return std::max(pars.dfloor, dens);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Atmosphere_()
//! \brief Sets initial conditions
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  // dP/dz = rho grad(Phi)
  // Phi = -0.5 \Omega^2 x^2

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto atmosphere_params = artemis_pkg->Param<AtmosphereParams>("atmosphere_params");
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v>((pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  const auto &pars = atmosphere_params;
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  pmb->par_for(
      "pgen_atmosphere", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates

        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto &xv = coords.GetCellCenter(vg, 0, k, j, i);

        const Real dens = InitialDensity(eos_d, pars, xv[0]);
        const Real sie = std::max(
            pars.siefloor, eos_d.InternalEnergyFromDensityTemperature(dens, pars.temp0));

        v(0, gas::prim::density(0), k, j, i) = dens;
        v(0, gas::prim::velocity(0), k, j, i) = 0.0;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = sie;
      });
}

template <Coordinates GEOM>
inline void ExtrapInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  // Packing
  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;

  // Extract artemis package and params
  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto &pars = artemis_pkg->Param<AtmosphereParams>("atmosphere_params");

  // Extract gas package and params
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  // Coordinates and indexing
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsK(IndexDomain::interior, TE::CC);
  const int ks = range.s;
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  pmb->par_for_bndry(
      "AtmosphereInnerX3", nb, IndexDomain::inner_x3, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        geometry::Coords<GEOM> coords_s(cpars, pco, ks, j, i);

        const Real x = coords.x1v();
        const Real x0 = coords_s.x1v();

        // isothermal through boundary

        for (int n = 0; n < v.GetSize(0, gas::prim::density()); ++n) {
          const Real &gv1 = v(0, gas::prim::velocity(VI(n, 0)), ks, j, i);
          const Real vx2g = v(0, gas::prim::velocity(VI(n, 1)), ks, j, i);
          const Real vx3g = v(0, gas::prim::velocity(VI(n, 2)), ks, j, i);
          const Real vx1g = (gv1 > 0.0) ? 0.0 : gv1;
          const Real &gd = v(0, gas::prim::density(n), ks, j, i);
          const Real &gsie = v(0, gas::prim::sie(n), ks, j, i);
          const Real Tg = eos_d.TemperatureFromDensityInternalEnergy(gd, gsie);

          const Real pm = eos_d.PressureFromDensityTemperature(gd * (1. - 1e-6), Tg);
          const Real pp = eos_d.PressureFromDensityTemperature(gd * (1. + 1e-6), Tg);
          const Real dPdrho = (pp - pm) / (gd * 1e-6);
          const Real efac = std::exp(-(SQR(x) - SQR(x0)) * SQR(pars.Om0) /
                                     (2.0 * dPdrho + Fuzz<Real>()));
          const Real rhog = std::max(gd * efac, pars.dfloor);

          v(0, gas::prim::velocity(VI(n, 0)), k, j, i) = vx1g;
          v(0, gas::prim::velocity(VI(n, 1)), k, j, i) = vx2g;
          v(0, gas::prim::velocity(VI(n, 2)), k, j, i) = vx3g;
          v(0, gas::prim::density(n), k, j, i) = rhog;
          v(0, gas::prim::sie(n), k, j, i) = std::max(
              pars.siefloor, eos_d.InternalEnergyFromDensityTemperature(rhog, Tg));
        }
      });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::AtmosphereOuterX3()
//! \brief Sets BCs on +z boundary in shearing box
template <Coordinates GEOM>
inline void ExtrapOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT
  //  Extrapolation bc + Outflow no inflow
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  // Packing
  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;

  // Extract artemis package and params
  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto &pars = artemis_pkg->Param<AtmosphereParams>("atmosphere_params");

  // Extract gas package and params
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  // Coordinates and indexing
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsK(IndexDomain::interior, TE::CC);
  const int ke = range.e;
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  pmb->par_for_bndry(
      "AtmosphereOuterX3", nb, IndexDomain::outer_x3, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        geometry::Coords<GEOM> coords_e(cpars, pco, ke, j, i);

        const Real x = coords.x1v();
        const Real x0 = coords_e.x1v();

        // isothermal through boundary

        for (int n = 0; n < v.GetSize(0, gas::prim::density()); ++n) {
          const Real &gv1 = v(0, gas::prim::velocity(VI(n, 0)), ke, j, i);
          const Real vx2g = v(0, gas::prim::velocity(VI(n, 1)), ke, j, i);
          const Real vx3g = v(0, gas::prim::velocity(VI(n, 2)), ke, j, i);
          const Real vx1g = (gv1 < 0.0) ? 0.0 : gv1;
          const Real &gd = v(0, gas::prim::density(n), ke, j, i);
          const Real &gsie = v(0, gas::prim::sie(n), ke, j, i);
          const Real Tg = eos_d.TemperatureFromDensityInternalEnergy(gd, gsie);
          const Real pm = eos_d.PressureFromDensityTemperature(gd * (1. - 1e-6), Tg);
          const Real pp = eos_d.PressureFromDensityTemperature(gd * (1. + 1e-6), Tg);
          const Real dPdrho = (pp - pm) / (gd * 1e-6);
          const Real efac = std::exp(-(SQR(x) - SQR(x0)) * SQR(pars.Om0) /
                                     (2.0 * dPdrho + Fuzz<Real>()));
          const Real rhog = std::max(pars.dfloor, gd * efac);

          v(0, gas::prim::velocity(VI(n, 0)), k, j, i) = vx1g;
          v(0, gas::prim::velocity(VI(n, 1)), k, j, i) = vx2g;
          v(0, gas::prim::velocity(VI(n, 2)), k, j, i) = vx3g;
          v(0, gas::prim::density(n), k, j, i) = rhog;
          v(0, gas::prim::sie(n), k, j, i) = std::max(
              pars.siefloor, eos_d.InternalEnergyFromDensityTemperature(rhog, Tg));
        }
      });

  return;
}

} // namespace atmosphere
#endif // PGEN_ATMOSPHERE_HPP_
