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
#ifndef PGEN_ESCAPE_1D_HPP_
#define PGEN_ESCAPE_1D_HPP_
//! \file escape_1d.hpp
//! \brief 1D spherical planetary atmospheric escape test.
//!
//! ## Problem Description
//! Two (or more) gas species occupy a 1D spherical atmosphere under the gravity of a
//! central point mass.  Species 0 is the "light" component (e.g., hydrogen) that has
//! enough thermal energy to escape via a Parker-type wind.  Species 1 is the "heavy"
//! component (e.g., oxygen) that is gravitationally bound.
//!
//! Chapman-Cowling drag couples the two fluids.  The test demonstrates:
//!   (a) Qualitative: heavy species is entrained and dragged outward by light species.
//!   (b) Quantitative: mass flux of light species matches the isothermal Parker wind
//!       solution at late times (in the absence of heavy species).
//!
//! ## Initial Conditions
//! Isothermal hydrostatic equilibrium is initialized from the stellar surface at r0.
//! For each species, the profile satisfies dP/dr = -rho*G*M/r^2 at fixed T0. The
//! density is integrated from the EOS-derived isothermal derivative dP/drho.
//!
//! ## Boundary Conditions
//! - inner_x1: hydrostatic inflow (density fixed, velocity set by Parker wind)
//! - outer_x1: supersonic outflow (zero-gradient extrapolation)
//!
//! ## Required input parameters (in [problem] block)
//! - rho0_0     : base density species 0 [code units]
//! - rho0_1     : base density species 1 [code units]
//! - T0         : isothermal temperature [EOS/code temperature units]
//! - r0         : inner radius (base of atmosphere) [code units]
//! - gm         : G*M [code units]  (central body, can match gravity package)
//! - npoints    : number of hydrostatic integration steps (optional; default 50)

// C/C++ headers
#include <cmath>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace escape_1d {

struct Escape1DParams {
  int nspecies;
  int npoints;
  Real rho0[8]; // base densities (up to 8 species)
  Real T0;      // common isothermal temperature [EOS/code temperature units]
  Real r0;      // inner boundary radius [code length]
  Real gm;      // G*M [code length^3 / code time^2]
  Real dfloor;
  Real siefloor;
};

//----------------------------------------------------------------------------------------
//! \fn void InitEscape1DParams
//! \brief Reads problem parameters and stores in artemis package.
inline void InitEscape1DParams(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (params.hasKey("escape1d_params")) return;

  auto &gas_pkg = pmb->packages.Get("gas");
  const int nspecies = gas_pkg->Param<int>("nspecies");
  PARTHENON_REQUIRE(nspecies >= 2, "escape_1d requires at least 2 gas species");
  PARTHENON_REQUIRE(nspecies <= 8, "escape_1d supports at most 8 species");

  Escape1DParams ep{};
  ep.nspecies = nspecies;
  ep.npoints = pin->GetOrAddInteger("problem", "npoints", 50);
  ep.T0 = pin->GetOrAddReal("problem", "T0", 1.0e4);
  ep.r0 = pin->GetOrAddReal("problem", "r0", 1.0);
  ep.gm = pin->GetOrAddReal("problem", "gm", 1.0);
  ep.dfloor = gas_pkg->Param<Real>("dfloor");
  ep.siefloor = gas_pkg->Param<Real>("siefloor");
  PARTHENON_REQUIRE(ep.npoints > 0, "escape_1d problem/npoints must be positive");
  PARTHENON_REQUIRE(ep.T0 > 0.0, "escape_1d problem/T0 must be positive");
  PARTHENON_REQUIRE(ep.r0 > 0.0, "escape_1d problem/r0 must be positive");
  PARTHENON_REQUIRE(ep.gm >= 0.0, "escape_1d problem/gm must be non-negative");

  const std::string rho0_key = "rho0_";
  for (int n = 0; n < nspecies; ++n) {
    ep.rho0[n] = pin->GetOrAddReal("problem", rho0_key + std::to_string(n), 1.0e-4);
    PARTHENON_REQUIRE(ep.rho0[n] > 0.0, "escape_1d base densities must be positive");
  }
  params.Add("escape1d_params", ep);
}

// ---------------------------------------------------------------------------
// Helpers: isothermal hydrostatic density at radius r
// ---------------------------------------------------------------------------
KOKKOS_FORCEINLINE_FUNCTION
Real IsothermalDPDRho(const EOS &eos, const Real rho, const Real temp) {
  constexpr Real dlnrho = 1.0e-6;
  const Real pp = eos.PressureFromDensityTemperature(rho * (1.0 + dlnrho), temp);
  const Real pm = eos.PressureFromDensityTemperature(rho * (1.0 - dlnrho), temp);
  return (pp - pm) / (2.0 * dlnrho * rho);
}

KOKKOS_FORCEINLINE_FUNCTION
Real IsothermalDens(const EOS &eos, const Escape1DParams &ep, const int n, const Real r) {
  // With Phi(r) - Phi(r0) = GM * (1/r0 - 1/r), hydrostatic equilibrium gives
  // d ln(rho) / d Phi = -1 / (dP/drho)_T.
  const Real dphi = ep.gm * (1.0 / ep.r0 - 1.0 / r);
  const Real dphi_step = dphi / static_cast<Real>(ep.npoints);
  Real lnrho = std::log(std::max(ep.rho0[n], ep.dfloor));

  for (int p = 0; p < ep.npoints; ++p) {
    const Real rho = std::exp(lnrho);
    if ((dphi_step > 0.0) && (rho <= ep.dfloor)) return ep.dfloor;

    const Real dpdrho = IsothermalDPDRho(eos, rho, ep.T0);
    const Real k1 = -1.0 / std::max(dpdrho, Fuzz<Real>());

    const Real rho_mid = std::exp(lnrho + 0.5 * dphi_step * k1);
    if ((dphi_step > 0.0) && (rho_mid <= ep.dfloor)) return ep.dfloor;

    const Real dpdrho_mid = IsothermalDPDRho(eos, rho_mid, ep.T0);
    lnrho -= dphi_step / std::max(dpdrho_mid, Fuzz<Real>());
  }
  return std::max(std::exp(lnrho), ep.dfloor);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator<GEOM>
//! \brief Sets initial conditions for the atmospheric escape problem.
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  PARTHENON_REQUIRE(do_gas, "escape_1d requires gas hydrodynamics!");

  auto gas_pkg = pmb->packages.Get("gas");
  const auto &eos_d = gas_pkg->Param<ParArray1D<EOS>>("eos_d");

  // Ensure params are initialised (called here and also via InitMeshBlockData)
  InitEscape1DParams(pmb, pin);
  const auto ep = artemis_pkg->Param<Escape1DParams>("escape1d_params");
  const int ns = ep.nspecies;

  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());

  static auto desc_g = MakePackDescriptor<geom::x1v>((pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());

  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");
  auto &pco = pmb->coords;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "escape_1d_ic", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const Real r = coords.x1v();
        for (int n = 0; n < ns; ++n) {
          const Real rho = IsothermalDens(eos_d(n), ep, n, r);
          const Real sie = std::max(
              eos_d(n).InternalEnergyFromDensityTemperature(rho, ep.T0), ep.siefloor);
          v(0, gas::prim::density(n), k, j, i) = rho;
          // Initialize at rest; the Parker wind will develop from the inner BC
          for (int d = 0; d < 3; ++d)
            v(0, gas::prim::velocity(VI(n, d)), k, j, i) = 0.0;
          v(0, gas::prim::sie(n), k, j, i) = sie;
        }
      }); // par_for IC
} // ProblemGenerator

//----------------------------------------------------------------------------------------
//! \fn Escape1DInnerX1
//! \brief Inner boundary: hydrostatic density fix, outward velocity extrapolation.
//! For the Parker wind test the inner boundary is below the sonic point so the
//! flow is subsonic and we fix density/temperature while allowing the velocity to
//! be set by the solution in the ghost cells (extrapolation inwards).
template <Coordinates GEOM, parthenon::IndexDomain BND>
void Escape1DInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  auto pmb = mbd->GetBlockPointer();
  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto ep = artemis_pkg->Param<Escape1DParams>("escape1d_params");
  const int ns = ep.nspecies;

  auto gas_pkg = pmb->packages.Get("gas");
  const auto eos_d = gas_pkg->Param<ParArray1D<EOS>>("eos_d");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");
  auto &pco = pmb->coords;

  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy>((pmb->resolved_packages).get());
  auto v = desc.GetPack(mbd.get());

  const IndexRange ib = pmb->cellbounds.GetBoundsI(BND, parthenon::TE::CC);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(BND, parthenon::TE::CC);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(BND, parthenon::TE::CC);
  const IndexRange ibi =
      pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior, parthenon::TE::CC);

  pmb->par_for(
      "escape_1d_innerX1", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Mirror index into interior
        const int im = ibi.s + (ibi.s - i - 1);
        const int isafe = std::max(im, ibi.s);

        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const Real r = coords.x1v();

        for (int n = 0; n < ns; ++n) {
          const Real rho = IsothermalDens(eos_d(n), ep, n, r);
          const Real sie = std::max(
              eos_d(n).InternalEnergyFromDensityTemperature(rho, ep.T0), ep.siefloor);

          // Velocity: extrapolate from interior (allow Parker wind to develop)
          const Real rho_int = v(0, gas::cons::density(n), k, j, isafe);
          const Real vr_int =
              (rho_int > 0.0) ? v(0, gas::cons::momentum(VI(n, 0)), k, j, isafe) / rho_int
                              : 0.0;
          const Real vr = std::max(vr_int, 0.0); // only outward flow at inner BC

          v(0, gas::cons::density(n), k, j, i) = rho;
          v(0, gas::cons::momentum(VI(n, 0)), k, j, i) = rho * vr;
          v(0, gas::cons::momentum(VI(n, 1)), k, j, i) = 0.0;
          v(0, gas::cons::momentum(VI(n, 2)), k, j, i) = 0.0;
          const Real ke = 0.5 * rho * vr * vr;
          v(0, gas::cons::internal_energy(n), k, j, i) = rho * sie;
          v(0, gas::cons::total_energy(n), k, j, i) = rho * sie + ke;
        }
      });
}
} // namespace escape_1d

#endif // PGEN_ESCAPE_1D_HPP_
