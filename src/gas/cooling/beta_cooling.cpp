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

// Artemis includes
#include "artemis.hpp"
#include "gas/cooling/cooling.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

using namespace parthenon::package::prelude;

namespace Gas {
namespace Cooling {
//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Cooling::BetaCooling
//! \brief Applies beta cooling to the gas
//!
//!   Backward Euler step
//!     dT/dt = - (T - T0)/tc
//!     Tp = T - dt/tc * Tp + dt/tc * T0
//!     Tp = (tc T + dtT0)/(tc + dt)
//!     For beta = tc * Om
//!     Tp = (beta T + om dt T0)/(beta + om dt)
//!     Tp - T = -om dt (T - T0) / (beta + om dt)
template <Coordinates GEOM, TempRefType T>
TaskStatus BetaCooling(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &gas_pkg = pm->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto de_switch = gas_pkg->template Param<Real>("de_switch");

  Real gm = Null<Real>();
  const bool do_gravity = pm->packages.Get("artemis")->template Param<bool>("do_gravity");
  if (do_gravity) gm = pm->packages.Get("gravity")->template Param<Real>("gm");

  auto &cooling_pkg = pm->packages.Get("cooling");
  const Real beta0 = cooling_pkg->template Param<Real>("beta0");
  const Real beta_min = cooling_pkg->template Param<Real>("beta_min");
  const Real escale = cooling_pkg->template Param<Real>("escale");
  TempParams tpars = cooling_pkg->template Param<TempParams>("tpars");

  static auto desc = MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy,
                                        gas::cons::internal_energy, gas::cons::density>(
      resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "BetaCooling", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        Real xv[3] = {Null<Real>()};
        coords.GetCellCenter(xv);

        Real xcyl[3] = {Null<Real>()};
        coords.ConvertToCyl(xv, xcyl);
        const Real rsph2 = xcyl[0] * xcyl[0] + xcyl[2] * xcyl[2];

        Real hx[3] = {Null<Real>()};
        coords.GetScaleFactors(hx);

        // Apply cooling
        const Real T0 = TemperatureProfile<GEOM, T>(coords, time, xv, tpars);
        const Real efac = (T0 > 0.) ? std::exp(-escale * xcyl[2] * xcyl[2] / T0) : 1.;
        const Real beta = beta_min + beta0 * efac;
        const Real omdt = dt * std::sqrt(gm / (rsph2 * std::sqrt(rsph2)));
        for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {
          const Real rv1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) / hx[0];
          const Real rv2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) / hx[1];
          const Real rv3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) / hx[2];

          // Get the specific internal energy
          Real &etot = vmesh(b, gas::cons::total_energy(n), k, j, i);
          Real &eint = vmesh(b, gas::cons::internal_energy(n), k, j, i);
          const Real &dens = vmesh(b, gas::cons::density(n), k, j, i);
          const Real ke = 0.5 * (SQR(rv1) + SQR(rv2) + SQR(rv3)) / dens;
          const Real u_trial = etot - ke;
          const Real sie = (u_trial > de_switch * etot) ? u_trial / dens : eint / dens;

          // Compute the energy change from the temperature change.
          const Real cv = eos_d.SpecificHeatFromDensityInternalEnergy(dens, sie);
          const Real Tn = eos_d.TemperatureFromDensityInternalEnergy(dens, sie);
          const Real dE = -dens * cv * omdt / (beta + omdt) * (Tn - T0);

          // Add this energy change to the conserved fields
          etot += dE;
          eint += dE;
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
template TaskStatus BetaCooling<Coordinates::cartesian, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::cylindrical, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::spherical, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::axisymmetric, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);

} // namespace Cooling
} // namespace Gas
