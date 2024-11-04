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
#include "gravity/nbody_gravity.hpp"
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
template <Coordinates GEOM, TempRefType TTYP>
TaskStatus BetaCooling(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &gas_pkg = pm->packages.Get("gas");
  const auto &eos_d = gas_pkg->template Param<EOS>("eos_d");
  const auto de_switch = gas_pkg->template Param<Real>("de_switch");
  const auto dflr_gas = gas_pkg->template Param<Real>("dfloor");
  const auto sieflr_gas = gas_pkg->template Param<Real>("siefloor");

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

  [[maybe_unused]] ParArray1D<NBody::Particle> particles;
  [[maybe_unused]] int npart = 0;
  if constexpr (TTYP == TempRefType::nbody) {
    auto &nbody_pkg = pm->packages.Get("nbody");
    particles = nbody_pkg->template Param<ParArray1D<NBody::Particle>>("particles");
    npart = static_cast<int>(particles.size());
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "BetaCooling", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();

        const auto &xcyl = coords.ConvertToCyl(xv);
        const Real rsph2 = xcyl[0] * xcyl[0] + xcyl[2] * xcyl[2];

        // Apply cooling
        Real ir1 = 1.0 / std::sqrt(rsph2);
        Real T0 = TemperatureProfile<GEOM, TTYP>(coords, time, xv, tpars);

        // capture these since they are first captured in the following constexpr
        [[maybe_unused]] auto &particles_ = particles;
        [[maybe_unused]] auto &npart_ = npart;
        [[maybe_unused]] auto &gm_ = gm;
        if constexpr (TTYP == TempRefType::nbody) {
          ir1 = -Gravity::NBodyPotential<GEOM>(coords, xv, particles_, npart_) / gm_;
          T0 = tpars.tfloor + tpars.tsph * std::pow(ir1, -tpars.sph_plaw);
        }
        const Real efac = (T0 > 0.) ? std::exp(-escale * xcyl[2] * xcyl[2] / T0) : 1.;
        const Real beta = beta_min + beta0 * efac;
        const Real omdt = dt * std::sqrt(gm * ir1 * ir1 * ir1);
        for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {

          // Get the specific internal energy
          Real &etot = vmesh(b, gas::cons::total_energy(n), k, j, i);
          Real &eint = vmesh(b, gas::cons::internal_energy(n), k, j, i);

          const Real sie = ArtemisUtils::GetSpecificInternalEnergy<GEOM>(
              vmesh, b, n, k, j, i, dflr_gas, sieflr_gas, de_switch);

          // Compute the energy change from the temperature change.
          const Real &dens = vmesh(b, gas::cons::density(n), k, j, i);
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
template TaskStatus
BetaCooling<Coordinates::cartesian, TempRefType::nbody>(MeshData<Real> *md,
                                                        const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::cylindrical, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus
BetaCooling<Coordinates::cylindrical, TempRefType::nbody>(MeshData<Real> *md,
                                                          const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::spherical3D, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus
BetaCooling<Coordinates::spherical3D, TempRefType::nbody>(MeshData<Real> *md,
                                                          const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::spherical1D, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus
BetaCooling<Coordinates::spherical1D, TempRefType::nbody>(MeshData<Real> *md,
                                                          const Real time, const Real dt);

template TaskStatus BetaCooling<Coordinates::spherical2D, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus
BetaCooling<Coordinates::spherical2D, TempRefType::nbody>(MeshData<Real> *md,
                                                          const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::axisymmetric, TempRefType::powerlaw>(
    MeshData<Real> *md, const Real time, const Real dt);
template TaskStatus BetaCooling<Coordinates::axisymmetric, TempRefType::nbody>(
    MeshData<Real> *md, const Real time, const Real dt);

} // namespace Cooling
} // namespace Gas
