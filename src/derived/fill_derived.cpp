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
#include "fill_derived.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace ArtemisDerived {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus ArtemisDerived::SetAuxillaryFields(MeshData<Real> *md)
//! \brief Sets auxillary fields over IndexDomain::interior after an integration stage
//! NOTE(PDM): Note that this function is not called during remeshing.
template <Coordinates GEOM>
TaskStatus SetAuxillaryFields(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  if (!(do_gas)) return TaskStatus::complete;

  // Extract gas parameters
  const Real dflr_gas = pm->packages.Get("gas").get()->template Param<Real>("dfloor");
  const Real sieflr_gas = pm->packages.Get("gas").get()->template Param<Real>("siefloor");
  const Real de_switch = pm->packages.Get("gas").get()->template Param<Real>("de_switch");

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetAuxillaryFields", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {
          // Extract conserved density
          Real u_d = vmesh(b, gas::cons::density(n), k, j, i);
          u_d = (u_d > dflr_gas) ? u_d : dflr_gas;

          // Sync the internal energy with the total energy
          Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);
          u_u = ArtemisUtils::GetSpecificInternalEnergy<GEOM>(
                    vmesh, b, n, k, j, i, de_switch, dflr_gas, sieflr_gas) *
                u_d;

          // Apply internal energy floor
          const Real uflr_gas = sieflr_gas * u_d;
          u_u = (u_u > uflr_gas) ? u_u : uflr_gas;
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisDerived::ConsToPrim(MeshData<Real> *md)
//! \brief Exectues C2P over IndexDomain::interior after an integration stage
//! or remeshing event in preparation for FillGhost
template <Coordinates GEOM>
void ConsToPrim(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  // Extract gas parameters
  Real dflr_gas = Null<Real>();
  Real sieflr_gas = Null<Real>();
  if (do_gas) {
    auto &gas_pkg = pm->packages.Get("gas");
    dflr_gas = gas_pkg->template Param<Real>("dfloor");
    sieflr_gas = gas_pkg->template Param<Real>("siefloor");
  }

  // Extract dust parameters
  Real dflr_dust = Null<Real>();
  if (do_dust) {
    dflr_dust = pm->packages.Get("dust").get()->template Param<Real>("dfloor");
  }

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum,
                         gas::cons::internal_energy, gas::prim::density,
                         gas::prim::velocity, gas::prim::sie, dust::cons::density,
                         dust::cons::momentum, dust::prim::density, dust::prim::velocity>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const int nblocks = md->NumBlocks();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConsToPrim", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &hx = coords.GetScaleFactors();

        if (do_gas) {
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            // Set primitive density
            const Real u_d = vmesh(b, gas::cons::density(n), k, j, i);
            Real &w_d = vmesh(b, gas::prim::density(n), k, j, i);
            w_d = (u_d > dflr_gas) ? u_d : dflr_gas;

            // Set primitive velocity
            Real &vel1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            Real &vel2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            Real &vel3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            vel1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) / (w_d * hx[0]);
            vel2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) / (w_d * hx[1]);
            vel3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) / (w_d * hx[2]);

            // Set primitive specific internal energy
            const Real w_s = vmesh(b, gas::cons::internal_energy(n), k, j, i) / w_d;
            vmesh(b, gas::prim::sie(n), k, j, i) = (w_s > sieflr_gas) ? w_s : sieflr_gas;
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Set primitive density
            const Real u_d = vmesh(b, dust::cons::density(n), k, j, i);
            Real &w_d = vmesh(b, dust::prim::density(n), k, j, i);
            w_d = (u_d > dflr_dust) ? u_d : dflr_dust;

            // set primitive velocity
            Real &vel1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
            Real &vel2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
            Real &vel3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);
            vel1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) / (w_d * hx[0]);
            vel2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) / (w_d * hx[1]);
            vel3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) / (w_d * hx[2]);
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisDerived::PrimToCons(MeshData<Real> *md)
//! \brief Executes P2C following integrator updates and/or remeshing events
template <typename T, Coordinates GEOM>
void PrimToCons(T *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  // Extract gas parameters
  Real dflr_gas = Null<Real>();
  Real sieflr_gas = Null<Real>();
  EOS eos_d;
  if (do_gas) {
    auto &gas_pkg = pm->packages.Get("gas");
    dflr_gas = gas_pkg->template Param<Real>("dfloor");
    sieflr_gas = gas_pkg->template Param<Real>("siefloor");
    eos_d = gas_pkg->template Param<EOS>("eos_d");
  }

  // Extract dust parameters
  Real dflr_dust = Null<Real>();
  if (do_dust) {
    dflr_dust = pm->packages.Get("dust").get()->template Param<Real>("dfloor");
  }

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy, gas::prim::density,
                         gas::prim::velocity, gas::prim::pressure, gas::prim::sie,
                         dust::cons::density, dust::cons::momentum, dust::prim::density,
                         dust::prim::velocity>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrimToCons", parthenon::DevExecSpace(), 0,
      vmesh.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &hx = coords.GetScaleFactors();

        if (do_gas) {
          Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            // Sync conserved and primitive density
            Real &w_d = vmesh(b, gas::prim::density(n), k, j, i);
            Real &u_d = vmesh(b, gas::cons::density(n), k, j, i);
            w_d = (w_d > dflr_gas) ? w_d : dflr_gas;
            u_d = w_d;

            // Sync conserved momenta and primitive velocity
            const Real vel1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            const Real vel2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            const Real vel3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            Real &mom1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i);
            mom1 = w_d * vel1 * hx[0];
            mom2 = w_d * vel2 * hx[1];
            mom3 = w_d * vel3 * hx[2];

            // Sync primtive sie, pressure, and conserved internal energy
            Real &w_s = vmesh(b, gas::prim::sie(n), k, j, i);
            Real &w_p = vmesh(b, gas::prim::pressure(n), k, j, i);
            Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);
            w_s = (w_s > sieflr_gas) ? w_s : sieflr_gas;
            u_u = w_s * u_d;
            w_p = eos_d.PressureFromDensityInternalEnergy(w_d, w_s, lambda);

            // Sync conserved total energy
            const Real ke = 0.5 * w_d * (SQR(vel1) + SQR(vel2) + SQR(vel3));
            Real &u_e = vmesh(b, gas::cons::total_energy(n), k, j, i);
            u_e = u_u + ke;
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Sync conserved and primitive density
            Real &w_d = vmesh(b, dust::prim::density(n), k, j, i);
            Real &u_d = vmesh(b, dust::cons::density(n), k, j, i);
            w_d = (w_d > dflr_dust) ? w_d : dflr_dust;
            u_d = w_d;

            // Sync conserved momenta and primitive velocity
            const Real vel1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
            const Real vel2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
            const Real vel3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);
            Real &mom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);
            mom1 = w_d * vel1 * hx[0];
            mom2 = w_d * vel2 * hx[1];
            mom3 = w_d * vel3 * hx[2];
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ArtemisDerived::PostInitialization(MeshBlock *pmb, ParameterInput *pin)
//! \brief Post-initialization hook which calls PrimToCons after ProblemGenerator
//! but before PreCommFillDerived
template <Coordinates GEOM>
void PostInitialization(MeshBlock *pmb, ParameterInput *pin) {
  auto &md = pmb->meshblock_data.Get();
  PrimToCons<MeshBlockData<Real>, GEOM>(md.get());
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef MeshBlock MB;
typedef MeshData<Real> MD;
typedef MeshBlockData<Real> MBD;
typedef ParameterInput PI;
template void ConsToPrim<Coordinates::cartesian>(MD *md);
template void ConsToPrim<Coordinates::cylindrical>(MD *md);
template void ConsToPrim<Coordinates::spherical1D>(MD *md);
template void ConsToPrim<Coordinates::spherical2D>(MD *md);
template void ConsToPrim<Coordinates::spherical3D>(MD *md);
template void ConsToPrim<Coordinates::axisymmetric>(MD *md);
template void PrimToCons<MBD, Coordinates::cartesian>(MBD *mbd);
template void PrimToCons<MBD, Coordinates::cylindrical>(MBD *mbd);
template void PrimToCons<MBD, Coordinates::spherical1D>(MBD *mbd);
template void PrimToCons<MBD, Coordinates::spherical2D>(MBD *mbd);
template void PrimToCons<MBD, Coordinates::spherical3D>(MBD *mbd);
template void PrimToCons<MBD, Coordinates::axisymmetric>(MBD *mbd);
template void PrimToCons<MD, Coordinates::cartesian>(MD *md);
template void PrimToCons<MD, Coordinates::cylindrical>(MD *md);
template void PrimToCons<MD, Coordinates::spherical1D>(MD *md);
template void PrimToCons<MD, Coordinates::spherical2D>(MD *md);
template void PrimToCons<MD, Coordinates::spherical3D>(MD *md);
template void PrimToCons<MD, Coordinates::axisymmetric>(MD *md);
template void PostInitialization<Coordinates::cartesian>(MB *pmb, PI *pin);
template void PostInitialization<Coordinates::cylindrical>(MB *pmb, PI *pin);
template void PostInitialization<Coordinates::spherical1D>(MB *pmb, PI *pin);
template void PostInitialization<Coordinates::spherical2D>(MB *pmb, PI *pin);
template void PostInitialization<Coordinates::spherical3D>(MB *pmb, PI *pin);
template void PostInitialization<Coordinates::axisymmetric>(MB *pmb, PI *pin);
template TaskStatus SetAuxillaryFields<Coordinates::cartesian>(MD *md);
template TaskStatus SetAuxillaryFields<Coordinates::cylindrical>(MD *md);
template TaskStatus SetAuxillaryFields<Coordinates::spherical1D>(MD *md);
template TaskStatus SetAuxillaryFields<Coordinates::spherical2D>(MD *md);
template TaskStatus SetAuxillaryFields<Coordinates::spherical3D>(MD *md);
template TaskStatus SetAuxillaryFields<Coordinates::axisymmetric>(MD *md);

} // namespace ArtemisDerived
