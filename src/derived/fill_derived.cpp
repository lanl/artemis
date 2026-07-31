//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#include "mhd/mhd.hpp"
#include "radiation/moments/moments.hpp"
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
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Return immediately if not evolving gas hydrodynamics
  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  if (!(do_gas)) return TaskStatus::complete;
  const Real mu0_code =
      do_mhd ? pm->packages.Get("mhd")->template Param<Real>("mu0_code") : 1.0;

  // Extract gas parameters
  const Real dflr_gas = pm->packages.Get("gas").get()->template Param<Real>("dfloor");
  const Real sieflr_gas = pm->packages.Get("gas").get()->template Param<Real>("siefloor");
  const Real de_switch = pm->packages.Get("gas").get()->template Param<Real>("de_switch");

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy, field::face::B>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;
  const int multid = ndim >= 2;
  const int threed = ndim == 3;

  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  // Apply dual energy formalism to sync internal energy and total energy
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetAuxillaryFields", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract geometry
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);

        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);
        Real emag = 0.0;
        if (do_mhd) {
          const auto xv = coords.GetCellCenter(vg, b, k, j, i);
          const auto &bnds = coords.GetBounds();
          const Real bx =
              ((bnds.x1[1] - xv[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i) +
               (xv[0] - bnds.x1[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i + 1)) /
              (bnds.x1[1] - bnds.x1[0]);
          const Real by =
              multid
                  ? (((bnds.x2[1] - xv[1]) * vmesh(b, TE::F2, field::face::B(), k, j, i) +
                      (xv[1] - bnds.x2[0]) *
                          vmesh(b, TE::F2, field::face::B(), k, j + multid, i)) /
                     (bnds.x2[1] - bnds.x2[0]))
                  : vmesh(b, TE::F2, field::face::B(), k, j, i);
          const Real bz =
              threed
                  ? (((bnds.x3[1] - xv[2]) * vmesh(b, TE::F3, field::face::B(), k, j, i) +
                      (xv[2] - bnds.x3[0]) *
                          vmesh(b, TE::F3, field::face::B(), k + threed, j, i)) /
                     (bnds.x3[1] - bnds.x3[0]))
                  : vmesh(b, TE::F3, field::face::B(), k, j, i);
          emag = MHD::MagneticEnergyDensity(bx, by, bz, mu0_code);
        }

        for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {
          // Extract state vector
          Real &u_d = vmesh(b, gas::cons::density(n), k, j, i);
          Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);

          // Apply density floor
          const bool dfloor = (u_d > dflr_gas);
          u_d = (dfloor)*u_d + (!dfloor) * dflr_gas;

          // Compute SIE via dual energy formalism and apply floor
          const Real species_emag = (do_mhd && (n == 0)) ? emag : 0.0;
          Real sie = ArtemisUtils::DualEnergySIE(vmesh, b, n, k, j, i, de_switch, hx,
                                                 species_emag);
          const Real efloor = (sie > sieflr_gas);
          sie = (efloor)*sie + (!efloor) * sieflr_gas;

          // Return internal energy
          u_u = u_d * sie;
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
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract artemis parameters
  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const bool do_rad = artemis_pkg->template Param<bool>("do_moment");
  const bool do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  const Real mu0_code =
      do_mhd ? pm->packages.Get("mhd")->template Param<Real>("mu0_code") : 1.0;

  // Extract gas parameters
  Real dflr_gas = Null<Real>(), sieflr_gas = Null<Real>();
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

  // Extract radiation parameters
  Real eflr_rad = Null<Real>(), c = Null<Real>();
  if (do_rad) {
    auto &rad_pkg = pm->packages.Get("moments");
    eflr_rad = rad_pkg->template Param<Real>("efloor");
    c = rad_pkg->template Param<Real>("c");
  }
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  // Packing and indexing
  static auto desc = MakePackDescriptor<
      gas::cons::density, gas::cons::momentum, gas::cons::internal_energy,
      gas::prim::density, gas::prim::velocity, gas::prim::sie, dust::cons::density,
      dust::cons::momentum, dust::prim::density, dust::prim::velocity, rad::cons::energy,
      rad::cons::flux, rad::prim::energy, rad::prim::flux, field::face::B, field::cell::B,
      field::cell::energy, field::cell::divB>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v, geom::hx2v,
                         geom::hx3v, geom::vol, geom::ax1, geom::ax2, geom::ax3>(
          resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);
  const int nblocks = md->NumBlocks();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);

  const int ndim = pm->ndim;
  const int multid = ndim >= 2;
  const int threed = ndim == 3;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConsToPrim", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter(vg, b, k, j, i);
        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);

        if (do_gas) {
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            // Set primitive density and apply floor
            const Real u_d = vmesh(b, TE::CC, gas::cons::density(n), k, j, i);
            const bool dfloor = (u_d > dflr_gas);
            Real &w_d = vmesh(b, TE::CC, gas::prim::density(n), k, j, i);
            w_d = (dfloor)*u_d + (!dfloor) * dflr_gas;

            // Set primitive velocity
            Real &vel1 = vmesh(b, TE::CC, gas::prim::velocity(VI(n, 0)), k, j, i);
            Real &vel2 = vmesh(b, TE::CC, gas::prim::velocity(VI(n, 1)), k, j, i);
            Real &vel3 = vmesh(b, TE::CC, gas::prim::velocity(VI(n, 2)), k, j, i);
            vel1 =
                vmesh(b, TE::CC, gas::cons::momentum(VI(n, 0)), k, j, i) / (w_d * hx[0]);
            vel2 =
                vmesh(b, TE::CC, gas::cons::momentum(VI(n, 1)), k, j, i) / (w_d * hx[1]);
            vel3 =
                vmesh(b, TE::CC, gas::cons::momentum(VI(n, 2)), k, j, i) / (w_d * hx[2]);

            // Set primitive specific internal energy and apply floor
            const Real w_s =
                vmesh(b, TE::CC, gas::cons::internal_energy(n), k, j, i) / w_d;
            const bool siefloor = (w_s > sieflr_gas);
            vmesh(b, gas::prim::sie(n), k, j, i) =
                (siefloor)*w_s + (!siefloor) * sieflr_gas;
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Set primitive density
            const Real u_d = vmesh(b, TE::CC, dust::cons::density(n), k, j, i);
            const bool dfloor = (u_d > dflr_dust);
            Real &w_d = vmesh(b, TE::CC, dust::prim::density(n), k, j, i);
            w_d = (dfloor)*u_d + (u_d <= dflr_dust) * dflr_dust;

            // Set primitive velocity
            Real &vel1 = vmesh(b, TE::CC, dust::prim::velocity(VI(n, 0)), k, j, i);
            Real &vel2 = vmesh(b, TE::CC, dust::prim::velocity(VI(n, 1)), k, j, i);
            Real &vel3 = vmesh(b, TE::CC, dust::prim::velocity(VI(n, 2)), k, j, i);
            vel1 =
                vmesh(b, TE::CC, dust::cons::momentum(VI(n, 0)), k, j, i) / (w_d * hx[0]);
            vel2 =
                vmesh(b, TE::CC, dust::cons::momentum(VI(n, 1)), k, j, i) / (w_d * hx[1]);
            vel3 =
                vmesh(b, TE::CC, dust::cons::momentum(VI(n, 2)), k, j, i) / (w_d * hx[2]);
          }
        }

        if (do_rad) {
          for (int n = 0; n < vmesh.GetSize(b, rad::prim::energy()); ++n) {
            // Set primitive radiation energy density
            const Real u_er = vmesh(b, rad::cons::energy(n), k, j, i);
            const bool efloor = (u_er > eflr_rad);
            Real &w_er = vmesh(b, rad::prim::energy(n), k, j, i);
            w_er = (efloor)*u_er + (!efloor) * eflr_rad;

            // Set primitive radiation flux
            const Real cer = c * w_er;
            const std::array<Real, 3> conv = {cer * hx[0], cer * hx[1], cer * hx[2]};
            const Real hfx1 =
                vmesh(b, TE::CC, rad::cons::flux(VI(n, 0)), k, j, i) / conv[0];
            const Real hfx2 =
                vmesh(b, TE::CC, rad::cons::flux(VI(n, 1)), k, j, i) / conv[1];
            const Real hfx3 =
                vmesh(b, TE::CC, rad::cons::flux(VI(n, 2)), k, j, i) / conv[2];
            const auto fx = Moments::NormalizeFlux(hfx1, hfx2, hfx3);
            vmesh(b, TE::CC, rad::cons::flux(VI(n, 0)), k, j, i) = fx[0] * conv[0];
            vmesh(b, TE::CC, rad::cons::flux(VI(n, 1)), k, j, i) = fx[1] * conv[1];
            vmesh(b, TE::CC, rad::cons::flux(VI(n, 2)), k, j, i) = fx[2] * conv[2];
            vmesh(b, TE::CC, rad::prim::flux(VI(n, 0)), k, j, i) = fx[0];
            vmesh(b, TE::CC, rad::prim::flux(VI(n, 1)), k, j, i) = fx[1];
            vmesh(b, TE::CC, rad::prim::flux(VI(n, 2)), k, j, i) = fx[2];
          }
        }
        if (do_mhd) {
          const Real vol = coords.GetVolume(vg, b, k, j, i);
          const auto ax1 = coords.GetFaceAreaX1(vg, b, k, j, i);
          const auto ax2 = coords.GetFaceAreaX2(vg, b, k, j, i);
          const auto ax3 = coords.GetFaceAreaX3(vg, b, k, j, i);
          const auto xv = coords.GetCellCenter(vg, b, k, j, i);
          const auto &bnds = coords.GetBounds();
          vmesh(b, TE::CC, field::cell::divB(), k, j, i) =
              ((ax1[1] * vmesh(b, TE::F1, field::face::B(), k, j, i + 1) -
                ax1[0] * vmesh(b, TE::F1, field::face::B(), k, j, i)) +
               (ax2[1] * vmesh(b, TE::F2, field::face::B(), k, j + multid, i) -
                ax2[0] * vmesh(b, TE::F2, field::face::B(), k, j, i)) +
               (ax3[1] * vmesh(b, TE::F3, field::face::B(), k + threed, j, i) -
                ax3[0] * vmesh(b, TE::F3, field::face::B(), k, j, i))) /
              vol;
          vmesh(b, TE::CC, field::cell::B(0), k, j, i) =
              ((bnds.x1[1] - xv[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i) +
               (xv[0] - bnds.x1[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i + 1)) /
              (bnds.x1[1] - bnds.x1[0]);
          vmesh(b, TE::CC, field::cell::B(1), k, j, i) =
              multid
                  ? (((bnds.x2[1] - xv[1]) * vmesh(b, TE::F2, field::face::B(), k, j, i) +
                      (xv[1] - bnds.x2[0]) *
                          vmesh(b, TE::F2, field::face::B(), k, j + multid, i)) /
                     (bnds.x2[1] - bnds.x2[0]))
                  : vmesh(b, TE::F2, field::face::B(), k, j, i);
          vmesh(b, TE::CC, field::cell::B(2), k, j, i) =
              threed
                  ? (((bnds.x3[1] - xv[2]) * vmesh(b, TE::F3, field::face::B(), k, j, i) +
                      (xv[2] - bnds.x3[0]) *
                          vmesh(b, TE::F3, field::face::B(), k + threed, j, i)) /
                     (bnds.x3[1] - bnds.x3[0]))
                  : vmesh(b, TE::F3, field::face::B(), k, j, i);
          vmesh(b, TE::CC, field::cell::energy(), k, j, i) = MHD::MagneticEnergyDensity(
              vmesh(b, TE::CC, field::cell::B(0), k, j, i),
              vmesh(b, TE::CC, field::cell::B(1), k, j, i),
              vmesh(b, TE::CC, field::cell::B(2), k, j, i), mu0_code);
        }
      });

  if (do_mhd) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ConsToPrim::MHDGhost", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
          const Real vol = coords.GetVolume(vg, b, k, j, i);
          const auto ax1 = coords.GetFaceAreaX1(vg, b, k, j, i);
          const auto ax2 = coords.GetFaceAreaX2(vg, b, k, j, i);
          const auto ax3 = coords.GetFaceAreaX3(vg, b, k, j, i);
          const auto xv = coords.GetCellCenter(vg, b, k, j, i);
          const auto &bnds = coords.GetBounds();
          vmesh(b, TE::CC, field::cell::divB(), k, j, i) =
              ((ax1[1] * vmesh(b, TE::F1, field::face::B(), k, j, i + 1) -
                ax1[0] * vmesh(b, TE::F1, field::face::B(), k, j, i)) +
               (ax2[1] * vmesh(b, TE::F2, field::face::B(), k, j + multid, i) -
                ax2[0] * vmesh(b, TE::F2, field::face::B(), k, j, i)) +
               (ax3[1] * vmesh(b, TE::F3, field::face::B(), k + threed, j, i) -
                ax3[0] * vmesh(b, TE::F3, field::face::B(), k, j, i))) /
              vol;
          vmesh(b, TE::CC, field::cell::B(0), k, j, i) =
              ((bnds.x1[1] - xv[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i) +
               (xv[0] - bnds.x1[0]) * vmesh(b, TE::F1, field::face::B(), k, j, i + 1)) /
              (bnds.x1[1] - bnds.x1[0] + Fuzz<Real>());
          vmesh(b, TE::CC, field::cell::B(1), k, j, i) =
              multid
                  ? (((bnds.x2[1] - xv[1]) * vmesh(b, TE::F2, field::face::B(), k, j, i) +
                      (xv[1] - bnds.x2[0]) *
                          vmesh(b, TE::F2, field::face::B(), k, j + multid, i)) /
                     (bnds.x2[1] - bnds.x2[0] + Fuzz<Real>()))
                  : vmesh(b, TE::F2, field::face::B(), k, j, i);
          vmesh(b, TE::CC, field::cell::B(2), k, j, i) =
              threed
                  ? (((bnds.x3[1] - xv[2]) * vmesh(b, TE::F3, field::face::B(), k, j, i) +
                      (xv[2] - bnds.x3[0]) *
                          vmesh(b, TE::F3, field::face::B(), k + threed, j, i)) /
                     (bnds.x3[1] - bnds.x3[0] + Fuzz<Real>()))
                  : vmesh(b, TE::F3, field::face::B(), k, j, i);
          vmesh(b, TE::CC, field::cell::energy(), k, j, i) = MHD::MagneticEnergyDensity(
              vmesh(b, TE::CC, field::cell::B(0), k, j, i),
              vmesh(b, TE::CC, field::cell::B(1), k, j, i),
              vmesh(b, TE::CC, field::cell::B(2), k, j, i), mu0_code);
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisDerived::PrimToCons(MeshData<Real> *md)
//! \brief Executes P2C following integrator updates and/or remeshing events
template <typename T, Coordinates GEOM>
void PrimToCons(T *md) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract artemis parameters
  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  const bool do_rad = artemis_pkg->template Param<bool>("do_moment");
  const Real mu0_code =
      do_mhd ? pm->packages.Get("mhd")->template Param<Real>("mu0_code") : 1.0;

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

  // Extract radiation parameters
  Real eflr_rad = Null<Real>();
  Real c = Null<Real>();
  if (do_rad) {
    auto &rad_pkg = pm->packages.Get("moments");
    eflr_rad = rad_pkg->template Param<Real>("efloor");
    c = rad_pkg->template Param<Real>("c");
  }
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy, gas::prim::density,
                         gas::prim::velocity, gas::prim::pressure, gas::prim::sie,
                         gas::prim::bmod, gas::prim::temperature, dust::cons::density,
                         dust::cons::momentum, dust::prim::density, dust::prim::velocity,
                         rad::cons::energy, rad::cons::flux, rad::prim::energy,
                         rad::prim::flux, rad::prim::pressure, field::face::B>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);
  const int ndim = md->GetMeshPointer()->ndim;
  const int multid = ndim >= 2;
  const int threed = ndim == 3;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrimToCons", parthenon::DevExecSpace(), 0,
      vmesh.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter(vg, b, k, j, i);
        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);

        if (do_gas) {
          Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            // Sync conserved and primitive density
            Real &w_d = vmesh(b, gas::prim::density(n), k, j, i);
            Real &u_d = vmesh(b, gas::cons::density(n), k, j, i);
            const bool dfloor = (w_d > dflr_gas);
            w_d = (dfloor)*w_d + (!dfloor) * dflr_gas;
            u_d = w_d;

            // Sync conserved momenta and primitive velocity
            const Real &vel1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            const Real &vel2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            const Real &vel3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            Real &mom1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i);
            mom1 = w_d * vel1 * hx[0];
            mom2 = w_d * vel2 * hx[1];
            mom3 = w_d * vel3 * hx[2];

            // Sync primtive sie, pressure, and conserved internal energy
            Real &w_s = vmesh(b, gas::prim::sie(n), k, j, i);
            Real &w_p = vmesh(b, gas::prim::pressure(n), k, j, i);
            Real &w_b = vmesh(b, gas::prim::bmod(n), k, j, i);
            Real &w_t = vmesh(b, gas::prim::temperature(n), k, j, i);
            Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);
            const bool siefloor = (w_s > sieflr_gas);
            w_s = (siefloor)*w_s + (!siefloor) * sieflr_gas;
            u_u = w_s * u_d;
            w_p = eos_d.PressureFromDensityInternalEnergy(w_d, w_s, lambda);
            w_b = eos_d.BulkModulusFromDensityInternalEnergy(w_d, w_s, lambda);
            w_t = eos_d.TemperatureFromDensityInternalEnergy(w_d, w_s, lambda);

            // Sync conserved total energy
            const Real ke = 0.5 * w_d * (SQR(vel1) + SQR(vel2) + SQR(vel3));
            Real me = 0.0;
            if (do_mhd && (n == 0)) {
              const Real bx = 0.5 * (vmesh(b, TE::F1, field::face::B(), k, j, i) +
                                     vmesh(b, TE::F1, field::face::B(), k, j, i + 1));
              const Real by =
                  0.5 * (vmesh(b, TE::F2, field::face::B(), k, j, i) +
                         vmesh(b, TE::F2, field::face::B(), k, j + multid, i));
              const Real bz =
                  0.5 * (vmesh(b, TE::F3, field::face::B(), k, j, i) +
                         vmesh(b, TE::F3, field::face::B(), k + threed, j, i));
              me = MHD::MagneticEnergyDensity(bx, by, bz, mu0_code);
            }
            Real &u_e = vmesh(b, gas::cons::total_energy(n), k, j, i);
            u_e = u_u + ke + me;
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Sync conserved and primitive density
            Real &w_d = vmesh(b, dust::prim::density(n), k, j, i);
            Real &u_d = vmesh(b, dust::cons::density(n), k, j, i);
            const bool dfloor = (w_d > dflr_dust);
            w_d = (dfloor)*w_d + (!dfloor) * dflr_dust;
            u_d = w_d;

            // Sync conserved momenta and primitive velocity
            const Real &vel1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
            const Real &vel2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
            const Real &vel3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);
            Real &mom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);
            mom1 = w_d * vel1 * hx[0];
            mom2 = w_d * vel2 * hx[1];
            mom3 = w_d * vel3 * hx[2];
          }
        }

        if (do_rad) {
          for (int n = 0; n < vmesh.GetSize(b, rad::prim::energy()); ++n) {
            // Energy Density
            Real &w_er = vmesh(b, rad::prim::energy(n), k, j, i);
            Real &u_er = vmesh(b, rad::cons::energy(n), k, j, i);
            const bool efloor = (w_er > eflr_rad);
            w_er = (efloor)*w_er + (!efloor) * eflr_rad;
            u_er = w_er;

            // Sync radiation fluxes
            const Real cer = c * w_er;
            const std::array<Real, 3> conv = {cer * hx[0], cer * hx[1], cer * hx[2]};
            const Real fx1 = vmesh(b, rad::prim::flux(VI(n, 0)), k, j, i);
            const Real fx2 = vmesh(b, rad::prim::flux(VI(n, 1)), k, j, i);
            const Real fx3 = vmesh(b, rad::prim::flux(VI(n, 2)), k, j, i);
            const auto fx = Moments::NormalizeFlux(fx1, fx2, fx3);
            vmesh(b, rad::cons::flux(VI(n, 0)), k, j, i) = fx[0] * conv[0];
            vmesh(b, rad::cons::flux(VI(n, 1)), k, j, i) = fx[1] * conv[1];
            vmesh(b, rad::cons::flux(VI(n, 2)), k, j, i) = fx[2] * conv[2];
            vmesh(b, rad::prim::flux(VI(n, 0)), k, j, i) = fx[0];
            vmesh(b, rad::prim::flux(VI(n, 1)), k, j, i) = fx[1];
            vmesh(b, rad::prim::flux(VI(n, 2)), k, j, i) = fx[2];

            // Radiation pressure
            Real &w_p = vmesh(b, rad::prim::pressure(n), k, j, i);
            w_p = ONE_3RD * w_er;
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
  PARTHENON_INSTRUMENT
  auto &md = pmb->meshblock_data.Get();
  MHD::SetCellCenteredMagneticFields<MeshBlockData<Real>, GEOM>(md.get());
  PrimToCons<MeshBlockData<Real>, GEOM>(md.get());
}

//----------------------------------------------------------------------------------------
//! \fn TaskCollection ArtemisDerived::SyncFields
//! \brief Syncs unsplit fields following an operator split update
template <Coordinates GEOM>
TaskCollection SyncFields(Mesh *pmesh, const Real time, const Real dt) {
  PARTHENON_INSTRUMENT
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &post_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = post_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
    auto c2p = tl.AddTask(start_recv, PreCommFillDerived<MeshData<Real>>, u0.get());
    auto bcs = parthenon::AddBoundaryExchangeTasks(c2p, tl, u0, pmesh->multilevel);
    auto p2c = tl.AddTask(bcs, FillDerived<MeshData<Real>>, u0.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef MeshBlock MB;
typedef MeshData<Real> MD;
typedef MeshBlockData<Real> MBD;
typedef ParameterInput PI;
template void ConsToPrim<G::cartesian>(MD *md);
template void ConsToPrim<G::cylindrical>(MD *md);
template void ConsToPrim<G::spherical1D>(MD *md);
template void ConsToPrim<G::spherical2D>(MD *md);
template void ConsToPrim<G::spherical3D>(MD *md);
template void ConsToPrim<G::axisymmetric>(MD *md);
template void PrimToCons<MBD, G::cartesian>(MBD *mbd);
template void PrimToCons<MBD, G::cylindrical>(MBD *mbd);
template void PrimToCons<MBD, G::spherical1D>(MBD *mbd);
template void PrimToCons<MBD, G::spherical2D>(MBD *mbd);
template void PrimToCons<MBD, G::spherical3D>(MBD *mbd);
template void PrimToCons<MBD, G::axisymmetric>(MBD *mbd);
template void PrimToCons<MD, G::cartesian>(MD *md);
template void PrimToCons<MD, G::cylindrical>(MD *md);
template void PrimToCons<MD, G::spherical1D>(MD *md);
template void PrimToCons<MD, G::spherical2D>(MD *md);
template void PrimToCons<MD, G::spherical3D>(MD *md);
template void PrimToCons<MD, G::axisymmetric>(MD *md);
template void PostInitialization<G::cartesian>(MB *pmb, PI *pin);
template void PostInitialization<G::cylindrical>(MB *pmb, PI *pin);
template void PostInitialization<G::spherical1D>(MB *pmb, PI *pin);
template void PostInitialization<G::spherical2D>(MB *pmb, PI *pin);
template void PostInitialization<G::spherical3D>(MB *pmb, PI *pin);
template void PostInitialization<G::axisymmetric>(MB *pmb, PI *pin);
template TaskStatus SetAuxillaryFields<G::cartesian>(MD *md);
template TaskStatus SetAuxillaryFields<G::cylindrical>(MD *md);
template TaskStatus SetAuxillaryFields<G::spherical1D>(MD *md);
template TaskStatus SetAuxillaryFields<G::spherical2D>(MD *md);
template TaskStatus SetAuxillaryFields<G::spherical3D>(MD *md);
template TaskStatus SetAuxillaryFields<G::axisymmetric>(MD *md);
template TaskCollection SyncFields<G::cartesian>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<G::cylindrical>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<G::spherical1D>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<G::spherical2D>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<G::spherical3D>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<G::axisymmetric>(Mesh *m, const Real t, const Real dt);

} // namespace ArtemisDerived
