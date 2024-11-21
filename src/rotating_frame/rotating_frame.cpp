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
#include "rotating_frame/rotating_frame.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

using ArtemisUtils::VI;

namespace RotatingFrame {

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RotatingFrame::Initialize
//! \brief Adds intialization function for rotating frame package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("rotating_frame");
  Params &params = pkg->AllParams();

  params.Add("Omega", pin->GetOrAddReal("rotating_frame", "omega", 0.0));
  params.Add("qshear", pin->GetOrAddReal("rotating_frame", "qshear", 0.0));

  const int ShBoxCoord = pin->GetOrAddInteger("rotating_frame", "shboxcoord", 1);
  const bool StratFlag = pin->GetOrAddBoolean("rotating_frame", "stratified_flag", true);
  params.Add("ShBoxCoord", ShBoxCoord);
  params.Add("StratFlag", StratFlag);

  if (pin->GetOrAddBoolean("physics", "dust", false)) {
    const Real Kai0 = pin->GetOrAddReal("dust", "Kai0", 0.0);
    params.Add("Kai0", Kai0);
  }

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RotatingFrame::RotatingFrameForce
//! \brief Calculate the rotating frame body forces
//!            dv/dt = -2 Omega x v - Omega x Omega x r
template <Coordinates GEOM>
TaskStatus RotatingFrameForce(MeshData<Real> *md, const Real time, const Real dt) {
  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "Rotating frame is only implemented for Cartesian at the moment!");

  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  auto &rframe_pkg = pm->packages.Get("rotating_frame");
  const Real om0 = rframe_pkg->template Param<Real>("Omega");
  const Real qshear = rframe_pkg->template Param<Real>("qshear");
  const Real omsq = SQR(om0);
  const int ShBboxCoord = rframe_pkg->template Param<int>("ShBoxCoord");
  const bool StratFlag = rframe_pkg->template Param<bool>("StratFlag");

  Real Kai0 = 0.0;
  if (do_dust) {
    Kai0 = rframe_pkg->template Param<Real>("Kai0");
  }

  static auto desc =
      MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy,
                         dust::cons::momentum, gas::prim::density, gas::prim::velocity,
                         dust::prim::density, dust::prim::velocity>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;
  const bool three_d = (ndim == 3 && StratFlag);

  if (ShBboxCoord == 1) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "RotatingFrame", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          // Extract coordinates
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);

          const Real dx = coords.bnds.x1[1] - coords.bnds.x1[0];
          const Real dz = coords.bnds.x3[1] - coords.bnds.x3[0];
          const Real phi_xm1 = -qshear * omsq * coords.bnds.x1[0] * coords.bnds.x1[0];
          const Real phi_xp1 = -qshear * omsq * coords.bnds.x1[1] * coords.bnds.x1[1];
          const Real phi_zm1 = 0.5 * omsq * coords.bnds.x3[0] * coords.bnds.x3[0];
          const Real phi_zp1 = 0.5 * omsq * coords.bnds.x3[1] * coords.bnds.x3[1];
          const Real dpx = (phi_xp1 - phi_xm1) / dx;
          const Real dpz = three_d * ((phi_zp1 - phi_zm1) / dz);

          if (do_gas) {
            for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
              const Real &dens = vmesh(b, gas::prim::density(n), k, j, i);
              const Real &v1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
              const Real &v2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
              const Real &v3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
              Real &cmom1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i);
              Real &cmom2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i);
              Real &cmom3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i);
              Real &ctote = vmesh(b, gas::cons::total_energy(n), k, j, i);

              const Real rdt = dens * dt;
              cmom1 -= rdt * (dpx - 2.0 * om0 * v2);
              cmom2 -= rdt * 2.0 * om0 * v1;
              cmom3 -= rdt * dpz;
              ctote -= rdt * (v1 * dpx + v3 * dpz); // NOTE(ADM): Change to use the fluxes
            }
          }

          if (do_dust) {
            for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
              const Real &dens = vmesh(b, dust::prim::density(n), k, j, i);
              const Real &v1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
              const Real &v2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
              const Real &v3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);
              Real &cmom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
              Real &cmom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
              Real &cmom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);

              const Real rdt = dens * dt;
              cmom1 -= rdt * (dpx - 2.0 * om0 * v2);
              cmom2 -= rdt * 2.0 * om0 * v1;
              cmom3 -= rdt * dpz;
            }
          }
        });
  } else if (ShBboxCoord == 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "RotatingFrame", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          // Extract coordinates
          geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);

          const Real dx = coords.bnds.x1[1] - coords.bnds.x1[0];
          const Real dz = coords.bnds.x2[1] - coords.bnds.x2[0];
          const Real phi_xm1 = -qshear * omsq * coords.bnds.x1[0] * coords.bnds.x1[0];
          const Real phi_xp1 = -qshear * omsq * coords.bnds.x1[1] * coords.bnds.x1[1];
          const Real phi_zm1 = 0.5 * omsq * coords.bnds.x2[0] * coords.bnds.x2[0];
          const Real phi_zp1 = 0.5 * omsq * coords.bnds.x2[1] * coords.bnds.x2[1];
          const Real dpx = (phi_xp1 - phi_xm1) / dx;
          const Real dpz = StratFlag * ((phi_zp1 - phi_zm1) / dz);

          if (do_gas) {
            for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
              const Real &dens = vmesh(b, gas::prim::density(n), k, j, i);
              const Real &v1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
              const Real &v2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
              const Real &v3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
              Real &cmom1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i);
              Real &cmom2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i);
              Real &cmom3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i);
              Real &ctote = vmesh(b, gas::cons::total_energy(n), k, j, i);

              const Real rdt = dens * dt;
              cmom1 -= rdt * (dpx - 2.0 * om0 * v2);
              cmom2 -= rdt * dpz;
              cmom3 -= rdt * 2.0 * om0 * v1;
              ctote -= rdt * (v1 * dpx + v2 * dpz); // NOTE(ADM): Change to use the fluxes
            }
          }

          if (do_dust) {
            if (Kai0 > 0.0) {
              // add artificial pressure gradient to gas
              for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
                const Real dens_g = vmesh(b, gas::prim::density(n), k, j, i);
                const Real v1_g = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
                const Real press_gra = dens_g * Kai0 * om0 * dt;
                vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += press_gra;
                vmesh(b, gas::cons::total_energy(n), k, j, i) += (press_gra * v1_g);
              }
            }

            for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
              const Real &dens = vmesh(b, dust::prim::density(n), k, j, i);
              const Real &v1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
              const Real &v2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
              const Real &v3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);
              Real &cmom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
              Real &cmom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
              Real &cmom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);

              const Real rdt = dens * dt;
              cmom1 -= rdt * (dpx - 2.0 * om0 * v2);
              cmom2 -= rdt * dpz;
              cmom3 -= rdt * 2.0 * om0 * v1;
            }
          }
        });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> M;
template TaskStatus RotatingFrameForce<C::cartesian>(M *m, const Real t, const Real d);
template TaskStatus RotatingFrameForce<C::cylindrical>(M *m, const Real t, const Real d);
template TaskStatus RotatingFrameForce<C::spherical>(M *m, const Real t, const Real d);
template TaskStatus RotatingFrameForce<C::axisymmetric>(M *m, const Real t, const Real d);

} // namespace RotatingFrame
