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
#ifndef PGEN_DISK_HPP_
#define PGEN_DISK_HPP_
//! \file disk.hpp
//! \brief Initializes a stratified Keplerian accretion disk. Initial conditions are in
//! vertical hydrostatic equilibrium.

// NOTE(PDM): The following is adapted from the open-source Athena++ disk.cpp
// problem generator, and adapted for Parthenon/Artemis by PDM on 10/20/23

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
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace disk {
//----------------------------------------------------------------------------------------
//! \struct DiskParams
//! \brief container for disk parameters
struct DiskParams {
  Real r0, h0;
  Real p, q, flare;
  Real rho0, dens_min, pres_min;
  Real gm, Omega0, l0;
  Real dust_to_gas;
  Real rexp;
  Real rcav;
  Real Gamma, gamma_gas;
  Real alpha, nu0, nu_indx;
  Real mdot;
  bool do_dust;
  int quiet_start;
};

//----------------------------------------------------------------------------------------
//! \fn Real DenProfile
//! \brief Computes density profile at cylindrical R and z
KOKKOS_INLINE_FUNCTION
Real DenProfile(struct DiskParams pgen, const Real R, const Real z) {
  const Real r = std::sqrt(R * R + z * z);
  const Real h = pgen.h0 * std::pow(R / pgen.r0, pgen.flare);
  const Real sig0 = pgen.rho0; // / (std::sqrt(2.0 * M_PI) * pgen.h0 * pgen.r0);
  const Real exp_fac = (pgen.rexp == 0.) ? 1. : std::exp(-SQR(R / pgen.rexp));
  const Real dmid =
      (sig0 * std::pow(R / pgen.r0, pgen.p)) *
      (1. - pgen.l0 * std::sqrt(pgen.r0 / R)) * // correction for an inner binary
      (pgen.dens_min / pgen.rho0 +              // The inner cavity
       (1. - pgen.dens_min / pgen.rho0) * std::exp(-std::pow(pgen.rcav / R, 12.0))) *
      exp_fac;                                // the outer cutoff
  const Real sint = (r == 0.0) ? 1.0 : R / r; // TODO(ADM): should it be 1?
  const Real efac = (1. - sint) / (h * h);
  if (pgen.Gamma == 1.) return std::max(pgen.dens_min, dmid * std::exp(-efac));
  // sint <= h^2/(gamma-1), efac*(g-1) = 1 - eps
  const Real pfac = std::max(Fuzz<Real>(), 1. - (pgen.Gamma - 1) * efac);
  return std::max(pgen.dens_min, dmid * std::pow(pfac, 1. / (pgen.Gamma - 1)));
}

//----------------------------------------------------------------------------------------
//! \fn Real DenProfile
//! \brief Computes temperature profile at cylindrical R and z
KOKKOS_INLINE_FUNCTION
Real TempProfile(struct DiskParams pgen, const Real R, const Real z) {
  // P = K rho^Gamma
  // T = T0 (rho/rho0)^(Gamma-1)
  const Real rho = DenProfile(pgen, R, z);
  const Real rho0 = DenProfile(pgen, R, 0.0);
  const Real H = R * pgen.h0 * std::pow(R / pgen.r0, pgen.flare);
  const Real omk2 = SQR(pgen.Omega0) / (R * R * R);
  const Real T0 = omk2 * H * H / pgen.Gamma;
  return T0 * std::pow(rho / rho0, pgen.Gamma - 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real PresProfile
//! \brief Computes pressure profile at cylindrical R and z (via dens and temp profiles)
KOKKOS_INLINE_FUNCTION
Real PresProfile(struct DiskParams pgen, EOS eos, const Real R, const Real z) {
  const Real df = DenProfile(pgen, R, z);
  const Real tf = TempProfile(pgen, R, z);
  return std::max(pgen.pres_min, eos.PressureFromDensityTemperature(df, tf));
}

//----------------------------------------------------------------------------------------
//! \fn Real ViscosityProfile
//! \brief Computes viscosity profile at cylindrical R and z (via dens and temp profiles)
KOKKOS_INLINE_FUNCTION
Real ViscosityProfile(struct DiskParams pgen, EOS eos, const Real R, const Real z) {
  return pgen.nu0 * std::pow(R / pgen.r0, pgen.nu_indx);
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeDiskProfile
//! \brief Initialize vertical hydrostatic and radial centrifugal equilibrium disk profile
//! at a specified index/coordinate
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION void
ComputeDiskProfile(const struct DiskParams pgen, const parthenon::Coordinates_t &pco,
                   const int k, const int j, const int i, EOS eos_d, Real &gdens,
                   Real &gtemp, Real &gvel1, Real &gvel2, Real &gvel3, const bool do_dust,
                   Real &ddens, Real &dvel1, Real &dvel2, Real &dvel3) {
  // Extract coordinates
  geometry::Coords<GEOM> coords(pco, k, j, i);
  Real xv[3] = {Null<Real>()};
  coords.GetCellCenter(xv);

  Real xcyl[3] = {Null<Real>()};
  Real ex1[3] = {Null<Real>()};
  Real ex2[3] = {Null<Real>()};
  Real ex3[3] = {Null<Real>()};
  coords.ConvertToCylWithVec(xv, xcyl, ex1, ex2, ex3);

  // compute Keplerian solution
  gdens = DenProfile(pgen, xcyl[0], xcyl[2]);
  gtemp = TempProfile(pgen, xcyl[0], xcyl[2]);

  // Construct grad(P) for this zone
  const Real fx1m[3] = {coords.bnds.x1[0], xv[1], xv[2]};
  const Real fx1p[3] = {coords.bnds.x1[1], xv[1], xv[2]};
  const Real fx2m[3] = {xv[0], coords.bnds.x2[0], xv[2]};
  const Real fx2p[3] = {xv[0], coords.bnds.x2[1], xv[2]};
  const Real fx3m[3] = {xv[0], xv[1], coords.bnds.x3[0]};
  const Real fx3p[3] = {xv[0], xv[1], coords.bnds.x3[1]};
  Real pgrad[3] = {Null<Real>()};
  Real xf[3] = {Null<Real>()};
  Real pfm = Null<Real>(), pfp = Null<Real>();

  // X1 Faces
  coords.ConvertToCyl(fx1m, xf);
  pfm = PresProfile(pgen, eos_d, xf[0], xf[2]);
  coords.ConvertToCyl(fx1p, xf);
  pfp = (pfm = pgen.pres_min) ? pgen.pres_min : PresProfile(pgen, eos_d, xf[0], xf[2]);
  pfm = (pfp == pgen.pres_min) ? pgen.pres_min : pfm;
  pgrad[0] = (pfp - pfm) / coords.GetCellWidthX1();

  // X2 Faces
  coords.ConvertToCyl(fx2m, xf);
  pfm = PresProfile(pgen, eos_d, xf[0], xf[2]);
  coords.ConvertToCyl(fx2p, xf);
  pfp = (pfm = pgen.pres_min) ? pgen.pres_min : PresProfile(pgen, eos_d, xf[0], xf[2]);
  pfm = (pfp == pgen.pres_min) ? pgen.pres_min : pfm;
  pgrad[1] = (pfp - pfm) / coords.GetCellWidthX2();

  // X3 Faces
  coords.ConvertToCyl(fx3m, xf);
  pfm = PresProfile(pgen, eos_d, xf[0], xf[2]);
  coords.ConvertToCyl(fx3p, xf);
  pfp = (pfm = pgen.pres_min) ? pgen.pres_min : PresProfile(pgen, eos_d, xf[0], xf[2]);
  pfm = (pfp == pgen.pres_min) ? pgen.pres_min : pfm;
  pgrad[2] = (pfp - pfm) / coords.GetCellWidthX3();

  // Convert to the cylindrical radial gradient
  const Real eR[3] = {ex1[0], ex2[0], ex3[0]};
  const Real dpdr = ArtemisUtils::VDot(pgrad, eR);

  // Set v_phi to centrifugal equilibrium
  //   vp^2/R = grad(p) + vk^2/R
  const Real r = std::sqrt(SQR(xcyl[0]) + SQR(xcyl[2]));
  const Real omk2 = pgen.gm / (r * r * r);
  const Real vk2 = omk2 * SQR(xcyl[0]);
  const Real vp = std::sqrt(vk2 + dpdr * xcyl[0] / gdens);
  const Real nu = ViscosityProfile(pgen, eos_d, xcyl[0], xcyl[2]);
  const Real vr = pgen.quiet_start * -1.5 * nu / xcyl[0];

  // Construct the total cylindrical velocity
  const Real vcyl[3] = {vr, vp, 0.0};

  // and convert it to the problem geometry
  gvel1 = ArtemisUtils::VDot(vcyl, ex1);
  gvel2 = ArtemisUtils::VDot(vcyl, ex2);
  gvel3 = ArtemisUtils::VDot(vcyl, ex3);

  if (!(do_dust)) return;

  // Dust is just Keplerian
  ddens = pgen.dust_to_gas * gdens;
  const Real vkep[3] = {0.0, std::sqrt(vk2), 0.0};
  dvel1 = ArtemisUtils::VDot(vkep, ex1);
  dvel2 = ArtemisUtils::VDot(vkep, ex2);
  dvel3 = ArtemisUtils::VDot(vkep, ex3);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void InitDiskParams
//! \brief Extracts disk parameters from ParameterInput.
//! NOTE(PDM): In order for our user-defined BCs to be compatible with restarts, we must
//! reset the DiskParams struct upon initialization.
inline void InitDiskParams(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("disk_params"))) {
    DiskParams disk_params;
    auto &grav_pkg = pmb->packages.Get("gravity");
    auto &gas_pkg = pmb->packages.Get("gas");

    disk_params.gm = grav_pkg->Param<Real>("gm");
    PARTHENON_REQUIRE(!(pin->DoesParameterExist("problem", "rho0") &&
                        pin->DoesParameterExist("problem", "mdot")),
                      "Specify either rho0 or mdot, not both");
    disk_params.r0 = pin->GetOrAddReal("problem", "r0", 1.0);
    disk_params.Omega0 =
        std::sqrt(disk_params.gm / (disk_params.r0 * disk_params.r0 * disk_params.r0));
    disk_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    disk_params.p = pin->GetOrAddReal("problem", "dslope", -2.25);
    disk_params.h0 = pin->GetOrAddReal("problem", "h0", 0.05);
    disk_params.gamma_gas = gas_pkg->Param<Real>("adiabatic_index");
    disk_params.Gamma =
        pin->GetOrAddReal("problem", "polytropic_index", disk_params.gamma_gas);

    PARTHENON_REQUIRE(disk_params.Gamma >= 1, "problem/gamma needs to be >= 1");

    disk_params.dens_min = pin->GetOrAddReal("problem", "dens_min", 1.0e-5);
    disk_params.pres_min = pin->GetOrAddReal("problem", "pres_min", 1.0e-8);
    disk_params.rexp = pin->GetOrAddReal("problem", "rexp", 0.0);
    disk_params.rcav = pin->GetOrAddReal("problem", "rcav", 0.0);
    disk_params.l0 = pin->GetOrAddReal("problem", "l0", 0.0);
    disk_params.dust_to_gas = pin->GetOrAddReal("problem", "dust_to_gas", 0.01);

    disk_params.do_dust = params.Get<bool>("do_dust");

    Real q = pin->GetOrAddReal("problem", "tslope", -Big<Real>());
    Real flare = pin->GetOrAddReal("problem", "flare", -Big<Real>());

    PARTHENON_REQUIRE((flare != -Big<Real>()) || (q != -Big<Real>()),
                      "Set flare or tslope in <problem>");

    if (flare == -Big<Real>()) {
      flare = 0.5 * (1.0 + q);
    } else if (q == -Big<Real>()) {
      q = 2.0 * flare - 1.;
    } else {
      PARTHENON_FAIL("Set either flare or tslope in <problem> not both!");
    }
    disk_params.flare = flare;
    disk_params.q = q;
    disk_params.alpha = 0.0;
    disk_params.nu0 = 0.0;
    disk_params.nu_indx = 0.0;
    disk_params.mdot = 0.0;
    disk_params.quiet_start =
        static_cast<int>(not pin->GetOrAddBoolean("problem", "quiet_start", false));

    if (params.Get<bool>("do_viscosity")) {
      const auto vtype = pin->GetString("gas/viscosity", "type");
      if (vtype == "alpha") {
        disk_params.alpha = pin->GetReal("gas/viscosity", "alpha");
        disk_params.nu0 = disk_params.alpha * disk_params.gamma_gas *
                          SQR(disk_params.h0 * disk_params.r0 * disk_params.Omega0);
        disk_params.nu_indx = 1.5 + disk_params.q;
      } else if (vtype == "constant") {
        disk_params.nu0 = pin->GetReal("gas/viscosity", "nu");
        disk_params.nu_indx = 0.0;
      } else {
        PARTHENON_FAIL("Disk pgen is only compatible with alpha or constant viscosity");
      }
      if (pin->DoesParameterExist("problem", "mdot")) {
        disk_params.mdot = pin->GetReal("problem", "mdot");
        disk_params.rho0 = disk_params.mdot / (3.0 * M_PI * disk_params.nu0);
      } else {
        disk_params.mdot = 3.0 * M_PI * disk_params.nu0 * disk_params.rho0;
      }
    }
    params.Add("disk_params", disk_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DiskICImpl
//! \brief Set the state vectors of cell to the initial conditions
template <Coordinates GEOM, typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void DiskICImpl(V1 v, const int b, const int k, const int j,
                                       const int i, V2 pco, EOS eos_d, DiskParams dp) {
  // gas
  Real gdens = Null<Real>(), gtemp = Null<Real>();
  Real gvel1 = Null<Real>(), gvel2 = Null<Real>(), gvel3 = Null<Real>();
  // dust
  Real ddens = Null<Real>();
  Real dvel1 = Null<Real>(), dvel2 = Null<Real>(), dvel3 = Null<Real>();
  ComputeDiskProfile<GEOM>(dp, pco, k, j, i, eos_d, gdens, gtemp, gvel1, gvel2, gvel3,
                           dp.do_dust, ddens, dvel1, dvel2, dvel3);

  // Set state vector
  v(b, gas::prim::density(0), k, j, i) = gdens;
  v(b, gas::prim::velocity(0), k, j, i) = gvel1;
  v(b, gas::prim::velocity(1), k, j, i) = gvel2;
  v(b, gas::prim::velocity(2), k, j, i) = gvel3;
  v(b, gas::prim::sie(0), k, j, i) =
      eos_d.InternalEnergyFromDensityTemperature(gdens, gtemp);
  if (dp.do_dust) {
    for (int n = 0; n < v.GetSize(b, dust::prim::density()); ++n) {
      v(b, dust::prim::density(n), k, j, i) = ddens;
      v(b, dust::prim::velocity(VI(n, 0)), k, j, i) = dvel1;
      v(b, dust::prim::velocity(VI(n, 1)), k, j, i) = dvel2;
      v(b, dust::prim::velocity(VI(n, 2)), k, j, i) = dvel3;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Disk()
//! \brief Sets initial conditions for disk problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  // Disk parameters
  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");

  // packing and capture variables for kernel
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
  auto &dp = disk_params;

  pmb->par_for(
      "disk", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        DiskICImpl<GEOM>(v, 0, k, j, i, pco, eos_d, dp);
      });
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::DiskBoundaryVisc()
//! \brief Sets inner or outer X1 boundary condition to the initial condition
template <Coordinates GEOM, IndexDomain BDY>
void DiskBoundaryVisc(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {

  PARTHENON_REQUIRE(GEOM == Coordinates::cylindrical || GEOM == Coordinates::spherical ||
                        GEOM == Coordinates::axisymmetric,
                    "Viscous boundary conditions only work with spherical/cylindrical "
                    "radial boundaries");
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  const bool fine = false;

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  auto &dp = disk_params;
  const auto nb = IndexRange{0, 0};

  // Boundary index arithmetic
  int is = Null<int>(), ie = Null<int>();
  int js = Null<int>(), je = Null<int>();
  int ks = Null<int>(), ke = Null<int>();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  if constexpr (BDY == IndexDomain::inner_x1) {
    is = bounds.GetBoundsI(IndexDomain::interior, TE::CC).s;
  } else if constexpr (BDY == IndexDomain::outer_x1) {
    ie = bounds.GetBoundsI(IndexDomain::interior, TE::CC).e;
  } else {
    PARTHENON_FAIL(
        "Viscous boundary conditions only work for the inner or outer radial boundary");
  }
  const int ix1 = 0;
  const int ix2 = 1;
  const int ix3 = 2;

  pmb->par_for_bndry(
      "DiskVisc", nb, BDY, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // We are extrapolating into the ghost zone. Extrapolation is done on cylinders.
        //   dP/dz = - rho grad(Phi)
        //   rho vp^2/R = rho vk^2/R + dP/dR
        // We estimate the pressure gradients as grad(P).eR and grad(P).ez
        // Vertical hydrostatic balance sets rho
        // Radial centrifugal balance sets vp
        const int ia[3] = {k, j, (BDY == IndexDomain::inner_x1) ? is : ie};
        const int ip1[3] = {k, j, (BDY == IndexDomain::inner_x1) ? is + 1 : ie};
        const int im1[3] = {k, j, (BDY == IndexDomain::inner_x1) ? is : ie - 1};

        // Extract coordinates at k, j, i
        geometry::Coords<GEOM> coords(pco, k, j, i);
        Real xv[3] = {Null<Real>()}, xcyl[3] = {Null<Real>()};
        Real ex1[3] = {Null<Real>()}, ex2[3] = {Null<Real>()}, ex3[3] = {Null<Real>()};
        coords.GetCellCenter(xv);
        coords.ConvertToCylWithVec(xv, xcyl, ex1, ex2, ex3);

        // Extract coordinates at ia, im, ic
        Real xva[3] = {Null<Real>()}, xcyla[3] = {Null<Real>()};
        Real xvp1[3] = {Null<Real>()}, xcylp1[3] = {Null<Real>()};
        Real xvm1[3] = {Null<Real>()}, xcylm1[3] = {Null<Real>()};
        Real scr1[3] = {Null<Real>()}, scr2[3] = {Null<Real>()}, scr3[3] = {Null<Real>()};
        geometry::Coords<GEOM> ca(pco, ia[0], ia[1], ia[2]);
        geometry::Coords<GEOM> cp1(pco, ip1[0], ip1[1], ip1[2]);
        geometry::Coords<GEOM> cm1(pco, im1[0], im1[1], im1[2]);
        ca.GetCellCenter(xva);
        ca.ConvertToCylWithVec(xva, xcyla, scr1, scr2, scr3);
        const Real eRa[3] = {scr1[0], scr2[0], scr3[0]};
        const Real epa[3] = {scr1[1], scr2[1], scr3[1]};
        const Real eza[3] = {scr1[2], scr2[2], scr3[2]};
        cp1.GetCellCenter(xvp1);
        cp1.ConvertToCylWithVec(xvp1, xcylp1, scr1, scr2, scr3);
        const Real epp1[3] = {scr1[1], scr2[1], scr3[1]};
        cm1.GetCellCenter(xvm1);
        cm1.ConvertToCylWithVec(xvm1, xcylm1, scr1, scr2, scr3);
        const Real epm1[3] = {scr1[1], scr2[1], scr3[1]};

        // Compute cell separations (using logarithmics if necessary)
        const Real xma = std::log(xv[ix1] / xva[ix1]);
        const Real dx = std::log(xvp1[ix1] / xvm1[ix1]);
        const Real xmadx = xma / dx;

        const Real nua = ViscosityProfile(dp, eos_d, xcyla[0], xcyla[2]);
        const Real nug = ViscosityProfile(dp, eos_d, xcyl[0], xcyl[2]);

        // Extrapolate gas density and specific internal energy
        Real dgsie = std::log(v(0, gas::prim::sie(0), ip1[0], ip1[1], ip1[2]) /
                              v(0, gas::prim::sie(0), im1[0], im1[1], im1[2]));
        const Real gsieexp = std::exp(dgsie * xmadx);
        const Real sieg = v(0, gas::prim::sie(0), ia[0], ia[1], ia[2]) * gsieexp;
        const Real rhoa = v(0, gas::prim::density(0), ia[0], ia[1], ia[2]);

        // Extrapolate gas velocity
        Real gva[3] = {v(0, gas::prim::velocity(0), ia[0], ia[1], ia[2]),
                       v(0, gas::prim::velocity(1), ia[0], ia[1], ia[2]),
                       v(0, gas::prim::velocity(2), ia[0], ia[1], ia[2])};
        Real gvp1[3] = {v(0, gas::prim::velocity(0), ip1[0], ip1[1], ip1[2]),
                        v(0, gas::prim::velocity(1), ip1[0], ip1[1], ip1[2]),
                        v(0, gas::prim::velocity(2), ip1[0], ip1[1], ip1[2])};
        Real gvm1[3] = {v(0, gas::prim::velocity(0), im1[0], im1[1], im1[2]),
                        v(0, gas::prim::velocity(1), im1[0], im1[1], im1[2]),
                        v(0, gas::prim::velocity(2), im1[0], im1[1], im1[2])};
        const Real gvp = ArtemisUtils::VDot(gva, epa);
        const Real gvz = ArtemisUtils::VDot(gva, eza);
        const Real gvp1p = ArtemisUtils::VDot(gvp1, epp1);
        const Real gvm1p = ArtemisUtils::VDot(gvm1, epm1);
        const Real dgvp = std::log(gvp1p / gvm1p);
        const Real vpg = gvp * std::exp(dgvp * xmadx);
        Real rhog, gvR;
        if constexpr (BDY == IndexDomain::inner_x1) {
          rhog = rhoa * nua / nug;
          gvR = -1.5 * nug / xcyl[0];
        } else if constexpr (BDY == IndexDomain::outer_x1) {
          // dFnu/dl = Mdot
          const Real lg = xcyl[0] * vpg;
          const Real la = xcyla[0] * gvp;
          rhog = (3.0 * M_PI * rhoa * nua * la + dp.mdot * (lg - la)) /
                 (3.0 * M_PI * nug * lg);
          gvR = -dp.mdot / (2 * M_PI * xcyl[0] * rhog);
        }

        const Real gvcyl[3] = {gvR, vpg, gvz};
        const Real gvel[3] = {ArtemisUtils::VDot(gvcyl, ex1),
                              ArtemisUtils::VDot(gvcyl, ex2),
                              ArtemisUtils::VDot(gvcyl, ex3)};
        // Set extrapolated values
        v(0, gas::prim::density(0), k, j, i) = rhog;
        v(0, gas::prim::sie(0), k, j, i) = sieg;
        v(0, gas::prim::velocity(ix1), k, j, i) = gvel[ix1];
        v(0, gas::prim::velocity(ix2), k, j, i) = gvel[ix2];
        v(0, gas::prim::velocity(ix3), k, j, i) = gvel[ix3];

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            // Extrapolate dust density
            Real ddrho = std::log(v(0, dust::prim::density(n), ip1[0], ip1[1], ip1[2]) /
                                  v(0, dust::prim::density(n), im1[0], im1[1], im1[2]));
            const Real drhoexp = std::exp(ddrho * xmadx);
            const Real rhod = v(0, dust::prim::density(n), ia[0], ia[1], ia[2]) * drhoexp;

            // Extrapolate dust velocity
            Real dva[3] = {v(0, dust::prim::velocity(VI(n, 0)), ia[0], ia[1], ia[2]),
                           v(0, dust::prim::velocity(VI(n, 1)), ia[0], ia[1], ia[2]),
                           v(0, dust::prim::velocity(VI(n, 2)), ia[0], ia[1], ia[2])};
            Real dvp1[3] = {v(0, dust::prim::velocity(VI(n, 0)), ip1[0], ip1[1], ip1[2]),
                            v(0, dust::prim::velocity(VI(n, 1)), ip1[0], ip1[1], ip1[2]),
                            v(0, dust::prim::velocity(VI(n, 2)), ip1[0], ip1[1], ip1[2])};
            Real dvm1[3] = {v(0, dust::prim::velocity(VI(n, 0)), im1[0], im1[1], im1[2]),
                            v(0, dust::prim::velocity(VI(n, 1)), im1[0], im1[1], im1[2]),
                            v(0, dust::prim::velocity(VI(n, 2)), im1[0], im1[1], im1[2])};
            const Real dvp = ArtemisUtils::VDot(dva, epa);
            const Real dvR = ArtemisUtils::VDot(dva, eRa);
            const Real dvz = ArtemisUtils::VDot(dva, eza);
            const Real dvp1p = ArtemisUtils::VDot(dvp1, epp1);
            const Real dvm1p = ArtemisUtils::VDot(dvm1, epm1);
            const Real ddvp = std::log(dvp1p / dvm1p);
            const Real dvcyl[3] = {dvR, dvp * std::exp(dgvp * xmadx), dvz};
            const Real dvel[3] = {ArtemisUtils::VDot(dvcyl, ex1),
                                  ArtemisUtils::VDot(dvcyl, ex2),
                                  ArtemisUtils::VDot(dvcyl, ex3)};

            // Set extrapolated values
            v(0, dust::prim::density(n), k, j, i) = rhod;
            v(0, dust::prim::velocity(VI(n, ix1)), k, j, i) = dvel[ix1];
            v(0, dust::prim::velocity(VI(n, ix2)), k, j, i) = dvel[ix2];
            v(0, dust::prim::velocity(VI(n, ix3)), k, j, i) = dvel[ix3];
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::DiskBoundaryIC()
//! \brief Sets inner or outer boundary condition to the initial condition
template <Coordinates GEOM, IndexDomain BDY>
void DiskBoundaryIC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");

  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  auto &dp = disk_params;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  pmb->par_for_bndry(
      "DiskInnerX1", nb, BDY, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        DiskICImpl<GEOM>(v, 0, k, j, i, pco, eos_d, dp);
      });
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::DiskBoundaryExtrap()
//! \brief Extrapolation boundary conditions
template <Coordinates GEOM, IndexDomain BDY>
void DiskBoundaryExtrap(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse,
                        const bool lnx) {
  auto pmb = mbd->GetBlockPointer();

  // Extract artemis parameters
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  // Extract gas parameters
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  // Packing
  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, dust::prim::density,
                                                 dust::prim::velocity>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  // Boundary index arithmetic
  int is = Null<int>(), ie = Null<int>();
  int js = Null<int>(), je = Null<int>();
  int ks = Null<int>(), ke = Null<int>();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  if constexpr (BDY == IndexDomain::inner_x1) {
    is = bounds.GetBoundsI(IndexDomain::interior, TE::CC).s;
  } else if constexpr (BDY == IndexDomain::outer_x1) {
    ie = bounds.GetBoundsI(IndexDomain::interior, TE::CC).e;
  } else if constexpr (BDY == IndexDomain::inner_x2) {
    js = bounds.GetBoundsJ(IndexDomain::interior, TE::CC).s;
  } else if constexpr (BDY == IndexDomain::outer_x2) {
    je = bounds.GetBoundsJ(IndexDomain::interior, TE::CC).e;
  } else if constexpr (BDY == IndexDomain::inner_x3) {
    ks = bounds.GetBoundsK(IndexDomain::interior, TE::CC).s;
  } else if constexpr (BDY == IndexDomain::outer_x3) {
    ke = bounds.GetBoundsK(IndexDomain::interior, TE::CC).e;
  }
  constexpr bool x1dir =
      ((BDY == IndexDomain::inner_x1) || (BDY == IndexDomain::outer_x1));
  constexpr bool x2dir =
      ((BDY == IndexDomain::inner_x2) || (BDY == IndexDomain::outer_x2));
  constexpr bool x3dir =
      ((BDY == IndexDomain::inner_x3) || (BDY == IndexDomain::outer_x3));
  constexpr int ix1 = x1dir ? 0 : (x2dir ? 1 : 2);
  constexpr int ix2 = (ix1 + 1) % 3;
  constexpr int ix3 = (ix1 + 2) % 3;

  pmb->par_for_bndry(
      "DiskExtrap", nb, BDY, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // We are extrapolating into the ghost zone. Extrapolation is done on cylinders.
        //   dP/dz = - rho grad(Phi)
        //   rho vp^2/R = rho vk^2/R + dP/dR
        // We estimate the pressure gradients as grad(P).eR and grad(P).ez
        // Vertical hydrostatic balance sets rho
        // Radial centrifugal balance sets vp
        const int ia[3] = {x3dir ? ((BDY == IndexDomain::inner_x3) ? ks : ke) : k,
                           x2dir ? ((BDY == IndexDomain::inner_x2) ? js : je) : j,
                           x1dir ? ((BDY == IndexDomain::inner_x1) ? is : ie) : i};
        const int ip1[3] = {x3dir ? ((BDY == IndexDomain::inner_x3) ? ks + 1 : ke) : k,
                            x2dir ? ((BDY == IndexDomain::inner_x2) ? js + 1 : je) : j,
                            x1dir ? ((BDY == IndexDomain::inner_x1) ? is + 1 : ie) : i};
        const int im1[3] = {x3dir ? ((BDY == IndexDomain::inner_x3) ? ks : ke - 1) : k,
                            x2dir ? ((BDY == IndexDomain::inner_x2) ? js : je - 1) : j,
                            x1dir ? ((BDY == IndexDomain::inner_x1) ? is : ie - 1) : i};

        // Extract coordinates at k, j, i
        geometry::Coords<GEOM> coords(pco, k, j, i);
        Real xv[3] = {Null<Real>()}, xcyl[3] = {Null<Real>()};
        Real ex1[3] = {Null<Real>()}, ex2[3] = {Null<Real>()}, ex3[3] = {Null<Real>()};
        coords.GetCellCenter(xv);
        coords.ConvertToCylWithVec(xv, xcyl, ex1, ex2, ex3);

        // Extract coordinates at ia, im, ic
        Real xva[3] = {Null<Real>()}, xcyla[3] = {Null<Real>()};
        Real xvp1[3] = {Null<Real>()}, xcylp1[3] = {Null<Real>()};
        Real xvm1[3] = {Null<Real>()}, xcylm1[3] = {Null<Real>()};
        Real scr1[3] = {Null<Real>()}, scr2[3] = {Null<Real>()}, scr3[3] = {Null<Real>()};
        geometry::Coords<GEOM> ca(pco, ia[0], ia[1], ia[2]);
        geometry::Coords<GEOM> cp1(pco, ip1[0], ip1[1], ip1[2]);
        geometry::Coords<GEOM> cm1(pco, im1[0], im1[1], im1[2]);
        ca.GetCellCenter(xva);
        ca.ConvertToCylWithVec(xva, xcyla, scr1, scr2, scr3);
        const Real eRa[3] = {scr1[0], scr2[0], scr3[0]};
        const Real epa[3] = {scr1[1], scr2[1], scr3[1]};
        const Real eza[3] = {scr1[2], scr2[2], scr3[2]};
        cp1.GetCellCenter(xvp1);
        cp1.ConvertToCylWithVec(xvp1, xcylp1, scr1, scr2, scr3);
        const Real epp1[3] = {scr1[1], scr2[1], scr3[1]};
        cm1.GetCellCenter(xvm1);
        cm1.ConvertToCylWithVec(xvm1, xcylm1, scr1, scr2, scr3);
        const Real epm1[3] = {scr1[1], scr2[1], scr3[1]};

        // Compute cell separations (using logarithmics if necessary)
        const Real xma = (lnx) ? std::log(xv[ix1] / xva[ix1]) : xv[ix1] - xva[ix1];
        const Real dx = (lnx) ? std::log(xvp1[ix1] / xvm1[ix1]) : xvp1[ix1] - xvm1[ix1];
        const Real xmadx = xma / dx;

        // Extrapolate gas density and specific internal energy
        Real dgrho = std::log(v(0, gas::prim::density(0), ip1[0], ip1[1], ip1[2]) /
                              v(0, gas::prim::density(0), im1[0], im1[1], im1[2]));
        Real dgsie = std::log(v(0, gas::prim::sie(0), ip1[0], ip1[1], ip1[2]) /
                              v(0, gas::prim::sie(0), im1[0], im1[1], im1[2]));
        const Real grhoexp = std::exp(dgrho * xmadx);
        const Real gsieexp = std::exp(dgsie * xmadx);
        const Real rhog = v(0, gas::prim::density(0), ia[0], ia[1], ia[2]) * grhoexp;
        const Real sieg = v(0, gas::prim::sie(0), ia[0], ia[1], ia[2]) * gsieexp;

        // Extrapolate gas velocity
        Real gva[3] = {v(0, gas::prim::velocity(0), ia[0], ia[1], ia[2]),
                       v(0, gas::prim::velocity(1), ia[0], ia[1], ia[2]),
                       v(0, gas::prim::velocity(2), ia[0], ia[1], ia[2])};
        Real gvp1[3] = {v(0, gas::prim::velocity(0), ip1[0], ip1[1], ip1[2]),
                        v(0, gas::prim::velocity(1), ip1[0], ip1[1], ip1[2]),
                        v(0, gas::prim::velocity(2), ip1[0], ip1[1], ip1[2])};
        Real gvm1[3] = {v(0, gas::prim::velocity(0), im1[0], im1[1], im1[2]),
                        v(0, gas::prim::velocity(1), im1[0], im1[1], im1[2]),
                        v(0, gas::prim::velocity(2), im1[0], im1[1], im1[2])};
        const Real gvp = ArtemisUtils::VDot(gva, epa);
        const Real gvR = ArtemisUtils::VDot(gva, eRa);
        const Real gvz = ArtemisUtils::VDot(gva, eza);
        const Real gvp1p = ArtemisUtils::VDot(gvp1, epp1);
        const Real gvm1p = ArtemisUtils::VDot(gvm1, epm1);
        const Real dgvp = std::log(gvp1p / gvm1p);
        const Real gvcyl[3] = {gvR, gvp * std::exp(dgvp * xmadx), gvz};
        const Real gvel[3] = {ArtemisUtils::VDot(gvcyl, ex1),
                              ArtemisUtils::VDot(gvcyl, ex2),
                              ArtemisUtils::VDot(gvcyl, ex3)};

        // Set extrapolated values
        v(0, gas::prim::density(0), k, j, i) = rhog;
        v(0, gas::prim::sie(0), k, j, i) = sieg;
        v(0, gas::prim::velocity(ix1), k, j, i) = gvel[ix1];
        v(0, gas::prim::velocity(ix2), k, j, i) = gvel[ix2];
        v(0, gas::prim::velocity(ix3), k, j, i) = gvel[ix3];

        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            // Extrapolate dust density
            Real ddrho = std::log(v(0, dust::prim::density(n), ip1[0], ip1[1], ip1[2]) /
                                  v(0, dust::prim::density(n), im1[0], im1[1], im1[2]));
            const Real drhoexp = std::exp(ddrho * xmadx);
            const Real rhod = v(0, dust::prim::density(n), ia[0], ia[1], ia[2]) * drhoexp;

            // Extrapolate dust velocity
            Real dva[3] = {v(0, dust::prim::velocity(VI(n, 0)), ia[0], ia[1], ia[2]),
                           v(0, dust::prim::velocity(VI(n, 1)), ia[0], ia[1], ia[2]),
                           v(0, dust::prim::velocity(VI(n, 2)), ia[0], ia[1], ia[2])};
            Real dvp1[3] = {v(0, dust::prim::velocity(VI(n, 0)), ip1[0], ip1[1], ip1[2]),
                            v(0, dust::prim::velocity(VI(n, 1)), ip1[0], ip1[1], ip1[2]),
                            v(0, dust::prim::velocity(VI(n, 2)), ip1[0], ip1[1], ip1[2])};
            Real dvm1[3] = {v(0, dust::prim::velocity(VI(n, 0)), im1[0], im1[1], im1[2]),
                            v(0, dust::prim::velocity(VI(n, 1)), im1[0], im1[1], im1[2]),
                            v(0, dust::prim::velocity(VI(n, 2)), im1[0], im1[1], im1[2])};
            const Real dvp = ArtemisUtils::VDot(dva, epa);
            const Real dvR = ArtemisUtils::VDot(dva, eRa);
            const Real dvz = ArtemisUtils::VDot(dva, eza);
            const Real dvp1p = ArtemisUtils::VDot(dvp1, epp1);
            const Real dvm1p = ArtemisUtils::VDot(dvm1, epm1);
            const Real ddvp = std::log(dvp1p / dvm1p);
            const Real dvcyl[3] = {dvR, dvp * std::exp(dgvp * xmadx), dvz};
            const Real dvel[3] = {ArtemisUtils::VDot(dvcyl, ex1),
                                  ArtemisUtils::VDot(dvcyl, ex2),
                                  ArtemisUtils::VDot(dvcyl, ex3)};

            // Set extrapolated values
            v(0, dust::prim::density(n), k, j, i) = rhod;
            v(0, dust::prim::velocity(VI(n, ix1)), k, j, i) = dvel[ix1];
            v(0, dust::prim::velocity(VI(n, ix2)), k, j, i) = dvel[ix2];
            v(0, dust::prim::velocity(VI(n, ix3)), k, j, i) = dvel[ix3];
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void DiskBoundary
//! \brief
template <Coordinates GEOM, IndexDomain BDY>
inline void DiskBoundary(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto artemis_pkg = pmb->packages.Get("artemis");

  if constexpr (BDY == IndexDomain::inner_x1) {
    ArtemisBC bc = artemis_pkg->Param<ArtemisBC>("ix1_bc");
    constexpr bool lnx = (GEOM != Coordinates::cartesian);
    if (bc == ArtemisBC::ic) {
      return DiskBoundaryIC<GEOM, IndexDomain::inner_x1>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return DiskBoundaryExtrap<GEOM, IndexDomain::inner_x1>(mbd, coarse, lnx);
    } else if (bc == ArtemisBC::visc) {
      return DiskBoundaryVisc<GEOM, IndexDomain::inner_x1>(mbd, coarse);
    }
  } else if constexpr (BDY == IndexDomain::outer_x1) {
    ArtemisBC bc = artemis_pkg->Param<ArtemisBC>("ox1_bc");
    constexpr bool lnx = (GEOM != Coordinates::cartesian);
    if (bc == ArtemisBC::ic) {
      return DiskBoundaryIC<GEOM, IndexDomain::outer_x1>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return DiskBoundaryExtrap<GEOM, IndexDomain::outer_x1>(mbd, coarse, lnx);
    } else if (bc == ArtemisBC::visc) {
      return DiskBoundaryVisc<GEOM, IndexDomain::outer_x1>(mbd, coarse);
    }
  } else if constexpr (BDY == IndexDomain::inner_x2) {
    ArtemisBC bc = artemis_pkg->Param<ArtemisBC>("ix2_bc");
    if (bc == ArtemisBC::ic) {
      return DiskBoundaryIC<GEOM, IndexDomain::inner_x2>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return DiskBoundaryExtrap<GEOM, IndexDomain::inner_x2>(mbd, coarse, false);
    }
  } else if constexpr (BDY == IndexDomain::outer_x2) {
    ArtemisBC bc = artemis_pkg->Param<ArtemisBC>("ox2_bc");
    if (bc == ArtemisBC::ic) {
      return DiskBoundaryIC<GEOM, IndexDomain::outer_x2>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return DiskBoundaryExtrap<GEOM, IndexDomain::outer_x2>(mbd, coarse, false);
    }
  } else if constexpr (BDY == IndexDomain::inner_x3) {
    ArtemisBC bc = artemis_pkg->Param<ArtemisBC>("ix3_bc");
    if (bc == ArtemisBC::ic) {
      return DiskBoundaryIC<GEOM, IndexDomain::inner_x3>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return DiskBoundaryExtrap<GEOM, IndexDomain::inner_x3>(mbd, coarse, false);
    }
  } else if constexpr (BDY == IndexDomain::outer_x3) {
    ArtemisBC bc = artemis_pkg->Param<ArtemisBC>("ox3_bc");
    if (bc == ArtemisBC::ic) {
      return DiskBoundaryIC<GEOM, IndexDomain::outer_x3>(mbd, coarse);
    } else if (bc == ArtemisBC::extrap) {
      return DiskBoundaryExtrap<GEOM, IndexDomain::outer_x3>(mbd, coarse, false);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn AmrTag ProblemCheckRefinementBlock()
//! \brief Refinement criterion for disk pgen
inline parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd) {
  PARTHENON_FAIL("Disk user-defined AMR criterion not yet implemented!");
  return AmrTag::same;
}

} // namespace disk

#endif // PGEN_DISK_HPP_
