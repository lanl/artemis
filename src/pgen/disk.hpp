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
#include "gravity/nbody_gravity.hpp"
#include "nbody/nbody.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/units.hpp"

// jaybenne includes
#include "jaybenne.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace disk {
//----------------------------------------------------------------------------------------
//! \struct DiskParams
//! \brief container for disk parameters
struct DiskParams {
  Real r0, h0;
  Real p, q, flare;
  Real rho0, dens_min, pres_min, sie_min, temp_min;
  Real gm, Omega0, l0;
  Real omf;
  Real dust_to_gas;
  Real rexp, exp_pow;
  Real rcav;
  Real Gamma;
  Real alpha, nu0, nu_indx;
  Real mdot;
  Real temp_soft2;
  Real kbmu, ar;
  bool do_gas, do_dust, do_moment, do_imc;
  bool nbody_temp;
  bool quiet_start;
  bool log;
  bool multi_d, three_d;
};

struct State {
  Real gdens = Null<Real>();
  Real gtemp = Null<Real>();
  Real gvel1 = Null<Real>();
  Real gvel2 = Null<Real>();
  Real gvel3 = Null<Real>();
  Real ddens = Null<Real>();
  Real dvel1 = Null<Real>();
  Real dvel2 = Null<Real>();
  Real dvel3 = Null<Real>();
};

//----------------------------------------------------------------------------------------
//! \fn Real DenProfile
//! \brief Computes density profile at cylindrical R and z
KOKKOS_INLINE_FUNCTION
Real DenProfile(struct DiskParams pgen, const Real R, const Real z) {
  const Real r = std::sqrt(R * R + z * z);
  const Real h = pgen.h0 * std::pow(R / pgen.r0, pgen.flare);
  const Real sig0 = pgen.rho0; // / (std::sqrt(2.0 * M_PI) * pgen.h0 * pgen.r0);
  const Real exp_fac =
      (pgen.rexp == 0.) ? 1. : std::exp(-std::pow(R / pgen.rexp, pgen.exp_pow));
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
  const Real pfac = 1. - (pgen.Gamma - 1) * efac;
  return std::max(pgen.dens_min,
                  dmid * std::pow(pfac + Fuzz<Real>(), 1. / (pgen.Gamma - 1)));
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
  const Real ir1 = 1.0 / std::sqrt(R * R + pgen.temp_soft2);
  const Real omk2 = SQR(pgen.Omega0) * ir1 * ir1 * ir1;
  // c_iso^2 = P/rho = kb/mu T = Omk^2 H^2
  const Real T0 = omk2 * H * H / (pgen.kbmu * pgen.Gamma);
  return std::max(pgen.temp_min, T0 * std::pow(rho / rho0, pgen.Gamma - 1.0));
}

//----------------------------------------------------------------------------------------
//! \fn Real PresProfile
//! \brief Computes pressure profile at cylindrical R and z (via dens and temp profiles)
KOKKOS_INLINE_FUNCTION
Real PresProfile(struct DiskParams pgen, const EOS &eos, const Real tf, const Real R,
                 const Real z) {
  const Real df = DenProfile(pgen, R, z);
  return std::max(pgen.pres_min, eos.PressureFromDensityTemperature(df, tf));
}

//----------------------------------------------------------------------------------------
//! \fn Real ViscosityProfile
//! \brief Computes viscosity profile at cylindrical R and z (via dens and temp profiles)
KOKKOS_INLINE_FUNCTION
Real ViscosityProfile(struct DiskParams pgen, const EOS &eos, const Real R,
                      const Real z) {
  return pgen.nu0 * std::pow(R / pgen.r0, pgen.nu_indx);
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeDiskProfile
//! \brief Initialize vertical hydrostatic and radial centrifugal equilibrium disk profile
//! at a specified index/coordinate
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION State ComputeDiskProfile(
    const struct DiskParams pgen, const geometry::Coords<GEOM> &coords,
    const std::array<Real, 3> &xv, const std::array<Real, 3> &dx, const int k,
    const int j, const int i, const EOS &eos_d, const bool do_gas, const bool do_dust,
    ParArray1D<NBody::Particle> particles, const int npart) {
  // Extract coordinates

  const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);
  State res;
  // compute Keplerian solution
  res.gdens = DenProfile(pgen, xcyl[0], xcyl[2]);
  const Real dxr = 1e-6 * std::sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));

  Real rt = xcyl[0];
  Real rtp = xcyl[0] + dxr;
  Real rtm = xcyl[0] - dxr;

  if (pgen.nbody_temp) {
    const Real pot = Gravity::NBodyPotential<GEOM>(coords, xv, particles, npart);
    const Real dxpot = Gravity::NBodyPotential<GEOM>(
        coords, {xv[0] + 1e-6 * dx[0], xv[1], xv[2]}, particles, npart);
    const Real dypot = Gravity::NBodyPotential<GEOM>(
        coords, {xv[0], xv[1] + 1e-6 * dx[1], xv[2]}, particles, npart);
    const Real dzpot = Gravity::NBodyPotential<GEOM>(
        coords, {xv[0], xv[1], xv[2] + 1e-6 * dx[2]}, particles, npart);
    // dPhi/dr = grad(Phi) . \hat{e}_r
    Real drpot = (dxpot - pot) / (2e-6 * dx[0]) * ex1[0];
    drpot += pgen.multi_d * (dypot - pot) / (2e-6 * dx[1]) * ex2[0];
    drpot += pgen.three_d * (dzpot - pot) / (2e-6 * dx[2]) * ex3[0];
    rt = -pgen.gm / pot;
    rtp = -pgen.gm / (pot + drpot * dxr);
    rtm = -pgen.gm / (pot - drpot * dxr);
  }

  res.gtemp = TempProfile(pgen, rt, xcyl[2]);
  const Real tp = TempProfile(pgen, rtp, xcyl[2]);
  const Real tm = TempProfile(pgen, rtm, xcyl[2]);

  // Note that pressure calls density and needs the true cylindrical radius
  const Real pres = PresProfile(pgen, eos_d, res.gtemp, xcyl[0], xcyl[2]);
  const Real dpdr = (PresProfile(pgen, eos_d, tp, xcyl[0] + dxr, xcyl[2]) -
                     PresProfile(pgen, eos_d, tm, xcyl[0] - dxr, xcyl[2])) /
                    (2. * dxr);
  // Set v_phi to centrifugal equilibrium
  //   vp^2/R = grad(p) + vk^2/R
  const Real r = pgen.nbody_temp ? rt : std::sqrt(SQR(xcyl[0]) + SQR(xcyl[2]));
  const Real omk2 = pgen.gm / (r * r * r);
  const Real vk2 = omk2 * SQR(xcyl[0]);
  const Real vp2 = vk2 + (dpdr / res.gdens) * xcyl[0];
  const Real vp = (vp2 < 0.0) ? 0.0 : std::sqrt(vp2);
  const Real nu = ViscosityProfile(pgen, eos_d, rt, xcyl[2]);
  const Real vr = pgen.quiet_start ? 0.0 : -1.5 * nu / xcyl[0];

  // Construct the total cylindrical velocity
  const Real vcyl[3] = {vr, vp - pgen.omf * xcyl[0], 0.0};

  // and convert it to the problem geometry
  res.gvel1 = ArtemisUtils::VDot(vcyl, ex1);
  res.gvel2 = ArtemisUtils::VDot(vcyl, ex2);
  res.gvel3 = ArtemisUtils::VDot(vcyl, ex3);

  if (!(do_dust)) return res;

  // Dust is just Keplerian
  res.ddens = pgen.dust_to_gas * res.gdens;
  const Real vkep[3] = {0.0, std::sqrt(vk2) - pgen.omf * xcyl[0], 0.0};
  res.dvel1 = ArtemisUtils::VDot(vkep, ex1);
  res.dvel2 = ArtemisUtils::VDot(vkep, ex2);
  res.dvel3 = ArtemisUtils::VDot(vkep, ex3);

  return res;
}

//----------------------------------------------------------------------------------------
//! \fn void InitDiskParams
//! \brief Extracts disk parameters from ParameterInput.
//! NOTE(PDM): In order for our user-defined BCs to be compatible with restarts, we must
//! reset the DiskParams struct upon initialization.
inline void InitDiskParams(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  auto &gas_pkg = pmb->packages.Get("gas");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("disk_params"))) {
    DiskParams disk_params;
    disk_params.log = artemis_pkg->Param<geometry::CoordParams>("coord_params").log;
    auto &grav_pkg = pmb->packages.Get("gravity");
    auto &gas_pkg = pmb->packages.Get("gas");

    disk_params.gm = grav_pkg->Param<Real>("gm");
    disk_params.r0 = pin->GetOrAddReal("problem", "r0", 1.0);
    disk_params.Omega0 =
        std::sqrt(disk_params.gm / (disk_params.r0 * disk_params.r0 * disk_params.r0));
    disk_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    disk_params.p = pin->GetOrAddReal("problem", "dslope", -2.25);
    disk_params.h0 = pin->GetOrAddReal("problem", "h0", 0.05);
    disk_params.Gamma = pin->GetOrAddReal("problem", "polytropic_index", 1.0);

    PARTHENON_REQUIRE(disk_params.Gamma >= 1, "problem/gamma needs to be >= 1");

    disk_params.dens_min = gas_pkg->Param<Real>("dfloor");
    disk_params.sie_min = gas_pkg->Param<Real>("siefloor");
    disk_params.rexp = pin->GetOrAddReal("problem", "rexp", 0.0);
    disk_params.exp_pow = pin->GetOrAddReal("problem", "exp_pow", 2.0);
    disk_params.rcav = pin->GetOrAddReal("problem", "rcav", 0.0);
    disk_params.l0 = pin->GetOrAddReal("problem", "l0", 0.0);
    disk_params.dust_to_gas = pin->GetOrAddReal("problem", "dust_to_gas", 0.01);
    disk_params.temp_soft2 = pin->GetOrAddReal("problem", "temp_soft", 0.0);
    const auto mu = gas_pkg->Param<Real>("mu");
    auto &constants = artemis_pkg->Param<ArtemisUtils::Constants>("constants");
    const auto &eos = gas_pkg->Param<ArtemisUtils::EOS>("eos_h");
    disk_params.kbmu = constants.GetKBCode() / (mu * constants.GetAMUCode());
    disk_params.pres_min =
        eos.PressureFromDensityInternalEnergy(disk_params.dens_min, disk_params.sie_min);
    disk_params.temp_min = eos.TemperatureFromDensityInternalEnergy(disk_params.dens_min,
                                                                    disk_params.sie_min);

    const auto nx = params.Get<std::array<int, 3>>("prob_dim");
    disk_params.three_d = nx[2] > 1;
    disk_params.multi_d = disk_params.three_d || (nx[1] > 1);

    disk_params.do_gas = true; // NOTE(@pdmullen): Hardcoded for now...
    disk_params.do_dust = params.Get<bool>("do_dust");

    disk_params.do_imc = params.Get<bool>("do_imc");
    disk_params.do_moment = params.Get<bool>("do_moment");

    disk_params.ar = constants.GetARCode();

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
    disk_params.quiet_start = pin->GetOrAddBoolean("problem", "quiet_start", false);

    if (params.Get<bool>("do_rotating_frame")) {
      auto &rf_pkg = pmb->packages.Get("rotating_frame");
      disk_params.omf = rf_pkg->Param<Real>("omega");
    } else {
      disk_params.omf = 0.0;
    }
    if (params.Get<bool>("do_viscosity")) {
      const auto vtype = pin->GetString("gas/viscosity", "type");
      if (vtype == "alpha") {
        disk_params.alpha = pin->GetReal("gas/viscosity", "alpha");
        disk_params.nu0 =
            disk_params.alpha * SQR(disk_params.h0 * disk_params.r0 * disk_params.Omega0);
        disk_params.nu_indx = 1.5 + disk_params.q;
      } else if ((vtype == "powerlaw") || (vtype == "constant")) {
        disk_params.nu0 = pin->GetReal("gas/viscosity", "nu");
        disk_params.nu_indx = pin->GetOrAddReal("gas/viscosity", "r_exp", 0.0);
      } else {
        PARTHENON_FAIL("Disk pgen is only compatible with alpha or powerlaw viscosity");
      }
      if (pin->DoesParameterExist("problem", "mdot")) {
        disk_params.mdot = pin->GetReal("problem", "mdot");
        disk_params.rho0 = disk_params.mdot / (3.0 * M_PI * disk_params.nu0);
      } else {
        disk_params.mdot = 3.0 * M_PI * disk_params.nu0 * disk_params.rho0;
      }
    }
    disk_params.nbody_temp = pin->GetOrAddBoolean("problem", "nbody_temp", false);
    disk_params.nbody_temp = disk_params.nbody_temp && params.Get<bool>("do_nbody");
    params.Add("disk_params", disk_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DiskICImpl
//! \brief Set the state vectors of cell to the initial conditions

template <Coordinates GEOM, typename V1, typename V2>
KOKKOS_INLINE_FUNCTION void
DiskICImpl(V1 v, const int b, const int k, const int j, const int i, V2 pco,
           const EOS &eos_d, DiskParams dp, ParArray1D<NBody::Particle> particles,
           const int npart) {

  geometry::Coords<GEOM> coords(dp.log, pco, k, j, i);
  const auto &xv = coords.GetCellCenter();
  const auto &dx = coords.GetCellWidths();

  const auto res = ComputeDiskProfile<GEOM>(dp, coords, xv, dx, k, j, i, eos_d, dp.do_gas,
                                            dp.do_dust, particles, npart);

  // Set state vector
  if (dp.do_gas) {
    for (int n = 0; n < v.GetSize(b, gas::prim::density()); ++n) {
      v(b, gas::prim::density(n), k, j, i) = res.gdens;
      v(b, gas::prim::velocity(VI(n, 0)), k, j, i) = res.gvel1;
      v(b, gas::prim::velocity(VI(n, 1)), k, j, i) = res.gvel2;
      v(b, gas::prim::velocity(VI(n, 2)), k, j, i) = res.gvel3;
      v(b, gas::prim::sie(n), k, j, i) =
          eos_d.InternalEnergyFromDensityTemperature(res.gdens, res.gtemp);
    }
  }
  if (dp.do_dust) {
    for (int n = 0; n < v.GetSize(b, dust::prim::density()); ++n) {
      v(b, dust::prim::density(n), k, j, i) = res.ddens;
      v(b, dust::prim::velocity(VI(n, 0)), k, j, i) = res.dvel1;
      v(b, dust::prim::velocity(VI(n, 1)), k, j, i) = res.dvel2;
      v(b, dust::prim::velocity(VI(n, 2)), k, j, i) = res.dvel3;
    }
  }
  if (dp.do_moment) {
    for (int n = 0; n < v.GetSize(b, rad::prim::energy()); ++n) {
      v(b, rad::prim::energy(n), k, j, i) = dp.ar * SQR(SQR(res.gtemp));
      v(b, rad::prim::flux(VI(n, 0)), k, j, i) = 0.0;
      v(b, rad::prim::flux(VI(n, 1)), k, j, i) = 0.0;
      v(b, rad::prim::flux(VI(n, 2)), k, j, i) = 0.0;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Disk()
//! \brief Sets initial conditions for disk problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;

  // Extract artemis package and params
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  // Extract gas package and params
  auto &gas_pkg = pmb->packages.Get("gas");
  const auto &eos_d = gas_pkg->template Param<EOS>("eos_d");

  // Disk parameters
  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");

  // Packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity, rad::prim::energy,
                         rad::prim::flux>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  auto &dp = disk_params;

  ParArray1D<NBody::Particle> particles;
  int npart = 0;
  if (dp.nbody_temp) {
    auto &nbody_pkg = pmb->packages.Get("nbody");
    particles = nbody_pkg->template Param<ParArray1D<NBody::Particle>>("particles");
    npart = static_cast<int>(particles.size());
  }

  pmb->par_for(
      "disk", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        DiskICImpl<GEOM>(v, 0, k, j, i, pco, eos_d, dp, particles, npart);
      });
  if (dp.do_imc) jaybenne::InitializeRadiation(md.get(), true);
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::DiskBoundaryVisc()
//! \brief Sets inner or outer X1 boundary condition to the initial condition
template <Coordinates GEOM, IndexDomain BDY>
void DiskBoundaryVisc(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT

  PARTHENON_REQUIRE(GEOM == Coordinates::cylindrical ||
                        GEOM == Coordinates::spherical3D ||
                        geometry::is_axisymmetric<GEOM>(),
                    "Viscous boundary conditions only work with spherical/cylindrical "
                    "radial boundaries");
  auto pmb = mbd->GetBlockPointer();
  if (coarse && !ArtemisUtils::FineNeighbor(pmb)) return;

  // Extract artemis apackage and params
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_rad = artemis_pkg->Param<bool>("do_moment");
  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");

  // Extract gas package and params
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  // Packing
  static auto descriptors = ArtemisUtils::GetBoundaryPackDescriptorMap<
      gas::prim::density, gas::prim::velocity, gas::prim::sie, dust::prim::density,
      dust::prim::velocity, rad::prim::energy, rad::prim::flux>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;
  // Coordinates and indexing
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  auto &dp = disk_params;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  // Boundary index arithmetic
  int is = Null<int>();
  int ie = Null<int>();
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
        geometry::Coords<GEOM> coords(dp.log, pco, k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // Extract coordinates at ia, im, ic
        geometry::Coords<GEOM> ca(dp.log, pco, ia[0], ia[1], ia[2]);
        geometry::Coords<GEOM> cp1(dp.log, pco, ip1[0], ip1[1], ip1[2]);
        geometry::Coords<GEOM> cm1(dp.log, pco, im1[0], im1[1], im1[2]);
        const auto &xva = ca.GetCellCenter();
        const auto &[xcyla, scr1, scr2, scr3] = ca.ConvertToCylWithVec(xva);
        const Real eRa[3] = {scr1[0], scr2[0], scr3[0]};
        const Real epa[3] = {scr1[1], scr2[1], scr3[1]};
        const Real eza[3] = {scr1[2], scr2[2], scr3[2]};

        const auto &xvp1 = cp1.GetCellCenter();
        const auto &[xcylp1, scr1p1, scr2p1, scr3p1] = cp1.ConvertToCylWithVec(xvp1);
        const Real epp1[3] = {scr1p1[1], scr2p1[1], scr3p1[1]};

        const auto &xvm1 = cm1.GetCellCenter();
        const auto &[xcylm1, scr1m1, scr2m1, scr3m1] = cm1.ConvertToCylWithVec(xvm1);
        const Real epm1[3] = {scr1m1[1], scr2m1[1], scr3m1[1]};

        // Compute cell separations (using logarithmics if necessary)
        const Real xma = std::log(xv[ix1] / xva[ix1]);
        const Real dx = std::log(xvp1[ix1] / xvm1[ix1]);
        const Real xmadx = xma / dx;

        const Real nua = ViscosityProfile(dp, eos_d, xcyla[0], xcyla[2]);
        const Real nug = ViscosityProfile(dp, eos_d, xcyl[0], xcyl[2]);

        // Viscous BC for gas
        if (do_gas) {
          for (int n = 0; n < v.GetSize(0, gas::prim::density()); ++n) {
            Real dgsie = std::log(v(0, gas::prim::sie(n), ip1[0], ip1[1], ip1[2]) /
                                  v(0, gas::prim::sie(n), im1[0], im1[1], im1[2]));
            const Real gsieexp = std::exp(dgsie * xmadx);
            const Real sieg = v(0, gas::prim::sie(n), ia[0], ia[1], ia[2]) * gsieexp;
            const Real rhoa = v(0, gas::prim::density(n), ia[0], ia[1], ia[2]);

            // Extrapolate gas velocity
            Real gva[3] = {v(0, gas::prim::velocity(VI(n, 0)), ia[0], ia[1], ia[2]),
                           v(0, gas::prim::velocity(VI(n, 1)), ia[0], ia[1], ia[2]),
                           v(0, gas::prim::velocity(VI(n, 2)), ia[0], ia[1], ia[2])};
            Real gvp1[3] = {v(0, gas::prim::velocity(VI(n, 0)), ip1[0], ip1[1], ip1[2]),
                            v(0, gas::prim::velocity(VI(n, 1)), ip1[0], ip1[1], ip1[2]),
                            v(0, gas::prim::velocity(VI(n, 2)), ip1[0], ip1[1], ip1[2])};
            Real gvm1[3] = {v(0, gas::prim::velocity(VI(n, 0)), im1[0], im1[1], im1[2]),
                            v(0, gas::prim::velocity(VI(n, 1)), im1[0], im1[1], im1[2]),
                            v(0, gas::prim::velocity(VI(n, 2)), im1[0], im1[1], im1[2])};
            const Real gvp = ArtemisUtils::VDot(gva, epa) + dp.omf * xcyla[0];
            const Real gvz = ArtemisUtils::VDot(gva, eza);
            const Real gvp1p = ArtemisUtils::VDot(gvp1, epp1) + dp.omf * xcylp1[0];
            const Real gvm1p = ArtemisUtils::VDot(gvm1, epm1) + dp.omf * xcylm1[0];
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

            const Real gvcyl[3] = {gvR, vpg - dp.omf * xcyl[0], gvz};
            const Real gvel[3] = {ArtemisUtils::VDot(gvcyl, ex1),
                                  ArtemisUtils::VDot(gvcyl, ex2),
                                  ArtemisUtils::VDot(gvcyl, ex3)};
            // Set extrapolated values
            v(0, gas::prim::density(n), k, j, i) = rhog;
            v(0, gas::prim::sie(n), k, j, i) = sieg;
            v(0, gas::prim::velocity(VI(n, ix1)), k, j, i) = gvel[ix1];
            v(0, gas::prim::velocity(VI(n, ix2)), k, j, i) = gvel[ix2];
            v(0, gas::prim::velocity(VI(n, ix3)), k, j, i) = gvel[ix3];
          }
        }

        // Viscous BC for dust
        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
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
            const Real dvp = ArtemisUtils::VDot(dva, epa) + dp.omf * xcyla[0];
            const Real dvR = ArtemisUtils::VDot(dva, eRa);
            const Real dvz = ArtemisUtils::VDot(dva, eza);
            const Real dvp1p = ArtemisUtils::VDot(dvp1, epp1) + dp.omf * xcylp1[0];
            const Real dvm1p = ArtemisUtils::VDot(dvm1, epm1) + dp.omf * xcylm1[0];
            const Real ddvp = std::log(dvp1p / dvm1p);
            const Real dvcyl[3] = {dvR, dvp * std::exp(ddvp * xmadx) - dp.omf * xcyl[0],
                                   dvz};
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

        // Moments
        if (do_rad) {
          for (int n = 0; n < v.GetSize(0, rad::prim::energy()); ++n) {
            v(0, rad::prim::energy(n), k, j, i) =
                v(0, rad::prim::energy(n), ia[0], ia[1], ia[2]);
            v(0, rad::prim::flux(VI(n, ix1)), k, j, i) =
                v(0, rad::prim::flux(VI(n, ix1)), ia[0], ia[1], ia[2]);
            v(0, rad::prim::flux(VI(n, ix2)), k, j, i) =
                v(0, rad::prim::flux(VI(n, ix2)), ia[0], ia[1], ia[2]);
            v(0, rad::prim::flux(VI(n, ix3)), k, j, i) =
                v(0, rad::prim::flux(VI(n, ix3)), ia[0], ia[1], ia[2]);
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::DiskBoundaryIC()
//! \brief Sets inner or outer boundary condition to the initial condition
template <Coordinates GEOM, IndexDomain BDY>
void DiskBoundaryIC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT
  auto pmb = mbd->GetBlockPointer();
  if (coarse && !ArtemisUtils::FineNeighbor(pmb)) return;

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");

  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  static auto descriptors = ArtemisUtils::GetBoundaryPackDescriptorMap<
      gas::prim::density, gas::prim::velocity, gas::prim::sie, dust::prim::density,
      dust::prim::velocity, rad::prim::energy, rad::prim::flux>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  auto &dp = disk_params;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  ParArray1D<NBody::Particle> particles;
  int npart = 0;
  if (dp.nbody_temp) {
    auto &nbody_pkg = pmb->packages.Get("nbody");
    particles = nbody_pkg->template Param<ParArray1D<NBody::Particle>>("particles");
    npart = static_cast<int>(particles.size());
  }

  pmb->par_for_bndry(
      "DiskInnerX1", nb, BDY, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        DiskICImpl<GEOM>(v, 0, k, j, i, pco, eos_d, dp, particles, npart);
      });
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::DiskBoundaryExtrap()
//! \brief Extrapolation boundary conditions
template <Coordinates GEOM, IndexDomain BDY>
void DiskBoundaryExtrap(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT

  auto pmb = mbd->GetBlockPointer();
  if (coarse && !ArtemisUtils::FineNeighbor(pmb)) return;

  const bool lnx = (GEOM != Coordinates::cartesian);
  // Extract artemis parameters
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_rad = artemis_pkg->Param<bool>("do_moment");

  // Extract gas parameters
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");
  auto &dp = disk_params;

  // Packing
  static auto descriptors = ArtemisUtils::GetBoundaryPackDescriptorMap<
      gas::prim::density, gas::prim::velocity, gas::prim::sie, dust::prim::density,
      dust::prim::velocity, rad::prim::energy, rad::prim::flux>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;

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
  constexpr bool inner =
      ((BDY == IndexDomain::inner_x1) || (BDY == IndexDomain::inner_x2) ||
       (BDY == IndexDomain::inner_x3));

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
        geometry::Coords<GEOM> coords(dp.log, pco, k, j, i);
        geometry::Coords<GEOM> ca(dp.log, pco, ia[0], ia[1], ia[2]);
        geometry::Coords<GEOM> cp(dp.log, pco, ip1[0], ip1[1], ip1[2]);
        geometry::Coords<GEOM> cm(dp.log, pco, im1[0], im1[1], im1[2]);
        const auto &xv = coords.GetCellCenter();
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // Extract coordinates at ia, im, ic
        const auto &xva = ca.GetCellCenter();
        const auto &[xcyla, scr1, scr2, scr3] = coords.ConvertToCylWithVec(xva);
        const Real eRa[3] = {scr1[0], scr2[0], scr3[0]};
        const Real epa[3] = {scr1[1], scr2[1], scr3[1]};
        const Real eza[3] = {scr1[2], scr2[2], scr3[2]};

        const auto &xvp1 = cp.GetCellCenter();
        const auto &[xcylp1, scr1p1, scr2p1, scr3p1] = coords.ConvertToCylWithVec(xvp1);
        const Real epp1[3] = {scr1p1[1], scr2p1[1], scr3p1[1]};

        const auto &xvm1 = cm.GetCellCenter();

        const auto &[xcylm1, scr1m1, scr2m1, scr3m1] = coords.ConvertToCylWithVec(xvm1);
        const Real epm1[3] = {scr1m1[1], scr2m1[1], scr3m1[1]};
        // Compute cell separations (using logarithmics if necessary)
        const Real xma = (lnx) ? std::log(xv[ix1] / xva[ix1]) : xv[ix1] - xva[ix1];
        const Real dx = (lnx) ? std::log(xvp1[ix1] / xvm1[ix1]) : xvp1[ix1] - xvm1[ix1];
        const Real xmadx = xma / dx;

        // Extrapolate gas density and specific internal energy
        if (do_gas) {
          for (int n = 0; n < v.GetSize(0, gas::prim::density()); ++n) {
            Real dgrho = std::log(v(0, gas::prim::density(n), ip1[0], ip1[1], ip1[2]) /
                                  v(0, gas::prim::density(n), im1[0], im1[1], im1[2]));
            Real dgsie = std::log(v(0, gas::prim::sie(n), ip1[0], ip1[1], ip1[2]) /
                                  v(0, gas::prim::sie(n), im1[0], im1[1], im1[2]));
            const Real grhoexp = std::exp(dgrho * xmadx);
            const Real gsieexp = std::exp(dgsie * xmadx);
            const Real rhog = v(0, gas::prim::density(n), ia[0], ia[1], ia[2]) * grhoexp;
            const Real sieg = v(0, gas::prim::sie(n), ia[0], ia[1], ia[2]) * gsieexp;

            // Extrapolate gas velocity
            Real gva[3] = {v(0, gas::prim::velocity(VI(n, 0)), ia[0], ia[1], ia[2]),
                           v(0, gas::prim::velocity(VI(n, 1)), ia[0], ia[1], ia[2]),
                           v(0, gas::prim::velocity(VI(n, 2)), ia[0], ia[1], ia[2])};
            Real gvp1[3] = {v(0, gas::prim::velocity(VI(n, 0)), ip1[0], ip1[1], ip1[2]),
                            v(0, gas::prim::velocity(VI(n, 1)), ip1[0], ip1[1], ip1[2]),
                            v(0, gas::prim::velocity(VI(n, 2)), ip1[0], ip1[1], ip1[2])};
            Real gvm1[3] = {v(0, gas::prim::velocity(VI(n, 0)), im1[0], im1[1], im1[2]),
                            v(0, gas::prim::velocity(VI(n, 1)), im1[0], im1[1], im1[2]),
                            v(0, gas::prim::velocity(VI(n, 2)), im1[0], im1[1], im1[2])};
            const Real gvp = ArtemisUtils::VDot(gva, epa) + dp.omf * xcyla[0];
            const Real gvR = ArtemisUtils::VDot(gva, eRa);
            const Real gvz = ArtemisUtils::VDot(gva, eza);
            const Real gvp1p = ArtemisUtils::VDot(gvp1, epp1) + dp.omf * xcylp1[0];
            const Real gvm1p = ArtemisUtils::VDot(gvm1, epm1) + dp.omf * xcylm1[0];
            const Real dgvp = std::log(gvp1p / gvm1p);
            const Real gvcyl[3] = {gvR, gvp * std::exp(dgvp * xmadx) - dp.omf * xcyl[0],
                                   gvz};
            const Real gvel[3] = {ArtemisUtils::VDot(gvcyl, ex1),
                                  ArtemisUtils::VDot(gvcyl, ex2),
                                  ArtemisUtils::VDot(gvcyl, ex3)};

            // Set extrapolated values
            v(0, gas::prim::density(n), k, j, i) = rhog;
            v(0, gas::prim::sie(n), k, j, i) = sieg;
            const bool inflow = (inner) ? gva[ix1] > 0.0 : gva[ix1] < 0.0;
            v(0, gas::prim::velocity(VI(n, ix1)), k, j, i) = (inflow) ? 0.0 : gvel[ix1];
            v(0, gas::prim::velocity(VI(n, ix2)), k, j, i) = gvel[ix2];
            v(0, gas::prim::velocity(VI(n, ix3)), k, j, i) = gvel[ix3];
          }
        }

        // Extrapolate dust density
        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
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
            const Real dvp = ArtemisUtils::VDot(dva, epa) + dp.omf * xcyla[0];
            const Real dvR = ArtemisUtils::VDot(dva, eRa);
            const Real dvz = ArtemisUtils::VDot(dva, eza);
            const Real dvp1p = ArtemisUtils::VDot(dvp1, epp1) + dp.omf * xcylp1[0];
            const Real dvm1p = ArtemisUtils::VDot(dvm1, epm1) + dp.omf * xcylm1[0];
            const Real ddvp = std::log(dvp1p / dvm1p);
            const Real dvcyl[3] = {dvR, dvp * std::exp(ddvp * xmadx) - dp.omf * xcyl[0],
                                   dvz};
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

        // Moments
        if (do_rad) {
          for (int n = 0; n < v.GetSize(0, rad::prim::energy()); ++n) {
            const Real er0 = v(0, rad::prim::energy(n), ia[0], ia[1], ia[2]);
            Real der = std::log(v(0, rad::prim::energy(n), ip1[0], ip1[1], ip1[2]) /
                                v(0, rad::prim::energy(n), im1[0], im1[1], im1[2]));
            const Real erg = er0 * std::exp(der * xmadx);
            v(0, rad::prim::energy(n), k, j, i) = erg;

            const Real fx1 = v(0, rad::prim::flux(VI(n, ix1)), ia[0], ia[1], ia[2]);
            const bool inflow = (inner) ? fx1 > 0.0 : fx1 < 0.0;
            v(0, rad::prim::flux(VI(n, ix1)), k, j, i) =
                (inflow) ? 0.0 : fx1 * erg / (er0 + Fuzz<Real>());
            v(0, rad::prim::flux(VI(n, ix2)), k, j, i) =
                v(0, rad::prim::flux(VI(n, ix2)), ia[0], ia[1], ia[2]) * erg /
                (er0 + Fuzz<Real>());
            v(0, rad::prim::flux(VI(n, ix3)), k, j, i) =
                v(0, rad::prim::flux(VI(n, ix3)), ia[0], ia[1], ia[2]) * erg /
                (er0 + Fuzz<Real>());
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn AmrTag ProblemCheckRefinementBlock()
//! \brief Refinement criterion for disk pgen
inline parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd) {
  PARTHENON_INSTRUMENT
  PARTHENON_FAIL("Disk user-defined AMR criterion not yet implemented!");
  return AmrTag::same;
}

} // namespace disk

#endif // PGEN_DISK_HPP_
