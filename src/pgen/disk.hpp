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
#include "gravity/nbody_gravity.hpp"
#include "nbody/nbody.hpp"
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
  Real omf;
  Real dust_to_gas;
  Real rexp;
  Real rcav;
  Real Gamma, gamma_gas;
  Real alpha, nu0, nu_indx;
  Real mdot;
  Real temp_soft2;
  bool const_nu;
  bool do_dust;
  bool nbody_temp;
  bool quiet_start;
};

//----------------------------------------------------------------------------------------
//! \fn Real DenProfile
//! \brief Computes density profile at cylindrical R and z
KOKKOS_INLINE_FUNCTION
Real DenProfile(struct DiskParams pgen, const Real R, const Real z) {
  const Real H = R * pgen.h0 * std::pow(R / pgen.r0, pgen.flare);
  return std::max(pgen.dens_min, pgen.rho0*std::pow(R/pgen.r0, pgen.p)*std::exp(-0.5*SQR(z/H)));
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
  const Real T0 = omk2 * H * H / pgen.Gamma;
  return T0 * std::pow(rho / rho0, pgen.Gamma - 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real PresProfile
//! \brief Computes pressure profile at cylindrical R and z (via dens and temp profiles)
KOKKOS_INLINE_FUNCTION
Real PresProfile(struct DiskParams pgen, EOS eos, const Real tf, const Real R,
                 const Real z) {
  const Real df = DenProfile(pgen, R, z);
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
                   Real &ddens, Real &dvel1, Real &dvel2, Real &dvel3,
                   ParArray1D<NBody::Particle> particles, const int npart) {
  // Extract coordinates
  geometry::Coords<GEOM> coords(pco, k, j, i);
  const auto &xv = coords.GetCellCenter();

  const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

  // compute Keplerian solution
  gdens = DenProfile(pgen, xcyl[0], xcyl[2]);
  const Real rt =
      pgen.nbody_temp
          ? -pgen.gm / Gravity::NBodyPotential<GEOM>(coords, xv, particles, npart)
          : xcyl[0];
  gtemp = TempProfile(pgen, rt, xcyl[2]);

  // Set v_phi to centrifugal equilibrium
  //   vp^2/R = grad(p) + vk^2/R
  if (!pgen.const_nu) {
    PARTHENON_FAIL("jtlaune: This version only works with nu viscosity.");
  }

  const Real H = xcyl[0] * pgen.h0 * std::pow(xcyl[0] / pgen.r0, pgen.flare);
  const Real OmKmid = std::sqrt(pgen.gm / (xcyl[0] * xcyl[0] * xcyl[0]));
  const Real Omg = OmKmid * (1 + 0.5 * SQR(H / xcyl[0]) *
                                     (pgen.p + pgen.q + 0.5 * pgen.q * SQR(xcyl[2] / H)));
  const Real vp = Omg * xcyl[0];
  const Real vr = - pgen.nu0*(6*pgen.p-2*pgen.q+3+(5*pgen.q+9)*SQR(xcyl[2]/H))/(2*cylv[0]);

  // Construct the total cylindrical velocity
  const Real vcyl[3] = {vr, vp - pgen.omf * xcyl[0], 0.0};

  // and convert it to the problem geometry
  gvel1 = ArtemisUtils::VDot(vcyl, ex1);
  gvel2 = ArtemisUtils::VDot(vcyl, ex2);
  gvel3 = ArtemisUtils::VDot(vcyl, ex3);

  // debugging code

  //if (i == 2 && j== 2&& k==2){
  //std::cout << "init:" << gvel1 << ", " << gvel2 << ", " << gvel3 << std::endl;
  //}
  //if (i==2 && j==2 && k==2){
  //  std::cout << "init cc:" << std::fixed << std::setprecision(12)<< xv[0] << ", " << xv[1] << ", " << xv[2] << std::endl;
  //  std::cout << "init v:" << std::fixed << std::setprecision(12)<< gvel1 << ", " << gvel2 << ", " << gvel3 << std::endl;
  //  std::cout << "init ex1:" << std::fixed << std::setprecision(12)<< ex1[0] << ", " << ex1[1] << ", " << ex1[2] << std::endl;
  //  std::cout << "init vcyl:" << std::fixed << std::setprecision(12) << vcyl[0]
  //            << ", " << vcyl[1] << ", " << vcyl[2] << std::endl;
  //}

  //if (!(do_dust)) return;

  //  PARTHENON_FAIL("jtlaune: This version only works without dust.");

  //return;
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
    disk_params.temp_soft2 = pin->GetOrAddReal("problem", "temp_soft", 0.0);

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
        disk_params.const_nu = false;
        disk_params.alpha = pin->GetReal("gas/viscosity", "alpha");
        disk_params.nu0 = disk_params.alpha * disk_params.gamma_gas *
                          SQR(disk_params.h0 * disk_params.r0 * disk_params.Omega0);
        disk_params.nu_indx = 1.5 + disk_params.q;
      } else if (vtype == "constant") {
        disk_params.const_nu = true;
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
DiskICImpl(V1 v, const int b, const int k, const int j, const int i, V2 pco, EOS eos_d,
           DiskParams dp, ParArray1D<NBody::Particle> particles, const int npart) {
  // gas
  Real gdens = Null<Real>(), gtemp = Null<Real>();
  Real gvel1 = Null<Real>(), gvel2 = Null<Real>(), gvel3 = Null<Real>();
  // dust
  Real ddens = Null<Real>();
  Real dvel1 = Null<Real>(), dvel2 = Null<Real>(), dvel3 = Null<Real>();
  ComputeDiskProfile<GEOM>(dp, pco, k, j, i, eos_d, gdens, gtemp, gvel1, gvel2, gvel3,
                           dp.do_dust, ddens, dvel1, dvel2, dvel3, particles, npart);

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
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::DiskBoundaryVisc()
//! \brief Sets inner or outer X1 boundary condition to the initial condition
template <Coordinates GEOM, IndexDomain BDY>
void DiskBoundaryVisc(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {

  PARTHENON_REQUIRE(GEOM == Coordinates::cylindrical ||
                        GEOM == Coordinates::spherical3D ||
                        geometry::is_axisymmetric<GEOM>(),
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
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // Extract coordinates at ia, im, ic
        geometry::Coords<GEOM> ca(pco, ia[0], ia[1], ia[2]);
        geometry::Coords<GEOM> cp1(pco, ip1[0], ip1[1], ip1[2]);
        geometry::Coords<GEOM> cm1(pco, im1[0], im1[1], im1[2]);
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
            const Real dvp = ArtemisUtils::VDot(dva, epa) + dp.omf * xcyla[0];
            const Real dvR = ArtemisUtils::VDot(dva, eRa);
            const Real dvz = ArtemisUtils::VDot(dva, eza);
            const Real dvp1p = ArtemisUtils::VDot(dvp1, epp1) + dp.omf * xcylp1[0];
            const Real dvm1p = ArtemisUtils::VDot(dvm1, epm1) + dp.omf * xcylm1[0];
            const Real ddvp = std::log(dvp1p / dvm1p);
            const Real dvcyl[3] = {dvR, dvp * std::exp(dgvp * xmadx) - dp.omf * xcyl[0],
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
  const bool lnx = (GEOM != Coordinates::cartesian);

  auto pmb = mbd->GetBlockPointer();

  // Extract artemis parameters
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  // Extract gas parameters
  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  auto disk_params = artemis_pkg->Param<DiskParams>("disk_params");
  auto &dp = disk_params;

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
        const auto &xv = coords.GetCellCenter();
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // Extract coordinates at ia, im, ic
        geometry::Coords<GEOM> ca(pco, ia[0], ia[1], ia[2]);
        geometry::Coords<GEOM> cp1(pco, ip1[0], ip1[1], ip1[2]);
        geometry::Coords<GEOM> cm1(pco, im1[0], im1[1], im1[2]);
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
        const Real gvp = ArtemisUtils::VDot(gva, epa) + dp.omf * xcyla[0];
        const Real gvR = ArtemisUtils::VDot(gva, eRa);
        const Real gvz = ArtemisUtils::VDot(gva, eza);
        const Real gvp1p = ArtemisUtils::VDot(gvp1, epp1) + dp.omf * xcylp1[0];
        const Real gvm1p = ArtemisUtils::VDot(gvm1, epm1) + dp.omf * xcylm1[0];
        const Real dgvp = std::log(gvp1p / gvm1p);
        const Real gvcyl[3] = {gvR, gvp * std::exp(dgvp * xmadx) - dp.omf * xcyl[0], gvz};
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
            const Real dvp = ArtemisUtils::VDot(dva, epa) + dp.omf * xcyla[0];
            const Real dvR = ArtemisUtils::VDot(dva, eRa);
            const Real dvz = ArtemisUtils::VDot(dva, eza);
            const Real dvp1p = ArtemisUtils::VDot(dvp1, epp1) + dp.omf * xcylp1[0];
            const Real dvm1p = ArtemisUtils::VDot(dvm1, epm1) + dp.omf * xcylm1[0];
            const Real ddvp = std::log(dvp1p / dvm1p);
            const Real dvcyl[3] = {dvR, dvp * std::exp(dgvp * xmadx) - dp.omf * xcyl[0],
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
      });
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
