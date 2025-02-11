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
#ifndef RADIATION_MOMENT_MATTER_COUPLING_HPP_
#define RADIATION_MOMENT_MATTER_COUPLING_HPP_

#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "radiation.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/opacity/opacity.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::Opacity;
using ArtemisUtils::Scattering;
using ArtemisUtils::VI;

namespace Radiation {

template <Coordinates GEOM, Fluid CLOSURE>
TaskStatus MatterCouplingSingleImpl(MeshData<Real> *u0, MeshData<Real> *u1,
                                    const Real dt) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &artemis_pkg = pm->packages.Get("artemis");
  auto &radiation_pkg = pm->packages.Get("radiation");

  auto &gas_pkg = pm->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto opac_d = gas_pkg->template Param<Opacity>("opacity_d");
  auto scat_d = gas_pkg->template Param<Scattering>("scattering_d");
  auto sieflr = gas_pkg->template Param<Real>("siefloor");
  auto dflr = gas_pkg->template Param<Real>("dfloor");
  auto de_switch = gas_pkg->template Param<Real>("de_switch");

  auto &rad_pkg = pm->packages.Get("radiation");
  auto &params = radiation_pkg->AllParams();
  const auto chat = params.template Get<Real>("chat");
  const auto c = params.template Get<Real>("c");
  const auto arad = params.template Get<Real>("arad");
  const auto outer_max = params.template Get<int>("outer_iteration_max");
  const auto inner_max = params.template Get<int>("inner_iteration_max");
  const auto outer_tol = params.template Get<Real>("outer_iteration_tol");
  const auto inner_tol = params.template Get<Real>("inner_iteration_tol");

  // Packing and indexing
  static auto desc =
      parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux,
                                    gas::cons::density, gas::cons::momentum,
                                    gas::cons::internal_energy, gas::cons::total_energy>(
          resolved_pkgs.get());
  static auto desc_guess =
      parthenon::MakePackDescriptor<gas::prim::density, gas::prim::sie,
                                    gas::prim::velocity, rad::cons::energy,
                                    rad::cons::flux>(resolved_pkgs.get());

  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc_guess.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);

  // Prepare scratch pad memory
  // const int ncells1 = ib.e - ib.s + 1 + 2 * parthenon::Globals::nghost;
  // int scr_size = ScratchPad1D<Real>::shmem_size(ncells1) * 12;
  // const int scr_level = rad_pkg->template Param<int>("scr_level");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MatterCoupling", DevExecSpace(), 0, u0->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
        const auto &hx = coords.GetScaleFactors();

        // The state after the explicit update
        const Real &dens = v0(b, gas::cons::density(0), k, j, i);
        const Real &e0 = v0(b, gas::cons::total_energy(0), k, j, i);
        const std::array<Real, 3> p0{v0(b, gas::cons::momentum(0), k, j, i) / hx[0],
                                     v0(b, gas::cons::momentum(1), k, j, i) / hx[1],
                                     v0(b, gas::cons::momentum(2), k, j, i) / hx[2]};
        const Real ke0 =
            (SQR(p0[0] / hx[0]) + SQR(p0[1] / hx[1]) + SQR(p0[2] / hx[2])) / (2. * dens);
        const Real Er0 = v0(b, rad::cons::energy(0), k, j, i);
        std::array<Real, 3> Fr0{v0(b, rad::cons::flux(0), k, j, i) / hx[0],
                                v0(b, rad::cons::flux(1), k, j, i) / hx[1],
                                v0(b, rad::cons::flux(2), k, j, i) / hx[2]};

        // The initial guesses use u1 register as that is fully synced
        Real Tg = eos_d.TemperatureFromDensityInternalEnergy(
            v1(b, gas::prim::density(), k, j, i), v1(b, gas::prim::sie(), k, j, i));
        Real B = arad * SQR(SQR(Tg));
        std::array<Real, 3> v{v1(b, gas::prim::velocity(0), k, j, i),
                              v1(b, gas::prim::velocity(1), k, j, i),
                              v1(b, gas::prim::velocity(2), k, j, i)};
        Real E = v1(b, rad::cons::energy(), k, j, i);
        std::array<Real, 3> F{v1(b, rad::cons::flux(0), k, j, i),
                              v1(b, rad::cons::flux(1), k, j, i),
                              v1(b, rad::cons::flux(2), k, j, i)};

        Real e = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2])) +
                 dens * v1(b, gas::prim::sie(), k, j, i);

        int outer_iter = 0;
        int inner_iter = 0;
        Real outer_err = 0.0;
        Real inner_err = 0.0;
        bool outer_conv = false;
        while (not outer_conv) {
          bool inner_conv = false;
          std::array<Real, 3> f0{F[0] / (c * E + Fuzz<Real>()),
                                 F[1] / (c * E + Fuzz<Real>()),
                                 F[2] / (c * E + Fuzz<Real>())};
          auto fedd = EddingtonTensor<CLOSURE>(f0);
          Real ke = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2]));
          inner_err = 0.0;
          while (not inner_conv) {
            // Compute new B,E
            Real T = std::pow(B / arad, 0.25);
            Real et = ke + dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
            const Real Cv = dens * eos_d.SpecificHeatFromDensityTemperature(dens, T);

            auto sigp = dt * opac_d.AbsorptionCoefficient(dens, T, 1.0);
            auto sigr = dt * opac_d.AbsorptionCoefficient(dens, T, 1.0);

            const auto f = FleckFactor(arad, T, Cv);

            const auto &[a, b, d] = EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);

            const auto &[Ri, Fi] = EnergyRHS(Cv, a, b, d, et, e0, E, B, c / chat);

            const Real scale = 1.0 / (1.0 + c / chat * f * a);
            Real ca = -(b - a) * scale;
            Real cb = -Fi * f * (b - a) * scale;
            Real Fr = (E - Er0) - Ri;
            // rad energy and B changes
            Real dE = (-Fr + cb) / (1.0 + ca);
            Real dB = -f * Fi * scale + cb * dE * scale;

            // Get relative change for this iteration and update
            inner_err = std::max(inner_err, std::abs(dE) / (E + Fuzz<Real>()));
            inner_err = std::max(inner_err, std::abs(dB) / (B + Fuzz<Real>()));
            B += dB;
            E += dE;

            inner_iter++;
            inner_conv = (inner_err <= inner_tol) || (inner_iter >= inner_max);
          } // Done inner

          // With the new T, compute the change in E from conservation
          // Etot = Eg + chat/c Erad
          Real T = std::pow(B / arad, 0.25);
          e = ke + dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
          E = Er0 - c / chat * (e - e0);

          // Now get new Fr at fixed T and E

          auto sigp = chat * dt * opac_d.AbsorptionCoefficient(dens, T, 1.0);
          auto sigr = chat * dt * opac_d.AbsorptionCoefficient(dens, T, 1.0);

          // the order v/c solution
          const Real ifac = 1. / (1. + sigr);
          std::array<Real, 3> dF{
              ifac * (-sigr * Fr0[0] + (sigp * (B - E) + sigr * E) * v[0] +
                      sigr * E *
                          (v[0] * fedd[TensIdx::X11] + v[1] * fedd[TensIdx::X12] +
                           v[2] * fedd[TensIdx::X13])),
              ifac * (-sigr * Fr0[1] + (sigp * (B - E) + sigr * E) * v[1] +
                      sigr * E *
                          (v[0] * fedd[TensIdx::X12] + v[1] * fedd[TensIdx::X22] +
                           v[2] * fedd[TensIdx::X23])),
              ifac * (-sigr * Fr0[2] + (sigp * (B - E) + sigr * E) * v[2] +
                      sigr * E *
                          (v[0] * fedd[TensIdx::X13] + v[1] * fedd[TensIdx::X23] +
                           v[2] * fedd[TensIdx::X33]))};

          // Momentum conservation sets rho*v
          // Ptot = rho*v + F/(c chat)
          const Real icc = 1. / (c * chat * dens);
          std::array<Real, 3> dv{-icc * dF[0], -icc * dF[1], -icc * dF[2]};

          // Get relative change and update
          for (int d = 0; d < 3; d++) {
            outer_err =
                std::max(outer_err, std::abs(dv[d]) / (std::abs(v[d]) + Fuzz<Real>()));
            outer_err =
                std::max(outer_err, std::abs(dF[d]) / (std::abs(F[d]) + Fuzz<Real>()));
            v[d] += dv[d];
            F[d] += dF[d];
          }
          outer_iter++;
          outer_conv = (outer_err <= outer_tol) || (outer_iter >= outer_max);
        } // done oterr

        // Converged values, now update the registers
        v0(b, gas::cons::total_energy(), k, j, i) = e;
        v0(b, rad::cons::energy(), k, j, i) = E;
        for (int d = 0; d < 3; d++) {
          v0(b, gas::cons::momentum(d), k, j, i) = dens * v[d] * hx[d];
          v0(b, rad::cons::flux(d), k, j, i) = F[d];
        }
      });

  return TaskStatus::complete;
}

// template <Coordinates GEOM, Fluid CLOSURE>
// TaskStatus MatterCouplingImpl(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt) {
//   using parthenon::MakePackDescriptor;
//   using parthenon::variable_names::any;
//   auto pm = u0->GetParentPointer();
//   auto &resolved_pkgs = pm->resolved_packages;
//   auto &artemis_pkg = pm->packages.Get("artemis");
//   auto &radiation_pkg = pm->packages.Get("radiation");

//   auto &gas_pkg = pm->packages.Get("gas");
//   auto eos_d = gas_pkg->template Param<EOS>("eos_d");
//   auto opac_d = gas_pkg->template Param<Opacity>("opacity_d");
//   auto scat_d = gas_pkg->template Param<Scattering>("scattering_d");
//   auto sieflr = gas_pkg->template Param<Real>("siefloor");
//   auto dflr = gas_pkg->template Param<Real>("dfloor");
//   auto de_switch = gas_pkg->template Param<Real>("de_switch");

//   auto &rad_pkg = pm->packages.Get("radiation");
//   auto &params = radiation_pkg->AllParams();
//   const auto chat = params.template Get<Real>("chat");
//   const auto c = params.template Get<Real>("c");
//   const auto arad = params.template Get<Real>("arad");
//   const auto outer_max = params.template Get<int>("outer_iteration_max");
//   const auto inner_max = params.template Get<int>("inner_iteration_max");
//   const auto outer_tol = params.template Get<Real>("outer_iteration_tol");
//   const auto inner_tol = params.template Get<Real>("inner_iteration_tol");

//   // Packing and indexing
//   static auto desc =
//       parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux,
//                                     gas::cons::density, gas::cons::momentum,
//                                     gas::cons::internal_energy,
//                                     gas::cons::total_energy>(
//           resolved_pkgs.get());

//   static auto desc_prim =
//       parthenon::MakePackDescriptor<gas::prim::velocity, gas::prim::density,
//                                     gas::prim::sie>(resolved_pkgs.get());
//   static auto desc_guess =
//       parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux,
//                                     gas::cons::momentum, gas::cons::internal_energy>(
//           resolved_pkgs.get());

//   const auto v0 = desc.GetPack(u0);
//   const auto vprim = desc_prim.GetPack(u0);
//   const auto vg = desc_guess.GetPack(u0);
//   const auto ib = u0->GetBoundsI(IndexDomain::interior);
//   const auto jb = u0->GetBoundsJ(IndexDomain::interior);
//   const auto kb = u0->GetBoundsK(IndexDomain::interior);
//   const bool multi_d = (pm->ndim > 1);
//   const bool three_d = (pm->ndim > 2);

//   // launch implementation on closure

//   // Prepare scratch pad memory
//   const int ncells1 = ib.e - ib.s + 1 + 2 * parthenon::Globals::nghost;
//   const int ngas = vprim.GetMaxNumberOfVars() / 5;
//   int scr_size = ScratchPad2D<Real>::shmem_size(ngas, ncells1) * 12;
//   const int scr_level = rad_pkg->template Param<int>("scr_level");
//   parthenon::par_for_outer(
//       DEFAULT_OUTER_LOOP_PATTERN, "MatterCoupling", DevExecSpace(), scr_size,
//       scr_level, 0, u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
//       KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j) {
//         ScratchPad2D<Real> cv(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> B(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> vx1(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> vx2(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> vx3(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> chip(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> chir(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> e0(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> eg(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> v0x1(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> v0x2(mbr.team_scratch(scr_level), ngas, ncells1);
//         ScratchPad2D<Real> v0x3(mbr.team_scratch(scr_level), ngas, ncells1);

//         // Save initial data
//         for (int n = 0; n < ngas; ++n) {
//           parthenon::par_for_inner(
//               DEFAULT_INNER_LOOP_PATTERN, mbr, ib.s, ib.e, [&](const int i) {
//                 geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
//                 const auto &hx = coords.GetScaleFactors();
//                 const Real &dens = v0(b, gas::cons::density(n), k, j, i);
//                 const Real sie = ArtemisUtils::GetSpecificInternalEnergy(
//                     v0, b, n, k, j, i, de_switch, dflr, sieflr, hx);
//                 cv(n, i) = dens * eos_d.SpecificHeatFromDensityInternalEnergy(dens,
//                 sie); const Real T = eos_d.TemperatureFromDensityInternalEnergy(dens,
//                 sie); e0(n, i) = v0(b, gas::cons::total_energy(n), k, j, i); eg(n, i) =
//                 e0(n, i); vx1(n, i) =
//                     v0(b, gas::cons::momentum(VI(n, 0)), k, j, i) / (dens * hx[0]);
//                 vx2(n, i) =
//                     v0(b, gas::cons::momentum(VI(n, 1)), k, j, i) / (dens * hx[1]);
//                 vx3(n, i) =
//                     v0(b, gas::cons::momentum(VI(n, 2)), k, j, i) / (dens * hx[2]);
//                 B(n, i) = arad * SQR(SQR(T));
//                 v0x1(n, i) = vx1(n, i);
//                 v0x2(n, i) = vx2(n, i);
//                 v0x3(n, i) = vx3(n, i);
//               });
//         }
//         mbr.team_barrier();

//         // Do the solve
//         // Par reduce to get max iterations?
//         parthenon::par_for_inner(
//             DEFAULT_INNER_LOOP_PATTERN, mbr, ib.s, ib.e, [&](const int i) {
//               // Outer iteratrion
//               // printf("(%d,%d,%d):\n", k, j, i);
//               Real outer_err = Null<Real>();
//               bool outer_conv = false;
//               int outer_iter = 0;
//               int inner_iter = 0;
//               Real inner_err = Null<Real>();
//               geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
//               const auto &hx = coords.GetScaleFactors();
//               std::array<Real, 3> F0{v0(b, rad::cons::flux(0), k, j, i) / hx[0],
//                                      v0(b, rad::cons::flux(1), k, j, i) / hx[1],
//                                      v0(b, rad::cons::flux(2), k, j, i) / hx[2]};
//               Real Er0 = v0(b, rad::cons::energy(), k, j, i);
//               std::array<Real, 3> f0{F0[0] / (c * Er0 + Fuzz<Real>()),
//                                      F0[1] / (c * Er0 + Fuzz<Real>()),
//                                      F0[2] / (c * Er0 + Fuzz<Real>())};
//               const auto fedd = EddingtonTensor<CLOSURE>(f0);

//               auto F = F0;
//               auto Er = Er0;
//               while (not outer_conv) {
//                 outer_err = 0.0;

//                 bool inner_conv = false;
//                 inner_iter = 0;
//                 // Real Ek = Er;

//                 // CHANGE THIS TO WORK WITH INTERNAL ENERGY?
//                 // IF KE IS FIXED, THEN WE CAN USE INTERNAL RIGHT?;p[[[]]]

//                 // E - E0  + c/chat R
//                 // E = Ek + Eg
//                 // dE/dB = dEg/dB
//                 // dE/d

//                 while (not inner_conv) {
//                   inner_err = 0.0;

//                   // Solve for new Ek
//                   Real Fr = Er - Er0;
//                   Real ca = 0.0;
//                   Real cb = 0.0;
//                   for (int n = 0; n < ngas; n++) {
//                     const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
//                     const Real &dens = v0(b, gas::cons::density(n), k, j, i);
//                     const Real ke = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2]));
//                     const Real T = std::pow(B(n, i) / arad, 0.25);
//                     const Real e =
//                         ke + dens * eos_d.InternalEnergyFromDensityTemperature(dens,
//                         T);
//                     // Evaluate opacities once per iteration
//                     chip(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
//                     chir(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
//                     Real sigp = chip(n, i) * dt;
//                     Real sigr = chir(n, i) * dt;

//                     const auto f = FleckFactor(arad, T, cv(n, i));

//                     const auto &[a, b, d] =
//                         EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);

//                     const auto &[Ri, Fi] =
//                         EnergyRHS(cv(n, i), a, b, d, e, e0(n, i), Er, B(n, i), c /
//                         chat);
//                     Fr -= Ri;
//                     const Real scale = 1.0 / (1.0 + c / chat * f * a);
//                     ca -= (b - a) * scale;
//                     cb -= Fi * f * (b - a) * scale;
//                   }

//                   // (1 + ca)*dEk = -Fr + cb

//                   const Real dEk = (-Fr + cb) / (1.0 + ca);
//                   Real dE = 0.0;

//                   for (int n = 0; n < ngas; n++) {
//                     const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
//                     const Real &dens = v0(b, gas::cons::density(n), k, j, i);
//                     const Real ke = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2]));
//                     const Real T = std::pow(B(n, i) / arad, 0.25);
//                     const Real e =
//                         ke + dens * eos_d.InternalEnergyFromDensityTemperature(dens,
//                         T);

//                     Real sigp = chip(n, i) * dt;
//                     Real sigr = chir(n, i) * dt;

//                     const auto f = FleckFactor(arad, T, cv(n, i));

//                     const auto &[a, b, d] =
//                         EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);
//                     const auto &[Ri, Fi] =
//                         EnergyRHS(cv(n, i), a, b, d, e, e0(n, i), Er, B(n, i), c /
//                         chat);

//                     const Real ca = c / chat * f * a;
//                     const Real cb = -c / chat * f * (b - a);
//                     const Real scale = 1.0 / (1.0 + ca);
//                     const Real dB = -f * Fi * scale + cb * dEk * scale;
//                     const Real dEg = -c / chat * Ri;
//                     dE += Ri;
//                     eg(n, i) = e0(n, i) + dEg;
//                     inner_err = std::max(inner_err, std::abs(dB / B(n, i)));
//                     B(n, i) += dB;
//                   }
//                   // dE *= c / chat;
//                   inner_err = std::max(inner_err, std::abs(dE / (Er + Fuzz<Real>())));
//                   Er = Er0 + dE;

//                   // Ek += dEk;
//                   inner_iter++;
//                   inner_conv = (inner_err <= inner_tol) || (inner_iter > inner_max);
//                 }
//                 if ((inner_iter > inner_max) && (inner_err > inner_tol)) {
//                   printf("No inner convergence after %d iterations: %lg > %lg\n",
//                          inner_iter, inner_err, inner_tol);
//                   PARTHENON_FAIL("");
//                 }

//                 // We have updated energies, now update the flux and velocity
//                 // Er = Ek;
//                 // Real dE = 0.0;
//                 // for (int n = 0; n < ngas; n++) {
//                 //   const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
//                 //   const Real &dens = v0(b, gas::cons::density(n), k, j, i);
//                 //   const Real ke = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2]));
//                 //   const Real T = std::pow(B(n, i) / arad, 0.25);
//                 //   const Real e =
//                 //       ke + dens * eos_d.InternalEnergyFromDensityTemperature(dens,
//                 T);

//                 //   Real sigp = chip(n, i) * dt;
//                 //   Real sigr = chir(n, i) * dt;

//                 //   const auto f = FleckFactor(arad, T, cv(n, i));

//                 //   const auto &[a, b, d] =
//                 //       EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);
//                 //   const auto &[dEg, Fi] =
//                 //       EnergyRHS(cv(n,i), a, b, d, e, e0(n, i), Er, B(n, i), c /
//                 chat);
//                 //   eg(n, i) = e0(n, i) - c / chat * dEg;

//                 //   dE += dEg;
//                 // }

//                 // Er = Er0 + dE;

//                 // Update flux fixing v/c , T and Er
//                 Real alpha = 0.0;
//                 Real exx = 0.0;
//                 Real eyy = 0.0;
//                 Real ezz = 0.0;
//                 Real exy = 0.0;
//                 Real exz = 0.0;
//                 Real eyz = 0.0;
//                 std::array<Real, 3> delta{0.0, 0.0, 0.0};
//                 for (int n = 0; n < ngas; n++) {
//                   const Real &dens = v0(b, gas::cons::density(n), k, j, i);
//                   const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
//                   const Real ke = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2]));
//                   const Real eint = (eg(n, i) - ke);
//                   const Real T = std::pow(B(n, i) / arad, 0.25);
//                   //    eos_d.TemperatureFromDensityInternalEnergy(dens, eint / dens);
//                   // const Real B = arad * SQR(SQR(T));
//                   // Evaluate opacities once per iteration
//                   chip(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
//                   chir(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
//                   Real sigp = chip(n, i) * dt;
//                   Real sigr = chir(n, i) * dt;
//                   const std::array<Real, 3> beta{vx1(n, i) / c, vx2(n, i) / c,
//                                                  vx3(n, i) / c};
//                   const auto &[a, b, d] = MomentumExchangeCoeffs(
//                       sigp, sigr, beta, B(n, i), fedd, Er, c, chat);
//                   alpha += a;
//                   exx += b * beta[0] * beta[0];
//                   eyy += b * beta[1] * beta[1];
//                   ezz += b * beta[2] * beta[2];
//                   exy += b * beta[0] * beta[1];
//                   exz += b * beta[0] * beta[2];
//                   eyz += b * beta[1] * beta[2];
//                   delta[0] += d[0];
//                   delta[1] += d[1];
//                   delta[2] += d[2];
//                 }
//                 delta[0] = F0[0] + c * chat * delta[0];
//                 delta[1] = F0[1] + c * chat * delta[1];
//                 delta[2] = F0[2] + c * chat * delta[2];

//                 alpha = 1.0 + chat * alpha;
//                 exx *= chat;
//                 eyy *= chat;
//                 ezz *= chat;
//                 exy *= chat;
//                 exz *= chat;
//                 eyz *= chat;

//                 const Real c11 = alpha * (alpha + eyy + ezz) + (eyy * ezz - eyz);
//                 const Real c22 = alpha * (alpha + exx + ezz) + (exx * ezz - exz);
//                 const Real c33 = alpha * (alpha + exx + eyy) + (exx * eyy - exy);
//                 const Real c12 = -alpha * exy + (exz * eyz - exy * ezz);
//                 const Real c13 = -alpha * exz + (exy * eyz - exz * eyy);
//                 const Real c23 = -alpha * eyz + (exy * exz - eyz * exx);

//                 const Real det =
//                     alpha * alpha * (alpha + exx + eyy + ezz) +
//                     alpha * ((ezz * exx - exz * exz) + (ezz * eyy - eyz * eyz)) +
//                     eyz * (exy * exz - eyz * exx) + exz * (exy * eyz - exz * eyy);

//                 std::array<Real, 3> Fn{0.0, 0.0, 0.0};

//                 Fn[0] = c11 * delta[0] + c12 * delta[1] + c13 * delta[2];
//                 Fn[1] = c12 * delta[0] + c22 * delta[1] + c23 * delta[2];
//                 Fn[2] = c13 * delta[0] + c23 * delta[1] + c33 * delta[2];
//                 const Real dFx1_ = Fn[0] - F0[0];
//                 const Real dFx2_ = Fn[1] - F0[1];
//                 const Real dFx3_ = Fn[2] - F0[2];

//                 Fn = NormalizeFlux(Fn[0] / (c * Er), Fn[1] / (c * Er), Fn[2] / (c *
//                 Er));

//                 const Real dFx1 = Fn[0] * c * Er - F0[0];
//                 const Real dFx2 = Fn[1] * c * Er - F0[1];
//                 const Real dFx3 = Fn[2] * c * Er - F0[2];

//                 Real err = 0.0;
//                 Real denom = 0.0;
//                 for (int d = 0; d < 3; d++) {
//                   Fn[d] *= c * Er;
//                   err += SQR(Fn[d] - F[d]);
//                   denom += SQR(F[d]);
//                   F[d] = Fn[d];
//                 }
//                 err = std::sqrt(err) / (std::sqrt(denom) + Fuzz<Real>());
//                 outer_err = std::max(outer_err, err);
//                 auto outer_err_ = outer_err;

//                 // Update material momentum from momentum conservation
//                 // this is wrong
//                 for (int n = 0; n < ngas; n++) {
//                   const Real dfx = F[0] - F0[0];
//                   const Real dfy = F[1] - F0[1];
//                   const Real dfz = F[2] - F0[2];
//                   const Real &dens = v0(b, gas::cons::density(n), k, j, i);
//                   const Real dvnx1 = -(F[0] - F0[0]) / (c * chat * dens);
//                   const Real dvnx2 = -(F[1] - F0[1]) / (c * chat * dens);
//                   const Real dvnx3 = -(F[2] - F0[2]) / (c * chat * dens);
//                   Real vx[3] = {vx1(n, i), vx2(n, i), vx3(n, i)};
//                   Real vnx1 = v0x1(n, i) + dvnx1;
//                   Real vnx2 = v0x2(n, i) + dvnx2;
//                   Real vnx3 = v0x3(n, i) + dvnx3;
//                   Real err_ =
//                       std::sqrt(SQR(vnx1 - vx[0]) + SQR(vnx2 - vx[1]) +
//                                 SQR(vnx3 - vx[2])) /
//                       (std::sqrt(SQR(vx[0]) + SQR(vx[1]) + SQR(vx[2])) + Fuzz<Real>());
//                   outer_err = std::max(outer_err, err_);
//                   vx1(n, i) = vnx1;
//                   vx2(n, i) = vnx2;
//                   vx3(n, i) = vnx3;
//                 }

//                 outer_iter++;
//                 outer_conv = (outer_err < outer_tol) || (outer_iter > outer_max);
//               }
//               // #ifdef DEBUG
//               // printf("inner: (%d, %lg), outer: (%d, %lg)\n", inner_iter, inner_err,
//               //        outer_iter, outer_err);
//               // #endif
//               if ((outer_iter > outer_max) && (outer_err > outer_tol)) {
//                 printf("No outer convergence after %d iterations: %lg > %lg\n",
//                        outer_iter, outer_err, outer_tol);
//                 PARTHENON_FAIL("");
//               }

//               // already have the new values...

//               Real dEr = 0.0;
//               std::array<Real, 3> dFr{0.0, 0.0, 0.0};
//               for (int n = 0; n < ngas; n++) {
//                 const Real &dens = v0(b, gas::cons::density(n), k, j, i);
//                 const Real v02 = SQR(v0x1(n, i)) + SQR(v0x2(n, i)) + SQR(v0x3(n, i));
//                 const Real v2 = SQR(vx1(n, i)) + SQR(vx2(n, i)) + SQR(vx3(n, i));
//                 const Real dEk = 0.5 * dens * (v2 - v02);
//                 const Real dEt = eg(n, i) - e0(n, i);
//                 // E = Eg + chat/c Er
//                 // dEg = -chat/c dEr
//                 const Real dEg = dEt - dEk;
//                 dEr -= c / chat * dEt;
//                 // const Real dEg = eg(n, i) - e0(n, i);

//                 const Real dmx1 = dens * hx[0] * (vx1(n, i) - v0x1(n, i));
//                 const Real dmx2 = dens * hx[1] * (vx2(n, i) - v0x2(n, i));
//                 const Real dmx3 = dens * hx[2] * (vx3(n, i) - v0x3(n, i));
//                 // mom = rhov + F/(c chat)
//                 // dm = - dF/(c chat)
//                 dFr[0] -= c * chat * dmx1;
//                 dFr[1] -= c * chat * dmx2;
//                 dFr[2] -= c * chat * dmx3;

//                 v0(b, gas::cons::momentum(VI(n, 0)), k, j, i) += dmx1;
//                 v0(b, gas::cons::momentum(VI(n, 1)), k, j, i) += dmx2;
//                 v0(b, gas::cons::momentum(VI(n, 2)), k, j, i) += dmx3;
//                 v0(b, gas::cons::internal_energy(n), k, j, i) += dEg;
//                 v0(b, gas::cons::total_energy(n), k, j, i) = eg(n, i); // dEt;
//               }

//               v0(b, rad::cons::energy(), k, j, i) = Er;
//               v0(b, rad::cons::flux(0), k, j, i) += dFr[0]; // F[0] * hx[0];
//               v0(b, rad::cons::flux(1), k, j, i) += dFr[1]; // F[1] * hx[1];
//               v0(b, rad::cons::flux(2), k, j, i) += dFr[2]; // F[2] * hx[2];
//             });
//       });

//   return TaskStatus::complete;
// }

} // namespace Radiation

#endif //  RADIATION_MOMENT_MATTER_COUPLING_HPP_
