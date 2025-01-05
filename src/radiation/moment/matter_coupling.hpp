#ifndef RADIATION_MOMENT_MATTER_COUPLING_HPP_
#define RADIATION_MOMENT_MATTER_COUPLING_HPP_

#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "radiation.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/opacity/opacity.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::Opacity;
using ArtemisUtils::Scattering;
using ArtemisUtils::VI;

namespace Radiation {

template<Coordinates GEOM, Fluid CLOSURE>
TaskStatus MatterCouplingImpl(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt) {
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
                                    gas::cons::momentum, gas::cons::internal_energy,
                                    gas::cons::total_energy>(resolved_pkgs.get());

  static auto desc_prim =
      parthenon::MakePackDescriptor<gas::prim::velocity, gas::prim::density,
                                    gas::prim::sie>(resolved_pkgs.get());
  static auto desc_guess =
      parthenon::MakePackDescriptor<rad::cons::energy, rad::cons::flux,
                                    gas::cons::momentum, gas::cons::internal_energy>(
          resolved_pkgs.get());

  const auto v0 = desc.GetPack(u0);
  const auto vprim = desc_prim.GetPack(u0);
  const auto vg = desc_guess.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  // launch implementation on closure
  
  // Prepare scratch pad memory
  const int ncells1 = ib.e - ib.s + 1 + 2 * parthenon::Globals::nghost;
  const int ngas = vprim.GetMaxNumberOfVars() / 5;
  int scr_size = ScratchPad2D<Real>::shmem_size(ngas, ncells1) * 12;
  const int scr_level = rad_pkg->template Param<int>("scr_level");
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "MatterCoupling", DevExecSpace(), scr_size, scr_level,
      0, u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j) {
        ScratchPad2D<Real> cv(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> B(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> vx1(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> vx2(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> vx3(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> chip(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> chir(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> e0(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> eg(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> v0x1(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> v0x2(mbr.team_scratch(scr_level), ngas, ncells1);
        ScratchPad2D<Real> v0x3(mbr.team_scratch(scr_level), ngas, ncells1);

        // Save initial data
        for (int n = 0; n < ngas; ++n) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, mbr, ib.s, ib.e, [&](const int i) {
                const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                const Real &sie = vprim(b, gas::prim::sie(n), k, j, i);
                cv(n, i) = dens * eos_d.SpecificHeatFromDensityInternalEnergy(dens, sie);
                const Real T = eos_d.TemperatureFromDensityInternalEnergy(dens, sie);
                e0(n, i) = v0(b, gas::cons::internal_energy(n), k, j, i);
                eg(n, i) = e0(n, i);
                vx1(n, i) = vprim(b, gas::prim::velocity(VI(n, 0)), k, j, i);
                vx2(n, i) = vprim(b, gas::prim::velocity(VI(n, 1)), k, j, i);
                vx3(n, i) = vprim(b, gas::prim::velocity(VI(n, 2)), k, j, i);
                B(n, i) = arad * SQR(SQR(T));
                v0x1(n, i) = vx1(n, i);
                v0x2(n, i) = vx2(n, i);
                v0x3(n, i) = vx3(n, i);
              });
        }
        mbr.team_barrier();

        // Do the solve
        // Par reduce to get max iterations?
        parthenon::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, mbr, ib.s, ib.e, [&](const int i) {
              // Outer iteratrion
              Real outer_err = 0.0;
              bool outer_conv = false;
              int outer_iter = 0;
              geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
              const auto &hx = coords.GetScaleFactors();
              std::array<Real, 3> F0{v0(b, rad::cons::flux(0), k, j, i) / hx[0],
                                     v0(b, rad::cons::flux(1), k, j, i) / hx[1],
                                     v0(b, rad::cons::flux(2), k, j, i) / hx[2]};
              Real Er0 = v0(b, rad::cons::energy(), k, j, i);
              std::array<Real, 3> f0{F0[0] / (c * Er0 + Fuzz<Real>()),
                                     F0[1] / (c * Er0 + Fuzz<Real>()),
                                     F0[2] / (c * Er0 + Fuzz<Real>())};
              const auto fedd = EddingtonTensor<CLOSURE>(f0);

              auto F = F0;
              auto Er = Er0;
              while (not outer_conv) {
                outer_err = 0.0;

                Real inner_err = 0.0;
                bool inner_conv = false;
                int inner_iter = 0;
                Real Ek = Er;

                while (not inner_conv) {
                  inner_err = 0.;

                  // Solve for new Ek
                  Real Fr = Ek - Er0;
                  Real ca = 0.0;
                  Real cb = 0.0;
                  for (int n = 0; n < ngas; n++) {
                    const Real B_ = B(n, i);
                    const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                    const Real T = std::pow(B(n, i) / arad, 0.25);
                    const Real e =
                        dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
                    const Real cv_ = cv(n, i);
                    // Evaluate opacities once per iteration
                    chip(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                    chir(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                    Real sigp = chip(n, i) * dt;
                    Real sigr = chir(n, i) * dt;
                    const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
                    const auto f = FleckFactor(arad, T, cv(n, i));

                    const auto &[a, b, d] =
                        EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);

                    const auto &[Ri, Fi] =
                        EnergyRHS(cv_, a, b, d, e, e0(n, i), Ek, B(n, i), c / chat);
                    Fr -= Ri;
                    const Real scale = 1.0 / (1.0 + c / chat * f * a);
                    ca -= (b - a) * scale;
                    const Real cb_ = -Fi * f * (b - a) * scale;
                    cb -= Fi * f * (b - a) * scale;
                  }

                  // (1 + ca)*dEk = -Fr + cb

                  const Real dEk = (-Fr + cb) / (1.0 + ca);
                  Real dE = 0.0;

                  for (int n = 0; n < ngas; n++) {
                    const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                    const Real T = std::pow(B(n, i) / arad, 0.25);
                    const Real e =
                        dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
                    const Real cv_ = cv(n, i);

                    Real sigp = chip(n, i) * dt;
                    Real sigr = chir(n, i) * dt;
                    const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
                    const auto f = FleckFactor(arad, T, cv(n, i));

                    const auto &[a, b, d] =
                        EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);
                    const auto &[Ri, Fi] =
                        EnergyRHS(cv_, a, b, d, e, e0(n, i), Ek, B(n, i), c / chat);

                    const Real ca = c / chat * f * a;
                    const Real cb = -c / chat * f * (b - a);
                    const Real scale = 1.0 / (1.0 + ca);
                    const Real dB = -f * Fi * scale + cb * dEk * scale;
                    inner_err = std::max(inner_err, std::abs(dB / B(n, i)));
                    B(n, i) += dB;
                    // const Real Tnew = std::pow(B(n, i) / arad, 0.25);
                    // const Real enew =
                    //     dens * eos_d.InternalEnergyFromDensityTemperature(dens, Tnew);
                    // // Should I reevaluate B -> T -> e
                    // // Use Ri?
                    // const Real deg = enew - e;
                    // dE -= deg;
                  }
                  // dE *= c / chat;
                  inner_err = std::max(inner_err, std::abs(dEk / (Ek + Fuzz<Real>())));
                  Ek += dEk;
                  // printf("dE %lg\n", std::abs(c / chat * dE / (Er + Fuzz<Real>())));
                  // Cap dE
                  // dE = ((Ek + c / chat * dE) < 0.0) ? -0.1 * chat / c * Ek : dE;
                  // inner_err =
                  //     std::max(inner_err, c / chat * std::abs(dE) / (Er +
                  //     Fuzz<Real>()));
                  // Ek += c / chat * dE;

                  inner_iter++;
                  inner_conv = (inner_err < inner_tol) || (inner_iter > inner_max);
                }
//                if (inner_iter > inner_max) {
// //                 std::stringstream msg;
// //                 msg << "No inner converge: " << inner_err << " " << Er - Ek;
//                  PARTHENON_FAIL("Inner iteration failed to converge");
//                }

                // We have updated energies, now update the flux and velocity
                Er = Ek;
                Real dE = 0.0;
                for (int n = 0; n < ngas; n++) {
                  const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                  const Real T = std::pow(B(n, i) / arad, 0.25);
                  const Real e =
                      dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
                  const Real cv_ = cv(n, i);

                  Real sigp = chip(n, i) * dt;
                  Real sigr = chir(n, i) * dt;
                  const std::array<Real, 3> v{vx1(n, i), vx2(n, i), vx3(n, i)};
                  const auto f = FleckFactor(arad, T, cv(n, i));

                  const auto &[a, b, d] =
                      EnergyExchangeCoeffs(sigp, sigr, v, F, fedd, c, chat);
                  const auto &[dEg, Fi] =
                      EnergyRHS(cv_, a, b, d, e, e0(n, i), Er, B(n, i), c / chat);
                  eg(n, i) = e0(n, i) - c / chat * dEg;

                  dE += dEg;
                }

                Er = Er0 + dE;

                // Update flux fixing v/c , T and Er
                Real alpha = 0.0;
                Real exx = 0.0;
                Real eyy = 0.0;
                Real ezz = 0.0;
                Real exy = 0.0;
                Real exz = 0.0;
                Real eyz = 0.0;
                std::array<Real, 3> delta{0.0, 0.0, 0.0};
                for (int n = 0; n < ngas; n++) {
                  const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                  const Real T =
                      eos_d.TemperatureFromDensityInternalEnergy(dens, eg(n, i));
                  const Real B = arad * SQR(SQR(T));
                  // Evaluate opacities once per iteration
                  chip(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                  chir(n, i) = opac_d.AbsorptionCoefficient(dens, T, 1.0);
                  Real sigp = chip(n, i) * dt;
                  Real sigr = chir(n, i) * dt;
                  const std::array<Real, 3> beta{vx1(n, i) / c, vx2(n, i) / c,
                                                 vx3(n, i) / c};
                  const auto &[a, b, d] =
                      MomentumExchangeCoeffs(sigp, sigr, beta, B, fedd, Er, c, chat);
                  alpha += a;
                  exx += b * beta[0] * beta[0];
                  eyy += b * beta[1] * beta[1];
                  ezz += b * beta[2] * beta[2];
                  exy += b * beta[0] * beta[1];
                  exz += b * beta[0] * beta[2];
                  eyz += b * beta[1] * beta[2];
                  delta[0] += d[0];
                  delta[1] += d[1];
                  delta[2] += d[2];
                }
                delta[0] = F0[0] + c * chat * delta[0];
                delta[1] = F0[1] + c * chat * delta[1];
                delta[2] = F0[2] + c * chat * delta[2];

                alpha = 1.0 + chat * alpha;
                exx *= chat;
                eyy *= chat;
                ezz *= chat;
                exy *= chat;
                exz *= chat;
                eyz *= chat;

                const Real c11 = alpha * (alpha + eyy + ezz) + (eyy * ezz - eyz);
                const Real c22 = alpha * (alpha + exx + ezz) + (exx * ezz - exz);
                const Real c33 = alpha * (alpha + exx + eyy) + (exx * eyy - exy);
                const Real c12 = -alpha * exy + (exz * eyz - exy * ezz);
                const Real c13 = -alpha * exz + (exy * eyz - exz * eyy);
                const Real c23 = -alpha * eyz + (exy * exz - eyz * exx);

                const Real det =
                    alpha * alpha * (alpha + exx + eyy + ezz) +
                    alpha * ((ezz * exx - exz * exz) + (ezz * eyy - eyz * eyz)) +
                    eyz * (exy * exz - eyz * exx) + exz * (exy * eyz - exz * eyy);

                std::array<Real, 3> Fn{0.0, 0.0, 0.0};

                Fn[0] = c11 * delta[0] + c12 * delta[1] + c13 * delta[2];
                Fn[1] = c12 * delta[0] + c22 * delta[1] + c23 * delta[2];
                Fn[2] = c13 * delta[0] + c23 * delta[1] + c33 * delta[2];
                const Real dFx1_ = Fn[0] - F0[0];
                const Real dFx2_ = Fn[1] - F0[1];
                const Real dFx3_ = Fn[2] - F0[2];

                // Fn = NormalizeFlux(Fn[0] / (c * Er), Fn[1] / (c * Er), Fn[2] / (c *
                // Er));

                const Real dFx1 = Fn[0] * c * Er - F0[0];
                const Real dFx2 = Fn[1] * c * Er - F0[1];
                const Real dFx3 = Fn[2] * c * Er - F0[2];

                for (int d = 0; d < 3; d++) {
                  // Fn[d] *= c * Er;
                  const Real err = std::abs((Fn[d] - F[d]) / (F[d] + Fuzz<Real>()));
                  outer_err = std::max(outer_err, err);
                  F[d] = Fn[d];
                }

                // Update material momentum from momentum conservation

                for (int n = 0; n < ngas; n++) {
                  const Real dfx = F[0] - F0[0];
                  const Real dfy = F[1] - F0[1];
                  const Real dfz = F[2] - F0[2];
                  const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                  const Real dvnx1 = -(F[0] - F0[0]) / (c * chat * dens);
                  const Real dvnx2 = -(F[1] - F0[1]) / (c * chat * dens);
                  const Real dvnx3 = -(F[2] - F0[2]) / (c * chat * dens);
                  Real vx[3] = {vx1(n, i), vx2(n, i), vx3(n, i)};
                  Real vnx1 = v0x1(n, i) + dvnx1;
                  Real vnx2 = v0x2(n, i) + dvnx2;
                  Real vnx3 = v0x3(n, i) + dvnx3;
                  const Real derr = vnx1 - vx1(n, i);
                  const Real fuzz_ = Fuzz<Real>();
                  const Real final_ = derr / (vx1(n, i) + Fuzz<Real>());
                  const Real one_ = 1.0 / (vx1(n, i) + Fuzz<Real>());
                  const Real two_ = std::abs(final_);
                  const Real three_ = std::abs(derr / (vx1(n, i) + Fuzz<Real>()));
                  // if ((i == 2) && (j == 3)) {
                  //   printf("v=[%lg,%lg,%lg]\n", vx1(n, i), vx2(n, i), vx3(n, i));
                  //   printf("v0=[%lg,%lg,%lg]\n", v0x1(n, i), v0x2(n, i), v0x3(n, i));
                  //   printf("derr=%lg\nfuzz=%lg\nfinal=%lg\none=%lg\ntwo=%lg\nthree=%"
                  //          "lg\nouter_err=%lg\n",
                  //          derr, fuzz_, final_, one_, two_, three_, outer_err);
                  // }
                  Real err[3] = {
                      std::abs((vnx1 - vx1(n, i)) / (vx1(n, i) + Fuzz<Real>())),
                      std::abs((vnx2 - vx2(n, i)) / (vx2(n, i) + Fuzz<Real>())),
                      std::abs((vnx3 - vx3(n, i)) / (vx3(n, i) + Fuzz<Real>()))};
                  outer_err = std::max(outer_err, err[0]);
                  outer_err = std::max(outer_err, err[1]);
                  outer_err = std::max(outer_err, err[2]);
                  // if ((i == 2) && (j == 3))
                  //   printf("err=[%lg,%lg,%lg] %lg\n", err[0], err[1], err[2],
                  //   outer_err);
                  vx1(n, i) = vnx1;
                  vx2(n, i) = vnx2;
                  vx3(n, i) = vnx3;
                }

                outer_iter++;
                outer_conv = (outer_err < outer_tol) || (outer_iter > outer_max);
              }
//              if (outer_iter > outer_max) {
////                std::stringstream msg;
////                msg << "No outer converge: " << outer_err;
//                PARTHENON_FAIL("Outer iteration failed to converge");
//              }

              v0(b, rad::cons::energy(), k, j, i) += (Er - Er0);
              v0(b, rad::cons::flux(0), k, j, i) = F[0] * hx[0];
              v0(b, rad::cons::flux(1), k, j, i) = F[1] * hx[1];
              v0(b, rad::cons::flux(2), k, j, i) = F[2] * hx[2];

              for (int n = 0; n < ngas; n++) {
                const Real &dens = vprim(b, gas::prim::density(n), k, j, i);
                const Real v02 = SQR(v0x1(n, i)) + SQR(v0x2(n, i)) + SQR(v0x3(n, i));
                const Real v2 = SQR(vx1(n, i)) + SQR(vx2(n, i)) + SQR(vx3(n, i));
                const Real dEk = 0.5 * dens * (v2 - v02);
                const Real dEg = eg(n, i) - e0(n, i);

                const Real dvx1 = vx1(n, i) - v0x1(n, i);
                const Real dvx2 = vx2(n, i) - v0x2(n, i);
                const Real dvx3 = vx3(n, i) - v0x3(n, i);

                v0(b, gas::cons::momentum(VI(n, 0)), k, j, i) += dens * dvx1 * hx[0];
                v0(b, gas::cons::momentum(VI(n, 1)), k, j, i) += dens * dvx2 * hx[1];
                v0(b, gas::cons::momentum(VI(n, 2)), k, j, i) += dens * dvx3 * hx[2];
                v0(b, gas::cons::internal_energy(n), k, j, i) += dEg;
                v0(b, gas::cons::total_energy(n), k, j, i) += dEk + dEg;
                // if (std::abs(std::max({dvx1, dvx2, dvx3, dEk, dEg})) > 1e-15) {
                //   printf("%d %d: %lg %lg %lg %lg %lg\n", j, i, dvx1, dvx2, dvx3, dEk,
                //          dEg);
                // }
              }
            });
      });

  return TaskStatus::complete;

}

} // namespace Radiation

#endif //  RADIATION_MOMENT_MATTER_COUPLING_HPP_
