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
#ifndef RADIATION_MOMENTS_MATTER_COUPLING_HPP_
#define RADIATION_MOMENTS_MATTER_COUPLING_HPP_

#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "moments.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/opacity/opacity.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::MeanOpacity;
using ArtemisUtils::MeanScattering;
using ArtemisUtils::VI;

namespace Moments {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Moments::MatterCouplingSimpleImpl
//! \brief Implementation for simple radiation-matter coupling source
template <Coordinates GEOM, Closure CLOSURE>
TaskStatus MatterCouplingSimpleImpl(MeshData<Real> *u0, const Real dt) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract gas package and params
  auto &gas_pkg = pm->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto opac_d = gas_pkg->template Param<MeanOpacity>("opacity_d");
  auto scat_d = gas_pkg->template Param<MeanScattering>("scattering_d");
  auto dflr = gas_pkg->template Param<Real>("dfloor");
  auto de_switch = gas_pkg->template Param<Real>("de_switch");

  // Extract radiation and moments package and params
  auto &moments_pkg = pm->packages.Get("moments");
  const auto chat = moments_pkg->template Param<Real>("chat");
  const auto c = moments_pkg->template Param<Real>("c");
  const auto arad = moments_pkg->template Param<Real>("arad");
  const auto rad_efloor = moments_pkg->template Param<Real>("efloor");
  const auto tfloor = moments_pkg->template Param<Real>("tfloor");
  const auto Bfloor = arad * SQR(SQR(tfloor));
  const auto efloor = Bfloor;
  const auto outer_max = moments_pkg->template Param<int>("outer_iteration_max");
  const auto inner_max = moments_pkg->template Param<int>("inner_iteration_max");
  const auto outer_tol = moments_pkg->template Param<Real>("outer_iteration_tol");
  const auto inner_tol = moments_pkg->template Param<Real>("inner_iteration_tol");
  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  const auto fatal_if_unconverged =
      moments_pkg->template Param<bool>("fatal_if_unconverged");

  // Extract rotating frame quantities
  Real om0 = 0.0;
  Real qshear = 0.0;
  Real gm_bg = 0.0;
  if (pm->packages.Get("artemis")->template Param<bool>("do_orbital_advection") ||
      pm->packages.Get("artemis")->template Param<bool>("do_rotating_frame")) {
    auto &rframe_pkg = pm->packages.Get("rotating_frame");
    qshear = rframe_pkg->template Param<Real>("qshear");
    om0 = rframe_pkg->template Param<Real>("omega");
    gm_bg = rframe_pkg->template Param<Real>("gm");
  }
  const bool do_raytrace =
      pm->packages.Get("artemis")->template Param<bool>("do_raytrace");

  // Packing and indexing
  static auto desc = parthenon::MakePackDescriptor<
      rad::cons::energy, rad::cons::flux, gas::cons::density, gas::cons::momentum,
      gas::cons::internal_energy, gas::cons::total_energy, gas::src::energy>(
      resolved_pkgs.get());
  const auto v0 = desc.GetPack(u0);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(u0);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MatterCoupling", DevExecSpace(), 0, u0->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, v0.GetCoordinates(b), k, j, i);
        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);
        // y = U^(0) + dt S(y)

        // U^(0) values
        const Real dens = v0(b, gas::cons::density(), k, j, i);
        Real Q = 0.0;
        if (do_raytrace) Q = dt * v0(b, gas::src::energy(), k, j, i);
        Real e0 = v0(b, gas::cons::internal_energy(), k, j, i);
        const auto vb = RotatingFrame::BackgroundVelocity<GEOM>(
            qshear, om0, gm_bg, coords.GetCellCenter(vg, b, k, j, i));
        std::array<Real, 3> v{
            vb[0] + v0(b, gas::cons::momentum(0), k, j, i) / (hx[0] * dens),
            vb[1] + v0(b, gas::cons::momentum(1), k, j, i) / (hx[1] * dens),
            vb[2] + v0(b, gas::cons::momentum(2), k, j, i) / (hx[2] * dens)};
        const Real Er0 = v0(b, rad::cons::energy(), k, j, i);
        std::array<Real, 3> Fr0{v0(b, rad::cons::flux(0), k, j, i) / hx[0],
                                v0(b, rad::cons::flux(1), k, j, i) / hx[1],
                                v0(b, rad::cons::flux(2), k, j, i) / hx[2]};

        // Note(AMD): There is some floating point difference between the internal energy
        // used to compute the temperature and the internal energy obtained from that
        // temperature: T = eos_d.TemperatureFromDensityInternalEnergy(dens, eg/dens); eg
        // /= dens * eos_d.InternalEnergyFromDensityTemperature(dens,T)
        //
        // Because of this, zero opacity problems will not result in zero change as
        // expected. Thus, we recalculate the internal and total energies from the
        // temperature. This does not affect energy conservation because at the end of the
        // step we update the energy with an increment.

        Real T = eos_d.TemperatureFromDensityInternalEnergy(dens, e0 / dens);
        e0 = dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
        Real e = e0;
        Real B = arad * SQR(SQR(T));

        Real E = Er0;
        Real etot = e + c / chat * E;

        // S(y) = sigma*chat*(E-B)
        //
        int inner_iter = 0;
        Real inner_err = 0.;
        for (inner_iter = 0; inner_iter < inner_max; inner_iter++) {
          T = std::pow(B / arad, 0.25);
          e = eos_d.InternalEnergyFromDensityTemperature(dens, T) * dens;
          const Real Cv = dens * eos_d.SpecificHeatFromDensityTemperature(dens, T);
          const Real a = chat * dt * opac_d.PlanckMeanAbsorptionCoefficient(dens, T);
          const Real fleck = FleckFactor(arad, T, Cv);

          const Real Ri = a * (E - B);
          const Real Fi = (e - e0) - c / chat * Ri - Q;
          const Real Fr = (E - Er0) + Ri;
          const Real idet = 1. / (1. + a + c / chat * fleck * a);
          Real dE = ((1. + c / chat * fleck * a) * (-Fr) + a * (-fleck * Fi)) * idet;
          Real dB = ((c / chat * fleck * a) * (-Fr) + (1. + a) * (-fleck * Fi)) * idet;

          Real Enew = E + dE;
          E = (Enew < efloor) ? efloor : Enew;
          Real Bnew = B + dB;
          B = (Bnew < Bfloor) ? Bfloor : Bnew;

          inner_err = std::max((std::abs(Fi) / etot), (c / chat * std::abs(Fr) / etot));
          if (inner_err <= inner_tol) {
            break;
          }
        }
        if ((inner_iter == inner_max) && (fatal_if_unconverged)) {
          printf("(%d,%d,%d,%d)  %lg > %lg after %d iterations\n", b, k, j, i, inner_err,
                 inner_tol, inner_max);
          PARTHENON_FAIL("Radiation matter coupling did not converge!");
        }
        T = std::pow(B / arad, 0.25);
        e = eos_d.InternalEnergyFromDensityTemperature(dens, T) * dens;
        const Real dEg = e - e0;
        Real a = chat * dt *
                 (opac_d.RosselandMeanAbsorptionCoefficient(dens, T) +
                  scat_d.RosselandMeanTotalScatteringCoefficient(dens, T));
        std::array<Real, 3> dF{-a / (1. + a) * Fr0[0], -a / (1. + a) * Fr0[1],
                               -a / (1. + a) * Fr0[2]};
        const Real icc = -1. / (c * chat * dens);
        std::array<Real, 3> dv{-icc * dF[0], -icc * dF[1], -icc * dF[2]};
        std::array<Real, 3> vn{v[0] + dv[0], v[1] + dv[1], v[2] + dv[2]};

        const Real dEk = 0.5 * dens *
                         ((SQR(vn[0]) - SQR(v[0])) + (SQR(vn[1]) - SQR(v[1])) +
                          (SQR(vn[2]) - SQR(v[2])));

        // Update state vector (both gas and radiation)
        v0(b, rad::cons::energy(), k, j, i) += chat / c * (dEk - dEg);
        v0(b, gas::cons::total_energy(), k, j, i) += dEg + dEk;
        v0(b, gas::cons::internal_energy(), k, j, i) += dEg;
        v0(b, rad::cons::flux(0), k, j, i) += dF[0] * hx[0];
        v0(b, gas::cons::momentum(0), k, j, i) += dv[0] * dens * hx[0];
        v0(b, rad::cons::flux(1), k, j, i) += dF[1] * hx[1];
        v0(b, gas::cons::momentum(1), k, j, i) += dv[1] * dens * hx[1];
        v0(b, rad::cons::flux(2), k, j, i) += dF[2] * hx[2];
        v0(b, gas::cons::momentum(2), k, j, i) += dv[2] * dens * hx[2];
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Moments::MatterCouplingSimpleImpl
//! \brief Implementation for "full" radiation-matter coupling source
template <Coordinates GEOM, Closure CLOSURE>
TaskStatus MatterCouplingFullSingleImpl(MeshData<Real> *u0, const Real dt) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract gas package and params
  auto &gas_pkg = pm->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");
  auto opac_d = gas_pkg->template Param<MeanOpacity>("opacity_d");
  auto scat_d = gas_pkg->template Param<MeanScattering>("scattering_d");
  auto dflr = gas_pkg->template Param<Real>("dfloor");
  auto de_switch = gas_pkg->template Param<Real>("de_switch");

  // Extract radiation package and params
  auto &moments_pkg = pm->packages.Get("moments");
  const auto chat = moments_pkg->template Param<Real>("chat");
  const auto c = moments_pkg->template Param<Real>("c");
  const auto arad = moments_pkg->template Param<Real>("arad");
  const auto rad_efloor = moments_pkg->template Param<Real>("efloor");
  const auto tfloor = moments_pkg->template Param<Real>("tfloor");
  const auto outer_max = moments_pkg->template Param<int>("outer_iteration_max");
  const auto inner_max = moments_pkg->template Param<int>("inner_iteration_max");
  const auto outer_tol = moments_pkg->template Param<Real>("outer_iteration_tol");
  const auto inner_tol = moments_pkg->template Param<Real>("inner_iteration_tol");
  const Real inner_practical_tol = std::max(inner_tol, 1.0e-8);
  const auto fatal_if_unconverged =
      moments_pkg->template Param<bool>("fatal_if_unconverged");

  // Extract rotating frame quantities
  Real om0 = 0.0;
  Real qshear = 0.0;
  Real gm_bg = 0.0;
  if (pm->packages.Get("artemis")->template Param<bool>("do_orbital_advection") ||
      pm->packages.Get("artemis")->template Param<bool>("do_rotating_frame")) {
    auto &rframe_pkg = pm->packages.Get("rotating_frame");
    qshear = rframe_pkg->template Param<Real>("qshear");
    om0 = rframe_pkg->template Param<Real>("omega");
    gm_bg = rframe_pkg->template Param<Real>("gm");
  }
  const auto &cpars =
      pm->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

  const bool do_raytrace =
      pm->packages.Get("artemis")->template Param<bool>("do_raytrace");

  // Packing and indexing
  static auto desc = parthenon::MakePackDescriptor<
      rad::cons::energy, rad::cons::flux, gas::cons::density, gas::cons::momentum,
      gas::cons::internal_energy, gas::cons::total_energy, gas::src::energy>(
      resolved_pkgs.get());

  const auto v0 = desc.GetPack(u0);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(u0);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MatterCoupling", DevExecSpace(), 0, u0->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, v0.GetCoordinates(b), k, j, i);
        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);
        // y = U^(0) + dt S(y)

        // U^(0) values
        const Real dens = std::max(v0(b, gas::cons::density(), k, j, i), dflr);
        // The full energy+momentum solve is not robust or physically useful in
        // floor-adjacent cells. Treat them as pure local heating updates.
        constexpr Real coupling_dfloor_factor = 100.0;
        Real Q = 0.0;
        if (do_raytrace) Q = dt * v0(b, gas::src::energy(), k, j, i);
        if (dens <= coupling_dfloor_factor * dflr) {
          v0(b, gas::cons::internal_energy(), k, j, i) += Q;
          v0(b, gas::cons::total_energy(), k, j, i) += Q;
          return;
        }

        // Note(AMD): There is some floating point difference between the internal energy
        // used to compute the temperature and the internal energy obtained from that
        // temperature: T = eos_d.TemperatureFromDensityInternalEnergy(dens, eg/dens); eg
        // /= dens * eos_d.InternalEnergyFromDensityTemperature(dens,T)
        //
        // Because of this, zero opacity problems will not result in zero change as
        // expected. Thus, we recalculate the internal and total energies from the
        // temperature. This does not affect energy conservation because at the end of the
        // step we update the energy with an increment.
        const Real Bfloor_phys = arad * SQR(SQR(tfloor));
        const Real eint0 =
            std::max(v0(b, gas::cons::internal_energy(), k, j, i) / dens, 0.0);
        Real T =
            std::max(tfloor, eos_d.TemperatureFromDensityInternalEnergy(dens, eint0));
        Real eg0 = dens * eos_d.InternalEnergyFromDensityTemperature(dens, T);
        Real B = std::max(Bfloor_phys, arad * SQR(SQR(T)));

        const auto vb = RotatingFrame::BackgroundVelocity<GEOM>(
            qshear, om0, gm_bg, coords.GetCellCenter(vg, b, k, j, i));
        const std::array<Real, 3> p0{
            vb[0] * dens + v0(b, gas::cons::momentum(0), k, j, i) / hx[0],
            vb[1] * dens + v0(b, gas::cons::momentum(1), k, j, i) / hx[1],
            vb[2] * dens + v0(b, gas::cons::momentum(2), k, j, i) / hx[2]};

        Real E0 = std::max(v0(b, rad::cons::energy(), k, j, i), rad_efloor);
        // choose the ref scale

        Real eref = std::max(std::sqrt(E0 * B), 0.5 * (E0 + B));
        eref = std::max(eref, std::max(rad_efloor, Bfloor_phys));
        const Real fref = c * eref;
        const Real efloor = rad_efloor / eref;
        const Real Bfloor = Bfloor_phys / eref;

        Q /= eref;
        E0 /= eref;
        eg0 /= eref;
        B /= eref;

        std::array<Real, 3> Fr0 =
            ProjectFlux({v0(b, rad::cons::flux(0), k, j, i) / hx[0] / fref,
                         v0(b, rad::cons::flux(1), k, j, i) / hx[1] / fref,
                         v0(b, rad::cons::flux(2), k, j, i) / hx[2] / fref},
                        E0);

        std::array<Real, 3> v{p0[0] / dens, p0[1] / dens, p0[2] / dens};
        const Real beta20 = SQR(v[0] / c) + SQR(v[1] / c) + SQR(v[2] / c);
        if (!(IsFinite(beta20)) || beta20 >= 1.0 - 1.0e-12) {
          v0(b, gas::cons::internal_energy(), k, j, i) += Q;
          v0(b, gas::cons::total_energy(), k, j, i) += Q;
          return;
        }
        const Real ke0 = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2])) / eref;
        const Real et0 = ke0 + eg0;

        Real E = E0;
        auto F = Fr0;

        // Start outer iteration

        int outer_iter = 0;
        int inner_iter = 0;
        Real outer_err = 0.0;
        Real inner_err = 0.0;

        std::array<Real, 3> dF{0., 0., 0.};
        std::array<Real, 3> dv{0., 0., 0.};
        Real dEk = 0.0;
        Real dEg = 0.0;
        Real dEr = 0.0;
        const Real icc = 1. / (c * chat * dens);
        Real escale = std::max(et0 + c / chat * E + std::abs(Q), efloor + Bfloor);
        bool inner_converged = false;
        bool outer_converged = false;
        bool solver_valid = IsFinite(escale);
        bool have_good_state = false;
        std::array<Real, 3> best_dF{0., 0., 0.};
        std::array<Real, 3> best_dv{0., 0., 0.};
        Real best_dEg = 0.0;
        Real best_dEk = 0.0;
        Real best_dEr = 0.0;

        if (!(IsFinite(eref) && IsFinite(E0) && IsFinite(B) && IsFinite(eg0) &&
              IsFinite(ke0) && IsFinite(escale) && eref > 0.0 && escale > 0.0)) {
          solver_valid = false;
        }

        for (outer_iter = 1; outer_iter <= outer_max; outer_iter++) {
          if (!solver_valid) break;
          inner_converged = false;

          // Set some v and F quantities
          F = ProjectFlux(F, std::max(E, efloor));
          const Real E_prev = E;
          const Real B_prev = B;
          const auto F_prev = F;
          Real ke = 0.5 * dens * (SQR(v[0]) + SQR(v[1]) + SQR(v[2])) / eref;
          std::array<Real, 3> beta{v[0] / c, v[1] / c, v[2] / c};
          Real beta2 = SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);
          constexpr Real beta2_max = 1.0 - 1.0e-12;
          if (!(IsFinite(beta2))) {
            solver_valid = false;
            break;
          }
          if (beta2 >= beta2_max) {
            const Real fac = std::sqrt(beta2_max / std::max(beta2, Fuzz<Real>()));
            for (int d = 0; d < 3; ++d) {
              beta[d] *= fac;
              v[d] = beta[d] * c;
            }
            beta2 = SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);
          }
          const Real g2 = 1. / (1. - beta2);
          const Real g = std::sqrt(g2);
          if (!(IsFinite(g2) && IsFinite(g))) {
            solver_valid = false;
            break;
          }

          auto fedd =
              EddingtonTensor<CLOSURE>({F[0] / (std::max(E, efloor) + Fuzz<Real>()),
                                        F[1] / (std::max(E, efloor) + Fuzz<Real>()),
                                        F[2] / (std::max(E, efloor) + Fuzz<Real>())});

          std::array<Real, 3> bdp{
              beta[0] * fedd[TensIdx::X11] + beta[1] * fedd[TensIdx::X12] +
                  beta[2] * fedd[TensIdx::X13],
              beta[0] * fedd[TensIdx::X12] + beta[1] * fedd[TensIdx::X22] +
                  beta[2] * fedd[TensIdx::X23],
              beta[0] * fedd[TensIdx::X13] + beta[1] * fedd[TensIdx::X23] +
                  beta[2] * fedd[TensIdx::X33]};
          const Real bdbdp = beta[0] * bdp[0] + beta[1] * bdp[1] + beta[2] * bdp[2];
          const Real bdf = beta[0] * F[0] + beta[1] * F[1] + beta[2] * F[2]; // 1/c

          // start inner iteration for (B,E)
          for (inner_iter = 1; inner_iter <= inner_max; inner_iter++) {
            T = std::pow(std::max(eref * B / arad, 0.0), 0.25);
            T = std::max(T, tfloor);
            Real eint = dens * eos_d.InternalEnergyFromDensityTemperature(dens, T) / eref;
            Real et = ke + eint;
            const Real Cv = dens * eos_d.SpecificHeatFromDensityTemperature(dens, T);
            const Real fleck = FleckFactor(arad, T, Cv);

            const Real sigp = chat * dt * opac_d.PlanckMeanAbsorptionCoefficient(dens, T);
            const Real sigs =
                chat * dt * scat_d.RosselandMeanTotalScatteringCoefficient(dens, T);
            const Real sigf = sigp + sigs;

            const Real ca = g * (sigf - g2 * sigs * (1. + bdbdp));
            const Real cb = g * sigp;
            const Real cd = -g * bdf * (sigf - 2. * g2 * sigs);

            const Real G0 = ca * E - cb * B + cd;
            const Real Fi = (et - et0) - c / chat * G0 - Q;
            const Real Fr = (E - E0) + G0;
            inner_err =
                std::max((std::abs(Fi) / escale), (c / chat * std::abs(Fr) / escale));
            if (!(IsFinite(inner_err) && IsFinite(Fi) && IsFinite(Fr) && IsFinite(G0) &&
                  IsFinite(Cv) && IsFinite(fleck) && Cv > 0.0)) {
              solver_valid = false;
              break;
            }
            if (inner_err <= inner_tol) {
              inner_converged = true;
              break;
            }

            // not converged yet
            const Real dfac = 1. + c / chat * fleck * cb;
            const Real denom = dfac + ca;
            if (!(IsFinite(denom)) || std::abs(denom) <= Fuzz<Real>()) {
              solver_valid = false;
              break;
            }
            const Real dE = dfac / denom * (-Fr) + fleck / denom * (-Fi * cb);
            const Real dB =
                c / chat * fleck / denom * (-ca * Fr) + (1. + ca) * fleck / denom * (-Fi);

            bool accepted = false;
            Real Etrial = E;
            Real Btrial = B;
            for (int ls = 0; ls < 8; ++ls) {
              const Real alpha = std::ldexp(1.0, -ls);
              Etrial = std::max(efloor, E + alpha * dE);
              Btrial = std::max(Bfloor, B + alpha * dB);
              const Real Ttrial =
                  std::max(tfloor, std::pow(std::max(eref * Btrial / arad, 0.0), 0.25));
              const Real eint_trial =
                  dens * eos_d.InternalEnergyFromDensityTemperature(dens, Ttrial) / eref;
              const Real et_trial = ke + eint_trial;
              const Real Cv_trial =
                  dens * eos_d.SpecificHeatFromDensityTemperature(dens, Ttrial);
              if (!(IsFinite(Cv_trial)) || Cv_trial <= 0.0) continue;
              const Real fleck_trial = FleckFactor(arad, Ttrial, Cv_trial);
              const Real sigp_trial =
                  chat * dt * opac_d.PlanckMeanAbsorptionCoefficient(dens, Ttrial);
              const Real sigs_trial =
                  chat * dt *
                  scat_d.RosselandMeanTotalScatteringCoefficient(dens, Ttrial);
              const Real sigf_trial = sigp_trial + sigs_trial;
              const Real ca_trial = g * (sigf_trial - g2 * sigs_trial * (1. + bdbdp));
              const Real cb_trial = g * sigp_trial;
              const Real cd_trial = -g * bdf * (sigf_trial - 2. * g2 * sigs_trial);
              const Real G0_trial = ca_trial * Etrial - cb_trial * Btrial + cd_trial;
              const Real Fi_trial = (et_trial - et0) - c / chat * G0_trial - Q;
              const Real Fr_trial = (Etrial - E0) + G0_trial;
              const Real err_trial = std::max(std::abs(Fi_trial) / escale,
                                              c / chat * std::abs(Fr_trial) / escale);
              if (IsFinite(err_trial) && err_trial < inner_err) {
                E = Etrial;
                B = Btrial;
                accepted = true;
                break;
              }
            }
            if (!accepted) {
              solver_valid = false;
              break;
            }

          } // inner_iter
          if (!solver_valid) break;
          if (!inner_converged && inner_err <= inner_practical_tol) {
            inner_converged = true;
          }
          if (!inner_converged) {
            if (fatal_if_unconverged) {
              printf("MatterCoupling inner fail (%d,%d,%d,%d): err=%lg tol=%lg "
                     "practical_tol=%lg E=%lg B=%lg |F|/E=%lg Q=%lg dens=%lg after %d "
                     "iterations\n",
                     b, k, j, i, inner_err, inner_tol, inner_practical_tol, E, B,
                     std::sqrt(SQR(F[0]) + SQR(F[1]) + SQR(F[2])) /
                         (std::max(E, efloor) + Fuzz<Real>()),
                     Q, dens, inner_iter - 1);
              PARTHENON_FAIL("Inner not converged");
            }
            break;
          }

          // Have new E and T

          T = std::max(tfloor, std::pow(std::max(eref * B / arad, 0.0), 0.25));
          Real eg = dens * eos_d.InternalEnergyFromDensityTemperature(dens, T) / eref;
          dEg = eg - eg0;

          const Real sigp =
              chat * dt * opac_d.RosselandMeanAbsorptionCoefficient(dens, T);
          const Real sigs =
              chat * dt * scat_d.RosselandMeanTotalScatteringCoefficient(dens, T);
          const Real sigf = sigp + sigs;

          const Real a = g * sigf;
          const Real b = 2. * g2 * g * sigs;
          const Real d1 = g * (sigp * B + g2 * sigs * (1. + bdbdp) * E); // * c
          const Real d2 = g * sigf * E;                                  // * c
          const std::array<Real, 3> rhs{Fr0[0] + d1 * beta[0] + d2 * bdp[0],
                                        Fr0[1] + d1 * beta[1] + d2 * bdp[1],
                                        Fr0[2] + d1 * beta[2] + d2 * bdp[2]};

          F = SolveRadFlux(1. + a, b, beta, rhs);
          F = ProjectFlux(F, std::max(E, efloor));

          for (int d = 0; d < 3; d++) {
            dF[d] = F[d] - Fr0[d];
            dv[d] = -icc * dF[d] * fref;
            v[d] = p0[d] / dens + dv[d];
          }

          const Real dEk_prev = dEk;
          dEk = 0.5 * dens *
                (dv[0] * (v[0] + p0[0] / dens) + dv[1] * (v[1] + p0[1] / dens) +
                 dv[2] * (v[2] + p0[2] / dens)) /
                eref;
          E = std::max(efloor, E0 - chat / c * (dEg + dEk));
          dEr = E - E0;
          F = ProjectFlux(F, std::max(E, efloor));
          for (int d = 0; d < 3; ++d)
            dF[d] = F[d] - Fr0[d];
          escale = std::max(et0 + c / chat * E + std::abs(Q), efloor + Bfloor);
          const Real dFmag = std::sqrt(SQR(F[0] - F_prev[0]) + SQR(F[1] - F_prev[1]) +
                                       SQR(F[2] - F_prev[2]));
          outer_err = std::max(std::abs(E - E_prev),
                               std::max(std::abs(B - B_prev),
                                        std::max(std::abs(dEk - dEk_prev), dFmag))) /
                      escale;
          solver_valid =
              IsFinite(outer_err) && IsFinite(dEg) && IsFinite(dEk) && IsFinite(dEr);
          if (!solver_valid) break;
          have_good_state = true;
          best_dEg = dEg;
          best_dEk = dEk;
          best_dEr = dEr;
          best_dF = dF;
          best_dv = dv;
          if (outer_err <= outer_tol) {
            outer_converged = true;
            break;
          }

        } // outer_iter
        if (!outer_converged) {
          if (have_good_state) {
            dEg = best_dEg;
            dEk = best_dEk;
            dEr = best_dEr;
            dF = best_dF;
            dv = best_dv;
          } else {
            // No valid outer iterate was ever accepted. Fall back to a purely local
            // heating update rather than aborting on a floor-adjacent or otherwise
            // ill-conditioned cell.
            dEg = Q;
            dEk = 0.0;
            dEr = 0.0;
            dF = {0.0, 0.0, 0.0};
            dv = {0.0, 0.0, 0.0};
          }
        }

        // Update state vector (both gas and radiation)
        v0(b, gas::cons::internal_energy(), k, j, i) += dEg * eref;
        v0(b, gas::cons::total_energy(), k, j, i) += (dEg + dEk) * eref;
        v0(b, rad::cons::energy(), k, j, i) += dEr * eref;
        v0(b, gas::cons::momentum(0), k, j, i) += dv[0] * dens * hx[0];
        v0(b, gas::cons::momentum(1), k, j, i) += dv[1] * dens * hx[1];
        v0(b, gas::cons::momentum(2), k, j, i) += dv[2] * dens * hx[2];
        v0(b, rad::cons::flux(0), k, j, i) += dF[0] * hx[0] * fref;
        v0(b, rad::cons::flux(1), k, j, i) += dF[1] * hx[1] * fref;
        v0(b, rad::cons::flux(2), k, j, i) += dF[2] * hx[2] * fref;
      });

  return TaskStatus::complete;
}

} // namespace Moments

#endif // RADIATION_MOMENTS_MATTER_COUPLING_HPP_
