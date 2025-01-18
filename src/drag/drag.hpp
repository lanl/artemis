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
#ifndef DRAG_DRAG_HPP_
#define DRAG_DRAG_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/diffusion/diffusion_coeff.hpp"
#include "utils/eos/eos.hpp"
#include "utils/units.hpp"

using namespace parthenon::package::prelude;
using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace Drag {

/*
  <physics>
  do_drag = true

  <gas>

  <gas/damping>
    inner_x1 = ..
    inner_x1_rate = ...

  <dust>

  <dust/damping>
    inner_x1 = ...

  <dust/stopping_time>
    type = constant  # constant, stokes
    tau = 1e-8

  <drag>
   type = simple_dust  # simple_dust, self

*/

enum class Coupling { simple_dust, self, null };
enum class DragModel { constant, stokes, dp15, null };

inline Coupling ChooseDrag(const std::string choice) {
  if (choice == "self") {
    return Coupling::self;
  } else if (choice == "simple_dust") {
    return Coupling::simple_dust;
  }
  PARTHENON_FAIL("Bad choice of drag type");
  return Coupling::null;
}

struct SelfDragParams {

  Real ix[3], ox[3];
  Real xmin[3], xmax[3];
  Real irate[3], orate[3];
  bool damp_to_visc;

  SelfDragParams() {
    for (int i = 0; i < 3; i++) {
      ix[i] = 0.;
      ox[i] = 0.;
      ix[i] = -Big<Real>();
      ox[i] = Big<Real>();
      irate[i] = 0.0;
      orate[i] = 0.0;
    }
    damp_to_visc = false;
  }
  SelfDragParams(std::string block_name, ParameterInput *pin) {
    ix[0] = pin->GetOrAddReal(block_name, "inner_x1", -Big<Real>());
    ix[1] = pin->GetOrAddReal(block_name, "inner_x2", -Big<Real>());
    ix[2] = pin->GetOrAddReal(block_name, "inner_x3", -Big<Real>());
    irate[0] = pin->GetOrAddReal(block_name, "inner_x1_rate", 0.0);
    irate[1] = pin->GetOrAddReal(block_name, "inner_x2_rate", 0.0);
    irate[2] = pin->GetOrAddReal(block_name, "inner_x3_rate", 0.0);

    ox[0] = pin->GetOrAddReal(block_name, "outer_x1", Big<Real>());
    ox[1] = pin->GetOrAddReal(block_name, "outer_x2", Big<Real>());
    ox[2] = pin->GetOrAddReal(block_name, "outer_x3", Big<Real>());
    orate[0] = pin->GetOrAddReal(block_name, "outer_x1_rate", 0.0);
    orate[1] = pin->GetOrAddReal(block_name, "outer_x2_rate", 0.0);
    orate[2] = pin->GetOrAddReal(block_name, "outer_x3_rate", 0.0);
    damp_to_visc = pin->GetOrAddBoolean(block_name, "damp_to_visc", false);

    for (int i = 0; i < 3; i++) {
      PARTHENON_REQUIRE(irate[i] >= 0.0,
                        "The damping rate in the x1 direction must be >= 0");
      PARTHENON_REQUIRE(ix[i] <= ox[i],
                        "The damping bounds must have inner_x1 <= outer_x1");
    }
  }
};

struct StoppingTimeParams {

  Real scale, dh, mass_scale, p1, p2, p3;
  DragModel model;
  ParArray1D<Real> tau;
  StoppingTimeParams(std::string block_name, ParameterInput *pin,
                     ArtemisUtils::Constants &constants, ArtemisUtils::Units &units) {
    const std::string choice = pin->GetString(block_name, "type");
    const int nd = pin->GetOrAddInteger("dust", "nspecies", 1);
    tau = ParArray1D<Real>("tau", nd);
    if (choice == "constant") {
      model = DragModel::constant;
      scale = pin->GetOrAddReal(block_name, "scale", 1.0);
      std::vector<Real> taus = pin->GetVector<Real>(block_name, "tau");
      auto h_tau = tau.GetHostMirror();
      for (int n = 0; n < nd; n++) {
        h_tau(n) = scale * taus[n];
      }
      tau.DeepCopy(h_tau);
    } else if (choice == "stokes") {
      // tau = rho_s/rho_g size / v_th ,  vth^2 = 8/pi R*T

      model = DragModel::stokes;
      scale = pin->GetOrAddReal(block_name, "scale", 1.0);
      auto h_tau = tau.GetHostMirror();
      for (int n = 0; n < nd; n++) {
        h_tau(n) = scale;
      }
      tau.DeepCopy(h_tau);
    } else if (choice == "dp15") {
      model = DragModel::dp15;
      scale = pin->GetOrAddReal(block_name, "scale", 1.0);
      auto h_tau = tau.GetHostMirror();
      for (int n = 0; n < nd; n++) {
        h_tau(n) = scale;
      }
      tau.DeepCopy(h_tau);
      dh = units.GetLengthPhysicalToCode() *
           pin->GetOrAddReal(block_name, "gas_diameter", 2.71e-8 /* cm */);
      p1 = pin->GetOrAddReal(block_name, "p1", 3.07);
      p2 = pin->GetOrAddReal(block_name, "p2", 0.6688);
      p3 = pin->GetOrAddReal(block_name, "p3", 0.681);
      // save this so that we don't have to pass units and constants struct to the Get()
      // routines
      mass_scale = units.GetMassPhysicalToCode() * constants.GetAMUCode();

    } else {
      PARTHENON_FAIL("bad type for stopping time model");
    }
  }
};

template <DragModel DTYP>
class DragCoeff {
 public:
  KOKKOS_INLINE_FUNCTION Real Get(const StoppingTimeParams &dp, const int id,
                                  const Real dg, const Real Tg, const Real u,
                                  const Real grain_density, const Real size,
                                  const EOS &eos) const {
    PARTHENON_FAIL("No default implementation for drag coefficient");
  }
};

// null
template <>
class DragCoeff<DragModel::null> {
 public:
  KOKKOS_INLINE_FUNCTION Real Get(const StoppingTimeParams &dp, const int id,
                                  const Real dg, const Real Tg, const Real u,
                                  const Real grain_density, const Real size,
                                  const EOS &eos) const {
    return Big<Real>();
  }
};

template <>
class DragCoeff<DragModel::constant> {
 public:
  KOKKOS_INLINE_FUNCTION Real Get(const StoppingTimeParams &dp, const int id,
                                  const Real dg, const Real Tg, const Real u,
                                  const Real grain_density, const Real size,
                                  const EOS &eos) const {
    return dp.tau(id);
  }
};

template <>
class DragCoeff<DragModel::stokes> {
 public:
  KOKKOS_INLINE_FUNCTION Real Get(const StoppingTimeParams &dp, const int id,
                                  const Real dg, const Real Tg, const Real u,
                                  const Real grain_density, const Real size,
                                  const EOS &eos) const {
    const Real cv = eos.SpecificHeatFromDensityTemperature(dg, Tg);
    const Real gm1 = eos.GruneisenParamFromDensityTemperature(dg, Tg);
    //  kb T/ mu = cv * gm1 * T
    const Real vth = std::sqrt(8.0 / M_PI * gm1 * cv * Tg);
    return dp.tau(id) * grain_density / dg * size / vth;
  }
};

template <>
class DragCoeff<DragModel::dp15> {
 public:
  KOKKOS_INLINE_FUNCTION Real Get(const StoppingTimeParams &dp, const int id,
                                  const Real dg, const Real Tg, const Real u,
                                  const Real grain_density, const Real size,
                                  const EOS &eos) const {
    const Real cv = eos.SpecificHeatFromDensityTemperature(dg, Tg);
    const Real gm1 = eos.GruneisenParamFromDensityTemperature(dg, Tg);
    const Real mu = dp.mass_scale * eos.MeanAtomicMass();
    const Real gamma = gm1 + 1.;
    const Real sgam = std::sqrt(gamma);
    //  kb T/ mu = cv * gm1 * T

    const Real vth = std::sqrt(8.0 / M_PI * gm1 * cv * Tg);
    // Mach and Reynolds numbers
    // Mu is non-zero, M can be zero
    const Real Mu = std::sqrt(8. / (M_PI * gamma)) / vth;
    const Real M = u * Mu;
    const Real K = 5. / (32. * std::sqrt(M_PI)) * mu / SQR(dp.dh) / (dg * size * sgam);
    // Ru is non-zero, R can be zero
    const Real Ru = Mu / K;
    const Real R = Ru * u;

    // CD = 2 + (CS - 2) * exp(-p1 sqrt(g) K * G(R)) + CE * exp(-1/(2*K))

    // The two coefficients (multiplied by u)
    const Real CEu = 1. / (sgam * Mu) * (4.6 / (1. + M) + 1.7 /* srt(Td/Tg) */);
    const Real CSu =
        24. / Ru * (1. + 0.15 * std::pow(R, dp.p3)) + 0.407 * R * u / (R + 8710.);

    // weight functions
    const Real x = std::pow(R / 312., dp.p2);
    const Real G = std::exp(2.5 * x / (1. + x));
    const Real ws = std::exp(-dp.p1 * sgam * K * G);
    const Real we = std::exp(-1. / (2. * K));

    // The final stopping time
    const Real CDu = 2 * u * (1. - ws) + CSu * ws + CEu * we;
    const Real alpha = 3. / 8 * CDu * dg / (grain_density * size);
    return dp.tau(id) / (alpha + Fuzz<Real>());
  }
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Constants &constants,
                                            ArtemisUtils::Units &units);

template <Coordinates GEOM>
TaskStatus DragSource(MeshData<Real> *md, const Real time, const Real dt);

template <Diffusion::DiffType DTYP, Coordinates GEOM>
TaskStatus SelfDragSourceImpl(MeshData<Real> *md, const Real time, const Real dt,
                              const Diffusion::DiffCoeffParams &dp, const EOS &eos_d,
                              const SelfDragParams &gasp, const SelfDragParams &dustp) {
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  auto &drag_pkg = pm->packages.Get("drag");

  Real de_switch = Null<Real>();
  Real dflr_gas = Null<Real>();
  Real sieflr_gas = Null<Real>();
  if (do_gas) {
    auto &gas_pkg = pm->packages.Get("gas");
    de_switch = gas_pkg->template Param<Real>("de_switch");
    dflr_gas = gas_pkg->template Param<Real>("dfloor");
    sieflr_gas = gas_pkg->template Param<Real>("siefloor");
  }

  const int ndim = pm->ndim;

  const Real x1min = drag_pkg->template Param<Real>("x1min");
  const Real x1max = drag_pkg->template Param<Real>("x1max");

  const Real x2min = drag_pkg->template Param<Real>("x2min");
  const Real x2max = drag_pkg->template Param<Real>("x2max");

  const Real x3min = drag_pkg->template Param<Real>("x3min");
  const Real x3max = drag_pkg->template Param<Real>("x3max");

  const int multi_d = (ndim >= 2);
  const int three_d = (ndim == 3);

  static auto desc =
      MakePackDescriptor<gas::cons::total_energy, gas::cons::momentum, gas::cons::density,
                         gas::cons::internal_energy, dust::cons::momentum,
                         dust::cons::density>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "VelocityDrag", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &hx = coords.GetScaleFactors();
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // Compute the ramp for this cell
        // Ramps are quadratic, eg. the left regions is SQR( (X - ix)/(ix - xmin) )
        if (do_gas) {
          const Real fx1 =
              dt * (gasp.irate[0] * ((xv[0] < gasp.ix[0]) *
                                     SQR((xv[0] - gasp.ix[0]) / (gasp.ix[0] - x1min))) +
                    gasp.orate[0] * ((xv[0] > gasp.ox[0]) *
                                     SQR((xv[0] - gasp.ox[0]) / (gasp.ox[0] - x1max))));
          const Real fx2 =
              multi_d * dt *
              (gasp.irate[1] * ((xv[1] < gasp.ix[1]) *
                                SQR((xv[1] - gasp.ix[1]) / (gasp.ix[1] - x2min))) +
               gasp.orate[1] * ((xv[1] > gasp.ox[1]) *
                                SQR((xv[1] - gasp.ox[1]) / (gasp.ox[1] - x2max))));
          const Real fx3 =
              three_d * dt *
              (gasp.irate[2] * ((xv[2] < gasp.ix[2]) *
                                SQR((xv[2] - gasp.ix[2]) / (gasp.ix[2] - x3min))) +
               gasp.orate[2] * ((xv[2] > gasp.ox[2]) *
                                SQR((xv[2] - gasp.ox[2]) / (gasp.ox[2] - x3max))));
          for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {
            const Real &dens = vmesh(b, gas::cons::density(n), k, j, i);
            const Real vg[3] = {
                vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) / (hx[0] * dens),
                vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) / (hx[1] * dens),
                vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) / (hx[2] * dens)};

            const Real sieg = ArtemisUtils::GetSpecificInternalEnergy(
                vmesh, b, n, k, j, i, de_switch, dflr_gas, sieflr_gas, hx);

            Real vd[3] = {0., 0., 0.};

            Diffusion::DiffusionCoeff<DTYP, GEOM, Fluid::gas> dcoeff;
            const Real mu = dcoeff.Get(dp, coords, dens, sieg, eos_d);
            const Real vR = -1.5 * mu / (xcyl[0] * dens);
            vd[0] = ex1[0] * vR;
            vd[1] = ex2[0] * vR;
            vd[2] = ex3[0] * vR;

            // Ep - E = 0.5 d ( vp^2 - v^2 )
            //  (vp-v) . (vp + v) = dv . (2v + dv) =  2 dv.v + dv.dv
            const Real dm1 = -fx1 * dens * (vg[0] - vd[0]) / (1.0 + fx1);
            const Real dm2 = -fx2 * dens * (vg[1] - vd[1]) / (1.0 + fx2);
            const Real dm3 = -fx3 * dens * (vg[2] - vd[2]) / (1.0 + fx3);
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += hx[0] * dm1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += hx[1] * dm2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += hx[2] * dm3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                dm1 * (vg[0] + 0.5 * dm1 / dens) + dm2 * (vg[1] + 0.5 * dm2 / dens) +
                dm3 * (vg[2] + 0.5 * dm3 / dens);
          }
        }
        if (do_dust) {
          const Real fx1 =
              dt *
              (dustp.irate[0] * ((xv[0] < dustp.ix[0]) *
                                 SQR((xv[0] - dustp.ix[0]) / (dustp.ix[0] - x1min))) +
               dustp.orate[0] * ((xv[0] > dustp.ox[0]) *
                                 SQR((xv[0] - dustp.ox[0]) / (dustp.ox[0] - x1max))));
          const Real fx2 =
              multi_d * dt *
              (dustp.irate[1] * ((xv[1] < dustp.ix[1]) *
                                 SQR((xv[1] - dustp.ix[1]) / (dustp.ix[1] - x2min))) +
               dustp.orate[1] * ((xv[1] > dustp.ox[1]) *
                                 SQR((xv[1] - dustp.ox[1]) / (dustp.ox[1] - x2max))));
          const Real fx3 =
              three_d * dt *
              (dustp.irate[2] * ((xv[2] < dustp.ix[2]) *
                                 SQR((xv[2] - dustp.ix[2]) / (dustp.ix[2] - x3min))) +
               dustp.orate[2] * ((xv[2] > dustp.ox[2]) *
                                 SQR((xv[2] - dustp.ox[2]) / (dustp.ox[2] - x3max))));
          for (int n = 0; n < vmesh.GetSize(b, dust::cons::density()); ++n) {
            const Real &dens = vmesh(b, dust::cons::density(n), k, j, i);
            const Real mom[3] = {vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i),
                                 vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i),
                                 vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i)};
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) -=
                fx1 * mom[0] / (1.0 + fx1);
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) -=
                fx2 * mom[1] / (1.0 + fx2);
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) -=
                fx3 * mom[2] / (1.0 + fx3);
          }
        }
      });

  return TaskStatus::complete;
}

template <Diffusion::DiffType DTYP, DragModel DRAG, Coordinates GEOM>
TaskStatus SimpleDragSourceImpl(MeshData<Real> *md, const Real time, const Real dt,
                                const Diffusion::DiffCoeffParams &dp, const EOS &eos_d,
                                const SelfDragParams &gasp, const SelfDragParams &dustp,
                                const StoppingTimeParams &tp) {
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  auto &drag_pkg = pm->packages.Get("drag");
  auto &dust_pkg = pm->packages.Get("dust");
  auto &gas_pkg = pm->packages.Get("gas");
  const Real de_switch = gas_pkg->template Param<Real>("de_switch");
  const Real dflr_gas = gas_pkg->template Param<Real>("dfloor");
  const Real sieflr_gas = gas_pkg->template Param<Real>("siefloor");

  const int ndim = pm->ndim;

  const Real x1min = drag_pkg->template Param<Real>("x1min");
  const Real x1max = drag_pkg->template Param<Real>("x1max");

  const Real x2min = drag_pkg->template Param<Real>("x2min");
  const Real x2max = drag_pkg->template Param<Real>("x2max");

  const Real x3min = drag_pkg->template Param<Real>("x3min");
  const Real x3max = drag_pkg->template Param<Real>("x3max");

  const auto &sizes = dust_pkg->template Param<ParArray1D<Real>>("sizes");
  const auto grain_density = dust_pkg->template Param<Real>("grain_density");

  const int multi_d = (ndim >= 2);
  const int three_d = (ndim == 3);

  static auto desc =
      MakePackDescriptor<gas::cons::total_energy, gas::cons::momentum, gas::cons::density,
                         gas::cons::internal_energy, dust::cons::momentum,
                         dust::cons::density>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "VelocityDrag", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &hx = coords.GetScaleFactors();
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // Compute the ramp for this cell
        // Ramps are quadratic, eg. the left regions is SQR( (X - ix)/(ix - xmin) )
        const Real bg[3] = {
            dt * (gasp.irate[0] * ((xv[0] < gasp.ix[0]) *
                                   SQR((xv[0] - gasp.ix[0]) / (gasp.ix[0] - x1min))) +
                  gasp.orate[0] * ((xv[0] > gasp.ox[0]) *
                                   SQR((xv[0] - gasp.ox[0]) / (gasp.ox[0] - x1max)))),
            multi_d * dt *
                (gasp.irate[1] * ((xv[1] < gasp.ix[1]) *
                                  SQR((xv[1] - gasp.ix[1]) / (gasp.ix[1] - x2min))) +
                 gasp.orate[1] * ((xv[1] > gasp.ox[1]) *
                                  SQR((xv[1] - gasp.ox[1]) / (gasp.ox[1] - x2max)))),
            three_d * dt *
                (gasp.irate[2] * ((xv[2] < gasp.ix[2]) *
                                  SQR((xv[2] - gasp.ix[2]) / (gasp.ix[2] - x3min))) +
                 gasp.orate[2] * ((xv[2] > gasp.ox[2]) *
                                  SQR((xv[2] - gasp.ox[2]) / (gasp.ox[2] - x3max))))};
        const Real bd[3] = {
            dt * (dustp.irate[0] * ((xv[0] < dustp.ix[0]) *
                                    SQR((xv[0] - dustp.ix[0]) / (dustp.ix[0] - x1min))) +
                  dustp.orate[0] * ((xv[0] > dustp.ox[0]) *
                                    SQR((xv[0] - dustp.ox[0]) / (dustp.ox[0] - x1max)))),
            multi_d * dt *
                (dustp.irate[1] * ((xv[1] < dustp.ix[1]) *
                                   SQR((xv[1] - dustp.ix[1]) / (dustp.ix[1] - x2min))) +
                 dustp.orate[1] * ((xv[1] > dustp.ox[1]) *
                                   SQR((xv[1] - dustp.ox[1]) / (dustp.ox[1] - x2max)))),
            three_d * dt *
                (dustp.irate[2] * ((xv[2] < dustp.ix[2]) *
                                   SQR((xv[2] - dustp.ix[2]) / (dustp.ix[2] - x3min))) +
                 dustp.orate[2] * ((xv[2] > dustp.ox[2]) *
                                   SQR((xv[2] - dustp.ox[2]) / (dustp.ox[2] - x3max))))};

        const Real &dg = vmesh(b, gas::cons::density(0), k, j, i);
        const Real vg[3] = {
            vmesh(b, gas::cons::momentum(VI(0, 0)), k, j, i) / (hx[0] * dg),
            vmesh(b, gas::cons::momentum(VI(0, 1)), k, j, i) / (hx[1] * dg),
            vmesh(b, gas::cons::momentum(VI(0, 2)), k, j, i) / (hx[2] * dg)};

        const Real sieg = ArtemisUtils::GetSpecificInternalEnergy(
            vmesh, b, 0, k, j, i, de_switch, dflr_gas, sieflr_gas, hx);

        // Target gas velocity
        Diffusion::DiffusionCoeff<DTYP, GEOM, Fluid::gas> dcoeff;
        const Real mu = dcoeff.Get(dp, coords, dg, sieg, eos_d);
        const Real vR = -1.5 * mu / (xcyl[0] * dg);
        const Real vt[3] = {ex1[0] * vR, ex2[0] * vR, ex3[0] * vR};

        Real fd[3] = {0.};
        Real fvd[3] = {0.};
        const auto nspecies = vmesh.GetSize(b, dust::cons::density());

        DragCoeff<DRAG> drag_coeff;
        Real Tg = eos_d.TemperatureFromDensityInternalEnergy(dg, sieg);

        // First pass to collect \sum rho' and \sum rho' v and compute new vg
        const Real vdt[3] = {0.0};
        for (int n = 0; n < nspecies; ++n) {
          const auto id = vmesh(b, dust::cons::density(n)).sparse_id;
          const Real &dens = vmesh(b, dust::cons::density(n), k, j, i);
          const Real vd[3] = {
              vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) / (hx[0] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) / (hx[1] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) / (hx[2] * dens)};

          // relative speed
          const Real u =
              std::sqrt(SQR(vd[0] - vg[0]) + SQR(vd[1] - vg[1]) + SQR(vd[2] - vg[2]));

          // Get the stopping time
          Real tc = drag_coeff.Get(tp, id, dg, Tg, u, grain_density, sizes(id), eos_d);

          const Real alpha = dt * ((tc <= 0.0) ? Big<Real>() : 1.0 / tc);
          for (int d = 0; d < 3; d++) {
            const Real rhop = dens * alpha / (1.0 + alpha + bd[d]);
            fd[d] += rhop * (1.0 + bd[d]);
            fvd[d] += rhop * (vd[d] + bd[d] * vdt[d]);
          }
        }
        // New vgas
        Real vgp[3] = {Null<Real>()};
        for (int d = 0; d < 3; d++) {
          vgp[d] = (dg * (vg[d] + bg[d] * vt[d]) + fvd[d]) / (dg * (1.0 + bg[d]) + fd[d]);
        }

        // Second pass to update all momenta

        // Total gas momentum change
        Real delta_g[3] = {0.0};
        for (int d = 0; d < 3; d++) {
          fvd[d] = 0.;
        }
        for (int n = 0; n < nspecies; ++n) {
          const auto id = vmesh(b, dust::cons::density(n)).sparse_id;
          const Real &dens = vmesh(b, dust::cons::density(n), k, j, i);
          const Real vd[3] = {
              vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) / (hx[0] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) / (hx[1] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) / (hx[2] * dens)};
          // relative speed
          const Real u =
              std::sqrt(SQR(vd[0] - vg[0]) + SQR(vd[1] - vg[1]) + SQR(vd[2] - vg[2]));

          // Get the stopping time
          Real tc = drag_coeff.Get(tp, id, dg, Tg, u, grain_density, sizes(id), eos_d);

          const Real alpha = dt * ((tc <= 0.0) ? Big<Real>() : 1.0 / tc);
          for (int d = 0; d < 3; d++) {
            Real delta_d = 0.;
            const Real rhop = dens * alpha / (1.0 + alpha + bd[d]);
            const Real delta = rhop * ((vgp[d] - vd[d] + bd[d] * (vgp[d] - vdt[d])));
            delta_d += delta;
            delta_g[d] -= delta;

            // self-drag coupling
            delta_d -= bd[d] * dens / (1. + alpha + bd[d]) *
                       (vd[d] - vdt[d] + alpha * (vgp[d] - vdt[d]));
            fvd[d] += rhop * (vd[d] - vt[d] + bd[d] * (vdt[d] - vt[d]));
            // Update dust momenta
            vmesh(b, dust::cons::momentum(VI(n, d)), k, j, i) += hx[d] * delta_d;
          }
        }
        for (int d = 0; d < 3; d++) {
          const Real prefac = dg * bg[d] / (1.0 + bg[d] + fd[d]);
          delta_g[d] -= prefac * (dg * (vg[d] - vt[d]) + fvd[d]);
          vmesh(b, gas::cons::momentum(VI(0, d)), k, j, i) += hx[d] * delta_g[d];
          vmesh(b, gas::cons::total_energy(0), k, j, i) +=
              0.5 * (vg[d] + vgp[d]) * delta_g[d];
        }
        // Update gas momenta
      });

  return TaskStatus::complete;
}

} // namespace Drag

#endif // DRAG_DRAG_HPP_
