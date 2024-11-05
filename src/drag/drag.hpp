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
#include <iostream>
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/diffusion/diffusion_coeff.hpp"
#include "utils/eos/eos.hpp"

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
enum class DragModel { constant, stokes, null };

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
    ix[1] = pin->GetOrAddReal(block_name, "pos_z_sh", Big<Real>());
    irate[0] = pin->GetOrAddReal(block_name, "inner_x1_rate", 0.0);
    irate[1] = pin->GetOrAddReal(block_name, "z_rate", 0.0);

    ox[0] = pin->GetOrAddReal(block_name, "outer_x1", Big<Real>());
    ox[1] = pin->GetOrAddReal(block_name, "neg_z_sh", Big<Real>());
    orate[0] = pin->GetOrAddReal(block_name, "outer_x1_rate", 0.0);
    orate[1] = pin->GetOrAddReal(block_name, "z_rate", 0.0);

    damp_to_visc = pin->GetOrAddBoolean(block_name, "damp_to_visc", false);

    for (int i = 0; i < 3; i++) {
      PARTHENON_REQUIRE(irate[i] >= 0.0,
                        "The damping rate in the x1 direction must be >= 0");
      if (i != 2) {
      PARTHENON_REQUIRE(ix[i] <= ox[i],
                        "The damping bounds must have inner_x1 <= outer_x1");
      }
    }
  }
};

struct StoppingTimeParams {

  Real scale;
  DragModel model;
  ParArray1D<Real> tau;
  StoppingTimeParams(std::string block_name, ParameterInput *pin) {
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
    } else {
      PARTHENON_FAIL("bad type for stopping time model");
    }
  }
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

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

  const Real p = drag_pkg->template Param<Real>("dslope");
  const Real q = drag_pkg->template Param<Real>("tslope");
  const Real h0 = drag_pkg->template Param<Real>("h0");
  const Real r0 = drag_pkg->template Param<Real>("r0");
  const Real gm = drag_pkg->template Param<Real>("gm");
  const Real omf = drag_pkg->template Param<Real>("omf");
  const Real nu = drag_pkg->template Param<Real>("nu");
  const Real flare = 0.5 * (1.0 + q);

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

        const Real PI_2 = 1.5707963267948966;

        // Compute the ramp for this cell
        // Ramps are quadratic, eg. the left regions is SQR( (X - ix)/(ix - xmin) )
        if (do_gas) {
          const Real H = xcyl[0] * h0 * std::pow(xcyl[0] / r0, flare);
          const Real fx1 =
              dt * (gasp.irate[0] * ((xv[0] < gasp.ix[0]) *
                                     SQR((xv[0] - gasp.ix[0]) / (gasp.ix[0] - x1min))) +
                    gasp.orate[0] * ((xv[0] > gasp.ox[0]) *
                                     SQR((xv[0] - gasp.ox[0]) / (gasp.ox[0] - x1max))));// + 
                    //gasp.irate[1] * ((xv[0] >= gasp.ix[0]) * (xcyl[2] > gasp.ix[1]*H) * // pos z
                    //                 SQR((xcyl[2] - gasp.ix[1]*H) / (gasp.ix[0]*H - xv[0]*std::cos(x2min)))) +
                    //gasp.orate[1] * ((xv[0] <= gasp.ox[0]) * (xcyl[2] < -gasp.ox[1]*H) * // neg z
                    //                 SQR((xcyl[2] + gasp.ox[1]*H) / (-gasp.ox[0]*H - xv[0]*std::cos(x2max)))));
          const Real fx2 = fx1;
          const Real fx3 = fx1;

          for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {
            const Real &dens = vmesh(b, gas::cons::density(n), k, j, i);
            const Real vg[3] = {
                vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) / (hx[0] * dens),
                vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) / (hx[1] * dens),
                vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) / (hx[2] * dens)};

            const Real sieg = ArtemisUtils::GetSpecificInternalEnergy(
                vmesh, b, n, k, j, i, de_switch, dflr_gas, sieflr_gas, hx);

            const Real OmKmid = std::sqrt(gm / (xcyl[0] * xcyl[0] * xcyl[0]));
            const Real Omg = OmKmid * (1 + 0.5 * SQR(H / xcyl[0]) *
                                               (p + q + 0.5 * q * SQR(xcyl[2] / H)));
            const Real vp = Omg * xcyl[0];
            const Real vR = -nu *
                            (6 * p - 2 * q + 3 + (5 * q + 9) * SQR(xcyl[2] / H)) /
                            (2 * xcyl[0]);

            const Real vcyl[3] = {vR, vp - omf * xcyl[0], 0.};

            const Real vd[3] = {
              ArtemisUtils::VDot(vcyl, ex1),
              ArtemisUtils::VDot(vcyl, ex2),
              ArtemisUtils::VDot(vcyl, ex3)
            };
            //Real vd[3] = {
            //  vR,
            //  0.,
            //  vp - omf * xcyl[0],
            //};

            //if (i==2 && j==2 && k==2) {
            //  std::cout << "(" << i << ", " << j << ", " << k << ")" << std::endl;
            //  std::cout << i << j << k << std::endl;
            //  std::cout << "drag vd:" << std::fixed << std::setprecision(12) << vd[0]
            //            << ", " << vd[1] << ", " << vd[2] << std::endl;
            //  std::cout << "drag fx1:" << fx1 << std::endl;
            //}
            //if (i==2 && j==32 && k==2) {
            //  std::cout << "(" << i << ", " << j << ", " << k << ")" << std::endl;
            //  std::cout << "drag vd:" << std::fixed << std::setprecision(12) << vd[0]
            //            << ", " << vd[1] << ", " << vd[2] << std::endl;
            //  std::cout << "drag fx1:" << fx1 << std::endl;
            //}
            //if (i==64 && j==2 && k==2) {
            //  std::cout << "(" << i << ", " << j << ", " << k << ")" << std::endl;
            //  std::cout << "drag vd:" << std::fixed << std::setprecision(12) << vd[0]
            //            << ", " << vd[1] << ", " << vd[2] << std::endl;
            //  std::cout << "drag fx1:" << fx1 << std::endl;
            //}
            //if (i==64 && j==32 && k==2) {
            //  std::cout << "(" << i << ", " << j << ", " << k << ")" << std::endl;
            //  std::cout << "drag vd:" << std::fixed << std::setprecision(12) << vd[0]
            //            << ", " << vd[1] << ", " << vd[2] << std::endl;
            //  std::cout << "drag fx1:" << fx1 << std::endl;
            //}
            // debugging code

            // std::cout << "drag:" << i << ", " << j << ", " << k << std::endl;

            // Ep - E = 0.5 d ( vp^2 - v^2 )
            //  (vp-v) . (vp + v) = dv . (2v + dv) =  2 dv.v + dv.dv
            const Real dm1 = -fx1 * dens * (vg[0] - vd[0]) / (1.0 + fx1);
            const Real dm2 = -fx2 * dens * (vg[1] - vd[1]) / (1.0 + fx2);
            const Real dm3 = -fx3 * dens * (vg[2] - vd[2]) / (1.0 + fx3);

            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += hx[0] * dm1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += hx[1] * dm2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += hx[2] * dm3;

            //if (i==2 && j==2 && k==2) {
            //  std::cout << std::fixed << std::setprecision(12) << "drag nu: " << nu << std::endl;
            //  std::cout << "gas vel after damping:" 
            //            << std::fixed << std::setprecision(12)
            //            << vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i)/dens << ", "
            //            << vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i)/dens << ", "
            //            << vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i)/dens << std::endl;
            //  std::cout << "drag cc:" << std::fixed << std::setprecision(12) << xv[0]
            //            << ", " << xv[1] << ", " << xv[2] << std::endl;
            //  std::cout << "drag vd:" << std::fixed << std::setprecision(12) << vd[0]
            //            << ", " << vd[1] << ", " << vd[2] << std::endl;
            //  std::cout << "drag vcyl:" << std::fixed << std::setprecision(12) << vcyl[0]
            //            << ", " << vcyl[1] << ", " << vcyl[2] << std::endl;
            //  std::cout << "drag ex1:" << std::fixed << std::setprecision(12)<< ex1[0] 
            //            << ", " << ex1[1] << ", " << ex1[2] << std::endl;
            //}

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

        [[maybe_unused]] auto &grain_density_ = grain_density;
        [[maybe_unused]] Real vth = Null<Real>();

        if constexpr (DRAG == DragModel::stokes) {
          const Real gm1 = eos_d.GruneisenParamFromDensityInternalEnergy(dg, sieg);
          vth = std::sqrt(8.0 / M_PI * gm1 * sieg);
        }

        // First pass to collect \sum rho' and \sum rho' v and compute new vg
        const Real vdt[3] = {0.0};
        for (int n = 0; n < nspecies; ++n) {
          const auto id = vmesh(b, dust::cons::density(n)).sparse_id;
          const Real &dens = vmesh(b, dust::cons::density(n), k, j, i);
          const Real vd[3] = {
              vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) / (hx[0] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) / (hx[1] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) / (hx[2] * dens)};
          Real tc = tp.tau(id);
          [[maybe_unused]] auto &sizes_ = sizes;
          if constexpr (DRAG == DragModel::stokes) {
            tc = tp.scale * grain_density_ / dg * sizes_(id) / vth;
          }
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
          Real tc = tp.tau(id);
          [[maybe_unused]] auto &sizes_ = sizes;
          if constexpr (DRAG == DragModel::stokes) {
            tc = tp.scale * grain_density_ / dg * sizes_(id) / vth;
          }
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
