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
#ifndef DRAG_DRAG_HPP_
#define DRAG_DRAG_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "dust/coagulation/coagulation.hpp"
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

// ... Coupling types
enum class Coupling { simple_dust, self, null };
// ... Drag models
enum class DragModel { constant, stokes, null };

//----------------------------------------------------------------------------------------
//! \fn  Coupling Drag::ChooseDrag
//! \brief Helper function to help select drag coupling type
inline Coupling ChooseDrag(const std::string choice) {
  if (choice == "self") {
    return Coupling::self;
  } else if (choice == "simple_dust") {
    return Coupling::simple_dust;
  } else {
    PARTHENON_FAIL("Bad choice of drag type");
    return Coupling::null;
  }
}

//----------------------------------------------------------------------------------------
//! \struct SelfDragParams
//!
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

//----------------------------------------------------------------------------------------
//! \struct StoppingTimeParams
//!
struct StoppingTimeParams {
  Real scale;
  DragModel model;
  ParArray1D<Real> tau;
  Real tau_max, tau_min;

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
      tau_max = pin->GetOrAddReal(block_name, "maximum", 1e99);
      tau_min = pin->GetOrAddReal(block_name, "minimum", 0.0);
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

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Drag::SelfDragSourceImpl
//! \brief Implementation for self drag
template <Diffusion::DiffType DTYP, Coordinates GEOM>
TaskStatus SelfDragSourceImpl(MeshData<Real> *md, const Real time, const Real dt,
                              const Diffusion::DiffCoeffParams &dp, const EOS &eos_d,
                              const SelfDragParams &gasp, const SelfDragParams &dustp) {
  PARTHENON_INSTRUMENT
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Dimensionality
  const int ndim = pm->ndim;
  const int multi_d = (ndim >= 2);
  const int three_d = (ndim == 3);

  // Extract artemis parameters
  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  // Extract gas parameters
  Real de_switch = Null<Real>();
  Real dflr_gas = Null<Real>();
  Real sieflr_gas = Null<Real>();
  if (do_gas) {
    auto &gas_pkg = pm->packages.Get("gas");
    de_switch = gas_pkg->template Param<Real>("de_switch");
    dflr_gas = gas_pkg->template Param<Real>("dfloor");
    sieflr_gas = gas_pkg->template Param<Real>("siefloor");
  }

  // Extract dust parameters
  Real dflr_dust = Null<Real>();
  if (do_dust) {
    auto &dust_pkg = pm->packages.Get("dust");
    dflr_dust = dust_pkg->template Param<Real>("dfloor");
  }
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  const Real x1min = artemis_pkg->template Param<Real>("x1min");
  const Real x1max = artemis_pkg->template Param<Real>("x1max");
  const Real x2min = artemis_pkg->template Param<Real>("x2min");
  const Real x2max = artemis_pkg->template Param<Real>("x2max");
  const Real x3min = artemis_pkg->template Param<Real>("x3min");
  const Real x3max = artemis_pkg->template Param<Real>("x3max");

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::total_energy, gas::cons::momentum, gas::cons::density,
                         gas::cons::internal_energy, dust::cons::momentum,
                         dust::cons::density>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SelfDrag", parthenon::DevExecSpace(), 0, md->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter(vg, b, k, j, i);
        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);

        // Compute the (gas) ramp for this cell
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

          // Update gas momenta and total energy
          for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {
            // Extract state vector
            Real &dens = vmesh(b, gas::cons::density(n), k, j, i);
            Real &mom1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i);
            Real &etot = vmesh(b, gas::cons::total_energy(n), k, j, i);

            // Apply density floor
            const bool dfloor = (dens > dflr_gas);
            dens = (dfloor)*dens + (!dfloor) * dflr_gas;

            // Compute SIE via dual energy formalism and apply floor
            Real sieg = ArtemisUtils::DualEnergySIE(vmesh, b, n, k, j, i, de_switch, hx);
            const Real efloor = (sieg > sieflr_gas);
            sieg = (efloor)*sieg + (!efloor) * sieflr_gas;

            // Get diffusion coefficient
            Diffusion::DiffusionCoeff<DTYP, GEOM, Fluid::gas> dcoeff;
            const Real mu = dcoeff.Get(dp, coords, xv, dens, sieg, eos_d);
            const Real vR = -1.5 * mu / (xcyl[0] * dens);
            const Real vg[3] = {mom1 / (hx[0] * dens), mom2 / (hx[1] * dens),
                                mom3 / (hx[2] * dens)};
            const Real vd[3] = {ex1[0] * vR, ex2[0] * vR, ex3[0] * vR};

            // Apply update
            // Ep - E = 0.5 d ( vp^2 - v^2 )
            //  (vp-v) . (vp + v) = dv . (2v + dv) =  2 dv.v + dv.dv
            const Real dm1 = -fx1 * dens * (vg[0] - vd[0]) / (1.0 + fx1);
            const Real dm2 = -fx2 * dens * (vg[1] - vd[1]) / (1.0 + fx2);
            const Real dm3 = -fx3 * dens * (vg[2] - vd[2]) / (1.0 + fx3);
            mom1 += hx[0] * dm1;
            mom2 += hx[1] * dm2;
            mom3 += hx[2] * dm3;
            etot += dm1 * (vg[0] + 0.5 * dm1 / dens) + dm2 * (vg[1] + 0.5 * dm2 / dens) +
                    dm3 * (vg[2] + 0.5 * dm3 / dens);

            // Apply total energy floor
            const Real utmp = dens * sieflr_gas;
            etot = std::max(etot, utmp);
          }
        }

        // Compute the (dust) ramp for this cell
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

          // Update dust momenta
          for (int n = 0; n < vmesh.GetSize(b, dust::cons::density()); ++n) {
            // Extract state vector
            Real &dens = vmesh(b, dust::cons::density(n), k, j, i);
            Real &mom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);

            // Apply density floor
            const bool dfloor = (dens > dflr_dust);
            dens = (dfloor)*dens + (!dfloor) * dflr_dust;

            // Apply update
            mom1 -= fx1 * mom1 / (1.0 + fx1);
            mom2 -= fx2 * mom2 / (1.0 + fx2);
            mom3 -= fx3 * mom3 / (1.0 + fx3);
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Drag::SimpleDragSourceImpl
//! \brief Implementation for simple drag
template <Diffusion::DiffType DTYP, DragModel DRAG, Coordinates GEOM>
TaskStatus SimpleDragSourceImpl(MeshData<Real> *md, const Real time, const Real dt,
                                const Diffusion::DiffCoeffParams &dp, const EOS &eos_d,
                                const SelfDragParams &gasp, const SelfDragParams &dustp,
                                const StoppingTimeParams &tp) {
  PARTHENON_INSTRUMENT
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Dimensionality
  const int ndim = pm->ndim;
  const int multi_d = (ndim >= 2);
  const int three_d = (ndim == 3);
  auto &artemis_pkg = pm->packages.Get("artemis");

  // Extract gas package and params
  auto &gas_pkg = pm->packages.Get("gas");
  const Real de_switch = gas_pkg->template Param<Real>("de_switch");
  const Real dflr_gas = gas_pkg->template Param<Real>("dfloor");
  const Real sieflr_gas = gas_pkg->template Param<Real>("siefloor");

  const Real x1min = artemis_pkg->template Param<Real>("x1min");
  const Real x1max = artemis_pkg->template Param<Real>("x1max");
  const Real x2min = artemis_pkg->template Param<Real>("x2min");
  const Real x2max = artemis_pkg->template Param<Real>("x2max");
  const Real x3min = artemis_pkg->template Param<Real>("x3min");
  const Real x3max = artemis_pkg->template Param<Real>("x3max");

  // Extract dust package and params
  auto &dust_pkg = pm->packages.Get("dust");
  const auto &sizes = dust_pkg->template Param<ParArray1D<Real>>("sizes");
  const auto grain_density = dust_pkg->template Param<Real>("grain_density");
  const Real dflr_dust = dust_pkg->template Param<Real>("dfloor");

  // Optional within-bin size reconstruction (Phase 4): when coagulation is enabled
  // and configured with bin_recon != Constant, the Stokes stopping time uses the
  // bin-averaged effective size <a>_i implied by the PL profile rather than the
  // point value sizes(id). When coagulation is off, we leave the legacy point-size
  // behavior intact via a Constant sentinel and zero-length ParArray1D guards.
  bool drag_use_bin_recon = false;
  Dust::Coagulation::BinRecon drag_bin_recon = Dust::Coagulation::BinRecon::Constant;
  ParArray1D<Real> drag_log_widths;
  ParArray1D<Real> drag_mass_grid;
  if (artemis_pkg->template Param<bool>("do_coagulation")) {
    auto &coag_pkg = pm->packages.Get("coagulation");
    const auto &coag_pars =
        coag_pkg->template Param<Dust::Coagulation::CoagParams>("coag_pars");
    if (coag_pars.bin_recon != Dust::Coagulation::BinRecon::Constant) {
      const auto &coag_arrs =
          coag_pkg->template Param<Dust::Coagulation::CoagArrays>("coag_arrs");
      drag_use_bin_recon = true;
      drag_bin_recon = coag_pars.bin_recon;
      drag_log_widths = coag_arrs.log_widths;
      drag_mass_grid = coag_arrs.mass_grid;
    }
  }

  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::total_energy, gas::cons::momentum, gas::cons::density,
                         gas::cons::internal_energy, dust::cons::momentum,
                         dust::cons::density>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  static auto desc_g = MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::hx1v,
                                          geom::hx2v, geom::hx3v>(resolved_pkgs.get());
  auto vg = desc_g.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SimpleDrag", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Bin-recon aliases (suffixed to mirror the sizes_/grain_density_ convention
        // used below for [[maybe_unused]] guarded captures).
        [[maybe_unused]] const bool drag_use_bin_recon_ = drag_use_bin_recon;
        [[maybe_unused]] auto &drag_log_widths_ = drag_log_widths;
        [[maybe_unused]] auto &drag_mass_grid_ = drag_mass_grid;
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter(vg, b, k, j, i);
        const auto &hx = coords.GetScaleFactors(vg, b, k, j, i);
        const auto &[xcyl, ex1, ex2, ex3] = coords.ConvertToCylWithVec(xv);
        const std::array<Real, 3> bg{
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
        const std::array<Real, 3> bd{
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

        // Extract gas state vector
        // NOTE(@pdmullen): Assumes single gas species
        Real &dg = vmesh(b, gas::cons::density(0), k, j, i);
        Real &gmom1 = vmesh(b, gas::cons::momentum(VI(0, 0)), k, j, i);
        Real &gmom2 = vmesh(b, gas::cons::momentum(VI(0, 1)), k, j, i);
        Real &gmom3 = vmesh(b, gas::cons::momentum(VI(0, 2)), k, j, i);
        Real &etot = vmesh(b, gas::cons::total_energy(0), k, j, i);

        // Apply density floor
        const bool dfloor = (dg > dflr_gas);
        dg = (dfloor)*dg + (!dfloor) * dflr_gas;

        // Compute SIE via dual energy formalism and apply floor
        Real sieg = ArtemisUtils::DualEnergySIE(vmesh, b, 0, k, j, i, de_switch, hx);
        const Real efloor = (sieg > sieflr_gas);
        sieg = (efloor)*sieg + (!efloor) * sieflr_gas;

        // Stash gas velocity
        const std::array<Real, 3> vg{gmom1 / (hx[0] * dg), gmom2 / (hx[1] * dg),
                                     gmom3 / (hx[2] * dg)};

        // Target gas velocity
        Diffusion::DiffusionCoeff<DTYP, GEOM, Fluid::gas> dcoeff;
        const Real mu = dcoeff.Get(dp, coords, xv, dg, sieg, eos_d);
        const Real vR = -1.5 * mu / (xcyl[0] * dg);
        const std::array<Real, 3> vt{ex1[0] * vR, ex2[0] * vR, ex3[0] * vR};

        // Extract Stokes specific parameters
        [[maybe_unused]] auto &grain_density_ = grain_density;
        [[maybe_unused]] Real vth = Null<Real>();
        if constexpr (DRAG == DragModel::stokes) {
          const Real gm1 = eos_d.GruneisenParamFromDensityInternalEnergy(dg, sieg);
          vth = std::sqrt(8.0 / M_PI * gm1 * sieg);
        }

        // First pass to collect \sum rho' and \sum rho' v and compute new vg
        std::array<Real, 3> fd{0.0, 0.0, 0.0};
        std::array<Real, 3> fvd{0.0, 0.0, 0.0};
        std::array<Real, 3> vdt{0.0, 0.0, 0.0};
        const int nspecies = vmesh.GetSize(b, dust::cons::density());
        for (int n = 0; n < nspecies; ++n) {
          // Extract state vector
          Real &dens = vmesh(b, dust::cons::density(n), k, j, i);
          Real &dmom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
          Real &dmom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
          Real &dmom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);

          // Apply density floor
          const bool dfloor = (dens > dflr_dust);
          dens = (dfloor)*dens + (!dfloor) * dflr_dust;

          // Stash nth dust velocity
          const std::array<Real, 3> vd{dmom1 / (hx[0] * dens), dmom2 / (hx[1] * dens),
                                       dmom3 / (hx[2] * dens)};

          // Coupling
          const auto id = vmesh(b, dust::cons::density(n)).sparse_id;
          Real tc = tp.tau(id);
          [[maybe_unused]] auto &sizes_ = sizes;
          if constexpr (DRAG == DragModel::stokes) {
            Real a_eff = sizes_(id);
            if (drag_use_bin_recon_) {
              const Real rho_i = dens * drag_mass_grid_(n);
              const int im1 = (n == 0) ? 0 : n - 1;
              const int ip1 = (n == nspecies - 1) ? nspecies - 1 : n + 1;
              const Real rho_im1 =
                  vmesh(b, dust::cons::density(im1), k, j, i) * drag_mass_grid_(im1);
              const Real rho_ip1 =
                  vmesh(b, dust::cons::density(ip1), k, j, i) * drag_mass_grid_(ip1);
              const Real sigma =
                  Dust::Coagulation::BinSlopeMCLimited(rho_im1, rho_i, rho_ip1);
              const Real dx = drag_log_widths_(n);
              if (rho_i > 0.0 && dx > 0.0) {
                const Real m1r =
                    Dust::Coagulation::BinMomentLogA(1.0, dx, rho_i, sigma, 1.0);
                if (m1r > 0.0) a_eff *= m1r;
              }
            }
            tc = std::max(tp.tau_min, std::min(tp.tau_max, tp.scale * grain_density_ /
                                                               dg * a_eff / vth));
          }
          const Real alpha = dt * ((tc <= 0.0) ? Big<Real>() : 1.0 / tc);
          for (int d = 0; d < 3; d++) {
            const Real rhop = dens * alpha / (1.0 + alpha + bd[d]);
            fd[d] += rhop * (1.0 + bd[d]);
            fvd[d] += rhop * (vd[d] + bd[d] * vdt[d]);
          }
        }

        // New vgas
        std::array<Real, 3> vgp{Null<Real>(), Null<Real>(), Null<Real>()};
        for (int d = 0; d < 3; d++) {
          vgp[d] = (dg * (vg[d] + bg[d] * vt[d]) + fvd[d]) / (dg * (1.0 + bg[d]) + fd[d]);
        }

        // Second pass to update all momenta
        fvd = {0.0, 0.0, 0.0};
        std::array<Real, 3> delta_g{0.0, 0.0, 0.0};
        for (int n = 0; n < nspecies; ++n) {
          // Extract dust density
          // NOTE(@pdmullen): Dust density already floored above
          const Real &dens = vmesh(b, dust::cons::density(n), k, j, i);

          // Stash nth dust velocity
          const std::array<Real, 3> vd{
              vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) / (hx[0] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) / (hx[1] * dens),
              vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) / (hx[2] * dens)};

          // Coupling
          const auto id = vmesh(b, dust::cons::density(n)).sparse_id;
          Real tc = tp.tau(id);
          [[maybe_unused]] auto &sizes_ = sizes;
          if constexpr (DRAG == DragModel::stokes) {
            Real a_eff = sizes_(id);
            if (drag_use_bin_recon_) {
              const Real rho_i = dens * drag_mass_grid_(n);
              const int im1 = (n == 0) ? 0 : n - 1;
              const int ip1 = (n == nspecies - 1) ? nspecies - 1 : n + 1;
              const Real rho_im1 =
                  vmesh(b, dust::cons::density(im1), k, j, i) * drag_mass_grid_(im1);
              const Real rho_ip1 =
                  vmesh(b, dust::cons::density(ip1), k, j, i) * drag_mass_grid_(ip1);
              const Real sigma =
                  Dust::Coagulation::BinSlopeMCLimited(rho_im1, rho_i, rho_ip1);
              const Real dx = drag_log_widths_(n);
              if (rho_i > 0.0 && dx > 0.0) {
                const Real m1r =
                    Dust::Coagulation::BinMomentLogA(1.0, dx, rho_i, sigma, 1.0);
                if (m1r > 0.0) a_eff *= m1r;
              }
            }
            tc = std::max(tp.tau_min, std::min(tp.tau_max, tp.scale * grain_density_ /
                                                               dg * a_eff / vth));
          }
          const Real alpha = dt * ((tc <= 0.0) ? Big<Real>() : 1.0 / tc);
          // Update dust momenta
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
            vmesh(b, dust::cons::momentum(VI(n, d)), k, j, i) += hx[d] * delta_d;
          }
        }

        // Final update to gas momenta and total energy
        for (int d = 0; d < 3; d++) {
          const Real prefac = dg * bg[d] / (1.0 + bg[d] + fd[d]);
          delta_g[d] -= prefac * (dg * (vg[d] - vt[d]) + fvd[d]);
          const Real vn = vg[d] + delta_g[d] / dg;
          vmesh(b, gas::cons::momentum(VI(0, d)), k, j, i) += hx[d] * delta_g[d];
          etot += 0.5 * (vg[d] + vn) * delta_g[d];
        }

        // Apply total energy floors
        const Real utmp = dg * sieflr_gas;
        etot = std::max(etot, utmp);
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
TaskStatus DragSource(MeshData<Real> *md, const Real time, const Real dt);

} // namespace Drag

#endif // DRAG_DRAG_HPP_
