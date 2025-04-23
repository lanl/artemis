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
#include "drag.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
namespace Drag {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Drag::Initialize
//! \brief Adds intialization function for damping package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto drag = std::make_shared<StateDescriptor>("drag");
  Params &params = drag->AllParams();

  // Below we store the bounds of the Mesh and parameters that define the bounds of the
  // damping zones and the damping rates
  //
  // We damp in the X* (* = 1,2,3) direction between
  //          x*min <= x* <= inner_x*, at the rate inner_x*_rate
  // and   outer_x* <= x* <=    x*max, at the rate outer_x*_rate
  const Coupling type = ChooseDrag(pin->GetString("drag", "type"));
  params.Add("type", type);

  params.Add("x1min", pin->GetReal("parthenon/mesh", "x1min"));
  params.Add("x2min", pin->GetReal("parthenon/mesh", "x2min"));
  params.Add("x3min", pin->GetReal("parthenon/mesh", "x3min"));
  params.Add("x1max", pin->GetReal("parthenon/mesh", "x1max"));
  params.Add("x2max", pin->GetReal("parthenon/mesh", "x2max"));
  params.Add("x3max", pin->GetReal("parthenon/mesh", "x3max"));

  const bool do_gas = pin->GetOrAddBoolean("physics", "gas", true);
  const bool do_dust = pin->GetOrAddBoolean("physics", "dust", false);

  if (do_gas) {
    if (type == Coupling::self) {
      PARTHENON_REQUIRE(
          pin->DoesBlockExist("gas/damping"),
          "With do_drag = true and do_gas = true, you need a gas/damping node");
    }
    if (pin->DoesBlockExist("gas/damping")) {
      params.Add("gas_self_drag", SelfDragParams("gas/damping", pin));
    } else {
      params.Add("gas_self_drag", SelfDragParams());
    }
  } else {
    params.Add("gas_self_drag", SelfDragParams());
  }
  if (do_dust) {
    if (type == Coupling::self) {
      PARTHENON_REQUIRE(
          pin->DoesBlockExist("dust/damping"),
          "With do_drag = true and do_dust = true, you need a dust/damping node");
    }
    if (pin->DoesBlockExist("dust/damping")) {
      params.Add("dust_self_drag", SelfDragParams("dust/damping", pin));
    } else {
      params.Add("dust_self_drag", SelfDragParams());
    }
  } else {
    params.Add("dust_self_drag", SelfDragParams());
  }

  if (type == Coupling::simple_dust) {
    PARTHENON_REQUIRE(do_gas && do_dust,
                      "drag type simple_dust requires do_gas = do_dust = true");
    PARTHENON_REQUIRE(pin->DoesBlockExist("dust/stopping_time"),
                      "drag type simple_dust requires a dust/stopping_time node");

    // Enforce 1 gas species

    params.Add("stopping_time_params", StoppingTimeParams("dust/stopping_time", pin));
  }

  return drag;
}

template <Coordinates GEOM>
TaskStatus DragSource(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_gas");

  auto &drag_pkg = pm->packages.Get("drag");
  const Coupling ctype = drag_pkg->template Param<Coupling>("type");
  // not this contains do_viscosity

  const auto gas_self_par = drag_pkg->template Param<SelfDragParams>("gas_self_drag");
  const auto dust_self_par = drag_pkg->template Param<SelfDragParams>("dust_self_drag");

  if (ctype == Coupling::self) {
    // Just self coupling
    if (do_gas && gas_self_par.damp_to_visc) {
      auto &gas_pkg = pm->packages.Get("gas");
      const auto &dp = gas_pkg->template Param<Diffusion::DiffCoeffParams>("visc_params");
      const auto &eos_d = gas_pkg->template Param<EOS>("eos_d");
      if (dp.type == Diffusion::DiffType::viscosity_plaw) {
        return SelfDragSourceImpl<Diffusion::DiffType::viscosity_plaw, GEOM>(
            md, time, dt, dp, eos_d, gas_self_par, dust_self_par);
      } else if (dp.type == Diffusion::DiffType::viscosity_alpha) {
        return SelfDragSourceImpl<Diffusion::DiffType::viscosity_alpha, GEOM>(
            md, time, dt, dp, eos_d, gas_self_par, dust_self_par);
      } else {
        PARTHENON_FAIL("The chosen viscosity model does not work with damping");
      }
    } else {
      Diffusion::DiffCoeffParams dp;
      EOS eos_d;
      return SelfDragSourceImpl<Diffusion::DiffType::null, GEOM>(
          md, time, dt, dp, eos_d, gas_self_par, dust_self_par);
    }
  } else if (ctype == Coupling::simple_dust) {
    // dust couples to gas
    auto &gas_pkg = pm->packages.Get("gas");
    auto &dust_pkg = pm->packages.Get("dust");
    const auto &eos_d = gas_pkg->template Param<EOS>("eos_d");
    const auto stop_par =
        drag_pkg->template Param<StoppingTimeParams>("stopping_time_params");
    if (gas_self_par.damp_to_visc) {
      auto &dp = gas_pkg->template Param<Diffusion::DiffCoeffParams>("visc_params");
      if (dp.type == Diffusion::DiffType::viscosity_plaw) {
        if (stop_par.model == DragModel::constant) {
          return SimpleDragSourceImpl<Diffusion::DiffType::viscosity_plaw,
                                      DragModel::constant, GEOM>(
              md, time, dt, dp, eos_d, gas_self_par, dust_self_par, stop_par);
        } else if (stop_par.model == DragModel::stokes) {
          return SimpleDragSourceImpl<Diffusion::DiffType::viscosity_plaw,
                                      DragModel::stokes, GEOM>(
              md, time, dt, dp, eos_d, gas_self_par, dust_self_par, stop_par);
        }
      } else if (dp.type == Diffusion::DiffType::viscosity_alpha) {
        if (stop_par.model == DragModel::constant) {
          return SimpleDragSourceImpl<Diffusion::DiffType::viscosity_alpha,
                                      DragModel::constant, GEOM>(
              md, time, dt, dp, eos_d, gas_self_par, dust_self_par, stop_par);
        } else if (stop_par.model == DragModel::stokes) {
          return SimpleDragSourceImpl<Diffusion::DiffType::viscosity_alpha,
                                      DragModel::stokes, GEOM>(
              md, time, dt, dp, eos_d, gas_self_par, dust_self_par, stop_par);
        }
      } else {
        PARTHENON_FAIL("The chosen viscosity model does not work with damping");
      }
    } else {
      Diffusion::DiffCoeffParams dp;
      if (stop_par.model == DragModel::constant) {
        return SimpleDragSourceImpl<Diffusion::DiffType::null, DragModel::constant, GEOM>(
            md, time, dt, dp, eos_d, gas_self_par, dust_self_par, stop_par);
      } else if (stop_par.model == DragModel::stokes) {
        return SimpleDragSourceImpl<Diffusion::DiffType::null, DragModel::stokes, GEOM>(
            md, time, dt, dp, eos_d, gas_self_par, dust_self_par, stop_par);
      }
    }
  } else {
    PARTHENON_FAIL("Invalid drag model!");
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> MD;
template TaskStatus DragSource<C::cartesian>(MD *md, const Real tt, const Real dt);
template TaskStatus DragSource<C::cylindrical>(MD *md, const Real tt, const Real dt);
template TaskStatus DragSource<C::spherical1D>(MD *md, const Real tt, const Real dt);
template TaskStatus DragSource<C::spherical2D>(MD *md, const Real tt, const Real dt);
template TaskStatus DragSource<C::spherical3D>(MD *md, const Real tt, const Real dt);
template TaskStatus DragSource<C::axisymmetric>(MD *md, const Real tt, const Real dt);

} // namespace Drag
