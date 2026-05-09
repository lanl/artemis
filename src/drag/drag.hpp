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
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
TaskStatus DragSource(MeshData<Real> *md, const Real time, const Real dt);

} // namespace Drag

#endif // DRAG_DRAG_HPP_
