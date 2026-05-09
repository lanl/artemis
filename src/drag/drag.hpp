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

// ... Coupling types
enum class Coupling { simple_dust, self, full, null };

// ... Interspecies drag models for the full Chapman-Cowling coupling path
// hard_sphere   - calibrated hard-sphere collision integrals (first backend)
// lj            - Lennard-Jones surrogate collision integrals
enum class GasDragModel { hard_sphere, lj, null };

// ... Drag models for the simple dust-gas coupling path
enum class DragModel { constant, stokes, null };

//----------------------------------------------------------------------------------------
//! \fn  Coupling Drag::ChooseDrag
//! \brief Helper function to help select drag coupling type
inline Coupling ChooseDrag(const std::string choice) {
  if (choice == "self") {
    return Coupling::self;
  } else if (choice == "simple_dust") {
    return Coupling::simple_dust;
  } else if (choice == "full") {
    return Coupling::full;
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
//! \struct FullCouplingParams
//! \brief Parameters for the full Chapman-Cowling interspecies coupling
//!
//! Species are indexed 0..nspecies-1, matching gas.nspecies ordering.
//! mu_s[n]       - mean molecular mass of species n (in AMU)
//! sigma_s[n]    - effective hard-sphere diameter (collision cross-section radius) in
//! code
//!                 length units, or the LJ sigma parameter for the lj model
//! eps_s[n]      - LJ epsilon/kB (K) for the lj model (unused for hard_sphere)
//! dof_s[n]      - internal degrees of freedom per molecule (e.g. 3 for monatomic,
//!                 5 for diatomic); used for energy-exchange weighting
struct FullCouplingParams {
  GasDragModel model;
  int nspecies;
  ParArray1D<Real> mu_s;    // molecular mass per species [AMU]
  ParArray1D<Real> sigma_s; // collision diameter / LJ sigma per species [code length]
  ParArray1D<Real> eps_s;   // LJ epsilon/kB per species [K] (lj model only)
  ParArray1D<Real> dof_s;   // internal DOF per molecule per species

  FullCouplingParams() : model(GasDragModel::null), nspecies(0) {}

  FullCouplingParams(ParameterInput *pin, const ArtemisUtils::Constants &constants) {
    const std::string block = "drag/full";
    nspecies = pin->GetOrAddInteger("gas", "nspecies", 1);

    const std::string model_str =
        pin->GetOrAddString(block, "collision_model", "hard_sphere");
    if (model_str == "hard_sphere") {
      model = GasDragModel::hard_sphere;
    } else if (model_str == "lj") {
      model = GasDragModel::lj;
    } else {
      PARTHENON_FAIL("drag/full collision_model must be hard_sphere or lj");
      model = GasDragModel::null;
    }

    mu_s = ParArray1D<Real>("drag_mu", nspecies);
    sigma_s = ParArray1D<Real>("drag_sigma", nspecies);
    eps_s = ParArray1D<Real>("drag_eps", nspecies);
    dof_s = ParArray1D<Real>("drag_dof", nspecies);

    auto h_mu = mu_s.GetHostMirror();
    auto h_sigma = sigma_s.GetHostMirror();
    auto h_eps = eps_s.GetHostMirror();
    auto h_dof = dof_s.GetHostMirror();

    std::vector<Real> mu_v = pin->GetVector<Real>(block, "mu");
    std::vector<Real> sigma_v = pin->GetVector<Real>(block, "sigma");

    // Optional: LJ epsilon and internal DOF (default to monatomic hard-sphere)
    std::vector<Real> eps_v(nspecies, 0.0);
    std::vector<Real> dof_v(nspecies, 3.0);
    if (pin->DoesParameterExist(block, "eps")) eps_v = pin->GetVector<Real>(block, "eps");
    if (pin->DoesParameterExist(block, "dof")) dof_v = pin->GetVector<Real>(block, "dof");

    PARTHENON_REQUIRE(static_cast<int>(mu_v.size()) == nspecies,
                      "drag/full mu must have nspecies entries");
    PARTHENON_REQUIRE(static_cast<int>(sigma_v.size()) == nspecies,
                      "drag/full sigma must have nspecies entries");

    for (int n = 0; n < nspecies; ++n) {
      h_mu(n) = mu_v[n];
      h_sigma(n) = sigma_v[n];
      h_eps(n) = eps_v[n];
      h_dof(n) = dof_v[n];
    }
    mu_s.DeepCopy(h_mu);
    sigma_s.DeepCopy(h_sigma);
    eps_s.DeepCopy(h_eps);
    dof_s.DeepCopy(h_dof);
  }
};

//----------------------------------------------------------------------------------------
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            const ArtemisUtils::Constants &constants);

template <Coordinates GEOM>
TaskStatus DragSource(MeshData<Real> *md, const Real time, const Real dt);

} // namespace Drag

#endif // DRAG_DRAG_HPP_
