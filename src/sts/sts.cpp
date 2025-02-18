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

// C++ headers
#include <limits>
#include <vector>

// Parthenon includes
#include <globals.hpp>

// Artemis includes
#include "artemis.hpp"
#include "gas/gas.hpp"
#include "sts.hpp"
#include "utils/artemis_utils.hpp"

using ArtemisUtils::VI;

namespace STS{

IntegratorPtr_t sts_integrator;

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor STS ::Initialize
//! \brief Adds intialization function for STS package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {

  auto STS = std::make_shared<StateDescriptor>("STS");
  Params &params = STS->AllParams();

  // initial check of getting the physics needed for the sts integrator
  // Determine input file specified physics
  const bool do_gas = pin->GetOrAddBoolean("physics", "gas", true);
  const bool do_viscosity = pin->GetOrAddBoolean("physics", "viscosity", false);
  const bool do_conduction = pin->GetOrAddBoolean("physics", "conduction", false);
  const bool do_sts = pin->GetOrAddBoolean("physics", "sts", false);
  const bool do_diffusion = do_conduction || do_viscosity;

  params.Add("do_sts", do_sts);
  params.Add("do_gas", do_gas);
  params.Add("do_viscosity", do_viscosity);
  params.Add("do_conduction", do_conduction);
  params.Add("do_diffusion", do_diffusion);

  if (!do_diffusion) {
    PARTHENON_FAIL("STS integrator requires diffusion to be enabled!");
  }

  // Getting the integrator & time ratio between hyperbolic and parabolic terms
  std::string sts_intg_mothod = pin->GetOrAddString("sts", "integrator", "none");
  Real sts_max_dt_ratio  = pin->GetOrAddReal("sts","sts_max_dt_ratio", -1.0);
  const bool info_output = pin->GetOrAddBoolean("sts", "info_output", false);

  STSInt sts_intg_mothod_param = STSInt::null;
  if (sts_intg_mothod == "rkl1") {
    sts_intg_mothod_param = STSInt::rkl1;
    sts_integrator = std::make_unique<Integrator_t>("rk1");
  } else if (sts_intg_mothod == "rkl2") {
    PARTHENON_FAIL("rkl2 STS integrator not implemented!");
    sts_intg_mothod_param = STSInt::rkl2;
  } else {
    PARTHENON_FAIL("STS integrator not recognized!");
  }

  params.Add("sts_intg_mothod", sts_intg_mothod_param);
  params.Add("sts_max_dt_ratio", sts_max_dt_ratio);
  params.Add("info_output", info_output);
  
  return STS;
}

//----------------------------------------------------------------------------------------
//! \fn PreStepSTSTasks
//! \brief Executes the pre-step tasks for the STS integrator
template <Coordinates GEOM>
void PreStepSTSTasks(Mesh *pmesh, const Real time, Real dt, int nstages) {

  // Getting the integrator & time ratio between hyperbolic and parabolic terms
  auto &sts = pmesh->packages.Get("STS");
  const STSInt sts_intg_mothod = sts->Param<STSInt>("sts_intg_mothod");

  // Check if the integrator is set
  if (sts_intg_mothod == STSInt::null) {
    PARTHENON_FAIL("STS integrator not set!");
  }

  // Execute the integrator tasks
  if (sts_intg_mothod == STSInt::rkl1) {
    // RKL1 : Full timestep dt_sts
    for (int stage = 1; stage <= nstages; ++stage) {
      //-----------------------------------------------------------------
      // RKL1 STS update
      // Y_{j} = nuj*Y_{j-2} + muj*Y_{j-1} + dt_sts*muj_tilde*F_diff(Y_{j-1}), 
      //  where F_diff(Y_{j-1}) = divf/vol
      //
      // Set up the STS stage coefficients
      // gam1 = nuj = (2.*j - 1.)/j;
      // gam0 = muj = (1. - j)/j;
      // beta_dt/dt = muj_tilde = pm->muj*2./(std::pow(s, 2.) + s);
      //
      // Update strategy for the registers
      // Let u0 = Y_{j-1}, u1 = Y_{j-2}
      // 1. After ApplyUpdate: u1 = nuj*Y_{j-2} + muj*Y_{j-1}, u1 -> Y'_{j}
      // 2. swap u0 <-> u1, u0 -> Y'_{j}, u0 -> Y_{j-1}
      // 3. After DiffusionUpdate: u0 = Y_{j}, u1 = Y_{j-1}

      Real muj = (2.*stage - 1.)/stage;
      Real nuj = (1. - stage)/stage;
      Real muj_tilde = muj * 2./(std::pow(nstages, 2.) + nstages);
      Real bdt = muj_tilde * dt;
  
      // We always use the stage 1 state for the coefficients
      sts_integrator->beta[0] = 0.0;
      sts_integrator->gam0[0] = nuj; // since we swap u0 and u1
      sts_integrator->gam1[0] = muj;
      STSRKL1<GEOM>(pmesh, time, bdt, stage, nstages).Execute();
    }
  } else if (sts_intg_mothod == STSInt::rkl2) {
    PARTHENON_FAIL("STS rkl2 integrator not implemented!");
    // (TODO) RKL2 : // eq (21) using half hyperbolic timestep 
    // due to Strang split
    //STSRKL2FirstStage<GEOM>(pmesh, time, 0.5*dt, nstages);
  }
}

//----------------------------------------------------------------------------------------
//! \fn STSRKL2FirstStage
//! \brief Assembles the tasks for first stage of the STS RKL2 integrator
template <Coordinates GEOM>
void STSRKL2FirstStage( Mesh *pm, const Real time, Real dt, int nstages) {
  // TODO: Implement RKL2 STS integration
}

//----------------------------------------------------------------------------------------
//! \fn STSRKL2SecondStage
//! \brief Assembles the tasks for first stage of the STS RKL2 integrator
template <Coordinates GEOM>
void STSRKL2SecondStage( Mesh *pm, const Real time, Real dt, int nstages) {
  // TODO: Implement RKL2 STS integration
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef Mesh M;
//RK2 first stage template instantiations
template void STSRKL2FirstStage<C::cartesian>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<C::cylindrical>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<C::spherical1D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<C::spherical2D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<C::spherical3D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<C::axisymmetric>(M *m, const Real time, Real dt, int nstages);
//RK2 second stage template instantiations
template void STSRKL2SecondStage<C::cartesian>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<C::cylindrical>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<C::spherical1D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<C::spherical2D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<C::spherical3D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<C::axisymmetric>(M *m, const Real time, Real dt, int nstages);
//PreStepSTSTasks template instantiations
template void PreStepSTSTasks<C::cartesian>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<C::cylindrical>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<C::spherical1D>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<C::spherical2D>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<C::spherical3D>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<C::axisymmetric>(M *m,const Real time, Real dt, int nstages);
} // namespace STS