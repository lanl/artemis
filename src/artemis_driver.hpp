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
#ifndef ARTEMIS_DRIVER_HPP_
#define ARTEMIS_DRIVER_HPP_

// Parthenon includes
#include <amr_criteria/refinement_package.hpp>
#include <parthenon/driver.hpp>
#include <prolong_restrict/prolong_restrict.hpp>

// Artemis Includes
#include "artemis.hpp"
#include "artemis_driver.hpp"
#include "derived/fill_derived.hpp"
#include "dust/dust.hpp"
#include "gas/gas.hpp"
#include "gravity/gravity.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/fluxes/fluid_fluxes.hpp"
#include "utils/integrators/artemis_integrator.hpp"

using parthenon::Packages_t;
using parthenon::StateDescriptor;
using namespace parthenon::driver::prelude;

namespace artemis {
//----------------------------------------------------------------------------------------
//! \class ArtemisDriver
//! \brief
template <Coordinates T>
class ArtemisDriver : public EvolutionDriver {
 public:
  using Integrator_t = parthenon::LowStorageIntegrator;
  using IntegratorPtr_t = std::unique_ptr<Integrator_t>;
  ArtemisDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm,
                const bool is_restart_in);
  TaskListStatus Step();
  void PreStepTasks();
  TaskCollection StepTasks();
  TaskCollection PostStepTasks();
  TaskCollection RadiationTasks();
  TaskListStatus RadiationDriver();

 protected:
  IntegratorPtr_t integrator, nbody_integrator;
  StateDescriptor *artemis_pkg;
  bool do_gas, do_dust, do_gravity, do_rotating_frame, do_cooling, do_drag, do_viscosity,
      do_nbody, do_conduction, do_diffusion, do_moment, do_imc;
  const bool is_restart;
  Real trad, dtr;
  int rad_stages;
};

using TaskCollectionFnPtr = TaskCollection (*)(Mesh *pm, const Real time, const Real dt);

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

} // namespace artemis

#endif // ARTEMIS_DRIVER_HPP_
