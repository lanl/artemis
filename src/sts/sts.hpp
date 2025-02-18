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
#ifndef STS_STS_HPP_
#define STS_STS_HPP_

#include "artemis.hpp"
#include "derived/fill_derived.hpp"
#include "gas/gas.hpp"
#include "geometry/geometry.hpp"
#include "utils/units.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/integrators/artemis_integrator.hpp"

namespace STS {

using Integrator_t = parthenon::LowStorageIntegrator;
using IntegratorPtr_t = std::unique_ptr<Integrator_t>;

// Global variable declaration
extern IntegratorPtr_t sts_integrator;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \fn STSRKL1
//! \brief Assembles the tasks for the STS RKL1 integrator
// comment: Maybe it should be moved back to artemis_driver.cpp
template <Coordinates GEOM>
TaskCollection STSRKL1(Mesh *pmesh, const Real time, Real dt, int stage, int nstages) {

  using TQ = TaskQualifier;
  TaskCollection tc;

  using namespace ::parthenon::Update;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  auto &pkg = pmesh->packages.Get("STS");
  const auto do_viscosity = pkg->template Param<bool>("do_viscosity");
  const auto do_conduction = pkg->template Param<bool>("do_conduction");
  const auto do_diffusion = pkg->template Param<bool>("do_diffusion");
  const auto do_gas = pkg->template Param<bool>("do_gas");

  // Deep copy u0 into u1 for integrator logic
  if (stage == 1) {
    auto &init_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = init_region[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
      tl.AddTask(none, ArtemisUtils::DeepCopyConservedData, u1.get(), u0.get());
    }
  }

  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);

    // Start looking for incoming messages (including for flux correction)
    auto start_recv_u0 = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
    auto start_flx_recv_u0 = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, u0);

    // Compute (gas) diffusive fluxes
    TaskID diff_flx = none;
    auto zf = tl.AddTask(none, Gas::ZeroDiffusionFlux, u0.get());
    TaskID vflx = zf, tflx = zf;
    if (do_viscosity) vflx = tl.AddTask(zf, Gas::ViscousFlux<GEOM>, u0.get());
    if (do_conduction) tflx = tl.AddTask(zf | vflx, Gas::ThermalFlux<GEOM>, u0.get());
    diff_flx = vflx | tflx;

    // TODO(KWHO) Dust diffusion fluxes in the future

    // Communicate and set fluxes
    auto send_flx = tl.AddTask(
      diff_flx, parthenon::SendBoundBufs<parthenon::BoundaryType::flxcor_send>, u0);
    auto recv_flx_u0 = 
        tl.AddTask(start_flx_recv_u0, parthenon::ReceiveFluxCorrections, u0);
    auto set_flx_u0 = tl.AddTask(recv_flx_u0, parthenon::SetFluxCorrections, u0);

    // Apply flux divergence
    auto update = none;
    update = tl.AddTask(diff_flx | set_flx_u0, ArtemisUtils::ApplyUpdate<GEOM>, u1.get(),
                        u0.get(), 1, sts_integrator.get());
   
    // swap u0 <-> u1
    auto swap_data_1 = tl.AddTask(update, ArtemisUtils::SwapData, u0.get(), u1.get());

    // Apply "coordinate source terms"
    TaskID gas_coord_src = swap_data_1, dust_coord_src = swap_data_1;
    if (do_gas) gas_coord_src = tl.AddTask(swap_data_1, Gas::FluxSource, u0.get(), dt);

    TaskID gas_diff_src = gas_coord_src | diff_flx | set_flx_u0;
    gas_diff_src = tl.AddTask(gas_coord_src | diff_flx | set_flx_u0,
                              Gas::DiffusionUpdate<GEOM>, u0.get(), dt);

    // Set auxillary fields
    auto set_aux_u0 =
        tl.AddTask(gas_diff_src, ArtemisDerived::SetAuxillaryFields<GEOM>, u0.get());

    // Set (remaining) fields to be communicated
    auto pre_comm_u0 = tl.AddTask(set_aux_u0, // update,
                                  PreCommFillDerived<MeshData<Real>>, u0.get());

    // Set boundary conditions (both physical and logical)
    auto bcs_u0 = 
        parthenon::AddBoundaryExchangeTasks(pre_comm_u0, tl, u0, pmesh->multilevel);

    // Update primitive variables
    auto c2p_u0 = 
        tl.AddTask(TQ::local_sync, bcs_u0, FillDerived<MeshData<Real>>, u0.get());

  }

  return tc;
}

// STS integrator functions
template <Coordinates GEOM>
void PreStepSTSTasks(Mesh *pmesh, const Real time, Real dt, int nstages);

} // namespace STS

#endif // STS_STS_HPP_