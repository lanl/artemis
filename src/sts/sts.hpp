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
#include "geometry/geometry.hpp"
#include "utils/units.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/integrators/artemis_integrator.hpp"

namespace STS{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// STS integrator functions
template <Coordinates GEOM>
void PreStepSTSTasks(Mesh *pmesh, const Real time, Real dt, int nstages);

// task list functions
template <Coordinates GEOM>
void STSRKL1( Mesh *pmesh, const Real time, Real dt, int nstages);

template <Coordinates GEOM>
void STSRKL2FirstStage( Mesh *pm, const Real time, Real dt, int nstages);
template <Coordinates GEOM>
void STSRKL2SecondStage( Mesh *pm, const Real time, Real dt, int nstages);

// task status functions
template <Coordinates GEOM>
TaskStatus RKL1FluxUpadte(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt,
                      const Real muj, const Real nuj, const Real muj_tilde);

//----------------------------------------------------------------------------------------
//! \fn STSRKL1
//! \brief Assembles the tasks for the STS RKL1 integrator
// comment: Maybe it should be moved back to artemis_driver.cpp
template <Coordinates GEOM>
void STSRKL1(Mesh *pmesh, const Real time, Real dt, int nstages) {
  
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  auto &pkg = pmesh->packages.Get("STS");
  const auto do_viscosity = pkg->template Param<bool>("do_viscosity");
  const auto do_conduction = pkg->template Param<bool>("do_conduction");
  const auto do_diffusion = pkg->template Param<bool>("do_diffusion");
  const auto do_gas = pkg->template Param<bool>("do_gas");

  // Deep copy u0 into u1 for integrator logic
  auto &init_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = init_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
    tl.AddTask(none, ArtemisUtils::DeepCopyConservedData, u1.get(), u0.get());
  }

  for (int stage = 1; stage <= nstages; stage++) {
    // Set up the STS stage coefficients
    // v0 = Y_{j-1}
    // v1 = Y_{j-2}
    // gam1 = muj = (2.*j - 1.)/j;
    // gam0 = nuj = (1. - j)/j;
    // beta_dt/dt = muj_tilde = pm->muj*2./(std::pow(s, 2.) + s);
    Real muj = (1. - stage)/stage;
    Real nuj = (2.*stage - 1.)/stage;
    Real muj_tilde = (2.*stage - 1.)/stage * 2./(std::pow(nstages, 2.) + nstages);

    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);

      // Start looking for incoming messages (including for flux correction)
      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
      auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, u0);

      // Compute (gas) diffusive fluxes
      TaskID diff_flx = none;
      if ((do_diffusion) && (do_gas)) {
        auto zf = tl.AddTask(none, Gas::ZeroDiffusionFlux, u0.get());
        TaskID vflx = zf, tflx = zf;
        if (do_viscosity) vflx = tl.AddTask(zf, Gas::ViscousFlux<GEOM>, u0.get());
        if (do_conduction) tflx = tl.AddTask(zf | vflx, Gas::ThermalFlux<GEOM>, u0.get());
        diff_flx = vflx | tflx;
      }

      // TODO(KWHO) Dust diffusion fluxes

      // Communicate and set fluxes
      auto send_flx =
          tl.AddTask(diff_flx,
                    parthenon::SendBoundBufs<parthenon::BoundaryType::flxcor_send>, u0);
      auto recv_flx = tl.AddTask(start_flx_recv, parthenon::ReceiveFluxCorrections, u0);
      auto set_flx = tl.AddTask(recv_flx, parthenon::SetFluxCorrections, u0);

      // Apply flux divergence, STS need stage to 0 for the sts ceofficients
      auto update =
          tl.AddTask(set_flx, RKL1FluxUpadte<GEOM>,
                    u0.get(), u1.get(),
                    dt, muj, nuj, muj_tilde);
    }
  }
}

}

#endif // STS_STS_HPP_