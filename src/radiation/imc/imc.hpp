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
#ifndef RADIATION_IMC_IMC_HPP_
#define RADIATION_IMC_IMC_HPP_

// Artemis includes
#include "artemis.hpp"

// Jaybenne includes
#include "jaybenne.hpp"

using namespace parthenon::driver::prelude;

namespace IMC {
//----------------------------------------------------------------------------------------
//! \fn TaskCollection IMC::SyncFields
//! \brief Syncs fields following a Jaybenne IMC update
template <Coordinates GEOM>
TaskCollection SyncFields(Mesh *pmesh, const Real time, const Real dt) {
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &post_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = post_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
    auto pre_comm = tl.AddTask(start_recv, PreCommFillDerived<MeshData<Real>>, u0.get());
    auto bcs = parthenon::AddBoundaryExchangeTasks(pre_comm, tl, u0, pmesh->multilevel);
    auto c2p = tl.AddTask(bcs, FillDerived<MeshData<Real>>, u0.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus IMC::JaybenneIMC
//! \brief Executes thermal IMC transport (Jaybenne) and syncs updated fields
template <Coordinates GEOM>
TaskListStatus JaybenneIMC(Mesh *pmesh, const Real time, const Real dt) {
  auto status = jaybenne::RadiationStep(pmesh, time, dt).Execute();
  if (status != TaskListStatus::complete) return status;
  status = IMC::SyncFields<GEOM>(pmesh, time, dt).Execute();
  return status;
}

} // namespace IMC

#endif // RADIATION_IMC_IMC_HPP_
