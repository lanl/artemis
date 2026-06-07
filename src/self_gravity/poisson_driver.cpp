//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
//! \file poisson_driver.cpp
//! \brief The code here is largely borrowed from the poisson_gmg example in Parthenon.

// C++ includes
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// Parthenon includes
#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/solver_utils.hpp>
#include <solvers/tridiag_solver.hpp>

// Artemis includes
#include "self_gravity/poisson_equation.hpp"
#include "self_gravity/self_gravity.hpp"
#include "utils/artemis_utils.hpp"

using namespace parthenon::driver::prelude;

namespace SelfGravity {

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus SelfGravity::PoissonDriver
//! \brief
void SolvePoisson(TaskCollection &tc, Mesh *pmesh, const Real time, const int stage) {
  using namespace parthenon;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("self_gravity");

  // Check if this package is active
  const auto active = ArtemisUtils::CheckPackageStatus(pkg, time);
  if (active == ArtemisUtils::PackageControl::inactive) {
    return;
  } else if (active == ArtemisUtils::PackageControl::shutdown) {
    if ((Globals::my_rank == 0) && (stage == 1)) {
      printf("Turning off self-gravity at t=%.8e...\n", time);
    }
    return;
  } else if (active == ArtemisUtils::PackageControl::initial) {
    if ((Globals::my_rank == 0) && (stage == 1)) {
      printf("Turning on self-gravity at t=%.8e...\n", time);
    }
  }
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    auto &md_phi = pmesh->mesh_data.Add("phi", md, {grav::phi::name()});
    auto &md_rhs = pmesh->mesh_data.Add("rhs", md, {grav::phi::name()});

    // Move the rhs variable into the rhs stage for stage based solver
    auto copy_rhs = tl.AddTask(
        none, TF(solvers::utils::between_fields::CopyData<grav::rhs, grav::phi>), md);
    copy_rhs =
        tl.AddTask(copy_rhs, TF(solvers::utils::CopyData<parthenon::TypeList<grav::phi>>),
                   md, md_rhs);

    // Solve
    auto setup = psolver->AddSetupTasks(tl, copy_rhs, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // Set BCs after solve
    auto bcs = parthenon::AddBoundaryExchangeTasks(solve, tl, md_phi, pmesh->multilevel);

    // Move the solution back so it is output
    auto copy_back = tl.AddTask(
        bcs, TF(solvers::utils::CopyData<parthenon::TypeList<grav::phi>>), md_phi, md);
  }
}

} // namespace SelfGravity
