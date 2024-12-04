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
#ifndef PGEN_PROBLEM_MODIFIER_HPP_
#define PGEN_PROBLEM_MODIFIER_HPP_

// C++ includes
#include <string>

// Parthenon includes
#include <parthenon_manager.hpp>

// Artemis includes
#include "pgen.hpp"
#include "utils/artemis_utils.hpp"

// Jaybenne includes
#include "jaybenne.hpp"

// User-defined refinement criterion callback
namespace artemis {

std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock = nullptr;

} // namespace artemis

// Problem modifiers
namespace artemis {
//----------------------------------------------------------------------------------------
//! \fn void ProblemModifier
//! \brief
template <Coordinates G>
void ProblemModifier(parthenon::ParthenonManager *pman) {
  using BF = parthenon::BoundaryFace;
  using ID = parthenon::IndexDomain;

  std::string artemis_problem =
      pman->pinput->GetOrAddString("artemis", "problem", "unset");

  // Enroll artemis problem-specific function calls and boundary conditions
  if (artemis_problem == "advection") {
    pman->app_input->UserWorkAfterLoop = advection::UserWorkAfterLoop<G>;
  } else if (artemis_problem == "conduction") {
    pman->app_input->InitMeshBlockUserData = cond::InitCondParams;

    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "conductive",
                                               cond::CondBoundary<G, ID::inner_x1>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "conductive",
                                               cond::CondBoundary<G, ID::outer_x1>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "conductive",
                                               cond::CondBoundary<G, ID::inner_x2>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "conductive",
                                               cond::CondBoundary<G, ID::outer_x2>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x3, "conductive",
                                               cond::CondBoundary<G, ID::inner_x3>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x3, "conductive",
                                               cond::CondBoundary<G, ID::outer_x3>);
  } else if (artemis_problem == "disk") {
    pman->app_input->InitMeshBlockUserData = disk::InitDiskParams;

    artemis::ProblemCheckRefinementBlock = disk::ProblemCheckRefinementBlock;

    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "ic",
                                               disk::DiskBoundaryIC<G, ID::inner_x1>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "ic",
                                               disk::DiskBoundaryIC<G, ID::outer_x1>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "ic",
                                               disk::DiskBoundaryIC<G, ID::inner_x2>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "ic",
                                               disk::DiskBoundaryIC<G, ID::outer_x2>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x3, "ic",
                                               disk::DiskBoundaryIC<G, ID::inner_x3>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x3, "ic",
                                               disk::DiskBoundaryIC<G, ID::outer_x3>);

    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "extrap",
                                               disk::DiskBoundaryExtrap<G, ID::inner_x1>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "extrap",
                                               disk::DiskBoundaryExtrap<G, ID::outer_x1>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "extrap",
                                               disk::DiskBoundaryExtrap<G, ID::inner_x2>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "extrap",
                                               disk::DiskBoundaryExtrap<G, ID::outer_x2>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x3, "extrap",
                                               disk::DiskBoundaryExtrap<G, ID::inner_x3>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x3, "extrap",
                                               disk::DiskBoundaryExtrap<G, ID::outer_x3>);

    if constexpr (geometry::is_axisymmetric<G>() || (G == Coordinates::cylindrical) ||
                  (G == Coordinates::spherical3D)) {
      pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "viscous",
                                                 disk::DiskBoundaryVisc<G, ID::inner_x1>);
      pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "viscous",
                                                 disk::DiskBoundaryVisc<G, ID::outer_x1>);
    }
  } else if (artemis_problem == "linear_wave") {
    pman->app_input->UserWorkAfterLoop = linear_wave::UserWorkAfterLoop<G>;
  } else if (artemis_problem == "shock") {
    pman->app_input->InitMeshBlockUserData = shock::InitShockParams;

    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "ic",
                                               shock::ShockInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "ic",
                                               shock::ShockOuterX1<G>);
  } else if (artemis_problem == "strat") {
    pman->app_input->InitMeshBlockUserData = strat::InitStratParams;

    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "extrap",
                                               strat::ExtrapInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "extrap",
                                               strat::ExtrapOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "inflow",
                                               strat::ShearInnerX2<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "inflow",
                                               strat::ShearOuterX2<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x3, "extrap",
                                               strat::ExtrapInnerX3<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x3, "extrap",
                                               strat::ExtrapOuterX3<G>);
  } else if (artemis_problem == "dust_collision") {
    pman->app_input->PreStepMeshUserWorkInLoop = dust_collision::PreStepUserWorkInLoop;
  }

  // Register jaybenne swarm boundary conditions
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::inner_x1, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::inner_x1>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::outer_x1, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::outer_x1>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::inner_x2, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::inner_x2>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::outer_x2, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::outer_x2>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::inner_x3, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::inner_x3>);
  pman->app_input->RegisterSwarmBoundaryCondition(
      BF::outer_x3, "jaybenne_reflecting", jaybenne::PhotonReflectBC<BF::outer_x3>);
}

} // namespace artemis

#endif // PGEN_PROBLEM_MODIFIER_HPP_
