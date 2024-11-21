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

// User-defined refinement criterion callback
namespace artemis {

std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock = nullptr;

} // namespace artemis

// User-defined ProblemGenerator functions
namespace advection {

template <Coordinates G>
void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);
} // namespace advection
namespace linear_wave {

template <Coordinates G>
void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);

} // namespace linear_wave
namespace disk {

void InitDiskParams(MeshBlock *pmb, ParameterInput *pin);

template <Coordinates G>
void DiskInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void DiskOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd);

} // namespace disk
namespace dust_collision {
void PreStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);
}
namespace strat {

void InitStratParams(MeshBlock *pmb, ParameterInput *pin);

template <Coordinates G>
void StratInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

} // namespace strat

namespace SI_strat {

void InitStratParams(MeshBlock *pmb, ParameterInput *pin);

template <Coordinates G>
void StratInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

template <Coordinates G>
void StratOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

} // namespace SI_strat

// Problem modifiers
namespace artemis {
//----------------------------------------------------------------------------------------
//! \fn void ProblemModifier
//! \brief
template <Coordinates G>
void ProblemModifier(parthenon::ParthenonManager *pman) {
  std::string artemis_problem =
      pman->pinput->GetOrAddString("artemis", "problem", "unset");

  if (artemis_problem == "advection") {
    pman->app_input->UserWorkAfterLoop = advection::UserWorkAfterLoop<G>;
  }
  if (artemis_problem == "disk") {
    pman->app_input->InitMeshBlockUserData = disk::InitDiskParams;

    artemis::ProblemCheckRefinementBlock = disk::ProblemCheckRefinementBlock;

    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        disk::DiskBoundary<G, parthenon::IndexDomain::inner_x1>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        disk::DiskBoundary<G, parthenon::IndexDomain::outer_x1>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] =
        disk::DiskBoundary<G, parthenon::IndexDomain::inner_x2>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x2] =
        disk::DiskBoundary<G, parthenon::IndexDomain::outer_x2>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x3] =
        disk::DiskBoundary<G, parthenon::IndexDomain::inner_x3>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x3] =
        disk::DiskBoundary<G, parthenon::IndexDomain::outer_x3>;
  }
  if (artemis_problem == "strat") {
    pman->app_input->InitMeshBlockUserData = strat::InitStratParams;

    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        strat::StratBoundary<G, parthenon::IndexDomain::inner_x1>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        strat::StratBoundary<G, parthenon::IndexDomain::outer_x1>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] =
        strat::StratBoundary<G, parthenon::IndexDomain::inner_x2>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x2] =
        strat::StratBoundary<G, parthenon::IndexDomain::outer_x2>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x3] =
        strat::StratBoundary<G, parthenon::IndexDomain::inner_x3>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x3] =
        strat::StratBoundary<G, parthenon::IndexDomain::outer_x3>;
  }
  if (artemis_problem == "SI_strat") {
    pman->app_input->InitMeshBlockUserData = SI_strat::InitStratParams;

    // Override with the pgen specific boundary routines
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        SI_strat::StratBoundary<G, parthenon::IndexDomain::inner_x1>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        SI_strat::StratBoundary<G, parthenon::IndexDomain::outer_x1>;

    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] =
        SI_strat::StratBoundary<G, parthenon::IndexDomain::inner_x2>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x2] =
        SI_strat::StratBoundary<G, parthenon::IndexDomain::outer_x2>;

    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x3] =
        SI_strat::StratBoundary<G, parthenon::IndexDomain::inner_x3>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x3] =
        SI_strat::StratBoundary<G, parthenon::IndexDomain::outer_x3>;
  }
  if (artemis_problem == "linear_wave") {
    pman->app_input->UserWorkAfterLoop = linear_wave::UserWorkAfterLoop<G>;
  }
  if (artemis_problem == "dust_collision") {
    pman->app_input->PreStepMeshUserWorkInLoop = dust_collision::PreStepUserWorkInLoop;
  }
  if (artemis_problem == "conduction") {
    pman->app_input->InitMeshBlockUserData = cond::InitCondParams;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        cond::CondBoundary<G, parthenon::IndexDomain::inner_x1>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        cond::CondBoundary<G, parthenon::IndexDomain::outer_x1>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] =
        cond::CondBoundary<G, parthenon::IndexDomain::inner_x2>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x2] =
        cond::CondBoundary<G, parthenon::IndexDomain::outer_x2>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::inner_x3] =
        cond::CondBoundary<G, parthenon::IndexDomain::inner_x3>;
    pman->app_input->boundary_conditions[parthenon::BoundaryFace::outer_x3] =
        cond::CondBoundary<G, parthenon::IndexDomain::outer_x3>;
  }
}

} // namespace artemis

#endif // PGEN_PROBLEM_MODIFIER_HPP_
