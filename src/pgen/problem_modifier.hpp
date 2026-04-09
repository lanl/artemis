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
  } else if (artemis_problem == "sst") {
    pman->app_input->InitMeshBlockUserData = sst::InitSSTParams;
  } else if (artemis_problem == "sst_Trelax") {
    pman->app_input->InitMeshBlockUserData = sst_Trelax::InitSST_TrelaxParams;
  } else if (artemis_problem == "brio_wu") {
    pman->app_input->InitMeshBlockUserData = brio_wu::Init_BRIO_WU_Params;
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "inflow_l",
                                               brio_wu::InflowInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "inflow_r",
                                               brio_wu::InflowOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "farfield",
                                               brio_wu::FarFieldOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "nrbc",
                                               brio_wu::NRBCOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "orlanski",
                                               brio_wu::OrlanskiOuterX1<G>);
  } else if (artemis_problem == "brio_wu_2sides") {
    pman->app_input->InitMeshBlockUserData = brio_wu_2sides::Init_BRIO_WU_2SIDES_Params;
  } else if (artemis_problem == "brio_wu_reversed") {
    pman->app_input->InitMeshBlockUserData = brio_wu_reversed::Init_BRIO_WU_REVERSED_Params;
  } else if (artemis_problem == "brio_wu_y") {
    pman->app_input->InitMeshBlockUserData = brio_wu_y::Init_BRIO_WU_Y_Params;
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "outflow_r",
                                               brio_wu_y::OutflowOuterX1<G>);
  } else if (artemis_problem == "blast_mhd") {
    pman->app_input->InitMeshBlockUserData = blast_mhd::Init_BLAST_MHD_Params;
  } else if (artemis_problem == "rotor_mhd") {
    pman->app_input->InitMeshBlockUserData = rotor_mhd::Init_ROTOR_MHD_Params;
  } else if (artemis_problem == "current_sheet_mhd") {
    pman->app_input->InitMeshBlockUserData = current_sheet_mhd::Init_CURRENT_SHEET_MHD_Params;
  } else if (artemis_problem == "blast3D_mhd") {
    pman->app_input->InitMeshBlockUserData = blast3D_mhd::Init_BLAST3D_MHD_Params;
  } else if (artemis_problem == "plasmoid_mhd") {
    pman->app_input->InitMeshBlockUserData = plasmoid_mhd::Init_PLASMOID_MHD_Params;
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "sym",
                                               plasmoid_mhd::SymInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "sym",
                                               plasmoid_mhd::SymInnerX2<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "conducting",
                                               plasmoid_mhd::CondOuterX2<G>);
  } else if (artemis_problem == "orszag_tang_mhd") {
    pman->app_input->InitMeshBlockUserData = orszag_tang_mhd::Init_ORSZAG_TANG_MHD_Params;
  } else if (artemis_problem == "linear_wave_mhd") {
    pman->app_input->UserWorkAfterLoop = linear_wave_mhd::UserWorkAfterLoop<G>;
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "periodic_x_l",
                                               linear_wave_mhd::PeriodicInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "periodic_x_r",
                                               linear_wave_mhd::PeriodicOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "periodic_y_l",
                                               linear_wave_mhd::PeriodicInnerX2<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "periodic_y_r",
                                               linear_wave_mhd::PeriodicOuterX2<G>);
  } else if (artemis_problem == "fast_shock") {
    pman->app_input->InitMeshBlockUserData = fast_shock::Init_FAST_SHOCK_Params;
  } else if (artemis_problem == "huba_hall") {
    pman->app_input->InitMeshBlockUserData = huba_hall::Init_HUBA_HALL_Params;
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "periodic_x_l",
                                               huba_hall::PeriodicInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "periodic_x_r",
                                               huba_hall::PeriodicOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "periodic_y_l",
                                               huba_hall::PeriodicInnerX2<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "periodic_y_r",
                                               huba_hall::PeriodicOuterX2<G>);
    //pman->app_input->UserWorkAfterLoop = huba_hall::UserWorkAfterLoop<G>;
  } else if (artemis_problem == "field_diffusion") {
    pman->app_input->InitMeshBlockUserData = field_diffusion::Init_FIELD_DIFFUSION_Params;
    pman->app_input->UserWorkAfterLoop = field_diffusion::UserWorkAfterLoop<G>;
  } else if (artemis_problem == "EMwave") {
    pman->app_input->InitMeshBlockUserData = EMwave::Init_EMWAVE_Params;
    pman->app_input->UserWorkAfterLoop = EMwave::UserWorkAfterLoop<G>;
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "periodic_x_l",
                                               EMwave::PeriodicInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "periodic_x_r",
                                               EMwave::PeriodicOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "periodic_y_l",
                                               EMwave::PeriodicInnerX2<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "periodic_y_r",
                                               EMwave::PeriodicOuterX2<G>);
  } else if (artemis_problem == "Zpinch_2D") {
    pman->app_input->InitMeshBlockUserData = Zpinch_2D::Init_ZPINCH_2D_Params;
  } else if (artemis_problem == "planar_foil") {
    pman->app_input->InitMeshBlockUserData = planar_foil::Init_PLANAR_FOIL_Params;
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "current_x_l",
                                               planar_foil::CurrentInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "current_x_r",
                                               planar_foil::CurrentOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x2, "conduct_y_l",
                                               planar_foil::ConductInnerX2<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x2, "conduct_y_r",
                                               planar_foil::ConductOuterX2<G>);
  } else if (artemis_problem == "one_planar_foil") {
    pman->app_input->InitMeshBlockUserData = one_planar_foil::Init_ONE_PLANAR_FOIL_Params;
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "dirichlet_x_l",
                                               one_planar_foil::DirichletInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "dirichlet_x_r",
                                               one_planar_foil::DirichletOuterX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::inner_x1, "current_x_l",
                                               one_planar_foil::CurrentInnerX1<G>);
    pman->app_input->RegisterBoundaryCondition(BF::outer_x1, "current_x_r",
                                               one_planar_foil::CurrentOuterX1<G>);
  } else if (artemis_problem == "whistler_wave") {
    pman->app_input->InitMeshBlockUserData = whistler_wave::Init_WHISTLER_WAVE_Params;
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
