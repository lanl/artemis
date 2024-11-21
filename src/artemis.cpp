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

// Artemis includes
#include "artemis.hpp"
#include "artemis_driver.hpp"
#include "damping/damping.hpp"
#include "dust/coagulation/coagulation.hpp"
#include "dust/dust.hpp"
#include "gas/cooling/cooling.hpp"
#include "gas/gas.hpp"
#include "geometry/geometry.hpp"
#include "gravity/gravity.hpp"
#include "nbody/nbody.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/history.hpp"

namespace artemis {

std::vector<TaskCollectionFnPtr> OperatorSplitTasks;

//----------------------------------------------------------------------------------------
//! \fn  Packages_t Artemis::ProcessPackages
//! \brief Process and initialize relevant packages
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  using ArtemisUtils::BCChoice;
  Packages_t packages;

  // Extract artemis package and params
  auto artemis = std::make_shared<StateDescriptor>("artemis");
  Params &params = artemis->AllParams();

  // Determine input file specified physics
  const bool do_gas = pin->GetOrAddBoolean("physics", "gas", true);
  const bool do_dust = pin->GetOrAddBoolean("physics", "dust", false);
  const bool do_gravity = pin->GetOrAddBoolean("physics", "gravity", false);
  const bool do_rotating_frame = pin->GetOrAddBoolean("physics", "rotating_frame", false);
  const bool do_cooling = pin->GetOrAddBoolean("physics", "cooling", false);
  const bool do_damping = pin->GetOrAddBoolean("physics", "damping", false);
  const bool do_nbody = pin->GetOrAddBoolean("physics", "nbody", false);
  const bool do_viscosity = pin->GetOrAddBoolean("physics", "viscosity", false);
  const bool do_conduction = pin->GetOrAddBoolean("physics", "conduction", false);
  const bool do_coagulation = pin->GetOrAddBoolean("physics", "coagulation", false);
  artemis->AddParam("do_gas", do_gas);
  artemis->AddParam("do_dust", do_dust);
  artemis->AddParam("do_gravity", do_gravity);
  artemis->AddParam("do_rotating_frame", do_rotating_frame);
  artemis->AddParam("do_cooling", do_cooling);
  artemis->AddParam("do_viscosity", do_viscosity);
  artemis->AddParam("do_conduction", do_conduction);
  artemis->AddParam("do_damping", do_damping);
  artemis->AddParam("do_nbody", do_nbody);
  artemis->AddParam("do_coagulation", do_coagulation);
  PARTHENON_REQUIRE(!(do_cooling) || (do_cooling && do_gas),
                    "Cooling requires the gas package, but there is not gas!");
  PARTHENON_REQUIRE(!(do_viscosity) || (do_viscosity && do_gas),
                    "Viscosity requires the gas package, but there is not gas!");
  PARTHENON_REQUIRE(!(do_conduction) || (do_conduction && do_gas),
                    "Conduction requires the gas package, but there is not gas!");
  PARTHENON_REQUIRE(!(do_coagulation) || (do_coagulation && do_dust),
                    "Coagulation requires the dust package, but there is not dust!");
  artemis->AddParam("do_diffusion", do_conduction || do_viscosity);

  // Store problem boundary conditions in params
  artemis->AddParam("ix1_bc", BCChoice(pin->GetOrAddString("problem", "ix1_bc", "none")));
  artemis->AddParam("ox1_bc", BCChoice(pin->GetOrAddString("problem", "ox1_bc", "none")));
  artemis->AddParam("ix2_bc", BCChoice(pin->GetOrAddString("problem", "ix2_bc", "none")));
  artemis->AddParam("ox2_bc", BCChoice(pin->GetOrAddString("problem", "ox2_bc", "none")));
  artemis->AddParam("ix3_bc", BCChoice(pin->GetOrAddString("problem", "ix3_bc", "none")));
  artemis->AddParam("ox3_bc", BCChoice(pin->GetOrAddString("problem", "ox3_bc", "none")));

  // Call package initializers here
  if (do_gas) packages.Add(Gas::Initialize(pin.get()));
  if (do_dust) {
    packages.Add(Dust::Initialize(pin.get()));
  }
  if (do_gravity) packages.Add(Gravity::Initialize(pin.get()));
  if (do_rotating_frame) packages.Add(RotatingFrame::Initialize(pin.get()));
  if (do_cooling) packages.Add(Gas::Cooling::Initialize(pin.get()));
  if (do_damping) packages.Add(Damping::Initialize(pin.get()));
  if (do_nbody) packages.Add(NBody::Initialize(pin.get()));
  if (do_coagulation) packages.Add(Dust::Coagulation::Initialize(pin.get()));

  // Set coordinate system
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys);
  artemis->AddParam("coords", coords);
  artemis->AddParam("coord_sys", sys);

  // Assign geometry-specific FillDerived functions
  if (do_gas || do_dust) {
    typedef Coordinates C;
    typedef MeshData<Real> MD;
    if (coords == C::cartesian) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::cartesian>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::cartesian>;
    } else if (coords == C::spherical) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::spherical>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::spherical>;
    } else if (coords == C::cylindrical) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::cylindrical>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::cylindrical>;
    } else if (coords == C::axisymmetric) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::axisymmetric>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::axisymmetric>;
    } else {
      PARTHENON_FAIL("Invalid artemis/coordinate system!");
    }
  }

  // Add in operator split physics (if any)
  parthenon::MetadataFlag MetadataOperatorSplit =
      parthenon::Metadata::AddUserFlag("OperatorSplit");
  // if (do_gas) {
  //   typedef Coordinates C;
  //   if (coords == C::cartesian) {
  //     OperatorSplitTasks.push_back(&Gas::OperatorSplitExample<C::cartesian>);
  //   } else if (coords == C::spherical) {
  //     OperatorSplitTasks.push_back(&Gas::OperatorSplitExample<C::spherical>);
  //   } else if (coords == C::cylindrical) {
  //     OperatorSplitTasks.push_back(&Gas::OperatorSplitExample<C::cylindrical>);
  //   } else if (coords == C::axisymmetric) {
  //     OperatorSplitTasks.push_back(&Gas::OperatorSplitExample<C::axisymmetric>);
  //   } else {
  //     PARTHENON_FAIL("Invalid artemis/coordinate system!");
  //   }
  // }

  if (do_coagulation) {
    typedef Coordinates C;
    if (coords == C::cartesian) {
      OperatorSplitTasks.push_back(&Dust::OperatorSplitDust<C::cartesian>);
    } else if (coords == C::spherical) {
      OperatorSplitTasks.push_back(&Dust::OperatorSplitDust<C::spherical>);
    } else if (coords == C::cylindrical) {
      OperatorSplitTasks.push_back(&Dust::OperatorSplitDust<C::cylindrical>);
    } else if (coords == C::axisymmetric) {
      OperatorSplitTasks.push_back(&Dust::OperatorSplitDust<C::axisymmetric>);
    } else {
      PARTHENON_FAIL("Invalid artemis/coordinate system!");
    }
  }

  // Add in user-defined AMR criterion callback
  const bool amr_user = pin->GetOrAddBoolean("artemis", "amr_user", false);
  if (amr_user) artemis->CheckRefinementBlock = artemis::ProblemCheckRefinementBlock;

  // Add history for all packages with history output
  if (do_gas) Gas::AddHistory(coords, packages.Get("gas")->AllParams());
  if (do_dust) Dust::AddHistory(coords, packages.Get("dust")->AllParams());

  // Add artemis package
  packages.Add(artemis);

  return packages;
}

} // namespace artemis
