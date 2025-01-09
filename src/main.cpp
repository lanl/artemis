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

// Parthenon includes
#include <defs.hpp>
#include <parthenon_manager.hpp>

// Artemis includes
#include "artemis_driver.hpp"
#include "derived/fill_derived.hpp"
#include "pgen/pgen.hpp"
#include "pgen/problem_modifier.hpp"
#include "utils/artemis_utils.hpp"

parthenon::DriverStatus LaunchWorkFlow(parthenon::ParthenonManager &pman,
                                       parthenon::ParameterInput *pin) {
  // Geometry specific routines
  // (1) Handle ProblemGenerators and associated modifiers
  // (2) Call ParthenonInit to set up the mesh and packages
  // (3) Execute driver
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  const int nx[3] = {pin->GetInteger("parthenon/mesh", "nx1"),
                     pin->GetInteger("parthenon/mesh", "nx2"),
                     pin->GetInteger("parthenon/mesh", "nx3")};

  const bool three_d = (nx[2] > 1);
  const bool two_d = (nx[1] > 1) && (!three_d);
  const bool one_d = (!two_d) && (!three_d);
  using namespace artemis;
  if (sys == "cartesian") { // (x,y,z)
    ProblemModifier<Coordinates::cartesian>(&pman);
    pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::cartesian>;
    pman.app_input->PostInitialization =
        ArtemisDerived::PostInitialization<Coordinates::cartesian>;
    pman.ParthenonInitPackagesAndMesh();
    ArtemisDriver<Coordinates::cartesian> driver(pin, pman.app_input.get(),
                                                 pman.pmesh.get(), pman.IsRestart());
    return driver.Execute();
  } else if (sys == "spherical") { // (r,theta,phi)
    if (one_d) {
      ProblemModifier<Coordinates::spherical1D>(&pman);
      pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::spherical1D>;
      pman.app_input->PostInitialization =
          ArtemisDerived::PostInitialization<Coordinates::spherical1D>;
      pman.ParthenonInitPackagesAndMesh();
      ArtemisDriver<Coordinates::spherical1D> driver(pin, pman.app_input.get(),
                                                     pman.pmesh.get(), pman.IsRestart());
      return driver.Execute();
    } else if (two_d) {
      ProblemModifier<Coordinates::spherical2D>(&pman);
      pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::spherical2D>;
      pman.app_input->PostInitialization =
          ArtemisDerived::PostInitialization<Coordinates::spherical2D>;
      pman.ParthenonInitPackagesAndMesh();
      ArtemisDriver<Coordinates::spherical2D> driver(pin, pman.app_input.get(),
                                                     pman.pmesh.get(), pman.IsRestart());
      return driver.Execute();
    } else {
      ProblemModifier<Coordinates::spherical3D>(&pman);
      pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::spherical3D>;
      pman.app_input->PostInitialization =
          ArtemisDerived::PostInitialization<Coordinates::spherical3D>;
      pman.ParthenonInitPackagesAndMesh();
      ArtemisDriver<Coordinates::spherical3D> driver(pin, pman.app_input.get(),
                                                     pman.pmesh.get(), pman.IsRestart());

      return driver.Execute();
    }
  } else if (sys == "cylindrical") { // (R,phi,z)
    PARTHENON_REQUIRE(
        (nx[1] > 1),
        "nx2 = 1. To run an axisymmetric, please use axisymmetric coordinates")
    ProblemModifier<Coordinates::cylindrical>(&pman);
    pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::cylindrical>;
    pman.app_input->PostInitialization =
        ArtemisDerived::PostInitialization<Coordinates::cylindrical>;
    pman.ParthenonInitPackagesAndMesh();
    ArtemisDriver<Coordinates::cylindrical> driver(pin, pman.app_input.get(),
                                                   pman.pmesh.get(), pman.IsRestart());
    return driver.Execute();
  } else if (sys == "axisymmetric") { // (R,z,phi)
    PARTHENON_REQUIRE(!three_d, "axisymmetric is only valid for 1D & 2D. To run 3D, "
                                "please use cylindrical coordinates!");
    ProblemModifier<Coordinates::axisymmetric>(&pman);
    pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::axisymmetric>;
    pman.app_input->PostInitialization =
        ArtemisDerived::PostInitialization<Coordinates::axisymmetric>;
    pman.ParthenonInitPackagesAndMesh();
    ArtemisDriver<Coordinates::axisymmetric> driver(pin, pman.app_input.get(),
                                                    pman.pmesh.get(), pman.IsRestart());
    return driver.Execute();
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }
  return DriverStatus::failed;
}

//----------------------------------------------------------------------------------------
//! \fn int main
//! \brief
int main(int argc, char *argv[]) {
  using namespace artemis;
  parthenon::ParthenonManager pman;

  // Set up kokkos and read pin
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = ProcessPackages;

  // Use Parthenon default reflecting boundary conditions
  pman.app_input->RegisterDefaultReflectingBoundaryConditions();

  // Run artemis
  auto status = LaunchWorkFlow(pman, pman.pinput.get());

  // Call MPI_Finalize and Kokkos::finalize if necessary
  // MPI and Kokkos can no longer be used
  if (pman.ParthenonFinalize() != ParthenonStatus::complete) {
    std::cout << "ParthenonFinalize() did not complete successfully!" << std::endl;
    return 4;
  }

  if (status == DriverStatus::complete) {
    std::cout << "artemis driver complete!" << std::endl;
    return 0;
  } else if (status == DriverStatus::failed) {
    std::cout << "artemis driver failed!" << std::endl;
    return 1;
  } else if (status == DriverStatus::timeout) {
    std::cout << "artemis driver timed out!" << std::endl;
    return 2;
  }
  PARTHENON_WARN("artemis driver returned with an uknown code!");
  return 3;
}
