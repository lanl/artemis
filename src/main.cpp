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

// NOTE(@PDM): The following is largely borrowed from the open-source LANL phoebus
// software

// Parthenon includes
#include <defs.hpp>
#include <parthenon_manager.hpp>

// Artemis includes
#include "artemis_driver.hpp"
#include "derived/fill_derived.hpp"
#include "pgen/pgen.hpp"
#include "pgen/problem_modifier.hpp"
#include "utils/artemis_utils.hpp"

parthenon::DriverStatus LaunchWorkFlow(std::string sys, parthenon::ParthenonManager &pman,
                                       parthenon::ParameterInput *pin) {
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
    ProblemModifier<Coordinates::spherical>(&pman);
    pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::spherical>;
    pman.app_input->PostInitialization =
        ArtemisDerived::PostInitialization<Coordinates::spherical>;
    pman.ParthenonInitPackagesAndMesh();
    ArtemisDriver<Coordinates::spherical> driver(pin, pman.app_input.get(),
                                                 pman.pmesh.get(), pman.IsRestart());
    return driver.Execute();
  } else if (sys == "cylindrical") { // (R,phi,z)
    ProblemModifier<Coordinates::cylindrical>(&pman);
    pman.app_input->ProblemGenerator = ProblemGenerator<Coordinates::cylindrical>;
    pman.app_input->PostInitialization =
        ArtemisDerived::PostInitialization<Coordinates::cylindrical>;
    pman.ParthenonInitPackagesAndMesh();
    ArtemisDriver<Coordinates::cylindrical> driver(pin, pman.app_input.get(),
                                                   pman.pmesh.get(), pman.IsRestart());
    return driver.Execute();
  } else if (sys == "axisymmetric") { // (R,z,phi)
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

  // Print basic facts of simulation
  auto pin = pman.pinput.get();
  if (parthenon::Globals::my_rank == 0) {
    printf("\n=====================================================\n");
    printf("  ARTEMIS\n");
    printf("    problem:     %s\n", pin->GetString("artemis", "problem").c_str());
    printf("    coordinates: %s\n", pin->GetString("artemis", "coordinates").c_str());
    printf("    MPI ranks:   %d\n", parthenon::Globals::nranks);
    printf("=====================================================\n\n");
  }

  // BC processing
  // NOTE(PDM): this should be eliminated once Parth PR#989 is merged...
  ArtemisUtils::ArtemisBoundaryCheck(pin);

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = ProcessPackages;

  // Geometry specific routines
  // (1) Handle ProblemGenerators and associated modifiers
  // (2) Call ParthenonInit to set up the mesh
  // (3) Execute driver
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  auto status = LaunchWorkFlow(sys, pman, pin);

  // Call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  if (status == DriverStatus::complete) {
    return 0;
  } else if (status == DriverStatus::failed) {
    return 1;
  } else if (status == DriverStatus::timeout) {
    return 2;
  }
  PARTHENON_WARN("Unknown driver status!");
  return 3;
}
