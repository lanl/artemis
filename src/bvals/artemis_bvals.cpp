//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#include "bvals/artemis_bvals.hpp"
#include "artemis.hpp"
//#include "pgen/pgen.hpp"

namespace artemis {

ArtemisBC BoundaryChoice(std::string bc) {
  // These are our supported BCs

  if (bc == "reflecting") {
    return ArtemisBC::reflect;
  } else if (bc == "outflow") {
    return ArtemisBC::outflow;
  } else if (bc == "extrapolate") {
    return ArtemisBC::extrap;
  } else if (bc == "inflow") {
    return ArtemisBC::inflow;
  } else if (bc == "ic") {
    return ArtemisBC::ic;
  } else if (bc == "user") {
    return ArtemisBC::user;
  } else if (bc == "periodic") {
    return ArtemisBC::periodic;
  } else if (bc == "none") {
    return ArtemisBC::none;
  } else {
    std::stringstream msg;
    msg << bc << " is not a valid BC choice!";
    PARTHENON_FAIL(msg);
  }
  return ArtemisBC::none;
}

void ArtemisBoundaryCheck(ParameterInput *pin) {
  // This transfers the ownership of a boundary condition from parthenon to artemis
  // For example, before this function is called the input file may look like:
  //
  // <parthenon/mesh>
  //  ix1_bc = outflow
  // <problem>
  //
  // And after this function it will look like
  //
  // <parthenon/mesh>
  //  ix1_bc = user
  // <problem>
  //  ix1_bc = outflow

  std::string pgen = pin->GetString("artemis", "problem");
  std::string sys = pin->GetString("artemis", "coordinates");

  std::vector<std::string> boundaries = {"ix1_bc", "ox1_bc", "ix2_bc",
                                         "ox2_bc", "ix3_bc", "ox3_bc"};
  for (auto bc : boundaries) {
    std::string choice = pin->GetString("parthenon/mesh", bc);
    // Transfer this bc if it is ours
    if (choice != "periodic") {
      pin->SetString("parthenon/mesh", bc, "user");
      pin->SetString("problem", bc, choice);
    }
  }
}

} // namespace artemis
