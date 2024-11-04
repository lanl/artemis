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
#include "rotating_frame.hpp"
#include "artemis.hpp"
#include "rotating_frame_impl.hpp"

namespace RotatingFrame {

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RotatingFrame::Initialize
//! \brief Adds intialization function for rotating frame package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("rotating_frame");
  Params &params = pkg->AllParams();

  const Real omega = pin->GetReal("rotating_frame", "omega");
  const Real qshear = pin->GetOrAddReal("rotating_frame", "qshear", 0.0);

  PARTHENON_REQUIRE(omega != 0.0, "rotating_frame/omega cannot be zero! To disable, set "
                                  "physics/rotating_frame = false");

  if (pin->GetString("artemis", "coordinates") != "cartesian") {
    PARTHENON_REQUIRE(
        qshear == 0.0,
        "rotating_frame/qshear must be zero for non-Cartesian coordinate systems!");
  }

  params.Add("omega", omega);
  params.Add("qshear", qshear);

  // turn off vertical gravity in shearingbox

  return pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RotatingFrame::RotatingFrameForce
//! \brief Calculate the rotating frame body forces
//!           For cartesian this function adds:
//!            dv/dt = -2 Omega x v - Omega x Omega x r
//!            dE/dt = - v. Omega x Omega x r
//!           For cyl/axi/sph this function only adds
//!             dE/dt = - v. Omega x Omega x r
TaskStatus RotatingFrameForce(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const auto coords = artemis_pkg->Param<Coordinates>("coords");

  auto &rframe_pkg = pm->packages.Get("rotating_frame");
  const Real om0 = rframe_pkg->Param<Real>("omega");
  const Real qshear = rframe_pkg->Param<Real>("qshear");

  // Switch for the different implementations based on coordinate system
  if (coords == Coordinates::cartesian) {
    return ShearingBoxImpl(md, om0, qshear, do_gas, do_dust, dt);
  } else if (coords == Coordinates::axisymmetric) {
    return RotatingFrameImpl<Coordinates::axisymmetric>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::spherical1D) {
    return RotatingFrameImpl<Coordinates::spherical1D>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::spherical2D) {
    return RotatingFrameImpl<Coordinates::spherical2D>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::spherical3D) {
    return RotatingFrameImpl<Coordinates::spherical3D>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::cylindrical) {
    return RotatingFrameImpl<Coordinates::cylindrical>(md, om0, do_gas, do_dust, dt);
  } else {
    PARTHENON_FAIL("Rotating frame is not consistent with this coordinate system");
  }

  return TaskStatus::complete;
}

} // namespace RotatingFrame
