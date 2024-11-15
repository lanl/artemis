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
#include "gravity/gravity.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "gravity/nbody_gravity.hpp"

using namespace parthenon::package::prelude;

namespace Gravity {

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Gravity::Initialize
//! \brief Adds intialization function for gravity package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto gravity = std::make_shared<StateDescriptor>("gravity");
  Params &params = gravity->AllParams();

  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);

  const Real gm = pin->GetOrAddReal("gravity", "gm", -Big<Real>());
  params.Add("tstart",
             pin->GetOrAddReal("gravity", "tstart", std::numeric_limits<Real>::lowest()));
  params.Add("tstop", pin->GetOrAddReal("gravity", "tstop", Big<Real>()));

  // Find which gravity type is requested
  // Note that we do not use if-elif statements to detect multiple gravity blocks
  int count = 0;
  GravityType gtype = GravityType::null;
  std::string block_name = "none";

  bool needs_gm = true;
  // params specific to the gravity type
  if (pin->DoesBlockExist("gravity/uniform")) {
    needs_gm = false;
    count++;
    block_name = "gravity/uniform";
    gtype = GravityType::uniform;
    params.Add("gx1", pin->GetReal(block_name, "gx1"));
    params.Add("gx2", pin->GetReal(block_name, "gx2"));
    params.Add("gx3", pin->GetReal(block_name, "gx3"));
  }
  if (pin->DoesBlockExist("gravity/point")) {
    count++;
    gtype = GravityType::point;
    block_name = "gravity/point";
    params.Add("soft", pin->GetOrAddReal(block_name, "soft", 0.0));
    params.Add("sink", pin->GetOrAddReal(block_name, "sink", 0.0));
    params.Add("sink_rate", pin->GetOrAddReal(block_name, "sink_rate", 0.0));
    const Real x = pin->GetOrAddReal(block_name, "x", 0.0);
    const Real y = pin->GetOrAddReal(block_name, "y", 0.0);
    const Real z = pin->GetOrAddReal(block_name, "z", 0.0);

    if (geometry::is_axisymmetric(coords)) {
      PARTHENON_REQUIRE(
          (x == 0.0) && (y == 0.0) && (z == 0.0),
          "In axisymmetric coordinates, the point mass must be at the origin!");
    }
    params.Add("x", x);
    params.Add("y", y);
    params.Add("z", z);
  }
  if (pin->DoesBlockExist("gravity/binary")) {
    count++;
    gtype = GravityType::binary;
    block_name = "gravity/binary";

    PARTHENON_REQUIRE(!geometry::is_axisymmetric(coords),
                      "Binary gravity is not compatable with axisymmetric coordinates!");

    params.Add("soft1", pin->GetOrAddReal(block_name, "soft1", 0.0));
    params.Add("soft2", pin->GetOrAddReal(block_name, "soft2", 0.0));
    params.Add("sink1", pin->GetOrAddReal(block_name, "sink1", 0.0));
    params.Add("sink2", pin->GetOrAddReal(block_name, "sink2", 0.0));
    params.Add("sink_rate1", pin->GetOrAddReal(block_name, "sink_rate1", 0.0));
    params.Add("sink_rate2", pin->GetOrAddReal(block_name, "sink_rate2", 0.0));
    params.Add("x", pin->GetOrAddReal(block_name, "x", 0.0));
    params.Add("y", pin->GetOrAddReal(block_name, "y", 0.0));
    params.Add("z", pin->GetOrAddReal(block_name, "z", 0.0));

    const Real qbin = pin->GetReal(block_name, "q");
    const Real abin = pin->GetReal(block_name, "a");
    const Real ebin = pin->GetOrAddReal(block_name, "e", 0.0);
    const Real ibin = pin->GetOrAddReal(block_name, "i", 0.0) * M_PI / 180.;
    const Real obin = pin->GetOrAddReal(block_name, "omega", 0.0) * M_PI / 180.;
    const Real Obin = pin->GetOrAddReal(block_name, "Omega", 0.0) * M_PI / 180.;
    const Real fbin = pin->GetOrAddReal(block_name, "f", 180.0) * M_PI / 180.;
    Orbit orb(gm, abin, ebin, ibin, obin, Obin, fbin);
    params.Add("q", qbin);
    params.Add("orb", orb);
  }
  if (pin->DoesBlockExist("gravity/nbody")) {
    count++;
    gtype = GravityType::nbody;
    block_name = "gravity/nbody";
    const bool do_nbody = pin->GetBoolean("physics", "nbody");
    PARTHENON_REQUIRE(do_nbody, "You have <gravity/nbody> but not physics/nbody = true!");
  }

  PARTHENON_REQUIRE((gm != -Big<Real>()) && (needs_gm),
                    "Please define gm in the <gravity> block!");

  PARTHENON_REQUIRE((count > 0) && (gtype != GravityType::null), "Unknown gravity node!");

  PARTHENON_REQUIRE(count == 1, "artemis only supports 1 gravity type at this time");

  params.Add("gm", gm);
  params.Add("type", gtype);

  return gravity;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::ExternalGravity
//! \brief Wrapper function for external gravity options
template <Coordinates GEOM>
TaskStatus ExternalGravity(MeshData<Real> *md, const Real time, const Real dt) {
  auto pm = md->GetParentPointer();

  auto &pkg = pm->packages.Get("gravity");
  GravityType gtype = pkg->template Param<GravityType>("type");
  const Real tstart = pkg->template Param<Real>("tstart");
  const Real tstop = pkg->template Param<Real>("tstop");

  if ((time >= tstart) && (time < tstop)) {
    if (gtype == GravityType::uniform) {
      return UniformGravity<GEOM>(md, time, dt);
    } else if (gtype == GravityType::point) {
      return PointMassGravity<GEOM>(md, time, dt);
    } else if (gtype == GravityType::binary) {
      if constexpr (!geometry::is_axisymmetric<GEOM>()) {
        return BinaryMassGravity<GEOM>(md, time, dt);
      }
    } else if (gtype == GravityType::nbody) {
      if constexpr (!geometry::is_axisymmetric<GEOM>()) {
        auto &pkg = pm->packages.Get("nbody");
        if (pkg->template Param<int>("npart") > 0)
          return NBodyGravity<GEOM>(md, time, dt);
      }
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> MD;
template TaskStatus ExternalGravity<C::cartesian>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<C::cylindrical>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<C::spherical1D>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<C::spherical2D>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<C::spherical3D>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<C::axisymmetric>(MD *m, const Real t, const Real dt);

} // namespace Gravity
