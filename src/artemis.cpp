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
#include "drag/drag.hpp"
#include "dust/dust.hpp"
#include "gas/cooling/cooling.hpp"
#include "gas/gas.hpp"
#include "geometry/geometry.hpp"
#include "gravity/gravity.hpp"
#include "nbody/nbody.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/history.hpp"
#include "utils/units.hpp"

// Jaybenne includes
#include "jaybenne.hpp"

namespace artemis {

//----------------------------------------------------------------------------------------
//! \fn  Packages_t Artemis::ProcessPackages
//! \brief Process and initialize relevant packages
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;

  // Extract artemis package and params
  auto artemis = std::make_shared<StateDescriptor>("artemis");
  Params &params = artemis->AllParams();

  // Store selected pgen name
  artemis->AddParam("pgen_name", pin->GetString("artemis", "problem"));
  artemis->AddParam("job_name", pin->GetString("parthenon/job", "problem_id"));
  std::array<int, 3> nx{pin->GetInteger("parthenon/mesh", "nx1"),
                        pin->GetInteger("parthenon/mesh", "nx2"),
                        pin->GetInteger("parthenon/mesh", "nx3")};
  artemis->AddParam("prob_dim", nx);
  std::array<int, 3> nb{pin->GetInteger("parthenon/meshblock", "nx1"),
                        pin->GetInteger("parthenon/meshblock", "nx2"),
                        pin->GetInteger("parthenon/meshblock", "nx3")};
  artemis->AddParam("mb_dim", nb);

  // Set up unit conversions for this problem
  ArtemisUtils::Units units(pin.get());
  artemis->AddParam("units", units);
  // TODO(BRR) store unit parameters for analysis usage? Pass artemis StateDescriptor to
  // units class?

  // Custom constants optionally, otherwise default to true values but in code units

  // Determine input file specified physics
  const bool do_gas = pin->GetOrAddBoolean("physics", "gas", true);
  const bool do_dust = pin->GetOrAddBoolean("physics", "dust", false);
  const bool do_gravity = pin->GetOrAddBoolean("physics", "gravity", false);
  const bool do_nbody = pin->GetOrAddBoolean("physics", "nbody", false);
  const bool do_rotating_frame = pin->GetOrAddBoolean("physics", "rotating_frame", false);
  const bool do_cooling = pin->GetOrAddBoolean("physics", "cooling", false);
  const bool do_drag = pin->GetOrAddBoolean("physics", "drag", false);
  const bool do_viscosity = pin->GetOrAddBoolean("physics", "viscosity", false);
  const bool do_conduction = pin->GetOrAddBoolean("physics", "conduction", false);
  const bool do_radiation = pin->GetOrAddBoolean("physics", "radiation", false);
  artemis->AddParam("do_gas", do_gas);
  artemis->AddParam("do_dust", do_dust);
  artemis->AddParam("do_gravity", do_gravity);
  artemis->AddParam("do_nbody", do_nbody);
  artemis->AddParam("do_rotating_frame", do_rotating_frame);
  artemis->AddParam("do_cooling", do_cooling);
  artemis->AddParam("do_drag", do_drag);
  artemis->AddParam("do_viscosity", do_viscosity);
  artemis->AddParam("do_conduction", do_conduction);
  artemis->AddParam("do_diffusion", do_conduction || do_viscosity);
  artemis->AddParam("do_radiation", do_radiation);
  PARTHENON_REQUIRE(!(do_cooling) || (do_cooling && do_gas),
                    "Cooling requires the gas package, but there is not gas!");
  PARTHENON_REQUIRE(!(do_viscosity) || (do_viscosity && do_gas),
                    "Viscosity requires the gas package, but there is not gas!");
  PARTHENON_REQUIRE(!(do_conduction) || (do_conduction && do_gas),
                    "Conduction requires the gas package, but there is not gas!");
  PARTHENON_REQUIRE(!(do_radiation) || (do_radiation && do_gas),
                    "Radiation requires the gas package, but there is not gas!");

  // Set coordinate system
  const int ndim = ProblemDimension(pin.get());
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);
  artemis->AddParam("coords", coords);
  artemis->AddParam("coord_sys", sys);

  // Call package initializers here
  if (do_gas) packages.Add(Gas::Initialize(pin.get()));
  if (do_dust) packages.Add(Dust::Initialize(pin.get()));
  if (do_gravity) packages.Add(Gravity::Initialize(pin.get()));
  if (do_rotating_frame) packages.Add(RotatingFrame::Initialize(pin.get()));
  if (do_cooling) packages.Add(Gas::Cooling::Initialize(pin.get()));
  if (do_drag) packages.Add(Drag::Initialize(pin.get()));
  if (do_nbody) packages.Add(NBody::Initialize(pin.get(), units));
  if (do_radiation) {
    auto eos_h = packages.Get("gas")->Param<EOS>("eos_h");
    auto opacity_h = packages.Get("gas")->Param<Opacity>("opacity_h");
    auto scattering_h = packages.Get("gas")->Param<Scattering>("scattering_h");
    packages.Add(jaybenne::Initialize(pin.get(), opacity_h, scattering_h, eos_h));
    PARTHENON_REQUIRE(coords == Coordinates::cartesian,
                      "Jaybenne currently supports only Cartesian coordinates!");
  }

  // Assign geometry-specific FillDerived functions
  if (do_gas || do_dust) {
    typedef Coordinates C;
    typedef MeshData<Real> MD;
    if (coords == C::cartesian) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::cartesian>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::cartesian>;
    } else if (coords == C::spherical1D) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::spherical1D>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::spherical1D>;
    } else if (coords == C::spherical2D) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::spherical2D>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::spherical2D>;
    } else if (coords == C::spherical3D) {
      artemis->PreCommFillDerivedMesh = ArtemisDerived::ConsToPrim<C::spherical3D>;
      artemis->PreFillDerivedMesh = ArtemisDerived::PrimToCons<MD, C::spherical3D>;
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

  // Add optionally enrollable operator split Metadata flag
  parthenon::MetadataFlag MetadataOperatorSplit =
      parthenon::Metadata::AddUserFlag("OperatorSplit");

  // Add in user-defined AMR criterion callback
  const bool amr_user = pin->GetOrAddBoolean("artemis", "amr_user", false);
  if (amr_user) artemis->CheckRefinementBlock = artemis::ProblemCheckRefinementBlock;

  // Add history for all packages with history output
  if (do_gas) Gas::AddHistory(coords, packages.Get("gas")->AllParams());
  if (do_dust) Dust::AddHistory(coords, packages.Get("dust")->AllParams());

  // Add artemis package
  packages.Add(artemis);

  // Report artemis simulation configuration
  const bool report = pin->GetOrAddBoolean("artemis", "print_artemis_config", true);
  if (report) ArtemisUtils::PrintArtemisConfiguration(packages);

  return packages;
}

} // namespace artemis
