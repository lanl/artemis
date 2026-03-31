//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// C++ headers
#include <limits>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "mhd.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/history.hpp"
#include "utils/refinement/amr_criteria.hpp"
#include "utils/units.hpp"

using ArtemisUtils::VI;

namespace MHD {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor MHD::Initialize
//! \brief Adds intialization function for gas hydrodynamics package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants,
                                            Packages_t &packages) {

  auto mhd = std::make_shared<StateDescriptor>("mhd");
  Params &params = mhd->AllParams();

  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);
  const bool log =
      pin->GetOrAddString("artemis", "radial_spacing", "uniform") == "logarithmic";

  Metadata m = Metadata({Metadata::Face, Metadata::Conserved, Metadata::Independent,
                         Metadata::WithFluxes, Metadata::FillGhost});
  ArtemisUtils::EnrollArtemisFaceRefinementOps(m, coords, log);
  mhd->AddField<field::face::B>(m);
  //   m = Metadata(
  //       {Metadata::Edge, Metadata::Conserved, Metadata::Independent,
  //       Metadata::WithFluxes});
  //   mhd->AddField<field::edge::E>(m);
  //   mhd->AddField<field::edge::J>(m);
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::WithFluxes},
               std::vector<int>({3}));
  mhd->AddField<field::cell::B>(m);

  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive, Metadata::OneCopy,
                Metadata::FillGhost, Metadata::WithFluxes});
  mhd->AddField<field::cell::energy>(m);

  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  mhd->AddField<field::cell::divB>(m);

  //   m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Intensive,
  //   Metadata::OneCopy,
  //                 Metadata::FillGhost},
  //                std::vector<int>({3}));
  //   mhd->AddField<field::cell::E>(m);
  //   mhd->AddField<field::cell::J>(m);
  return mhd;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::AssembleEdgeEMF
//! \brief Runtime dispatch for geometry-aware edge EMF assembly.
TaskStatus AssembleEdgeEMF(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  const auto &artemis_pkg = pm->packages.Get("artemis");
  const auto do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  if (!do_mhd) return TaskStatus::complete;

  static auto desc = MakePackDescriptor<field::cell::B, field::face::B>(
      resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::dx1, geom::dx2, geom::dx3,
                         geom::hx1f1, geom::hx2f1, geom::hx3f1, geom::hx1f2, geom::hx2f2,
                         geom::hx3f2, geom::hx1f3, geom::hx2f3, geom::hx3f3, geom::hx1e1,
                         geom::hx2e2, geom::hx3e3>(resolved_pkgs.get());
  const auto v = desc.GetPack(md);
  const auto vg = desc_g.GetPack(md);

  const auto sys = artemis_pkg->template Param<Coordinates>("coords");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");
  typedef Coordinates G;
  if (sys == G::cartesian) {
    return AssembleEdgeEMFImpl<G::cartesian>(md, v, vg, cpars);
  } else if (sys == G::spherical3D) {
    return AssembleEdgeEMFImpl<G::spherical3D>(md, v, vg, cpars);
  } else if (sys == G::spherical1D) {
    return AssembleEdgeEMFImpl<G::spherical1D>(md, v, vg, cpars);
  } else if (sys == G::spherical2D) {
    return AssembleEdgeEMFImpl<G::spherical2D>(md, v, vg, cpars);
  } else if (sys == G::cylindrical) {
    return AssembleEdgeEMFImpl<G::cylindrical>(md, v, vg, cpars);
  } else if (sys == G::axisymmetric) {
    return AssembleEdgeEMFImpl<G::axisymmetric>(md, v, vg, cpars);
  } else {
    PARTHENON_FAIL("Coordinate type not recognized!");
  }
}

} // namespace MHD