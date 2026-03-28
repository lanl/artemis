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

  Metadata m = Metadata({Metadata::Face, Metadata::Conserved, Metadata::Independent,
                         Metadata::WithFluxes, Metadata::FillGhost});
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
} // namespace MHD