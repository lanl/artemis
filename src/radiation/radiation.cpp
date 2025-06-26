//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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
#include "radiation.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/opacity/opacity.hpp"
#include "utils/units.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::MeanOpacity;
using ArtemisUtils::MeanScattering;

namespace Radiation {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Radiation::Initialize
//! \brief Adds intialization function for radiation package
//! NOTE(@pdmullen): ...to become a top-level package for radiation utils commmon to impl
std::shared_ptr<StateDescriptor>
Initialize(ParameterInput *pin, ArtemisUtils::Constants &constants, const bool do_imc) {
  auto radiation = std::make_shared<StateDescriptor>("radiation");
  Params &params = radiation->AllParams();

  // Metadata flags
  auto MetadataRadiation = radiation->GetMetadataFlag();
  auto MetadataOperatorSplit = Metadata::GetUserFlag("OperatorSplit");

  // Radiation constants (including chat for Moments)
  // NOTE(@pdmullen): We require constants again in moments sector (when
  // moments enabled) due to our anonymous flux machinery
  const Real light = constants.GetCCode();
  params.Add("c", light);
  const Real arad = constants.GetARCode();
  params.Add("arad", arad);
  const Real creduc = pin->GetOrAddReal("radiation/moment", "creduc", 1.0);
  params.Add("chat", light / creduc);

  // Add derived radiation fields expected by Jaybenne
  if (do_imc) {
    // Number of radiation species (i.e., groups)
    const int nspecies = pin->GetOrAddInteger("radiation/imc", "nspecies", 1);
    params.Add("nspecies", nspecies);
    PARTHENON_REQUIRE(nspecies == 1, "Jaybenne IMC only works with nspecies=1!");
    std::vector<int> fluidids;
    for (int n = 0; n < nspecies; ++n)
      fluidids.push_back(n);

    // Control field for sparse gas fields
    const std::string control_field = rad::opac::absorption::name();

    // Absorption and scattering opacity
    Metadata m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                           Metadata::Sparse, MetadataRadiation, MetadataOperatorSplit});
    radiation->AddSparsePool<rad::opac::absorption>(m, control_field, fluidids);
    radiation->AddSparsePool<rad::opac::scattering>(m, control_field, fluidids);
  }

  return radiation;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Radiation::SetOpacities
//! \brief Routine to set opacitiy fields (when required, e.g., for Jaybenne IMC)
TaskStatus SetOpacities(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &gas_pkg = pm->packages.Get("gas");

  EOS eos_d = gas_pkg->template Param<EOS>("eos_d");
  MeanOpacity opacity_d = gas_pkg->template Param<MeanOpacity>("opacity_d");
  MeanScattering scattering_d = gas_pkg->template Param<MeanScattering>("scattering_d");

  // Packing and indexing
  // TODO(): Will eventually incorporate other fluids
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::sie, rad::opac::absorption,
                         rad::opac::scattering>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  // Set opacities
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConsToPrim", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const Real &rho = vmesh(b, gas::prim::density(), k, j, i);
        const Real &sie = vmesh(b, gas::prim::sie(), k, j, i);
        const Real temp = eos_d.TemperatureFromDensityInternalEnergy(rho, sie);
        Real &aa = vmesh(b, rad::opac::absorption(), k, j, i);
        Real &ss = vmesh(b, rad::opac::scattering(), k, j, i);

        aa = opacity_d.AbsorptionCoefficient(rho, temp);
        ss = scattering_d.RosselandMeanTotalScatteringCoefficient(rho, temp);
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskCollection Radiation::UpdateRadiationFields
//! \brief TaskCollection to set radiation fields (when required, e.g., for Jaybenne IMC)
TaskCollection UpdateRadiationFields(Mesh *pmesh) {
  TaskCollection tc;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto set_opac = tl.AddTask(none, SetOpacities, base.get());
  }

  return tc;
}

} // namespace Radiation
