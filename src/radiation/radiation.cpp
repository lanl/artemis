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
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/opacity/opacity.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::MeanOpacity;
using ArtemisUtils::MeanScattering;

namespace rad {

std::shared_ptr<StateDescriptor> InitializeRadDataFields(ParameterInput *pin) {

  // instantiate a state descriptor for the fields
  auto rad_fs = std::make_shared<StateDescriptor>("rad_fields");

  // Only one photon species
  std::vector<int> radids = {0};

  // Control field for sparse gas fields
  const std::string control_field = "rad_fields";

  // const int ndim = ProblemDimension(pin);
  // std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  // Coordinates coords = geometry::CoordSelect(sys, ndim);

  // Absorption and scattering opacity
  Metadata m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Sparse});
  //ArtemisUtils::EnrollArtemisRefinementOps(m, coords);
  //m.SetSparseThresholds(0.0, 0.0, 0.0);
  rad_fs->AddSparsePool<rad::opac::absorption>(m, control_field, radids);
  rad_fs->AddSparsePool<rad::opac::scattering>(m, control_field, radids);

  // return rad_fields state descriptor
  return rad_fs;
}

TaskStatus UpRadDataFields(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &gas_pkg = pm->packages.Get("gas");

  EOS eos_d = gas_pkg->template Param<EOS>("eos_d");
  MeanOpacity opacity_d = gas_pkg->template Param<MeanOpacity>("opacity_d");
  MeanScattering scattering_d = gas_pkg->template Param<MeanScattering>("scattering_d");

  // Packing and indexing (TODO: use dust sie, density)
  static auto desc = MakePackDescriptor<
      gas::prim::density, gas::prim::sie,
      rad::opac::absorption, rad::opac::scattering>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const int nblocks = md->NumBlocks();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

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

// task collection for updating data fields
TaskCollection UpdateRadDataFields(Mesh *pmesh) {
  TaskCollection tc;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &reg = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto upradf = tl.AddTask(none, UpRadDataFields, base.get());
  }

  return tc;
}

} // namespace rad
