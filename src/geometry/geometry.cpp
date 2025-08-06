//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

#include "geometry.hpp"

namespace geometry {

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RotatingFrame::Initialize
//! \brief Adds intialization function for rotating frame package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto geom = std::make_shared<StateDescriptor>("geometry");
  Params &params = geom->AllParams();

  const bool log =
      pin->GetOrAddString("artemis", "radial_spacing", "uniform") == "logarithmic";

  // Coordinates
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);

  const int nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
  const int nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
  const int nx3 = pin->GetInteger("parthenon/meshblock", "nx3");

  const std::array<int, 3> cc_shape = {std::max(1, x1dep(coords) * nx1),
                                       std::max(1, x2dep(coords) * nx2),
                                       std::max(1, x3dep(coords) * nx3)};
  const int cc_size = cc_shape[0] * cc_shape[1] * cc_shape[2];
  const int fx_size = (cc_shape[0] + 1) * cc_shape[1] * cc_shape[2];
  const int fy_size = cc_shape[0] * (cc_shape[1] + 1) * cc_shape[2];
  const int fz_size = cc_shape[0] * cc_shape[1] * (cc_shape[2] + 1);

  printf("SHAPE %d %d %d\n", cc_shape[0], cc_shape[1], cc_shape[2]);
  Metadata m =
      Metadata({Metadata::None, Metadata::OneCopy}, std::vector<int>({cc_size, 3}));
  geom->AddField<geom::xv>(m);

  return geom;
}

template <Coordinates GEOM>
void InitBlockGeom(MeshBlock *pmb, ParameterInput *pin) {
  printf("Is this working;");
  printf("INIT BLOCK GEOM");
}

parthenon::TaskStatus UpdateGeom(parthenon::MeshBlockData<Real> *md) {
  printf("UpdateGEOM\n");
  return TaskStatus::complete;
}

template void InitBlockGeom<Coordinates::cartesian>(MeshBlock *pmb, ParameterInput *pin);
template void InitBlockGeom<Coordinates::spherical1D>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::spherical2D>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::spherical3D>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::cylindrical>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::axisymmetric>(MeshBlock *pmb,
                                                       ParameterInput *pin);

} // namespace geometry