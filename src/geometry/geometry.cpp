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

// helper macro
#define ADD_FIELD(name)                                                                  \
  shape = coords.template shape<name>();                                                 \
  pkg->AddField<name>(Metadata({Metadata::None, Metadata::OneCopy},                      \
                               std::vector<int>({shape[0] * shape[1] * shape[2]})));

template <Coordinates GEOM>
void EnrollFields(StateDescriptor *pkg, CoordParams &cpars) {

  Coords<GEOM> coords(cpars);

  std::array<int, 3> shape;

  ADD_FIELD(geom::x1v);
  ADD_FIELD(geom::x2v);
  ADD_FIELD(geom::x3v);
  ADD_FIELD(geom::vol);
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RotatingFrame::Initialize
//! \brief Adds intialization function for rotating frame package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto geom = std::make_shared<StateDescriptor>("geometry");
  Params &params = geom->AllParams();

  CoordParams cpars(pin);

  if (cpars.sys == Coordinates::cartesian) {
    EnrollFields<Coordinates::cartesian>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::axisymmetric) {
    EnrollFields<Coordinates::axisymmetric>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::cylindrical) {
    EnrollFields<Coordinates::cylindrical>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::spherical1D) {
    EnrollFields<Coordinates::spherical1D>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::spherical2D) {
    EnrollFields<Coordinates::spherical2D>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::spherical3D) {
    EnrollFields<Coordinates::spherical3D>(geom.get(), cpars);
  }

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

template void EnrollFields<Coordinates::cartesian>(StateDescriptor *pkg,
                                                   CoordParams &cpars);
template void EnrollFields<Coordinates::axisymmetric>(StateDescriptor *pkg,
                                                      CoordParams &cpars);
template void EnrollFields<Coordinates::cylindrical>(StateDescriptor *pkg,
                                                     CoordParams &cpars);
template void EnrollFields<Coordinates::spherical1D>(StateDescriptor *pkg,
                                                     CoordParams &cpars);
template void EnrollFields<Coordinates::spherical2D>(StateDescriptor *pkg,
                                                     CoordParams &cpars);
template void EnrollFields<Coordinates::spherical3D>(StateDescriptor *pkg,
                                                     CoordParams &cpars);

} // namespace geometry
