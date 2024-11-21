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
#ifndef DERIVED_FILL_DERIVED_HPP_
#define DERIVED_FILL_DERIVED_HPP_

// Artemis includes
#include "artemis.hpp"
#include "utils/artemis_utils.hpp"

namespace ArtemisDerived {

template <Coordinates GEOM>
TaskStatus SetAuxillaryFields(MeshData<Real> *md);

template <Coordinates GEOM>
void ConsToPrim(MeshData<Real> *md);

template <typename T, Coordinates GEOM>
void PrimToCons(T *md);

template <Coordinates GEOM>
void PostInitialization(MeshBlock *pmb, ParameterInput *pin);

} // namespace ArtemisDerived

#endif // DERIVED_FILL_DERIVED_HPP_
