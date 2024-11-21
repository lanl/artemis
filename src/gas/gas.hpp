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
#ifndef GAS_GAS_HPP_
#define GAS_GAS_HPP_

#include "artemis.hpp"

namespace Gas {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md);

TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm);
TaskStatus FluxSource(MeshData<Real> *md, const Real dt);

template <Coordinates GEOM>
TaskStatus DiffusionUpdate(MeshData<Real> *md, const Real dt);

template <Coordinates GEOM>
TaskStatus ThermalFlux(MeshData<Real> *md);
template <Coordinates GEOM>
TaskStatus ViscousFlux(MeshData<Real> *md);

TaskStatus ZeroDiffusionFlux(MeshData<Real> *md);

void AddHistory(Coordinates coords, Params &params);

// // Example OperatorSplit useage
// template <Coordinates GEOM>
// TaskCollection OperatorSplitExample(Mesh *pm, parthenon::SimTime &tm, const Real dt);
// template <Coordinates GEOM>
// TaskStatus PrintMessage(MeshData<Real> *md);

//----------------------------------------------------------------------------------------
//! External template instantiations
extern template Real EstimateTimestepMesh<Coordinates::cartesian>(MeshData<Real> *md);
extern template Real EstimateTimestepMesh<Coordinates::cylindrical>(MeshData<Real> *md);
extern template Real EstimateTimestepMesh<Coordinates::spherical>(MeshData<Real> *md);
extern template Real EstimateTimestepMesh<Coordinates::axisymmetric>(MeshData<Real> *md);

extern template TaskStatus ViscousFlux<Coordinates::cartesian>(MeshData<Real> *md);
extern template TaskStatus ViscousFlux<Coordinates::spherical>(MeshData<Real> *md);
extern template TaskStatus ViscousFlux<Coordinates::cylindrical>(MeshData<Real> *md);
extern template TaskStatus ViscousFlux<Coordinates::axisymmetric>(MeshData<Real> *md);

extern template TaskStatus ThermalFlux<Coordinates::cartesian>(MeshData<Real> *md);
extern template TaskStatus ThermalFlux<Coordinates::spherical>(MeshData<Real> *md);
extern template TaskStatus ThermalFlux<Coordinates::cylindrical>(MeshData<Real> *md);
extern template TaskStatus ThermalFlux<Coordinates::axisymmetric>(MeshData<Real> *md);

extern template TaskStatus DiffusionUpdate<Coordinates::cartesian>(MeshData<Real> *md,
                                                                   const Real dt);
extern template TaskStatus DiffusionUpdate<Coordinates::spherical>(MeshData<Real> *md,
                                                                   const Real dt);
extern template TaskStatus DiffusionUpdate<Coordinates::cylindrical>(MeshData<Real> *md,
                                                                     const Real dt);
extern template TaskStatus DiffusionUpdate<Coordinates::axisymmetric>(MeshData<Real> *md,
                                                                      const Real dt);

// extern template TaskStatus PrintMessage<Coordinates::cartesian>(MeshData<Real> *md);
// extern template TaskStatus PrintMessage<Coordinates::spherical>(MeshData<Real> *md);
// extern template TaskStatus PrintMessage<Coordinates::cylindrical>(MeshData<Real> *md);
// extern template TaskStatus PrintMessage<Coordinates::axisymmetric>(MeshData<Real> *md);

// extern template TaskCollection
// OperatorSplitExample<Coordinates::cartesian>(Mesh *pm, parthenon::SimTime &tm, const
// Real dt); extern template TaskCollection
// OperatorSplitExample<Coordinates::spherical>(Mesh *pm, parthenon::SimTime &tm, const
// Real dt); extern template TaskCollection
// OperatorSplitExample<Coordinates::cylindrical>(Mesh *pm, parthenon::SimTime &tm, const
// Real dt); extern template TaskCollection
// OperatorSplitExample<Coordinates::axisymmetric>(Mesh *pm, parthenon::SimTime &tm, const
// Real dt);

} // namespace Gas

#endif // GAS_GAS_HPP_
