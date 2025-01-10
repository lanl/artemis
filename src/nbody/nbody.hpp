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
#ifndef NBODY_NBODY_HPP_
#define NBODY_NBODY_HPP_

#include <parthenon/package.hpp>

#include "artemis.hpp"
#include "nbody/particle_base.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/units.hpp"

namespace NBody {

const std::string rebound_filename = "rebound_temporary_output.bin";

namespace RebAttrs {
extern Real PN;
extern Real c;
extern int include_pn2;
extern bool extras;
extern bool merge_on_collision;
} // namespace RebAttrs

struct Orbit {
  Real a;
  Real e;
  Real i;
  Real o;
  Real O;
  Real f;
};

std::map<int, ParticleParams> NBodySetup(ParameterInput *pin, const Real G, Real &mresc);

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            const ArtemisUtils::Constants &constants);

Real EstimateTimestepMesh(MeshData<Real> *md);

TaskStatus Advance(Mesh *pm, const Real time, const int stage,
                   const parthenon::LowStorageIntegrator *nbody_integ);

template <Coordinates GEOM>
AmrTag DistanceRefinement(MeshBlockData<Real> *md);

void InitializeFromRestart(Mesh *pm);

void SaveForRestart(Mesh *pm);

void Outputs(Mesh *pm, const Real time);

} // namespace NBody

#endif // NBODY_NBODY_HPP_
