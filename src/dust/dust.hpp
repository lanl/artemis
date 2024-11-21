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
#ifndef DUST_DUST_HPP_
#define DUST_DUST_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

using namespace parthenon::package::prelude;

namespace Dust {

struct CGSUnit {
  Real mass0 = 1.0;
  Real time0 = 1.0;
  Real length0 = 1.0;
  Real vol0 = 1.0;
  bool isurface_den = true;
  bool Code2PhysicalUnit_Set = false;

  bool isSet() const { return Code2PhysicalUnit_Set; }

  void SetCGSUnit(const Real &mass0_in, const Real &length0_in, const Real &time0_in,
                  const int isurf) {
    if (!Code2PhysicalUnit_Set) {
      mass0 = mass0_in;
      length0 = length0_in;
      vol0 = SQR(length0_in);
      if (isurf == 0) vol0 *= length0_in; // 3D volume
      time0 = time0_in;
      Code2PhysicalUnit_Set = true;
    }
  }

  void SetCGSUnit(ParameterInput *pin) {
    if (!Code2PhysicalUnit_Set) {
      const Real M_SUN = 1.9891e33;         // gram (sun)
      const Real AU_LENGTH = 1.49597892e13; // cm
      const Real GRAV_CONST = 6.67259e-8;   // gravitational const in cm^3 g^-1 s^-2

      Real mstar = pin->GetOrAddReal("problem", "mstar", 1.0) * M_SUN;
      length0 = pin->GetOrAddReal("problem", "r0_length", 1.0) * AU_LENGTH;
      const Real omega0 = std::sqrt(GRAV_CONST * mstar / pow(length0, 3));
      time0 = 1. / omega0;

      const Real rho0 = pin->GetReal("problem", "rho0");
      mass0 = rho0 * mstar;

      isurface_den = pin->GetOrAddBoolean("dust", "surface_density_flag", true);
      vol0 = SQR(length0);
      if (isurface_den == 0) vol0 *= length0; // 3D volume
      Code2PhysicalUnit_Set = true;
    }
  }
};

extern CGSUnit *cgsunit;
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md);

TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm);
TaskStatus FluxSource(MeshData<Real> *md, const Real dt);

template <Coordinates GEOM>
TaskStatus UpdateDustStoppingTime(MeshData<Real> *md);

TaskStatus ApplyDragForce(MeshData<Real> *md, const Real dt);

// OperatorSplit tasks
template <Coordinates GEOM>
TaskCollection OperatorSplitDust(Mesh *pm, parthenon::SimTime &tm, const Real dt);

template <Coordinates GEOM>
TaskStatus CoagulationOneStep(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION Real StoppingTime(Real dens_g, Real Cs, Real Omega_k, Real rho_p,
                                         Real dust_size) {
  if constexpr (GEOM == Coordinates::cylindrical) {
    // for surface denstiy, Stokes number = Pi/2*rho_p*s_p/sigma_g
    Real St = rho_p * dust_size / dens_g; // dimensionless
    // stopping_time = Stokes_number/Omega_k
    return (St / Omega_k);

  } else if constexpr ((GEOM == Coordinates::spherical) ||
                       (GEOM == Coordinates::axisymmetric)) {
    // for density (g/cc), Stokes number = sqrt(Pi/8)*rho_p*s_p*Omega_k/rho_g/Cs
    Real StOom = rho_p * dust_size / dens_g;
    return (StOom / Cs); // dimensionless
  }

  return (dust_size);
}

template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION void ApplyCoagulationOneCell() {}

void AddHistory(Coordinates coords, Params &params);

} // namespace Dust

#endif // DUST_DUST_HPP_
