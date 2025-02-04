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
#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"
#include "utils/units.hpp"

namespace Gravity {

enum class GravityType { uniform, point, binary, nbody, null };

//----------------------------------------------------------------------------------------
//! \struct Orbit
//! \brief
struct Orbit {
  Real a;
  Real e;
  Real i;
  Real o;
  Real O;
  Real f0;
  Real n;
  Real coso;
  Real sino;
  Real cosO;
  Real sinO;
  Real cosI;
  Real sinI;
  Real cosf0;
  Real sinf0;

  KOKKOS_FUNCTION
  Orbit(Real mb, Real abin, Real ebin, Real ibin, Real obin, Real Obin, Real fbin) {
    a = abin;
    e = ebin;
    i = ibin;
    o = obin;
    O = Obin;
    f0 = fbin;
    n = std::sqrt(mb / (abin * abin * abin));
    coso = std::cos(o);
    sino = std::sin(o);
    cosI = std::cos(i);
    sinI = std::sin(i);
    cosO = std::cos(O);
    sinO = std::sin(O);
    cosf0 = std::cos(f0);
    sinf0 = std::sin(f0);
  }

  KOKKOS_INLINE_FUNCTION
  void solve(Real t, const Real omf, Real *pos, Real *vel) {
    const Real sint = std::sin(t * (n - omf));
    const Real cost = std::cos(t * (n - omf));
    // sin(f + t * (n - omf)), cos(f + t * (n - omf))
    Real cosf = cosf0 * cost - sinf0 * sint;
    Real sinf = cosf0 * sint + sinf0 * cost;
    const Real vb = a * n / std::sqrt(1.0 - SQR(e));
    const Real rb = a * (1.0 - SQR(e)) / (1.0 + e * cosf);
    const Real xb = rb * cosf;
    const Real yb = rb * sinf;
    const Real vxb = -sinf * vb;
    const Real vyb = (cosf + e) * vb;

    // In lab frame
    // See eg Murray&Dermott Section 2.8
    cosf = xb * coso - sino * yb;
    sinf = xb * sino + coso * yb;
    pos[0] = (cosO * cosf - sinO * sinf * cosI);
    pos[1] = (sinO * cosf + cosO * sinf * cosI);
    pos[2] = sinf * sinI;

    cosf = vxb * coso - sino * vyb;
    sinf = vxb * sino + coso * vyb;
    vel[0] = (cosO * cosf - sinO * sinf * cosI);
    vel[1] = (sinO * cosf + cosO * sinf * cosI);
    vel[2] = sinf * sinI;
  }
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            const ArtemisUtils::Constants &constants,
                                            const Packages_t &packages);

template <Coordinates GEOM>
TaskStatus UniformGravity(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
TaskStatus PointMassGravity(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
TaskStatus BinaryMassGravity(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
TaskStatus NBodyGravityFixed(MeshData<Real> *md, const Real time, const Real dt);

template <Coordinates GEOM>
TaskStatus ExternalGravity(MeshData<Real> *md, const Real time, const Real dt);

KOKKOS_INLINE_FUNCTION
Real quad_ramp(const Real x) { return SQR(x); }

} // namespace Gravity

#endif // GRAVITY_GRAVITY_HPP_
