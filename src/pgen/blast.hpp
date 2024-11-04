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
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef PGEN_BLAST_HPP_
#define PGEN_BLAST_HPP_
//! \file blast.hpp
//! \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//!        cylindrical, and spherical coordinates.
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \struct BlastParams
//! \brief container for blast parameters
struct BlastParams {
  Real rinit;
  Real internal_energy;
  Real p0;
  Real d0;
  std::array<Real, 3> x0;
  Real dz;
  int type;
  int samples;
};

} // end anonymous namespace

namespace blast {

static BlastParams blast_params;

//----------------------------------------------------------------------------------------
//! \fn void blast::compute_overlap_cyl()
//! \brief Helper function to subsample cylindrical blast profile
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION Real compute_overlap_cyl(geometry::BBox bnds, Real rad,
                                                int samples) {
  // We have a circle at the origin of radius rad intersecting a zone given by bnds.
  if constexpr (GEOM == Coordinates::cartesian) {
    const Real dxf = (bnds.x1[1] - bnds.x1[0]) / (Real)samples;
    const Real dyf = (bnds.x2[1] - bnds.x2[0]) / (Real)samples;
    int tot = 0;
    for (int i = 0; i < samples; i++) {
      const Real xc = bnds.x1[0] + ((Real)i + 0.5) * dxf;
      for (int j = 0; j < samples; j++) {
        const Real yc = bnds.x2[0] + ((Real)j + 0.5) * dyf;
        if (SQR(xc) + SQR(yc) <= SQR(rad)) tot++;
      }
    }
    return tot * dxf * dyf;
  }
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void blast::compute_overlap_sph()
//! \brief Helper function to subsample spherical blast profile
template <Coordinates GEOM>
KOKKOS_INLINE_FUNCTION Real compute_overlap_sph(geometry::BBox bnds, Real rad,
                                                int samples) {
  // We have a circle at the origin of radius rad intersecting a zone given by bnds.
  if constexpr (GEOM == Coordinates::cartesian) {
    const Real dxf = (bnds.x1[1] - bnds.x1[0]) / (Real)samples;
    const Real dyf = (bnds.x2[1] - bnds.x2[0]) / (Real)samples;
    const Real dzf = (bnds.x3[1] - bnds.x3[0]) / (Real)samples;
    int tot = 0;
    for (int i = 0; i < samples; i++) {
      const Real xc = bnds.x1[0] + (i + 0.5) * dxf;
      for (int j = 0; j < samples; j++) {
        const Real yc = bnds.x2[0] + (j + 0.5) * dyf;
        for (int k = 0; k < samples; k++) {
          const Real zc = bnds.x3[0] + (k + 0.5) * dzf;
          if (SQR(xc) + SQR(yc) + SQR(zc) <= SQR(rad)) tot++;
        }
      }
    }
    return tot * dxf * dyf * dzf;
  } else if constexpr (GEOM == Coordinates::axisymmetric) {
    // Vol = \int r dr dp dz = r_c * dr*dp*dz
    const Real dxf = (bnds.x1[1] - bnds.x1[0]) / (Real)samples;
    const Real dyf = (bnds.x2[1] - bnds.x2[0]) / (Real)samples;
    Real dV = dxf * dyf;
    Real tot = 0.0;
    for (int i = 0; i < samples; i++) {
      const Real xc = bnds.x1[0] + (i + 0.5) * dxf;
      for (int j = 0; j < samples; j++) {
        const Real yc = bnds.x2[0] + (j + 0.5) * dyf;
        if (SQR(xc) + SQR(yc) <= SQR(rad)) tot += xc * dV;
      }
    }
    return tot;
  }
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Blast_()
//! \brief Sedov blast wave
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract blast parameters
  blast_params.rinit = pin->GetOrAddReal("problem", "radius", 1.0);
  blast_params.dz = pin->GetOrAddReal("problem", "height", 1.0);
  blast_params.internal_energy = pin->GetOrAddReal("problem", "internal_energy", 1.0);
  blast_params.p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  blast_params.d0 = pin->GetOrAddReal("problem", "d0", 1.0);
  blast_params.x0[0] = pin->GetOrAddReal("problem", "x1", 0.0);
  blast_params.x0[1] = pin->GetOrAddReal("problem", "x2", 0.0);
  blast_params.x0[2] = pin->GetOrAddReal("problem", "x3", 0.0);
  blast_params.samples = pin->GetOrAddInteger("problem", "samples", -1);
  std::string sym = pin->GetOrAddString("problem", "symmetry", "spherical");
  if (sym == "spherical") {
    blast_params.type = 1;
  } else if (sym == "cylindrical") {
    blast_params.type = 2;
  } else {
    PARTHENON_FAIL("Bad blast wave symmetry parameter in <problem>!");
  }

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  // TODO(PDM): Replace the below with a call to singularity-eos
  auto gas_pkg = pmb->packages.Get("gas");
  const Real gm1 = gas_pkg->Param<Real>("adiabatic_index") - 1.0;

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  auto pars = blast_params;

  // setup uniform ambient medium with spherical over-pressured region
  pmb->par_for(
      "blast", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        Real total_vol = coords.Volume();
        const auto &xv = coords.GetCellCenter();
        Real den = pars.d0;
        Real e0 = pars.p0 / gm1;
        Real internal_energy = 0.0;
        auto xcart = coords.ConvertToCart(xv);
        const auto &xc = coords.ConvertToCart(pars.x0);
        for (int n = 0; n < 3; n++) {
          xcart[n] -= xc[n];
        }

        if (pars.type == 1) { // spherical
          // Intersection volume
          Real vol =
              (pars.samples > 0)
                  ? compute_overlap_sph<GEOM>(coords.bnds, pars.rinit, pars.samples)
                  : ((SQR(xcart[0]) + SQR(xcart[1]) + SQR(xcart[2]) <
                      pars.rinit * pars.rinit)
                         ? total_vol
                         : 0.0);
          internal_energy = e0 * (1.0 - vol / total_vol) +
                            pars.internal_energy * vol / total_vol /
                                (4.0 * M_PI / 3.0 * pars.rinit * pars.rinit * pars.rinit);
        } else if (pars.type == 2) { // cylindrical
          Real vol =
              (pars.samples > 0)
                  ? compute_overlap_cyl<GEOM>(coords.bnds, pars.rinit, pars.samples)
                  : ((SQR(xcart[0]) + SQR(xcart[1]) + SQR(xcart[2]) <
                      pars.rinit * pars.rinit)
                         ? total_vol
                         : 0.0);
          internal_energy =
              e0 * (1.0 - vol / total_vol) +
              pars.internal_energy * vol / total_vol / (M_PI * pars.rinit * pars.rinit);
        } else {
          PARTHENON_FAIL("Blast coordinate system unrecognized!");
        }

        // velocities
        const Real vx1 = 0.0;
        const Real vx2 = 0.0;
        const Real vx3 = 0.0;

        // compute cell-centered conserved variables
        v(0, gas::prim::density(), k, j, i) = den;
        v(0, gas::prim::velocity(0), k, j, i) = vx1;
        v(0, gas::prim::velocity(1), k, j, i) = vx2;
        v(0, gas::prim::velocity(2), k, j, i) = vx3;
        v(0, gas::prim::sie(), k, j, i) = internal_energy / den;
      });
}

} // namespace blast
#endif // PGEN_BLAST_HPP_
