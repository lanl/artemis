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
#ifndef RADIATION_MOMENT_RADIATION_HPP_
#define RADIATION_MOMENT_RADIATION_HPP_

#include "artemis.hpp"

namespace Radiation {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md);

TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm);
TaskStatus FluxSource(MeshData<Real> *md, const Real dt);

template <Coordinates GEOM>
TaskStatus ApplyUpdate(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       const Real gam0, const Real gam1, const Real beta_dt);

void AddHistory(Coordinates coords, Params &params);

KOKKOS_INLINE_FUNCTION
Real EddingtonFactor(const Real f) {
  const Real f2 = f * f;
  return (3. + 4. * f2) / (5. + 2. * std::sqrt(4. - 3. * f2));
}

KOKKOS_INLINE_FUNCTION
std::tuple<Real, Real> WaveSpeed(const Real mu, const Real f) {
  const Real f2 = f * f;
  const Real det = 4. - 3 * f2;
  const Real sdet = std::sqrt(det);
  const Real fac = std::sqrt(2. / 3 * (det - sdet) + 2 * mu * mu * (2. - f2 - sdet));
  const Real norm = 1. / (sdet + Fuzz<Real>());
  return {norm * (mu * f - fac), norm * (mu * f + fac)};
}

KOKKOS_INLINE_FUNCTION
std::array<Real, 3> NormalizeFlux(const Real fx1, const Real fx2, const Real fx3) {
  Real f = std::sqrt(SQR(fx1) + SQR(fx2) + SQR(fx3));
  const Real nx1 = fx1 / (f + Fuzz<Real>());
  const Real nx2 = fx2 / (f + Fuzz<Real>());
  const Real nx3 = fx3 / (f + Fuzz<Real>());
  f = std::min(1.0, f);

  return {nx1 * f, nx2 * f, nx3 * f};
}

template <Coordinates GEOM>
Real EstimateTimeStep(parthenon::Mesh *pmesh) {
  auto &radiation_pkg = pmesh->packages.Get("radiation");
  auto &params = radiation_pkg->AllParams();
  Real dxmin = Big<Real>();
  for (auto const &pmb : pmesh->block_list) {
    //   if constexpr (geometry::is_cartesian<GEOM>()) {
    const auto &reg = pmb->block_size;
    for (int d = 0; d < pmesh->ndim; d++) {
      const Real dx = (reg.xmax_[d] - reg.xmin_[d]) / reg.nx_[d];
      dxmin = std::min(dxmin, dx);
    }
    // } else {
    //   const auto &md = pmb->meshblock_data.Get().get();
    //   PackDescriptor desc;
    //   auto vmesh = desc.GetPack(md);
    //   IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    //   IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    //   IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    //   Real min_dx = Big<Real>();
    //   const auto ndim = pmesh->ndim;
    //  parthenon::par_reduce(
    //     parthenon::loop_pattern_mdrange_tag, "Radiation::EstimateTimestepMesh",
    //     DevExecSpace(), 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    //     KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldx_m)
    //     {
    //     // Extract coordinates
    //     geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
    //     const auto &dx = coords.GetCellWidths();
    //     Real dx_m = Big<Real>();
    //     for (int d = 0; d < ndim; d++) {
    //       dx_m = std::min(dx_m, dx[d]);
    //     }
    //     ldx_m = std::min(ldx_m, dx_m);
    //   },
    //   Kokkos::Min<Real>(min_dx));

    //   dxmin = std::min(dxmin, min_dx);
    // }
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &dxmin, 1, MPI_PARTHENON_REAL, MPI_MIN,
                                    MPI_COMM_WORLD));
#endif
  const auto chat = params.template Get<Real>("chat");
  const auto cfl = params.template Get<Real>("cfl");
  return cfl * dxmin / chat;
}

} // namespace Radiation

#endif // RADIATION_MOMENT_RADIATION_HPP_
