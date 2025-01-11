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
#include "utils/units.hpp"

namespace Radiation {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Constants &Constants);

template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md);

TaskStatus CalculateFluxes(MeshData<Real> *md, const bool pcm);
TaskStatus FluxSource(MeshData<Real> *md, const Real dt);

template <Coordinates GEOM>
TaskStatus MatterCoupling(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt);

template <Coordinates GEOM>
TaskStatus ApplyUpdate(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       const Real gam0, const Real gam1, const Real beta_dt);

void AddHistory(Coordinates coords, Params &params);

template <Fluid CTYP>
KOKKOS_INLINE_FUNCTION Real EddingtonFactor(const Real f) {
  if constexpr (CTYP == Fluid::greyP1) {
    return 1. / 3.;
  }
  const Real f2 = f * f;
  return (3. + 4. * f2) / (5. + 2. * std::sqrt(4. - 3. * f2));
}

template <Fluid CTYP>
KOKKOS_INLINE_FUNCTION std::array<Real, 6>
EddingtonTensor(const std::array<Real, 3> fred) {

  if constexpr (CTYP == Fluid::greyP1) {
    return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0};
  }

  Real fmag = std::sqrt(SQR(fred[0]) + SQR(fred[1]) + SQR(fred[2]));
  std::array<Real, 3> n{fred[0] / (fmag + Fuzz<Real>()), fred[1] / (fmag + Fuzz<Real>()),
                        fred[2] / (fmag + Fuzz<Real>())};
  fmag = std::min(1.0, fmag);
  const std::array<Real, 3> f{n[0] * fmag, n[1] * fmag, n[2] * fmag};
  const Real chi = EddingtonFactor<CTYP>(fmag);
  const Real ca = 0.5 * (1. - chi);
  const Real cb = 0.5 * (3 * chi - 1.);
  return {ca + cb * n[0] * n[0], ca + cb * n[1] * n[1], ca + cb * n[2] * n[2],
          cb * n[1] * n[2],      cb * n[0] * n[2],      cb * n[0] * n[1]};
}

template <Fluid CTYP>
KOKKOS_INLINE_FUNCTION std::tuple<Real, Real> WaveSpeed(const Real mu, const Real f) {
  if constexpr (CTYP == Fluid::greyP1) {
    Real val = std::sqrt(1. / 3);
    return {-val, val};
  }

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

KOKKOS_INLINE_FUNCTION
std::tuple<Real, Real> EnergyRHS(const Real cv, const Real a, const Real b, const Real d,
                                 const Real e, const Real e0, const Real Er, const Real B,
                                 const Real coverc) {
  const Real R = (a * (B - Er) + b * Er + d);
  const Real Fi = (e - e0) + coverc * R;
  return {R, Fi};
}

KOKKOS_INLINE_FUNCTION
std::tuple<Real, Real> TemperatureCoeffs(const Real cv, const Real a, const Real b,
                                         const Real d, const Real dB, const Real coverc) {
  const Real c1 = cv + coverc * a * dB;
  const Real c2 = b - a;
  return {c1, c2};
}

KOKKOS_INLINE_FUNCTION
std::tuple<Real, Real> PlanckEnergy(const Real ar, const Real T) {
  const Real T2 = SQR(T);
  const Real B = ar * SQR(T2);
  const Real dB = 4.0 * ar * T2 * T;
  return {B, dB};
}

KOKKOS_INLINE_FUNCTION
Real FleckFactor(const Real ar, const Real T, const Real cv) {
  return 4.0 * ar * T * T * T / cv;
}

KOKKOS_INLINE_FUNCTION
std::tuple<Real, Real, Real> EnergyExchangeCoeffs(const Real sigp, const Real sigr,
                                                  std::array<Real, 3> v,
                                                  std::array<Real, 3> F,
                                                  std::array<Real, 6> fedd, const Real c,
                                                  const Real chat) {
  // R = a * (B - E) + b * E  + d
  Real beta2 = (SQR(v[0]) + SQR(v[1]) + SQR(v[2])) / (c * c);
  Real bdf = (v[0] * F[0] + v[1] * F[1] + v[2] * F[2]) / (c * c);
  Real fb[3] = {fedd[TensIdx::X11] * v[0] / c + fedd[TensIdx::X12] * v[1] / c +
                    fedd[TensIdx::X13] * v[2] / c,
                fedd[TensIdx::X12] * v[0] / c + fedd[TensIdx::X22] * v[1] / c +
                    fedd[TensIdx::X23] * v[2] / c,
                fedd[TensIdx::X13] * v[0] / c + fedd[TensIdx::X23] * v[1] / c +
                    fedd[TensIdx::X33] * v[2] / c};
  Real fbf = fb[0] * v[0] / c + fb[1] * v[1] / c + fb[2] * v[2] / c;

  Real a = chat * sigp * (1. + 0.5 * beta2);
  Real b = chat * (sigr - sigp) * (beta2 + fbf);
  Real d = chat * (sigp + (sigp - sigr)) * bdf;
  return {a, b, d};
}

KOKKOS_INLINE_FUNCTION
std::tuple<Real, Real, std::array<Real, 3>>
MomentumExchangeCoeffs(const Real sigp, const Real sigr, const std::array<Real, 3> beta,
                       const Real B, std::array<Real, 6> fedd, const Real Er,
                       const Real c, const Real chat) {

  const Real beta2 = SQR(beta[0]) + SQR(beta[1]) + SQR(beta[2]);
  const Real a = sigr * (1. + 0.5 * beta2);
  const Real b = 2 * (sigr - sigp);
  const std::array<Real, 3> bdp = {
      beta[0] * fedd[TensIdx::X11] + beta[1] * fedd[TensIdx::X12] +
          beta[2] * fedd[TensIdx::X13],
      beta[0] * fedd[TensIdx::X12] + beta[1] * fedd[TensIdx::X22] +
          beta[2] * fedd[TensIdx::X23],
      beta[0] * fedd[TensIdx::X13] + beta[1] * fedd[TensIdx::X23] +
          beta[2] * fedd[TensIdx::X33]};

  const std::array<Real, 3> d{sigr * (bdp[0] + beta[0]) * Er + sigp * (B - Er) * beta[0],
                              sigr * (bdp[1] + beta[1]) * Er + sigp * (B - Er) * beta[1],
                              sigr * (bdp[2] + beta[2]) * Er + sigp * (B - Er) * beta[2]};
  return {a, b, d};
}

} // namespace Radiation

#endif // RADIATION_MOMENT_RADIATION_HPP_
