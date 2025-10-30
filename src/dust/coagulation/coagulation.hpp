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
// NOTE(@sli):
// The dust coagulation code is modified from public available Dustpy package
//          https://github.com/stammler/dustpy
//   and from their paper (Stammler and Birnstiel (2022) ApJ 935:35)
//          "DustPy: A Python Package for Dust Evolution in Protoplanetary Disks"
//========================================================================================
#ifndef DUST_COAGULATION_COAGULATION_HPP_
#define DUST_COAGULATION_COAGULATION_HPP_

#include "utils/artemis_utils.hpp"
#include "utils/units.hpp"

namespace Dust {
namespace Coagulation {
static constexpr Real mom_scale = 1.0e10;
static constexpr Real mom_iscale = 1.0e-10;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Params &gas_params,
                                            Params &dust_params,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants);

// OperatorSplit tasks
template <Coordinates GEOM>
TaskListStatus CoagulationDriver(Mesh *pm, parthenon::SimTime &tm);

template <Coordinates GEOM>
TaskStatus CoagulationStep(MeshData<Real> *md, const Real time, const Real dt);

// Initialization functions
void InitializeArray(const int nm, int &pgrid, const Real &rho_p, const Real &chi,
                     const Real &a, const ParArray1D<Real> dsize, ParArray2D<int> klf,
                     ParArray1D<Real> mass_grid, ParArray3D<Real> coag3d,
                     ParArray3D<int> cpod_notzero, ParArray3D<Real> cpod_short);

// Diagonstics functions
using DiagPack_t = parthenon::SparsePack<gas::prim::density, gas::prim::sie,
                                         dust::cons::density, dust::cons::momentum,
                                         dust::prim::density, dust::prim::velocity>;
template <Coordinates GEOM>
void CoagulationDiagnostics(MeshData<Real> *md, DiagPack_t &vmesh,
                            const geometry::CoordParams &cpars, const Real &dfloor,
                            Real &mass_d, int &max_size);
void WriteCoagulationDiagnostics(MeshData<Real> *md, const Real time, const Real dt,
                                 const int max_size1, const int max_size0,
                                 const Real mass_d1, const Real mass_d0);

// Constants that enumerate coagulation kernel
enum coag2drv { dpod, afrag, phifrag, epsfrag, dalp, kdelta, coef_fett, last2 };

// Coagulation State Parameters
// NOTE(@pdmullen): Shared between various inline function calls
struct StateParams {
  Real gdens;
  Real alpha;
  Real cs;
  Real omega;
};

// Struct that holds coagulation parameters
struct CoagParams {
  int nm;            // nspecies
  int ncall_max;     // max coag calls
  Real rho_p;        // grain density
  Real dfloor;       // dust density floor
  int integrator;    // coag time integrator
  bool use_adaptive; // adaptive step size
  bool mom_coag;     // mom-preserving coagulation

  int pgrid;        // dust grid
  Real chi;         // chi parameter
  Real pgrow;       // Power for increasing step size
  Real pshrink;     // Power for decreasing step size
  Real err_eps;     // Relative tolerance for adaptive step sizing
  Real S;           // Safety margin for adaptive step sizing
  Real cfl;         // CFL number
  Real errcon;      // Needed for increasing step size
  bool const_omega; // for shearing-box or testing

  Real rho0;    // physical-to-code unit conversion density
  Real length0; // physical-to-code unit conversion length
};

// Coagulation Rate Parameters
// NOTE(@pdmullen): Shared between CoagulationRate calls
struct RateParams {
  bool coord; // true--surface density, false: 3D

  Real mmw;           // mean molecular weight (mu * amu in CGS)
  Real cross_section; // cross section of gas species

  Real vfrag;   // fragmentation velocity
  bool ibounce; // include bouncing
};

// Struct that holds coagulation arrays
struct CoagArrays {
  // pre-calculated arrays
  ParArray2D<int> klf;
  ParArray1D<Real> mass_grid;
  ParArray3D<Real> coagR3D;
  ParArray3D<int> cpod_notzero;
  ParArray3D<Real> cpod_short;
};

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::v_rel_ormel
//  \brief
KOKKOS_INLINE_FUNCTION
Real v_rel_ormel(const Real &tau_1, const Real &tau_2, const Real &t0, const Real &v0,
                 const Real &ts, const Real &vs, const Real &reynolds) {
  static constexpr Real ya = 1.6; // approx solution for st*=y*st1; valid for st1 << 1

  const Real tau_mx = (tau_1 >= tau_2) ? tau_1 : tau_2;
  const Real tau_mn = (tau_1 >= tau_2) ? tau_2 : tau_1;
  const Real st1 = tau_mx / t0;
  const Real st2 = tau_mn / t0;
  const Real vg2 = 1.5 * SQR(v0); // note the square

  // Return appropriate v_rel_ormel for regime
  const Real sqRe = 1.0 / std::sqrt(reynolds);
  if (tau_mx < 0.2 * ts) {
    // Very small regime
    return 1.5 * SQR((vs / ts * (tau_mx - tau_mn)));
  } else if (tau_mx < ts / ya) {
    return vg2 * (st1 - st2) / (st1 + st2) *
           (SQR(st1) / (st1 + sqRe) - SQR(st2) / (st2 + sqRe));
  } else if (tau_mx < 5.0 * ts) {
    // Eq. 17 of oc07. the second term with st_i**2.0 is negligible (assuming re>>1)
    // hulp1 = eq. 17; hulp2 = eq. 18
    const Real hulp1 =
        ((st1 - st2) / (st1 + st2) *
         (SQR(st1) / (st1 + ya * st1) - SQR(st2) / (st2 + ya * st1))); // note the -sign
    const Real hulp2 = 2.0 * (ya * st1 - sqRe) + SQR(st1) / (ya * st1 + st1) -
                       SQR(st1) / (st1 + sqRe) + SQR(st2) / (ya * st1 + st2) -
                       SQR(st2) / (st2 + sqRe);
    return vg2 * (hulp1 + hulp2);
  } else if (tau_mx < t0 / 5.0) {
    // Full intermediate regime
    const Real eps = st2 / st1; // stopping time ratio
    return vg2 * (st1 * (2.0 * ya - (1.0 + eps) +
                         2.0 / (1.0 + eps) *
                             (1.0 / (1.0 + ya) + (eps * eps * eps) / (ya + eps))));
  } else if (tau_mx < t0) {
    // now y* lies between 1.6 (st1 << 1) and 1.0 (st1>=1). the fit below fits ystar to
    // less than 1%
    const Real c3 = -0.29847604;
    const Real c2 = 0.32938936;
    const Real c1 = -0.63119577;
    const Real c0 = 1.6015125;
    const Real y_star = c0 + c1 * st1 + c2 * SQR(st1) + c3 * (st1 * st1 * st1);
    // we can then employ the same formula as before
    const Real eps = st2 / st1; // stopping time ratio
    return vg2 *
           (st1 * (2.0 * y_star - (1.0 + eps) +
                   2.0 / (1.0 + eps) *
                       (1.0 / (1.0 + y_star) + (eps * eps * eps) / (y_star + eps))));
  } else {
    // heavy particle limit
    return vg2 * (1.0 / (1.0 + st1) + 1.0 / (1.0 + st2));
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::theta
//  \brief
KOKKOS_INLINE_FUNCTION
Real theta(const Real &x) { return (x < 0 ? 0.0 : 1.0); }

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::Qplus
//  \brief Function calculates the new Q value of a particle resulting of a collision of
//         particles with masses m1, m2 and Q values Q1, Q2
KOKKOS_INLINE_FUNCTION
Real Qplus(const Real &m1, const Real &Q1, const Real &m2, const Real &Q2) {
  return (m1 * Q1 + m2 * Q2) / (m1 + m2);
}

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::CoagulationRate
//  \brief
KOKKOS_INLINE_FUNCTION
Real CoagulationRate(const int &i, const int &j, const int &nm1,
                     const StateParams &kernel4, const ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime,
                     const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                     const RateParams &rate, const int &itype) {
  const Real &mass_gridi = mass_grid(i);
  const Real &mass_gridj = mass_grid(j);
  const Real &mass_gride = mass_grid(nm1);
  if (mass_gridi + mass_gridj >= mass_gride) return 0.0;

  const Real &gdens = kernel4.gdens;
  const Real &alpha = kernel4.alpha;
  const Real &cs = kernel4.cs;
  const Real &omega = kernel4.omega;
  const Real &tau_i = stoppingTime(i);
  const Real &tau_j = stoppingTime(j);
  const Real *vel_i = &vel(3 * i);
  const Real *vel_j = &vel(3 * j);

  const Real &sig = rate.cross_section; //! cross section of gas species
  const Real &mmw = rate.mmw;           //! mean molecular weight (mu * mp)

  // Calculate some basic properties
  const Real hg = cs / omega;
  Real re = alpha * sig * gdens / (2.0 * mmw);
  if (!(rate.coord)) re *= std::sqrt(2.0 * M_PI) * hg;

  const Real tn = 1.0 / omega;
  const Real ts = tn / std::sqrt(re);
  const Real vn = std::sqrt(alpha) * cs;
  const Real vs = vn * std::pow(re, -0.25);

  // Calculate the relative velocities
  const Real c1 = 8.0 / M_PI * cs * cs * mmw;

  // Calculate Stokes number
  const Real stokes_i = tau_i * omega;
  const Real stokes_j = tau_j * omega;

  // Calculate turbulent relative velocity
  Real dv0 = v_rel_ormel(tau_i, tau_j, tn, vn, ts, vs, re);

  // Brownian motion relative velocities
  const Real mredu = (mass_gridi * mass_gridj / (mass_gridi + mass_gridj));
  dv0 += (c1 / mredu);

  Real dv2_ij;
  Real hij = 1.0;
  if (rate.coord) { // surface density
    const Real hi =
        std::min(std::sqrt(alpha / (std::min(0.5, stokes_i) * (SQR(stokes_i) + 1.0))),
                 1.0) *
        hg;
    const Real hj =
        std::min(std::sqrt(alpha / (std::min(0.5, stokes_j) * (SQR(stokes_j) + 1.0))),
                 1.0) *
        hg;
    const Real vs_i = std::min(stokes_i, 1.0) * omega * hi;
    const Real vs_j = std::min(stokes_j, 1.0) * omega * hj;
    // relative velocity from vertical settling
    dv2_ij = SQR(vs_i - vs_j);
    dv2_ij += (SQR(vel_i[0] - vel_j[0]) + SQR(vel_i[1] - vel_j[1]));
    hij = std::sqrt(2.0 * M_PI * (SQR(hi) + SQR(hj)));
  } else { // 3D
    dv2_ij =
        (SQR(vel_i[0] - vel_j[0]) + SQR(vel_i[1] - vel_j[1]) + SQR(vel_i[2] - vel_j[2]));
  }

  // After adding up all v_rel**2 take the square root
  const Real dv = std::sqrt(dv0 + dv2_ij);

  // New pf calculation: for fragmenation
  Real pf = 0.0;
  if (dv > 0.0) {
    const Real tmp = 1.5 * SQR(rate.vfrag / dv);
    pf = (tmp + 1.0) * std::exp(-tmp);
  }

  const int icoef_fett = coag2drv::coef_fett;
  const Real &coef_fettij = coagR3D(icoef_fett, i, j);
  if (itype == 0) {                 // coagulation
    if (rate.ibounce && dv > 0.0) { // including bouncing effect
      const Real froll = 1e-4;      // Heim et al.(PRL) 1999
      const Real amono = 1e-4;      // micro-size
      const Real vbounce = std::sqrt(5.0 * M_PI * amono * froll / mredu);
      if (vbounce < rate.vfrag) {
        const Real tmp = 1.5 * SQR(vbounce / dv);
        pf = (tmp + 1.0) * std::exp(-tmp); // using bouncing vel
      }
    }
    const Real pc = 1.0 - pf;
    return (coef_fettij * dv * pc / hij);
  } else { // fragmentation
    return (coef_fettij * dv * pf / hij);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void ZeroSource(parthenon::team_mbr_t const &mbr, const int &nm1,
                ScratchPad1D<Real> &source) {
  // Initialize source(*)
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { source(i) = 0.0; });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void InitializeSource(parthenon::team_mbr_t const &mbr, const int &nm1, const int &mimax,
                      ScratchPad1D<Real> &source, const ScratchPad1D<Real> &distri,
                      const ScratchPad1D<Real> &vel,
                      const ScratchPad1D<Real> &stoppingTime, const StateParams &kernel4,
                      const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                      const ParArray3D<int> &cpod_notzero,
                      const ParArray3D<Real> &cpod_short, const RateParams &rate) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // Calculate the Rate
      const Real fett_t = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 0);
      const Real Rc1 = distri(i) * distri(j) * fett_t;
      for (int nz = 0; nz < 4; nz++) {
        const int k = cpod_notzero(i, j, nz);
        if (k < 0) continue;
        const Real src_coag0 = cpod_short(i, j, nz) * Rc1;
        Kokkos::atomic_add(&source(k), src_coag0);
      }
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void FragmentationSource(parthenon::team_mbr_t const &mbr, const int &nm1,
                         const int &mimax, const int iafrag, const int iphifrag,
                         const ScratchPad1D<Real> &source,
                         const ScratchPad1D<Real> &distri, const ScratchPad1D<Real> &vel,
                         const ScratchPad1D<Real> &stoppingTime,
                         const StateParams &kernel4, const ParArray1D<Real> &mass_grid,
                         const ParArray3D<Real> &coagR3D, const ParArray2D<int> &klf,
                         const RateParams &rate) {
  // Adding fragment distribution
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int k) {
    for (int j = k; j <= mimax; j++) {
      // Calculate As(j) on fly
      Real As_j = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          if (klf(i2, j2) == j) {
            const Real fett_l = CoagulationRate(i2, j2, nm1, kernel4, vel, stoppingTime,
                                                mass_grid, coagR3D, rate, 1);
            As_j += coagR3D(iafrag, i2, j2) * distri(i2) * distri(j2) * fett_l;
          }
        }
      }
      source(k) += coagR3D(iphifrag, k, j) / mass_grid(k) * As_j;
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void CrateringSource(parthenon::team_mbr_t const &mbr, const int &nm1, const int &mimax,
                     const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &distri,
                     const ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const StateParams &kernel4,
                     const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                     const RateParams &rate) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real fett_l = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum0 -= Rf1;
    }
    source(j) += sum0;
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void FinalizeSource(parthenon::team_mbr_t const &mbr, const int &nm1, const int &mimax,
                    const int &pgrid, const int iepsfrag,
                    const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &distri,
                    const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stoppingTime,
                    const StateParams &kernel4, const ParArray1D<Real> &mass_grid,
                    const ParArray3D<Real> &coagR3D, const RateParams &rate) {
  // Cratering
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0 = 0.0;
    for (int j = 0; j <= i - pgrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coagR3D(iepsfrag, i, j) * Rf1;
      sum0 += dummy;
    }

    if (i - pgrid - 1 >= 0) {
      Kokkos::atomic_add(&source(i - 1), sum0);
    }

    Real sum1 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum1 -= Rf1;
    }
    Kokkos::atomic_add(&source(i), sum1);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void ZeroSourceNQ(parthenon::team_mbr_t const &mbr, const int &n, const int &nm1,
                  ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                  ScratchPad1D<Real> &vel) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    Q(i) = vel(n + i * 3) * mom_scale;
    nQs(i) = 0.0;
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void InitializeSourceNQ(parthenon::team_mbr_t const &mbr, const int &nm1,
                        const int &mimax, const int idalp, const int idpod,
                        const ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                        const ScratchPad1D<Real> &distri, const ScratchPad1D<Real> &vel,
                        const ScratchPad1D<Real> &stoppingTime,
                        const StateParams &kernel4, const ParArray1D<Real> &mass_grid,
                        const ParArray3D<Real> &coagR3D,
                        const ParArray3D<int> &cpod_notzero,
                        const ParArray3D<Real> &cpod_short, const RateParams &rate) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
      const Real fett_t = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 0);
      const Real Rc1 = distri(i) * distri(j) * fett_t;
      const Real &mass_gridi = mass_grid(i);
      const Real &mass_gridj = mass_grid(j);
      const Real &dalpji = coagR3D(idalp, j, i);
      const Real &dpodji = coagR3D(idpod, j, i);
      const Real &dalpij = coagR3D(idalp, i, j);
      const Real &dpodij = coagR3D(idpod, i, j);

      const Real Qp1 = Qplus(mass_gridi, Q(i), mass_gridj, Q(j));
      const Real dphijQ = dalpji * Qp1 * (1.0 + dpodji) - Q(j);
      const Real dphjiQ = dalpij * Qp1 * (1.0 + dpodij) - Q(i);

      const Real tmp1 = dphijQ - Qp1 * dpodji;
      const Real tmp2 = dphjiQ - Qp1 * dpodij;
      for (int nz = 0; nz < 4; nz++) {
        const int k = cpod_notzero(i, j, nz);
        if (k < 0) continue;
        const bool kdeltajk = (j == k);
        const bool kdeltaik = (i == k);
        const Real nqs_k =
            (Qp1 * cpod_short(i, j, nz) + tmp1 * kdeltajk + tmp2 * kdeltaik) * Rc1;
        Kokkos::atomic_add(&nQs(k), nqs_k);
      }
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void FragmentationSourceNQ(parthenon::team_mbr_t const &mbr, const int &nm1,
                           const int &mimax, const int &pgrid, const int iafrag,
                           const int iphifrag, const ScratchPad1D<Real> &Q,
                           ScratchPad1D<Real> &nQs, const ScratchPad1D<Real> &distri,
                           const ScratchPad1D<Real> &vel,
                           const ScratchPad1D<Real> &stoppingTime,
                           const StateParams &kernel4, const ParArray1D<Real> &mass_grid,
                           const ParArray3D<Real> &coagR3D, const Real &chi,
                           const RateParams &rate) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int k) {
    for (int j = k; j <= mimax; j++) {
      // calculate As(j) on fly
      Real As_j = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          int klf = (j2 <= i2 - pgrid - 1) ? j2 : i2;
          if (klf == j) {
            const Real fett_l = CoagulationRate(i2, j2, nm1, kernel4, vel, stoppingTime,
                                                mass_grid, coagR3D, rate, 1);
            Real Qf1;
            const Real &mass_gridi2 = mass_grid(i2);
            const Real &mass_gridj2 = mass_grid(j2);
            const Real &afragi2j2 = coagR3D(iafrag, i2, j2);
            if (j2 <= i2 - pgrid - 1) {
              Qf1 = Qplus(chi * mass_gridj2, Q(i2), mass_gridj2, Q(j2));
            } else {
              Qf1 = Qplus(mass_gridi2, Q(i2), mass_gridj2, Q(j2));
            }
            As_j += afragi2j2 * distri(i2) * distri(j2) * fett_l * Qf1;
          }
        }
      }
      const Real &mass_gridk = mass_grid(k);
      nQs(k) += coagR3D(iphifrag, k, j) / mass_gridk * As_j;
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void CrateringSourceNQ(parthenon::team_mbr_t const &mbr, const int &nm1, const int &mimax,
                       const ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                       const ScratchPad1D<Real> &distri, const ScratchPad1D<Real> &vel,
                       const ScratchPad1D<Real> &stoppingTime, const StateParams &kernel4,
                       const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                       const RateParams &rate) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    // Cratering
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real fett_l = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum0 -= Rf1;
    }
    nQs(j) += (sum0 * Q(j));
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void FinalizeSourceNQ(parthenon::team_mbr_t const &mbr, const int &nm1, const int &mimax,
                      const int &pgrid, const int iepsfrag, const ScratchPad1D<Real> &Q,
                      ScratchPad1D<Real> &nQs, const ScratchPad1D<Real> &distri,
                      const ScratchPad1D<Real> &vel,
                      const ScratchPad1D<Real> &stoppingTime, const StateParams &kernel4,
                      const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                      const RateParams &rate) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0 = 0.0;
    for (int j = 0; j <= i - pgrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coagR3D(iepsfrag, i, j) * Rf1 * Q(i);
      sum0 += dummy;
    }

    if (i - pgrid - 1 >= 0) {
      Kokkos::atomic_add(&nQs(i - 1), sum0);
    }

    Real sum1 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, nm1, kernel4, vel, stoppingTime,
                                          mass_grid, coagR3D, rate, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l * Q(i);
      sum1 -= Rf1;
    }
    Kokkos::atomic_add(&nQs(i), sum1);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void UpdateVelocityNQ(parthenon::team_mbr_t const &mbr, const int &n, const int &nm1,
                      ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                      ScratchPad1D<Real> &vel, ScratchPad1D<Real> &distri,
                      ScratchPad1D<Real> &source, const ParArray1D<Real> &mass_grid,
                      const Real &dt, const Real &dfloor) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    const Real distri_i = distri(i) + dt * source(i);
    if (distri_i > dfloor / mass_grid(i)) {
      const Real nQ1 = distri(i) * Q(i) + dt * nQs(i);
      vel(n + i * 3) = nQ1 / distri_i * mom_iscale;
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
int FindMIMax(parthenon::team_mbr_t const &mbr, const int &nm1,
              ScratchPad1D<Real> &dustdens, const ParArray1D<Real> &mass_grid,
              const Real &dfloor) {
  int mimax = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(mbr, nm1 + 1), // exclusive
      [&](const int i, int &lmax) {
        if (dustdens(i) > dfloor / mass_grid(i)) lmax = std::max(lmax, i);
      },
      Kokkos::Max<int>(mimax));
  return mimax;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
Real TimeStepControl(parthenon::team_mbr_t const &mbr, const int &nm1,
                     ScratchPad1D<Real> &dustdens, ScratchPad1D<Real> &source,
                     const ParArray1D<Real> &mass_grid, const Real &dfloor,
                     const Real &cfl) {
  Real dt_sync = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(mbr, nm1 + 1),
      [&](const int i, Real &lmin) {
        if (dustdens(i) > dfloor / mass_grid(i) && source(i) < 0.0) {
          lmin = std::min(lmin, std::abs(dustdens(i) / source(i)));
        }
      },
      Kokkos::Min<Real>(dt_sync));
  return cfl * dt_sync;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void UpdateDensity(parthenon::team_mbr_t const &mbr, const int &nm1,
                   ScratchPad1D<Real> &dustdens, ScratchPad1D<Real> &source,
                   const Real &dt) {
  // Update the density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { dustdens(i) += dt * source(i); });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void ConvertToNumberDensity(parthenon::team_mbr_t const &mbr, const int &nm1,
                            ScratchPad1D<Real> &dustdens,
                            const ParArray1D<Real> &mass_grid, const Real &dfloor) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    const Real &mass_gridi = mass_grid(i);
    dustdens(i) /= mass_gridi;
    dustdens(i) = std::max(dustdens(i), 0.01 * dfloor / mass_gridi);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::
//  \brief
KOKKOS_INLINE_FUNCTION
void ConvertToVolumeDensity(parthenon::team_mbr_t const &mbr, const int &nm1,
                            ScratchPad1D<Real> &dustdens,
                            const ParArray1D<Real> &mass_grid) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { dustdens(i) *= mass_grid(i); });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CoagulationOneCell
//  \brief
KOKKOS_INLINE_FUNCTION
void CoagulationOneCell(parthenon::team_mbr_t const &mbr, const Real &time, Real &dt_sync,
                        const Real &gdens, ScratchPad1D<Real> &dustdens,
                        ScratchPad1D<Real> &stime, ScratchPad1D<Real> &vel,
                        const int nvel, ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                        const Real &alpha, const Real &cs, const Real &omega,
                        const CoagParams &coag, const CoagArrays &coag_arrays,
                        const RateParams &rate, ScratchPad1D<Real> &source, int &nCall,
                        ScratchPad1D<Real> &Q2) {
  nCall = 0;
  Real time_dummy = time;
  Real dt_sync1 = dt_sync;
  Real time_goal = time_dummy + dt_sync1;
  Real dt = dt_sync1;
  Real hnext = dt;
  Real cfl = coag.cfl;
  dt_sync = 1e-15; // works H5

  // Params
  const int nm1 = coag.nm - 1;
  const int pgrid = coag.pgrid;
  const Real &dfloor = coag.dfloor;
  const Real &chi = coag.chi;
  const bool &do_momentum_conserving_update = coag.mom_coag;
  const bool &do_adaptive = coag.use_adaptive;
  const int &ncall_max = coag.ncall_max;
  const StateParams kernel{gdens, alpha, cs, omega};

  // Arrays
  auto &klf = coag_arrays.klf;
  auto &mass_grid = coag_arrays.mass_grid;
  auto &coagR3D = coag_arrays.coagR3D;
  auto &cpod_notzero = coag_arrays.cpod_notzero;
  auto &cpod_short = coag_arrays.cpod_short;

  // Update distribution
  ConvertToNumberDensity(mbr, nm1, dustdens, mass_grid, dfloor);
  while (std::abs(time_dummy - time_goal) > 1e-6 * dt) {
    const int mimax = FindMIMax(mbr, nm1, dustdens, mass_grid, dfloor);

    // Set source
    ZeroSource(mbr, nm1, source);
    InitializeSource(mbr, nm1, mimax, source, dustdens, vel, stime, kernel, mass_grid,
                     coagR3D, cpod_notzero, cpod_short, rate);
    FragmentationSource(mbr, nm1, mimax, coag2drv::afrag, coag2drv::phifrag, source,
                        dustdens, vel, stime, kernel, mass_grid, coagR3D, klf, rate);
    CrateringSource(mbr, nm1, mimax, source, dustdens, vel, stime, kernel, mass_grid,
                    coagR3D, rate);
    FinalizeSource(mbr, nm1, mimax, pgrid, coag2drv::epsfrag, source, dustdens, vel,
                   stime, kernel, mass_grid, coagR3D, rate);

    // Time step control
    dt_sync1 = TimeStepControl(mbr, nm1, dustdens, source, mass_grid, dfloor, cfl);
    dt = std::min(dt_sync1, time_goal - time_dummy);
    dt_sync = dt_sync1;

    // Momentum Conserving Update (iff do_momentum_conserving_update)
    for (int n = 0; n < do_momentum_conserving_update * nvel; n++) {
      ZeroSourceNQ(mbr, n, nm1, Q, nQs, vel);
      InitializeSourceNQ(mbr, nm1, mimax, coag2drv::dalp, coag2drv::dpod, Q, nQs,
                         dustdens, vel, stime, kernel, mass_grid, coagR3D, cpod_notzero,
                         cpod_short, rate);
      FragmentationSourceNQ(mbr, nm1, mimax, pgrid, coag2drv::afrag, coag2drv::phifrag, Q,
                            nQs, dustdens, vel, stime, kernel, mass_grid, coagR3D, chi,
                            rate);
      CrateringSourceNQ(mbr, nm1, mimax, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
                        coagR3D, rate);
      FinalizeSourceNQ(mbr, nm1, mimax, pgrid, coag2drv::epsfrag, Q, nQs, dustdens, vel,
                       stime, kernel, mass_grid, coagR3D, rate);
      UpdateVelocityNQ(mbr, n, nm1, Q, nQs, vel, dustdens, source, mass_grid, dt, dfloor);
    }

    // Update dust density
    UpdateDensity(mbr, nm1, dustdens, source, dt);

    // Update time and increment ncall
    time_dummy += dt;
    nCall++;
    dt_sync = (do_adaptive) ? std::max(hnext, dt_sync) : dt_sync;
    hnext = (do_adaptive) ? std::min(hnext, time_goal - time_dummy) : hnext;

    // Warn and break upon reaching ncall_max
    if (nCall > ncall_max) {
      PARTHENON_WARN("(Coagulation): Reach ncall_max in coagulation kernel!");
      break;
    }
  }
  ConvertToVolumeDensity(mbr, nm1, dustdens, mass_grid);

} // end of CoagulationOneCell
//----------------------------------------------------------------------------------------

} // namespace Coagulation
} // namespace Dust

#endif // DUST_COAGULATION_COAGULATION_HPP_
