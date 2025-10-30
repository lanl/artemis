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

// Coagulation Kernel Parameters
// NOTE(@pdmullen): Shared between various inline function calls
struct KernelParams {
  Real gdens = Null<Real>();
  Real alpha = Null<Real>();
  Real cs = Null<Real>();
  Real omega = Null<Real>();
};

// Struct that holds coagulation parameters
struct CoagParams {
  bool coord = false; // true--surface density, false: 3D

  int nm;               // nspecies
  int ncall_max = 1000; // max coag calls
  bool ibounce = false; // include bouncing
  Real rho_p;           // grain density
  Real vfrag;           // fragmentation velocity
  Real dfloor;          // dust density floor
  int integrator;       // coag time integrator
  bool use_adaptive;    // adaptive step size
  bool mom_coag;        // mom-preserving coagulation

  int pgrid;        // dust grid
  Real chi;         // chi parameter
  Real pgrow;       // Power for increasing step size
  Real pshrink;     // Power for decreasing step size
  Real err_eps;     // Relative tolerance for adaptive step sizing
  Real S;           // Safety margin for adaptive step sizing
  Real cfl;         // CFL number
  Real errcon;      // Needed for increasing step size
  bool const_omega; // for shearing-box or testing

  Real mmw;           // mean molecular weight (mu * amu in CGS)
  Real cross_section; // cross section of gas species

  Real rho0;    // physical-to-code unit conversion density
  Real length0; // physical-to-code unit conversion length

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
Real CoagulationRate(const int &i, const int &j, const KernelParams &kernel4,
                     const ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag,
                     const int &itype) {
  const Real &mass_gridi = coag.mass_grid(i);
  const Real &mass_gridj = coag.mass_grid(j);
  const Real &mass_gride = coag.mass_grid(coag.nm - 1);
  if (mass_gridi + mass_gridj >= mass_gride) return 0.0;

  const Real &gdens = kernel4.gdens;
  const Real &alpha = kernel4.alpha;
  const Real &cs = kernel4.cs;
  const Real &omega = kernel4.omega;
  const Real &tau_i = stoppingTime(i);
  const Real &tau_j = stoppingTime(j);
  const Real *vel_i = &vel(3 * i);
  const Real *vel_j = &vel(3 * j);

  const Real &sig = coag.cross_section; //! cross section of gas species
  const Real &mmw = coag.mmw;           //! mean molecular weight (mu * mp)

  // Calculate some basic properties
  const Real hg = cs / omega;
  Real re = alpha * sig * gdens / (2.0 * mmw);
  if (!(coag.coord)) re *= std::sqrt(2.0 * M_PI) * hg;

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
  if (coag.coord) { // surface density
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
    const Real tmp = 1.5 * SQR(coag.vfrag / dv);
    pf = (tmp + 1.0) * std::exp(-tmp);
  }

  const int icoef_fett = coag2drv::coef_fett;
  const Real &coef_fettij = coag.coagR3D(icoef_fett, i, j);
  if (itype == 0) {                 // coagulation
    if (coag.ibounce && dv > 0.0) { // including bouncing effect
      const Real froll = 1e-4;      // Heim et al.(PRL) 1999
      const Real amono = 1e-4;      // micro-size
      const Real vbounce = std::sqrt(5.0 * M_PI * amono * froll / mredu);
      if (vbounce < coag.vfrag) {
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
//! \fn  void Dust::Coagulation::CoagulationSource
//  \brief
KOKKOS_INLINE_FUNCTION
void CoagulationSource(parthenon::team_mbr_t const &mbr, ScratchPad1D<Real> &source,
                       const ScratchPad1D<Real> &distri, const int &mimax,
                       const KernelParams &kernel4, const ScratchPad1D<Real> &vel,
                       const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag) {
  // Initialize source(*)
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                           [&](const int k) { source(k) = 0.0; });
  mbr.team_barrier();

  // Adding coagulation source terms
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // Calculate the Rate
      const Real fett_t = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 0);
      const Real Rc1 = distri(i) * distri(j) * fett_t;
      for (int nz = 0; nz < 4; nz++) {
        const int k = coag.cpod_notzero(i, j, nz);
        if (k < 0) continue;
        const Real src_coag0 = coag.cpod_short(i, j, nz) * Rc1;
        Kokkos::atomic_add(&source(k), src_coag0);
      }
    }
  });
  mbr.team_barrier();

  // FRAGMENTATION------------------------------------------------------------------------

  // Adding fragment distribution
  const int pgrid = coag.pgrid;
  const int iafrag = coag2drv::afrag;
  const int iphifrag = coag2drv::phifrag, iepsfrag = coag2drv::epsfrag;
  parthenon::par_for_inner(
      DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1, [&](const int k) {
        for (int j = k; j <= mimax; j++) {
          // Calculate As(j) on fly
          Real As_j = 0.0;
          for (int i2 = 0; i2 <= mimax; i2++) {
            for (int j2 = 0; j2 <= i2; j2++) {
              if (coag.klf(i2, j2) == j) {
                const Real fett_l =
                    CoagulationRate(i2, j2, kernel4, vel, stoppingTime, coag, 1);
                As_j += coag.coagR3D(iafrag, i2, j2) * distri(i2) * distri(j2) * fett_l;
              }
            }
          }
          source(k) += coag.coagR3D(iphifrag, k, j) / coag.mass_grid(k) * As_j;
        }
      });
  mbr.team_barrier();

  // Negative terms and cratering remnants
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum0 -= Rf1;
    }
    source(j) += sum0;
  });
  mbr.team_barrier();

  // Cratering
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0 = 0.0;
    for (int j = 0; j <= i - pgrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coag.coagR3D(iepsfrag, i, j) * Rf1;
      sum0 += dummy;
    }

    if (i - pgrid - 1 >= 0) {
      Kokkos::atomic_add(&source(i - 1), sum0);
    }

    Real sum1 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum1 -= Rf1;
    }
    Kokkos::atomic_add(&source(i), sum1);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::Coagulation_nQ
//  \brief
KOKKOS_INLINE_FUNCTION
void Coagulation_nQ(parthenon::team_mbr_t const &mbr, ScratchPad1D<Real> &nQs,
                    const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &distri,
                    const int &mimax, const KernelParams &kernel4,
                    const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stoppingTime,
                    const CoagParams &coag) {
  // Adding coagulation source terms
  const int iafrag = coag2drv::afrag;
  const int iphifrag = coag2drv::phifrag;
  const int iepsfrag = coag2drv::epsfrag;
  const int idalp = coag2drv::dalp;
  const int idpod = coag2drv::dpod;
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
      const Real fett_t = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 0);
      const Real Rc1 = distri(i) * distri(j) * fett_t;
      const Real &mass_gridi = coag.mass_grid(i);
      const Real &mass_gridj = coag.mass_grid(j);
      const Real &dalpji = coag.coagR3D(idalp, j, i);
      const Real &dpodji = coag.coagR3D(idpod, j, i);
      const Real &dalpij = coag.coagR3D(idalp, i, j);
      const Real &dpodij = coag.coagR3D(idpod, i, j);

      const Real Qp1 = Qplus(mass_gridi, Q(i), mass_gridj, Q(j));
      const Real dphijQ = dalpji * Qp1 * (1.0 + dpodji) - Q(j);
      const Real dphjiQ = dalpij * Qp1 * (1.0 + dpodij) - Q(i);

      const Real tmp1 = dphijQ - Qp1 * dpodji;
      const Real tmp2 = dphjiQ - Qp1 * dpodij;
      for (int nz = 0; nz < 4; nz++) {
        const int k = coag.cpod_notzero(i, j, nz);
        if (k < 0) continue;
        const Real kdeltajk = (j == k) ? 1.0 : 0.0;
        const Real kdeltaik = (i == k) ? 1.0 : 0.0;
        const Real nqs_k =
            (Qp1 * coag.cpod_short(i, j, nz) + tmp1 * kdeltajk + tmp2 * kdeltaik) * Rc1;
        Kokkos::atomic_add(&nQs(k), nqs_k);
      }
    }
  });
  mbr.team_barrier();

  // FRAGMENTATION------------------------------------------------------------------------

  // Adding fragment distribution
  const int pgrid = coag.pgrid;
  parthenon::par_for_inner(
      DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1, [&](const int k) {
        for (int j = k; j <= mimax; j++) {
          // calculate As(j) on fly
          Real As_j = 0.0;
          for (int i2 = 0; i2 <= mimax; i2++) {
            for (int j2 = 0; j2 <= i2; j2++) {
              int klf = (j2 <= i2 - pgrid - 1) ? j2 : i2;
              if (klf == j) {
                const Real fett_l =
                    CoagulationRate(i2, j2, kernel4, vel, stoppingTime, coag, 1);
                Real Qf1;
                const Real &mass_gridi2 = coag.mass_grid(i2);
                const Real &mass_gridj2 = coag.mass_grid(j2);
                const Real &afragi2j2 = coag.coagR3D(iafrag, i2, j2);
                if (j2 <= i2 - pgrid - 1) {
                  Qf1 = Qplus(coag.chi * mass_gridj2, Q(i2), mass_gridj2, Q(j2));
                } else {
                  Qf1 = Qplus(mass_gridi2, Q(i2), mass_gridj2, Q(j2));
                }
                As_j += afragi2j2 * distri(i2) * distri(j2) * fett_l * Qf1;
              }
            }
          }
          const Real &mass_gridk = coag.mass_grid(k);
          nQs(k) += coag.coagR3D(iphifrag, k, j) / mass_gridk * As_j;
        }
      });
  mbr.team_barrier();

  // Negative terms and cratering remnants
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    // Cratering
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum0 -= Rf1;
    }
    nQs(j) += (sum0 * Q(j));
  });
  mbr.team_barrier();

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0 = 0.0;
    for (int j = 0; j <= i - pgrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coag.coagR3D(iepsfrag, i, j) * Rf1 * Q(i);
      sum0 += dummy;
    }

    if (i - pgrid - 1 >= 0) {
      Kokkos::atomic_add(&nQs(i - 1), sum0);
    }

    Real sum1 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l * Q(i);
      sum1 -= Rf1;
    }
    Kokkos::atomic_add(&nQs(i), sum1);
  });
  mbr.team_barrier();
} // end of subroutine source

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CoagulationOneCell
//  \brief
KOKKOS_INLINE_FUNCTION
void CoagulationOneCell(parthenon::team_mbr_t const &mbr, const Real &time, Real &dt_sync,
                        const Real &gdens, ScratchPad1D<Real> &dustdens,
                        ScratchPad1D<Real> &stime, ScratchPad1D<Real> &vel,
                        const int nvel, ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                        const Real &alpha, const Real &cs, const Real &omega,
                        const CoagParams &coag, ScratchPad1D<Real> &source, int &nCall,
                        ScratchPad1D<Real> &Q2) {
  static constexpr Real mom_scale = 1.0e10;
  static constexpr Real mom_iscale = 1.0e-10;

  parthenon::par_for_inner(
      DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1, [&](const int i) {
        // convert to number density
        const Real &mass_gridi = coag.mass_grid(i);
        dustdens(i) /= mass_gridi; // number density
        // take the floor value
        dustdens(i) = std::max(dustdens(i), 0.01 * coag.dfloor / mass_gridi);
      });
  mbr.team_barrier();

  // do time steps
  nCall = 0;
  Real time_dummy = time;
  Real dt_sync1 = dt_sync;
  Real time_goal = time_dummy + dt_sync1;
  Real dt = dt_sync1;
  Real hnext = dt;
  dt_sync = 1e-15; // works H5

  const KernelParams kernel{gdens, alpha, cs, omega};
  while (std::abs(time_dummy - time_goal) > 1e-6 * dt) {
    int mimax = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(mbr, coag.nm),
        [&](const int i, int &lmax) {
          if (dustdens(i) > coag.dfloor / coag.mass_grid(i)) {
            lmax = std::max(lmax, i);
          }
        },
        Kokkos::Max<int>(mimax));

    CoagulationSource(mbr, source, dustdens, mimax, kernel, vel, stime, coag);

    // time step control
    dt_sync1 = std::numeric_limits<Real>::max(); // start with a large number
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(mbr, coag.nm),
        [&](const int i, Real &lmin) {
          if (dustdens(i) > coag.dfloor / coag.mass_grid(i) && source(i) < 0.0) {
            lmin = std::min(lmin, std::abs(dustdens(i) / source(i)));
          }
        },
        Kokkos::Min<Real>(dt_sync1));
    dt_sync1 *= coag.cfl;
    dt = std::min(dt_sync1, time_goal - time_dummy);
    dt_sync = dt_sync1;

    for (int n = 0; n < coag.mom_coag * nvel; n++) {
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                               [&](const int i) {
                                 Q(i) = vel(n + i * 3) * mom_scale;
                                 nQs(i) = 0.0; // initialize source(*)
                               });
      mbr.team_barrier();

      // Coagulation_nQ(mbr, nQs, Q, dustdens, mimax, kernel, vel, stime, coag);

      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                               [&](const int i) {
                                 const Real distri_i = dustdens(i) + dt * source(i);
                                 if (distri_i > coag.dfloor / coag.mass_grid(i)) {
                                   const Real nQ1 = dustdens(i) * Q(i) + dt * nQs(i);
                                   vel(n + i * 3) = nQ1 / distri_i * mom_iscale;
                                 }
                               });
      mbr.team_barrier();
    }

    // Update the density
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                             [&](const int i) { dustdens(i) += dt * source(i); });
    mbr.team_barrier();

    // Update time and increment ncall
    time_dummy += dt;
    nCall++;
    dt_sync = (coag.use_adaptive) ? std::max(hnext, dt_sync) : dt_sync;
    hnext = (coag.use_adaptive) ? std::min(hnext, time_goal - time_dummy) : hnext;

    // Warn and break upon reaching ncall_max
    if (nCall > coag.ncall_max) {
      PARTHENON_WARN("(Coagulation): Reach ncall_max in coagulation kernel!");
      break;
    }
  } // end of internal timestep

  // from number density to volume density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                           [&](const int i) { dustdens(i) *= coag.mass_grid(i); });
  mbr.team_barrier();

} // end of CoagulationOneCell

} // namespace Coagulation
} // namespace Dust

#endif // DUST_COAGULATION_COAGULATION_HPP_
