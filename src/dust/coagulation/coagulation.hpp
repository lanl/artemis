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
#ifndef DUST_COAGULATION_HPP_
#define DUST_COAGULATION_HPP_

#include "utils/artemis_utils.hpp"
#include "utils/units.hpp"

//#define  COAGULATION_DEBUG
namespace Dust {
namespace Coagulation {

enum coag2DRv { dpod, aFrag, phiFrag, epsFrag, dalp, kdelta, coef_fett, last2 };

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Params &dustPars,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants);

struct CoagParams {
  int coord = 0; // 1--surface density, 0: 3D

  int nm;
  int nCall_mx = 1000;
  bool ibounce = false;
  Real nlim;
  Real rho_p;
  Real vfrag;
  Real dfloor;
  int integrator;
  bool use_adaptive; // adaptive step size
  bool mom_coag;     // mom-preserving coagulation

  int pGrid;
  Real chi;
  Real pgrow;   // Power for increasing step size
  Real pshrink; // Power for decreasing step size
  Real err_eps; // Relative tolerance for adaptive step sizing
  Real S;       // Safety margin for adaptive step sizing
  Real cfl;
  Real errcon;      // Needed for increasing step size
  bool const_omega; // for shearing-box or testing

  Real rho0; // physical-to-code unit conversion

  // pre-calculated array, once-for-all
  ParArray2D<int> klf;
  ParArray1D<Real> mass_grid;
  ParArray3D<Real> coagR3D;
  ParArray3D<int> cpod_notzero;
  ParArray3D<Real> cpod_short;
};

void initializeArray(const int nm, int &pGrid, const Real &rho_p, const Real &chi,
                     const Real &a, const ParArray1D<Real> dsize, ParArray2D<int> klf,
                     ParArray1D<Real> mass_grid, ParArray3D<Real> coag3D,
                     ParArray3D<int> cpod_notzero, ParArray3D<Real> cpod_short);

KOKKOS_INLINE_FUNCTION
Real v_rel_ormel(Real tau_1, Real tau_2, Real t0, Real v0, Real ts, Real vs,
                 Real reynolds) {
  Real st1, st2, tau_mx, tau_mn, vg2;
  Real c0, c1, c2, c3, y_star, ya, eps;
  Real hulp1, hulp2;

  // sort tau's 1--> correspond to the max. now
  if (tau_1 >= tau_2) {
    tau_mx = tau_1;
    tau_mn = tau_2;
    st1 = tau_mx / t0;
    st2 = tau_mn / t0;
  } else {
    tau_mx = tau_2;
    tau_mn = tau_1;
    st1 = tau_mx / t0;
    st2 = tau_mn / t0;
  }

  vg2 = 1.5 * SQR(v0); // note the square
  ya = 1.6;            // approximate solution for st*=y*st1; valid for st1 << 1.

  Real sqRe = 1.0 / sqrt(reynolds);
  if (tau_mx < 0.2 * ts) {
    // very small regime
    return 1.5 * SQR((vs / ts * (tau_mx - tau_mn)));
  } else if (tau_mx < ts / ya) {
    return vg2 * (st1 - st2) / (st1 + st2) *
           (SQR(st1) / (st1 + sqRe) - SQR(st2) / (st2 + sqRe));
  } else if (tau_mx < 5.0 * ts) {
    // eq. 17 of oc07. the second term with st_i**2.0 is negligible (assuming re>>1)
    // hulp1 = eq. 17; hulp2 = eq. 18
    hulp1 =
        ((st1 - st2) / (st1 + st2) *
         (SQR(st1) / (st1 + ya * st1) - SQR(st2) / (st2 + ya * st1))); // note the -sign
    hulp2 = 2.0 * (ya * st1 - sqRe) + SQR(st1) / (ya * st1 + st1) -
            SQR(st1) / (st1 + sqRe) + SQR(st2) / (ya * st1 + st2) -
            SQR(st2) / (st2 + sqRe);
    return vg2 * (hulp1 + hulp2);
  } else if (tau_mx < t0 / 5.0) {
    // full intermediate regime
    eps = st2 / st1; // stopping time ratio
    return vg2 * (st1 * (2.0 * ya - (1.0 + eps) +
                         2.0 / (1.0 + eps) *
                             (1.0 / (1.0 + ya) + (eps * eps * eps) / (ya + eps))));
  } else if (tau_mx < t0) {
    // now y* lies between 1.6 (st1 << 1) and 1.0 (st1>=1). the fit below fits ystar to
    // less than 1%
    c3 = -0.29847604;
    c2 = 0.32938936;
    c1 = -0.63119577;
    c0 = 1.6015125;
    y_star = c0 + c1 * st1 + c2 * SQR(st1) + c3 * (st1 * st1 * st1);
    // we can then employ the same formula as before
    eps = st2 / st1; // stopping time ratio
    return vg2 *
           (st1 * (2.0 * y_star - (1.0 + eps) +
                   2.0 / (1.0 + eps) *
                       (1.0 / (1.0 + y_star) + (eps * eps * eps) / (y_star + eps))));
  } else {
    // heavy particle limit
    return vg2 * (1.0 / (1.0 + st1) + 1.0 / (1.0 + st2));
  }
}

KOKKOS_INLINE_FUNCTION
Real theta(Real x) { return (x < 0 ? 0.0 : 1.0); }

// Function calculates the new Q value of a particle resulting of a
// collision of particles with masses m1, m2 and Q values Q1, Q2
KOKKOS_INLINE_FUNCTION
Real Qplus(Real m1, Real Q1, Real m2, Real Q2) { return (m1 * Q1 + m2 * Q2) / (m1 + m2); }

KOKKOS_INLINE_FUNCTION
Real CoagulationRate(const int i, const int j, const Real kernel4[],
                     const ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag,
                     const int itype) {

  const Real &mass_gridi = coag.mass_grid(i);
  const Real &mass_gridj = coag.mass_grid(j);
  const Real &mass_gride = coag.mass_grid(coag.nm - 1);
  if (mass_gridi + mass_gridj >= mass_gride) {
    return 0.0;
  }

  const Real &gasdens = kernel4[0];
  const Real &alpha = kernel4[1];
  const Real &cs = kernel4[2];
  const Real &omega = kernel4[3];
  const Real &tau_i = stoppingTime(i);
  const Real &tau_j = stoppingTime(j);
  const Real *vel_i = &vel(3 * i);
  const Real *vel_j = &vel(3 * j);

  const Real sig_h2 = 2.0e-15;    //! cross section of H2
  const Real mu = 2.3;            //! mean molecular mass in proton masses
  const Real m_p = 1.6726231e-24; //! proton mass in g

  // calculate some basic properties
  const Real hg = cs / omega;
  Real re = alpha * sig_h2 * gasdens / (2.0 * mu * m_p);
  if (coag.coord == 0) {
    re *= std::sqrt(2.0 * M_PI) * hg;
  }

  const Real tn = 1.0 / omega;
  const Real ts = tn / sqrt(re);
  const Real vn = std::sqrt(alpha) * cs;
  const Real vs = vn * std::pow(re, -0.25);

  // calculate the relative velocities
  const Real c1 = 8.0 / M_PI * cs * cs * mu * m_p;

  const Real stokes_i = tau_i * omega;
  const Real stokes_j = tau_j * omega;

  // turbulent relative velocity
  Real dv0 = v_rel_ormel(tau_i, tau_j, tn, vn, ts, vs, re);

  // Brownian motion relative velocities
  const Real mredu = (mass_gridi * mass_gridj / (mass_gridi + mass_gridj));
  dv0 += (c1 / mredu);

  Real dv2_ij;
  Real hij = 1.0;
  if (coag.coord == 1) { // surface density
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
  } else {
    dv2_ij =
        (SQR(vel_i[0] - vel_j[0]) + SQR(vel_i[1] - vel_j[1]) + SQR(vel_i[2] - vel_j[2]));
  }

  // after adding up all v_rel**2 take the square root
  const Real dv = std::sqrt(dv0 + dv2_ij);

  // new pf calculation: for fragmenation
  Real pf = 0.0;
  if (dv > 0.0) {
    const Real tmp = 1.5 * SQR(coag.vfrag / dv);
    pf = (tmp + 1.0) * std::exp(-tmp);
  }

  int icoef_fett = coag2DRv::coef_fett;
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

KOKKOS_INLINE_FUNCTION
void CoagulationSource(parthenon::team_mbr_t const &mbr, ScratchPad1D<Real> &source,
                       const ScratchPad1D<Real> &distri, const int mimax,
                       const Real kernel4[], const ScratchPad1D<Real> &vel,
                       const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag) {

  const int nm = coag.nm;

  // initialize source(*)
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int k) {
    // for (int k = 0; k < nm; k++) {
    source(k) = 0.0;
  });
  mbr.team_barrier();

  // Adding coagulation source terms
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    // for (int i = 0; i <= mimax; i++) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
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

  // FRAGMENTATION------------------------------------------------------

  // Adding fragment distribution
  const int pGrid = coag.pGrid;
  int iaFrag = coag2DRv::aFrag;
  int iphiFrag = coag2DRv::phiFrag, iepsFrag = coag2DRv::epsFrag;
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int k) {
    // for (int k = 0; k < nm; k++) {
    for (int j = k; j <= mimax; j++) {
      // calculate As(j) on fly
      Real As_j = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          // int klf = (j2 <= i2 - pGrid - 1) ? j2 : i2;
          // if (klf == j) {
          if (coag.klf(i2, j2) == j) {
            const Real fett_l =
                CoagulationRate(i2, j2, kernel4, vel, stoppingTime, coag, 1);
            As_j += coag.coagR3D(iaFrag, i2, j2) * distri(i2) * distri(j2) * fett_l;
          }
        }
      }
      source(k) += coag.coagR3D(iphiFrag, k, j) / coag.mass_grid(k) * As_j;
    }
  });
  mbr.team_barrier();

  // Negative terms and cratering remnants
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    Real sum0(0.0);
    for (int i = j; i <= mimax; i++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum0 -= Rf1;
    }
    source(j) += sum0;
  });
  mbr.team_barrier();

  // for (int i = 0; i <= mimax; i++) {
  // Cratering
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0(0.0);
    for (int j = 0; j <= i - pGrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coag.coagR3D(iepsFrag, i, j) * Rf1;
      sum0 += dummy;
    }

    if (i - pGrid - 1 >= 0) {
      Kokkos::atomic_add(&source(i - 1), sum0);
    }

    Real sum1 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pGrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum1 -= Rf1;
    }
    Kokkos::atomic_add(&source(i), sum1);
  });
  mbr.team_barrier();
} // end of subroutine source

KOKKOS_INLINE_FUNCTION
void Coagulation_nQ(parthenon::team_mbr_t const &mbr, ScratchPad1D<Real> &nQs,
                    const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &distri,
                    const int mimax, const Real kernel4[], const ScratchPad1D<Real> &vel,
                    const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag) {

  const int nm = coag.nm;

  int iaFrag = coag2DRv::aFrag;
  int iphiFrag = coag2DRv::phiFrag, iepsFrag = coag2DRv::epsFrag;
  int idalp = coag2DRv::dalp, idpod = coag2DRv::dpod;
  // int ikdelta = coag2DRv::kdelta;
  // Adding coagulation source terms
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    // for (int i = 0; i <= mimax; i++) {
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

  // FRAGMENTATION------------------------------------------------------

  const int pGrid = coag.pGrid;
  // Adding fragment distribution
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int k) {
    // for (int k = 0; k < nm; k++) {
    for (int j = k; j <= mimax; j++) {
      // calculate As(j) on fly
      Real As_j = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          int klf = (j2 <= i2 - pGrid - 1) ? j2 : i2;
          // if (coag.klf(i2, j2) == j) {
          if (klf == j) {
            const Real fett_l =
                CoagulationRate(i2, j2, kernel4, vel, stoppingTime, coag, 1);
            Real Qf1;
            const Real &mass_gridi2 = coag.mass_grid(i2);
            const Real &mass_gridj2 = coag.mass_grid(j2);
            const Real &aFragi2j2 = coag.coagR3D(iaFrag, i2, j2);
            if (j2 <= i2 - pGrid - 1) {
              Qf1 = Qplus(coag.chi * mass_gridj2, Q(i2), mass_gridj2, Q(j2));
            } else {
              Qf1 = Qplus(mass_gridi2, Q(i2), mass_gridj2, Q(j2));
            }
            As_j += aFragi2j2 * distri(i2) * distri(j2) * fett_l * Qf1;
          }
        }
      }
      const Real &mass_gridk = coag.mass_grid(k);
      nQs(k) += coag.coagR3D(iphiFrag, k, j) / mass_gridk * As_j;
    }
  });
  mbr.team_barrier();

  // Negative terms and cratering remnants
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    // Cratering
    Real sum0(0.0);
    for (int i = j; i <= mimax; i++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      sum0 -= Rf1;
    }
    nQs(j) += (sum0 * Q(j));
  });
  mbr.team_barrier();

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0(0.0);
    for (int j = 0; j <= i - pGrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coag.coagR3D(iepsFrag, i, j) * Rf1 * Q(i);
      sum0 += dummy;
    }

    if (i - pGrid - 1 >= 0) {
      Kokkos::atomic_add(&nQs(i - 1), sum0);
    }

    Real sum1 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pGrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l * Q(i);
      sum1 -= Rf1;
    }
    Kokkos::atomic_add(&nQs(i), sum1);
  });
  mbr.team_barrier();
} // end of subroutine source

KOKKOS_INLINE_FUNCTION
void Coagulation_nQs(parthenon::team_mbr_t const &mbr, const Real &dt,
                     ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                     ScratchPad1D<Real> &distri, const int mimax, const int nvel,
                     const Real kernel4[], ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag,
                     ScratchPad1D<Real> &source) {

  const int nm = coag.nm - 1;

  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;

  for (int n = 0; n < nvel; n++) {
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm, [&](const int k) {
      Q(k) = vel(n + k * 3) * mom_scale;
      nQs(k) = 0.0; // initialize source(*)
    });
    mbr.team_barrier();

    Coagulation_nQ(mbr, nQs, Q, distri, mimax, kernel4, vel, stoppingTime, coag);

    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm, [&](const int k) {
      const Real distri_k = distri(k) + dt * source(k);
      if (distri_k > coag.dfloor / coag.mass_grid(k)) {
        const Real nQ1 = distri(k) * Q(k) + dt * nQs(k);
        vel(n + k * 3) = nQ1 / distri_k * mom_iscale;
      }
    });
    mbr.team_barrier();
  } // loop of n

  // update the density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm,
                           [&](const int k) { distri(k) += dt * source(k); });
  mbr.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void Coagulation_nQs3(parthenon::team_mbr_t const &mbr, const Real &dt,
                      ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                      ScratchPad1D<Real> &distri, const int mimax, const int nvel,
                      const Real kernel4[], ScratchPad1D<Real> &vel,
                      const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag,
                      ScratchPad1D<Real> &source, ScratchPad1D<Real> &Q2,
                      const int mimax2) {

  const int nm = coag.nm - 1;

  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm, [&](const int k) {
    Q2(k) = nQs(k); // 2nd stage source
  });
  mbr.team_barrier();

  for (int n = 0; n < nvel; n++) {
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm, [&](const int k) {
      Q(k) = vel(n + k * 3) * mom_scale;
      nQs(k) = 0.0;
    });
    mbr.team_barrier();

    // 1st stage
    Coagulation_nQ(mbr, nQs, Q, distri, mimax, kernel4, vel, stoppingTime, coag);

    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm, [&](const int k) {
      const Real distri_k = distri(k) + dt * source(k);
      const Real nQ1 = distri(k) * Q(k) + dt * nQs(k);
      Q(k) = nQ1 / distri_k; // intermediate Q
    });
    mbr.team_barrier();

    // 2nd stage
    Coagulation_nQ(mbr, nQs, Q, distri, mimax2, kernel4, vel, stoppingTime, coag);

    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm, [&](const int k) {
      const Real nQ_o = distri(k) * vel(n + k * 3) * mom_scale;
      const Real distri_k = distri(k) + 0.5 * dt * (source(k) + Q2(k));
      if (distri_k > coag.dfloor / coag.mass_grid(k)) {
        Real nQ1 = nQ_o + 0.5 * dt * nQs(k);
        vel(n + k * 3) = nQ1 / distri_k * mom_iscale;
      }
    });
    mbr.team_barrier();
  }
  // update the density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm, [&](const int k) {
    distri(k) += 0.5 * dt * (source(k) + Q2(k));
  });
  mbr.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void CoagulationOneCell(parthenon::team_mbr_t const &mbr, const int cell_i,
                        const Real &time, Real &dt_sync, const Real &gasdens,
                        ScratchPad1D<Real> &dustdens, ScratchPad1D<Real> &stime,
                        ScratchPad1D<Real> &vel, const int nvel, ScratchPad1D<Real> &Q,
                        ScratchPad1D<Real> &nQs, const Real &alpha, const Real &cs,
                        const Real &omega, const CoagParams &coag,
                        ScratchPad1D<Real> &source, int &nCall, ScratchPad1D<Real> &Q2) {

  int nm = coag.nm;

  Real dt, time_dummy, time_goal, hnext;
  Real dt_sync1 = dt_sync;

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int i) {
    // convert to number density
    const Real &mass_gridi = coag.mass_grid(i);
    dustdens(i) /= mass_gridi; // number density
    // take the floor value
    dustdens(i) = std::max(dustdens(i), 0.01 * coag.dfloor / mass_gridi);
#ifdef COAGULATION_DEBUG
    if (cell_i == 2 && dustdens[i] > coag.dfloor / mass_gridi) {
      std::cout << " oneCellBeg: " << time << " " << i << " " << dustdens[i] << " "
                << mass_gridi << " " << stime[i] << std::endl;
    }
#endif
  });
  mbr.team_barrier();

  nCall = 0;

  // do time steps
  time_dummy = time;
  time_goal = time_dummy + dt_sync1;
  dt = dt_sync1;
  hnext = dt;
  dt_sync = 1e-15; // works H5

  Real kernel[4];
  kernel[0] = gasdens;
  kernel[1] = alpha;
  kernel[2] = cs;
  kernel[3] = omega;

  while (std::abs(time_dummy - time_goal) > 1e-6 * dt) {
    int mimax = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(mbr, nm),
        [&](const int i, int &lmax) {
          if (dustdens(i) > coag.dfloor / coag.mass_grid(i)) {
            lmax = std::max(lmax, i);
          }
        },
        Kokkos::Max<int>(mimax));

    CoagulationSource(mbr, source, dustdens, mimax, kernel, vel, stime, coag);

    if (coag.use_adaptive == 0 || coag.integrator == 1) {
      // time step control
      dt_sync1 = std::numeric_limits<Real>::max(); // start with a large number
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(mbr, nm),
          [&](const int i, Real &lmin) {
            if (dustdens(i) > coag.dfloor / coag.mass_grid(i) && source(i) < 0.0) {
              lmin = std::min(lmin, std::abs(dustdens(i) / source(i)));
            }
          },
          Kokkos::Min<Real>(dt_sync1));
      dt_sync1 *= coag.cfl;
      dt = std::min(dt_sync1, time_goal - time_dummy);
      dt_sync = dt_sync1;

      if (coag.mom_coag) {
        Coagulation_nQs(mbr, dt, Q, nQs, dustdens, mimax, nvel, kernel, vel, stime, coag,
                        source);
      } else {
        // integration: first-order
        parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1,
                                 [&](const int i) { dustdens(i) += dt * source(i); });
        mbr.team_barrier();
      }

    } else {
      // third-order method
      Real h0 = hnext;
      Real h = h0;
      Real errmax;
      int mimax2 = 0;
      // Heun's Method
      while (1) {
        // Q(*) is temprary variable to store the dust number density
        mimax2 = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(mbr, nm),
            [&](const int i, int &lmax) {
              Q(i) = dustdens(i) + h * source(i);
              if (Q(i) > coag.dfloor / coag.mass_grid(i)) {
                lmax = std::max(lmax, i);
              }
            },
            Kokkos::Max<int>(mimax2));

        CoagulationSource(mbr, nQs, Q, mimax2, kernel, vel, stime, coag);

        errmax = 0.0;
        const int nm1 = std::min(mimax2, mimax);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(mbr, nm1),
            [&](const int i, Real &lmax) {
              Real dscale = std::abs(dustdens(i)) + std::abs(h0 * source(i));
              Real derr = 0.5 * h * (nQs(i) - source(i)) / dscale;
              lmax = std::max(lmax, std::abs(derr));
            },
            Kokkos::Max<Real>(errmax));
#ifdef COAGULATION_DEBUG
        if (cell_i == 2)
          std::cout << nCall << " " << errmax << " " << coag.err_eps << " " << h << " "
                    << mimax << " " << mimax2 << std::endl;
#endif
        errmax /= coag.err_eps;

        if (errmax <= 1.0) break;

        h = std::max(coag.S * h * std::pow(errmax, coag.pshrink), 0.1 * h);
      } // end of while (1)

      if (errmax > coag.errcon) {
        hnext = coag.S * h * std::pow(errmax, coag.pgrow);
      } else {
        hnext = 5.0 * h;
      }

      // Actual taken step
      dt = h;

      if (coag.mom_coag) {
        Coagulation_nQs3(mbr, dt, Q, nQs, dustdens, mimax, nvel, kernel, vel, stime, coag,
                         source, Q2, mimax2);
      } else {
        parthenon::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1,
            [&](const int i) { dustdens(i) += 0.5 * dt * (source(i) + nQs(i)); });
        mbr.team_barrier();
      }

    } // end of if (coag.use_adaptive == 0)

    // Kokkos::single (Kokkos::PerTeam(mbr), [&]() {
    time_dummy += dt;
    nCall++;
    //}); mbr.team_barrier();

    if (coag.use_adaptive == 1) {
      dt_sync = std::max(hnext, dt_sync);
      hnext = std::min(hnext, time_goal - time_dummy);
    }

    if (nCall > coag.nCall_mx) break;
  } // end of internal timestep

  // from number density to volume density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1,
                           [&](const int i) { dustdens(i) *= coag.mass_grid(i); });
  mbr.team_barrier();

} // end of CoagulationOneCell

} // namespace Coagulation
} // namespace Dust

#endif // DUST_COAGULATION_HPP_
