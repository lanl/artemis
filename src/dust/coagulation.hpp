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

//#define  COAGULATION_DEBUG
namespace Dust {

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

  vg2 = 1.5 * pow(v0, 2); // note the square
  ya = 1.6;               // approximate solution for st*=y*st1; valid for st1 << 1.

  Real sqRe = 1.0 / sqrt(reynolds);
  if (tau_mx < 0.2 * ts) {
    // very small regime
    return 1.5 * pow((vs / ts * (tau_mx - tau_mn)), 2);
  } else if (tau_mx < ts / ya) {
    return vg2 * (st1 - st2) / (st1 + st2) *
           (pow(st1, 2) / (st1 + sqRe) - pow(st2, 2) / (st2 + sqRe));
  } else if (tau_mx < 5.0 * ts) {
    // eq. 17 of oc07. the second term with st_i**2.0 is negligible (assuming re>>1)
    // hulp1 = eq. 17; hulp2 = eq. 18
    hulp1 = ((st1 - st2) / (st1 + st2) *
             (pow(st1, 2) / (st1 + ya * st1) -
              pow(st2, 2) / (st2 + ya * st1))); // note the -sign
    hulp2 = 2.0 * (ya * st1 - sqRe) + pow(st1, 2.0) / (ya * st1 + st1) -
            pow(st1, 2) / (st1 + sqRe) + pow(st2, 2) / (ya * st1 + st2) -
            pow(st2, 2) / (st2 + sqRe);
    return vg2 * (hulp1 + hulp2);
  } else if (tau_mx < t0 / 5.0) {
    // full intermediate regime
    eps = st2 / st1; // stopping time ratio
    return vg2 *
           (st1 * (2.0 * ya - (1.0 + eps) +
                   2.0 / (1.0 + eps) * (1.0 / (1.0 + ya) + pow(eps, 3) / (ya + eps))));
  } else if (tau_mx < t0) {
    // now y* lies between 1.6 (st1 << 1) and 1.0 (st1>=1). the fit below fits ystar to
    // less than 1%
    c3 = -0.29847604;
    c2 = 0.32938936;
    c1 = -0.63119577;
    c0 = 1.6015125;
    y_star = c0 + c1 * st1 + c2 * pow(st1, 2) + c3 * pow(st1, 3);
    // we can then employ the same formula as before
    eps = st2 / st1; // stopping time ratio
    return vg2 * (st1 * (2.0 * y_star - (1.0 + eps) +
                         2.0 / (1.0 + eps) *
                             (1.0 / (1.0 + y_star) + pow(eps, 3) / (y_star + eps))));
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

class Coagulation {
 public:
  Coagulation(int nspecies, ParArray1D<Real> dustSize, Real rho_p, Real vfrag, Real nlim,
              int integrator, bool use_adaptive, Real dfloor, bool imom_coag, int nsubmx,
              int coord)
      : nm(nspecies), rho_p(rho_p), vfrag(vfrag), nlim(nlim), integrator(integrator),
        use_adaptive(use_adaptive), dfloor(dfloor), mom_coag(imom_coag), nCall_mx(nsubmx),
        coord(coord) {
    dsize = ParArray1D<Real>("dsize", nm);
    Kokkos::deep_copy(dsize, dustSize);

    ij_idx = ParArray1D<int>("ij_index", nm);
    cpod_notzero = ParArray3D<int>("idx_nzcpod", nm, nm, 4);
    cpod_short = ParArray3D<Real>("nzcpod", nm, nm, 4);
    mass_grid = ParArray1D<Real>("mass grid", nm);
    dpod = ParArray2D<Real>("dpod", nm, nm);
    aFrag = ParArray2D<Real>("afrag", nm, nm);
    phiFrag = ParArray2D<Real>("phifrag", nm, nm);
    epsFrag = ParArray2D<Real>("epsfrag", nm, nm);

    dalp = ParArray2D<Real>("dalp", nm, nm);
    kdelta = ParArray2D<Real>("kdelta", nm, nm);

    coef_fett = ParArray2D<Real>("coef_fett", nm, nm);
    floor_dnum = ParArray1D<Real>("floor_dnum", nm);

    klf = ParArray2D<int>("klf", nm, nm);
    // set constant
    err_eps = 1.0e-1;
    S = 0.9;
    cfl = 1.0e-1;
    chi = 1.0;

    // some checks and definition
    if (use_adaptive) {
      if (integrator == 3) {
        pgrow = -0.5;
        pshrink = -1.0;
      } else if (integrator == 5) {
        pgrow = -0.2;
        pshrink = -0.25;
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in dust coagulation initialization: " << std::endl
            << "###   You can not use this integrator with adaptive step sizing: "
            << integrator << std::endl;
        PARTHENON_FAIL(msg);
      }
      errcon = std::pow(5. / S, (1. / pgrow));
    }
  }

  KOKKOS_INLINE_FUNCTION
  ~Coagulation() {}

  // private:
  int coord = 0; // 1--surface density, 0: 3D

  int nm;
  bool ibounce = false;
  int nCall_mx = 1000;
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
  Real errcon; // Needed for increasing step size

  ParArray1D<int> ij_idx;
  ParArray1D<Real> dsize;
  ParArray1D<Real> mass_grid;

  ParArray3D<int> cpod_notzero;
  ParArray3D<Real> cpod_short;
  ParArray2D<Real> dpod;
  ParArray2D<Real> aFrag;
  ParArray2D<Real> phiFrag;
  ParArray2D<Real> epsFrag;
  ParArray2D<Real> dalp, kdelta;
  ParArray2D<Real> coef_fett;
  ParArray1D<Real> floor_dnum;

  ParArray2D<int> klf;
};

// using Kokkos par_for() loop
void initializeArray(const int nm, int &pGrid, const Real &rho_p, const Real &chi,
                     const ParArray1D<Real> dsize, ParArray1D<int> ij_idx,
                     ParArray2D<Real> kdelta, ParArray1D<Real> mass_grid,
                     ParArray2D<Real> coef_fett, ParArray2D<Real> epsFrag,
                     ParArray2D<Real> phiFrag, ParArray2D<int> klf,
                     ParArray2D<Real> aFrag, ParArray2D<Real> dalp, ParArray2D<Real> dpod,
                     ParArray3D<int> cpod_notzero, ParArray3D<Real> cpod_short,
                     const Real &dfloor, ParArray1D<Real> &floor_dnum) {

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag1", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        // initialize in_idx(*) array
        ij_idx(i) = 0;
        for (int j = 1; j <= i; j++) {
          ij_idx(i) += j;
        }
        // initialize kdelta array
        for (int j = 0; j < nm; j++) {
          kdelta(i, j) = 0.0;
        }
        kdelta(i, i) = 1.0;
        mass_grid(i) = 4.0 * M_PI / 3.0 * rho_p * std::pow(dsize(i), 3);
        floor_dnum(i) = dfloor / mass_grid(i);
        // initialize coef_fett(*,*)
        for (int j = 0; j < nm; j++) {
          Real tmp1 = (1.0 - 0.5 * kdelta(i, j));
          coef_fett(i, j) = M_PI * SQR(dsize(i) + dsize(j)) * tmp1;
        }
      });

  // set fragmentation variables
  auto massGrid_h =
      Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), mass_grid);

  Real a = std::log10(massGrid_h(0) / massGrid_h(nm - 1)) / (1 - nm);
  Real ten_a = std::pow(10.0, a);
  Real ten_ma = 1.0 / ten_a;
  int ce = int(-1.0 / a * std::log10(1.0 - ten_ma)) + 1;

  pGrid = floor(1.0 / a); // used in integration

  Real frag_slope = 2.0 - 11.0 / 6.0;

  if (parthenon::Globals::my_rank == 0) {
    auto floorh =
        Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), floor_dnum);
    auto ij_idxh = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), ij_idx);
    std::cout << "coag initialization: pGrid,order=" << pGrid
              << " "
              //<< integrator<< " " << mom_coag
              << ", and mass grid is defined as:" << std::endl;
    for (int n = 0; n < nm; n++) {
      std::cout << n << " " << massGrid_h(n) << " " << floorh(n) << " " << ij_idxh(n)
                << std::endl;
    }
  }

  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag2", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        Real sum_pF = 0.0;
        for (int j = 0; j <= i; j++) {
          phiFrag(j, i) = std::pow(mass_grid(j), frag_slope);
          sum_pF += phiFrag(j, i);
        }
        // normalization
        for (int j = 0; j <= i; j++) {
          phiFrag(j, i) /= sum_pF; // switch (i,j) from fortran
        }

        // Cratering
        for (int j = 0; j <= i - pGrid - 1; j++) {
          // FRAGMENT DISTRIBUTION
          // The largest fragment has the mass of the smaller collision partner

          // Mass bin of largest fragment
          klf(i, j) = j;

          aFrag(i, j) = ((1.0 + chi) * mass_grid(j));
          //            |_____________|
          //                    |
          //                    Mass of fragments

          epsFrag(i, j) = chi * mass_grid(j) / (mass_grid(i) * (1.0 - ten_ma));
        }

        int i1 = std::max(0, i - pGrid);
        for (int j = i1; j <= i; j++) {
          // The largest fragment has the mass of the larger collison partner
          klf(i, j) = i;
          aFrag(i, j) = (mass_grid(i) + mass_grid(j));
        }
      });

  // initialize dalp array
  // Calculate the D matrix
  // Calculate the E matrix
  ParArray2D<Real> e("epod", nm, nm);
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag4", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int k) {
        for (int j = 0; j < nm; j++) {
          if (j <= k + 1 - ce) {
            dalp(k, j) = 1.0;
            dpod(k, j) = -mass_grid(j) / (mass_grid(k) * (ten_a - 1.0));
          } else {
            dpod(k, j) = -1.0;
            dalp(k, j) = 0.0;
          }
        }
        // for E matrix-------------
        Real mkkme = mass_grid(k) * (1.0 - ten_ma);
        Real mkpek = mass_grid(k) * (ten_a - 1.0);
        Real mkpeme = mass_grid(k) * (ten_a - ten_ma);
        for (int j = 0; j < nm; ++j) {
          if (j <= k - ce) {
            e(k, j) = mass_grid(j) / mkkme;
          } else {
            Real theta1 = (mkpeme - mass_grid(j) < 0.0) ? 0.0 : 1.0;
            e(k, j) = (1.0 - (mass_grid(j) - mkkme) / mkpek) * theta1;
          }
        }
      });

  // calculate the coagualtion variables
  ParArray3D<Real> cpod("cpod", nm, nm, nm);

  // initialize cpod array;
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "initializeCoag6", parthenon::DevExecSpace(),
      0, nm - 1, 0, nm - 1, KOKKOS_LAMBDA(const int i, const int j) {
        // initialize to zero first
        for (int k = 0; k < nm; k++) {
          cpod(i, j, k) = 0.0;
        }
        Real mloc = mass_grid(i) + mass_grid(j);
        if (mloc < mass_grid(nm - 1)) {
          int gg = 0;
          for (int k = std::max(i, j); k < nm - 1; k++) {
            if (mloc >= mass_grid(k) && mloc < mass_grid(k + 1)) {
              gg = k;
              break;
            }
          }

          cpod(i, j, gg) =
              (mass_grid(gg + 1) - mloc) / (mass_grid(gg + 1) - mass_grid(gg));
          cpod(i, j, gg + 1) = 1.0 - cpod(i, j, gg);

          // modified cpod(*) array-------------------------
          Real dtheta_ji = (j - i - 0.5 < 0.0) ? 0.0 : 1.0; // theta(j - i - 0.5);
          for (int k = 0; k < nm; k++) {
            Real theta_kj = (k - j - 1.5 < 0.0) ? 0.0 : 1.0; // theta(k - j - 1.5)
            cpod(i, j, k) = (0.5 * kdelta(i, j) * cpod(i, j, k) +
                             cpod(i, j, k) * theta_kj * dtheta_ji);
          }
          cpod(i, j, j) += dpod(j, i);
          cpod(i, j, j + 1) += e(j + 1, i) * dtheta_ji;

        } //  end if
      });

  // initialize cpod_nonzero and cpod_short array
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "initializeCoag7", parthenon::DevExecSpace(),
      0, nm - 1, 0, nm - 1, KOKKOS_LAMBDA(const int i, const int j) {
        if (j <= i) {
          // initialize cpod_notzero(i, j, 4) and cpod_short(i, j, 4)
          for (int k = 0; k < 4; k++) {
            cpod_notzero(i, j, k) = 0;
            cpod_short(i, j, k) = 0.0;
          }
          int inc = 0;
          for (int k = 0; k < nm; ++k) {
            Real dum = cpod(i, j, k) + cpod(j, i, k);
            if (dum != 0.0) {
              cpod_notzero(i, j, inc) = k;
              cpod_short(i, j, inc) = dum;
              inc++;
            }
          }
        }
      });
}

KOKKOS_INLINE_FUNCTION
Real CoagulationRate(const int i, const int j, const Real kernel4[], const Real vel[][3],
                     const Real stoppingTime[], Coagulation &coag, const int itype) {

  if (coag.mass_grid(i) + coag.mass_grid(j) >= coag.mass_grid(coag.nm - 1)) {
    return 0.0;
  }

  const Real &gasdens = kernel4[0];
  const Real &alpha = kernel4[1];
  const Real &cs = kernel4[2];
  const Real &omega = kernel4[3];
  const Real &tau_i = stoppingTime[i];
  const Real &tau_j = stoppingTime[j];
  const Real *vel_i = vel[i];
  const Real *vel_j = vel[j];

  const Real sig_h2 = 2.0e-15;    //! cross section of H2
  const Real mu = 2.3;            //! mean molecular mass in proton masses
  const Real m_p = 1.6726231e-24; //! proton mass in g

  // calculate some basic properties
  const Real hg = cs / omega;
  const Real re = alpha * sig_h2 * gasdens / (2.0 * mu * m_p);
  const Real tn = 1.0 / omega;
  const Real ts = tn / sqrt(re);
  const Real vn = std::sqrt(alpha) * cs;
  const Real vs = vn * std::pow(re, -0.25);

  // calculate the relative velocities
  const Real c1 = 8.0 / M_PI * cs * cs * mu * m_p;

  const Real stokes_i = tau_i * omega;
  const Real stokes_j = tau_j * omega;

  const Real hi =
      std::min(std::sqrt(alpha / (std::min(0.5, stokes_i) * (stokes_i + 1.0))), 1.0) * hg;
  const Real hj =
      std::min(std::sqrt(alpha / (std::min(0.5, stokes_j) * (stokes_j + 1.0))), 1.0) * hg;

  // turbulent relative velocity
  Real dv0 = v_rel_ormel(tau_i, tau_j, tn, vn, ts, vs, re);

  // Brownian motion relative velocities
  const Real mredu =
      (coag.mass_grid(i) * coag.mass_grid(j) / (coag.mass_grid(i) + coag.mass_grid(j)));
  dv0 += (c1 / mredu);

  Real dv2_ij;
  if (coag.coord == 1) { // surface density
    const Real vs_i = std::min(stokes_i, 1.0) * omega * hi;
    const Real vs_j = std::min(stokes_j, 1.0) * omega * hj;
    dv2_ij = std::pow(vs_i - vs_j, 2);
    dv2_ij += (SQR(vel_i[0] - vel_j[0]) + SQR(vel_i[1] - vel_j[1]));
  } else {
    dv2_ij =
        (SQR(vel_i[0] - vel_j[0]) + SQR(vel_i[1] - vel_j[1]) + SQR(vel_i[2] - vel_j[2]));
  }

  // after adding up all v_rel**2 take the square root
  const Real dv = std::sqrt(dv0 + dv2_ij);

  Real hij = 1.0;
  if (coag.coord == 1) {
    hij = std::sqrt(2.0 * M_PI * (SQR(hi) + SQR(hj)));
  }

  // new pf calculation: for fragmenation
  Real pf = 0.0;
  if (dv > 0.0) {
    const Real tmp = 1.5 * SQR(coag.vfrag / dv);
    pf = (tmp + 1.0) * std::exp(-tmp);
  }

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
    return (coag.coef_fett(i, j) * dv * pc / hij);
  } else { // fragmentation
    return (coag.coef_fett(i, j) * dv * pf / hij);
  }
}

KOKKOS_INLINE_FUNCTION
void CoagulationSource(Real *source, const Real *distri, const int mimax,
                       const Real kernel4[], const Real vel[][3],
                       const Real stoppingTime[], Coagulation &coag) {

  const int nm = coag.nm;

  // initialize source(*)
  for (int k = 0; k < nm; k++) {
    source[k] = 0.0;
  }

  // Adding coagulation source terms
  for (int i = 0; i <= mimax; i++) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
      const Real fett_t = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 0);
      const Real Rc1 = distri[i] * distri[j] * fett_t;
      for (int nz = 0; nz < 4; nz++) {
        const int k = coag.cpod_notzero(i, j, nz);
        if (k < 0) continue;
        const Real src_coag0 = coag.cpod_short(i, j, nz) * Rc1;
        source[k] += src_coag0;
      }
    }
  }

  // FRAGMENTATION------------------------------------------------------

  // Adding fragment distribution
  for (int k = 0; k < nm; k++) {
    for (int j = k; j <= mimax; j++) {
      // calculate As(j) on fly
      Real As_j = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          if (coag.klf(i2, j2) == j) {
            const Real fett_l =
                CoagulationRate(i2, j2, kernel4, vel, stoppingTime, coag, 1);
            As_j += coag.aFrag(i2, j2) * distri[i2] * distri[j2] * fett_l;
          }
        }
      }
      source[k] += coag.phiFrag(k, j) / coag.mass_grid(k) * As_j;
    }
  }

  const int pGrid = coag.pGrid;
  // Negative terms and cratering remnants
  for (int i = 0; i <= mimax; i++) {
    // Cratering
    for (int j = 0; j <= i - pGrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri[i] * distri[j] * fett_l;
      const Real dummy = coag.epsFrag(i, j) * Rf1;
      source[i - 1] += dummy;
      source[i] -= dummy;
      source[j] -= Rf1;
    }

    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pGrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri[i] * distri[j] * fett_l;
      source[i] -= Rf1;
      source[j] -= Rf1;
    }
  }
} // end of subroutine source

KOKKOS_INLINE_FUNCTION
void Coagulation_nQ(Real *nQs, const Real *Q, const Real *distri, const int mimax,
                    const Real kernel4[], const Real vel[][3], const Real stoppingTime[],
                    Coagulation &coag) {

  const int nm = coag.nm;

  // initialize source(*)
  for (int k = 0; k < nm; k++) {
    nQs[k] = 0.0;
  }

  // Adding coagulation source terms
  for (int i = 0; i <= mimax; i++) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
      const Real fett_t = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 0);
      const Real Rc1 = distri[i] * distri[j] * fett_t;
      const Real Qp1 = Qplus(coag.mass_grid(i), Q[i], coag.mass_grid(j), Q[j]);
      const Real dphijQ = coag.dalp(j, i) * Qp1 * (1.0 + coag.dpod(j, i)) - Q[j];
      const Real dphjiQ = coag.dalp(i, j) * Qp1 * (1.0 + coag.dpod(i, j)) - Q[i];

      const Real tmp1 = dphijQ - Qp1 * coag.dpod(j, i);
      const Real tmp2 = dphjiQ - Qp1 * coag.dpod(i, j);
      for (int nz = 0; nz < 4; nz++) {
        const int k = coag.cpod_notzero(i, j, nz);
        if (k < 0) continue;
        const Real nqs_k = (Qp1 * coag.cpod_short(i, j, nz) + tmp1 * coag.kdelta(j, k) +
                            tmp2 * coag.kdelta(i, k)) *
                           Rc1;
        nQs[k] += nqs_k;
      }
    }
  }

  // FRAGMENTATION------------------------------------------------------

  // Adding fragment distribution
  const int pGrid = coag.pGrid;
  for (int k = 0; k < nm; k++) {
    for (int j = k; j <= mimax; j++) {
      // calculate As(j) on fly
      Real As_j = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          if (coag.klf(i2, j2) == j) {
            const Real fett_l =
                CoagulationRate(i2, j2, kernel4, vel, stoppingTime, coag, 1);
            Real Qf1;
            if (j2 <= i2 - pGrid - 1) {
              Qf1 =
                  Qplus(coag.chi * coag.mass_grid(j2), Q[i2], coag.mass_grid(j2), Q[j2]);
            } else {
              Qf1 = Qplus(coag.mass_grid(i2), Q[i2], coag.mass_grid(j2), Q[j2]);
            }
            As_j += coag.aFrag(i2, j2) * distri[i2] * distri[j2] * fett_l * Qf1;
          }
        }
      }
      nQs[k] += coag.phiFrag(k, j) / coag.mass_grid(k) * As_j;
    }
  }

  // Negative terms and cratering remnants
  for (int i = 0; i <= mimax; i++) {
    // Cratering
    for (int j = 0; j <= i - pGrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri[i] * distri[j] * fett_l;
      const Real dummy = coag.epsFrag(i, j) * Rf1 * Q[i];
      nQs[i - 1] += dummy;
      nQs[i] -= dummy;
      nQs[j] -= Rf1 * Q[j];
    }

    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pGrid);
    for (int j = i1; j <= i; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri[i] * distri[j] * fett_l;
      nQs[i] -= Rf1 * Q[i];
      nQs[j] -= Rf1 * Q[j];
    }
  }
} // end of subroutine source

KOKKOS_INLINE_FUNCTION
void Coagulation_nQs(const Real &dt, Real *Q, Real *nQs, const Real *distri,
                     const int mimax, const int nvel, const Real kernel4[], Real vel[][3],
                     const Real stoppingTime[], Coagulation &coag) {

  const int nm = coag.nm;

  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;
  for (int n = 0; n < nvel; n++) {
    for (int k = 0; k < nm; k++) {
      Q[k] = vel[k][n] * mom_scale;
    }

    Coagulation_nQ(nQs, Q, distri, mimax, kernel4, vel, stoppingTime, coag);

    for (int k = 0; k < nm; k++) {
      if (distri[k] > coag.floor_dnum(k)) {
        Real nQ1 = distri[k] * Q[k] + dt * nQs[k];
        vel[k][n] = nQ1 / distri[k] * mom_iscale;
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CoagulationOneCell(const int cell_i, const Real &time, Real &dt_sync,
                        const Real &gasdens, Real *dustdens, Real *stime, Real vel[][3],
                        const int nvel, Real *Q, Real *nQs, const Real &alpha,
                        const Real &cs, const Real &omega, const Real &volume,
                        Coagulation coag, Real *source, int &nCall) {

  int nm = coag.nm;
  // convert to number density
  for (int i = 0; i < nm; ++i) {
    dustdens[i] /= coag.mass_grid(i); // number density
  }

  const Real floorVal = coag.nlim / volume;
  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;

  Real dt, time_dummy, time_goal, hnext;
  Real dt_sync1 = dt_sync;

  for (int i = 0; i < nm; ++i) {
    dustdens[i] = std::max(dustdens[i], 0.01 * std::min(coag.floor_dnum(i), floorVal));
#ifdef COAGULATION_DEBUG
    if (cell_i == 2 && dustdens[i] > floorVal) {
      std::cout << " oneCellBeg: " << time << " " << i << " "
                << dustdens[i] * coag.mass_grid(i) << " " << coag.floor_dnum(i) << " "
                << stime[i] << std::endl;
    }
#endif
  }

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
    for (int i = nm - 1; i >= 0; i--) {
      if (dustdens[i] > coag.floor_dnum(i)) {
        mimax = i;
        break;
      }
    }

    CoagulationSource(source, dustdens, mimax, kernel, vel, stime, coag);

    // time step control
    dt_sync1 = std::numeric_limits<Real>::max(); // start with a large number
    for (int i = 0; i < nm; i++) {
      if (dustdens[i] > coag.floor_dnum(i) && source[i] < 0.0) {
        dt_sync1 = std::min(dt_sync1, std::abs(dustdens[i] / source[i]));
      }
    }
    dt = std::min(dt_sync1, time_goal - time_dummy);
    dt_sync = dt_sync1;

    if (coag.mom_coag) {
      Coagulation_nQs(dt, Q, nQs, dustdens, mimax, nvel, kernel, vel, stime, coag);
    }

    // integration: first-order
    for (int i = 0; i < nm; i++) {
      dustdens[i] += dt * source[i];
    }

    time_dummy += dt;
    nCall++;

    if (nCall > coag.nCall_mx) break;
  } // end of internal timestep

  // back to density and velocity
  for (int i = 0; i < nm; i++) {
    if (dustdens[i] > coag.floor_dnum(i)) {
      dustdens[i] *= coag.mass_grid(i);
    } else {
      dustdens[i] = 0.0;
    }
  }
} // end of CoagulationOneCell

// inner-loop using scratch space----------------------------------------------------
//  ---------------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
Real CoagulationRate(const int i, const int j, const Real kernel4[],
                     const ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const Coagulation &coag,
                     const int itype) {

  auto &mass_grid = coag.mass_grid;
  if (mass_grid(i) + mass_grid(j) >= mass_grid(coag.nm - 1)) {
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
  const Real re = alpha * sig_h2 * gasdens / (2.0 * mu * m_p);
  const Real tn = 1.0 / omega;
  const Real ts = tn / sqrt(re);
  const Real vn = std::sqrt(alpha) * cs;
  const Real vs = vn * std::pow(re, -0.25);

  // calculate the relative velocities
  const Real c1 = 8.0 / M_PI * cs * cs * mu * m_p;

  const Real stokes_i = tau_i * omega;
  const Real stokes_j = tau_j * omega;

  const Real hi =
      std::min(std::sqrt(alpha / (std::min(0.5, stokes_i) * (stokes_i + 1.0))), 1.0) * hg;
  const Real hj =
      std::min(std::sqrt(alpha / (std::min(0.5, stokes_j) * (stokes_j + 1.0))), 1.0) * hg;

  // turbulent relative velocity
  Real dv0 = v_rel_ormel(tau_i, tau_j, tn, vn, ts, vs, re);

  // Brownian motion relative velocities
  const Real mredu = (mass_grid(i) * mass_grid(j) / (mass_grid(i) + mass_grid(j)));
  dv0 += (c1 / mredu);

  Real dv2_ij;
  Real hij = 1.0;
  if (coag.coord == 1) { // surface density
    const Real vs_i = std::min(stokes_i, 1.0) * omega * hi;
    const Real vs_j = std::min(stokes_j, 1.0) * omega * hj;
    dv2_ij = std::pow(vs_i - vs_j, 2);
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
    return (coag.coef_fett(i, j) * dv * pc / hij);
  } else { // fragmentation
    return (coag.coef_fett(i, j) * dv * pf / hij);
  }
}

KOKKOS_INLINE_FUNCTION
void CoagulationSource(parthenon::team_mbr_t const &mbr, ScratchPad1D<Real> &source,
                       const ScratchPad1D<Real> &distri, const int mimax,
                       const Real kernel4[], const ScratchPad1D<Real> &vel,
                       const ScratchPad1D<Real> &stoppingTime, const Coagulation &coag) {

  const int nm = coag.nm;

  // initialize source(*)
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, nm), [=](const int k) {
    // for (int k = 0; k < nm; k++) {
    source(k) = 0.0;
  });
  mbr.team_barrier();

  // Adding coagulation source terms
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, mimax + 1), [=](const int i) {
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
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, nm), [=](const int k) {
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
            As_j += coag.aFrag(i2, j2) * distri(i2) * distri(j2) * fett_l;
          }
        }
      }
      source(k) += coag.phiFrag(k, j) / coag.mass_grid(k) * As_j;
    }
  });
  mbr.team_barrier();

  // Negative terms and cratering remnants
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, mimax + 1), [=](const int j) {
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
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, mimax + 1), [=](const int i) {
    Real sum0(0.0);
    for (int j = 0; j <= i - pGrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coag.epsFrag(i, j) * Rf1;
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
                    const ScratchPad1D<Real> &stoppingTime, const Coagulation &coag) {

  const int nm = coag.nm;

  // initialize source(*)
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, nm), [=](const int k) {
    // for (int k = 0; k < nm; k++) {
    nQs(k) = 0.0;
  });
  mbr.team_barrier();

  // Adding coagulation source terms
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, mimax + 1), [=](const int i) {
    // for (int i = 0; i <= mimax; i++) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
      const Real fett_t = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 0);
      const Real Rc1 = distri(i) * distri(j) * fett_t;
      const Real Qp1 = Qplus(coag.mass_grid(i), Q(i), coag.mass_grid(j), Q(j));
      const Real dphijQ = coag.dalp(j, i) * Qp1 * (1.0 + coag.dpod(j, i)) - Q(j);
      const Real dphjiQ = coag.dalp(i, j) * Qp1 * (1.0 + coag.dpod(i, j)) - Q(i);

      const Real tmp1 = dphijQ - Qp1 * coag.dpod(j, i);
      const Real tmp2 = dphjiQ - Qp1 * coag.dpod(i, j);
      for (int nz = 0; nz < 4; nz++) {
        const int k = coag.cpod_notzero(i, j, nz);
        if (k < 0) continue;
        const Real nqs_k = (Qp1 * coag.cpod_short(i, j, nz) + tmp1 * coag.kdelta(j, k) +
                            tmp2 * coag.kdelta(i, k)) *
                           Rc1;
        Kokkos::atomic_add(&nQs(k), nqs_k);
      }
    }
  });
  mbr.team_barrier();

  // FRAGMENTATION------------------------------------------------------

  const int pGrid = coag.pGrid;
  // Adding fragment distribution
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, nm), [=](const int k) {
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
            if (j2 <= i2 - pGrid - 1) {
              Qf1 =
                  Qplus(coag.chi * coag.mass_grid(j2), Q[i2], coag.mass_grid(j2), Q[j2]);
            } else {
              Qf1 = Qplus(coag.mass_grid(i2), Q[i2], coag.mass_grid(j2), Q[j2]);
            }
            As_j += coag.aFrag(i2, j2) * distri(i2) * distri(j2) * fett_l * Qf1;
          }
        }
      }
      nQs(k) += coag.phiFrag(k, j) / coag.mass_grid(k) * As_j;
    }
  });
  mbr.team_barrier();

  // Negative terms and cratering remnants
  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, mimax + 1), [=](const int j) {
    // for (int i = 0; i <= mimax; i++) {
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

  Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, mimax + 1), [=](const int i) {
    Real sum0(0.0);
    for (int j = 0; j <= i - pGrid - 1; j++) {
      const Real fett_l = CoagulationRate(i, j, kernel4, vel, stoppingTime, coag, 1);
      const Real Rf1 = distri(i) * distri(j) * fett_l;
      const Real dummy = coag.epsFrag(i, j) * Rf1 * Q(i);
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
                     const ScratchPad1D<Real> &distri, const int mimax, const int nvel,
                     const Real kernel4[], ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const Coagulation &coag) {

  const int nm = coag.nm;

  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;

  for (int n = 0; n < nvel; n++) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, nm), [=](const int k) {
      // for (int k = 0; k < nm; k++) {
      Q(k) = vel(n + k * 3) * mom_scale;
    });
    mbr.team_barrier();

    Coagulation_nQ(mbr, nQs, Q, distri, mimax, kernel4, vel, stoppingTime, coag);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(mbr, nm), [=](const int k) {
      // for (int k = 0; k < nm; k++) {
      if (distri(k) > coag.floor_dnum(k)) {
        Real nQ1 = distri(k) * Q(k) + dt * nQs(k);
        vel(n + k * 3) = nQ1 / distri(k) * mom_iscale;
      }
    });
    mbr.team_barrier();
  }
}

KOKKOS_INLINE_FUNCTION
void CoagulationOneCell(parthenon::team_mbr_t const &mbr, const int cell_i,
                        const Real &time, Real &dt_sync, const Real &gasdens,
                        ScratchPad1D<Real> &dustdens, ScratchPad1D<Real> &stime,
                        ScratchPad1D<Real> &vel, const int nvel, ScratchPad1D<Real> &Q,
                        ScratchPad1D<Real> &nQs, const Real &alpha, const Real &cs,
                        const Real &omega, const Real &volume, Coagulation coag,
                        ScratchPad1D<Real> &source, int &nCall) {

  int nm = coag.nm;

  const Real floorVal = coag.nlim / volume;
  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;

  Real dt, time_dummy, time_goal, hnext;
  Real dt_sync1 = dt_sync;

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int i) {
    // convert to number density
    dustdens(i) /= coag.mass_grid(i); // number density
    // take the floor value
    dustdens(i) = std::max(dustdens(i), 0.01 * std::min(coag.floor_dnum(i), floorVal));
#ifdef COAGULATION_DEBUG
    if (cell_i == 2 && dustdens[i] > floorVal) {
      std::cout << " oneCellBeg: " << time << " " << i << " "
                << dustdens[i] * coag.mass_grid(i) << " " << coag.floor_dnum(i) << " "
                << stime[i] << std::endl;
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
          if (dustdens(i) > coag.floor_dnum(i)) {
            lmax = std::max(lmax, i);
          }
        },
        Kokkos::Max<int>(mimax));

    CoagulationSource(mbr, source, dustdens, mimax, kernel, vel, stime, coag);

    // time step control
    dt_sync1 = std::numeric_limits<Real>::max(); // start with a large number
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(mbr, nm),
        [&](const int i, Real &lmin) {
          if (dustdens(i) > coag.floor_dnum(i) && source(i) < 0.0) {
            lmin = std::min(lmin, std::abs(dustdens(i) / source(i)));
          }
        },
        Kokkos::Min<Real>(dt_sync1));

    dt = std::min(dt_sync1, time_goal - time_dummy);
    dt_sync = dt_sync1;

    if (coag.mom_coag) {
      Coagulation_nQs(mbr, dt, Q, nQs, dustdens, mimax, nvel, kernel, vel, stime, coag);
    }

    // integration: first-order
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1,
                             [&](const int i) { dustdens(i) += dt * source(i); });

    // Kokkos::single (Kokkos::PerTeam(mbr), [&]() {
    time_dummy += dt;
    nCall++;
    //}); mbr.team_barrier();

    if (nCall > coag.nCall_mx) break;
  } // end of internal timestep

  // back to density and velocity
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm - 1, [&](const int i) {
    if (dustdens(i) > coag.floor_dnum(i)) {
      dustdens(i) *= coag.mass_grid(i);
    } else {
      dustdens(i) = 0.0;
    }
  });
  mbr.team_barrier();

} // end of CoagulationOneCell

} // namespace Dust

#endif // DUST_COAGULATION_HPP_
