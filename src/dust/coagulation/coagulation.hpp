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
//! \fn  void Dust::Coagulation::PreCoagulationDiagnostics
//  \brief Gather pre-coagulation diagnostics
template <Coordinates GEOM, typename T>
static void CoagulationDiagnostics(MeshData<Real> *md, T &vmesh,
                                   const geometry::CoordParams &cpars, const Real &dfloor,
                                   Real &mass_d, int &max_size) {
  // Indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Reduction
  Real lmass_d = 0.0;
  int lmax_size = 1;
  Kokkos::parallel_reduce(
      "coag::diag",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {md->NumBlocks(), kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum,
                    int &lmax) {
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const Real &vol = coords.Volume();

        // Sum over nspecies
        for (int n = 0; n < vmesh.GetSize(b, dust::cons::density()); ++n) {
          lsum += vmesh(b, dust::cons::density(n), k, j, i) * vol;
        }

        // Max
        for (int n = vmesh.GetSize(b, dust::cons::density()) - 1; n >= 0; --n) {
          const Real &dens_d = vmesh(b, dust::cons::density(n), k, j, i);
          if (dens_d > dfloor) {
            lmax = std::max(lmax, n);
            break;
          }
        }
      },
      lmass_d, Kokkos::Max<int>(lmax_size));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  // Sum over all processors
  MPI_Allreduce(MPI_IN_PLACE, &lmax_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &lmass_d, 1, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  mass_d = lmass_d;
  max_size = lmax_size;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::WriteCoagulationDiagnostics
//  \brief Write coagulation diagnostics to file
static void WriteCoagulationDiagnostics(MeshData<Real> *md, const Real time,
                                        const Real dt, const int max_size1,
                                        const int max_size0, const Real mass_d1,
                                        const Real mass_d0) {
  if (parthenon::Globals::my_rank == 0) {
    auto pm = md->GetParentPointer();
    auto &artemis_pkg = pm->packages.Get("artemis");
    std::string fname;
    fname.assign(artemis_pkg->template Param<std::string>("job_name"));
    fname.append("_info.dat");
    static FILE *pfile = NULL;

    // The file exists -- reopen the file in append mode
    if (pfile == NULL) {
      if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
        if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
          PARTHENON_FAIL("Error output file could not be opened");
        }
        // The file does not exist -- open the file in write mode and add headers
      } else {
        if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
          PARTHENON_FAIL("Error output file could not be opened");
        }
        std::string label = "# time dt max_size1 max_size0 mass_d1 mass_d0 delta \n";
        std::fprintf(pfile, "%s", label.c_str());
      }
    }
    std::fprintf(pfile, "  %24.16e ", time);
    std::fprintf(pfile, "  %24.16e ", dt);
    std::fprintf(pfile, "  %d  %d ", max_size1, max_size0);
    std::fprintf(pfile, "  %24.16e  %24.16e  %24.16e", mass_d1, mass_d0,
                 1.0 - mass_d0 / mass_d1);
    std::fprintf(pfile, "\n");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::InitializeArray
//  \brief Initialize static coagulation arrays
static void InitializeArray(const int nm, int &pgrid, const Real &rho_p, const Real &chi,
                            const Real &a, const ParArray1D<Real> dsize,
                            ParArray2D<int> klf, ParArray1D<Real> mass_grid,
                            ParArray3D<Real> coag3d, ParArray3D<int> cpod_notzero,
                            ParArray3D<Real> cpod_short) {
  // Initialization Part I
  const int ikdelta = coag2drv::kdelta;
  const int icoef_fett = coag2drv::coef_fett;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag1", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < nm; j++) {
          coag3d(ikdelta, i, j) = 0.0;
        }
        coag3d(ikdelta, i, i) = 1.0;
        mass_grid(i) = 4.0 * M_PI / 3.0 * rho_p * dsize(i) * dsize(i) * dsize(i);
        for (int j = 0; j < nm; j++) {
          Real tmp1 = (1.0 - 0.5 * coag3d(ikdelta, i, j));
          coag3d(icoef_fett, i, j) = M_PI * SQR(dsize(i) + dsize(j)) * tmp1;
        }
      });

  // Set fragmentation variables
  const Real ten_a = std::pow(10.0, a);
  const Real ten_ma = 1.0 / ten_a;
  const int ce = static_cast<int>( std::floor(-1.0 / a * std::log10(1.0 - ten_ma)))  + 1;

  // Used in integration
  pgrid = static_cast<int>(std::floor(1.0 / a));

  // Initialization Part II
  const int iphifrag = coag2drv::phifrag;
  const int iepsfrag = coag2drv::epsfrag;
  const int iafrag = coag2drv::afrag;
  const Real frag_slope = 1.0/6.0; // = 2.0 - 11.0 / 6.0;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag2", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        Real sum_pF = 0.0;
        for (int j = 0; j <= i; j++) {
          coag3d(iphifrag, j, i) = std::pow(mass_grid(j), frag_slope);
          sum_pF += coag3d(iphifrag, j, i);
        }
        // normalization
        for (int j = 0; j <= i; j++) {
          coag3d(iphifrag, j, i) /= sum_pF; // switch (i,j) from fortran
        }

        // Cratering
        for (int j = 0; j <= i - pgrid - 1; j++) {
          // FRAGMENT DISTRIBUTION
          // The largest fragment has the mass of the smaller collision partner

          // Mass bin of largest fragment
          klf(i, j) = j;

          coag3d(iafrag, i, j) = (1.0 + chi) * mass_grid(j);
          //                      |_______|
          //                           |
          //                    Mass of fragments
          coag3d(iepsfrag, i, j) = chi * mass_grid(j) / (mass_grid(i) * (1.0 - ten_ma));
        }

        int i1 = std::max(0, i - pgrid);
        for (int j = i1; j <= i; j++) {
          // The largest fragment has the mass of the larger collison partner
          klf(i, j) = i;
          coag3d(iafrag, i, j) = (mass_grid(i) + mass_grid(j));
        }
      });

  // Initialization Part III
  // --> dalp array
  // --> D matrix
  // --> E matrix
  ParArray2D<Real> e("epod", nm, nm);
  int idalp = coag2drv::dalp, idpod = coag2drv::dpod;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag4", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int k) {
        for (int j = 0; j < nm; j++) {
          if (j <= k + 1 - ce) {
            coag3d(idalp, k, j) = 1.0;
            coag3d(idpod, k, j) = -mass_grid(j) / (mass_grid(k) * (ten_a - 1.0));
          } else {
            coag3d(idpod, k, j) = -1.0;
            coag3d(idalp, k, j) = 0.0;
          }
        }
        // for E matrix-------------
        const Real mkkme = mass_grid(k) * (1.0 - ten_ma);
        const Real mkpek = mass_grid(k) * (ten_a - 1.0);
        const Real mkpeme = mass_grid(k) * (ten_a - ten_ma);
        for (int j = 0; j < nm; ++j) {
          if (j <= k - ce) {
            e(k, j) = mass_grid(j) / mkkme;
          } else {
            Real theta1 = (mkpeme - mass_grid(j) < 0.0) ? 0.0 : 1.0;
            e(k, j) = (1.0 - (mass_grid(j) - mkkme) / mkpek) * theta1;
          }
        }
      });

  // Initialization Part IV
  // -->cpod array;
  ParArray3D<Real> cpod("cpod", nm, nm, nm);
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
            cpod(i, j, k) = (0.5 * coag3d(ikdelta, i, j) * cpod(i, j, k) +
                             cpod(i, j, k) * theta_kj * dtheta_ji);
          }
          cpod(i, j, j) += coag3d(idpod, j, i);
          cpod(i, j, j + 1) += e(j + 1, i) * dtheta_ji;

        } //  end if
      });

  // Initialization Part V
  // -->cpod_nonzero and cpod_short array
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

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::v_rel_ormel
//  \brief
KOKKOS_FORCEINLINE_FUNCTION
Real v_rel_ormel(Real tau_1, Real tau_2, Real t0, Real v0, Real ts, Real vs,
                 Real reynolds) {
  // Initialize variables to Null
  Real st1 = Null<Real>(), st2 = Null<Real>();
  Real tau_mx = Null<Real>(), tau_mn = Null<Real>();
  Real vg2 = Null<Real>();
  Real c0 = Null<Real>(), c1 = Null<Real>(), c2 = Null<Real>(), c3 = Null<Real>();
  Real y_star = Null<Real>(), ya = Null<Real>();
  Real eps = Null<Real>();
  Real hulp1 = Null<Real>(), hulp2 = Null<Real>();

  // Sort tau's 1--> correspond to the max. now
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

  // Return appropriate v_rel_ormel for regime
  Real sqRe = 1.0 / sqrt(reynolds);
  if (tau_mx < 0.2 * ts) {
    // Very small regime
    return 1.5 * SQR((vs / ts * (tau_mx - tau_mn)));
  } else if (tau_mx < ts / ya) {
    return vg2 * (st1 - st2) / (st1 + st2) *
           (SQR(st1) / (st1 + sqRe) - SQR(st2) / (st2 + sqRe));
  } else if (tau_mx < 5.0 * ts) {
    // Eq. 17 of oc07. the second term with st_i**2.0 is negligible (assuming re>>1)
    // hulp1 = eq. 17; hulp2 = eq. 18
    hulp1 =
        ((st1 - st2) / (st1 + st2) *
         (SQR(st1) / (st1 + ya * st1) - SQR(st2) / (st2 + ya * st1))); // note the -sign
    hulp2 = 2.0 * (ya * st1 - sqRe) + SQR(st1) / (ya * st1 + st1) -
            SQR(st1) / (st1 + sqRe) + SQR(st2) / (ya * st1 + st2) -
            SQR(st2) / (st2 + sqRe);
    return vg2 * (hulp1 + hulp2);
  } else if (tau_mx < t0 / 5.0) {
    // Full intermediate regime
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

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::theta
//  \brief
KOKKOS_FORCEINLINE_FUNCTION
Real theta(Real x) { return (x < 0 ? 0.0 : 1.0); }

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::Qplus
//  \brief Function calculates the new Q value of a particle resulting of a collision of
//         particles with masses m1, m2 and Q values Q1, Q2
KOKKOS_FORCEINLINE_FUNCTION
Real Qplus(Real m1, Real Q1, Real m2, Real Q2) { return (m1 * Q1 + m2 * Q2) / (m1 + m2); }

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::CoagulationRate
//  \brief
KOKKOS_FORCEINLINE_FUNCTION
Real CoagulationRate(const int i, const int j, const KernelParams &kernel4,
                     const ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag,
                     const int itype) {
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
  const Real ts = tn / sqrt(re);
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
KOKKOS_FORCEINLINE_FUNCTION
void CoagulationSource(parthenon::team_mbr_t const &mbr, ScratchPad1D<Real> &source,
                       const ScratchPad1D<Real> &distri, const int mimax,
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
  int iafrag = coag2drv::afrag;
  int iphifrag = coag2drv::phifrag, iepsfrag = coag2drv::epsfrag;
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
KOKKOS_FORCEINLINE_FUNCTION
void Coagulation_nQ(parthenon::team_mbr_t const &mbr, ScratchPad1D<Real> &nQs,
                    const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &distri,
                    const int mimax, const KernelParams &kernel4,
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
//! \fn  void Dust::Coagulation::Coagulation_nQs
//  \brief
KOKKOS_FORCEINLINE_FUNCTION
void Coagulation_nQs(parthenon::team_mbr_t const &mbr, const Real &dt,
                     ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                     ScratchPad1D<Real> &distri, const int mimax, const int nvel,
                     const KernelParams &kernel4, ScratchPad1D<Real> &vel,
                     const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag,
                     ScratchPad1D<Real> &source) {
  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;
  for (int n = 0; n < nvel; n++) {
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                             [&](const int k) {
                               Q(k) = vel(n + k * 3) * mom_scale;
                               nQs(k) = 0.0; // initialize source(*)
                             });
    mbr.team_barrier();

    Coagulation_nQ(mbr, nQs, Q, distri, mimax, kernel4, vel, stoppingTime, coag);

    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                             [&](const int k) {
                               const Real distri_k = distri(k) + dt * source(k);
                               if (distri_k > coag.dfloor / coag.mass_grid(k)) {
                                 const Real nQ1 = distri(k) * Q(k) + dt * nQs(k);
                                 vel(n + k * 3) = nQ1 / distri_k * mom_iscale;
                               }
                             });
    mbr.team_barrier();
  }

  // update the density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                           [&](const int k) { distri(k) += dt * source(k); });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::Coagulation_nQs3
//  \brief
KOKKOS_FORCEINLINE_FUNCTION
void Coagulation_nQs3(parthenon::team_mbr_t const &mbr, const Real &dt,
                      ScratchPad1D<Real> &Q, ScratchPad1D<Real> &nQs,
                      ScratchPad1D<Real> &distri, const int mimax, const int nvel,
                      const KernelParams &kernel4, ScratchPad1D<Real> &vel,
                      const ScratchPad1D<Real> &stoppingTime, const CoagParams &coag,
                      ScratchPad1D<Real> &source, ScratchPad1D<Real> &Q2,
                      const int mimax2) {
  const Real mom_scale = 1.0e10;
  const Real mom_iscale = 1.0e-10;
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                           [&](const int k) {
                             Q2(k) = nQs(k); // 2nd stage source
                           });
  mbr.team_barrier();

  for (int n = 0; n < nvel; n++) {
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                             [&](const int k) {
                               Q(k) = vel(n + k * 3) * mom_scale;
                               nQs(k) = 0.0;
                             });
    mbr.team_barrier();

    // 1st stage
    Coagulation_nQ(mbr, nQs, Q, distri, mimax, kernel4, vel, stoppingTime, coag);

    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
                             [&](const int k) {
                               const Real distri_k = distri(k) + dt * source(k);
                               const Real nQ1 = distri(k) * Q(k) + dt * nQs(k);
                               Q(k) = nQ1 / distri_k; // intermediate Q
                             });
    mbr.team_barrier();

    // 2nd stage
    Coagulation_nQ(mbr, nQs, Q, distri, mimax2, kernel4, vel, stoppingTime, coag);

    parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1, [&](const int k) {
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
  parthenon::par_for_inner(
      DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
      [&](const int k) { distri(k) += 0.5 * dt * (source(k) + Q2(k)); });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CoagulationOneCell
//  \brief
KOKKOS_FORCEINLINE_FUNCTION
void CoagulationOneCell(parthenon::team_mbr_t const &mbr, const int cell_i,
                        const Real &time, Real &dt_sync, const Real &gdens,
                        ScratchPad1D<Real> &dustdens, ScratchPad1D<Real> &stime,
                        ScratchPad1D<Real> &vel, const int nvel, ScratchPad1D<Real> &Q,
                        ScratchPad1D<Real> &nQs, const Real &alpha, const Real &cs,
                        const Real &omega, const CoagParams &coag,
                        ScratchPad1D<Real> &source, int &nCall, ScratchPad1D<Real> &Q2) {
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

    if (coag.use_adaptive == 0 || coag.integrator == 1) {
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

      if (coag.mom_coag) {
        Coagulation_nQs(mbr, dt, Q, nQs, dustdens, mimax, nvel, kernel, vel, stime, coag,
                        source);
      } else {
        // integration: first-order
        parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
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
            Kokkos::TeamThreadRange(mbr, coag.nm),
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
        errmax /= coag.err_eps;

        if (errmax <= 1.0) break;

        h = std::max(coag.S * h * std::pow(errmax, coag.pshrink), 0.1 * h);
      }

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
            DEFAULT_INNER_LOOP_PATTERN, mbr, 0, coag.nm - 1,
            [&](const int i) { dustdens(i) += 0.5 * dt * (source(i) + nQs(i)); });
        mbr.team_barrier();
      }
    }

    // Update time and increment ncall
    time_dummy += dt;
    nCall++;

    // Adaptivity
    if (coag.use_adaptive) {
      dt_sync = std::max(hnext, dt_sync);
      hnext = std::min(hnext, time_goal - time_dummy);
    }

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
