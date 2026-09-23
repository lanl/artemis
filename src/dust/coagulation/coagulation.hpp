//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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
// This closely follows the implementation in Stammler and Birnstiel (2022) ApJ 935:35
//========================================================================================
#ifndef DUST_COAGULATION_COAGULATION_HPP_
#define DUST_COAGULATION_COAGULATION_HPP_

#include "utils/artemis_utils.hpp"
#include "utils/units.hpp"
using ArtemisUtils::VI;

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
                     const Real &a, const ParArray1D<Real> dsize,
                     ParArray2D<int> idx_largest, ParArray1D<Real> mass_grid,
                     ParArray3D<Real> coag3d, ParArray3D<int> Kijk_sym_ind,
                     ParArray3D<Real> Kijk_sym);

// Diagnostics functions
// using DiagPack_t = parthenon::SparsePack<gas::prim::density, gas::prim::sie,
//                                          dust::cons::density, dust::cons::momentum,
//                                          dust::prim::density, dust::prim::velocity>;
//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::PreCoagulationDiagnostics
//  \brief Gather pre-coagulation diagnostics
template <Coordinates GEOM, typename DiagPack_t>
void CoagulationDiagnostics(MeshData<Real> *md, DiagPack_t &vmesh,
                            const geometry::CoordParams &cpars, const Real &dfloor,
                            Real &mass_d, int &max_size) {
  // Indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Reduction
  Real lmass_d = 0.0;
  int lmax_size = 1;

  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "coag::diag", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
          if (dens_d > 5.0 * dfloor) {
            lmax = Kokkos::max(lmax, n);
            break;
          }
        }
      },
      Kokkos::Sum<Real>(lmass_d), Kokkos::Max<int>(lmax_size));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  // Sum over all processors
  MPI_Allreduce(MPI_IN_PLACE, &lmax_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &lmass_d, 1, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  mass_d = lmass_d;
  max_size = lmax_size;
}

void WriteCoagulationDiagnostics(MeshData<Real> *md, const Real time, const Real dt,
                                 const int max_size1, const int max_size0,
                                 const Real mass_d1, const Real mass_d0);

// Constants that enumerate coagulation kernel
enum cidx { Djk, Aij, Pij, epsij, dalp, rate_coef };
enum class DustInteractionType { Coagulation, Fragmentation };

// Coagulation State Parameters
// NOTE(@pdmullen): Shared between various inline function calls
struct StateParams {
  Real gdens;
  Real alpha;
  Real cs;
  Real kT;
  Real omega;
  int nvel;
};

// Struct that holds coagulation parameters
struct CoagParams {
  bool coord; // true--surface density, false: 3D

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
  Real err_con;     // Needed for increasing step size
  bool const_omega; // for shearing-box or testing

  Real rho0;    // physical-to-code unit conversion density
  Real length0; // physical-to-code unit conversion length
};

// Coagulation Rate Parameters
// NOTE(@pdmullen): Shared between CoagulationRate calls
struct RateParams {
  Real mmw;           // mean molecular weight (mu * amu in CGS)
  Real cross_section; // cross section of gas species
  Real vfrag;         // fragmentation velocity
  bool ibounce;       // include bouncing
};

// Struct that holds coagulation arrays
struct CoagArrays {
  // pre-calculated arrays
  ParArray2D<int> idx_largest; // index of largest fragment
  ParArray1D<Real> mass_grid;
  ParArray3D<Real> coagR3D;
  ParArray3D<int> Kijk_sym_ind;
  ParArray3D<Real> Kijk_sym;
};

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::GetRelativeTurbulentVelocity
//  \brief
//  Implementation of Ormel & Cuzzi 2007 relative turbulent velocity between two particles
KOKKOS_INLINE_FUNCTION
Real GetRelativeTurbulentVelocity(const Real &tau_1, const Real &tau_2, const Real &t0,
                                  const Real &v0, const Real &ts, const Real &vs,
                                  const Real &reynolds) {
  constexpr Real ya =
      1.600307102383245; // approx solution for st*=y*st1; valid for st1 << 1

  const Real tau_mx = (tau_1 >= tau_2) ? tau_1 : tau_2;
  const Real tau_mn = (tau_1 >= tau_2) ? tau_2 : tau_1;
  const Real st1 = tau_mx / t0;
  const Real st2 = tau_mn / t0;
  const Real vg2 = 1.5 * SQR(v0); // note the square

  const Real sqRe = 1.0 / Kokkos::sqrt(reynolds);
  if (tau_mx < 0.2 * ts) {
    // Very small regime
    return 1.5 * SQR((vs / ts * (tau_mx - tau_mn)));
  } else if (tau_mx < ts / ya) {
    return vg2 * (st1 - st2) / (st1 + st2) *
           (SQR(st1) / (st1 + sqRe) - SQR(st2) / (st2 + sqRe));
  } else if (tau_mx < 5.0 * ts) {
    // Eq. 17 & 18 of oc07. the second term with st_i**2.0 is negligible (assuming re>>1)
    const Real dv1 =
        ((st1 - st2) / (st1 + st2) *
         (SQR(st1) / (st1 + ya * st1) - SQR(st2) / (st2 + ya * st1))); // note the -sign
    const Real dv2 = 2.0 * (ya * st1 - sqRe) + SQR(st1) / (ya * st1 + st1) -
                     SQR(st1) / (st1 + sqRe) + SQR(st2) / (ya * st1 + st2) -
                     SQR(st2) / (st2 + sqRe);
    return vg2 * (dv1 + dv2);
  } else if (tau_mx <= t0) { // st1 <= 1
    // now y* lies between 1.6 (st1 << 1) and 1.0 (st1>=1).
    // This fits a quartic to y_star from oc07. Extra weight is given to the end points.
    // The maximum relative error is 0.00198 for a Newton-Raphson solution to
    // Eq. 21d from St=[1e-5,1] logspaced that had a tolerance of 1e-14
    constexpr Real c[5] = {ya, -0.5862123458950033, 0.08425017502816909,
                           0.11676288996447193, -0.2149691024550422};
    const Real y_star =
        Kokkos::max(1.0, c[0] + (c[1] + (c[2] + (c[3] * st1) * st1) * st1) * st1);
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
//! \fn  Real Dust::Coagulation::iHeaviSide
//  \brief inverted Heviside function
KOKKOS_FORCEINLINE_FUNCTION
Real iHeaviSide(const Real &x) { return (x < 0 ? 0.0 : 1.0); }

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::Qplus
//  \brief Function calculates the new Q value of a particle resulting of a collision of
//         particles with masses m1, m2 and Q values Q1, Q2
KOKKOS_FORCEINLINE_FUNCTION
Real Qplus(const Real &m1, const Real &Q1, const Real &m2, const Real &Q2) {
  return (m1 * Q1 + m2 * Q2) / (m1 + m2);
}

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::CoagulationRate
//  \brief Calculate Rij
template <DustInteractionType DIT>
KOKKOS_INLINE_FUNCTION Real CoagulationRate(const int &i, const int &j, const int &nm1,
                                            const StateParams &kernel,
                                            const ScratchPad1D<Real> &vel,
                                            const ScratchPad1D<Real> &stime,
                                            const ParArray1D<Real> &mass_grid,
                                            const ParArray3D<Real> &coagR3D,
                                            const bool &surface, const RateParams &rate) {
  const Real &mass_gridi = mass_grid(i);
  const Real &mass_gridj = mass_grid(j);
  const Real &mass_gride = mass_grid(nm1);
  if (mass_gridi + mass_gridj >= mass_gride) return 0.0;

  const Real &gdens = kernel.gdens;
  const Real &alpha = kernel.alpha;
  const Real &cs = kernel.cs;
  const Real &omega = kernel.omega;
  const int &nvel = kernel.nvel;
  const Real &tau_i = stime(i);
  const Real &tau_j = stime(j);
  const Real *vel_i = &vel(nvel * i);
  const Real *vel_j = &vel(nvel * j);

  const Real &sig = rate.cross_section; //! cross section of gas species
  const Real &mmw = rate.mmw;           //! mean molecular weight (mu * mp)

  // Calculate some basic properties
  const Real hg = cs / omega;
  Real re = alpha * sig * gdens / (2.0 * mmw);
  if (!(surface)) re *= Kokkos::sqrt(2.0 * M_PI) * hg;

  const Real tn = 1.0 / omega;
  const Real ts = tn / Kokkos::sqrt(re);
  const Real vn = Kokkos::sqrt(alpha) * cs;
  const Real vs = vn * Kokkos::pow(re, -0.25);

  // Calculate Stokes number
  const Real stokes_i = tau_i * omega;
  const Real stokes_j = tau_j * omega;

  const Real muij = (mass_gridi * mass_gridj / (mass_gridi + mass_gridj));
  // Calculate turbulent relative velocity
  // turbulent + brownian + actual
  Real dv2 = GetRelativeTurbulentVelocity(tau_i, tau_j, tn, vn, ts, vs, re) +
             Kokkos::min(cs * cs, 8 / M_PI * kernel.kT / muij) +
             SQR(vel_i[0] - vel_j[0]) + SQR(vel_i[1] - vel_j[1]);
  if (nvel > 2) {
    dv2 += SQR(vel_i[2] - vel_j[2]);
  }
  Real hij = 1.0;
  if (surface) { // surface density
    const Real hi = Kokkos::sqrt(1.0 / (1.0 + stokes_i / alpha)) * hg;
    const Real hj = Kokkos::sqrt(1.0 / (1.0 + stokes_j / alpha)) * hg;
    const Real vs_i = Kokkos::min(stokes_i, 0.5) * omega * hi;
    const Real vs_j = Kokkos::min(stokes_j, 0.5) * omega * hj;
    // relative velocity from vertical settling
    dv2 += SQR(vs_i - vs_j);
    hij = Kokkos::sqrt(2.0 * M_PI * (SQR(hi) + SQR(hj)));
  }

  const Real dv = Kokkos::sqrt(dv2);

  // New pf calculation: for fragmenation
  Real pf = 0.0;
  if (dv > 0.0) {
    // Eq. 60 of SB22
    const Real tmp = 1.5 * SQR(rate.vfrag / dv);
    pf = (tmp + 1.0) * Kokkos::exp(-tmp);
  }

  if constexpr (DIT == DustInteractionType::Coagulation) {
    if (rate.ibounce && dv > 0.0) { // including bouncing effect
      // NOTE(AMD): these should be input params
      const Real froll = 1e-4; // Heim et al.(PRL) 1999
      const Real amono = 1e-4; // micro-size
      const Real vbounce = Kokkos::sqrt(5.0 * M_PI * amono * froll / muij);
      if (vbounce < rate.vfrag) {
        const Real tmp = 1.5 * SQR(vbounce / dv);
        pf = (tmp + 1.0) * Kokkos::exp(-tmp); // using bouncing vel
      }
    }
    pf = 1.0 - pf;
  }

  return coagR3D(cidx::rate_coef, i, j) * dv * pf / hij;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::Calculate_Kenels
//  \brief Calculate the rate matrix fett_t, fett_l
KOKKOS_INLINE_FUNCTION
void Calculate_Kernels(const parthenon::team_mbr_t &mbr, const StateParams &kernel,
                       const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &dustdens,
                       const ScratchPad1D<Real> &stime, const int &nm1,
                       const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                       const bool &surface, const RateParams &rate,
                       ScratchPad1D<Real> &fett_t, ScratchPad1D<Real> &fett_l) {

  const Real &gdens = kernel.gdens;
  const Real &alpha = kernel.alpha;
  const Real &cs = kernel.cs;
  const Real &omega = kernel.omega;
  const int &nvel = kernel.nvel;

  // calculate some basic properties
  const Real &sig = rate.cross_section; //! cross section of gas species
  const Real &mmw = rate.mmw;           //! mean molecular weight (mu * mp)

  // Calculate some basic properties
  const Real hg = cs / omega;
  Real re = alpha * sig * gdens / (2.0 * mmw);
  if (!(surface)) re *= Kokkos::sqrt(2.0 * M_PI) * hg;

  const Real tn = 1.0 / omega;
  const Real ts = tn / Kokkos::sqrt(re);
  const Real vn = Kokkos::sqrt(alpha) * cs;
  const Real vs = vn * Kokkos::pow(re, -0.25);

  const int nm = nm1 + 1;

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    for (int j = 0; j <= i; ++j) {
      const Real &mass_gridi = mass_grid(i);
      const Real &mass_gridj = mass_grid(j);
      const Real &mass_gride = mass_grid(nm1);
      // avoid outflow
      if (mass_gridi + mass_gridj >= mass_gride) {
        fett_t(j + i * nm) = fett_l(j + i * nm) = 0.0;
      } else {

        // turbulent relative velocity
        const Real &tau_i = stime(i);
        const Real &tau_j = stime(j);
        const Real *vel_i = &vel(nvel * i);
        const Real *vel_j = &vel(nvel * j);

        // Calculate Stokes number
        const Real stokes_i = tau_i * omega;
        const Real stokes_j = tau_j * omega;

        const Real muij = (mass_gridi * mass_gridj / (mass_gridi + mass_gridj));
        // Calculate turbulent relative velocity
        // turbulent + brownian + actual
        Real dv2 = (GetRelativeTurbulentVelocity(tau_i, tau_j, tn, vn, ts, vs, re) +
                    Kokkos::min(cs * cs, 8 / M_PI * kernel.kT / muij));
        if (dustdens(i) * dustdens(j) > 0.0) {
          // add velocity difference only when none of two bins is empty
          for (int n = 0; n < nvel; n++) {
            dv2 += SQR(vel_i[n] - vel_j[n]);
          }
        }

        Real hij = 1.0;
        if (surface) { // surface density
          const Real hi = Kokkos::sqrt(1.0 / (1.0 + stokes_i / alpha)) * hg;
          const Real hj = Kokkos::sqrt(1.0 / (1.0 + stokes_j / alpha)) * hg;
          const Real vs_i = Kokkos::min(stokes_i, 0.5) * omega * hi;
          const Real vs_j = Kokkos::min(stokes_j, 0.5) * omega * hj;
          // relative velocity from vertical settling
          dv2 += SQR(vs_i - vs_j);
          hij = Kokkos::sqrt(2.0 * M_PI * (SQR(hi) + SQR(hj)));
        }

        const Real dv = Kokkos::sqrt(dv2);

        // New pf calculation: for fragmenation
        Real pf = 0.0;
        if (dv > 0.0) {
          // Eq. 60 of SB22
          const Real tmp = 1.5 * SQR(rate.vfrag / dv);
          pf = (tmp + 1.0) * Kokkos::exp(-tmp);
        }

        if (rate.ibounce && dv > 0.0) { // including bouncing effect
          // NOTE(AMD): these should be input params
          const Real froll = 1e-4; // Heim et al.(PRL) 1999
          const Real amono = 1e-4; // micro-size
          const Real vbounce = Kokkos::sqrt(5.0 * M_PI * amono * froll / muij);
          if (vbounce < rate.vfrag) {
            const Real tmp = 1.5 * SQR(vbounce / dv);
            pf = (tmp + 1.0) * Kokkos::exp(-tmp); // using bouncing vel
          }
        }
        const Real tmp = coagR3D(cidx::rate_coef, i, j) * dv / hij;

        fett_t(j + i * nm) = tmp * (1.0 - pf); // coagulation rate
        fett_l(j + i * nm) = tmp * pf;         // fragmentation rate
      }
    } // end loop of j [0,i)
  }); // end loop of i
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::ConvertToNumberDensity
//  \brief convert to number density
KOKKOS_INLINE_FUNCTION
void ConvertToNumberDensity(const parthenon::team_mbr_t &mbr, const int &nm1,
                            const ScratchPad1D<Real> &dustdens,
                            const ParArray1D<Real> &mass_grid, const Real &dfloor) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    const Real &mass_gridi = mass_grid(i);
    dustdens(i) /= mass_gridi;
    dustdens(i) = Kokkos::max(dustdens(i), 0.01 * dfloor / mass_gridi);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::ConvertToVolumeDensity
//  \brief convert to volume density
KOKKOS_INLINE_FUNCTION
void ConvertToVolumeDensity(const parthenon::team_mbr_t &mbr, const int &nm1,
                            const ScratchPad1D<Real> &dustdens,
                            const ParArray1D<Real> &mass_grid, const Real &dfloor) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    const bool flg_dfloor = (dustdens(i) > dfloor / mass_grid(i));
    dustdens(i) *= (flg_dfloor)*mass_grid(i);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::TimeStepControl
//  \brief set the coagulation time step
KOKKOS_INLINE_FUNCTION
Real TimeStepControl(const parthenon::team_mbr_t &mbr, const int &nm1,
                     const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &source,
                     const ParArray1D<Real> &mass_grid, const Real &dfloor,
                     const Real &cfl) {
  Real dt_sync = Big<Real>();

  parthenon::par_reduce_inner(
      parthenon::inner_loop_pattern_ttr_tag, mbr, 0, nm1,
      [&](const int i, Real &lmin) {
        if (dustdens(i) > dfloor / mass_grid(i) && source(i) < 0.0) {
          lmin = Kokkos::min(lmin, Kokkos::abs(dustdens(i) / source(i)));
        }
      },
      Kokkos::Min<Real>(dt_sync));
  return cfl * dt_sync;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::ZeroSource
//  \brief zero out the mass source
KOKKOS_INLINE_FUNCTION
void ZeroSource(const parthenon::team_mbr_t &mbr, const int &nm1,
                const ScratchPad1D<Real> &source) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { source(i) = 0.0; });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::InitializeSource
//  \brief initialize the mass source
KOKKOS_INLINE_FUNCTION
void InitializeSource(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
                      const ScratchPad1D<Real> &source,
                      const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &vel,
                      const ScratchPad1D<Real> &stime, const StateParams &kernel,
                      const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                      const ParArray3D<int> &Kijk_sym_ind,
                      const ParArray3D<Real> &Kijk_sym, const bool &surface,
                      const RateParams &rate_par, const ScratchPad1D<Real> &fett_t) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // Calculate the Rate
      const Real rate = fett_t(j + i * (nm1 + 1));
      // const Real rate =  CoagulationRate<DustInteractionType::Coagulation>(
      //     i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
      const Real val = dustdens(i) * dustdens(j) * rate;
      for (int nz = 0; nz < 4; nz++) {
        const int k = Kijk_sym_ind(i, j, nz);
        if (k >= 0) {
          const Real src_coag0 = Kijk_sym(i, j, nz) * val;
          Kokkos::atomic_add(&source(k), src_coag0);
        }
      }
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::FragmentationSource
//  \brief compute the fragmentation source
KOKKOS_INLINE_FUNCTION
void FragmentationSource(const parthenon::team_mbr_t &mbr, const int &nm1,
                         const int &mimax, const int &pgrid,
                         const ScratchPad1D<Real> &source,
                         const ScratchPad1D<Real> &dustdens,
                         const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
                         const StateParams &kernel, const ParArray1D<Real> &mass_grid,
                         const ScratchPad1D<Real> &As, const ParArray3D<Real> &coagR3D,
                         const bool &surface, const RateParams &rate_par,
                         const ScratchPad1D<Real> &fett_l) {

  // Adding the collision rates to the fragments distribution
  // first initialize the array As(*) to zero
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { As(i) = 0.0; });
  mbr.team_barrier();

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    const Real di = dustdens(i);
    const int jsplit = (i > pgrid) ? i - pgrid : 0;
    // Region 1: k = j
    for (int j = 0; j < jsplit; ++j) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      // const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
      // 	  i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, dustdens, surface,
      // rate_par);
      const Real val = coagR3D(cidx::Aij, i, j) * di * dustdens(j) * rate;
      Kokkos::atomic_add(&As(j), val);
    }

    // Region 2: k = i
    Real sum_i = 0.0;
    for (int j = jsplit; j <= i; ++j) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      // const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
      // 	  i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, dustdens, surface,
      // rate_par);
      sum_i += coagR3D(cidx::Aij, i, j) * di * dustdens(j) * rate;
    }
    Kokkos::atomic_add(&As(i), sum_i);
  });
  mbr.team_barrier();

  // Adding fragment distribution
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int k) {
    for (int j = k; j <= mimax; j++) {
      source(k) += coagR3D(cidx::Pij, k, j) / mass_grid(k) * As(j);
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CrateringSource
//  \brief compute the cratering source
KOKKOS_INLINE_FUNCTION
void CrateringSource(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
                     const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &dustdens,
                     const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
                     const StateParams &kernel, const ParArray1D<Real> &mass_grid,
                     const ParArray3D<Real> &coagR3D, const bool &surface,
                     const RateParams &rate_par, const ScratchPad1D<Real> &fett_l) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      // const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
      //     i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
      sum0 -= dustdens(i) * dustdens(j) * rate;
    }
    source(j) += sum0;
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::FinalizeSource
//  \brief Finish up the mass source
KOKKOS_INLINE_FUNCTION
void FinalizeSource(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
                    const int &pgrid, const ScratchPad1D<Real> &source,
                    const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &vel,
                    const ScratchPad1D<Real> &stime, const StateParams &kernel,
                    const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                    const bool &surface, const RateParams &rate_par,
                    const ScratchPad1D<Real> &fett_l) {
  // Cratering
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    const int jsplit = (i > pgrid) ? i - pgrid : 0;
    Real sum0 = 0.0;
    for (int j = 0; j < jsplit; j++) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      // const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
      //     i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
      const Real val = dustdens(i) * dustdens(j) * rate;
      sum0 += coagR3D(cidx::epsij, i, j) * val;
    }

    if (jsplit > 0) {
      Kokkos::atomic_add(&source(i - 1), sum0);
    }

    sum0 = -sum0;
    // Full fragmentation (only negative terms)
    for (int j = jsplit; j <= i; j++) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      // const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
      //     i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
      sum0 -= dustdens(i) * dustdens(j) * rate;
    }
    Kokkos::atomic_add(&source(i), sum0);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::ZeroSourceNQ
//  \brief Zero the momentum source for direction n
KOKKOS_INLINE_FUNCTION
void ZeroSourceNQ(const parthenon::team_mbr_t &mbr, const int &n, const int &nm1,
                  const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &nQs,
                  const ScratchPad1D<Real> &vel, const int nvel) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    Q(i) = vel(n + i * nvel) * mom_scale;
    nQs(i) = 0.0;
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::InitializeSourceNQ
//  \brief Initialize the momentum source for direction n
KOKKOS_INLINE_FUNCTION
void InitializeSourceNQ(const parthenon::team_mbr_t &mbr, const int &nm1,
                        const int &mimax, const ScratchPad1D<Real> &Q,
                        const ScratchPad1D<Real> &nQs, const ScratchPad1D<Real> &dustdens,
                        const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
                        const StateParams &kernel, const ParArray1D<Real> &mass_grid,
                        const ParArray3D<Real> &coagR3D,
                        const ParArray3D<int> &Kijk_sym_ind,
                        const ParArray3D<Real> &Kijk_sym, const bool &surface,
                        const RateParams &rate_par, const ScratchPad1D<Real> &fett_t) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
      const Real rate = fett_t(j + i * (nm1 + 1));
      const Real val = dustdens(i) * dustdens(j) * rate;
      const Real &mass_gridi = mass_grid(i);
      const Real &mass_gridj = mass_grid(j);
      const Real &dalpji = coagR3D(cidx::dalp, j, i);
      const Real &Djkji = coagR3D(cidx::Djk, j, i);
      const Real &dalpij = coagR3D(cidx::dalp, i, j);
      const Real &Djkij = coagR3D(cidx::Djk, i, j);

      const Real Qp1 = Qplus(mass_gridi, Q(i), mass_gridj, Q(j));
      const Real dphijQ = dalpji * Qp1 * (1.0 + Djkji) - Q(j);
      const Real dphjiQ = dalpij * Qp1 * (1.0 + Djkij) - Q(i);

      const Real tmp1 = dphijQ - Qp1 * Djkji;
      const Real tmp2 = dphjiQ - Qp1 * Djkij;
      for (int nz = 0; nz < 4; nz++) {
        const int k = Kijk_sym_ind(i, j, nz);
        if (k >= 0) {
          const Real nqs_k =
              (Qp1 * Kijk_sym(i, j, nz) + tmp1 * (j == k) + tmp2 * (i == k)) * val;
          Kokkos::atomic_add(&nQs(k), nqs_k);
        }
      }
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::FragmentationSourceNQ
//  \brief Fragmentation momentum source for direction n
KOKKOS_INLINE_FUNCTION
void FragmentationSourceNQ(const parthenon::team_mbr_t &mbr, const int &nm1,
                           const int &mimax, const int &pgrid,
                           const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &nQs,
                           const ScratchPad1D<Real> &dustdens,
                           const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
                           const StateParams &kernel, const ParArray1D<Real> &mass_grid,
                           const ScratchPad1D<Real> &As, const ParArray3D<Real> &coagR3D,
                           const Real &chi, const bool &surface,
                           const RateParams &rate_par, const ScratchPad1D<Real> &fett_l) {

  // Adding the collision rates to the fragments distribution
  // first initialize the array to zero
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { As(i) = 0.0; });
  mbr.team_barrier();

  // calculate As(j) on fly
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    const Real di = dustdens(i);
    const int jsplit = (i > pgrid) ? i - pgrid : 0;
    // Region 1: k = j
    for (int j = 0; j < jsplit; ++j) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      // const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
      // 	  i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, dustdens, surface,
      // rate_par);
      const Real Qf1 = Qplus(chi * mass_grid(j), Q(i), mass_grid(j), Q(j));
      const Real val = coagR3D(cidx::Aij, i, j) * di * dustdens(j) * rate * Qf1;
      Kokkos::atomic_add(&As(j), val);
    }

    // Region 2: k = i
    Real sum_i = 0.0;
    for (int j = jsplit; j <= i; ++j) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      // const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
      // 	  i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, dustdens, surface,
      // rate_par);
      const Real Qf1 = Qplus(mass_grid(i), Q(i), mass_grid(j), Q(j));
      sum_i += coagR3D(cidx::Aij, i, j) * di * dustdens(j) * rate * Qf1;
    }
    Kokkos::atomic_add(&As(i), sum_i);
  });
  mbr.team_barrier();

  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int k) {
    for (int j = k; j <= mimax; j++) {
      nQs(k) += coagR3D(cidx::Pij, k, j) / mass_grid(k) * As(j);
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CrateringSourceNQ
//  \brief Cratering momentum source for direction n
KOKKOS_INLINE_FUNCTION
void CrateringSourceNQ(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
                       const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &nQs,
                       const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &vel,
                       const ScratchPad1D<Real> &stime, const StateParams &kernel,
                       const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                       const bool &surface, const RateParams &rate_par,
                       const ScratchPad1D<Real> &fett_l) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    // Cratering
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      sum0 -= dustdens(i) * dustdens(j) * rate;
    }
    nQs(j) += (sum0 * Q(j));
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::FinalizeSourceNQ
//  \brief Finalize momentum source for direction n
KOKKOS_INLINE_FUNCTION
void FinalizeSourceNQ(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
                      const int &pgrid, const ScratchPad1D<Real> &Q,
                      const ScratchPad1D<Real> &nQs, const ScratchPad1D<Real> &dustdens,
                      const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
                      const StateParams &kernel, const ParArray1D<Real> &mass_grid,
                      const ParArray3D<Real> &coagR3D, const bool &surface,
                      const RateParams &rate_par, const ScratchPad1D<Real> &fett_l) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    const int jsplit = (i > pgrid) ? i - pgrid : 0;
    Real sum0 = 0.0;
    for (int j = 0; j < jsplit; j++) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      const Real Rf1 = dustdens(i) * dustdens(j) * rate;
      sum0 += coagR3D(cidx::epsij, i, j) * Rf1 * Q(i);
    }

    if (jsplit > 0) {
      Kokkos::atomic_add(&nQs(i - 1), sum0);
    }

    sum0 = -sum0;
    // Full fragmentation (only negative terms)
    for (int j = jsplit; j <= i; j++) {
      const Real rate = fett_l(j + i * (nm1 + 1));
      sum0 -= dustdens(i) * dustdens(j) * rate * Q(i);
    }
    Kokkos::atomic_add(&nQs(i), sum0);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::UpdateVelocityNQ
//  \brief Update velocity from momentum evolution for direction n
KOKKOS_INLINE_FUNCTION
void UpdateVelocityNQ(const parthenon::team_mbr_t &mbr, const int &n, const int &nm1,
                      const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &nQs,
                      const ScratchPad1D<Real> &vel, const int nvel,
                      const ScratchPad1D<Real> &dustdens,
                      const ScratchPad1D<Real> &source, const ParArray1D<Real> &mass_grid,
                      const Real &dt, const Real &dfloor) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    const Real distri_i = dustdens(i) + dt * source(i);
    if (distri_i > dfloor / mass_grid(i)) {
      const Real nQ1 = dustdens(i) * Q(i) + dt * nQs(i);
      vel(n + nvel * i) = nQ1 / distri_i * mom_iscale;
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::UpdateDensity
//  \brief Update density from new mass
KOKKOS_INLINE_FUNCTION
void UpdateDensity(const parthenon::team_mbr_t &mbr, const int &nm1,
                   const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &source,
                   const Real &dt) {
  // Update the density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { dustdens(i) += dt * source(i); });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::FindMIMax
//  \brief Find mimax
KOKKOS_INLINE_FUNCTION
int FindMIMax(parthenon::team_mbr_t const &mbr, const int &nm1,
              const ScratchPad1D<Real> &dustdens, const ParArray1D<Real> &mass_grid,
              const Real &dfloor) {
  int mimax = 0;
  parthenon::par_reduce_inner(
      parthenon::inner_loop_pattern_ttr_tag, mbr, 0, nm1,
      [&](const int i, int &lmax) {
        if (dustdens(i) > dfloor / mass_grid(i)) lmax = Kokkos::max(lmax, i);
      },
      Kokkos::Max<int>(mimax));
  return mimax;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::UpdateDensityNQS3
//  \brief
KOKKOS_INLINE_FUNCTION
void UpdateDensityNQS3(const parthenon::team_mbr_t &mbr, const int &nm1,
                       const ScratchPad1D<Real> &dustdens,
                       const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &Q2,
                       const Real &dt) {
  // Update the density
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    dustdens(i) += 0.5 * dt * (source(i) + Q2(i));
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::FindMIMaxNQS3
//  \brief
KOKKOS_INLINE_FUNCTION
int FindMIMaxNQS3(parthenon::team_mbr_t const &mbr, const int &nm1, const Real &h,
                  const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &source,
                  const ScratchPad1D<Real> &Q, const ParArray1D<Real> &mass_grid,
                  const Real &dfloor) {
  int mimax = 0;
  parthenon::par_reduce_inner(
      parthenon::inner_loop_pattern_ttr_tag, mbr, 0, nm1,
      [&](const int i, int &lmax) {
        Q(i) = dustdens(i) + h * source(i);
        if (Q(i) > dfloor / mass_grid(i)) lmax = Kokkos::max(lmax, i);
      },
      Kokkos::Max<int>(mimax));
  return mimax;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::InitializeTempNQS3
//  \brief
KOKKOS_INLINE_FUNCTION
void InitializeTempNQS3(const parthenon::team_mbr_t &mbr, const int &n, const int &nm1,
                        const ScratchPad1D<Real> &Q2, const ScratchPad1D<Real> &nQs) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { Q2(i) = nQs(i); });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::IntermediateNQS3
//  \brief
KOKKOS_INLINE_FUNCTION
void IntermediateNQS3(const parthenon::team_mbr_t &mbr, const int &nm1,
                      const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &nQs,
                      const ScratchPad1D<Real> &dustdens,
                      const ScratchPad1D<Real> &source, const Real &dt) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    const Real distri_i = dustdens(i) + dt * source(i);
    const Real nQ1 = dustdens(i) * Q(i) + dt * nQs(i);
    Q(i) = nQ1 / distri_i; // intermediate Q
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::UpdateVelocityNQ
//  \brief
KOKKOS_INLINE_FUNCTION
void UpdateVelocityNQS3(const parthenon::team_mbr_t &mbr, const int &n, const int &nm1,
                        const ScratchPad1D<Real> &Q2, const ScratchPad1D<Real> &nQs,
                        const ScratchPad1D<Real> &vel, const int nvel,
                        const ScratchPad1D<Real> &dustdens,
                        const ScratchPad1D<Real> &source,
                        const ParArray1D<Real> &mass_grid, const Real &dt,
                        const Real &dfloor) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    const Real distri_i = dustdens(i) + 0.5 * dt * (source(i) + Q2(i));
    if (distri_i > dfloor / mass_grid(i)) {
      const Real nQ1 = (dustdens(i) * vel(n + i * nvel) * mom_scale) + 0.5 * dt * nQs(i);
      vel(n + i * nvel) = nQ1 / distri_i * mom_iscale;
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::ComputeError
//  \brief
KOKKOS_INLINE_FUNCTION
Real ComputeError(parthenon::team_mbr_t const &mbr, const int &mimax, const int &mimax2,
                  const Real &h, const Real &h0, const ScratchPad1D<Real> &dustdens,
                  const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &nQs,
                  const Real &err_eps) {
  Real errmax = 0.0;
  parthenon::par_reduce_inner(
      parthenon::inner_loop_pattern_ttr_tag, mbr, 0, Kokkos::min(mimax, mimax2) - 1,
      [&](const int i, Real &lmax) {
        const Real dscale = Kokkos::abs(dustdens(i)) + Kokkos::abs(h0 * source(i));
        lmax = Kokkos::max(lmax, Kokkos::abs(0.5 * h * (nQs(i) - source(i)) / dscale));
      },
      Kokkos::Max<Real>(errmax));
  return errmax / err_eps;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::Source
//  \brief Source driver
KOKKOS_INLINE_FUNCTION
void Source(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
            const int &pgrid, const ScratchPad1D<Real> &source,
            const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &vel,
            const ScratchPad1D<Real> &stime, const StateParams &kernel,
            const ParArray2D<int> &idx_largest, const ParArray1D<Real> &mass_grid,
            const ScratchPad1D<Real> &As, const ParArray3D<Real> &coagR3D,
            const ParArray3D<int> &Kijk_sym_ind, const ParArray3D<Real> &Kijk_sym,
            const bool &surface, const RateParams &rate, const ScratchPad1D<Real> &fett_t,
            const ScratchPad1D<Real> &fett_l) {
  ZeroSource(mbr, nm1, source);
  InitializeSource(mbr, nm1, mimax, source, dustdens, vel, stime, kernel, mass_grid,
                   coagR3D, Kijk_sym_ind, Kijk_sym, surface, rate, fett_t);
  FragmentationSource(mbr, nm1, mimax, pgrid, source, dustdens, vel, stime, kernel,
                      mass_grid, As, coagR3D, surface, rate, fett_l);
  CrateringSource(mbr, nm1, mimax, source, dustdens, vel, stime, kernel, mass_grid,
                  coagR3D, surface, rate, fett_l);
  FinalizeSource(mbr, nm1, mimax, pgrid, source, dustdens, vel, stime, kernel, mass_grid,
                 coagR3D, surface, rate, fett_l);
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::SourceNQ
//  \brief Momentum source driver
KOKKOS_INLINE_FUNCTION
void SourceNQ(const parthenon::team_mbr_t &mbr, const int &n, const int &nm1,
              const int &mimax, const int &pgrid, const ScratchPad1D<Real> &Q,
              const ScratchPad1D<Real> &nQs, const ScratchPad1D<Real> &dustdens,
              const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
              const StateParams &kernel, const ParArray1D<Real> &mass_grid,
              const ScratchPad1D<Real> &As, const ParArray3D<Real> &coagR3D,
              const ParArray3D<int> &Kijk_sym_ind, const ParArray3D<Real> &Kijk_sym,
              const Real &chi, const bool &surface, const RateParams &rate,
              const ScratchPad1D<Real> &fett_t, const ScratchPad1D<Real> &fett_l) {
  InitializeSourceNQ(mbr, nm1, mimax, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
                     coagR3D, Kijk_sym_ind, Kijk_sym, surface, rate, fett_t);
  FragmentationSourceNQ(mbr, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel,
                        mass_grid, As, coagR3D, chi, surface, rate, fett_l);
  CrateringSourceNQ(mbr, nm1, mimax, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
                    coagR3D, surface, rate, fett_l);
  FinalizeSourceNQ(mbr, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel,
                   mass_grid, coagR3D, surface, rate, fett_l);
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::SourceNQS3
//  \brief
KOKKOS_INLINE_FUNCTION
void SourceNQS3(const parthenon::team_mbr_t &mbr, const int &n, const int &nm1,
                const int &mimax, const int &mimax2, const int &pgrid,
                const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &Q2,
                const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &nQs,
                const ScratchPad1D<Real> &dustdens, const ScratchPad1D<Real> &vel,
                const ScratchPad1D<Real> &stime, const StateParams &kernel,
                const ParArray1D<Real> &mass_grid, const ScratchPad1D<Real> &As,
                const ParArray3D<Real> &coagR3D, const ParArray3D<int> &Kijk_sym_ind,
                const ParArray3D<Real> &Kijk_sym, const Real &chi, const bool &surface,
                const RateParams &rate, const Real &dt, const ScratchPad1D<Real> &fett_t,
                const ScratchPad1D<Real> &fett_l) {
  SourceNQ(mbr, n, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel, mass_grid, As,
           coagR3D, Kijk_sym_ind, Kijk_sym, chi, surface, rate, fett_t, fett_l);
  IntermediateNQS3(mbr, nm1, Q, nQs, dustdens, source, dt);
  SourceNQ(mbr, n, nm1, mimax2, pgrid, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
           As, coagR3D, Kijk_sym_ind, Kijk_sym, chi, surface, rate, fett_t, fett_l);
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CoagulationOneCell
//  \brief
KOKKOS_INLINE_FUNCTION
int CoagulationOneCell(parthenon::team_mbr_t const &mbr, const bool &surface,
                       const Real &time, Real &dt_sync, const StateParams &kernel,
                       const ScratchPad1D<Real> &dustdens,
                       const ScratchPad1D<Real> &stime, const ScratchPad1D<Real> &vel,
                       const int &nvel, const ScratchPad1D<Real> &Q,
                       const ScratchPad1D<Real> &nQs, const CoagParams &coag,
                       const ScratchPad1D<Real> &As, const CoagArrays &coag_arrays,
                       const RateParams &rate, const ScratchPad1D<Real> &source,
                       const ScratchPad1D<Real> &Q2, ScratchPad1D<Real> &fett_t,
                       ScratchPad1D<Real> &fett_l, const int ix) {
  // Params
  const int nm1 = coag.nm - 1;
  const Real &cfl = coag.cfl;
  const int &pgrid = coag.pgrid;
  const Real &dfloor = coag.dfloor;
  const Real &chi = coag.chi;
  const bool &do_momentum_conserving_update = coag.mom_coag;
  const bool &do_adaptive = coag.use_adaptive;
  const int &ncall_max = coag.ncall_max;

  // Higher order params
  const int &coag_int = coag.integrator;
  const Real &err_eps = coag.err_eps;
  const Real &err_con = coag.err_con;
  const Real &S = coag.S;
  const Real &pshrink = coag.pshrink;
  const Real &pgrow = coag.pgrow;

  // Arrays
  auto &idx_largest = coag_arrays.idx_largest;
  auto &mass_grid = coag_arrays.mass_grid;
  auto &coagR3D = coag_arrays.coagR3D;
  auto &Kijk_sym_ind = coag_arrays.Kijk_sym_ind;
  auto &Kijk_sym = coag_arrays.Kijk_sym;

  // Timestepping
  int ncall = 0;
  Real time_dummy = time;
  Real dt_sync1 = dt_sync;
  Real dt = dt_sync1;
  Real hnext = dt;
  dt_sync = 1e-15;
  const Real time_goal = time_dummy + dt_sync1;

  // calculate the rate via kernel first
  Calculate_Kernels(mbr, kernel, vel, dustdens, stime, nm1, mass_grid, coagR3D, surface,
                    rate, fett_t, fett_l);

  // Update distribution
  ConvertToNumberDensity(mbr, nm1, dustdens, mass_grid, dfloor);

  while (Kokkos::abs(time_dummy - time_goal) > 1e-6 * dt) {
    // Set source
    const int mimax = FindMIMax(mbr, nm1, dustdens, mass_grid, dfloor);
    Source(mbr, nm1, mimax, pgrid, source, dustdens, vel, stime, kernel, idx_largest,
           mass_grid, As, coagR3D, Kijk_sym_ind, Kijk_sym, surface, rate, fett_t, fett_l);

    int mimax2 = Null<int>();
    if (!(do_adaptive) || (coag_int == 1)) {
      dt_sync1 = TimeStepControl(mbr, nm1, dustdens, source, mass_grid, dfloor, cfl);
      dt = Kokkos::min(dt_sync1, time_goal - time_dummy);
      dt_sync = dt_sync1;
      if (coag_int == 3) {
        mimax2 = FindMIMaxNQS3(mbr, nm1, dt, dustdens, source, Q, mass_grid, dfloor);
        // now Q stores dustdens + dt*source(), nQs will be used for temperary source(*)
        ZeroSource(mbr, nm1, nQs);
        Source(mbr, nm1, mimax2, pgrid, nQs, Q, vel, stime, kernel, idx_largest,
               mass_grid, As, coagR3D, Kijk_sym_ind, Kijk_sym, surface, rate, fett_t,
               fett_l);
      }
    } else { // adaptive third-order method
      // Set source
      Real h0 = hnext, h = h0;
      Real emax = Null<Real>();
      while (1) {
        mimax2 = FindMIMaxNQS3(mbr, nm1, h, dustdens, source, Q, mass_grid, dfloor);
        // now Q stores dustdens + dt*source(), nQs will be used for temperary source(*)
        ZeroSource(mbr, nm1, nQs);
        Source(mbr, nm1, mimax2, pgrid, nQs, Q, vel, stime, kernel, idx_largest,
               mass_grid, As, coagR3D, Kijk_sym_ind, Kijk_sym, surface, rate, fett_t,
               fett_l);
        emax = ComputeError(mbr, mimax, mimax2, h, h0, dustdens, source, nQs, err_eps);
        if (emax <= 1.0) break;
        h = Kokkos::max(S * h * Kokkos::pow(emax, pshrink), 0.1 * h);
      }
      hnext = (emax > err_con) ? S * h * Kokkos::pow(emax, pgrow) : 5.0 * h;
      dt = h;
    }

    if (coag_int == 1) {
      // Momentum Conserving Update (iff do_momentum_conserving_update)
      for (int n = 0; n < do_momentum_conserving_update * nvel; n++) {
        ZeroSourceNQ(mbr, n, nm1, Q, nQs, vel, kernel.nvel);
        SourceNQ(mbr, n, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel,
                 mass_grid, As, coagR3D, Kijk_sym_ind, Kijk_sym, chi, surface, rate,
                 fett_t, fett_l);
        UpdateVelocityNQ(mbr, n, nm1, Q, nQs, vel, kernel.nvel, dustdens, source,
                         mass_grid, dt, dfloor);
      }

      // Update dust density
      UpdateDensity(mbr, nm1, dustdens, source, dt);

    } else { // third-order method

      // Momentum Conserving Update (iff do_momentum_conserving_update)
      for (int n = 0; n < do_momentum_conserving_update * nvel; n++) {
        if (n == 0) InitializeTempNQS3(mbr, n, nm1, Q2, nQs);
        ZeroSourceNQ(mbr, n, nm1, Q, nQs, vel, kernel.nvel);
        SourceNQS3(mbr, n, nm1, mimax, mimax2, pgrid, Q, Q2, source, nQs, dustdens, vel,
                   stime, kernel, mass_grid, As, coagR3D, Kijk_sym_ind, Kijk_sym, chi,
                   surface, rate, dt, fett_t, fett_l);
        UpdateVelocityNQS3(mbr, n, nm1, Q2, nQs, vel, kernel.nvel, dustdens, source,
                           mass_grid, dt, dfloor);
      }

      // Update dust density
      UpdateDensityNQS3(mbr, nm1, dustdens, source,
                        (do_momentum_conserving_update ? Q2 : nQs), dt);
    }

    // Update time and increment ncall: debug info
    // if (ix==2) printf("coagdt: %d %g %g %g\n", ncall, time_dummy, dt, time_goal);
    time_dummy += dt;
    ncall++;

    dt_sync = (do_adaptive) ? Kokkos::max(hnext, dt_sync) : dt_sync;
    hnext = (do_adaptive) ? Kokkos::min(hnext, time_goal - time_dummy) : hnext;

    // Warn and break upon reaching ncall_max
    if (ncall > ncall_max) {
      printf("(Coagulation): Reach ncall_max in coagulation kernel! i=%d\n", ix);
      break;
    }

    // calling the kernels for subcycling
    if (time_goal <= time_dummy + 1e-6 * dt && do_momentum_conserving_update) {
      Calculate_Kernels(mbr, kernel, vel, dustdens, stime, nm1, mass_grid, coagR3D,
                        surface, rate, fett_t, fett_l);
    }
  }
  ConvertToVolumeDensity(mbr, nm1, dustdens, mass_grid, dfloor);
  return ncall;
} // end of CoagulationOneCell

} // namespace Coagulation
} // namespace Dust

#endif // DUST_COAGULATION_COAGULATION_HPP_
