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

#include <Kokkos_Core.hpp>

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"

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
                     ParArray3D<Real> Kijk_sym, ParArray1D<Real> log_centers,
                     ParArray1D<Real> log_widths);

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
enum cidx { Djk, Aij, Pij, epsij, dalp, rate_coef };
enum class DustInteractionType { Coagulation, Fragmentation };

// Within-bin reconstruction model used by coagulation (and Stokes drag when wired).
//   Constant     -- each bin is a delta function at its representative size (default).
//   PLMLogSize   -- piecewise linear in ln(a), slope-limited from neighbors; evolved
//                   state is unchanged, only microphysical coefficients differ.
enum class BinRecon { Constant, PLMLogSize };

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

  // Backward-Euler implicit step controls (Part 1).
  // When `implicit` is true, CoagulationOneCell takes a single implicit step
  // for the dust number-density evolution over the coagulation dt instead of
  // the explicit (possibly adaptive) substepping driven by `integrator` /
  // `use_adaptive`. Velocities are advanced using the existing
  // momentum-conserving update at the accepted implicit state.
  bool implicit;       // use backward-Euler instead of explicit RK
  int newton_max_iter; // max Newton iterations per cell
  Real newton_tol;     // relative tolerance on Newton residual
  Real newton_fd_eps;  // relative perturbation for FD Jacobian (unused with analytic J)
  bool newton_verbose; // print per-iteration Newton diagnostics
  int newton_jac_lag; // reuse factored Jacobian for this many Newton iters before rebuild

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

  // Within-bin size reconstruction (Phase 2 opt-in; see BinRecon).
  // Default is BinRecon::Constant which preserves existing behavior exactly.
  BinRecon bin_recon;
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

  // Within-bin size geometry. Populated unconditionally from the (log-spaced)
  // dust sizes so they are always available; only consumed when bin_recon !=
  // Constant. log_centers(i) = ln(a_i); log_widths(i) is the width of bin i in
  // ln(a) measured between geometric-mean edges to neighboring bins.
  ParArray1D<Real> log_centers;
  ParArray1D<Real> log_widths;
  // Physical dust size grid (already multiplied by length0); shared with the
  // coagulation initialisation. Needed by within-bin reconstruction to convert
  // moment ratios back to absolute sizes when correcting collision cross sections.
  ParArray1D<Real> dsize;
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

  const Real sqRe = 1.0 / std::sqrt(reynolds);
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
        std::max(1.0, c[0] + (c[1] + (c[2] + (c[3] * st1) * st1) * st1) * st1);
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
//! \fn  Real Dust::Coagulation::BinSlopeMCLimited
//  \brief Monotonized-central limited slope of the mass-per-log-size density rho_log
//         across one size bin, given the bin-averaged densities of bin i and its two
//         neighbors. Returns the slope sigma_i scaled so the within-bin profile is
//             f_i(xi) = rho_avg_i + sigma_i * xi,    xi = (x - x_i)/dx_i in [-1/2, 1/2].
//         Includes a positivity cap |sigma| <= 2 rho_avg so f_i >= 0 at bin edges.
//         Returns 0 when neighbor differences disagree in sign (extremum) or when
//         rho_avg_i is non-positive (empty bin).
KOKKOS_FORCEINLINE_FUNCTION
Real BinSlopeMCLimited(const Real rho_im1, const Real rho_i, const Real rho_ip1) {
  if (!(rho_i > 0.0)) return 0.0;
  const Real dL = rho_i - rho_im1;
  const Real dR = rho_ip1 - rho_i;
  if (dL * dR <= 0.0) return 0.0;
  const Real s = (dL > 0.0) ? 1.0 : -1.0;
  const Real adL = (dL > 0.0) ? dL : -dL;
  const Real adR = (dR > 0.0) ? dR : -dR;
  const Real adC = 0.5 * (adL + adR); // |0.5*(rho_ip1 - rho_im1)| with consistent sign
  Real m = 2.0 * adL;
  if (2.0 * adR < m) m = 2.0 * adR;
  if (adC < m) m = adC;
  // Positivity cap: keep f_i >= 0 at bin edges.
  if (2.0 * rho_i < m) m = 2.0 * rho_i;
  return s * m;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::BinMomentLogA
//  \brief Bin-averaged moment <a^p> for a piecewise-linear-in-log-size profile of the
//         mass-per-log-size density f_i(xi) = rho_avg + sigma * xi over bin i, with
//         xi = (x - x_i)/dx_i in [-1/2, 1/2] and x = ln(a). Mass per unit log a is the
//         conserved quantity so the bin-mass weighted moment evaluates to
//             <a^p> = a_i^p * ( I0(alpha) + (sigma/rho_avg) * I1(alpha) ),
//         where alpha = p * dx_i, I0(alpha) = (2/alpha) sinh(alpha/2), and
//         I1(alpha) = dI0/dalpha. The constant (sigma=0) branch reproduces the
//         standard log-bin moment correction. Stable small-alpha series fallback is
//         used to avoid 0/0 near alpha = 0.
KOKKOS_FORCEINLINE_FUNCTION
Real BinMomentLogA(const Real ai, const Real dx, const Real rho_avg, const Real sigma,
                   const Real p) {
  const Real alpha = p * dx;
  const Real abs_a = (alpha >= 0.0) ? alpha : -alpha;
  Real I0, I1;
  if (abs_a < 1.0e-3) {
    const Real a2 = alpha * alpha;
    // I0 ≈ 1 + α^2/24 + α^4/1920
    I0 = 1.0 + a2 * (1.0 / 24.0) + a2 * a2 * (1.0 / 1920.0);
    // I1 ≈ α/12 + α^3/480
    I1 = alpha * (1.0 / 12.0) + alpha * a2 * (1.0 / 480.0);
  } else {
    const Real h = 0.5 * alpha;
    const Real sh = std::sinh(h);
    const Real ch = std::cosh(h);
    I0 = (2.0 / alpha) * sh;
    // I1 = (cosh(α/2) - 2 sinh(α/2)/α) / α
    I1 = (ch - 2.0 * sh / alpha) / alpha;
  }
  const Real ai_p = std::pow(ai, p);
  const Real ratio = (rho_avg > 0.0) ? (sigma / rho_avg) : 0.0;
  return ai_p * (I0 + ratio * I1);
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::ComputeBinSizeMomentRatios
//  \brief Fill per-bin moment ratios mPr(i) = <a^P>_i / a_i^P (P=1,2,3) from a
//         piecewise-linear-in-log-size reconstruction of the dust mass density.
//         Reconstruction is built from the cell-local bin-averaged mass density,
//         which is dustdens(i)*mass_grid(i) (i.e. the mass-per-bin), with an
//         MC-limited slope across neighbors. Only bins 0..mimax are touched; the
//         caller is expected to size m1r/m2r/m3r at least nm long.
//             m1r weights stopping times (Epstein: tau ~ a),
//             m2r weights cross sections (sigma ~ a^2),
//             m3r weights particle masses (m ~ a^3).
KOKKOS_INLINE_FUNCTION
void ComputeBinSizeMomentRatios(const parthenon::team_mbr_t &mbr, const int &nm1,
                                const int &mimax, const ScratchPad1D<Real> &dustdens,
                                const ParArray1D<Real> &mass_grid,
                                const ParArray1D<Real> &log_widths,
                                const ScratchPad1D<Real> &m1r,
                                const ScratchPad1D<Real> &m2r,
                                const ScratchPad1D<Real> &m3r) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    m1r(i) = 1.0;
    m2r(i) = 1.0;
    m3r(i) = 1.0;
  });
  mbr.team_barrier();
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    const Real rho_i = dustdens(i) * mass_grid(i);
    const int im1 = (i == 0) ? 0 : i - 1;
    const int ip1 = (i == nm1) ? nm1 : i + 1;
    const Real rho_im1 = dustdens(im1) * mass_grid(im1);
    const Real rho_ip1 = dustdens(ip1) * mass_grid(ip1);
    const Real sigma = BinSlopeMCLimited(rho_im1, rho_i, rho_ip1);
    const Real dx = log_widths(i);
    if (!(rho_i > 0.0) || dx <= 0.0) {
      m1r(i) = 1.0;
      m2r(i) = 1.0;
      m3r(i) = 1.0;
      return;
    }
    // <a^p>/a_i^p = I0(p*dx) + (sigma/rho_i) * I1(p*dx). a_i cancels.
    // ai = 1.0 sentinel: BinMomentLogA(1, dx, rho, sigma, p) returns the ratio.
    m1r(i) = BinMomentLogA(1.0, dx, rho_i, sigma, 1.0);
    m2r(i) = BinMomentLogA(1.0, dx, rho_i, sigma, 2.0);
    m3r(i) = BinMomentLogA(1.0, dx, rho_i, sigma, 3.0);
    // Defensive floors: keep ratios positive in case of pathological inputs.
    if (!(m1r(i) > 0.0)) m1r(i) = 1.0;
    if (!(m2r(i) > 0.0)) m2r(i) = 1.0;
    if (!(m3r(i) > 0.0)) m3r(i) = 1.0;
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::CoagulationRatePair
//  \brief Scalar-input core of CoagulationRate. Computes R_{ij} for a pair of bins
//         given explicit per-bin scalar values for the mean particle mass and
//         stopping time. The wrapper CoagulationRate() below preserves the legacy
//         point-bin behavior; the PLM-corrected path in BuildRateCache calls this
//         directly with bin-averaged effective masses (m_i = mass_grid(i)*m3r(i))
//         and stopping times (tau_i = stime(i)*m1r(i)) implied by the within-bin
//         piecewise linear profile. mass_gride is the upper-bin mass used only for
//         the structural "merged particle exceeds grid" cutoff and is left as the
//         constant-bin value mass_grid(nm-1) regardless of mode.
template <DustInteractionType DIT>
KOKKOS_INLINE_FUNCTION Real CoagulationRatePair(
    const int &i, const int &j, const Real &mass_gridi, const Real &mass_gridj,
    const Real &mass_gride, const StateParams &kernel, const ScratchPad1D<Real> &vel,
    const Real &tau_i, const Real &tau_j, const ParArray3D<Real> &coagR3D,
    const bool &surface, const RateParams &rate) {
  if (mass_gridi + mass_gridj >= mass_gride) return 0.0;

  const Real &gdens = kernel.gdens;
  const Real &alpha = kernel.alpha;
  const Real &cs = kernel.cs;
  const Real &omega = kernel.omega;
  const int &nvel = kernel.nvel;
  const Real *vel_i = &vel(nvel * i);
  const Real *vel_j = &vel(nvel * j);

  const Real &sig = rate.cross_section; //! cross section of gas species
  const Real &mmw = rate.mmw;           //! mean molecular weight (mu * mp)

  // Calculate some basic properties
  const Real hg = cs / omega;
  Real re = alpha * sig * gdens / (2.0 * mmw);
  if (!(surface)) re *= std::sqrt(2.0 * M_PI) * hg;

  const Real tn = 1.0 / omega;
  const Real ts = tn / std::sqrt(re);
  const Real vn = std::sqrt(alpha) * cs;
  const Real vs = vn * std::pow(re, -0.25);

  // Calculate Stokes number
  const Real stokes_i = tau_i * omega;
  const Real stokes_j = tau_j * omega;

  const Real muij = (mass_gridi * mass_gridj / (mass_gridi + mass_gridj));
  // Calculate turbulent relative velocity
  // turbulent + brownian + actual
  Real dv = GetRelativeTurbulentVelocity(tau_i, tau_j, tn, vn, ts, vs, re) +
            std::min(cs * cs, 8 / M_PI * kernel.kT / muij) + SQR(vel_i[0] - vel_j[0]) +
            SQR(vel_i[1] - vel_j[1]);
  if (surface) {
    dv += SQR(vel_i[2] - vel_j[2]);
  }
  Real hij = 1.0;
  if (surface) { // surface density
    const Real hi =
        std::min(std::sqrt(alpha / (std::min(0.5, stokes_i) * (SQR(stokes_i) + 1.0))),
                 1.0) *
        hg;
    const Real hj =
        std::min(std::sqrt(alpha / (std::min(0.5, stokes_j) * (SQR(stokes_j) + 1.0))),
                 1.0) *
        hg;

    // NOTE(AMD): why not 0.5 in the min following Eq. 55?
    const Real vs_i = std::min(stokes_i, 1.0) * omega * hi;
    const Real vs_j = std::min(stokes_j, 1.0) * omega * hj;
    // relative velocity from vertical settling
    dv += SQR(vs_i - vs_j);
    hij = std::sqrt(2.0 * M_PI * (SQR(hi) + SQR(hj)));
  }

  dv = std::sqrt(dv);

  // New pf calculation: for fragmenation
  Real pf = 0.0;
  if (dv > 0.0) {
    // Eq. 60 of SB22
    const Real tmp = 1.5 * SQR(rate.vfrag / dv);
    pf = (tmp + 1.0) * std::exp(-tmp);
  }

  if constexpr (DIT == DustInteractionType::Coagulation) {
    if (rate.ibounce && dv > 0.0) { // including bouncing effect
      // NOTE(AMD): these should be input params
      const Real froll = 1e-4; // Heim et al.(PRL) 1999
      const Real amono = 1e-4; // micro-size
      const Real vbounce = std::sqrt(5.0 * M_PI * amono * froll / muij);
      if (vbounce < rate.vfrag) {
        const Real tmp = 1.5 * SQR(vbounce / dv);
        pf = (tmp + 1.0) * std::exp(-tmp); // using bouncing vel
      }
    }
    pf = 1.0 - pf;
  }

  return coagR3D(cidx::rate_coef, i, j) * dv * pf / hij;
}

//----------------------------------------------------------------------------------------
//! \fn  Real Dust::Coagulation::CoagulationRate
//  \brief Calculate Rij (legacy wrapper; uses bin-center mass/stopping-time values).
template <DustInteractionType DIT>
KOKKOS_INLINE_FUNCTION Real CoagulationRate(const int &i, const int &j, const int &nm1,
                                            const StateParams &kernel,
                                            const ScratchPad1D<Real> &vel,
                                            const ScratchPad1D<Real> &stime,
                                            const ParArray1D<Real> &mass_grid,
                                            const ParArray3D<Real> &coagR3D,
                                            const bool &surface, const RateParams &rate) {
  return CoagulationRatePair<DIT>(i, j, mass_grid(i), mass_grid(j), mass_grid(nm1),
                                  kernel, vel, stime(i), stime(j), coagR3D, surface,
                                  rate);
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
    dustdens(i) = std::max(dustdens(i), 0.01 * dfloor / mass_gridi);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::ConvertToVolumeDensity
//  \brief convert to volume density
KOKKOS_INLINE_FUNCTION
void ConvertToVolumeDensity(const parthenon::team_mbr_t &mbr, const int &nm1,
                            const ScratchPad1D<Real> &dustdens,
                            const ParArray1D<Real> &mass_grid) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { dustdens(i) *= mass_grid(i); });
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
          lmin = std::min(lmin, std::abs(dustdens(i) / source(i)));
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
                      const RateParams &rate_par) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // Calculate the Rate
      const Real rate = CoagulationRate<DustInteractionType::Coagulation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
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
                         const int &mimax, const ScratchPad1D<Real> &source,
                         const ScratchPad1D<Real> &dustdens,
                         const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
                         const StateParams &kernel, const ParArray2D<int> &idx_largest,
                         const ParArray1D<Real> &mass_grid,
                         const ParArray3D<Real> &coagR3D, const bool &surface,
                         const RateParams &rate_par) {
  // Adding fragment distribution
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int k) {
    for (int j = k; j <= mimax; j++) {
      // Calculate As(j) on fly
      Real val = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          if (idx_largest(i2, j2) == j) {
            const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
                i2, j2, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
            val += coagR3D(cidx::Aij, i2, j2) * dustdens(i2) * dustdens(j2) * rate;
          }
        }
      }
      source(k) += coagR3D(cidx::Pij, k, j) / mass_grid(k) * val;
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
                     const RateParams &rate_par) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
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
                    const bool &surface, const RateParams &rate_par) {
  // Cratering
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0 = 0.0;
    for (int j = 0; j <= i - pgrid - 1; j++) {
      const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
      const Real val = dustdens(i) * dustdens(j) * rate;
      sum0 += coagR3D(cidx::epsij, i, j) * val;
    }

    if (i - pgrid - 1 >= 0) {
      Kokkos::atomic_add(&source(i - 1), sum0);
    }

    sum0 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i; j++) {
      const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
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
                        const RateParams &rate_par) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      // calculate the rate
      const Real rate = CoagulationRate<DustInteractionType::Coagulation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
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
                           const ParArray3D<Real> &coagR3D, const Real &chi,
                           const bool &surface, const RateParams &rate_par) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int k) {
    for (int j = k; j <= mimax; j++) {
      // calculate As(j) on fly
      Real val = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++) {
        for (int j2 = 0; j2 <= i2; j2++) {
          int idx_largest = (j2 <= i2 - pgrid - 1) ? j2 : i2;
          if (idx_largest == j) {
            const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
                i2, j2, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
            const Real &mass_gridi2 = mass_grid(i2);
            const Real &mass_gridj2 = mass_grid(j2);
            const Real &Ai2j2 = coagR3D(cidx::Aij, i2, j2);
            const Real Qf1 = (j2 <= i2 - pgrid - 1)
                                 ? Qplus(chi * mass_gridj2, Q(i2), mass_gridj2, Q(j2))
                                 : Qplus(mass_gridi2, Q(i2), mass_gridj2, Q(j2));
            val += Ai2j2 * dustdens(i2) * dustdens(j2) * rate * Qf1;
          }
        }
      }
      nQs(k) += coagR3D(cidx::Pij, k, j) / mass_grid(k) * val;
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
                       const bool &surface, const RateParams &rate_par) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    // Cratering
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++) {
      const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
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
                      const RateParams &rate_par) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0 = 0.0;
    for (int j = 0; j <= i - pgrid - 1; j++) {
      const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
      const Real Rf1 = dustdens(i) * dustdens(j) * rate;
      sum0 += coagR3D(cidx::epsij, i, j) * Rf1 * Q(i);
    }

    if (i - pgrid - 1 >= 0) {
      Kokkos::atomic_add(&nQs(i - 1), sum0);
    }

    sum0 = -sum0;
    // Full fragmentation (only negative terms)
    int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i; j++) {
      const Real rate = CoagulationRate<DustInteractionType::Fragmentation>(
          i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate_par);
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
        if (dustdens(i) > dfloor / mass_grid(i)) lmax = std::max(lmax, i);
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
        if (Q(i) > dfloor / mass_grid(i)) lmax = std::max(lmax, i);
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
      parthenon::inner_loop_pattern_ttr_tag, mbr, 0, std::min(mimax, mimax2) - 1,
      [&](const int i, Real &lmax) {
        const Real dscale = std::abs(dustdens(i)) + std::abs(h0 * source(i));
        lmax = std::max(lmax, std::abs(0.5 * h * (nQs(i) - source(i)) / dscale));
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
            const ParArray3D<Real> &coagR3D, const ParArray3D<int> &Kijk_sym_ind,
            const ParArray3D<Real> &Kijk_sym, const bool &surface,
            const RateParams &rate) {
  ZeroSource(mbr, nm1, source);
  InitializeSource(mbr, nm1, mimax, source, dustdens, vel, stime, kernel, mass_grid,
                   coagR3D, Kijk_sym_ind, Kijk_sym, surface, rate);
  FragmentationSource(mbr, nm1, mimax, source, dustdens, vel, stime, kernel, idx_largest,
                      mass_grid, coagR3D, surface, rate);
  CrateringSource(mbr, nm1, mimax, source, dustdens, vel, stime, kernel, mass_grid,
                  coagR3D, surface, rate);
  FinalizeSource(mbr, nm1, mimax, pgrid, source, dustdens, vel, stime, kernel, mass_grid,
                 coagR3D, surface, rate);
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
              const ParArray3D<Real> &coagR3D, const ParArray3D<int> &Kijk_sym_ind,
              const ParArray3D<Real> &Kijk_sym, const Real &chi, const bool &surface,
              const RateParams &rate) {
  InitializeSourceNQ(mbr, nm1, mimax, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
                     coagR3D, Kijk_sym_ind, Kijk_sym, surface, rate);
  FragmentationSourceNQ(mbr, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel,
                        mass_grid, coagR3D, chi, surface, rate);
  CrateringSourceNQ(mbr, nm1, mimax, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
                    coagR3D, surface, rate);
  FinalizeSourceNQ(mbr, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel,
                   mass_grid, coagR3D, surface, rate);
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
                const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                const ParArray3D<int> &Kijk_sym_ind, const ParArray3D<Real> &Kijk_sym,
                const Real &chi, const bool &surface, const RateParams &rate,
                const Real &dt) {
  SourceNQ(mbr, n, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
           coagR3D, Kijk_sym_ind, Kijk_sym, chi, surface, rate);
  IntermediateNQS3(mbr, nm1, Q, nQs, dustdens, source, dt);
  SourceNQ(mbr, n, nm1, mimax2, pgrid, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
           coagR3D, Kijk_sym_ind, Kijk_sym, chi, surface, rate);
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::BuildRateCache
//  \brief Precompute collision rates for all active pairs (i,j) with i<=mimax.
//         Rates are independent of number density so they need only be computed once
//         per cell per implicit solve (they depend only on vel, stime, mass_grid, etc.).
//         rcoag(i,j) = R^coag_{ij},  rfrag(i,j) = R^frag_{ij}, both stored symmetrically.
//
//         When bin_recon == BinRecon::PLMLogSize, all size-dependent kinematic
//         coefficients are reconstructed from the within-bin PL profile:
//             - mean particle mass:      m_i    -> mass_grid(i) * m3r(i)  (m ~ a^3),
//                                                  affecting muij in the Brownian
//                                                  velocity and bouncing threshold,
//             - mean stopping time:      tau_i  -> stime(i)     * m1r(i)  (Epstein,
//                                                  tau ~ a), affecting turbulent
//                                                  v_rel and vertical settling,
//             - geometric cross section: pi (a_i+a_j)^2 * <(a_i+a_j)^2>/(a_i+a_j)^2
//                                        with the moment expanded as
//                                        m2r_i a_i^2 + 2 m1r_i m1r_j a_i a_j +
//                                        m2r_j a_j^2.
//         The "merged particle exceeds grid" cutoff uses the constant mass_grid(nm-1)
//         regardless of mode. Mass-redistribution outcome weights (Aij, Pij, epsij,
//         Djk in coagR3D) are *not* PLM-corrected: they are exact mass-bookkeeping
//         coefficients and perturbing them would break conservation of total dust
//         mass per cell.
template <typename RateView2D>
KOKKOS_INLINE_FUNCTION void
BuildRateCache(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
               const ScratchPad1D<Real> &vel, const ScratchPad1D<Real> &stime,
               const StateParams &kernel, const ParArray1D<Real> &mass_grid,
               const ParArray3D<Real> &coagR3D, const bool &surface,
               const RateParams &rate, const RateView2D &rcoag, const RateView2D &rfrag,
               const BinRecon bin_recon, const ParArray1D<Real> &dsize,
               const ScratchPad1D<Real> &m1r, const ScratchPad1D<Real> &m2r,
               const ScratchPad1D<Real> &m3r) {
  const bool plm = (bin_recon == BinRecon::PLMLogSize);
  const Real mass_gride = mass_grid(nm1);
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      Real rc, rf;
      if (plm) {
        const Real mi_eff = mass_grid(i) * m3r(i);
        const Real mj_eff = mass_grid(j) * m3r(j);
        const Real tau_i_eff = stime(i) * m1r(i);
        const Real tau_j_eff = stime(j) * m1r(j);
        rc = CoagulationRatePair<DustInteractionType::Coagulation>(
            i, j, mi_eff, mj_eff, mass_gride, kernel, vel, tau_i_eff, tau_j_eff, coagR3D,
            surface, rate);
        rf = CoagulationRatePair<DustInteractionType::Fragmentation>(
            i, j, mi_eff, mj_eff, mass_gride, kernel, vel, tau_i_eff, tau_j_eff, coagR3D,
            surface, rate);
        // Cross-section moment correction.
        const Real ai = dsize(i);
        const Real aj = dsize(j);
        const Real denom = (ai + aj) * (ai + aj);
        if (denom > 0.0) {
          const Real num =
              m2r(i) * ai * ai + 2.0 * m1r(i) * ai * m1r(j) * aj + m2r(j) * aj * aj;
          const Real factor = num / denom;
          rc *= factor;
          rf *= factor;
        }
      } else {
        rc = CoagulationRate<DustInteractionType::Coagulation>(
            i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate);
        rf = CoagulationRate<DustInteractionType::Fragmentation>(
            i, j, nm1, kernel, vel, stime, mass_grid, coagR3D, surface, rate);
      }
      rcoag(i, j) = rc;
      rcoag(j, i) = rc;
      rfrag(i, j) = rf;
      rfrag(j, i) = rf;
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::SourceFromCache
//  \brief Evaluate the coagulation+fragmentation source using precomputed rate arrays.
//         Replaces Source() inside the Newton loop; does not recompute CoagulationRate.
template <typename RateView2D>
KOKKOS_INLINE_FUNCTION void
SourceFromCache(const parthenon::team_mbr_t &mbr, const int &nm1, const int &mimax,
                const int &pgrid, const ScratchPad1D<Real> &source,
                const ScratchPad1D<Real> &dustdens, const ParArray2D<int> &idx_largest,
                const ParArray1D<Real> &mass_grid, const ParArray3D<Real> &coagR3D,
                const ParArray3D<int> &Kijk_sym_ind, const ParArray3D<Real> &Kijk_sym,
                const RateView2D &rcoag, const RateView2D &rfrag) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { source(i) = 0.0; });
  mbr.team_barrier();
  // Coagulation gain/loss.
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    for (int j = 0; j <= i; j++) {
      const Real val = dustdens(i) * dustdens(j) * rcoag(i, j);
      for (int nz = 0; nz < 4; nz++) {
        const int k = Kijk_sym_ind(i, j, nz);
        if (k >= 0) Kokkos::atomic_add(&source(k), Kijk_sym(i, j, nz) * val);
      }
    }
  });
  mbr.team_barrier();
  // Fragmentation redistribution.
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int k) {
    for (int j = k; j <= mimax; j++) {
      Real val = 0.0;
      for (int i2 = 0; i2 <= mimax; i2++)
        for (int j2 = 0; j2 <= i2; j2++)
          if (idx_largest(i2, j2) == j)
            val +=
                coagR3D(cidx::Aij, i2, j2) * dustdens(i2) * dustdens(j2) * rfrag(i2, j2);
      source(k) += coagR3D(cidx::Pij, k, j) / mass_grid(k) * val;
    }
  });
  mbr.team_barrier();
  // Cratering small partner.
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int j) {
    Real sum0 = 0.0;
    for (int i = j; i <= mimax; i++)
      sum0 -= dustdens(i) * dustdens(j) * rfrag(i, j);
    source(j) += sum0;
  });
  mbr.team_barrier();
  // Cratering large partner + full fragmentation.
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, mimax, [&](const int i) {
    Real sum0 = 0.0;
    for (int j = 0; j <= i - pgrid - 1; j++)
      sum0 += coagR3D(cidx::epsij, i, j) * dustdens(i) * dustdens(j) * rfrag(i, j);
    if (i - pgrid - 1 >= 0) Kokkos::atomic_add(&source(i - 1), sum0);
    sum0 = -sum0;
    const int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i; j++)
      sum0 -= dustdens(i) * dustdens(j) * rfrag(i, j);
    Kokkos::atomic_add(&source(i), sum0);
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::AnalyticJacobian
//  \brief Build J = I - dt * dS/dn analytically for the active block [0,nact_m1],
//         treating all collision rates as frozen at their cached values.
//
//  dS_k/dn_q is derived by differentiating the bilinear products n_i*n_j in each
//  source term. The rate factors (rcoag, rfrag, Kijk, Pij, epsij, Aij) are constants.
template <typename JacView2D, typename RateView2D>
KOKKOS_INLINE_FUNCTION void
AnalyticJacobian(const parthenon::team_mbr_t &mbr, const int &nm1, const int &nact_m1,
                 const int &pgrid, const ScratchPad1D<Real> &dustdens,
                 const ParArray2D<int> &idx_largest, const ParArray1D<Real> &mass_grid,
                 const ParArray3D<Real> &coagR3D, const ParArray3D<int> &Kijk_sym_ind,
                 const ParArray3D<Real> &Kijk_sym, const RateView2D &rcoag,
                 const RateView2D &rfrag, const Real &dt, const JacView2D &jac) {
  // Initialize to identity.
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nact_m1,
                           [&](const int row) {
                             for (int col = 0; col <= nact_m1; col++)
                               jac(row, col) = (row == col) ? Real(1.0) : Real(0.0);
                           });
  mbr.team_barrier();

  // Contribution from coagulation gain/loss (InitializeSource).
  // S_k += Kijk*R^coag_{ij}*n_i*n_j  =>  dS_k/dn_q = Kijk*R^coag_{qj}*n_j (i=q term)
  //                                                  + Kijk*R^coag_{iq}*n_i (j=q term,
  //                                                  i!=j)
  // When i==j the stored val=n_i^2 so d/dn_i = 2*n_i; we account for this by adding
  // the n_j contribution twice (since n_j=n_i in that case).
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nact_m1, [&](const int i) {
    for (int j = 0; j <= i && j <= nact_m1; j++) {
      const Real rij = rcoag(i, j);
      const Real ni = dustdens(i), nj = dustdens(j);
      for (int nz = 0; nz < 4; nz++) {
        const int k = Kijk_sym_ind(i, j, nz);
        if (k < 0 || k > nact_m1) continue;
        const Real Kval = Kijk_sym(i, j, nz) * rij;
        // Perturbing n_i: dS_k/dn_i += Kval * n_j
        Kokkos::atomic_add(&jac(k, i), -dt * Kval * nj);
        // Perturbing n_j (i!=j): dS_k/dn_j += Kval * n_i
        if (i != j)
          Kokkos::atomic_add(&jac(k, j), -dt * Kval * ni);
        else
          // i==j: product is n_i^2, d/dn_i = 2*n_i => add n_j=n_i once more
          Kokkos::atomic_add(&jac(k, i), -dt * Kval * nj);
      }
    }
  });
  mbr.team_barrier();

  // Contribution from FragmentationSource.
  // S_k += Pij(k,j)/m_k * Aij(i2,j2)*rfrag(i2,j2)*n_i2*n_j2  for each (i2,j2) with idx==j
  // => dS_k/dn_i2 += Pij(k,j)/m_k * Aij*rfrag * n_j2
  //    dS_k/dn_j2 += Pij(k,j)/m_k * Aij*rfrag * n_i2  (i2!=j2)
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nact_m1, [&](const int k) {
    for (int j = k; j <= nact_m1; j++) {
      const Real Pkj_mk = coagR3D(cidx::Pij, k, j) / mass_grid(k);
      for (int i2 = 0; i2 <= nact_m1; i2++) {
        for (int j2 = 0; j2 <= i2 && j2 <= nact_m1; j2++) {
          if (idx_largest(i2, j2) != j) continue;
          const Real coeff = Pkj_mk * coagR3D(cidx::Aij, i2, j2) * rfrag(i2, j2);
          Kokkos::atomic_add(&jac(k, i2), -dt * coeff * dustdens(j2));
          if (i2 != j2)
            Kokkos::atomic_add(&jac(k, j2), -dt * coeff * dustdens(i2));
          else
            Kokkos::atomic_add(&jac(k, i2), -dt * coeff * dustdens(j2));
        }
      }
    }
  });
  mbr.team_barrier();

  // Contribution from CrateringSource.
  // S_j += -sum_{i>=j} rfrag(i,j)*n_i*n_j
  // dS_j/dn_i (i!=j): +rfrag(i,j)*n_j   (becomes +dt in J)
  // dS_j/dn_j:        +sum_{i>=j} rfrag(i,j)*n_i
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nact_m1, [&](const int j) {
    Real dSj_dnjj = 0.0;
    for (int i = j; i <= nact_m1; i++) {
      const Real rij = rfrag(i, j);
      if (i != j) Kokkos::atomic_add(&jac(j, i), dt * rij * dustdens(j));
      dSj_dnjj += rij * dustdens(i);
    }
    Kokkos::atomic_add(&jac(j, j), dt * dSj_dnjj);
  });
  mbr.team_barrier();

  // Contribution from FinalizeSource.
  // Sub-term A: S_{i-1} += epsij(i,j)*rfrag(i,j)*n_i*n_j  for j <= i-pgrid-1
  // Sub-term B: S_i     -= sum_{j=i1..i} rfrag(i,j)*n_i*n_j
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nact_m1, [&](const int i) {
    // Sub-term A: deposit into row i-1
    for (int j = 0; j <= i - pgrid - 1 && j <= nact_m1; j++) {
      const int krow = i - 1;
      if (krow < 0 || krow > nact_m1) continue;
      const Real coeff = coagR3D(cidx::epsij, i, j) * rfrag(i, j);
      Kokkos::atomic_add(&jac(krow, i), -dt * coeff * dustdens(j));
      Kokkos::atomic_add(&jac(krow, j), -dt * coeff * dustdens(i));
    }
    // Sub-term B: deposit into row i (negative source terms)
    const int i1 = std::max(0, i - pgrid);
    for (int j = i1; j <= i && j <= nact_m1; j++) {
      const Real rij = rfrag(i, j);
      Kokkos::atomic_add(&jac(i, i), dt * rij * dustdens(j));
      if (i != j)
        Kokkos::atomic_add(&jac(i, j), dt * rij * dustdens(i));
      else
        Kokkos::atomic_add(&jac(i, i), dt * rij * dustdens(j));
    }
  });
  mbr.team_barrier();
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CoagulationOneCellImplicit
//  \brief One backward-Euler implicit step over the coagulation dt.
//
//  Three performance optimizations over the naive FD-Jacobian approach:
//  1. Rate caching: pairwise collision rates precomputed once before the Newton loop.
//  2. Analytic Jacobian: J = I - dt*dS/dn built analytically from the bilinear source
//     structure (rates frozen), replacing nact full Source() evaluations per iteration.
//  3. Jacobian lagging: the factored Jacobian is reused for coag.newton_jac_lag Newton
//     iterations before rebuilding, amortising the O(nact^2) Jacobian build cost.
template <bool kDebugPrint, typename JacView2D, typename ResidView2D, typename RateView2D>
KOKKOS_INLINE_FUNCTION int CoagulationOneCellImplicit(
    parthenon::team_mbr_t const &mbr, const bool &surface, const Real & /*time*/,
    const Real &dt, const StateParams &kernel, const ScratchPad1D<Real> &dustdens,
    const ScratchPad1D<Real> &stime, const ScratchPad1D<Real> &vel, const int &nvel,
    const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &nQs, const CoagParams &coag,
    const CoagArrays &coag_arrays, const RateParams &rate,
    const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &n_new,
    const ScratchPad1D<Real> & /*n_trial*/, const ScratchPad1D<Real> & /*src_pert*/,
    const ScratchPad1D<Real> &n_old, const ResidView2D &residual, const JacView2D &jac,
    const RateView2D &rcoag, const RateView2D &rfrag, const ScratchPad1D<Real> &m1r,
    const ScratchPad1D<Real> &m2r, const ScratchPad1D<Real> &m3r) {
  const int nm1 = coag.nm - 1;
  const int &pgrid = coag.pgrid;
  const Real &dfloor = coag.dfloor;
  const Real &chi = coag.chi;
  const bool &do_momentum_conserving_update = coag.mom_coag;
  const int max_iter = coag.newton_max_iter;
  const Real tol = coag.newton_tol;
  const int jac_lag = std::max(1, coag.newton_jac_lag);

  auto &idx_largest = coag_arrays.idx_largest;
  auto &mass_grid = coag_arrays.mass_grid;
  auto &coagR3D = coag_arrays.coagR3D;
  auto &Kijk_sym_ind = coag_arrays.Kijk_sym_ind;
  auto &Kijk_sym = coag_arrays.Kijk_sym;
  auto &log_widths = coag_arrays.log_widths;
  auto &dsize = coag_arrays.dsize;
  const BinRecon bin_recon = coag.bin_recon;

  // Convert to number densities and stash initial state.
  ConvertToNumberDensity(mbr, nm1, dustdens, mass_grid, dfloor);
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    n_old(i) = dustdens(i);
    n_new(i) = dustdens(i);
  });
  mbr.team_barrier();

  constexpr int pgrid_pad_extra = 2;
  constexpr Real kStagnFactor = 0.9;

  // Precompute rates once over the initial active range.
  // FindMIMax returns INT_MIN identity when no bin is populated; clamp to -1 to signal.
  const int mimax_init_raw = FindMIMax(mbr, nm1, n_new, mass_grid, dfloor);
  if (mimax_init_raw < 0) {
    // Cell has no populated bins; nothing to do.
    ConvertToVolumeDensity(mbr, nm1, dustdens, mass_grid);
    return 0;
  }
  const int nact_cache_m1 = std::min(nm1, mimax_init_raw + pgrid + pgrid_pad_extra);
  if (bin_recon == BinRecon::PLMLogSize) {
    ComputeBinSizeMomentRatios(mbr, nm1, nact_cache_m1, n_new, mass_grid, log_widths, m1r,
                               m2r, m3r);
  }
  BuildRateCache(mbr, nm1, nact_cache_m1, vel, stime, kernel, mass_grid, coagR3D, surface,
                 rate, rcoag, rfrag, bin_recon, dsize, m1r, m2r, m3r);

  int iter = 0;
  Real last_res_norm = 0.0;
  Real prev_res_norm = std::numeric_limits<Real>::max();
  int last_nact = 0;
  bool stagnated = false;
  int jac_age = jac_lag; // force build on first iteration
  int last_nact_m1 = -1; // track nact from previous iter to detect size change

  for (iter = 0; iter < max_iter; iter++) {
    // Active subsystem size from current iterate.
    const int mimax_raw = FindMIMax(mbr, nm1, n_new, mass_grid, dfloor);
    if (mimax_raw < 0) break; // all bins collapsed below floor
    const int mimax = mimax_raw;
    const int nact_m1 = std::min(nm1, mimax + pgrid + pgrid_pad_extra);
    const int nact = nact_m1 + 1;

    // Evaluate source using cached rates.
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                             [&](const int i) { dustdens(i) = n_new(i); });
    mbr.team_barrier();
    SourceFromCache(mbr, nm1, nact_m1, pgrid, source, dustdens, idx_largest, mass_grid,
                    coagR3D, Kijk_sym_ind, Kijk_sym, rcoag, rfrag);

    // Residual: ||F||_inf / ||n||_inf.
    Real F_inf = 0.0, n_inf = 0.0;
    parthenon::par_reduce_inner(
        parthenon::inner_loop_pattern_ttr_tag, mbr, 0, nact_m1,
        [&](const int i, Real &lmax) {
          const Real Ri = n_new(i) - n_old(i) - dt * source(i);
          residual(i, 0) = Ri;
          const Real ai = std::abs(Ri);
          if (ai > lmax) lmax = ai;
        },
        Kokkos::Max<Real>(F_inf));
    parthenon::par_reduce_inner(
        parthenon::inner_loop_pattern_ttr_tag, mbr, 0, nact_m1,
        [&](const int i, Real &lmax) {
          const Real ai = std::abs(n_new(i));
          if (ai > lmax) lmax = ai;
        },
        Kokkos::Max<Real>(n_inf));
    const Real res_norm = F_inf / std::max(n_inf, std::numeric_limits<Real>::min());
    last_res_norm = res_norm;
    last_nact = nact;

    if (kDebugPrint) {
      Kokkos::single(Kokkos::PerTeam(mbr), [&]() {
        printf("[coag-impl] iter=%d nact=%d mimax=%d res=%.3e tol=%.3e jac_age=%d\n",
               iter, nact, mimax, res_norm, tol, jac_age);
      });
    }
    if (res_norm < tol) break;
    if (iter > 0 && res_norm > kStagnFactor * prev_res_norm) {
      stagnated = true;
      if (kDebugPrint) {
        Kokkos::single(Kokkos::PerTeam(mbr), [&]() {
          printf("[coag-impl] stagnation detected res=%.3e prev=%.3e iter=%d\n", res_norm,
                 prev_res_norm, iter);
        });
      }
      break;
    }
    prev_res_norm = res_norm;

    // Build analytic Jacobian if the lag has expired or nact grew; otherwise reuse.
    auto jac_act = Kokkos::subview(jac, Kokkos::pair<int, int>(0, nact),
                                   Kokkos::pair<int, int>(0, nact));
    auto rhs_act =
        Kokkos::subview(residual, Kokkos::pair<int, int>(0, nact), Kokkos::ALL());
    if (jac_age >= jac_lag || nact_m1 > last_nact_m1) {
      AnalyticJacobian(mbr, nm1, nact_m1, pgrid, dustdens, idx_largest, mass_grid,
                       coagR3D, Kijk_sym_ind, Kijk_sym, rcoag, rfrag, dt, jac);
      KokkosBatched::TeamLU<parthenon::team_mbr_t,
                            KokkosBatched::Algo::LU::Unblocked>::invoke(mbr, jac_act);
      mbr.team_barrier();
      jac_age = 0;
      last_nact_m1 = nact_m1;
    }

    // Solve with the (possibly lagged) factored Jacobian.
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nact_m1,
                             [&](const int i) { residual(i, 0) = -residual(i, 0); });
    mbr.team_barrier();
    KokkosBatched::TeamSolveLU<parthenon::team_mbr_t, KokkosBatched::Trans::NoTranspose,
                               KokkosBatched::Algo::Trsm::Unblocked>::invoke(mbr, jac_act,
                                                                             rhs_act);
    mbr.team_barrier();

    // Newton update with non-negativity floor.
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nact_m1,
                             [&](const int i) {
                               Real ni = n_new(i) + residual(i, 0);
                               const Real nfloor = 0.01 * dfloor / mass_grid(i);
                               if (ni < nfloor) ni = nfloor;
                               n_new(i) = ni;
                             });
    mbr.team_barrier();
    jac_age++;
  }

  if (kDebugPrint) {
    Kokkos::single(Kokkos::PerTeam(mbr), [&]() {
      const int iters_done = (iter < max_iter) ? iter + 1 : max_iter;
      if (stagnated) {
        printf("[coag-impl] STAGNATED after %d iters res=%.3e tol=%.3e nact=%d dt=%.3e\n",
               iters_done, last_res_norm, tol, last_nact, dt);
      } else if (iter >= max_iter) {
        printf("[coag-impl] WARNING: hit max_iter=%d last_res=%.3e tol=%.3e nact=%d "
               "dt=%.3e\n",
               max_iter, last_res_norm, tol, last_nact, dt);
      } else {
        printf("[coag-impl] converged in %d iters res=%.3e nact=%d dt=%.3e\n", iters_done,
               last_res_norm, last_nact, dt);
      }
    });
  }

  // Commit source = (n_new - n_old)/dt for UpdateVelocityNQ.
  const Real inv_dt = 1.0 / dt;
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1, [&](const int i) {
    dustdens(i) = n_old(i);
    source(i) = (n_new(i) - n_old(i)) * inv_dt;
  });
  mbr.team_barrier();

  // Find mimax for momentum source (uses accepted state n_new).
  const int mimax_final_raw = FindMIMax(mbr, nm1, n_new, mass_grid, dfloor);
  const int mimax = (mimax_final_raw >= 0) ? mimax_final_raw : 0;

  // Momentum-conserving velocity update at the accepted implicit state.
  if (do_momentum_conserving_update) {
    parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                             [&](const int i) { dustdens(i) = n_new(i); });
    mbr.team_barrier();
    for (int n = 0; n < nvel; n++) {
      ZeroSourceNQ(mbr, n, nm1, Q, nQs, vel, kernel.nvel);
      SourceNQ(mbr, n, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel, mass_grid,
               coagR3D, Kijk_sym_ind, Kijk_sym, chi, surface, rate);
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                               [&](const int i) { dustdens(i) = n_old(i); });
      mbr.team_barrier();
      UpdateVelocityNQ(mbr, n, nm1, Q, nQs, vel, kernel.nvel, dustdens, source, mass_grid,
                       dt, dfloor);
      parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                               [&](const int i) { dustdens(i) = n_new(i); });
      mbr.team_barrier();
    }
  }

  // Commit and convert back to volume density.
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, mbr, 0, nm1,
                           [&](const int i) { dustdens(i) = n_new(i); });
  mbr.team_barrier();
  ConvertToVolumeDensity(mbr, nm1, dustdens, mass_grid);
  return iter + 1;
}

//----------------------------------------------------------------------------------------
//! \fn  void Dust::Coagulation::CoagulationOneCell
//  \brief
template <typename RateView2D>
KOKKOS_INLINE_FUNCTION int CoagulationOneCell(
    parthenon::team_mbr_t const &mbr, const bool &surface, const Real &time,
    Real &dt_sync, const StateParams &kernel, const ScratchPad1D<Real> &dustdens,
    const ScratchPad1D<Real> &stime, const ScratchPad1D<Real> &vel, const int &nvel,
    const ScratchPad1D<Real> &Q, const ScratchPad1D<Real> &nQs, const CoagParams &coag,
    const CoagArrays &coag_arrays, const RateParams &rate,
    const ScratchPad1D<Real> &source, const ScratchPad1D<Real> &Q2,
    const RateView2D &rcoag, const RateView2D &rfrag, const ScratchPad1D<Real> &m1r,
    const ScratchPad1D<Real> &m2r, const ScratchPad1D<Real> &m3r) {
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
  auto &log_widths = coag_arrays.log_widths;
  auto &dsize = coag_arrays.dsize;
  const BinRecon bin_recon = coag.bin_recon;

  // Timestepping
  int ncall = 0;
  Real time_dummy = time;
  Real dt_sync1 = dt_sync;
  Real dt = dt_sync1;
  Real hnext = dt;
  dt_sync = 1e-15;
  const Real time_goal = time_dummy + dt_sync1;

  // Update distribution
  ConvertToNumberDensity(mbr, nm1, dustdens, mass_grid, dfloor);
  while (std::abs(time_dummy - time_goal) > 1e-6 * dt) {
    // Set source using cached rates (BuildRateCache once per step, then SourceFromCache).
    const int mimax = FindMIMax(mbr, nm1, dustdens, mass_grid, dfloor);
    if (bin_recon == BinRecon::PLMLogSize) {
      ComputeBinSizeMomentRatios(mbr, nm1, mimax, dustdens, mass_grid, log_widths, m1r,
                                 m2r, m3r);
    }
    BuildRateCache(mbr, nm1, mimax, vel, stime, kernel, mass_grid, coagR3D, surface, rate,
                   rcoag, rfrag, bin_recon, dsize, m1r, m2r, m3r);
    SourceFromCache(mbr, nm1, mimax, pgrid, source, dustdens, idx_largest, mass_grid,
                    coagR3D, Kijk_sym_ind, Kijk_sym, rcoag, rfrag);

    int mimax2 = Null<int>();
    if (!(do_adaptive) || (coag_int == 1)) {
      dt_sync1 = TimeStepControl(mbr, nm1, dustdens, source, mass_grid, dfloor, cfl);
      dt = std::min(dt_sync1, time_goal - time_dummy);
      dt_sync = dt_sync1;
      if (coag_int == 3) {
        mimax2 = FindMIMaxNQS3(mbr, nm1, dt, dustdens, source, Q, mass_grid, dfloor);
        // now Q stores dustdens + dt*source(), nQs will be used for temporary source(*)
        // Rates are density-independent so the cached rcoag/rfrag are still valid.
        SourceFromCache(mbr, nm1, mimax2, pgrid, nQs, Q, idx_largest, mass_grid, coagR3D,
                        Kijk_sym_ind, Kijk_sym, rcoag, rfrag);
      }
    } else { // adaptive third-order method
      // Set source
      Real h0 = hnext, h = h0;
      Real emax = Null<Real>();
      while (1) {
        mimax2 = FindMIMaxNQS3(mbr, nm1, h, dustdens, source, Q, mass_grid, dfloor);
        // now Q stores dustdens + dt*source(), nQs will be used for temporary source(*)
        // Rates are density-independent so the cached rcoag/rfrag are still valid.
        SourceFromCache(mbr, nm1, mimax2, pgrid, nQs, Q, idx_largest, mass_grid, coagR3D,
                        Kijk_sym_ind, Kijk_sym, rcoag, rfrag);
        emax = ComputeError(mbr, mimax, mimax2, h, h0, dustdens, source, nQs, err_eps);
        if (emax <= 1.0) break;
        h = std::max(S * h * std::pow(emax, pshrink), 0.1 * h);
      }
      hnext = (emax > err_con) ? S * h * std::pow(emax, pgrow) : 5.0 * h;
      dt = h;
    }

    if (coag_int == 1) {
      // Momentum Conserving Update (iff do_momentum_conserving_update)
      for (int n = 0; n < do_momentum_conserving_update * nvel; n++) {
        ZeroSourceNQ(mbr, n, nm1, Q, nQs, vel, kernel.nvel);
        SourceNQ(mbr, n, nm1, mimax, pgrid, Q, nQs, dustdens, vel, stime, kernel,
                 mass_grid, coagR3D, Kijk_sym_ind, Kijk_sym, chi, surface, rate);
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
                   stime, kernel, mass_grid, coagR3D, Kijk_sym_ind, Kijk_sym, chi,
                   surface, rate, dt);
        UpdateVelocityNQS3(mbr, n, nm1, Q2, nQs, vel, kernel.nvel, dustdens, source,
                           mass_grid, dt, dfloor);
      }

      // Update dust density
      UpdateDensityNQS3(mbr, nm1, dustdens, source,
                        (do_momentum_conserving_update ? Q2 : nQs), dt);
    }

    // Update time and increment ncall
    time_dummy += dt;
    ncall++;

    dt_sync = (do_adaptive) ? std::max(hnext, dt_sync) : dt_sync;
    hnext = (do_adaptive) ? std::min(hnext, time_goal - time_dummy) : hnext;

    // Warn and break upon reaching ncall_max
    if (ncall > ncall_max) {
      printf("(Coagulation): Reach ncall_max in coagulation kernel!");
      break;
    }
  }
  ConvertToVolumeDensity(mbr, nm1, dustdens, mass_grid);
  return ncall;
} // end of CoagulationOneCell

} // namespace Coagulation
} // namespace Dust

#endif // DUST_COAGULATION_COAGULATION_HPP_
