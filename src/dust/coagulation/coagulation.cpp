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

// Artemis includes
#include "dust/coagulation/coagulation.hpp"
#include "artemis.hpp"
#include "dust/dust.hpp"
#include "geometry/geometry.hpp"

namespace Dust {
namespace Coagulation {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Coagulalation::Initialize
//! \brief Adds intialization function for coagulation package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Params &dustPars) {
  auto coag = std::make_shared<StateDescriptor>("coagulation");
  Params &params = coag->AllParams();

  CoagParams cpars;

  const int nm = dustPars.template Get<int>("nspecies");
  const Real dfloor = dustPars.template Get<Real>("dfloor");
  const Real rho_p = dustPars.template Get<Real>("grain_density");
  cpars.nm = nm;
  cpars.vfrag = pin->GetOrAddReal("dust", "vfrag", 1.e3); // cm/s
  cpars.nlim = 1e10;
  cpars.integrator = pin->GetOrAddInteger("dust/coagulation", "coag_int", 3);
  cpars.use_adaptive =
      pin->GetOrAddBoolean("dust/coagulation", "coag_use_adaptiveStep", true);
  cpars.mom_coag = pin->GetOrAddBoolean("dust/coagulation", "coag_mom_preserve", true);
  cpars.nCall_mx = pin->GetOrAddInteger("dust/coagulation", "coag_nsteps_mx", 1000);
  cpars.rho_p = rho_p;

  const Real M_SUN = 1.988409870698051e33;      // gram (sun)
  const Real GRAV_CONST = 6.674299999999999e-8; // gravitational const in cm^3 g^-1 s^-2
  const Real mstar = pin->GetOrAddReal("problem", "mstar", 1.0) * M_SUN;
  cpars.gm = std::sqrt(GRAV_CONST * mstar);
  const bool const_omega = pin->GetOrAddBoolean("problem", "const_coag_omega", false);
  cpars.const_omega = const_omega;
  if (const_omega) {
    const Real AU_LENGTH = 1.4959787070000e13; // cm
    const Real r0 = pin->GetOrAddReal("problem", "r0_length", 1.0) * AU_LENGTH;
    cpars.gm /= (std::sqrt(r0) * r0);
  }

  cpars.ibounce = pin->GetOrAddBoolean("dust/coagulation", "coag_bounce", false);

  int coord_type = 0; // density

  const bool isurface_den = pin->GetOrAddBoolean("dust", "surface_density_flag", true);
  if (isurface_den) coord_type = 1;
  cpars.coord = coord_type; // 1--surface density, 0: 3D

  cpars.err_eps = 1.0e-1;
  cpars.S = 0.9;
  cpars.cfl = 1.0e-1;
  cpars.chi = 1.0;

  // some checks and definition
  if (cpars.use_adaptive) {
    if (cpars.integrator == 3) {
      cpars.pgrow = -0.5;
      cpars.pshrink = -1.0;
    } else if (cpars.integrator == 5) {
      cpars.pgrow = -0.2;
      cpars.pshrink = -0.25;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in dust coagulation initialization: " << std::endl
          << "###   You can not use this integrator with adaptive step sizing: "
          << cpars.integrator << std::endl;
      PARTHENON_FAIL(msg);
    }
    cpars.errcon = std::pow((5. / cpars.S), (1. / cpars.pgrow));
  }

  auto h_sizes = dustPars.template Get<ParArray1D<Real>>("h_sizes");
  const Real cond = 3.0 / (1.0 - nm) * std::log(h_sizes(0) / h_sizes(nm - 1));
  if (std::exp(cond) > std::sqrt(2.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in dust with coagulation: using nspecies >"
        << 3.0 * std::log(h_sizes(nm - 1) / h_sizes(0)) / (std::log(std::sqrt(2.0))) + 1.
        << " instead of " << nm << std::endl;
    PARTHENON_FAIL(msg);
  }

  auto dust_size = dustPars.template Get<ParArray1D<Real>>("sizes");

  // allocate array and assign values
  cpars.klf = ParArray2D<int>("klf", nm, nm);
  cpars.mass_grid = ParArray1D<Real>("mass_grid", nm);
  const int n2DRv = coag2DRv::last2;
  cpars.coagR3D = ParArray3D<Real>("coagReal3D", n2DRv, nm, nm);
  cpars.cpod_notzero = ParArray3D<int>("idx_nzcpod", nm, nm, 4);
  cpars.cpod_short = ParArray3D<Real>("nzcpod", nm, nm, 4);

  cpars.dfloor = dfloor;
  Real a = 3.0 * std::log10(h_sizes(0) / h_sizes(nm - 1)) / static_cast<Real>(1 - nm);

  initializeArray(nm, cpars.pGrid, cpars.rho_p, cpars.chi, a, dust_size, cpars.klf,
                  cpars.mass_grid, cpars.coagR3D, cpars.cpod_notzero, cpars.cpod_short);

  params.Add("coag_pars", cpars);

  // other parameters for coagulation
  const int nstep1Coag = pin->GetOrAddReal("problem", "nstep1Coag", 50);
  params.Add("nstep1Coag", nstep1Coag);
  Real dtCoag = 0.0;
  params.Add("dtCoag", dtCoag, Params::Mutability::Restart);
  const Real alpha = pin->GetOrAddReal("dust/coagulation", "coag_alpha", 1.e-3);
  params.Add("coag_alpha", alpha);
  const int scr_level = pin->GetOrAddReal("dust/coagulation", "coag_scr_level", 0);
  params.Add("coag_scr_level", scr_level);
  const bool info_out = pin->GetOrAddBoolean("dust/coagulation", "coag_info_out", false);
  params.Add("coag_info_out", info_out);

  return coag;
}

// using Kokkos par_for() loop
void initializeArray(const int nm, int &pGrid, const Real &rho_p, const Real &chi,
                     const Real &a, const ParArray1D<Real> dsize, ParArray2D<int> klf,
                     ParArray1D<Real> mass_grid, ParArray3D<Real> coag3D,
                     ParArray3D<int> cpod_notzero, ParArray3D<Real> cpod_short) {

  int ikdelta = coag2DRv::kdelta;
  int icoef_fett = coag2DRv::coef_fett;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag1", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        // initialize in_idx(*) array
        // initialize kdelta array
        for (int j = 0; j < nm; j++) {
          coag3D(ikdelta, i, j) = 0.0;
        }
        coag3D(ikdelta, i, i) = 1.0;
        mass_grid(i) = 4.0 * M_PI / 3.0 * rho_p * dsize(i) * dsize(i) * dsize(i);
        // initialize coag3D(icoef_fett, *,*)
        for (int j = 0; j < nm; j++) {
          Real tmp1 = (1.0 - 0.5 * coag3D(ikdelta, i, j));
          coag3D(icoef_fett, i, j) = M_PI * SQR(dsize(i) + dsize(j)) * tmp1;
        }
      });

  // set fragmentation variables
  // Real a = std::log10(massGrid_h(0) / massGrid_h(nm - 1)) / (1 - nm);
  Real ten_a = std::pow(10.0, a);
  Real ten_ma = 1.0 / ten_a;
  int ce = int(-1.0 / a * std::log10(1.0 - ten_ma)) + 1;

  pGrid = floor(1.0 / a); // used in integration

  Real frag_slope = 2.0 - 11.0 / 6.0;

  int iphiFrag = coag2DRv::phiFrag, iepsFrag = coag2DRv::epsFrag;
  int iaFrag = coag2DRv::aFrag;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag2", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        Real sum_pF = 0.0;
        for (int j = 0; j <= i; j++) {
          coag3D(iphiFrag, j, i) = std::pow(mass_grid(j), frag_slope);
          sum_pF += coag3D(iphiFrag, j, i);
        }
        // normalization
        for (int j = 0; j <= i; j++) {
          coag3D(iphiFrag, j, i) /= sum_pF; // switch (i,j) from fortran
        }

        // Cratering
        for (int j = 0; j <= i - pGrid - 1; j++) {
          // FRAGMENT DISTRIBUTION
          // The largest fragment has the mass of the smaller collision partner

          // Mass bin of largest fragment
          klf(i, j) = j;

          coag3D(iaFrag, i, j) = (1.0 + chi) * mass_grid(j);
          //                      |_______|
          //                           |
          //                    Mass of fragments
          coag3D(iepsFrag, i, j) = chi * mass_grid(j) / (mass_grid(i) * (1.0 - ten_ma));
        }

        int i1 = std::max(0, i - pGrid);
        for (int j = i1; j <= i; j++) {
          // The largest fragment has the mass of the larger collison partner
          klf(i, j) = i;
          coag3D(iaFrag, i, j) = (mass_grid(i) + mass_grid(j));
        }
      });

  // initialize dalp array
  // Calculate the D matrix
  // Calculate the E matrix
  ParArray2D<Real> e("epod", nm, nm);
  int idalp = coag2DRv::dalp, idpod = coag2DRv::dpod;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag4", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int k) {
        for (int j = 0; j < nm; j++) {
          if (j <= k + 1 - ce) {
            coag3D(idalp, k, j) = 1.0;
            coag3D(idpod, k, j) = -mass_grid(j) / (mass_grid(k) * (ten_a - 1.0));
          } else {
            coag3D(idpod, k, j) = -1.0;
            coag3D(idalp, k, j) = 0.0;
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
            cpod(i, j, k) = (0.5 * coag3D(ikdelta, i, j) * cpod(i, j, k) +
                             cpod(i, j, k) * theta_kj * dtheta_ji);
          }
          cpod(i, j, j) += coag3D(idpod, j, i);
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

} // namespace Coagulation
} // namespace Dust
