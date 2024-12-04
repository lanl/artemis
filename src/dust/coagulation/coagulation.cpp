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

using namespace parthenon::package::prelude;

namespace Dust {
namespace Coagulation {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Coagulalation::Initialize
//! \brief Adds intialization function for coagulation package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto coag = std::make_shared<StateDescriptor>("coagulation");
  Params &params = coag->AllParams();

  CoagParams cpars;
  const int nm = pin->GetOrAddInteger("dust", "nspecies", 1);
  cpars.nm = nm;
  cpars.vfrag = pin->GetOrAddReal("dust", "vfrag", 1.e3); // cm/s
  cpars.nlim = 1e10;
  cpars.integrator = pin->GetOrAddInteger("dust", "coag_int", 3);
  cpars.use_adaptive = pin->GetOrAddBoolean("dust", "coag_use_adaptiveStep", true);
  cpars.mom_coag = pin->GetOrAddBoolean("dust", "coag_mom_preserve", true);
  cpars.nCall_mx = pin->GetOrAddInteger("dust", "coag_nsteps_mx", 1000);
  // dust particle internal density g/cc
  const Real rho_p = pin->GetOrAddReal("dust", "rho_p", 1.25);
  cpars.rho_p = rho_p;

  cpars.ibounce = pin->GetOrAddBoolean("dust", "coag_bounce", false);

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

  const Real s_min = pin->GetReal("dust", "min_size");
  const Real s_max = pin->GetReal("dust", "max_size");
  const Real mmin = 4.0 * M_PI / 3.0 * rho_p * std::pow(s_min, 3);
  const Real mmax = 4.0 * M_PI / 3.0 * rho_p * std::pow(s_max, 3);
  const Real cond = 1.0 / (1.0 - nm) * std::log(mmin / mmax);
  const Real conc = std::log(mmin);
  if (std::exp(cond) > std::sqrt(2.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in dust with coagulation: using nspecies >"
        << std::log(mmax / mmin) / (std::log(std::sqrt(2.0))) + 1. << " instead of " << nm
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  ParArray1D<Real> dust_size("dustSize", nm);
  auto dust_size_host = dust_size.GetHostMirror();

  for (int n = 0; n < nm; ++n) {
    Real mgrid = std::exp(conc + cond * n);
    dust_size_host(n) = std::pow(3.0 * mgrid / (4.0 * M_PI * cpars.rho_p), 1. / 3.);
  }
  dust_size.DeepCopy(dust_size_host);

  // allocate array and assign values
  cpars.klf = ParArray2D<int>("klf", nm, nm);
  cpars.mass_grid = ParArray1D<Real>("mass_grid", nm);
  const int n2DRv = coag2DRv::last2;
  cpars.coagR3D = ParArray3D<Real>("coagReal3D", n2DRv, nm, nm);
  cpars.cpod_notzero = ParArray3D<int>("idx_nzcpod", nm, nm, 4);
  cpars.cpod_short = ParArray3D<Real>("nzcpod", nm, nm, 4);

  if (Dust::cgsunit == NULL) {
    Dust::cgsunit = new CGSUnit;
  }
  if (!Dust::cgsunit->isSet()) {
    Dust::cgsunit->SetCGSUnit(pin);
  }
  const Real dfloor = pin->GetOrAddReal("dust", "dfloor", 1.0e-25);
  // convert to CGS unit
  cpars.dfloor = dfloor * Dust::cgsunit->mass0 / Dust::cgsunit->vol0;
  Real a = std::log10(mmin / mmax) / static_cast<Real>(1 - nm);

  initializeArray(nm, cpars.pGrid, cpars.rho_p, cpars.chi, a, dust_size, cpars.klf,
                  cpars.mass_grid, cpars.coagR3D, cpars.cpod_notzero, cpars.cpod_short);

  params.Add("coag_pars", cpars);

  // other parameters for coagulation
  const int nstep1Coag = pin->GetOrAddReal("problem", "nstep1Coag", 50);
  params.Add("nstep1Coag", nstep1Coag);
  Real dtCoag = 0.0;
  params.Add("dtCoag", dtCoag, Params::Mutability::Restart);
  const Real alpha = pin->GetOrAddReal("dust", "coag_alpha", 1.e-3);
  params.Add("coag_alpha", alpha);
  const int scr_level = pin->GetOrAddReal("dust", "coag_scr_level", 0);
  params.Add("coag_scr_level", scr_level);

  return coag;
}

// using Kokkos par_for() loop
void initializeArray(const int nm, int &pGrid, const Real &rho_p, const Real &chi,
                     const Real &a, const ParArray1D<Real> dsize, ParArray2D<int> klf,
                     ParArray1D<Real> mass_grid, ParArray3D<Real> coag3D,
                     ParArray3D<int> cpod_notzero, ParArray3D<Real> cpod_short) {

  int ikdelta = coag2DRv::kdelta;
  auto kdelta = Kokkos::subview(coag3D, ikdelta, Kokkos::ALL, Kokkos::ALL);
  int icoef_fett = coag2DRv::coef_fett;
  auto coef_fett = Kokkos::subview(coag3D, icoef_fett, Kokkos::ALL, Kokkos::ALL);
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeCoag1", parthenon::DevExecSpace(),
      0, nm - 1, KOKKOS_LAMBDA(const int i) {
        // initialize in_idx(*) array
        // initialize kdelta array
        for (int j = 0; j < nm; j++) {
          kdelta(i, j) = 0.0;
        }
        kdelta(i, i) = 1.0;
        mass_grid(i) = 4.0 * M_PI / 3.0 * rho_p * std::pow(dsize(i), 3);
        // initialize coef_fett(*,*)
        for (int j = 0; j < nm; j++) {
          Real tmp1 = (1.0 - 0.5 * kdelta(i, j));
          coef_fett(i, j) = M_PI * SQR(dsize(i) + dsize(j)) * tmp1;
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
  auto phiFrag = Kokkos::subview(coag3D, iphiFrag, Kokkos::ALL, Kokkos::ALL);
  auto epsFrag = Kokkos::subview(coag3D, iepsFrag, Kokkos::ALL, Kokkos::ALL);
  auto aFrag = Kokkos::subview(coag3D, iaFrag, Kokkos::ALL, Kokkos::ALL);
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

          aFrag(i, j) = (1.0 + chi) * mass_grid(j);
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
  int idalp = coag2DRv::dalp, idpod = coag2DRv::dpod;
  auto dalp = Kokkos::subview(coag3D, idalp, Kokkos::ALL, Kokkos::ALL);
  auto dpod = Kokkos::subview(coag3D, idpod, Kokkos::ALL, Kokkos::ALL);
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

} // namespace Coagulation
} // namespace Dust
