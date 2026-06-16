//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld.hpp
//! \brief Contains HLLD Riemann solver for hydrodynamics
//!
//! Computes fluxes using the Harten-Lax-vanLeer-Discontinuities (HLLD) Riemann solver.
//!
#ifndef UTILS_FLUXES_RIEMANN_HLLD_HPP_
#define UTILS_FLUXES_RIEMANN_HLLD_HPP_

// C++ headers
#include <algorithm>
#include <cmath>

// Artemis headers
#include "artemis.hpp"
#include "radiation/moments/moments.hpp"
#include "utils/eos/eos.hpp"

// NOTE(AMD): The following is mostly taken directly from the open-source Athena++/AthenaK
// software

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \class ArtemisUtils::RiemannSolver<RSolver::hlld, ...>
//! \brief The HLLD Riemann solver for ideal gas
template <Fluid FLUID_TYPE, Closure CTYPE>
struct RiemannSolver<RSolver::hlld, FLUID_TYPE, CTYPE,
                     std::enable_if_t<FLUID_TYPE == Fluid::gas>> {
  template <typename V1, typename V2, typename V3>
  KOKKOS_INLINE_FUNCTION void
  operator()(const EOS &eos, const Real c, const Real chat, const Real mu0,
             const bool do_mhd, parthenon::team_mbr_t const &member, const int b,
             const int k, const int j, const int il, const int iu, const int dir,
             const parthenon::ScratchPad2D<Real> &wl,
             const parthenon::ScratchPad2D<Real> &wr, const V1 &p, const V2 &q,
             const V3 &vf) const {
    using TE = parthenon::TopologicalElement;
    // Check sensibility of flux direction
    PARTHENON_REQUIRE(do_mhd, "HLLD solver only implemented for MHD!");
    PARTHENON_REQUIRE(dir > 0 && dir <= 3, "Invalid flux direction!");
    auto fdir = (dir == 1) ? TE::F1 : ((dir == 2) ? TE::F2 : TE::F3);

    // Obtain number of species
    int nspecies = q.GetSize(b, gas::cons::density());

    const int IBX = nspecies * 7 + (dir - 1);
    const int IBY = nspecies * 7 + ((dir - 1) + 1) % 3;
    const int IBZ = nspecies * 7 + ((dir - 1) + 2) % 3;
    const int IBM = nspecies * 7 + 3;
    const int IBXG = dir - 1;
    const int IBYG = dir % 3;
    const int IBZG = (dir + 1) % 3;

    parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          // Create local references for L/R states (helps compiler vectorize)
          const int n = 0;
          const int IDN = 0;
          const int ivx = nspecies + ((dir - 1));
          const int ivy = nspecies + ((dir - 1) + 1) % 3;
          const int ivz = nspecies + ((dir - 1) + 2) % 3;
          // Unused indices for dust hydrodynamics
          const int IPR = nspecies * 4;
          const int ISE = nspecies * 5;
          const int IBL = nspecies * 6;
          const int IEN = IPR;
          const int IEG = ISE;
          Real &wl_idn = wl(IDN, i);
          Real &wl_ivx = wl(ivx, i);
          Real &wl_ivy = wl(ivy, i);
          Real &wl_ivz = wl(ivz, i);
          Real &wl_ipr = wl(IPR, i);
          Real &wl_ise = wl(ISE, i);
          Real &wl_ibl = wl(IBL, i);

          Real &wr_idn = wr(IDN, i);
          Real &wr_ivx = wr(ivx, i);
          Real &wr_ivy = wr(ivy, i);
          Real &wr_ivz = wr(ivz, i);
          Real &wr_ipr = wr(IPR, i);
          Real &wr_ise = wr(ISE, i);
          Real &wr_ibl = wr(IBL, i);

          Real &wl_ibx = wl(IBX, i);
          Real &wl_iby = wl(IBY, i);
          Real &wl_ibz = wl(IBZ, i);
          Real &wr_ibx = wr(IBX, i);
          Real &wr_iby = wr(IBY, i);
          Real &wr_ibz = wr(IBZ, i);

          const Real small = 1.0e-20;
          const Real eps = 1.0e-12;

          Real frho = Null<Real>();
          Real fmx = Null<Real>();
          Real fmy = Null<Real>();
          Real fmz = Null<Real>();
          Real fe = Null<Real>();
          Real pface = 0.0;
          Real pmag_face = 0.0;
          Real fby = 0.0;
          Real fbz = 0.0;

          const Real bxi = 0.5 * (wl_ibx + wr_ibx);
          const Real sqrt_mu0 = std::sqrt(mu0);
          const Real inv_sqrt_mu0 = 1.0 / sqrt_mu0;

          const Real bxi_n = bxi * inv_sqrt_mu0;
          const Real wl_iby_n = wl_iby * inv_sqrt_mu0;
          const Real wl_ibz_n = wl_ibz * inv_sqrt_mu0;
          const Real wr_iby_n = wr_iby * inv_sqrt_mu0;
          const Real wr_ibz_n = wr_ibz * inv_sqrt_mu0;
          const Real bxsq_n = SQR(bxi_n);
          const Real pbl = 0.5 * (bxsq_n + SQR(wl_iby_n) + SQR(wl_ibz_n));
          const Real pbr = 0.5 * (bxsq_n + SQR(wr_iby_n) + SQR(wr_ibz_n));
          const Real ptl = wl_ipr + pbl;
          const Real ptr = wr_ipr + pbr;
          const Real vdotBl = wl_ivx * bxi_n + wl_ivy * wl_iby_n + wl_ivz * wl_ibz_n;
          const Real vdotBr = wr_ivx * bxi_n + wr_ivy * wr_iby_n + wr_ivz * wr_ibz_n;
          const Real el = wl_idn * wl_ise +
                          0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz)) + pbl;
          const Real er = wr_idn * wr_ise +
                          0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz)) + pbr;

          const Real fl_d = wl_idn * wl_ivx;
          const Real fl_mx = wl_idn * SQR(wl_ivx) + ptl - bxsq_n;
          const Real fl_my = wl_idn * wl_ivx * wl_ivy - bxi_n * wl_iby_n;
          const Real fl_mz = wl_idn * wl_ivx * wl_ivz - bxi_n * wl_ibz_n;
          const Real fl_e = (el + ptl) * wl_ivx - bxi_n * vdotBl;
          const Real fl_by = wl_ivx * wl_iby - bxi * wl_ivy;
          const Real fl_bz = wl_ivx * wl_ibz - bxi * wl_ivz;

          const Real fr_d = wr_idn * wr_ivx;
          const Real fr_mx = wr_idn * SQR(wr_ivx) + ptr - bxsq_n;
          const Real fr_my = wr_idn * wr_ivx * wr_ivy - bxi_n * wr_iby_n;
          const Real fr_mz = wr_idn * wr_ivx * wr_ivz - bxi_n * wr_ibz_n;
          const Real fr_e = (er + ptr) * wr_ivx - bxi_n * vdotBr;
          const Real fr_by = wr_ivx * wr_iby - bxi * wr_ivy;
          const Real fr_bz = wr_ivx * wr_ibz - bxi * wr_ivz;

          const Real cfl =
              MHD::FastMagnetosonicSpeed(wl_ibl, wl_idn, bxi, wl_iby, wl_ibz, mu0);
          const Real cfr =
              MHD::FastMagnetosonicSpeed(wr_ibl, wr_idn, bxi, wr_iby, wr_ibz, mu0);

          const Real sl = std::min(wl_ivx - cfl, wr_ivx - cfr);
          const Real sr = std::max(wl_ivx + cfl, wr_ivx + cfr);

          const Real hlle_qa = sr > 0.0 ? sr : small;
          const Real hlle_qb = sl < 0.0 ? sl : -small;
          const Real hlle_qc = 0.5 * (hlle_qa + hlle_qb) / (hlle_qa - hlle_qb);
          const Real hlle_d = 0.5 * (wl_ipr + wr_ipr) + hlle_qc * (wl_ipr - wr_ipr);
          const Real hlle_db = 0.5 * (pbl + pbr) + hlle_qc * (pbl - pbr);
          const Real hlle_frho =
              0.5 * ((wl_ivx - hlle_qb) * wl_idn + (wr_ivx - hlle_qa) * wr_idn) +
              hlle_qc * ((wl_ivx - hlle_qb) * wl_idn - (wr_ivx - hlle_qa) * wr_idn);
          const Real hlle_fmx =
              0.5 * ((wl_ivx - hlle_qb) * wl_idn * wl_ivx - bxsq_n +
                     (wr_ivx - hlle_qa) * wr_idn * wr_ivx - bxsq_n) +
              hlle_qc * ((wl_ivx - hlle_qb) * wl_idn * wl_ivx - bxsq_n -
                         ((wr_ivx - hlle_qa) * wr_idn * wr_ivx - bxsq_n));
          const Real hlle_fmy =
              0.5 * ((wl_ivx - hlle_qb) * wl_idn * wl_ivy - bxi_n * wl_iby_n +
                     (wr_ivx - hlle_qa) * wr_idn * wr_ivy - bxi_n * wr_iby_n) +
              hlle_qc * ((wl_ivx - hlle_qb) * wl_idn * wl_ivy - bxi_n * wl_iby_n -
                         ((wr_ivx - hlle_qa) * wr_idn * wr_ivy - bxi_n * wr_iby_n));
          const Real hlle_fmz =
              0.5 * ((wl_ivx - hlle_qb) * wl_idn * wl_ivz - bxi_n * wl_ibz_n +
                     (wr_ivx - hlle_qa) * wr_idn * wr_ivz - bxi_n * wr_ibz_n) +
              hlle_qc * ((wl_ivx - hlle_qb) * wl_idn * wl_ivz - bxi_n * wl_ibz_n -
                         ((wr_ivx - hlle_qa) * wr_idn * wr_ivz - bxi_n * wr_ibz_n));
          const Real hlle_fe =
              0.5 * (el * (wl_ivx - hlle_qb) + ptl * wl_ivx - bxi_n * vdotBl +
                     er * (wr_ivx - hlle_qa) + ptr * wr_ivx - bxi_n * vdotBr) +
              hlle_qc * (el * (wl_ivx - hlle_qb) + ptl * wl_ivx - bxi_n * vdotBl -
                         (er * (wr_ivx - hlle_qa) + ptr * wr_ivx - bxi_n * vdotBr));
          const Real hlle_fby =
              0.5 * (fl_by - hlle_qb * wl_iby + fr_by - hlle_qa * wr_iby) +
              hlle_qc * ((fl_by - hlle_qb * wl_iby) - (fr_by - hlle_qa * wr_iby));
          const Real hlle_fbz =
              0.5 * (fl_bz - hlle_qb * wl_ibz + fr_bz - hlle_qa * wr_ibz) +
              hlle_qc * ((fl_bz - hlle_qb * wl_ibz) - (fr_bz - hlle_qa * wr_ibz));

          const Real sdl = sl - wl_ivx;
          const Real sdr = sr - wr_ivx;
          const Real denom = wr_idn * sdr - wl_idn * sdl;

          bool use_hlle = (std::abs(bxi) <= eps) || (std::abs(denom) <= small) ||
                          !std::isfinite(sl) || !std::isfinite(sr);

          Real sm = 0.0;
          Real ptst = 0.0;
          Real dlst = 0.0;
          Real drst = 0.0;
          Real sal = 0.0;
          Real sar = 0.0;

          Real vlst_y = 0.0, vlst_z = 0.0, blst_y = 0.0, blst_z = 0.0, elst = 0.0;
          Real vrst_y = 0.0, vrst_z = 0.0, brst_y = 0.0, brst_z = 0.0, erst = 0.0;
          Real vdst_y = 0.0, vdst_z = 0.0, bdst_y = 0.0, bdst_z = 0.0;
          Real eldst = 0.0, erdst = 0.0;

          if (!use_hlle) {
            sm = (wr_idn * sdr * wr_ivx - wl_idn * sdl * wl_ivx + ptl - ptr) / denom;
            const Real ptstl = ptl + wl_idn * sdl * (sm - wl_ivx);
            const Real ptstr = ptr + wr_idn * sdr * (sm - wr_ivx);
            ptst = 0.5 * (ptstl + ptstr);

            const Real sdml = sl - sm;
            const Real sdmr = sr - sm;
            if (std::abs(sdml) <= small || std::abs(sdmr) <= small) {
              use_hlle = true;
            } else {
              dlst = wl_idn * sdl / sdml;
              drst = wr_idn * sdr / sdmr;

              if (dlst <= 0.0 || drst <= 0.0 || !std::isfinite(dlst) ||
                  !std::isfinite(drst) || ptst <= 0.0 || !std::isfinite(ptst)) {
                use_hlle = true;
              } else {
                const Real dsl = wl_idn * sdl * sdml - bxsq_n;
                const Real dsr = wr_idn * sdr * sdmr - bxsq_n;
                const Real deg_tol = 1.0e-4 * ptst;

                if (std::abs(dsl) < deg_tol) {
                  vlst_y = wl_ivy;
                  vlst_z = wl_ivz;
                  blst_y = wl_iby;
                  blst_z = wl_ibz;
                } else {
                  const Real inv_dsl = 1.0 / dsl;
                  const Real mfact = bxi_n * (sm - wl_ivx) * inv_dsl;
                  const Real bfact = (wl_idn * SQR(sdl) - bxsq_n) * inv_dsl;
                  vlst_y = wl_ivy - wl_iby_n * mfact;
                  vlst_z = wl_ivz - wl_ibz_n * mfact;
                  blst_y = wl_iby_n * bfact;
                  blst_z = wl_ibz_n * bfact;
                }

                if (std::abs(dsr) < deg_tol) {
                  vrst_y = wr_ivy;
                  vrst_z = wr_ivz;
                  brst_y = wr_iby;
                  brst_z = wr_ibz;
                } else {
                  const Real inv_dsr = 1.0 / dsr;
                  const Real mfact = bxi_n * (sm - wr_ivx) * inv_dsr;
                  const Real bfact = (wr_idn * SQR(sdr) - bxsq_n) * inv_dsr;
                  vrst_y = wr_ivy - wr_iby_n * mfact;
                  vrst_z = wr_ivz - wr_ibz_n * mfact;
                  brst_y = wr_iby_n * bfact;
                  brst_z = wr_ibz_n * bfact;
                }

                const Real vbstl = sm * bxi_n + vlst_y * blst_y + vlst_z * blst_z;
                const Real vbstr = sm * bxi_n + vrst_y * brst_y + vrst_z * brst_z;
                elst = (sdl * el - ptl * wl_ivx + ptst * sm + bxi_n * (vdotBl - vbstl)) /
                       sdml;
                erst = (sdr * er - ptr * wr_ivx + ptst * sm + bxi_n * (vdotBr - vbstr)) /
                       sdmr;

                sal = sm - std::abs(bxi_n) / std::sqrt(dlst);
                sar = sm + std::abs(bxi_n) / std::sqrt(drst);

                const Real sqrtdlst = std::sqrt(dlst);
                const Real sqrtdrst = std::sqrt(drst);
                const Real denom_dst = sqrtdlst + sqrtdrst;
                if (std::abs(denom_dst) <= small || !std::isfinite(denom_dst)) {
                  use_hlle = true;
                } else {
                  const Real sgnbx = (bxi_n >= 0.0) ? 1.0 : -1.0;
                  const Real inv_dst = 1.0 / denom_dst;

                  if (0.5 * bxsq_n < deg_tol) {
                    vdst_y = vlst_y;
                    vdst_z = vlst_z;
                    bdst_y = blst_y;
                    bdst_z = blst_z;
                    eldst = elst;
                    erdst = erst;
                  } else {
                    vdst_y = (sqrtdlst * vlst_y + sqrtdrst * vrst_y +
                              (brst_y - blst_y) * sgnbx) *
                             inv_dst;
                    vdst_z = (sqrtdlst * vlst_z + sqrtdrst * vrst_z +
                              (brst_z - blst_z) * sgnbx) *
                             inv_dst;
                    bdst_y = (sqrtdlst * brst_y + sqrtdrst * blst_y +
                              sqrtdlst * sqrtdrst * (vrst_y - vlst_y) * sgnbx) *
                             inv_dst;
                    bdst_z = (sqrtdlst * brst_z + sqrtdrst * blst_z +
                              sqrtdlst * sqrtdrst * (vrst_z - vlst_z) * sgnbx) *
                             inv_dst;

                    const Real vbdst = sm * bxi_n + vdst_y * bdst_y + vdst_z * bdst_z;
                    eldst = elst - sqrtdlst *
                                       ((sm * bxi_n + vlst_y * blst_y + vlst_z * blst_z) -
                                        vbdst) *
                                       sgnbx;
                    erdst = erst + sqrtdrst *
                                       ((sm * bxi_n + vrst_y * brst_y + vrst_z * brst_z) -
                                        vbdst) *
                                       sgnbx;
                  }

                  use_hlle = !std::isfinite(sal) || !std::isfinite(sar) ||
                             !std::isfinite(elst) || !std::isfinite(erst) ||
                             !std::isfinite(eldst) || !std::isfinite(erdst);
                }
              }
            }
          }

          if (use_hlle) {
            frho = hlle_frho;
            fmx = hlle_fmx;
            fmy = hlle_fmy;
            fmz = hlle_fmz;
            fe = hlle_fe;
            fby = hlle_fby;
            fbz = hlle_fbz;
            pface = hlle_d;
            pmag_face = hlle_db;
          } else if (sl >= 0.0) {
            frho = fl_d;
            fmx = fl_mx;
            fmy = fl_my;
            fmz = fl_mz;
            fe = fl_e;
            fby = fl_by;
            fbz = fl_bz;
            pface = wl_ipr;
            pmag_face = pbl;
          } else if (sal >= 0.0) {
            frho = fl_d + sl * (dlst - wl_idn);
            fmx = fl_mx + sl * (dlst * sm - wl_idn * wl_ivx);
            fmy = fl_my + sl * (dlst * vlst_y - wl_idn * wl_ivy);
            fmz = fl_mz + sl * (dlst * vlst_z - wl_idn * wl_ivz);
            fe = fl_e + sl * (elst - el);
            fby = fl_by + sl * (sqrt_mu0 * blst_y - wl_iby);
            fbz = fl_bz + sl * (sqrt_mu0 * blst_z - wl_ibz);
            pface = ptst - 0.5 * (bxsq_n + SQR(blst_y) + SQR(blst_z));
            pmag_face = 0.5 * (bxsq_n + SQR(blst_y) + SQR(blst_z));
          } else if (sm >= 0.0) {
            frho = fl_d + sl * (dlst - wl_idn);
            fmx = fl_mx + sl * (dlst * sm - wl_idn * wl_ivx);
            fmy = fl_my + sl * (dlst * vlst_y - wl_idn * wl_ivy) +
                  sal * dlst * (vdst_y - vlst_y);
            fmz = fl_mz + sl * (dlst * vlst_z - wl_idn * wl_ivz) +
                  sal * dlst * (vdst_z - vlst_z);
            fe = fl_e + sl * (elst - el) + sal * (eldst - elst);
            fby = fl_by + sl * (sqrt_mu0 * blst_y - wl_iby) +
                  sal * sqrt_mu0 * (bdst_y - blst_y);
            fbz = fl_bz + sl * (sqrt_mu0 * blst_z - wl_ibz) +
                  sal * sqrt_mu0 * (bdst_z - blst_z);
            pface = ptst - 0.5 * (bxsq_n + SQR(bdst_y) + SQR(bdst_z));
            pmag_face = 0.5 * (bxsq_n + SQR(bdst_y) + SQR(bdst_z));
          } else if (sar > 0.0) {
            frho = fr_d + sr * (drst - wr_idn);
            fmx = fr_mx + sr * (drst * sm - wr_idn * wr_ivx);
            fmy = fr_my + sr * (drst * vrst_y - wr_idn * wr_ivy) +
                  sar * drst * (vdst_y - vrst_y);
            fmz = fr_mz + sr * (drst * vrst_z - wr_idn * wr_ivz) +
                  sar * drst * (vdst_z - vrst_z);
            fe = fr_e + sr * (erst - er) + sar * (erdst - erst);
            fby = fr_by + sr * (sqrt_mu0 * brst_y - wr_iby) +
                  sar * sqrt_mu0 * (bdst_y - brst_y);
            fbz = fr_bz + sr * (sqrt_mu0 * brst_z - wr_ibz) +
                  sar * sqrt_mu0 * (bdst_z - brst_z);
            pface = ptst - 0.5 * (bxsq_n + SQR(bdst_y) + SQR(bdst_z));
            pmag_face = 0.5 * (bxsq_n + SQR(bdst_y) + SQR(bdst_z));
          } else if (sr > 0.0) {
            frho = fr_d + sr * (drst - wr_idn);
            fmx = fr_mx + sr * (drst * sm - wr_idn * wr_ivx);
            fmy = fr_my + sr * (drst * vrst_y - wr_idn * wr_ivy);
            fmz = fr_mz + sr * (drst * vrst_z - wr_idn * wr_ivz);
            fe = fr_e + sr * (erst - er);
            fby = fr_by + sr * (sqrt_mu0 * brst_y - wr_iby);
            fbz = fr_bz + sr * (sqrt_mu0 * brst_z - wr_ibz);
            pface = ptst - 0.5 * (bxsq_n + SQR(brst_y) + SQR(brst_z));
            pmag_face = 0.5 * (bxsq_n + SQR(brst_y) + SQR(brst_z));
          } else {
            frho = fr_d;
            fmx = fr_mx;
            fmy = fr_my;
            fmz = fr_mz;
            fe = fr_e;
            fby = fr_by;
            fbz = fr_bz;
            pface = wr_ipr;
            pmag_face = pbr;
          }

          q.flux(b, dir, IDN, k, j, i) = frho;
          q.flux(b, dir, ivx, k, j, i) = fmx;
          q.flux(b, dir, ivy, k, j, i) = fmy;
          q.flux(b, dir, ivz, k, j, i) = fmz;
          p.flux(b, dir, IPR, k, j, i) = pface;
          p.flux(b, dir, IBM, k, j, i) = pmag_face;
          p.flux(b, dir, field::cell::B(IBXG), k, j, i) = 0.0;
          p.flux(b, dir, field::cell::B(IBYG), k, j, i) = fby;
          p.flux(b, dir, field::cell::B(IBZG), k, j, i) = fbz;

          q.flux(b, dir, IEN, k, j, i) = fe;

          // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
          q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
          vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
        });

    // other species are just hllc
    for (int n = 1; n < nspecies; ++n) {
      const int IDN = n;
      const int ivx = nspecies + (n * 3) + ((dir - 1));
      const int ivy = nspecies + (n * 3) + ((dir - 1) + 1) % 3;
      const int ivz = nspecies + (n * 3) + ((dir - 1) + 2) % 3;
      // Unused indices for dust hydrodynamics
      const int IPR = nspecies * 4 + n;
      const int ISE = nspecies * 5 + n;
      const int IBL = nspecies * 6 + n;
      const int IEN = IPR;
      const int IEG = ISE;
      parthenon::par_for_inner(
          DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
            // Create local references for L/R states (helps compiler vectorize)
            Real &wl_idn = wl(IDN, i);
            Real &wl_ivx = wl(ivx, i);
            Real &wl_ivy = wl(ivy, i);
            Real &wl_ivz = wl(ivz, i);
            Real &wl_ipr = wl(IPR, i);
            Real &wl_ise = wl(ISE, i);
            Real &wl_ibl = wl(IBL, i);

            Real &wr_idn = wr(IDN, i);
            Real &wr_ivx = wr(ivx, i);
            Real &wr_ivy = wr(ivy, i);
            Real &wr_ivz = wr(ivz, i);
            Real &wr_ipr = wr(IPR, i);
            Real &wr_ise = wr(ISE, i);
            Real &wr_ibl = wr(IBL, i);

            // Compute middle state estimates with PVRS (Toro 10.5.2)
            // define 6 registers used below
            Real qa, qb, qc, qd, qe, qf;
            qa = wl_ibl / wl_idn;
            qb = wr_ibl / wr_idn;
            Real el = wl_idn * (wl_ise + 0.5 * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz)));
            Real er = wr_idn * (wr_ise + 0.5 * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz)));

            // NOTE(@adempsey)
            // The below choices are taken from Batten et al 1997 and Fleischmann et al
            // 2020 Roe averages
            Real sqrtl = std::sqrt(wl_idn);
            Real sqrtr = std::sqrt(wr_idn);
            const Real isqrt = 1.0 / (sqrtl + sqrtr);
            sqrtl *= isqrt;
            sqrtr *= isqrt;
            const Real vxh = sqrtl * wl_ivx + sqrtr * wr_ivx;
            const Real csh = std::sqrt(sqrtl * qa + sqrtr * qb +
                                       0.5 * sqrtl * sqrtr * SQR(wl_ivx - wr_ivx));
            qa = std::sqrt(qa);
            qb = std::sqrt(qb);

            // Compute the max/min wave speeds based on L/R
            Real sl = std::min(wl_ivx - qa, vxh - csh);
            Real sr = std::max(wr_ivx + qb, vxh + csh);

            // following min/max set to TINY_NUMBER to fix bug found in converging
            // supersonic flow
            qa = sr > 0.0 ? sr : 1.0e-20;  // bp
            qb = sl < 0.0 ? sl : -1.0e-20; // bm

            // Compute the contact wave speed and pressure
            qe = wl_ivx - sl; // vxl
            qf = wr_ivx - sr; // vxr

            qc = wl_ipr + qe * wl_idn * wl_ivx; // tl
            qd = wr_ipr + qf * wr_idn * wr_ivx; // tr

            Real ml = wl_idn * qe;
            Real mr = -(wr_idn * qf);

            // Determine the contact wave speed...
            Real am = (qc - qd) / (ml + mr);
            // ...and the pressure at the contact surface
            Real cp = (ml * qd + mr * qc) / (ml + mr);
            cp = cp > 0.0 ? cp : 0.0;

            // Compute L/R fluxes along the line bm (qb), bp (qa)
            qe = wl_idn * (wl_ivx - qb);
            qf = wr_idn * (wr_ivx - qa);

            Real fld = qe;
            Real frd = qf;
            Real flmx = qe * wl_ivx; // + wl_ipr;
            Real frmx = qf * wr_ivx; // + wr_ipr;
            Real flmy = qe * wl_ivy;
            Real frmy = qf * wr_ivy;
            Real flmz = qe * wl_ivz;
            Real frmz = qf * wr_ivz;
            Real fle = el * (wl_ivx - qb) + wl_ipr * wl_ivx;
            Real fre = er * (wr_ivx - qa) + wr_ipr * wr_ivx;

            // Compute flux weights or scales.  Set an approximate interface pressure for
            // coordinate source terms and pressure contribution to flux.
            if (am >= 0.0) {
              qc = am / (am - qb);
              qd = 0.0;
              qe = -qb / (am - qb);
            } else {
              qc = 0.0;
              qd = -am / (qa - am);
              qe = qa / (qa - am);
            }
            p.flux(b, dir, IPR, k, j, i) = qc * wl_ipr + qd * wr_ipr + qe * cp;

            // Compute the HLLC flux at interface, including weighted contribution of the
            // flux along the contact
            const Real frho = qc * fld + qd * frd;
            q.flux(b, dir, IDN, k, j, i) = frho;
            q.flux(b, dir, ivx, k, j, i) = qc * flmx + qd * frmx; // + qe * cp;
            q.flux(b, dir, ivy, k, j, i) = qc * flmy + qd * frmy;
            q.flux(b, dir, ivz, k, j, i) = qc * flmz + qd * frmz;
            q.flux(b, dir, IEN, k, j, i) = qc * fle + qd * fre + qe * cp * am;

            // Li, 2008, https://ui.adsabs.harvard.edu/abs/2008ASPC..385..273L/abstract
            q.flux(b, dir, IEG, k, j, i) = frho * ((frho >= 0.0) ? wl_ise : wr_ise);
            vf(b, fdir, n, k, j, i) = frho / ((frho >= 0.0) ? wl_idn : wr_idn);
          });
    }
  }
};
} // namespace ArtemisUtils

#endif // UTILS_FLUXES_RIEMANN_HLLD_HPP_
