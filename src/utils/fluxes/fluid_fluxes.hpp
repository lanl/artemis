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
#ifndef UTILS_FLUXES_FLUID_FLUXES_HPP_
#define UTILS_FLUXES_FLUID_FLUXES_HPP_

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "rotating_frame/rotating_frame.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/fluxes/reconstruction/reconstruction.hpp"
#include "utils/fluxes/riemann/riemann.hpp"

using ArtemisUtils::VI;
using parthenon::ScratchPad1D;
using parthenon::ScratchPad2D;

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn  void ArtemisUtils::ScaleMomentumFlux
//! \brief Scales the momentum fluxes by scale factors associated with relevant coord sys
template <Coordinates GEOM, Fluid FLUID_TYPE, int DIR, typename V3>
KOKKOS_INLINE_FUNCTION void ScaleMomentumFlux(parthenon::team_mbr_t const &member,
                                              const int b, const int k, const int j,
                                              const int il, const int iu, const V3 &q) {
  if constexpr (GEOM == Coordinates::cartesian) return;
  PARTHENON_REQUIRE(DIR > 0 && DIR <= 3, "Invalid flux direction!");

  // Obtain number of species
  int nvar = Null<int>();
  if constexpr (FLUID_TYPE == Fluid::gas) {
    nvar = 6;
  } else if constexpr (FLUID_TYPE == Fluid::dust) {
    nvar = 4;
  }
  const int nspecies = q.GetMaxNumberOfVars() / nvar;

  // Scale the Momentum Flux in the DIR direction
  for (int n = 0; n < nspecies; ++n) {
    const int IVX = nspecies + VI(n, 0);
    const int IVY = nspecies + VI(n, 1);
    const int IVZ = nspecies + VI(n, 2);
    parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          geometry::Coords<GEOM> coords(q.GetCoordinates(b), k, j, i);
          auto xf = NewArray<Real, 3>();
          if constexpr (DIR == X1DIR) {
            xf = coords.FaceCenX1(geometry::CellFace::lower);
          } else if constexpr (DIR == X2DIR) {
            xf = coords.FaceCenX2(geometry::CellFace::lower);
          } else if constexpr (DIR == X3DIR) {
            xf = coords.FaceCenX3(geometry::CellFace::lower);
          }
          q.flux(b, DIR, IVX, k, j, i) *= coords.hx1(xf[0], xf[1], xf[2]);
          q.flux(b, DIR, IVY, k, j, i) *= coords.hx2(xf[0], xf[1], xf[2]);
          q.flux(b, DIR, IVZ, k, j, i) *= coords.hx3(xf[0], xf[1], xf[2]);
        });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::CalculateFluxesImpl
//! \brief Calculate hydrodynamic fluxes from reconstructed primitive variables.
//! NOTE(PDM): flux kernel largely borrowed from AthenaPK/Parthenon-Hydro/AthenaK
template <Coordinates GEOM, Fluid FLUID_TYPE, RSolver RIEMANN, ReconstructionMethod RECON,
          typename PackPrim, typename PackFlux, typename PackFace, typename PKG>
TaskStatus CalculateFluxesImpl(MeshData<Real> *md, PKG &pkg, PackPrim vprim,
                               PackFlux vflux, PackFace vface) {
  using parthenon::MakePackDescriptor;

  auto pm = md->GetParentPointer();
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int ncells1 = (ib.e - ib.s + 1) + 2 * parthenon::Globals::nghost; 
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  auto &artemis_pkg = pm->packages.Get("artemis");
  const auto do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  auto &gas_pkg = pm->packages.Get("gas");
  const auto tvd_type = gas_pkg->template Param<TVDType>("tvd_type");

  // YH: for positive preserving scheme
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index"); 
  const double dt = gas_pkg->Param<double>("track_dt");
  const Real dW_idn = gas_pkg->template Param<Real>("dW_idn");
  const Real dW_ipr = gas_pkg->template Param<Real>("dW_ipr");
  const Real dW_ise = gas_pkg->template Param<Real>("dW_ise");
  const Real dW_ivx = gas_pkg->template Param<Real>("dW_ivx");
  const Real dW_ivy = gas_pkg->template Param<Real>("dW_ivy");
  const Real dW_ivz = gas_pkg->template Param<Real>("dW_ivz");
  const Real dW_ibx = gas_pkg->template Param<Real>("dW_ibx");
  const Real dW_iby = gas_pkg->template Param<Real>("dW_iby");
  const Real dW_ibz = gas_pkg->template Param<Real>("dW_ibz");
  const Real dW_iJx = gas_pkg->template Param<Real>("dW_iJx");
  const Real dW_iJy = gas_pkg->template Param<Real>("dW_iJy");
  const Real dW_iJz = gas_pkg->template Param<Real>("dW_iJz");
  const Real dW_iPe = gas_pkg->template Param<Real>("dW_iPe");

  // Adiabatic index, if used
  EOS eos;
  if constexpr (FLUID_TYPE == Fluid::gas) {
    eos = pkg->template Param<EOS>("eos_d");
  }

  // Scratch properties
  // NOTE(PDM): Scratch here must be able to contain up to the total number of species,
  // even if some blocks don't contain all species
  const int nspecies = pkg->template Param<int>("nspecies");
  const int nvars = vprim.GetMaxNumberOfVars();
  const ReconstructionMethod recon_method =
      pkg->template Param<ReconstructionMethod>("recon");
  const bool applyPP = (recon_method == ReconstructionMethod::plm_pp);
  int sz = applyPP ? 3 : 2; // YH: Need store dW for PP
  int scr_size = ScratchPad2D<Real>::shmem_size(nvars, ncells1) * sz;
  const int scr_level = pkg->template Param<int>("scr_level");
  // X1-Flux
  // -> For some reason my code need -2 (otherwise NaN) & +2 (otherwise oscillation at right end), thus, I need to use 3 ghost cells for now. I need to figure out which part of my code (probably interpolation) requires this additional flux computation at boundaries.
  int il = ib.s - 2, iu = ib.e + 2;
  int jl = jb.s, ju = jb.e;
  if (multi_d) {jl -= 2; ju += 2;} // YH: to account for fcc bfield in CT 
  int kl = kb.s, ku = kb.e;
  if (three_d) {kl -= 2; ku += 2;}
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "CalculateFluxes::X1-Flux", DevExecSpace(), scr_size,
      scr_level, 0, md->NumBlocks() - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j) {
        ScratchPad2D<Real> wl(mbr.team_scratch(scr_level), nvars, ncells1);
        ScratchPad2D<Real> wr(mbr.team_scratch(scr_level), nvars, ncells1);

        // Reconstruct qR[i] and qL[i+1]
        Reconstruction<RECON, X1DIR, GEOM> recon;
	if constexpr (RECON == ReconstructionMethod::plm_pp) {
	  ScratchPad2D<Real> dw(mbr.team_scratch(scr_level), nvars, ncells1);
	  const Real dW_sw[16] = {dW_idn,dW_ivx,dW_ivy,dW_ivz,dW_ipr,dW_ise,
                          dW_ibx,dW_iby,dW_ibz,1,1,1,dW_iJx,dW_iJy,dW_iJz,dW_iPe};
	  recon.apply_pp(mbr, b, k, j, il, iu, vprim, wl, wr, dw, gamma, 
			 dt, tvd_type,dW_sw);
	} else {
          recon.apply(mbr, b, k, j, il, iu, vprim, wl, wr, tvd_type);
	}
	// YH: Add wl(il) as may need it for flux which is used for my XMHD where the v0_B.flux(i=1) from CT
        //     is used through interpolation... -> For now temp fix to avoid nan at boundary
        for (int nv=0; nv<nvars; ++nv) {
          wl(nv,il) = wl(nv,il+1); // YH: cause recon update ql(i+1) & qr(i)
        }
	mbr.team_barrier();
	if (do_mhd) {
	  Reconstruction<ReconstructionMethod::Bcorrection, X1DIR, GEOM> recon_mhd;
	  recon_mhd.apply_fcc(mbr, b, k, j, il, iu, vprim, wl, wr, vface);
	} 
        mbr.team_barrier();

        // Compute fluxes over[is, ie + 1]
        RiemannSolver<RIEMANN, FLUID_TYPE> riemann; 
        riemann.solve(eos, mbr, b, k, j, il, iu, X1DIR, wl, wr, vprim, vflux, vface, do_mhd);
        mbr.team_barrier();

        // Scale X1-momentum flux by appropriate scale factor for coord system
        ScaleMomentumFlux<GEOM, FLUID_TYPE, X1DIR>(mbr, b, k, j, il, iu, vflux);
      });

  // X2-Flux
  if (multi_d) {
    jl = jb.s - 2, ju = jb.e + 2;
    il = ib.s - 2, iu = ib.e + 2;
    kl = kb.s, ku = kb.e;
    if (three_d) {kl -= 2; ku += 2;}
    sz = applyPP ? 4 : 3; // Need store dW for PP
    scr_size = ScratchPad2D<Real>::shmem_size(nvars, ncells1) * sz;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "CalculateFluxes::X2-Flux", DevExecSpace(), scr_size,
        scr_level, 0, md->NumBlocks() - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k) {
          ScratchPad2D<Real> scr1(mbr.team_scratch(scr_level), nvars, ncells1);
          ScratchPad2D<Real> scr2(mbr.team_scratch(scr_level), nvars, ncells1);
          ScratchPad2D<Real> scr3(mbr.team_scratch(scr_level), nvars, ncells1);

          for (int j = jl; j <= ju; ++j) {
            // Permute scratch arrays.
            auto wl = scr1;
            auto wl_jp1 = scr2;
            auto wr = scr3;
            if ((j % 2) == 0) {
              wl = scr2;
              wl_jp1 = scr1;
            }

            // Reconstruct qR[j] and qL[j+1]
            Reconstruction<RECON, X2DIR, GEOM> recon;
	    if constexpr (RECON == ReconstructionMethod::plm_pp) {
	      ScratchPad2D<Real> dw(mbr.team_scratch(scr_level), nvars, ncells1);
	      const Real dW_sw[16] = {dW_idn,dW_ivx,dW_ivy,dW_ivz,dW_ipr,dW_ise,
                          dW_ibx,dW_iby,dW_ibz,1,1,1,dW_iJx,dW_iJy,dW_iJz,dW_iPe};
	      recon.apply_pp(mbr, b, k, j, il, iu, vprim, wl_jp1, wr, dw, gamma, 
			     dt, tvd_type,dW_sw);
	    } else {
              recon.apply(mbr, b, k, j, il, iu, vprim, wl_jp1, wr, tvd_type);
	    }
	    if (j==jl) {
	      for (int nv=0; nv<nvars; ++nv) {
                wl(nv,il) = wl_jp1(nv,il); // YH: cause recon update ql(i+1) & qr(i)
              }
	    }
	    mbr.team_barrier();
	    if (do_mhd) {
	      Reconstruction<ReconstructionMethod::Bcorrection, X2DIR, GEOM> recon_mhd;
	      recon_mhd.apply_fcc(mbr, b, k, j, il, iu, vprim, wl, wr, vface);
	    }
            mbr.team_barrier();

            if (j > jl) { 
              // compute fluxes over [js,je+1]
              RiemannSolver<RIEMANN, FLUID_TYPE> riemann;
              riemann.solve(eos, mbr, b, k, j, il, iu, X2DIR, wl, wr, vprim, vflux,
                            vface, do_mhd);
              mbr.team_barrier();

              // Scale X2-momentum flux by appropriate scale factor for coord system
              ScaleMomentumFlux<GEOM, FLUID_TYPE, X2DIR>(mbr, b, k, j, il, iu, vflux);
            }
          }
        });
  }

  // X3-Flux
  if (three_d) {
    kl = kb.s - 2, ku = kb.e + 2;
    il = ib.s - 2, iu = ib.e + 2; 
    jl = jb.s - 2, ju = jb.e + 2;
    sz = applyPP ? 4 : 3; // Need store dW for PP
    scr_size = ScratchPad2D<Real>::shmem_size(nvars, ncells1) * sz;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Hydro::X3-Flux", DevExecSpace(), scr_size, scr_level,
        0, md->NumBlocks() - 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int j) {
          ScratchPad2D<Real> scr1(mbr.team_scratch(scr_level), nvars, ncells1);
          ScratchPad2D<Real> scr2(mbr.team_scratch(scr_level), nvars, ncells1);
          ScratchPad2D<Real> scr3(mbr.team_scratch(scr_level), nvars, ncells1);

          for (int k = kl; k <= ku; ++k) {
            // Permute scratch arrays.
            auto wl = scr1;
            auto wl_kp1 = scr2;
            auto wr = scr3;
            if ((k % 2) == 0) {
              wl = scr2;
              wl_kp1 = scr1;
            }

            // Reconstruct qR[k] and qL[k+1]
            Reconstruction<RECON, X3DIR, GEOM> recon;
	    if constexpr (RECON == ReconstructionMethod::plm_pp) {
	      ScratchPad2D<Real> dw(mbr.team_scratch(scr_level), nvars, ncells1);
	      const Real dW_sw[16] = {dW_idn,dW_ivx,dW_ivy,dW_ivz,dW_ipr,dW_ise,
                          dW_ibx,dW_iby,dW_ibz,1,1,1,dW_iJx,dW_iJy,dW_iJz,dW_iPe};
	      recon.apply_pp(mbr, b, k, j, il, iu, vprim, wl_kp1, wr, dw, gamma, 
			     dt, tvd_type,dW_sw);
	    } else {
              recon.apply(mbr, b, k, j, il, iu, vprim, wl_kp1, wr, tvd_type);
	    }
	    if (k==kl) {
              for (int nv=0; nv<nvars; ++nv) {
                wl(nv,il) = wl_kp1(nv,il); // YH: cause recon update ql(i+1) & qr(i)
              }
            }
	    mbr.team_barrier();
	    if (do_mhd) {
	      Reconstruction<ReconstructionMethod::Bcorrection, X3DIR, GEOM> recon_mhd;
	      recon_mhd.apply_fcc(mbr, b, k, j, il, iu, vprim, wl, wr, vface);
	    }
            mbr.team_barrier();

            // compute fluxes over [ks,ke+1]
            if (k > kl) { 
              RiemannSolver<RIEMANN, FLUID_TYPE> riemann;
              riemann.solve(eos, mbr, b, k, j, il, iu, X3DIR, wl, wr, vprim, vflux,
                            vface, do_mhd);
              mbr.team_barrier();

              // Scale X3-momentum flux by appropriate scale factor for coord system
              ScaleMomentumFlux<GEOM, FLUID_TYPE, X3DIR>(mbr, b, k, j, il, iu, vflux);
            }
          }
        });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::CalculateFluxesReconSelect
//! \brief Dispatch templated function depending on runtime reconstruction option.
template <Coordinates GEOM, Fluid FLUID_TYPE, RSolver RIEMANN, typename PackPrim,
          typename PackFlux, typename PackFace, typename PKG>
TaskStatus CalculateFluxesReconSelect(MeshData<Real> *md, PKG &pkg, PackPrim vprim,
                                      PackFlux vflux, PackFace vface, const bool pcm) { 
  const ReconstructionMethod recon_method =
      pkg->template Param<ReconstructionMethod>("recon");

  if ((recon_method == ReconstructionMethod::pcm) || (pcm)) {
    return CalculateFluxesImpl<GEOM, FLUID_TYPE, RIEMANN, ReconstructionMethod::pcm>(
        md, pkg, vprim, vflux, vface);
  } else if (recon_method == ReconstructionMethod::plm) {
    return CalculateFluxesImpl<GEOM, FLUID_TYPE, RIEMANN, ReconstructionMethod::plm>(
        md, pkg, vprim, vflux, vface);
  } else if (recon_method == ReconstructionMethod::ppm) {
    return CalculateFluxesImpl<GEOM, FLUID_TYPE, RIEMANN, ReconstructionMethod::ppm>(
        md, pkg, vprim, vflux, vface);
  } else if (recon_method == ReconstructionMethod::plm_rho) {
    return CalculateFluxesImpl<GEOM, FLUID_TYPE, RIEMANN, ReconstructionMethod::plm_rho>(
        md, pkg, vprim, vflux, vface);
  } else if (recon_method == ReconstructionMethod::plm_pp) {
    return CalculateFluxesImpl<GEOM, FLUID_TYPE, RIEMANN, ReconstructionMethod::plm_pp>(
        md, pkg, vprim, vflux, vface);
  } else if (recon_method == ReconstructionMethod::plm_modPe) {
    return CalculateFluxesImpl<GEOM, FLUID_TYPE, RIEMANN, ReconstructionMethod::plm_modPe>(
        md, pkg, vprim, vflux, vface);
  } else {
    PARTHENON_FAIL("Reconstruction method not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::CalculateFluxesRiemannSelect
//! \brief Dispatch templated function depending on runtime Riemann solver option.
template <Coordinates GEOM, Fluid FLUID_TYPE, typename PackPrim, typename PackFlux,
          typename PackFace, typename PKG>
TaskStatus CalculateFluxesRiemannSelect(MeshData<Real> *md, PKG &pkg, PackPrim vprim,
                                        PackFlux vflux, PackFace vface, const bool pcm) { 
  const RSolver riemann_method = pkg->template Param<RSolver>("rsolver");

  if (riemann_method == RSolver::hllc) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hllc>(md, pkg, vprim,
                                                                       vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlle) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlle>(md, pkg, vprim,
                                                                       vflux, vface, pcm);
  } else if (riemann_method == RSolver::llf) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::llf>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlld) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlld>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::llf_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::llf_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlld_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlld_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::llf_hall_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::llf_hall_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hll_hall_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hll_hall_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlle_hall_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlle_hall_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlldc_llf_hall_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlldc_llf_hall_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlldc_hall_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlldc_hall_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlldc_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlldc_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else if (riemann_method == RSolver::hlldc_llf_xmhd) {
    return CalculateFluxesReconSelect<GEOM, FLUID_TYPE, RSolver::hlldc_llf_xmhd>(md, pkg, vprim,
                                                                      vflux, vface, pcm);
  } else {
    PARTHENON_FAIL("Riemann solver not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::CalculateFluxes
//! \brief Hierarchically dispatch templated function depending on runtime coord system
template <Fluid FLUID_TYPE, typename PackPrim, typename PackFlux, typename PackFace,
          typename PKG>
TaskStatus CalculateFluxes(MeshData<Real> *md, PKG &pkg, PackPrim vprim, PackFlux vflux,
                           PackFace vface, const bool pcm) { 
  const Coordinates sys = pkg->template Param<Coordinates>("coords");

  if (sys == Coordinates::cartesian) {
    return CalculateFluxesRiemannSelect<Coordinates::cartesian, FLUID_TYPE>(
        md, pkg, vprim, vflux, vface, pcm);
  } else if (sys == Coordinates::spherical3D) {
    return CalculateFluxesRiemannSelect<Coordinates::spherical3D, FLUID_TYPE>(
        md, pkg, vprim, vflux, vface, pcm);
  } else if (sys == Coordinates::spherical1D) {
    return CalculateFluxesRiemannSelect<Coordinates::spherical1D, FLUID_TYPE>(
        md, pkg, vprim, vflux, vface, pcm);
  } else if (sys == Coordinates::spherical2D) {
    return CalculateFluxesRiemannSelect<Coordinates::spherical2D, FLUID_TYPE>(
        md, pkg, vprim, vflux, vface, pcm);
  } else if (sys == Coordinates::cylindrical) {
    return CalculateFluxesRiemannSelect<Coordinates::cylindrical, FLUID_TYPE>(
        md, pkg, vprim, vflux, vface, pcm);
  } else if (sys == Coordinates::axisymmetric) {
    return CalculateFluxesRiemannSelect<Coordinates::axisymmetric, FLUID_TYPE>(
        md, pkg, vprim, vflux, vface, pcm);
  } else {
    PARTHENON_FAIL("Coordinate type not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::FluxSourceImpl
//!  \brief Adds the pressure gradient force, PdV work term, and geometric source terms
//!           <1/h_k * dh_k/dxi>  (rho*v_i^2 + P)
template <Coordinates GEOM, Fluid FLUID_TYPE, typename PackPrim, typename PackCons,
          typename PackFace, typename PKG>
TaskStatus FluxSourceImpl(MeshData<Real> *md, PKG &pkg, PackPrim vprim, PackCons vcons,
                          PackFace vface, const Real omf, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;
  const bool multi_d = (ndim >= 2);
  const bool three_d = (ndim == 3);
  const bool x1dep = geometry::x1dep<GEOM>();
  const bool x2dep = (geometry::x2dep<GEOM>()) && (multi_d);
  const bool x3dep = (geometry::x3dep<GEOM>()) && (three_d);

  // Obtain number of species
  int nvar = Null<int>();
  if constexpr (FLUID_TYPE == Fluid::gas) {
    nvar = 5;
  } else if constexpr (FLUID_TYPE == Fluid::dust) {
    nvar = 4;
  }
  const int nspecies = vprim.GetMaxNumberOfVars() / nvar;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GeometricSourceTerms", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s - 2, ib.e + 1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vprim.GetCoordinates(b), k, j, i);
        const auto dhdx1 = (x1dep) ? coords.GetConnX1() : NewArray<Real, 3>(0.0);
        const auto dhdx2 = (x2dep) ? coords.GetConnX2() : NewArray<Real, 3>(0.0);
        const auto dhdx3 = (x3dep) ? coords.GetConnX3() : NewArray<Real, 3>(0.0);

        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = (multi_d) ? coords.GetFaceAreaX2() : NewArray<Real, 2>(0.0);
        const auto ax3 = (three_d) ? coords.GetFaceAreaX3() : NewArray<Real, 2>(0.0);

        const Real vol = coords.Volume();
        geometry::BBox bnds = coords.bnds;
        const Real dx[3] = {bnds.x1[1] - bnds.x1[0], bnds.x2[1] - bnds.x2[0],
                            bnds.x3[1] - bnds.x3[0]};

        const auto &xv = coords.GetCellCenter();

        // Get the rotational velocity
        const auto &vf = RotatingFrame::RotationVelocity<GEOM>(xv, omf);

        for (int n = 0; n < nspecies; ++n) {
          const int IMX = VI(n, 0);
          const int IMY = VI(n, 1);
          const int IMZ = VI(n, 2);
          const int IVX = nspecies + IMX;
          const int IVY = nspecies + IMY;
          const int IVZ = nspecies + IMZ;

          // Add the pressure gradient and PdV term
          [[maybe_unused]] auto &vprim_ = vprim;
          [[maybe_unused]] auto &vcons_ = vcons;
          [[maybe_unused]] auto &vface_ = vface;
          [[maybe_unused]] auto &dt_ = dt;
          [[maybe_unused]] auto &multi_d_ = multi_d;
          [[maybe_unused]] auto &three_d_ = three_d;
          if constexpr (FLUID_TYPE == Fluid::gas) {
            const int IEG = nspecies * 3 + n;
            const int IPR = nspecies * 4 + n;
            vcons_(b, IMX, k, j, i) += dt_ / dx[0] *
                                       (vprim_.flux(b, X1DIR, IPR, k, j, i) -
                                        vprim_.flux(b, X1DIR, IPR, k, j, i + 1));
            vcons_(b, IEG, k, j, i) -= dt_ / vol * 0.5 *
                                       (vprim_.flux(b, X1DIR, IPR, k, j, i) +
                                        vprim_.flux(b, X1DIR, IPR, k, j, i + 1)) *
                                       (ax1[1] * vface_(b, TE::F1, n, k, j, i + 1) -
                                        ax1[0] * vface_(b, TE::F1, n, k, j, i));
            if (multi_d_) {
              vcons_(b, IMY, k, j, i) += dt_ / dx[1] *
                                         (vprim_.flux(b, X2DIR, IPR, k, j, i) -
                                          vprim_.flux(b, X2DIR, IPR, k, j + 1, i));
              vcons_(b, IEG, k, j, i) -= dt_ / vol * 0.5 *
                                         (vprim_.flux(b, X2DIR, IPR, k, j, i) +
                                          vprim_.flux(b, X2DIR, IPR, k, j + 1, i)) *
                                         (ax2[1] * vface_(b, TE::F2, n, k, j + 1, i) -
                                          ax2[0] * vface_(b, TE::F2, n, k, j, i));
            }
            if (three_d_) {
              vcons_(b, IMZ, k, j, i) += dt_ / dx[2] *
                                         (vprim_.flux(b, X3DIR, IPR, k, j, i) -
                                          vprim_.flux(b, X3DIR, IPR, k + 1, j, i));
              vcons_(b, IEG, k, j, i) -= dt_ / vol * 0.5 *
                                         (vprim_.flux(b, X3DIR, IPR, k, j, i) +
                                          vprim_.flux(b, X3DIR, IPR, k + 1, j, i)) *
                                         (ax3[1] * vface_(b, TE::F3, n, k + 1, j, i) -
                                          ax3[0] * vface_(b, TE::F3, n, k, j, i));
            }
          }

          // Add coordinate source term
          const Real &dens = vprim(b, n, k, j, i);
          const Real rdt = dens * dt;
          if (x1dep) {
            vcons(b, IMX, k, j, i) +=
                rdt * (dhdx1[0] * SQR(vprim(b, IVX, k, j, i) + vf[0]) +
                       dhdx1[1] * SQR(vprim(b, IVY, k, j, i) + vf[1]) +
                       dhdx1[2] * SQR(vprim(b, IVZ, k, j, i) + vf[2]));
          }
          if (x2dep) {
            vcons(b, IMY, k, j, i) +=
                rdt * (dhdx2[0] * SQR(vprim(b, IVX, k, j, i) + vf[0]) +
                       dhdx2[1] * SQR(vprim(b, IVY, k, j, i) + vf[1]) +
                       dhdx2[2] * SQR(vprim(b, IVZ, k, j, i) + vf[2]));
          }
          if (x3dep) {
            vcons(b, IMZ, k, j, i) +=
                rdt * (dhdx3[0] * SQR(vprim(b, IVX, k, j, i) + vf[0]) +
                       dhdx3[1] * SQR(vprim(b, IVY, k, j, i) + vf[1]) +
                       dhdx3[2] * SQR(vprim(b, IVZ, k, j, i) + vf[2]));
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::FluxSourceGeomSelect
//! \brief Dispatch templated function depending on runtime coordinate system.
template <Fluid FLUID_TYPE, typename PackPrim, typename PackCons, typename PackFace,
          typename PKG>
TaskStatus FluxSourceGeomSelect(MeshData<Real> *md, PKG &pkg, PackPrim vprim,
                                PackCons vcons, PackFace vface, const Real dt) {
  typedef Coordinates C;
  const C sys = pkg->template Param<C>("coords");
  auto pm = md->GetParentPointer();
  auto &artemis_pkg = pm->packages.Get("artemis");
  Real omf = 0.0;
  if (artemis_pkg->template Param<bool>("do_rotating_frame")) {
    auto &rf_pkg = pm->packages.Get("rotating_frame");
    omf = rf_pkg->template Param<Real>("omega");
  }

  if (sys == C::cartesian) {
    return FluxSourceImpl<C::cartesian, FLUID_TYPE>(md, pkg, vprim, vcons, vface, omf,
                                                    dt);
  } else if (sys == C::spherical3D) {
    return FluxSourceImpl<C::spherical3D, FLUID_TYPE>(md, pkg, vprim, vcons, vface, omf,
                                                      dt);
  } else if (sys == C::spherical1D) {
    return FluxSourceImpl<C::spherical1D, FLUID_TYPE>(md, pkg, vprim, vcons, vface, omf,
                                                      dt);
  } else if (sys == C::spherical2D) {
    return FluxSourceImpl<C::spherical2D, FLUID_TYPE>(md, pkg, vprim, vcons, vface, omf,
                                                      dt);
  } else if (sys == C::cylindrical) {
    return FluxSourceImpl<C::cylindrical, FLUID_TYPE>(md, pkg, vprim, vcons, vface, omf,
                                                      dt);
  } else if (sys == C::axisymmetric) {
    return FluxSourceImpl<C::axisymmetric, FLUID_TYPE>(md, pkg, vprim, vcons, vface, omf,
                                                       dt);
  } else {
    PARTHENON_FAIL("Coordinate type not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::FluxSource
//! \brief Dispatch templated function depending on fluid type.
template <typename PackPrim, typename PackCons, typename PackFace, typename PKG>
TaskStatus FluxSource(MeshData<Real> *md, PKG &pkg, PackPrim vprim, PackCons vcons,
                      PackFace vface, const Real dt) {
  const Fluid fluid_type = pkg->template Param<Fluid>("fluid_type");
  if (fluid_type == Fluid::gas) {
    return FluxSourceGeomSelect<Fluid::gas>(md, pkg, vprim, vcons, vface, dt);
  } else if (fluid_type == Fluid::dust) {
    return FluxSourceGeomSelect<Fluid::dust>(md, pkg, vprim, vcons, vface, dt);
  } else {
    PARTHENON_FAIL("Fluid type not recognized!");
  }
}

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_FLUID_FLUXES_HPP_
