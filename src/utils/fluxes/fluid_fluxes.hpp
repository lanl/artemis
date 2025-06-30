//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
using parthenon::MakePackDescriptor;
using parthenon::ScratchPad1D;
using parthenon::ScratchPad2D;
using TE = parthenon::TopologicalElement;

namespace ArtemisUtils {
//----------------------------------------------------------------------------------------
//! \fn  void ArtemisUtils::ScaleMomentumFlux
//! \brief Scales the momentum fluxes by scale factors associated with relevant coord sys
template <Coordinates G, Fluid F, int DIR, typename V3>
KOKKOS_INLINE_FUNCTION void ScaleMomentumFlux(parthenon::team_mbr_t const &member,
                                              const int b, const int k, const int j,
                                              const int il, const int iu, const V3 &q) {
  // Immediately return if Cartesian (i.e., do not scale momentum flux)
  if constexpr (G == Coordinates::cartesian) return;
  PARTHENON_REQUIRE(DIR > 0 && DIR <= 3, "Invalid flux direction!");

  // Obtain number of species
  int nspecies = Null<int>();
  if constexpr (F == Fluid::gas) {
    nspecies = q.GetSize(b, gas::cons::density());
  } else if constexpr (F == Fluid::dust) {
    nspecies = q.GetSize(b, dust::cons::density());
  } else if constexpr (F == Fluid::radiation) {
    nspecies = q.GetSize(b, rad::cons::energy());
  }

  // Scale the Momentum Flux in the DIR direction
  for (int n = 0; n < nspecies; ++n) {
    const int IVX = nspecies + VI(n, 0);
    const int IVY = nspecies + VI(n, 1);
    const int IVZ = nspecies + VI(n, 2);
    parthenon::par_for_inner(
        DEFAULT_INNER_LOOP_PATTERN, member, il, iu, [&](const int i) {
          geometry::Coords<G> coords(q.GetCoordinates(b), k, j, i);
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
template <Coordinates G, Fluid F, Closure C, RSolver RIEMANN, ReconstructionMethod RECON,
          typename PKG, typename PRIM, typename FLUX, typename FACE>
TaskStatus CalculateFluxesImpl(MeshData<Real> *md, PKG &pkg, PRIM vp, FLUX vflx,
                               FACE vface) {
  auto pm = md->GetParentPointer();

  // Bounds and indexing
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int ncells1 = (ib.e - ib.s + 1) + 2 * parthenon::Globals::nghost;
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  // Adiabatic index, if used
  EOS eos;
  if constexpr (F == Fluid::gas) {
    eos = pkg->template Param<EOS>("eos_d");
  }

  // Speed of light (and reduced), if used
  Real chat = Null<Real>();
  Real c = Null<Real>();
  if constexpr (F == Fluid::radiation) {
    chat = pkg->template Param<Real>("chat");
    c = pkg->template Param<Real>("c");
  }

  // Scratch properties
  // NOTE(PDM): Scratch here must be able to contain up to the total number of species,
  // even if some blocks don't contain all species
  const int nvars = vp.GetMaxNumberOfVars();
  const int scr_level = pkg->template Param<int>("scr_level");
  int scr_size = ScratchPad2D<Real>::shmem_size(nvars, ncells1) * 2;

  // X1-Flux
  int il = ib.s, iu = ib.e + 1;
  int jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "CalculateFluxes::X1-Flux", DevExecSpace(), scr_size,
      scr_level, 0, md->NumBlocks() - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int b, const int k, const int j) {
        ScratchPad2D<Real> wl(mbr.team_scratch(scr_level), nvars, ncells1);
        ScratchPad2D<Real> wr(mbr.team_scratch(scr_level), nvars, ncells1);

        // Reconstruct qR[i] and qL[i+1]
        Reconstruction<RECON, X1DIR, G> recon;
        recon(mbr, b, k, j, il - 1, iu, vp, wl, wr);
        mbr.team_barrier();

        // NOTE(@adempsey): Moments radiation currently requires zeroing of reconstructed
        // fluxes when ~round-off.  Future work may invoke PCM fallbacks.  Such a utility
        // could also be employed for e.g., dust, FOFC, etc...
        if constexpr (F == Fluid::radiation) {
          correct_recon(mbr, X1DIR, b, k, j, il - 1, iu, vp, wl, wr);
          mbr.team_barrier();
        }

        // Compute fluxes over[is, ie + 1]
        RiemannSolver<RIEMANN, F, C> riemann;
        riemann(eos, c, chat, mbr, b, k, j, il, iu, X1DIR, wl, wr, vp, vflx, vface);
        mbr.team_barrier();

        // Scale X1-momentum flux by appropriate scale factor for coord system
        ScaleMomentumFlux<G, F, X1DIR>(mbr, b, k, j, il, iu, vflx);
      });

  // X2-Flux
  if (multi_d) {
    jl = jb.s - 1, ju = jb.e + 1;
    il = ib.s, iu = ib.e, kl = kb.s, ku = kb.e;
    scr_size = ScratchPad2D<Real>::shmem_size(nvars, ncells1) * 3;
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
            Reconstruction<RECON, X2DIR, G> recon;
            recon(mbr, b, k, j, il, iu, vp, wl_jp1, wr);
            mbr.team_barrier();

            // NOTE(@adempsey): See comments above
            if constexpr (F == Fluid::radiation) {
              correct_recon(mbr, X2DIR, b, k, j, il, iu, vp, wl_jp1, wr);
              mbr.team_barrier();
            }

            if (j > jl) {
              // compute fluxes over [js,je+1]
              RiemannSolver<RIEMANN, F, C> riemann;
              riemann(eos, c, chat, mbr, b, k, j, il, iu, X2DIR, wl, wr, vp, vflx, vface);
              mbr.team_barrier();

              // Scale X2-momentum flux by appropriate scale factor for coord system
              ScaleMomentumFlux<G, F, X2DIR>(mbr, b, k, j, il, iu, vflx);
            }
          }
        });
  }

  // X3-Flux
  if (three_d) {
    kl = kb.s - 1, ku = kb.e + 1;
    il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e;
    scr_size = ScratchPad2D<Real>::shmem_size(nvars, ncells1) * 3;
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
            Reconstruction<RECON, X3DIR, G> recon;
            recon(mbr, b, k, j, il, iu, vp, wl_kp1, wr);
            mbr.team_barrier();

            // NOTE(@adempsey): See comments above
            if constexpr (F == Fluid::radiation) {
              correct_recon(mbr, X3DIR, b, k, j, il, iu, vp, wl_kp1, wr);
              mbr.team_barrier();
            }

            // compute fluxes over [ks,ke+1]
            if (k > kl) {
              RiemannSolver<RIEMANN, F, C> riemann;
              riemann(eos, c, chat, mbr, b, k, j, il, iu, X3DIR, wl, wr, vp, vflx, vface);
              mbr.team_barrier();

              // Scale X3-momentum flux by appropriate scale factor for coord system
              ScaleMomentumFlux<G, F, X3DIR>(mbr, b, k, j, il, iu, vflx);
            }
          }
        });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::FluxSourceImpl
//!  \brief Adds source terms "affiliated with the flux", e.g.,
//!         - the pressure gradient force (including for radiation moments)
//!         - the PdV work term
//!         - the geometric source terms (be wary of rotating frame...)
//!           <1/h_k * dh_k/dxi>  (rho*v_i^2 + P)
template <Coordinates G, Fluid F, Closure C, typename PKG, typename PRIM, typename CONS,
          typename FACE>
TaskStatus FluxSourceImpl(MeshData<Real> *md, PKG &pkg, PRIM vp, CONS vcons, FACE vface,
                          const Real omf, const Real dt) {
  // Indexing and geometry
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = md->GetParentPointer()->ndim;
  const bool multi_d = (ndim >= 2);
  const bool three_d = (ndim == 3);
  const bool x1dep = geometry::x1dep<G>();
  const bool x2dep = (geometry::x2dep<G>() && multi_d);
  const bool x3dep = (geometry::x3dep<G>() && three_d);

  // Extract (reduced, weighted) speed of light
  Real hcchat = Null<Real>();
  if constexpr (F == Fluid::radiation) {
    hcchat = 0.5 * pkg->template Param<Real>("c") * pkg->template Param<Real>("chat");
  }

  // Apply flux sources
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FluxSourceTerms", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<G> coords(vp.GetCoordinates(b), k, j, i);
        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = (multi_d) ? coords.GetFaceAreaX2() : NewArray<Real, 2>(0.0);
        const auto ax3 = (three_d) ? coords.GetFaceAreaX3() : NewArray<Real, 2>(0.0);
        const auto dh1 = (x1dep) ? coords.GetConnX1() : NewArray<Real, 3>(0.0);
        const auto dh2 = (x2dep) ? coords.GetConnX2() : NewArray<Real, 3>(0.0);
        const auto dh3 = (x3dep) ? coords.GetConnX3() : NewArray<Real, 3>(0.0);

        // Get the rotational velocity
        std::array<Real, 3> rfv{0.0};
        [[maybe_unused]] Real omf_ = omf;
        if constexpr (F != Fluid::radiation) {
          rfv = RotatingFrame::RotationVelocity<G>(coords.GetCellCenter(), omf_);
        }

        // Timestep weighted by dx
        geometry::BBox bnds = coords.bnds;
        const std::array<Real, 3> dtdx = {dt / (bnds.x1[1] - bnds.x1[0]),
                                          multi_d * dt / (bnds.x2[1] - bnds.x2[0]),
                                          three_d * dt / (bnds.x3[1] - bnds.x3[0])};

        // Timestep weighted by (half) volume
        const Real hdtv = 0.5 * dt / coords.Volume();
        const std::array<Real, 3> hdtvol = {hdtv, multi_d * hdtv, three_d * hdtv};

        // Face indexing
        // NOTE(@pdmullen): Including outside kernel seems to encroach upon internal NVCC
        // capture limits? Would need an extra capture for constexpr anyways...
        const int d1 = X1DIR;
        const int d2 = d1 + multi_d;
        const int d3 = d2 + three_d;
        const auto f1 = TE::F1;
        const auto f2 = (multi_d) ? TE::F2 : f1;
        const auto f3 = (three_d) ? TE::F3 : f2;

        // (Remaining) captures for constexpr
        [[maybe_unused]] auto &vp_ = vp;
        [[maybe_unused]] auto &vc_ = vcons;
        [[maybe_unused]] auto &vface_ = vface;
        [[maybe_unused]] auto &hcchat_ = hcchat;

        // Extract nspecies
        int nspecies = Null<int>();
        if constexpr (F == Fluid::gas) {
          nspecies = vp_.GetSize(b, gas::prim::density());
        } else if constexpr (F == Fluid::dust) {
          nspecies = vp_.GetSize(b, dust::prim::density());
        } else if constexpr (F == Fluid::radiation) {
          nspecies = vp_.GetSize(b, rad::prim::energy());
        }

        // Add the "flux source terms"
        for (int n = 0; n < nspecies; ++n) {
          const int IMX = VI(n, 0);
          const int IMY = VI(n, 1);
          const int IMZ = VI(n, 2);
          const int IVX = nspecies + IMX;
          const int IVY = nspecies + IMY;
          const int IVZ = nspecies + IMZ;
          const int IPR = nspecies * 4 + n; // may not be used
          const int IEG = nspecies * 3 + n; // may not be used

          // Pressure gradient force for gas and radiation
          if constexpr (F == Fluid::gas || F == Fluid::radiation) {
            // Pressure gradient force
            vc_(b, IMX, k, j, i) += dtdx[0] * (vp_.flux(b, d1, IPR, k, j, i) -
                                               vp_.flux(b, d1, IPR, k, j, i + 1));
            vc_(b, IMY, k, j, i) += dtdx[1] * (vp_.flux(b, d2, IPR, k, j, i) -
                                               vp_.flux(b, d2, IPR, k, j + multi_d, i));
            vc_(b, IMZ, k, j, i) += dtdx[2] * (vp_.flux(b, d3, IPR, k, j, i) -
                                               vp_.flux(b, d3, IPR, k + three_d, j, i));
          }

          // pdV source for gas internal energy equation
          if constexpr (F == Fluid::gas) {
            // pdV source term
            // clang-format off
            vc_(b, IEG, k, j, i) += hdtvol[0] *
                                    (vp_.flux(b, d1, IPR, k, j, i) +
                                     vp_.flux(b, d1, IPR, k, j, i + 1)) *
                                    (ax1[0] * vface_(b, f1,n, k, j, i) -
                                     ax1[1] * vface_(b, f1,n, k, j, i + 1));
            vc_(b, IEG, k, j, i) += hdtvol[1] *
                                    (vp_.flux(b, d2, IPR, k, j, i) +
                                     vp_.flux(b, d2, IPR, k, j + multi_d, i)) *
                                    (ax2[0] * vface_(b, f2, n, k, j, i) -
                                     ax2[1] * vface_(b, f2, n, k, j + multi_d, i));
            vc_(b, IEG, k, j, i) += hdtvol[2] *
                                    (vp_.flux(b, d3, IPR, k, j, i) +
                                     vp_.flux(b, d3, IPR, k + three_d, j, i)) *
                                    (ax3[0] * vface_(b, f3, n, k, j, i) -
                                     ax3[1] * vface_(b, f3, n, k + three_d, j, i));
            // clang-format on
          }

          // Apply "coordinate source terms" (if not Cartesian)
          if constexpr (G != Coordinates::cartesian) {
            // Extract primitive weighted timestep
            Real wdt = vp_(b, n, k, j, i) * dt;

            // Additionally weight by closure for radiation moments
            if constexpr (F == Fluid::radiation) {
              const Real &fx = vp_(b, IVX, k, j, i);
              const Real &fy = vp_(b, IVY, k, j, i);
              const Real &fz = vp_(b, IVZ, k, j, i);
              const Real ff = std::sqrt(SQR(fx) + SQR(fy) + SQR(fz));
              const Real chi = Moments::ThriceEddingtonFactor<C>(ff);
              wdt *= ((chi - 1.) / (ff + Fuzz<Real>())) * hcchat_;
            }

            // Update momenta
            // clang-format off
            const Real t1 = SQR(vp_(b, IVX, k, j, i) + rfv[0]);
            const Real t2 = SQR(vp_(b, IVY, k, j, i) + rfv[1]);
            const Real t3 = SQR(vp_(b, IVZ, k, j, i) + rfv[2]);
            vc_(b, IMX, k, j, i) += x1dep * wdt * (dh1[0]*t1 + dh1[1]*t2 + dh1[2]*t3);
            vc_(b, IMY, k, j, i) += x2dep * wdt * (dh2[0]*t1 + dh2[1]*t2 + dh2[2]*t3);
            vc_(b, IMZ, k, j, i) += x3dep * wdt * (dh3[0]*t1 + dh3[1]*t2 + dh3[2]*t3);
            // clang-format on
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::CalculateFluxesReconSelect
//! \brief Dispatch templated function depending on runtime reconstruction option.
template <Coordinates G, Fluid F, Closure C, RSolver R, typename PKG, typename PRIM,
          typename FLUX, typename FACE>
TaskStatus CalculateFluxesReconSelect(MeshData<Real> *md, PKG &pkg, PRIM vp, FLUX vflx,
                                      FACE vface, const bool pcm) {
  const auto recon_method = pkg->template Param<ReconstructionMethod>("recon");

  // Select CalculateFluxesImpl based on reconstruction method
  typedef ReconstructionMethod S;
  if ((recon_method == ReconstructionMethod::pcm) || (pcm)) {
    return CalculateFluxesImpl<G, F, C, R, S::pcm>(md, pkg, vp, vflx, vface);
  } else if (recon_method == S::plm) {
    return CalculateFluxesImpl<G, F, C, R, S::plm>(md, pkg, vp, vflx, vface);
  } else if (recon_method == S::ppm) {
    return CalculateFluxesImpl<G, F, C, R, S::ppm>(md, pkg, vp, vflx, vface);
  } else {
    PARTHENON_FAIL("Reconstruction method not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::CalculateFluxesRiemannSelect
//! \brief Dispatch templated function depending on runtime Riemann solver option.
template <Coordinates G, Fluid F, Closure C, typename PKG, typename PRIM, typename FLUX,
          typename FACE>
TaskStatus CalculateFluxesRiemannSelect(MeshData<Real> *md, PKG &pkg, PRIM vp, FLUX vflx,
                                        FACE vface, const bool pcm) {
  const auto riemann_method = pkg->template Param<RSolver>("rsolver");

  // Select CalculateFluxesReconSelect based on Riemann solver
  typedef RSolver R;
  if (riemann_method == R::hllc) {
    if constexpr (F != Fluid::radiation && F != Fluid::dust) {
      return CalculateFluxesReconSelect<G, F, C, R::hllc>(md, pkg, vp, vflx, vface, pcm);
    } else {
      PARTHENON_FAIL("Radiation fluid does not support an HLLC solver")
    }
  } else if (riemann_method == R::hlle) {
    return CalculateFluxesReconSelect<G, F, C, R::hlle>(md, pkg, vp, vflx, vface, pcm);
  } else if (riemann_method == R::llf) {
    return CalculateFluxesReconSelect<G, F, C, R::llf>(md, pkg, vp, vflx, vface, pcm);
  } else {
    PARTHENON_FAIL("Riemann solver not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::CalculateFluxes
//! \brief Hierarchically dispatch templated function depending on runtime coord system
template <Fluid F, Closure C = Closure::null, typename PKG, typename PRIM, typename FLUX,
          typename FACE>
TaskStatus CalculateFluxes(MeshData<Real> *md, PKG &pkg, PRIM vp, FLUX vflx, FACE vf,
                           const bool dc) {
  const auto sys = pkg->template Param<Coordinates>("coords");

  // Select CalculateFluxesRiemannSelect based on coordinate system
  typedef Coordinates G;
  if (sys == G::cartesian) {
    return CalculateFluxesRiemannSelect<G::cartesian, F, C>(md, pkg, vp, vflx, vf, dc);
  } else if (sys == G::spherical3D) {
    return CalculateFluxesRiemannSelect<G::spherical3D, F, C>(md, pkg, vp, vflx, vf, dc);
  } else if (sys == G::spherical1D) {
    return CalculateFluxesRiemannSelect<G::spherical1D, F, C>(md, pkg, vp, vflx, vf, dc);
  } else if (sys == G::spherical2D) {
    return CalculateFluxesRiemannSelect<G::spherical2D, F, C>(md, pkg, vp, vflx, vf, dc);
  } else if (sys == G::cylindrical) {
    return CalculateFluxesRiemannSelect<G::cylindrical, F, C>(md, pkg, vp, vflx, vf, dc);
  } else if (sys == G::axisymmetric) {
    return CalculateFluxesRiemannSelect<G::axisymmetric, F, C>(md, pkg, vp, vflx, vf, dc);
  } else {
    PARTHENON_FAIL("Coordinate type not recognized!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::FluxSourceGeomSelect
//! \brief Dispatch templated function depending on runtime coordinate system.
template <Fluid F, Closure C = Closure::null, typename PKG, typename PRIM, typename CONS,
          typename FACE>
TaskStatus FluxSource(MeshData<Real> *md, PKG &pkg, PRIM vp, CONS vcons, FACE vface,
                      const Real dt) {
  auto pm = md->GetParentPointer();

  // Extract rotating frame omega
  Real omf = 0.0;
  if (pm->packages.Get("artemis")->template Param<bool>("do_rotating_frame")) {
    auto &rf_pkg = pm->packages.Get("rotating_frame");
    omf = rf_pkg->template Param<Real>("omega");
  }

  // Call FluxSourceImpl given
  typedef Coordinates G;
  const auto sys = pkg->template Param<Coordinates>("coords");
  if (sys == G::cartesian) {
    return FluxSourceImpl<G::cartesian, F, C>(md, pkg, vp, vcons, vface, omf, dt);
  } else if (sys == G::spherical3D) {
    return FluxSourceImpl<G::spherical3D, F, C>(md, pkg, vp, vcons, vface, omf, dt);
  } else if (sys == G::spherical1D) {
    return FluxSourceImpl<G::spherical1D, F, C>(md, pkg, vp, vcons, vface, omf, dt);
  } else if (sys == G::spherical2D) {
    return FluxSourceImpl<G::spherical2D, F, C>(md, pkg, vp, vcons, vface, omf, dt);
  } else if (sys == G::cylindrical) {
    return FluxSourceImpl<G::cylindrical, F, C>(md, pkg, vp, vcons, vface, omf, dt);
  } else if (sys == G::axisymmetric) {
    return FluxSourceImpl<G::axisymmetric, F, C>(md, pkg, vp, vcons, vface, omf, dt);
  } else {
    PARTHENON_FAIL("Coordinate type not recognized!");
  }
}

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_FLUID_FLUXES_HPP_
