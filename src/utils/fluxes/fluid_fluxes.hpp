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
  } else if constexpr (FLUID_TYPE == Fluid::radiation) {
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

  // Adiabatic index, if used
  EOS eos;
  if constexpr (FLUID_TYPE == Fluid::gas) {
    eos = pkg->template Param<EOS>("eos_d");
  }
  Real chat = Null<Real>();
  if constexpr (FLUID_TYPE == Fluid::radiation) {
    chat = pkg->template Param<Real>("chat");
  }

  // Scratch properties
  // NOTE(PDM): Scratch here must be able to contain up to the total number of species,
  // even if some blocks don't contain all species
  const int nspecies = pkg->template Param<int>("nspecies");
  const int nvars = vprim.GetMaxNumberOfVars();
  int scr_size = ScratchPad2D<Real>::shmem_size(nvars, ncells1) * 2;
  const int scr_level = pkg->template Param<int>("scr_level");

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
        Reconstruction<RECON, X1DIR, GEOM> recon;
        recon.apply(mbr, b, k, j, il - 1, iu, vprim, wl, wr);
        mbr.team_barrier();

        // Compute fluxes over[is, ie + 1]
        RiemannSolver<RIEMANN, FLUID_TYPE> riemann;
        riemann.solve(eos, chat, mbr, b, k, j, il, iu, X1DIR, wl, wr, vprim, vflux,
                      vface);
        mbr.team_barrier();

        // Scale X1-momentum flux by appropriate scale factor for coord system
        ScaleMomentumFlux<GEOM, FLUID_TYPE, X1DIR>(mbr, b, k, j, il, iu, vflux);
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
            Reconstruction<RECON, X2DIR, GEOM> recon;
            recon.apply(mbr, b, k, j, il, iu, vprim, wl_jp1, wr);
            mbr.team_barrier();

            if (j > jl) {
              // compute fluxes over [js,je+1]
              RiemannSolver<RIEMANN, FLUID_TYPE> riemann;
              riemann.solve(eos, chat, mbr, b, k, j, il, iu, X2DIR, wl, wr, vprim, vflux,
                            vface);
              mbr.team_barrier();

              // Scale X2-momentum flux by appropriate scale factor for coord system
              ScaleMomentumFlux<GEOM, FLUID_TYPE, X2DIR>(mbr, b, k, j, il, iu, vflux);
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
            Reconstruction<RECON, X3DIR, GEOM> recon;
            recon.apply(mbr, b, k, j, il, iu, vprim, wl_kp1, wr);
            mbr.team_barrier();

            // compute fluxes over [ks,ke+1]
            if (k > kl) {
              RiemannSolver<RIEMANN, FLUID_TYPE> riemann;
              riemann.solve(eos, chat, mbr, b, k, j, il, iu, X3DIR, wl, wr, vprim, vflux,
                            vface);
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
  } else if constexpr (FLUID_TYPE == Fluid::radiation) {
    nvar = 5;
  }
  const int nspecies = vprim.GetMaxNumberOfVars() / nvar;
  Real c = Null<Real>(), chat = Null<Real>();
  if constexpr (FLUID_TYPE == Fluid::radiation) {
    c = pkg->template Param<Real>("c");
    chat = pkg->template Param<Real>("chat");
  }
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
          } else if constexpr (FLUID_TYPE == Fluid::radiation) {
            const int IPR = nspecies * 4 + n;
            vcons_(b, IMX, k, j, i) += dt_ / dx[0] *
                                       (vprim_.flux(b, X1DIR, IPR, k, j, i) -
                                        vprim_.flux(b, X1DIR, IPR, k, j, i + 1));
            if (multi_d_) {
              vcons_(b, IMY, k, j, i) += dt_ / dx[1] *
                                         (vprim_.flux(b, X2DIR, IPR, k, j, i) -
                                          vprim_.flux(b, X2DIR, IPR, k, j + 1, i));
            }
            if (three_d_) {
              vcons_(b, IMZ, k, j, i) += dt_ / dx[2] *
                                         (vprim_.flux(b, X3DIR, IPR, k, j, i) -
                                          vprim_.flux(b, X3DIR, IPR, k + 1, j, i));
            }
          }

          // Add coordinate source term
          Real rdt = vprim(b, n, k, j, i) * dt;
          const Real cfac_ = c * chat;
          if constexpr (FLUID_TYPE == Fluid::radiation) {
            const Real f2 =
                std::sqrt(SQR(vprim(b, IVX, k, j, i)) + SQR(vprim(b, IVY, k, j, i)) +
                          SQR(vprim(b, IVZ, k, j, i)));
            const Real chi = Radiation::EddingtonFactor(std::sqrt(f2));
            rdt *= (3. * chi - 1.) * 0.5 * cfac_ / (f2 + Fuzz<Real>());
          }
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
TaskStatus FluxSource(MeshData<Real> *md, PKG &pkg, PackPrim vprim, PackCons vcons,
                      PackFace vface, const Real dt) {
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

} // namespace ArtemisUtils

#endif // UTILS_FLUXES_FLUID_FLUXES_HPP_
