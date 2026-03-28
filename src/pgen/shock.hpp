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
#ifndef PGEN_SHOCK_HPP_
#define PGEN_SHOCK_HPP_
//! \file shock.hpp
//! \brief
//!
//! This is the Mach=3 problem from Lowrie & Edwards (2008).
//! The specific values are taken from the Fornax and Quokka code papers
//!
//!  mu = mH, gamma = 5/3, rho*kappa = 577 /cm
//!  c/chat = 43.3526011561
//!  left state:         |  right state:
//!      T = 2.18e6 K    |   T = 7.98e6 K
//!    rho = 5.69 g/cc   | rho = 17.1 g/cc
//!     vx = 5.19e7 cm/s |  vx = 1.73e7 cm/s

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

// jaybenne includes
#include "jaybenne.hpp"

using ArtemisUtils::EOS;

namespace shock {

struct ShockParams {
  Real rhol, vxl, tl, pl;
  Real rhor, vxr, tr, pr;
  Real bx;
  Real byl, byr;
  Real bzl, bzr;
  Real xdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void InitShockParams
//! \brief Extracts shock parameters from ParameterInput.
inline void InitShockParams(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("shock_params"))) {
    ShockParams shock_params;
    shock_params.rhol = pin->GetOrAddReal("problem", "rhol", 5.69);
    shock_params.vxl = pin->GetOrAddReal("problem", "vxl", 5.19e7);
    shock_params.tl = pin->GetOrAddReal("problem", "tl", 2.18e6);
    shock_params.pl = pin->GetOrAddReal("problem", "pl", Null<Real>());
    shock_params.rhor = pin->GetOrAddReal("problem", "rhor", 17.1);
    shock_params.vxr = pin->GetOrAddReal("problem", "vxr", 1.73e7);
    shock_params.tr = pin->GetOrAddReal("problem", "tr", 7.98e6);
    shock_params.pr = pin->GetOrAddReal("problem", "pr", Null<Real>());
    shock_params.bx = pin->GetOrAddReal("problem", "bx", 0.0);
    shock_params.byl = pin->GetOrAddReal("problem", "byl", 0.0);
    shock_params.byr = pin->GetOrAddReal("problem", "byr", 0.0);
    shock_params.bzl = pin->GetOrAddReal("problem", "bzl", 0.0);
    shock_params.bzr = pin->GetOrAddReal("problem", "bzr", 0.0);
    shock_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.0005);
    params.Add("shock_params", shock_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Shock()
//! \brief Sets initial conditions for shock problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_imc = artemis_pkg->Param<bool>("do_imc");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas, "The shock problem requires gas hydrodynamics!");
  PARTHENON_REQUIRE(!(do_dust), "The shock problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");

  Real ar = Null<Real>();
  if (do_moment) {
    ar = pmb->packages.Get("moments")->Param<Real>("arad");
  }

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         rad::prim::energy, rad::prim::flux, field::face::B>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  static auto desc_g = MakePackDescriptor<geom::x1v>((pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib1 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
  IndexRange jb1 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  IndexRange kb1 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  IndexRange ib2 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2);
  IndexRange jb2 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  IndexRange kb2 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  IndexRange ib3 = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);
  IndexRange jb3 = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  IndexRange kb3 = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  auto &pco = pmb->coords;

  // Shock parameters
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  // Setup shock state
  pmb->par_for(
      "shock", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto xi = coords.x1v();
        const bool upwind = (xi <= shkp.xdisc);
        const Real rho = upwind ? shkp.rhol : shkp.rhor;
        const Real vx = upwind ? shkp.vxl : shkp.vxr;
        const Real pres = upwind ? shkp.pl : shkp.pr;
        const Real sie = (pres != Null<Real>())
                             ? ArtemisUtils::EofPR(eos_d, pres, rho)
                             : eos_d.InternalEnergyFromDensityTemperature(
                                   rho, upwind ? shkp.tl : shkp.tr);
        const Real T = eos_d.TemperatureFromDensityInternalEnergy(rho, sie);
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = sie;
        if (do_moment) {
          v(0, rad::prim::energy(0), k, j, i) = ar * SQR(SQR(T));
          v(0, rad::prim::flux(0), k, j, i) = 0.0;
          v(0, rad::prim::flux(1), k, j, i) = 0.0;
          v(0, rad::prim::flux(2), k, j, i) = 0.0;
        }
      });

  if (do_mhd) {
    pmb->par_for(
        "shock_b1", kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v(0, TE::F1, field::face::B(), k, j, i) = shkp.bx;
        });
    pmb->par_for(
        "shock_b2", kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
          const auto xi = coords.x1v();
          const bool upwind = (xi <= shkp.xdisc);
          v(0, TE::F2, field::face::B(), k, j, i) = upwind ? shkp.byl : shkp.byr;
        });
    pmb->par_for(
        "shock_b3", kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
          const auto xi = coords.x1v();
          const bool upwind = (xi <= shkp.xdisc);
          v(0, TE::F3, field::face::B(), k, j, i) = upwind ? shkp.bzl : shkp.bzr;
        });
  }

  if (do_imc) jaybenne::InitializeRadiation(md.get(), true);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ShockInnerX1()
template <Coordinates GEOM>
inline void ShockInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  Real ar = Null<Real>();
  if (do_moment) {
    ar = pmb->packages.Get("moments")->Param<Real>("arad");
  }
  const auto nb = IndexRange{0, 0};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, rad::prim::energy,
                                                 rad::prim::flux>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;
  static auto descriptors_b =
      ArtemisUtils::GetBoundaryPackDescriptorMap<field::face::B>(mbd);
  auto vb = descriptors_b[coarse].GetPack(mbd.get());
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  pmb->par_for_bndry(
      "ShockInnerX1", nb, IndexDomain::inner_x1, TE::CC, coarse, false,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const Real sie =
            (shkp.pl != Null<Real>())
                ? ArtemisUtils::EofPR(eos_d, shkp.pl, shkp.rhol)
                : eos_d.InternalEnergyFromDensityTemperature(shkp.rhol, shkp.tl);
        const Real T = eos_d.TemperatureFromDensityInternalEnergy(shkp.rhol, sie);
        for (int n = 0; n < v.GetSize(0, gas::prim::density()); ++n) {
          v(0, gas::prim::density(n), k, j, i) = shkp.rhol;
          v(0, gas::prim::velocity(VI(n, 0)), k, j, i) = shkp.vxl;
          v(0, gas::prim::velocity(VI(n, 1)), k, j, i) = 0.0;
          v(0, gas::prim::velocity(VI(n, 2)), k, j, i) = 0.0;
          v(0, gas::prim::sie(n), k, j, i) = sie;
        }
        if (do_moment) {
          for (int n = 0; n < v.GetSize(0, rad::prim::energy()); ++n) {
            v(0, rad::prim::energy(n), k, j, i) = ar * SQR(SQR(T));
            v(0, rad::prim::flux(VI(n, 0)), k, j, i) = 0.0;
            v(0, rad::prim::flux(VI(n, 1)), k, j, i) = 0.0;
            v(0, rad::prim::flux(VI(n, 2)), k, j, i) = 0.0;
          }
        }
      });

  if (do_mhd && vb.GetMaxNumberOfVars() > 0) {
    pmb->par_for_bndry(
        "ShockInnerX1::B1", nb, IndexDomain::inner_x1, TE::F1, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          vb(0, TE::F1, field::face::B(), k, j, i) = shkp.bx;
        });
    pmb->par_for_bndry(
        "ShockInnerX1::B2", nb, IndexDomain::inner_x1, TE::F2, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          vb(0, TE::F2, field::face::B(), k, j, i) = shkp.byl;
        });
    pmb->par_for_bndry(
        "ShockInnerX1::B3", nb, IndexDomain::inner_x1, TE::F3, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          vb(0, TE::F3, field::face::B(), k, j, i) = shkp.bzl;
        });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ShockOuterX1()
template <Coordinates GEOM>
inline void ShockOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  auto shkp = artemis_pkg->Param<ShockParams>("shock_params");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  Real ar = Null<Real>();
  if (do_moment) {
    ar = pmb->packages.Get("moments")->Param<Real>("arad");
  }
  const auto nb = IndexRange{0, 0};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, rad::prim::energy,
                                                 rad::prim::flux>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;
  static auto descriptors_b =
      ArtemisUtils::GetBoundaryPackDescriptorMap<field::face::B>(mbd);
  auto vb = descriptors_b[coarse].GetPack(mbd.get());

  pmb->par_for_bndry(
      "ShockOuterX1", nb, IndexDomain::outer_x1, TE::CC, coarse, false,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const Real sie =
            (shkp.pr != Null<Real>())
                ? ArtemisUtils::EofPR(eos_d, shkp.pr, shkp.rhor)
                : eos_d.InternalEnergyFromDensityTemperature(shkp.rhor, shkp.tr);
        const Real T = eos_d.TemperatureFromDensityInternalEnergy(shkp.rhor, sie);
        for (int n = 0; n < v.GetSize(0, gas::prim::density()); ++n) {
          v(0, gas::prim::density(n), k, j, i) = shkp.rhor;
          v(0, gas::prim::velocity(VI(n, 0)), k, j, i) = shkp.vxr;
          v(0, gas::prim::velocity(VI(n, 1)), k, j, i) = 0.0;
          v(0, gas::prim::velocity(VI(n, 2)), k, j, i) = 0.0;
          v(0, gas::prim::sie(n), k, j, i) = sie;
        }
        if (do_moment) {
          for (int n = 0; n < v.GetSize(0, rad::prim::energy()); ++n) {
            v(0, rad::prim::energy(n), k, j, i) = ar * SQR(SQR(T));
            v(0, rad::prim::flux(VI(n, 0)), k, j, i) = 0.0;
            v(0, rad::prim::flux(VI(n, 1)), k, j, i) = 0.0;
            v(0, rad::prim::flux(VI(n, 2)), k, j, i) = 0.0;
          }
        }
      });

  if (do_mhd && vb.GetMaxNumberOfVars() > 0) {
    pmb->par_for_bndry(
        "ShockOuterX1::B1", nb, IndexDomain::outer_x1, TE::F1, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          vb(0, TE::F1, field::face::B(), k, j, i) = shkp.bx;
        });
    pmb->par_for_bndry(
        "ShockOuterX1::B2", nb, IndexDomain::outer_x1, TE::F2, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          vb(0, TE::F2, field::face::B(), k, j, i) = shkp.byr;
        });
    pmb->par_for_bndry(
        "ShockOuterX1::B3", nb, IndexDomain::outer_x1, TE::F3, coarse, false,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          vb(0, TE::F3, field::face::B(), k, j, i) = shkp.bzr;
        });
  }

  return;
}

} // namespace shock
#endif // PGEN_SHOCK_HPP_
