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
#ifndef PGEN_BRIO_WU_2SIDES_HPP_
#define PGEN_BRIO_WU_2SIDES_HPP_
//! \file brio_wu_2sides.hpp
//! \brief
//!
//! This is the Mach=3 problem from Lowrie & Edwards (2008).
//! The specific values are taken from the Fornax and Quokka code papers
//!
//!  mu = mH, gamma = 5/3, rho*kappa = 577 /cm
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
#include "mhd/extended/defs.hpp"
#include "utils/integrators/artemis_integrator.hpp"

using ArtemisUtils::EOS;

namespace brio_wu_2sides {

struct BRIO_WU_2SIDES_Params {
  Real rho0;
  Real rhol, vxl, pl, pel, bxl, byl, bzl;
  Real rhor, vxr, pr, per, bxr, byr, bzr;
  Real xdiscl, xdiscr;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_BRIO_WU_2SIDES_Params
//! \brief Extracts brio_wu_2sides parameters from ParameterInput.
inline void Init_BRIO_WU_2SIDES_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("brio_wu_2sides_params"))) {
    BRIO_WU_2SIDES_Params brio_wu_2sides_params;
    brio_wu_2sides_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    brio_wu_2sides_params.rhol = pin->GetOrAddReal("problem", "rhol", 1.0);
    brio_wu_2sides_params.vxl = pin->GetOrAddReal("problem", "vxl", 0.0);
    brio_wu_2sides_params.pl = pin->GetOrAddReal("problem", "pl", 1.0);
    brio_wu_2sides_params.pel = pin->GetOrAddReal("problem", "pel", 0.5);
    brio_wu_2sides_params.bxl = pin->GetOrAddReal("problem", "bxl", 0.75);
    brio_wu_2sides_params.byl = pin->GetOrAddReal("problem", "byl", 1.0);
    brio_wu_2sides_params.bzl = pin->GetOrAddReal("problem", "bzl", 0.0);
    brio_wu_2sides_params.rhor = pin->GetOrAddReal("problem", "rhor", 0.125);
    brio_wu_2sides_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.0);
    brio_wu_2sides_params.pr = pin->GetOrAddReal("problem", "pr", 0.1);
    brio_wu_2sides_params.per = pin->GetOrAddReal("problem", "per", 0.05);
    brio_wu_2sides_params.bxr = pin->GetOrAddReal("problem", "bxr", 0.75);
    brio_wu_2sides_params.byr = pin->GetOrAddReal("problem", "byr", -1.0);
    brio_wu_2sides_params.bzr = pin->GetOrAddReal("problem", "bzr", 0.0);
    brio_wu_2sides_params.xdiscl = pin->GetOrAddReal("problem", "xdiscl", 0.5);
    brio_wu_2sides_params.xdiscr = pin->GetOrAddReal("problem", "xdiscr", 0.5);
    params.Add("brio_wu_2sides_params", brio_wu_2sides_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BRIO_WU_2SIDES()
//! \brief Sets initial conditions for brio_wu_2sides problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  int nghosts = parthenon::Globals::nghost;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The brio wu problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The brio wu problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",2.0);
  Real gm1 = gamma - 1.;
  const int ndim = ProblemDimension(pin);

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  // -> Init prim.B & press so that BCs whichneed these will not run into segfault 
  //    during initialization
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
	  gas::prim::Pe, gas::prim::Te, gas::prim::Ti,
	  gas::prim::pressure,gas::prim::Bfield>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // BRIO_WU_2SIDES parameters
  auto shkp = artemis_pkg->Param<BRIO_WU_2SIDES_Params>("brio_wu_2sides_params");
  const Real rho0 = shkp.rho0;
  // Setup brio_wu_2sides state
  pmb->par_for(
      "brio_wu_2sides", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const bool upwind = (xi[0]>= shkp.xdiscl and xi[0] <= shkp.xdiscr);
        const Real rho = upwind ? shkp.rhol : shkp.rhor;
        const Real vx = upwind ? shkp.vxl : shkp.vxr;
        const Real P = upwind ? shkp.pl : shkp.pr;
	const Real bx = upwind ? shkp.bxl : shkp.bxr;
	const Real by = upwind ? shkp.byl : shkp.byr;
	const Real Pe = upwind ? shkp.pel : shkp.per;
        v(0, gas::prim::density(0), k, j, i) = rho*rho0;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P*rho0/(rho*rho0*gm1);
	v(0, gas::prim::pressure(0), k, j, i) = P*rho0;
	v(0, gas::prim::Bfield(0), k, j, i) = bx*sqrt(rho0);
	v(0, gas::prim::Bfield(1), k, j, i) = by*sqrt(rho0);
	v(0, gas::prim::Bfield(2), k, j, i) = 0.;
	v(0, gas::prim::Pe(0), k, j, i) = Pe*rho0;

	// Need compute temperature for electron & ion energy exchange
	v(0, gas::prim::Ti(0), k, j, i) = (v(0, gas::prim::pressure(0), k, j, i) -
			v(0, gas::prim::Pe(0), k, j, i)) / v(0, gas::prim::density(0), k, j, i);
	v(0, gas::prim::Te(0), k, j, i) = v(0, gas::prim::Pe(0), k, j, i) / (Z_ion * v(0, gas::prim::density(0), k, j, i));
      });
  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "brio_wu_2sides", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	const bool upwind = (xf[0]>= shkp.xdiscl and xf[0] <= shkp.xdiscr);
	// YH: add magnetic field - refer to problem gen from fine_adv
	const Real bx = upwind ? shkp.bxl : shkp.bxr;
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = bx*sqrt(rho0);
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "brio_wu_2sides", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
        const bool upwind = (xf[0]>= shkp.xdiscl and xf[0] <= shkp.xdiscr);
        const Real by = upwind ? shkp.byl : shkp.byr;
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = by*sqrt(rho0);
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "brio_wu_2sides", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        const bool upwind = (xf[0]>= shkp.xdiscl and xf[0] <= shkp.xdiscr);
        const Real bz = upwind ? shkp.bzl : shkp.bzr;
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = bz*sqrt(rho0);
      });
  static auto desc_EJ =
      MakePackDescriptor<gas::edge::Efield,
      			 gas::edge::J>((pmb->resolved_packages).get());
  auto vEJ = desc_EJ.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E1);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E1);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E1, gas::edge::Efield(), k, j, i) = 0.;
	Real dBz_dy = -(vmag(0, TE::F3, 0, k, j - (ndim > 1 and j>0), i)
                        - vmag(0, TE::F3, 0, k, j, i)) / dx[1];
        Real dBy_dz = -(vmag(0, TE::F2, 0, k - (ndim > 2 and k>0), j, i)
                        - vmag(0, TE::F2, 0, k, j, i)) / dx[2];
        vEJ(0, TE::E1, gas::edge::J(), k, j, i) = dBz_dy - dBy_dz;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E2);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E2);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E2, gas::edge::Efield(), k, j, i) = 0.;
	Real dBz_dx = -(vmag(0, TE::F3, 0, k, j, i - (i>0))
                        - vmag(0, TE::F3, 0, k, j, i)) / dx[0];
        Real dBx_dz = -(vmag(0, TE::F1, 0, k - (ndim > 2 and k>0), j, i)
                        - vmag(0, TE::F1, 0, k, j, i)) / dx[2];
        vEJ(0, TE::E2, gas::edge::J(), k, j, i) = -dBz_dx + dBx_dz;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E3);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E3);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E3, gas::edge::Efield(), k, j, i) = 0.;
	Real dBy_dx = -(vmag(0, TE::F2, 0, k, j, i - (i>0))
                        - vmag(0, TE::F2, 0, k, j, i)) / dx[0];
        Real dBx_dy = -(vmag(0, TE::F1, 0, k, j - (ndim > 1 and j>0), i)
                        - vmag(0, TE::F1, 0, k, j, i)) / dx[1];
        vEJ(0, TE::E3, gas::edge::J(), k, j, i) = dBy_dx - dBx_dy;
      });
}


} // namespace brio_wu_2sides
#endif // PGEN_BRIO_WU_2SIDES_HPP_
