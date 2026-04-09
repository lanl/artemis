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
#ifndef PGEN_BRIO_WU_REVERSED_HPP_
#define PGEN_BRIO_WU_REVERSED_HPP_
//! \file brio_wu_reversed.hpp
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

namespace brio_wu_reversed {

struct BRIO_WU_REVERSED_Params {
  Real rho0;
  Real rhol, vxl, pl, pel, bxl, byl, bzl;
  Real rhor, vxr, pr, per, bxr, byr, bzr;
  Real xdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_BRIO_WU_REVERSED_Params
//! \brief Extracts brio_wu_reversed parameters from ParameterInput.
inline void Init_BRIO_WU_REVERSED_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("brio_wu_reversed_params"))) {
    BRIO_WU_REVERSED_Params brio_wu_reversed_params;
    brio_wu_reversed_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    brio_wu_reversed_params.rhor = pin->GetOrAddReal("problem", "rhor", 1.0);
    brio_wu_reversed_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.0);
    brio_wu_reversed_params.pr = pin->GetOrAddReal("problem", "pr", 1.0);
    brio_wu_reversed_params.per = pin->GetOrAddReal("problem", "per", 0.5);
    brio_wu_reversed_params.bxr = pin->GetOrAddReal("problem", "bxr", 0.75);
    brio_wu_reversed_params.byr = pin->GetOrAddReal("problem", "byr", 1.0);
    brio_wu_reversed_params.bzr = pin->GetOrAddReal("problem", "bzr", 0.0);
    brio_wu_reversed_params.rhol = pin->GetOrAddReal("problem", "rhol", 0.125);
    brio_wu_reversed_params.vxl = pin->GetOrAddReal("problem", "vxl", 0.0);
    brio_wu_reversed_params.pl = pin->GetOrAddReal("problem", "pl", 0.1);
    brio_wu_reversed_params.pel = pin->GetOrAddReal("problem", "pel", 0.05);
    brio_wu_reversed_params.bxl = pin->GetOrAddReal("problem", "bxl", 0.75);
    brio_wu_reversed_params.byl = pin->GetOrAddReal("problem", "byl", -1.0);
    brio_wu_reversed_params.bzl = pin->GetOrAddReal("problem", "bzl", 0.0);
    brio_wu_reversed_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.);
    params.Add("brio_wu_reversed_params", brio_wu_reversed_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BRIO_WU_REVERSED()
//! \brief Sets initial conditions for brio_wu_reversed problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  int nghosts = parthenon::Globals::nghost;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The brio wu reversed problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The brio wu reversed problem does not permit dust hydrodynamics!");
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
	  gas::prim::Pe,
	  gas::prim::pressure,gas::prim::Bfield>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // BRIO_WU_REVERSED parameters
  auto shkp = artemis_pkg->Param<BRIO_WU_REVERSED_Params>("brio_wu_reversed_params");
  const Real rho0 = shkp.rho0;

  // Setup brio_wu_reversed state
  pmb->par_for(
      "brio_wu_reversed", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const bool upwind = (xi[0] <= shkp.xdisc);
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
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "brio_wu_reversed", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	const bool upwind = (xf[0] <= shkp.xdisc);
	// YH: add magnetic field - refer to problem gen from fine_adv
	const Real bx = upwind ? shkp.bxl : shkp.bxr;
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = bx*sqrt(rho0);
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "brio_wu_reversed", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
        const bool upwind = (xf[0] <= shkp.xdisc);
        const Real by = upwind ? shkp.byl : shkp.byr;
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = by*sqrt(rho0);
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "brio_wu_reversed", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        const bool upwind = (xf[0] <= shkp.xdisc);
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
	Real dBz_dy = -(vmag(0, TE::F3, 0, k, j - (ndim > 1), i)
                        - vmag(0, TE::F3, 0, k, j, i)) / dx[1];
        Real dBy_dz = -(vmag(0, TE::F2, 0, k - (ndim > 2), j, i)
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
        Real dBx_dz = -(vmag(0, TE::F1, 0, k - (ndim > 2), j, i)
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
        Real dBx_dy = -(vmag(0, TE::F1, 0, k, j - (ndim > 1), i)
                        - vmag(0, TE::F1, 0, k, j, i)) / dx[1];
        vEJ(0, TE::E3, gas::edge::J(), k, j, i) = dBy_dx - dBx_dy;
      });

}


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::InflowInnerX1()
//! \brief Sets BCs on -x boundary 
template <Coordinates GEOM>
inline void InflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<BRIO_WU_REVERSED_Params>("brio_wu_reversed_params");
  const Real rho0 = shkp.rho0;

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int is = range.s;
  pmb->par_for_bndry(
      "InflowInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = is; 
	const Real vx = v(l, gas::prim::velocity(0), k, j, iref);

	if (vx>=0.) {
          v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
          v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
          v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
          v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
          v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	  v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
	} else {
	  v(l, gas::prim::density(0), k, j, i) = 1.0*rho0;
          v(l, gas::prim::sie(0), k, j, i)     = 1.0*rho0/(1.0*rho0*gm1);
          v(l, gas::prim::velocity(0), k, j, i) = 0.;
          v(l, gas::prim::velocity(1), k, j, i) = 0.;          
	  v(l, gas::prim::velocity(2), k, j, i) = 0.;
	  v(l, gas::prim::Pe(0), k, j, i)     = 0.5*rho0;
	}
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int is_x = range_x.s;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = is_x;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);	
	const bool outflow = (vx >= 0.);
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = outflow ? v(l, TE::F1, gas::face::bfield(0), k, j, iref) : 0.75*sqrt(rho0);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int is_y = range_y.s;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = is_y;
	const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
	const bool outflow = (vx >= 0.);
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = outflow ? v(l, TE::F2, gas::face::bfield(0), k, j, iref) : 1.*sqrt(rho0); 
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int is_z = range_z.s;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = is_z;
	const Real vx = v(l, gas::prim::velocity(0), k, j, iref);                   
	const bool outflow = (vx >= 0.);
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = outflow ? v(l, TE::F3, gas::face::bfield(0), k, j, iref) : 0.; 
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int isE_x = rangeE_x.s;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = isE_x;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
        const bool outflow = (vx >= 0.);
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = outflow ? v(l, TE::E1, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E1, gas::edge::J(0), k, j, i) = outflow ? v(l, TE::E1, gas::edge::J(0), k, j, iref) : 0.;
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int isE_y = rangeE_y.s;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = isE_y;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
        const bool outflow = (vx >= 0.);
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = outflow ? v(l, TE::E2, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E2, gas::edge::J(0), k, j, i) = outflow ? v(l, TE::E2, gas::edge::J(0), k, j, iref) : 0.;
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int isE_z = rangeE_z.s;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = isE_z;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);            
	const bool outflow = (vx >= 0.);
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = outflow ? v(l, TE::F3, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E3, gas::edge::J(0), k, j, i) = outflow ? v(l, TE::F3, gas::edge::J(0), k, j, iref) : 0.;
        });
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::InflowOuterX1()
//! \brief Sets BCs on -x boundary 
template <Coordinates GEOM>
inline void InflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto shkp = artemis_pkg->Param<BRIO_WU_REVERSED_Params>("brio_wu_reversed_params");
  const Real rho0 = shkp.rho0;
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  EOS eos_d;
  eos_d = gas_pkg->template Param<EOS>("eos_d");
  Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::prim::Bfield,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int ie = range.e;
  pmb->par_for_bndry(
      "InflowOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie; 
	const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref),
                        v(l, gas::prim::sie(0), k, j, iref), lambda);;
        const Real dens = v(l, gas::prim::density(0), k, j, iref);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));

	if (vx>=0.) {
          v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
          v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
          v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
          v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
          v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	  v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
	} else {
	  v(l, gas::prim::density(0), k, j, i) = 0.125*rho0;
          v(l, gas::prim::sie(0), k, j, i)     = 0.1*rho0/(0.125*rho0*gm1);
          v(l, gas::prim::velocity(0), k, j, i) = 0.;
          v(l, gas::prim::velocity(1), k, j, i) = 0.;          
	  v(l, gas::prim::velocity(2), k, j, i) = 0.;
	  v(l, gas::prim::Pe(0), k, j, i)     = 0.05*rho0;
	}
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int ie_x = range_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = ie_x;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref),
                        v(l, gas::prim::sie(0), k, j, iref), lambda);;
        const Real dens = v(l, gas::prim::density(0), k, j, iref);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));	
	const bool outflow =  (vx >= 0.);
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = outflow ? v(l, TE::F1, gas::face::bfield(0), k, j, iref) : 0.75*sqrt(rho0);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int ie_y = range_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = ie_y;
	const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref),
                        v(l, gas::prim::sie(0), k, j, iref), lambda);;
        const Real dens = v(l, gas::prim::density(0), k, j, iref);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
	const bool outflow = (vx>=0.);
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = outflow ? v(l, TE::F2, gas::face::bfield(0), k, j, iref) : -1.*sqrt(rho0); 
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int ie_z = range_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie_z;
	const Real vx = v(l, gas::prim::velocity(0), k, j, iref);                  
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref),
                        v(l, gas::prim::sie(0), k, j, iref), lambda);;
        const Real dens = v(l, gas::prim::density(0), k, j, iref);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
	const bool outflow = (vx>=0.);
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = outflow ? v(l, TE::F3, gas::face::bfield(0), k, j, iref) : 0.; 
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int ieE_x = rangeE_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = ieE_x;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref),
                        v(l, gas::prim::sie(0), k, j, iref), lambda);;
        const Real dens = v(l, gas::prim::density(0), k, j, iref);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
        const bool outflow = (vx>=0.);
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = outflow ? v(l, TE::E1, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E1, gas::edge::J(0), k, j, i) = outflow ? v(l, TE::E1, gas::edge::J(0), k, j, iref) : 0.;
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int ieE_y = rangeE_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_y;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref),
                        v(l, gas::prim::sie(0), k, j, iref), lambda);;
        const Real dens = v(l, gas::prim::density(0), k, j, iref);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
        const bool outflow = (vx>=0.);
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = outflow ? v(l, TE::E2, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E2, gas::edge::J(0), k, j, i) = outflow ? v(l, TE::E2, gas::edge::J(0), k, j, iref) : 0.;
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int ieE_z = rangeE_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_z;
        const Real vx = v(l, gas::prim::velocity(0), k, j, iref);            
        const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref),
                        v(l, gas::prim::sie(0), k, j, iref), lambda);;
        const Real dens = v(l, gas::prim::density(0), k, j, iref);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));	
	const bool outflow = (vx>=0.);
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = outflow ? v(l, TE::F3, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E3, gas::edge::J(0), k, j, i) = outflow ? v(l, TE::F3, gas::edge::J(0), k, j, iref) : 0.;
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::FarFieldOuterX1()
//! \brief Sets BCs on -x boundary 
template <Coordinates GEOM>
inline void FarFieldOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto shkp = artemis_pkg->Param<BRIO_WU_REVERSED_Params>("brio_wu_reversed_params");
  const Real rho0 = shkp.rho0;
  Params &params = artemis_pkg->AllParams();
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  EOS eos_d;
  eos_d = gas_pkg->template Param<EOS>("eos_d");
  Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::prim::pressure, gas::prim::Bfield,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int ie = range.e;
  pmb->par_for_bndry(
      "FarFieldOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie;
	geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);
        const auto &dx = coords.GetCellWidths();
	const Real vx = v(l, gas::prim::velocity(0), k, j, iref);
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, iref), 
			v(l, gas::prim::sie(0), k, j, iref), lambda);;
	const Real dens = v(l, gas::prim::density(0), k, j, iref);
	const Real csound = sqrt(gamma*press/dens);
	const Real Mach = vx/csound;
	const Real bx = v(l, gas::prim::Bfield(0), k, j, iref);
        const Real by = v(l, gas::prim::Bfield(1), k, j, iref);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, iref);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
	const Real cs = std::sqrt(0.5*(csa - std::sqrt(SQR(csa) - 4*SQR(acan))));

	const Real p_inf = 0.1*rho0;
	const Real sigma = 25.;
	const Real K = sigma*(1.-SQR(Mach))*csound/L0;
	
	const Real alphaf = std::sqrt((SQR(csound)-SQR(cs))/(SQR(cf)-SQR(cs)));
	const Real alphas = std::sqrt((SQR(cf)-SQR(csound))/(SQR(cf)-SQR(cs)));
	const Real b_perp = std::sqrt(SQR(by) + SQR(bz));
	const Real betay = by/b_perp;
	const Real betaz = bz/b_perp;
	const Real S = bx==0. ? 0. : (bx>0. ? 1. : -1.);
	const Real dwdx_coef = (alphas*betaz*csound*S) / (2.*SQR(csound));
	const Real dvdx_coef = (alphas*betay*csound*S) / (2.*SQR(csound));
	const Real dudx_coef = (alphaf*cf) / (2.*SQR(csound));
	const Real dpdx_coef = (alphaf) / (2.*SQR(csound)*dens);
	const Real dBzdx_coef = (alphas*betaz) / (2.*csound*std::sqrt(dens));
	const Real dBydx_coef = (alphas*betay) / (2.*csound*std::sqrt(dens));
	const Real dwdx = (v(l, gas::prim::velocity(2), k, j, iref) - 
			v(l, gas::prim::velocity(2), k, j, iref-1))/dx[0];
	const Real dvdx = (v(l, gas::prim::velocity(1), k, j, iref) -     
			v(l, gas::prim::velocity(1), k, j, iref-1))/dx[0];
	const Real dudx = (v(l, gas::prim::velocity(0), k, j, iref) -             
			v(l, gas::prim::velocity(0), k, j, iref-1))/dx[0];
	const Real dpdx = (v(l, gas::prim::pressure(0), k, j, iref) -        
			v(l, gas::prim::pressure(0), k, j, iref-1))/dx[0];
	const Real dBzdx = (v(l, gas::prim::Bfield(2), k, j, iref) -    
			v(l, gas::prim::Bfield(2), k, j, iref-1))/dx[0];
        const Real dBydx = (v(l, gas::prim::Bfield(1), k, j, iref) -         
			v(l, gas::prim::Bfield(1), k, j, iref-1))/dx[0];
	const Real L1 = (vx-cf) * (dwdx_coef*dwdx + dvdx_coef*dvdx - dudx_coef*dudx
			+ dpdx_coef*dpdx + dBzdx_coef*dBzdx + dBydx_coef*dBydx);
	const Real dt = dx[0]/(abs(vx)+cf);
	const Real newP = p_inf;//press - dt*K*(press-p_inf);//(L1/K) + p_inf;

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
	if (vx>=cf) {
          v(l, gas::prim::sie(0), k, j, i) = v(l, gas::prim::sie(0), k, j, iref);
	} else {
	  v(l, gas::prim::sie(0), k, j, i) = newP/(v(l, gas::prim::density(0), k, j, i)*gm1);
	}
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int ie_x = range_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = ie_x;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int ie_y = range_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = ie_y;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref); 
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int ie_z = range_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie_z;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, j, iref); 
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int ieE_x = rangeE_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = ieE_x;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int ieE_y = rangeE_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_y;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int ieE_z = rangeE_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_z;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, j, iref);
        });
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator: NRBCOuterX1()
//! \brief Sets BCs on +x boundary 
//  ->. Non-reflecting boundary condition: Distinct from interior values as boundary values are evolved using the quasi-linear PDE. Interior values only come in via the backward difference of primitive variables and flow characteristics are modified to only allow outgoing flow.
//  -> For now, characteristic only applied to ideal MHD as E & J I will just extrapolate.
template <Coordinates GEOM>
inline void NRBCOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  auto shkp = artemis_pkg->Param<BRIO_WU_REVERSED_Params>("brio_wu_reversed_params");
  const Real rho0 = shkp.rho0;
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  EOS eos_d;
  eos_d = gas_pkg->template Param<EOS>("eos_d");
  Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};
  const Real dt = gas_pkg->Param<Real>("dt");
  const int ncycle = gas_pkg->Param<int>("ncycle");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::prim::pressure, gas::prim::Bfield,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
					gas::boundary::density, gas::boundary::velocity,
					gas::boundary::pressure, gas::boundary::Bfield>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  int nghosts = parthenon::Globals::nghost;
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int ie = range.e;
  pmb->par_for_bndry(
      "FarFieldOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie;
	geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);
        const auto &dx = coords.GetCellWidths();
	const Real vx = v(l, gas::prim::velocity(0), k, j, i);
        const bool outflow = (v(l, gas::prim::velocity(0), k, j, iref) >= 0.);
	const Real press = eos_d.PressureFromDensityInternalEnergy(v(l, gas::prim::density(0), k, j, i), 
			v(l, gas::prim::sie(0), k, j, iref), lambda);;
	const Real dens = v(l, gas::prim::density(0), k, j, i);
	const Real csound = sqrt(gamma*press/dens);
	const Real Mach = vx/csound;
	const Real bx = v(l, gas::prim::Bfield(0), k, j, i);
        const Real by = v(l, gas::prim::Bfield(1), k, j, i);
        const Real bz = v(l, gas::prim::Bfield(2), k, j, i);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
	const Real cs = std::sqrt(0.5*(csa - std::sqrt(SQR(csa) - 4*SQR(acan))));

	const Real p_inf = 0.1*rho0;
	const Real sigma = 25000.;
	const Real K = sigma*(1.-SQR(Mach))*csound/L0;
	
	const Real alphaf = std::sqrt((SQR(csound)-SQR(cs))/(SQR(cf)-SQR(cs)));
	const Real alphas = std::sqrt((SQR(cf)-SQR(csound))/(SQR(cf)-SQR(cs)));
	const Real b_perp = std::sqrt(SQR(by) + SQR(bz));
	const Real betay = by/b_perp;
	const Real betaz = bz/b_perp;
	const Real S = bx==0. ? 0. : (bx>0. ? 1. : -1.);
	int ighost = i-ie-1;
	//const Real dpdt = ncycle==0 ? (v(l, gas::prim::pressure(0), k, j, i-1) - shkp.pr*shkp.rho0)/dt
	//	: (v(l, gas::prim::pressure(0), k, j, i-1) - v(l, gas::boundary::pressure(0), k, j, i-1))/dt;
	if (ighost==0) {
          v(l, gas::boundary::density(0), k, j, i-1) = v(l, gas::prim::density(0), k, j, i-1);
          v(l, gas::boundary::velocity(0), k, j, i-1) = v(l, gas::prim::velocity(0), k, j, i-1);
          v(l, gas::boundary::velocity(1), k, j, i-1) = v(l, gas::prim::velocity(1), k, j, i-1);
          v(l, gas::boundary::velocity(2), k, j, i-1) = v(l, gas::prim::velocity(2), k, j, i-1);
          v(l, gas::boundary::pressure(0), k, j, i-1) = v(l, gas::prim::pressure(0), k, j, i-1);
        }

	const Real drhodx = (v(l, gas::boundary::density(0), k, j, i) -
                        v(l, gas::boundary::density(0), k, j, i-1))/dx[0];
	const Real dwdx = (v(l, gas::boundary::velocity(2), k, j, i) - 
			v(l, gas::boundary::velocity(2), k, j, i-1))/dx[0];
	const Real dvdx = (v(l, gas::boundary::velocity(1), k, j, i) -     
			v(l, gas::boundary::velocity(1), k, j, i-1))/dx[0];
	const Real dudx = (v(l, gas::boundary::velocity(0), k, j, i) -             
			v(l, gas::boundary::velocity(0), k, j, i-1))/dx[0];
	const Real dpdx = (v(l, gas::boundary::pressure(0), k, j, i) -        
			v(l, gas::boundary::pressure(0), k, j, i-1))/dx[0];
	const Real dBzdx = (v(l, gas::boundary::Bfield(0), k, j, i) -    
			v(l, gas::boundary::Bfield(0), k, j, i-1))/dx[0];
        const Real dBydx = (v(l, gas::boundary::Bfield(0), k, j, i) -         
			v(l, gas::boundary::Bfield(0), k, j, i-1))/dx[0];
	
	const Real dwdx_L1coef = (alphas*betaz*cs*S) / (2.*SQR(csound));         
	const Real dvdx_L1coef = (alphas*betay*cs*S) / (2.*SQR(csound));    
	const Real dudx_L1coef = (alphaf*cf) / (2.*SQR(csound));          
	const Real dpdx_L1coef = (alphaf) / (2.*SQR(csound)*dens);
        const Real dBzdx_L1coef = (alphas*betaz) / (2.*csound*std::sqrt(dens));
        const Real dBydx_L1coef = (alphas*betay) / (2.*csound*std::sqrt(dens));
	Real L1 = (vx-cf) * (dwdx_L1coef*dwdx + dvdx_L1coef*dvdx - dudx_L1coef*dudx
			+ dpdx_L1coef*dpdx + dBzdx_L1coef*dBzdx + dBydx_L1coef*dBydx);

	const Real dwdx_L2coef = 0.5*betay;
        const Real dvdx_L2coef = 0.5*betaz;
        const Real dBzdx_L2coef = (betay*S)/(2.*sqrt(dens));
        const Real dBydx_L2coef = (betaz*S)/(2.*sqrt(dens));
	Real L2 = (vx-cax) * (dwdx_L2coef*dwdx - dvdx_L2coef*dvdx + 
			dBzdx_L2coef*dBzdx - dBydx_L2coef*dBydx);

	const Real dwdx_L3coef = (alphaf*betaz*cf*S)/(2.*SQR(csound));
	const Real dvdx_L3coef = (alphaf*betay*cf*S)/(2.*SQR(csound));
	const Real dudx_L3coef = (alphas*cs)/(2.*SQR(csound));
	const Real dpdx_L3coef = (alphas)/(2.*SQR(csound)*dens);
	const Real dBzdx_L3coef = (alphaf*betaz)/(2.*csound*sqrt(dens));
	const Real dBydx_L3coef = (alphaf*betay)/(2.*csound*sqrt(dens));
	Real L3 = (vx-cs) * (-dwdx_L3coef*dwdx - dvdx_L3coef*dvdx - dudx_L3coef*dudx
			+ dpdx_L3coef*dpdx - dBzdx_L3coef*dBzdx - dBydx_L3coef*dBydx);

	const Real drhodx_L4coef = 1.;
	const Real dpdx_L4coef = 1./SQR(csound);
	Real L4 = vx * (drhodx_L4coef*drhodx - dpdx_L4coef*dpdx);

	const Real dwdx_L5coef = (alphaf*betaz*cf*S) / (2.*SQR(csound));
	const Real dvdx_L5coef = (alphaf*betay*cf*S) / (2.*SQR(csound));
	const Real dudx_L5coef = (alphas*cs) / (2.*SQR(csound));
	const Real dpdx_L5coef = (alphas) / (2.*SQR(csound)*dens);
	const Real dBzdx_L5coef = (alphaf*betaz) / (2.*csound*sqrt(dens));
	const Real dBydx_L5coef = (alphaf*betay) / (2.*csound*sqrt(dens));
	Real L5 = (vx + cs) * (dwdx_L5coef*dwdx + dvdx_L5coef*dvdx + dudx_L5coef*dudx
			+ dpdx_L5coef*dpdx - dBzdx_L5coef*dBzdx - dBydx_L5coef*dBydx);

	const Real dwdx_L6coef = 0.5*betay;
	const Real dvdx_L6coef = 0.5*betaz;
	const Real dBzdx_L6coef = (betay*S) / (2.*sqrt(dens));
	const Real dBydx_L6coef = (betaz*S) / (2.*sqrt(dens));
	Real L6 = (vx + cax) * (-dwdx_L6coef*dwdx + dvdx_L6coef*dvdx +
			dBzdx_L6coef*dBzdx - dBydx_L6coef*dBydx);

	const Real dwdx_L7coef = (alphas*betaz*cs*S) / (2.*SQR(csound));
	const Real dvdx_L7coef = (alphas*betay*cs*S) / (2.*SQR(csound));
	const Real dudx_L7coef = (alphaf*cf) / (2.*SQR(csound));
	const Real dpdx_L7coef = (alphaf) / (2.*SQR(csound)*dens);
	const Real dBzdx_L7coef = (alphas*betaz) / (2.*csound*sqrt(dens));
	const Real dBydx_L7coef = (alphas*betay) / (2.*csound*sqrt(dens));
	Real L7 = (vx + cf) * (-dwdx_L7coef*dwdx - dvdx_L7coef*dvdx + 
	     dudx_L7coef*dudx + dpdx_L7coef*dpdx + dBzdx_L7coef*dBzdx + dBydx_L7coef*dBydx);

	// To ensure non-reflecting, at least for the ideal MHD component
	//if (vx<cs)  L3 = 0.;
	//if (vx<cax) L2 = 0.;
	//if (vx<cf)  L1 = 0.;
	//if (vx<0.)  L4 = 0.;
	L1=0.; L2=0.; L3=0.;
	//if (vx<cf) L1 = K*(v(l, gas::boundary::pressure(0), k, j, i) - p_inf);
	// For constant pressure at outlet
	//L1 = -(alphas*L5/alphaf) - (alphas*L3/alphaf) - L7 - (1./(SQR(csound)*alphaf*dens))*dpdt;
	if (ncycle==0) {
	  v(l, gas::boundary::density(0), k, j, i) = shkp.rhor*shkp.rho0;
	  v(l, gas::boundary::pressure(0), k, j, i) = shkp.pr*shkp.rho0;
	  v(l, gas::boundary::velocity(0), k, j, i) = shkp.vxr;
	  v(l, gas::boundary::velocity(1), k, j, i) = 0.;
	  v(l, gas::boundary::velocity(2), k, j, i) = 0.;
	  v(l, gas::boundary::Bfield(1), k, j, i) = shkp.byr*sqrt(shkp.rho0);
	  v(l, gas::boundary::Bfield(2), k, j, i) = shkp.bzr*sqrt(shkp.rho0);
	} else {
          v(l, gas::boundary::density(0), k, j, i) -= dt * (
			alphaf*dens*(L7+L1) + alphas*dens*(L5+L3) + L4);
	  v(l, gas::boundary::pressure(0), k, j, i) -= dt * (
			dens*SQR(csound)*alphaf*(L7+L1) + dens*SQR(csound)*alphas*(L5+L3));
          v(l, gas::boundary::velocity(0), k, j, i) -= dt * (
			alphaf*cf*(L7-L1) + alphas*cs*(L5-L3));
          v(l, gas::boundary::velocity(1), k, j, i) -= dt * (
			-alphas*betay*cs*(L7-L1) + betaz*(L6-L2) + alphaf*betay*cf*(L5-L3));
          v(l, gas::boundary::velocity(2), k, j, i) -= dt * (
			-alphas*betaz*cs*(L7-L1) - betay*(L6-L2) + alphaf*betaz*cf*(L5-L3));
	  // Update CC Bfield then interpolate to FCC as easier
	  Real checkBy= v(l, gas::boundary::Bfield(1), k, j, i);
	  Real checkBz= v(l, gas::boundary::Bfield(2), k, j, i);
	  Real term1 = vx-cf==0. ? 0. : 
		  (sqrt(dens)/(vx-cf))*(csound*alphas*betay*vx - alphas*betay*bx*cs - alphaf*by*cf)*L1;
	  Real term2 = vx-cs==0. ? 0. :
		  (sqrt(dens)/(vx-cs))*(csound*alphaf*betay*vx + alphas*by*cs - alphaf*betay*bx*cf)*L3;
	  Real term3 = vx+cs==0. ? 0. :
		  (sqrt(dens)/(vx+cs))*(csound*alphaf*betay*vx - alphas*by*cs + alphaf*betay*bx*cf)*L5;
	  Real term4 = vx+cf==0. ? 0. :
		  (sqrt(dens)/(vx+cf))*(csound*alphas*betay*vx + alphas*betay*bx*cs + alphaf*by*cf)*L7;
	  v(l, gas::boundary::Bfield(1), k, j, i) -= dt * (term1 - term2 - term3 + term4 -
		 betaz*sqrt(dens)*(L2 + L6));
	  term1 = vx+cf==0. ? 0. :
		  (sqrt(dens)/(vx+cf))*(csound*alphas*betaz*vx + alphas*betaz*bx*cs + alphaf*bz*cf)*L7;
	  term2 = vx+cs==0. ? 0. :
		  (sqrt(dens)/(vx+cs))*(csound*alphaf*betaz*vx - alphas*bz*cs + alphaf*betaz*bx*cf)*L5;
	  term3 = vx-cs==0. ? 0. :
		  (sqrt(dens)/(vx-cs))*(csound*alphaf*betaz*vx + alphas*bz*cs - alphaf*betaz*bx*cf)*L3;
	  term4 = vx-cf==0. ? 0. :
		  (sqrt(dens)/(vx-cf))*(csound*alphas*betaz*vx - alphas*betaz*bx*cs - alphaf*bz*cf)*L1;
	  v(l, gas::boundary::Bfield(2), k, j, i) -= dt * (term1 - term2 - term3 + term4 +
		 betay*sqrt(dens)*(L2 + L6));
	}
	// Update primitive variables with boundary variables
	v(l, gas::prim::density(0), k, j, i) = v(l, gas::boundary::density(0), k, j, i);
	v(l, gas::prim::pressure(0), k, j, i) = v(l, gas::boundary::pressure(0), k, j, i);
	v(l, gas::prim::sie(0), k, j, i) = v(l, gas::boundary::pressure(0), k, j, i)/
                  (v(l, gas::boundary::density(0), k, j, i)*gm1);
	v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::boundary::velocity(0), k, j, i);
	v(l, gas::prim::velocity(1), k, j, i) = v(l, gas::boundary::velocity(1), k, j, i);
	v(l, gas::prim::velocity(2), k, j, i) = v(l, gas::boundary::velocity(2), k, j, i);
	v(l, gas::prim::Bfield(1), k, j, i) = v(l, gas::boundary::Bfield(1), k, j, i);
	v(l, gas::prim::Bfield(2), k, j, i) = v(l, gas::boundary::Bfield(2), k, j, i);
	/*v(l, gas::prim::density(0), k, j, i) = outflow ? v(l, gas::prim::density(0), k, j, iref) : shkp.rhor*shkp.rho0;
        v(l, gas::prim::pressure(0), k, j, i) = v(l, gas::boundary::pressure(0), k, j, i);
        v(l, gas::prim::sie(0), k, j, i) = v(l, gas::prim::pressure(0), k, j, i)/
                  (v(l, gas::prim::density(0), k, j, i)*gm1);
        v(l, gas::prim::velocity(0), k, j, i) = outflow ? v(l, gas::prim::velocity(0), k, j, iref) : shkp.vxr;
        v(l, gas::prim::velocity(1), k, j, i) = outflow ? v(l, gas::prim::velocity(1), k, j, iref) : 0.;
        v(l, gas::prim::velocity(2), k, j, i) = outflow ? v(l, gas::prim::velocity(2), k, j, iref) : 0.;
        v(l, gas::prim::Bfield(1), k, j, i) = outflow ? v(l, gas::prim::Bfield(1), k, j, iref) : shkp.byr*sqrt(shkp.rho0);
        v(l, gas::prim::Bfield(2), k, j, i) = outflow ? v(l, gas::prim::Bfield(2), k, j, iref) : shkp.bzr*sqrt(shkp.rho0);*/
	// For now just extrapolate Pe
	v(l, gas::prim::Pe(0), k, j, i)     = outflow ? v(l, gas::prim::Pe(0), k, j, iref) : shkp.per*shkp.rho0;
      });
 
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int ie_x = range_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, ie_x);
        });
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int ie_y = range_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const Real vx = v(l, gas::prim::velocity(0), k, j, ie_y);
        const bool outflow = (vx >= 0.);
        //v(l, TE::F2, gas::face::bfield(0), k, j, i) = 0.5*(
	//		v(l, gas::prim::Bfield(1), k, j, i) +
	//		v(l, gas::prim::Bfield(1), k, j, i + (i+1<=ie_y+nghosts)));
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = outflow ? 
			v(l, TE::F2, gas::face::bfield(0), k, j, ie_y) : shkp.byr*sqrt(shkp.rho0);
        });
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int ie_z = range_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
      	const Real vx = v(l, gas::prim::velocity(0), k, j, ie_z);
        const bool outflow = (vx >= 0.);
        //v(l, TE::F3, gas::face::bfield(0), k, j, i) = 0.5*(
	//		v(l, gas::prim::Bfield(2), k, j, i) +
	//		v(l, gas::prim::Bfield(2), k, j, i + (i+1<=ie_z+nghosts)));
	v(l, TE::F3, gas::face::bfield(0), k, j, i) = outflow ?
			v(l, TE::F3, gas::face::bfield(0), k, j, ie_z) : shkp.bzr*sqrt(shkp.rho0);
        });


  // YH: For now E & J are not reflecting as NRBC only applied to ideal MHD component.
  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int ieE_x = rangeE_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
	const Real vx = v(l, gas::prim::velocity(0), k, j, ieE_x);
        const bool outflow = (vx >= 0.);
        const int iref = ieE_x;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = outflow ? 
			v(l, TE::E1, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E1, gas::edge::J(0), k, j, i) = outflow ? 
			v(l, TE::E1, gas::edge::J(0), k, j, iref) : 0.;
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int ieE_y = rangeE_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const Real vx = v(l, gas::prim::velocity(0), k, j, ieE_y);
        const bool outflow = (vx >= 0.);
        const int iref = ieE_y;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = outflow ? 
			v(l, TE::E2, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E2, gas::edge::J(0), k, j, i) = outflow ? 
			v(l, TE::E2, gas::edge::J(0), k, j, iref) : 0.;
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int ieE_z = rangeE_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
      	const Real vx = v(l, gas::prim::velocity(0), k, j, ieE_z);
        const bool outflow = (vx >= 0.);
        const int iref = ieE_z;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = outflow ?
			v(l, TE::F3, gas::edge::Efield(0), k, j, iref) : 0.;
	v(l, TE::E3, gas::edge::J(0), k, j, i) = outflow ?
			v(l, TE::F3, gas::edge::J(0), k, j, iref) : 0.;
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::OrlanskiOuterX1()
//! \brief Sets BCs on +x boundary 
//  -> another type of nrbc but simpler and can applied to XMHD
//  -> To match OpenFoam: https://caefn.com/openfoam/bc-advective-wavetransmissive
//     https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/fields/fvPatchFields/derived/advective/advectiveFvPatchField.C
//     // If the wave is incoming set the speed to 0.
//     const scalarField w(Foam::max(advectionSpeed(), scalar(0)));
//  -> double alpha = c * dt / dx; // c from advectionSpeed() or estimated
//  -> phi[N+1] = phi[N] - alpha * (phi[N] - phi[N-1]);
//  -> phi[N+2] = phi[N+1] - alpha * (phi[N+1] - phi[N]);
//  -> From CFDonline, can set U to mean velocity too
template <Coordinates GEOM>
inline void OrlanskiOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto shkp = artemis_pkg->Param<BRIO_WU_REVERSED_Params>("brio_wu_reversed_params");
  const Real rho0 = shkp.rho0;
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  EOS eos_d;
  eos_d = gas_pkg->template Param<EOS>("eos_d");
  Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::prim::pressure, gas::prim::Bfield,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
		gas::boundary::density, gas::boundary::velocity,
		gas::boundary::pressure, gas::boundary::Bfield_fcc, gas::boundary::Bfield,
		gas::boundary::Efield, gas::boundary::J, gas::boundary::Pe>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  int nghosts = parthenon::Globals::nghost;
  Kokkos::View<Real*> adv("adv", nghosts);

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int ie = range.e;
  const Real dt = gas_pkg->Param<Real>("dt");
  const int ncycle = gas_pkg->Param<int>("ncycle");
  pmb->par_for_bndry(
      "InflowOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie;
        geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);           
	const auto &dx = coords.GetCellWidths();
	int ighost = i-ie-1;	
	if (ighost==0) {
          v(l, gas::boundary::density(0), k, j, i-1) = v(l, gas::prim::density(0), k, j, i-1);
          v(l, gas::boundary::velocity(0), k, j, i-1) = v(l, gas::prim::velocity(0), k, j, i-1);
          v(l, gas::boundary::velocity(1), k, j, i-1) = v(l, gas::prim::velocity(1), k, j, i-1);
          v(l, gas::boundary::velocity(2), k, j, i-1) = v(l, gas::prim::velocity(2), k, j, i-1);
          v(l, gas::boundary::pressure(0), k, j, i-1) = v(l, gas::prim::pressure(0), k, j, i-1);
	  v(l, gas::boundary::Bfield(0), k, j, i-1) = v(l, gas::prim::Bfield(0), k, j, i-1);
          v(l, gas::boundary::Bfield(1), k, j, i-1) = v(l, gas::prim::Bfield(1), k, j, i-1);
          v(l, gas::boundary::Bfield(2), k, j, i-1) = v(l, gas::prim::Bfield(2), k, j, i-1);
          v(l, gas::boundary::Pe(0), k, j, i-1) = v(l, gas::prim::Pe(0), k, j, i-1);
        } else {
	  v(l, gas::boundary::Bfield(0), k, j, i-1) = 0.5*(v(l, TE::F1, gas::boundary::Bfield_fcc(0), k, j, i-1)
			  + v(l, TE::F1, gas::boundary::Bfield_fcc(0), k, j, i));
          v(l, gas::boundary::Bfield(1), k, j, i-1) = 0.5*(v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i-1) 
			  + v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i));
          v(l, gas::boundary::Bfield(2), k, j, i-1) = 0.5*(v(l, TE::F3, gas::boundary::Bfield_fcc(0), k, j, i-1)
			  + v(l, TE::F3, gas::boundary::Bfield_fcc(0), k, j, i));
	}
	const Real vx = v(l, gas::boundary::velocity(0), k, j, i-1);
	const Real press = v(l, gas::boundary::pressure(0), k, j, i-1);
        const Real dens = v(l, gas::boundary::density(0), k, j, i-1);
        const Real csound = sqrt(gamma*press/dens);
        const Real bx = v(l, gas::boundary::Bfield(0), k, j, i-1);
        const Real by = v(l, gas::boundary::Bfield(1), k, j, i-1);
        const Real bz = v(l, gas::boundary::Bfield(2), k, j, i-1);
        const Real ca = std::sqrt((SQR(bx)+SQR(by)+SQR(bz))/dens);
        const Real cax = std::sqrt((SQR(bx))/dens);
        const Real cay = std::sqrt((SQR(by))/dens);
        const Real caz = std::sqrt((SQR(bz))/dens);
        const Real can = std::min(std::min(cax,cay),caz);
        const Real csa = SQR(csound) + SQR(ca);
        const Real acan = csound * can;
        const Real cf = std::sqrt(0.5*(csa + std::sqrt(SQR(csa) - 4*SQR(acan))));
	const Real ne = Z_ion * dens;
	const Real Jx = ighost==0 ? v(l, TE::E1, gas::edge::J(0), k, j, i-1) :
				    v(l, TE::E1, gas::boundary::J(0), k, j, i-1);
	const Real ue = vx - (J0/(e_charge*n0*char_speed))*(Jx/ne);	
        adv(ighost)=std::max(vx + std::max(cf,ue), 0.);

	const Real drhodx = (v(l, gas::boundary::density(0), k, j, i) -
                        v(l, gas::boundary::density(0), k, j, i-1))/dx[0];
	const Real dwdx = (v(l, gas::boundary::velocity(2), k, j, i) - 
			v(l, gas::boundary::velocity(2), k, j, i-1))/dx[0];
	const Real dvdx = (v(l, gas::boundary::velocity(1), k, j, i) -     
			v(l, gas::boundary::velocity(1), k, j, i-1))/dx[0];
	const Real dudx = (v(l, gas::boundary::velocity(0), k, j, i) -             
			v(l, gas::boundary::velocity(0), k, j, i-1))/dx[0];
	const Real dpdx = (v(l, gas::boundary::pressure(0), k, j, i) -        
			v(l, gas::boundary::pressure(0), k, j, i-1))/dx[0];
	const Real dPedx = (v(l, gas::boundary::Pe(0), k, j, i) -
                        v(l, gas::boundary::Pe(0), k, j, i-1))/dx[0];
	
	if (ncycle==0) {
          v(l, gas::boundary::density(0), k, j, i) = shkp.rhor*shkp.rho0;
          v(l, gas::boundary::pressure(0), k, j, i) = shkp.pr*shkp.rho0;
          v(l, gas::boundary::velocity(0), k, j, i) = shkp.vxr;
          v(l, gas::boundary::velocity(1), k, j, i) = 0.;
          v(l, gas::boundary::velocity(2), k, j, i) = 0.;
	  v(l, gas::boundary::Pe(0), k, j, i) = shkp.per*shkp.rho0;
        } else {
          v(l, gas::boundary::density(0), k, j, i)  -= adv(ighost)*dt*drhodx;
          v(l, gas::boundary::pressure(0), k, j, i) -= adv(ighost)*dt*dpdx;
          v(l, gas::boundary::velocity(0), k, j, i) -= adv(ighost)*dt*dudx;
          v(l, gas::boundary::velocity(1), k, j, i) -= adv(ighost)*dt*dvdx;
          v(l, gas::boundary::velocity(2), k, j, i) -= adv(ighost)*dt*dwdx;
	  v(l, gas::boundary::Pe(0), k, j, i)      -= adv(ighost)*dt*dPedx;
	}

	v(l, gas::prim::density(0), k, j, i) = v(l, gas::boundary::density(0), k, j, i);
        v(l, gas::prim::pressure(0), k, j, i) = v(l, gas::boundary::pressure(0), k, j, i);
        v(l, gas::prim::sie(0), k, j, i) = v(l, gas::boundary::pressure(0), k, j, i)/
                  (v(l, gas::boundary::density(0), k, j, i)*gm1);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::boundary::velocity(0), k, j, i);
        v(l, gas::prim::velocity(1), k, j, i) = v(l, gas::boundary::velocity(1), k, j, i);
        v(l, gas::prim::velocity(2), k, j, i) = v(l, gas::boundary::velocity(2), k, j, i);
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int ie_x = range_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = ie_x;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, i-1);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int ie_y = range_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = ie_y;
	geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);
	const auto &dx = coords.GetCellWidths();                             
	const int ighost = i- iref - 1;
	if (ighost==0) v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i-1) = v(l, TE::F2, gas::face::bfield(0), k, j, i-1);
	const Real dBydx = (v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i) -
                        v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i-1))/dx[0];
	if (ncycle==0) {
	  v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i) = shkp.byr*sqrt(shkp.rho0);
	} else {
	  v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i) -= adv(ighost)*dt*dBydx; 
	}
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::boundary::Bfield_fcc(0), k, j, i);
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int ie_z = range_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie_z;
	geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);
	const auto &dx = coords.GetCellWidths();                                 
	const int ighost = i - iref - 1;
	if (ighost==0) v(l, TE::F3, gas::boundary::Bfield_fcc(0), k, j, i-1) = v(l, TE::F3, gas::face::bfield(0), k, j, i-1);
	const Real dBzdx = (v(l, TE::F3, gas::face::bfield(0), k, j, i) -
                        v(l, TE::F3, gas::face::bfield(0), k, j, i-1))/dx[0];
	if (ncycle==0) {
	  v(l, TE::F3, gas::boundary::Bfield_fcc(0), k, j, i) = shkp.bzr*sqrt(shkp.rho0);
	} else {
    	  v(l, TE::F3, gas::boundary::Bfield_fcc(0), k, j, i) -= adv(ighost)*dt*dBzdx; 
	}
	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::boundary::Bfield_fcc(0), k, j, i);
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int ieE_x = rangeE_x.e;
  pmb->par_for_bndry(
      "SymOuterX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = ieE_x;
	geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);
	const auto &dx = coords.GetCellWidths();                
        const int ighost = i - iref - 1;
	if (ighost==0) {
	  v(l, TE::E1, gas::boundary::Efield(0), k, j, i-1) = v(l, TE::E1, gas::edge::Efield(0), k, j, i-1);
	  v(l, TE::E1, gas::boundary::J(0), k, j, i-1) = v(l, TE::E1, gas::edge::J(0), k, j, i-1);
	}	
        const Real dExdx = (v(l, TE::E1, gas::boundary::Efield(0), k, j, i) -
			v(l, TE::E1, gas::boundary::Efield(0), k, j, i-1)) / dx[0];
	const Real dJxdx = (v(l, TE::E1, gas::boundary::J(0), k, j, i) -  
		       	v(l, TE::E1, gas::boundary::J(0), k, j, i-1)) / dx[0];	
	if (ncycle==0) {
	  v(l, TE::E1, gas::boundary::Efield(0), k, j, i) = 0.;
	  v(l, TE::E1, gas::boundary::J(0), k, j, i) = 0.;
	} else { 
          v(l, TE::E1, gas::boundary::Efield(0), k, j, i) -= adv(ighost)*dt*dExdx;
	  v(l, TE::E1, gas::boundary::J(0), k, j, i) -= adv(ighost)*dt*dJxdx;
	}
	v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::boundary::Efield(0), k, j, i);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::boundary::J(0), k, j, i);
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int ieE_y = rangeE_y.e;
  pmb->par_for_bndry(
      "SymOuterX1_F2", nb, IndexDomain::outer_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_y;
	geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);
	const auto &dx = coords.GetCellWidths();
	const int ighost = i - iref - 1;
	if (ighost==0) {
          v(l, TE::E2, gas::boundary::Efield(0), k, j, i-1) = v(l, TE::E2, gas::edge::Efield(0), k, j, i-1);
          v(l, TE::E2, gas::boundary::J(0), k, j, i-1) = v(l, TE::E2, gas::edge::J(0), k, j, i-1);
        }
	const Real dEydx = (v(l, TE::E2, gas::boundary::Efield(0), k, j, i) -
			v(l, TE::E2, gas::boundary::Efield(0), k, j, i-1)) / dx[0];
	const Real dJydx = (v(l, TE::E2, gas::boundary::J(0), k, j, i) -  
		       	v(l, TE::E2, gas::boundary::J(0), k, j, i-1)) / dx[0];	
	if (ncycle==0) {
	  v(l, TE::E2, gas::boundary::Efield(0), k, j, i) = 0.;
	  v(l, TE::E2, gas::boundary::J(0), k, j, i) = 0.;
	} else {
          v(l, TE::E2, gas::boundary::Efield(0), k, j, i) -= adv(ighost)*dt*dEydx;
	  v(l, TE::E2, gas::boundary::J(0), k, j, i) -= adv(ighost)*dt*dJydx;
	}
	v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::boundary::Efield(0), k, j, i);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::boundary::J(0), k, j, i);
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int ieE_z = rangeE_z.e;
  pmb->par_for_bndry(
      "SymOuterX1_F3", nb, IndexDomain::outer_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_z;
	geometry::Coords<GEOM> coords(v.GetCoordinates(l), k, j, i);
	const auto &dx = coords.GetCellWidths();
	const int ighost = i - iref - 1;
	if (ighost==0) {
          v(l, TE::E3, gas::boundary::Efield(0), k, j, i-1) = v(l, TE::E3, gas::edge::Efield(0), k, j, i-1);
          v(l, TE::E3, gas::boundary::J(0), k, j, i-1) = v(l, TE::E3, gas::edge::J(0), k, j, i-1);
        }
	const Real dEzdx = (v(l, TE::E3, gas::boundary::Efield(0), k, j, i) -
			v(l, TE::E3, gas::boundary::Efield(0), k, j, i-1)) / dx[0];
	const Real dJzdx = (v(l, TE::E3, gas::boundary::J(0), k, j, i) -  
		       	v(l, TE::E3, gas::boundary::J(0), k, j, i-1)) / dx[0];	
	if (ncycle==0) {
          v(l, TE::E3, gas::boundary::Efield(0), k, j, i) = 0.;
          v(l, TE::E3, gas::boundary::J(0), k, j, i) = 0.;
        } else {
	  v(l, TE::E3, gas::boundary::Efield(0), k, j, i) -= adv(ighost)*dt*dEzdx;
	  v(l, TE::E3, gas::boundary::J(0), k, j, i) -= adv(ighost)*dt*dJzdx;
	}
	v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::E3, gas::boundary::Efield(0), k, j, i);
        v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::E3, gas::boundary::J(0), k, j, i);
        });
  return;
}



} // namespace brio_wu_reversed
#endif // PGEN_BRIO_WU_REVERSED_HPP_
