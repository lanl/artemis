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
#ifndef PGEN_BRIO_WU_Y_HPP_
#define PGEN_BRIO_WU_Y_HPP_
//! \file brio_wu_y.hpp
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

namespace brio_wu_y {

struct BRIO_WU_Y_Params {
  Real rho0;
  Real rhol, vxl, pl, pel, bxl, byl, bzl;
  Real rhor, vxr, pr, per, bxr, byr, bzr;
  Real xdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_BRIO_WU_Y_Params
//! \brief Extracts brio_wu_y parameters from ParameterInput.
inline void Init_BRIO_WU_Y_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("brio_wu_y_params"))) {
    BRIO_WU_Y_Params brio_wu_y_params;
    brio_wu_y_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    brio_wu_y_params.rhol = pin->GetOrAddReal("problem", "rhol", 1.0);
    brio_wu_y_params.vxl = pin->GetOrAddReal("problem", "vxl", 0.0);
    brio_wu_y_params.pl = pin->GetOrAddReal("problem", "pl", 1.0);
    brio_wu_y_params.pel = pin->GetOrAddReal("problem", "pel", 0.5);
    brio_wu_y_params.byl = pin->GetOrAddReal("problem", "byl", 0.75);
    brio_wu_y_params.bxl = pin->GetOrAddReal("problem", "bxl", 1.0);
    brio_wu_y_params.bzl = pin->GetOrAddReal("problem", "bzl", 0.0);
    brio_wu_y_params.rhor = pin->GetOrAddReal("problem", "rhor", 0.125);
    brio_wu_y_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.0);
    brio_wu_y_params.pr = pin->GetOrAddReal("problem", "pr", 0.1);
    brio_wu_y_params.per = pin->GetOrAddReal("problem", "per", 0.05);
    brio_wu_y_params.byr = pin->GetOrAddReal("problem", "byr", 0.75);
    brio_wu_y_params.bxr = pin->GetOrAddReal("problem", "bxr", -1.0);
    brio_wu_y_params.bzr = pin->GetOrAddReal("problem", "bzr", 0.0);
    brio_wu_y_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.5);
    params.Add("brio_wu_y_params", brio_wu_y_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BRIO_WU_Y()
//! \brief Sets initial conditions for brio_wu_y problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  int nghosts = parthenon::Globals::nghost;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The brio wu y problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The brio wu y problem does not permit dust hydrodynamics!");
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

  // BRIO_WU_Y parameters
  auto shkp = artemis_pkg->Param<BRIO_WU_Y_Params>("brio_wu_y_params");
  const Real rho0 = shkp.rho0;
  // Setup brio_wu_y state
  pmb->par_for(
      "brio_wu_y", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const bool upwind = (xi[1] <= shkp.xdisc);
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
      "brio_wu_y", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	const bool upwind = (xf[1] <= shkp.xdisc);
	// YH: add magnetic field - refer to problem gen from fine_adv
	const Real bx = upwind ? shkp.bxl : shkp.bxr;
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = bx*sqrt(rho0);
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "brio_wu_y", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
        const bool upwind = (xf[1] <= shkp.xdisc);
        const Real by = upwind ? shkp.byl : shkp.byr;
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = by*sqrt(rho0);
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "brio_wu_y", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        const bool upwind = (xf[1] <= shkp.xdisc);
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

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::InflowOuterX1()
//! \brief Sets BCs on -x boundary 
template <Coordinates GEOM>
inline void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto shkp = artemis_pkg->Param<BRIO_WU_Y_Params>("brio_wu_y_params");
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
        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
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








} // namespace brio_wu_y
#endif // PGEN_BRIO_WU_Y_HPP_
