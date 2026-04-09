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
#ifndef PGEN_GEM_RECONNECTION_MHD_HPP_
#define PGEN_GEM_RECONNECTION_MHD_HPP_
//! \file GEM_reconnection_mhd.hpp
//! \brief
//!

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "cmath"

using ArtemisUtils::EOS;

namespace GEM_reconnection_mhd {

struct GEM_RECONNECTION_MHD_Params {
  Real vA, w0;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_GEM_RECONNECTION_MHD_Params
//! \brief Extracts GEM_reconnection_mhd parameters from ParameterInput.
inline void Init_GEM_RECONNECTION_MHD_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("GEM_reconnection_mhd_params"))) {
    GEM_RECONNECTION_MHD_Params GEM_reconnection_mhd_params;
    GEM_reconnection_mhd_params.vA = pin->GetOrAddReal("problem", "vA", 1.1e6);
    // COME back next time as need change L0
    params.Add("GEM_reconnection_mhd_params", GEM_reconnection_mhd_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::GEM_RECONNECTION_MHD()
//! \brief Sets initial conditions for GEM_reconnection_mhd problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The GEM_reconnection mhd problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The GEM_reconnection mhd problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",1.6666666666666666666666667);
  Real gm1 = gamma - 1.;

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
	  gas::prim::Bfield>( // YH: remove after fcc bfield work
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // GEM_RECONNECTION_MHD parameters
  auto shkp = artemis_pkg->Param<GEM_RECONNECTION_MHD_Params>("GEM_reconnection_mhd_params");

  // Setup GEM_reconnection_mhd state
  pmb->par_for(
      "GEM_reconnection_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const Real rho = 1. + (1. / shkp.beta_up) * SQR(1. / cosh(xi[1])) ;
        const Real P = 0.5 * (shkp.beta_up + SQR(1. / cosh(xi[1])));
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = 0.0;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes},{parthenon::PDOpt::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "GEM_reconnection_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	const Real x = xf[0];
	const Real y = xf[1];
	const Real bx = tanh(y);
	const Real delta_bx = - 0.06 * 0.5 * y *exp(-(x*x + y*y)/4.0);
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = bx + delta_bx;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "GEM_reconnection_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
	const Real x = xf[0];
	const Real y = xf[1];
	const Real delta_by = 0.06 * 0.5 * x *exp(-(x*x + y*y)/4.0);
        const Real by = 0.0;
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = by + delta_by;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "GEM_reconnection_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        const Real bz = 0.0;
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = bz;
      });

}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SymInnerX1()
//! \brief Sets BCs on -x boundary in magnetic reconnection
template <Coordinates GEOM>
inline void SymInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie,
						 gas::face::bfield>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsI(IndexDomain::interior, TE::CC);
  const int is = range.s;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	// Reflect across x=0
	const int iref = 2 * is - i - 1;  // mirror index (e.g., i=-1 → iref=0)

	// Copy even variables
        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);

        // Velocity components
        v(l, gas::prim::velocity(0), k, j, i) = -v(l, gas::prim::velocity(0), k, j, iref);  // v_x: odd
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);  // v_y: even
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);  // v_z: even
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int is_x = range_x.s;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = 2 * is - i - 1;  // mirror index (e.g., i=-1 → iref=0)
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = -v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int is_y = range_y.s;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = 2 * is_y - i - 1;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref); 
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int is_z = range_z.s;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = 2 * is_z - i - 1;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, j, iref); 
  	});

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SymInnerX2()
//! \brief Sets BCs on -y boundary in magnetic reconnection
template <Coordinates GEOM>
inline void SymInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie,
						 gas::face::bfield>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;

  pmb->par_for_bndry(
    "SymInnerX2_CC", nb, IndexDomain::inner_x2, TE::CC,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = 2 * js - j - 1;  // Mirror index across y=0
	// Even scalars
    	v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
    	v(l, gas::prim::sie(0),     k, j, i) = v(l, gas::prim::sie(0),     k, jref, i);
    	// Velocity components
    	v(l, gas::prim::velocity(0), k, j, i) =  v(l, gas::prim::velocity(0), k, jref, i);  // vx: even
    	v(l, gas::prim::velocity(1), k, j, i) = -v(l, gas::prim::velocity(1), k, jref, i);  // vy: odd
    	v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);  // vz: even
  	});

  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int js_x = range_x.s;
  pmb->par_for_bndry(
    "SymInnerX2_F1", nb, IndexDomain::inner_x2, TE::F1,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = 2 * js_x - j - 1;
    	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);  
  	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int js_y = range_y.s;
  pmb->par_for_bndry(
    "SymInnerX2_F2", nb, IndexDomain::inner_x2, TE::F2,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = 2 * js_y - j - 1;
    	v(l, TE::F2, gas::face::bfield(0), k, j, i) = -v(l, TE::F2, gas::face::bfield(0), k, jref, i); 
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int js_z = range_z.s;
  pmb->par_for_bndry(
    "SymInnerX2_F3", nb, IndexDomain::inner_x2, TE::F3,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = 2 * js_z - j - 1;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);  
  	});
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::CondOuterX2()
//! \brief Sets BCs on +y boundary in magnetic reconnection
template <Coordinates GEOM>
inline void CondOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie,
						 gas::face::bfield>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);  
  const int je = range.e;

  pmb->par_for_bndry(
    "CondOuterX2_CC", nb, IndexDomain::outer_x2, TE::CC,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = 2 * je - j + 1;  
	// Even scalars
    	v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
    	v(l, gas::prim::sie(0),     k, j, i) = v(l, gas::prim::sie(0),     k, jref, i);
    	// Velocity components
    	v(l, gas::prim::velocity(0), k, j, i) =  v(l, gas::prim::velocity(0), k, jref, i);  
    	v(l, gas::prim::velocity(1), k, j, i) = 0.;
    	v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);  
  	});

  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int je_x = range_x.e;
  pmb->par_for_bndry(
    "CondOuterX2_F1", nb, IndexDomain::outer_x2, TE::F1,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = 2 * je_x - j + 1;
    	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);  
  	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int je_y = range_y.e;
  pmb->par_for_bndry(
    "CondOuterX2_F2", nb, IndexDomain::outer_x2, TE::F2,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	v(l, TE::F2, gas::face::bfield(0), k, j, i) = 0.0; 
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int je_z = range_z.e;
  pmb->par_for_bndry(
    "CondOuterX2_F3", nb, IndexDomain::outer_x2, TE::F3,
    coarse, fine,
    KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = 2 * je_z - j + 1;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);  
  	});
  return;
}


} // namespace GEM_reconnection_mhd
#endif // PGEN_GEM_RECONNECTION_MHD_HPP_
