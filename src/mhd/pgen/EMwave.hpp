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
#ifndef PGEN_EMWAVE_HPP_
#define PGEN_EMWAVE_HPP_
//! \file EMwave.hpp
//! \brief
//!

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "mhd/extended/defs.hpp"

using ArtemisUtils::EOS;

namespace EMwave {

struct EMWAVE_Params {
  Real rho0, dB, mode, Lx;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_EMWAVE_Params
//! \brief Extracts EMwave parameters from ParameterInput.
inline void Init_EMWAVE_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("EMwave_params"))) {
    EMWAVE_Params EMwave_params;
    // Left and right region
    EMwave_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    EMwave_params.dB = pin->GetOrAddReal("problem", "dB", 0.001);
    EMwave_params.mode = pin->GetOrAddReal("problem", "mode", 1.);
    EMwave_params.Lx = pin->GetOrAddReal("problem", "Lx", 0.3);
    params.Add("EMwave_params", EMwave_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::EMWAVE()
//! \brief Sets initial conditions for EMwave problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The hubba_hall mhd problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The hubba_hall mhd problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",2.0);
  Real gm1 = gamma - 1.;
  const int ndim = ProblemDimension(pin);

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
	  gas::prim::Bfield,
	  gas::prim::Pe>( 
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // EMWAVE parameters
  auto shkp = artemis_pkg->Param<EMWAVE_Params>("EMwave_params");
  const Real deltaB = shkp.dB;

  const Real Lx = shkp.Lx;
  const Real Temp = 2.7; // Temperature of interstellar medium

  // Setup EMwave state
  pmb->par_for(
      "EMwave", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
	const auto &dx = coords.GetCellWidths();
	const Real rho = 1.*shkp.rho0;
	const Real P = rho*Temp*eV_to_K/T0;
        v(0, gas::prim::density(0), k, j, i)  = rho;
        v(0, gas::prim::velocity(0), k, j, i) = 0.0;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
	v(0, gas::prim::Pe(0), k, j, i) = 0.5*P;
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes},{parthenon::PDOpt::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "EMwave", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = 0.;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "EMwave", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = 0.;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "EMwave", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        const Real bz = deltaB*std::cos(2.*M_PI*shkp.mode*xf[0]/Lx) / c_per_v;
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = bz;
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
	vEJ(0, TE::E1, gas::edge::Efield(), k, j, i) = 0.;
        vEJ(0, TE::E1, gas::edge::J(), k, j, i) = 0.;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E2);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E2);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX2();
	const Real ey = deltaB*std::cos(2.*M_PI*shkp.mode*xf[0]/Lx);
	vEJ(0, TE::E2, gas::edge::Efield(), k, j, i) = ey;
        vEJ(0, TE::E2, gas::edge::J(), k, j, i) = 0.;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E3);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E3);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	vEJ(0, TE::E3, gas::edge::Efield(), k, j, i) = 0.;
        vEJ(0, TE::E3, gas::edge::J(), k, j, i) = 0.;
      });

}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicInnerX1()
//! \brief Sets BCs on -x boundary 
template <Coordinates GEOM>
inline void PeriodicInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<EMWAVE_Params>("EMwave_params");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
                                                 gas::source::mom,
                                                 gas::source::ener,
                                                 gas::source::Se>(mbd);

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
  const int ie = range.e;
  const int shift = ie-is+1;
  pmb->par_for_bndry(
      "InflowInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i + shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
	// For source term
        for (int im=0; im<3; im++)
          v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, j, iref);
        v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, j, iref);
        v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, j, iref);
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int is_x = range_x.s;
  const int ie_x = range_x.e;
  const int shift_f1 = ie_x - is_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i + shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int is_y = range_y.s;
  const int ie_y = range_y.e;
  const int shift_f2 = ie_y - is_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = i + shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, j, iref);
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int is_z = range_z.s;
  const int ie_z = range_z.e;
  const int shift_f3 = ie_z - is_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i + shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, j, iref);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, j, iref);
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int isE_x = rangeE_x.s;
  const int ieE_x = rangeE_x.e;
  const int shift_e1 = ieE_x - isE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i + shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int isE_y = rangeE_y.s;
  const int ieE_y = rangeE_y.e;
  const int shift_e2 = ieE_y - isE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i + shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int isE_z = rangeE_z.s;
  const int ieE_z = rangeE_z.e;
  const int shift_e3 = ieE_z - isE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i + shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, j, iref);
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicOuterX1()
//! \brief Sets BCs on +x boundary 
template <Coordinates GEOM>
inline void PeriodicOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<EMWAVE_Params>("EMwave_params");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
                                                 gas::source::mom,
                                                 gas::source::ener,
                                                 gas::source::Se>(mbd);

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
  const int ie = range.e;
  const int shift = ie-is+1;
  pmb->par_for_bndry(
      "InflowInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i - shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
	// For source term
        for (int im=0; im<3; im++)
          v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, j, iref);
        v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, j, iref);
        v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, j, iref);
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int is_x = range_x.s;
  const int ie_x = range_x.e;
  const int shift_f1 = ie_x - is_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i - shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int is_y = range_y.s;
  const int ie_y = range_y.e;
  const int shift_f2 = ie_y - is_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = i - shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, j, iref);
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int is_z = range_z.s;
  const int ie_z = range_z.e;
  const int shift_f3 = ie_z - is_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = i - shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, j, iref);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, j, iref);
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int isE_x = rangeE_x.s;
  const int ieE_x = rangeE_x.e;
  const int shift_e1 = ieE_x - isE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = i - shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int isE_y = rangeE_y.s;
  const int ieE_y = rangeE_y.e;
  const int shift_e2 = ieE_y - isE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i - shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, j, iref);
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int isE_z = rangeE_z.s;
  const int ieE_z = rangeE_z.e;
  const int shift_e3 = ieE_z - isE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = i - shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, j, iref);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, j, iref);
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicInnerX2()
//! \brief Sets BCs on -y boundary 
template <Coordinates GEOM>
inline void PeriodicInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<EMWAVE_Params>("EMwave_params");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
						 gas::source::mom,
						 gas::source::ener,
						 gas::source::Se>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;
  const int je = range.e;
  const int shift = je-js+1;
  pmb->par_for_bndry(
      "InflowInnerX2", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j + shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, jref, i);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, jref, i);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, jref, i);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, jref, i);
	// For source term
	for (int im=0; im<3; im++)
	  v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, jref, i);
	v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, jref, i);
	v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, jref, i);
      });
  
  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int js_x = range_x.s;
  const int je_x = range_x.e;
  const int shift_f1 = je_x - js_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j + shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, jref, i);
	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int js_y = range_y.s;
  const int je_y = range_y.e;
  const int shift_f2 = je_y - js_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x2, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = j + shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, jref, i);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, jref, i);
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int js_z = range_z.s;
  const int je_z = range_z.e;
  const int shift_f3 = je_z - js_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x2, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j + shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, jref, i);
  	});

  const auto &rangeE_x = bounds.GetBoundsJ(IndexDomain::interior, TE::E1);
  const int jsE_x = rangeE_x.s;
  const int jeE_x = rangeE_x.e;
  const int shift_e1 = jeE_x - jsE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int jref = j + shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_y = bounds.GetBoundsJ(IndexDomain::interior, TE::E2);
  const int jsE_y = rangeE_y.s;
  const int jeE_y = rangeE_y.e;
  const int shift_e2 = jeE_y - jsE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x2, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j + shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_z = bounds.GetBoundsJ(IndexDomain::interior, TE::E3);
  const int jsE_z = rangeE_z.s;
  const int jeE_z = rangeE_z.e;
  const int shift_e3 = jeE_z - jsE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x2, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j + shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, jref, i);
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PeriodicOuterX2()
//! \brief Sets BCs on +y boundary 
template <Coordinates GEOM>
inline void PeriodicOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<EMWAVE_Params>("EMwave_params");

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie, gas::prim::Pe,
						 gas::face::bfield,
						 gas::edge::Efield,gas::edge::J,
						 gas::source::Bfield,
                                                 gas::source::mom,
                                                 gas::source::ener,
                                                 gas::source::Se>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  if (v.GetMaxNumberOfVars() == 0) return;

  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");
  const Real gm1=gamma-1.;
  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;
  const int je = range.e;
  const int shift = je-js+1;
  pmb->par_for_bndry(
      "InflowInnerX2", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j - shift; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, jref, i);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, jref, i);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, jref, i);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, jref, i);
	// For source term
        for (int im=0; im<3; im++)
          v(l, gas::source::mom(im), k, j, i) = v(l, gas::source::mom(im), k, jref, i);
        v(l, gas::source::ener(0), k, j, i) = v(l, gas::source::ener(0), k, jref, i);
        v(l, gas::source::Se(0), k, j, i) = v(l, gas::source::Se(0), k, jref, i);
      });
  
  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int js_x = range_x.s;
  const int je_x = range_x.e;
  const int shift_f1 = je_x - js_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_f1;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);
	v(l, TE::F1, gas::source::Bfield(0), k, j, i) = v(l, TE::F1, gas::source::Bfield(0), k, jref, i);
	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int js_y = range_y.s;
  const int je_y = range_y.e;
  const int shift_f2 = je_y - js_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x2, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = j - shift_f2;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, jref, i);
        v(l, TE::F2, gas::source::Bfield(0), k, j, i) = v(l, TE::F2, gas::source::Bfield(0), k, jref, i);
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int js_z = range_z.s;
  const int je_z = range_z.e;
  const int shift_f3 = je_z - js_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x2, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = j - shift_f3;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);
        v(l, TE::F3, gas::source::Bfield(0), k, j, i) = v(l, TE::F3, gas::source::Bfield(0), k, jref, i);
  	});

  const auto &rangeE_x = bounds.GetBoundsJ(IndexDomain::interior, TE::E1);
  const int jsE_x = rangeE_x.s;
  const int jeE_x = rangeE_x.e;
  const int shift_e1 = jeE_x - jsE_x + 1;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_e1;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E1, gas::edge::J(0), k, j, i) = v(l, TE::E1, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_y = bounds.GetBoundsJ(IndexDomain::interior, TE::E2);
  const int jsE_y = rangeE_y.s;
  const int jeE_y = rangeE_y.e;
  const int shift_e2 = jeE_y - jsE_y + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x2, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_e2;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E2, gas::edge::J(0), k, j, i) = v(l, TE::E2, gas::edge::J(0), k, jref, i);
        });
  const auto &rangeE_z = bounds.GetBoundsJ(IndexDomain::interior, TE::E3);
  const int jsE_z = rangeE_z.s;
  const int jeE_z = rangeE_z.e;
  const int shift_e3 = jeE_z - jsE_z + 1;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x2, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = j - shift_e3;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, jref, i);
	v(l, TE::E3, gas::edge::J(0), k, j, i) = v(l, TE::F3, gas::edge::J(0), k, jref, i);
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Computes errors in field diffusion mhd solution by subtracting current solution from
//! ICs, and outputting errors to file. Problem must be run for an integer number of wave
//! periods.
template <Coordinates GEOM>
inline void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  using parthenon::MakePackDescriptor;
  const int nhydro = 5;
  const int nvars = nhydro + 3 + 7;

  // packing and capture variables for kernel
  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  static auto desc =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get());
  auto vmag = desc.GetPack(md.get());
  static auto desc2 =
      MakePackDescriptor<gas::edge::Efield>((pmb->resolved_packages).get());
  auto vE = desc2.GetPack(md.get());
  const int ndim = ProblemDimension(pin);

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  // EMWAVE parameters
  auto shkp = artemis_pkg->Param<EMWAVE_Params>("EMwave_params");
  const Real deltaB = shkp.dB;
  const Real Lx = shkp.Lx;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, TE::F2);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, TE::F2);
  ArtemisUtils::array_type<Real, 2> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "EMwave", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    ArtemisUtils::array_type<Real, 2> &lsum) {
        // Capture coordinates this Meshblock
        geometry::Coords<GEOM> coords(vmag.GetCoordinates(b), k, j, i);
	const auto &xf = coords.GetFaceCenterX2();
        const auto &xv = coords.GetCellCenter();
        Real x1v = xv[0];
        Real x2v = xv[1];
        Real x3v = xv[2];
        Real vol = coords.Volume();
	const auto &dx = coords.GetCellWidths();
	const Real x2 = SQR(xf[0]);
        const Real y2 = ndim>1 ? SQR(xf[1]) : 0.;
        const Real z2 = ndim>2 ? SQR(xf[2]) : 0.;
        const Real bz = deltaB*std::cos((2.*M_PI*shkp.mode/Lx)*(xf[0]-c_per_v*tm.time)) 
			/ c_per_v;
	const Real ey = deltaB*std::cos((2.*M_PI*shkp.mode/Lx)*(xf[0]-c_per_v*tm.time));
        lsum.myArray[0] += abs(vmag(0, TE::F3, gas::face::bfield(0), k, j, i) - bz)*dx[0];
	lsum.myArray[1] += abs(vE(0,   TE::E2, gas::edge::Efield(0), k, j, i) - ey)*dx[0];
      },
      ArtemisUtils::SumMyArray<Real, Kokkos::HostSpace, 2>(l1_err));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &(l1_err.myArray[0]), 2, MPI_PARTHENON_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  Real rms_err = 0.0; 
  for (int i = 0; i < 2; ++i) rms_err += SQR(l1_err.myArray[i]);
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (parthenon::Globals::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("parthenon/job", "problem_id"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1         ");
      std::fprintf(pfile, "Bz_L1         Ey_L1");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmesh->mesh_size.nx(X1DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X3DIR));
    std::fprintf(pfile, "  %05d  %e ", tm.ncycle, rms_err);
    for (int i = 0; i < 2; ++i) {
      std::fprintf(pfile, "  %e", l1_err.myArray[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}


} // namespace EMwave
#endif // PGEN_EMWAVE_HPP_
