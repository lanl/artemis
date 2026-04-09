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
#ifndef PGEN_PLANAR_FOIL_HPP_
#define PGEN_PLANAR_FOIL_HPP_
//! \file planar_foil.hpp
//! \brief
//!

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include <random>

using ArtemisUtils::EOS;

namespace planar_foil {

struct PLANAR_FOIL_Params {
  Real rho_f;   // Liner
  Real rho_o;  // Outeside Coronal layer
  Real rho_amp;
  Real Temp;
  Real loc1_foil, loc2_foil;
  Real bx, by, bz;
  Real tau;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_PLANAR_FOIL_Params
//! \brief Extracts planar_foil parameters from ParameterInput.
inline void Init_PLANAR_FOIL_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("planar_foil_params"))) {
    PLANAR_FOIL_Params planar_foil_params;
    planar_foil_params.rho_f = pin->GetOrAddReal("problem", "rho_f", 1.0);
    planar_foil_params.rho_o = pin->GetOrAddReal("problem", "rho_o", 1.e-9);
    planar_foil_params.rho_amp = pin->GetOrAddReal("problem", "rho_amp", 0.2);
    planar_foil_params.bx = pin->GetOrAddReal("problem", "bx", 0.0);
    planar_foil_params.by = pin->GetOrAddReal("problem", "by", 0.0);
    planar_foil_params.bz = pin->GetOrAddReal("problem", "bz", 0.0);
    planar_foil_params.tau = pin->GetOrAddReal("problem", "tau", 340);
    // Temperature in terms of eV
    planar_foil_params.Temp = pin->GetOrAddReal("problem", "Temp", 1.0);
    // Location of foil
    planar_foil_params.loc1_foil = pin->GetOrAddReal("problem", "loc1_foil", 8);
    planar_foil_params.loc2_foil = pin->GetOrAddReal("problem", "loc2_foil", 32);
    params.Add("planar_foil_params", planar_foil_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PLANAR_FOIL()
//! \brief Sets initial conditions for planar_foil problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The planar foil problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The planar foil problem does not permit dust hydrodynamics!");
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

  // PLANAR_FOIL parameters
  auto shkp = artemis_pkg->Param<PLANAR_FOIL_Params>("planar_foil_params");

  // Setup planar_foil state
  pmb->par_for(
      "planar_foil", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
	const auto &dx = coords.GetCellWidths();
	bool in_foil1 = std::abs(xi[0] - shkp.loc1_foil) <= 0.5 * dx[0];
	bool in_foil2 = std::abs(xi[0] - shkp.loc2_foil) <= 0.5 * dx[0];
	Real rho, P;
	if (in_foil1 or in_foil2) {
	  rho = shkp.rho_f;
	  // Random distribution as pertubation for density
	  std::mt19937 rng(i+j); // random seed
	  std::uniform_real_distribution<double> dist(-1.0,1.0);
 	  double xi = dist(rng);
	  rho *= (1.0+shkp.rho_amp*xi); // YH: add random perturbation
	  P = shkp.rho_f*(shkp.Temp*eV_to_K/T0);
	} else {
	  rho = shkp.rho_o;
	  P = shkp.rho_f*(shkp.Temp*eV_to_K/T0);
	}

	const Real Pe = 0.75 * P; // as 3 electrons per Al ions
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = 0.0;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
	v(0, gas::prim::Pe(0), k, j, i) = Pe;
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
		      {parthenon::Metadata::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "planar_foil", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = shkp.bx;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "planar_foil", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = shkp.by;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "planar_foil", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = shkp.bz;
      });

  static auto desc_EJ =
      MakePackDescriptor<gas::edge::Efield,
      			 gas::edge::J>((pmb->resolved_packages).get());
  auto vEJ = desc_EJ.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E1);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E1);
  pmb->par_for(
      "planar_foil", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
      "planar_foil", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
      "planar_foil", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
//! \fn void ProblemGenerator::CurrentInnerX1()
//! \brief Sets BCs on -x boundary 
template <Coordinates GEOM>
inline void CurrentInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<PLANAR_FOIL_Params>("planar_foil_params");

  // For the magnetic field
  const double time = gas_pkg->Param<double>("track_time");
  const double I0 = 1.e6;
  const double tau = 100.e-9/t0;
  const double s = std::sin(2.*pi*time/tau);
  const double current = I0 * s * s;
  const double Ly = 40.e-6/L0;
  const double Bz_dimension = - mu0 * current / (2. * Ly); 
  const double Bz = Bz_dimension / B0;
  /*const double tau = shkp.tau*1.e-9/t0;
  const double s = (time<0.25*tau) ? std::sin(2.*pi*time/tau) : 1.;
  const double By_dimension = 3.14875135722 * s * s;
  const double By = By_dimension / B0;*/

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
  const int is = range.s;
  pmb->par_for_bndry(
      "InflowInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = is; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, j, iref);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, j, iref);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, j, iref);
        v(l, gas::prim::velocity(1), k, j, i) =  v(l, gas::prim::velocity(1), k, j, iref);
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, j, iref);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, j, iref);
      });
  
  const auto &range_x = bounds.GetBoundsI(IndexDomain::interior, TE::F1);
  const int is_x = range_x.s;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = is_x;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int is_y = range_y.s;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = is_y;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref);
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int is_z = range_z.s;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = is_z;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = Bz; //v(l, TE::F3, gas::face::bfield(0), k, j, iref); 
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int isE_x = rangeE_x.s;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = isE_x;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, j, iref);
	geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBz_dy = -(v(0, TE::F3, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[1];
        Real dBy_dz = -(v(0, TE::F2, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E1, gas::edge::J(0), k, j, i) = dBz_dy - dBy_dz;
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int isE_y = rangeE_y.s;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = isE_y;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, j, iref);
	geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBz_dx = -(v(0, TE::F3, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dz = -(v(0, TE::F1, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E2, gas::edge::J(0), k, j, i) = -dBz_dx + dBx_dz;
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int isE_z = rangeE_z.s;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = isE_z;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, j, iref);
	geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBy_dx = -(v(0, TE::F2, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dy = -(v(0, TE::F1, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[1];
	v(l, TE::E3, gas::edge::J(0), k, j, i) = dBy_dx - dBx_dy;
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::CurrentOuterX1()
//! \brief Sets BCs on +x boundary 
template <Coordinates GEOM>
inline void CurrentOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<PLANAR_FOIL_Params>("planar_foil_params");

  // For the magnetic field
  const double time = gas_pkg->Param<double>("track_time");
  const double I0 = 1.e6;
  const double tau = 100.e-9/t0;
  const double s = std::sin(2.*pi*time/tau);
  const double current = I0 * s * s;
  const double Ly = 40.e-6/L0;
  const double Bz_dimension = mu0 * current / (2. * Ly);
  const double Bz = Bz_dimension / B0;
  /*const double tau = shkp.tau*1.e-9/t0;
  const double s = (time<0.25*tau) ? std::sin(2.*pi*time/tau) : 1.;
  const double By_dimension = 3.14875135722 * s * s;
  const double By = -By_dimension / B0;*/

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
      "InflowInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
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
      "SymInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Reflect across x=0
        const int iref = ie_x;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, j, iref);
	});
  const auto &range_y = bounds.GetBoundsI(IndexDomain::interior, TE::F2);
  const int ie_y = range_y.e;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x1, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int iref = ie_y;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, j, iref);
  	});
  const auto &range_z = bounds.GetBoundsI(IndexDomain::interior, TE::F3);
  const int ie_z = range_z.e;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x1, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int iref = ie_z;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = Bz; //v(l, TE::F3, gas::face::bfield(0), k, j, iref); 
  	});

  const auto &rangeE_x = bounds.GetBoundsI(IndexDomain::interior, TE::E1);
  const int ieE_x = rangeE_x.e;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_x;
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = v(l, TE::E1, gas::edge::Efield(0), k, j, iref);
	geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBz_dy = -(v(0, TE::F3, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[1];
        Real dBy_dz = -(v(0, TE::F2, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E1, gas::edge::J(0), k, j, i) = dBz_dy - dBy_dz;
        });
  const auto &rangeE_y = bounds.GetBoundsI(IndexDomain::interior, TE::E2);
  const int ieE_y = rangeE_y.e;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x1, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_y;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, j, iref);
	geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBz_dx = -(v(0, TE::F3, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dz = -(v(0, TE::F1, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E2, gas::edge::J(0), k, j, i) = -dBz_dx + dBx_dz;
        });
  const auto &rangeE_z = bounds.GetBoundsI(IndexDomain::interior, TE::E3);
  const int ieE_z = rangeE_z.e;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x1, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int iref = ieE_z;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, j, iref);
	geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBy_dx = -(v(0, TE::F2, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dy = -(v(0, TE::F1, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[1];
	v(l, TE::E3, gas::edge::J(0), k, j, i) = dBy_dx - dBx_dy;
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ConductInnerX2()
//! \brief Sets BCs on -y boundary 
template <Coordinates GEOM>
inline void ConductInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<PLANAR_FOIL_Params>("planar_foil_params");

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
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;
  pmb->par_for_bndry(
      "InflowInnerX2", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = js; 

        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, jref, i);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, jref, i);
        v(l, gas::prim::velocity(1), k, j, i) =  0.; // Impenetrable
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, jref, i);
      });
 
  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int js_x = range_x.s;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = js_x;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);
	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int js_y = range_y.s;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x2, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = js_y;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, jref, i);
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int js_z = range_z.s;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x2, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = js_z;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);
  	});

  const auto &rangeE_x = bounds.GetBoundsJ(IndexDomain::interior, TE::E1);
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &dx = coords.GetCellWidths();
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = 0.;
	Real dBz_dy = -(v(0, TE::F3, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[1];
        Real dBy_dz = -(v(0, TE::F2, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E1, gas::edge::J(0), k, j, i) = dBz_dy - dBy_dz;
        });
  const auto &rangeE_y = bounds.GetBoundsJ(IndexDomain::interior, TE::E2);
  const int jEs_y = rangeE_y.s;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::inner_x2, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &dx = coords.GetCellWidths();
	const int jref = jEs_y;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, jref, i);
	Real dBz_dx = -(v(0, TE::F3, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dz = -(v(0, TE::F1, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E2, gas::edge::J(0), k, j, i) =  -dBz_dx + dBx_dz;
        });
  const auto &rangeE_z = bounds.GetBoundsJ(IndexDomain::interior, TE::E3);
  const int jEs_z = rangeE_z.s;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::inner_x2, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &dx = coords.GetCellWidths();
	const int jref = jEs_z;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, jref, i);
	Real dBy_dx = -(v(0, TE::F2, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dy = -(v(0, TE::F1, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[1];
	v(l, TE::E3, gas::edge::J(0), k, j, i) = dBy_dx - dBx_dy;
        });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ConductOuterX2()
//! \brief Sets BCs on +y boundary 
template <Coordinates GEOM>
inline void ConductOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pmb = mbd->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  auto shkp = artemis_pkg->Param<PLANAR_FOIL_Params>("planar_foil_params");

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
  const auto &range = bounds.GetBoundsJ(IndexDomain::interior, TE::CC);
  const int js = range.s;
  const int je = range.e;
  pmb->par_for_bndry(
      "InflowInnerX2", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::CC,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = je; 
        v(l, gas::prim::density(0), k, j, i) = v(l, gas::prim::density(0), k, jref, i);
        v(l, gas::prim::sie(0), k, j, i)     = v(l, gas::prim::sie(0), k, jref, i);
        v(l, gas::prim::velocity(0), k, j, i) = v(l, gas::prim::velocity(0), k, jref, i);
        v(l, gas::prim::velocity(1), k, j, i) = 0.0; // Impenetrable
        v(l, gas::prim::velocity(2), k, j, i) =  v(l, gas::prim::velocity(2), k, jref, i);
	v(l, gas::prim::Pe(0), k, j, i)     = v(l, gas::prim::Pe(0), k, jref, i);
      });
  
  const auto &range_x = bounds.GetBoundsJ(IndexDomain::interior, TE::F1);
  const int je_x = range_x.e;
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::F1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        const int jref = je_x;
	v(l, TE::F1, gas::face::bfield(0), k, j, i) = v(l, TE::F1, gas::face::bfield(0), k, jref, i);
	});
  const auto &range_y = bounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  const int je_y = range_y.e;
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x2, TE::F2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    	const int jref = je_y;
	v(l, TE::F2, gas::face::bfield(0), k, j, i) = v(l, TE::F2, gas::face::bfield(0), k, jref, i);
  	});
  const auto &range_z = bounds.GetBoundsJ(IndexDomain::interior, TE::F3);
  const int je_z = range_z.e;
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x2, TE::F3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
	const int jref = je_z;
    	v(l, TE::F3, gas::face::bfield(0), k, j, i) = v(l, TE::F3, gas::face::bfield(0), k, jref, i);
  	});

  const auto &rangeE_x = bounds.GetBoundsJ(IndexDomain::interior, TE::E1);
  pmb->par_for_bndry(
      "SymInnerX1", nb, IndexDomain::outer_x2, parthenon::TopologicalElement::E1,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
        v(l, TE::E1, gas::edge::Efield(0), k, j, i) = 0.;
	Real dBz_dy = -(v(0, TE::F3, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[1];
        Real dBy_dz = -(v(0, TE::F2, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E1, gas::edge::J(0), k, j, i) = dBz_dy - dBy_dz;
        });
  const auto &rangeE_y = bounds.GetBoundsJ(IndexDomain::interior, TE::E2);
  pmb->par_for_bndry(
      "SymInnerX1_F2", nb, IndexDomain::outer_x2, TE::E2,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	const int jref = rangeE_y.e;
        v(l, TE::E2, gas::edge::Efield(0), k, j, i) = v(l, TE::E2, gas::edge::Efield(0), k, jref, i);
	Real dBz_dx = -(v(0, TE::F3, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F3, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dz = -(v(0, TE::F1, gas::face::bfield(0), k - (ndim > 2 and k>0), j, i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[2];
	v(l, TE::E2, gas::edge::J(0), k, j, i) = -dBz_dx + dBx_dz;
        });
  const auto &rangeE_z = bounds.GetBoundsJ(IndexDomain::interior, TE::E3);
  pmb->par_for_bndry(
      "SymInnerX1_F3", nb, IndexDomain::outer_x2, TE::E3,
      coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	const int jref = rangeE_z.e;
        v(l, TE::E3, gas::edge::Efield(0), k, j, i) = v(l, TE::F3, gas::edge::Efield(0), k, jref, i);
	Real dBy_dx = -(v(0, TE::F2, gas::face::bfield(0), k, j, i - (i>0))
                        - v(0, TE::F2, gas::face::bfield(0), k, j, i)) / dx[0];
        Real dBx_dy = -(v(0, TE::F1, gas::face::bfield(0), k, j - (ndim > 1 and j>0), i)
                        - v(0, TE::F1, gas::face::bfield(0), k, j, i)) / dx[1];
	v(l, TE::E3, gas::edge::J(0), k, j, i) = dBy_dx - dBx_dy;
        });
  return;
}


} // namespace planar_foil
#endif // PGEN_PLANAR_FOIL_HPP_
