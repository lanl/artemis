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
#ifndef XMHD_XMHD_HPP_
#define XMHD_XMHD_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

#include "defs.hpp"
#include "resistivity.hpp"
#include "interp.hpp"
#include "../integrator/defs_int.hpp"
#include "collision.hpp"
using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace XMHD {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::DeepCopyConservedFaceData
//! \brief
inline TaskStatus DeepCopyConservedEdgeData(MeshData<Real> *to, MeshData<Real> *from) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;

  std::vector<MetadataFlag> flags({Metadata::Edge, Metadata::Conserved});
  static auto desc = MakePackDescriptor<any>(to, flags);
  const auto vt = desc.GetPack(to);
  const auto vf = desc.GetPack(from);

  const auto ibe_E1 = to->GetBoundsI(IndexDomain::entire,TE::E1);
  const auto jbe_E1 = to->GetBoundsJ(IndexDomain::entire,TE::E1);
  const auto kbe_E1 = to->GetBoundsK(IndexDomain::entire,TE::E1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DeepCopyConservedData_E1", parthenon::DevExecSpace(), 0,
      to->NumBlocks() - 1, kbe_E1.s, kbe_E1.e, jbe_E1.s, jbe_E1.e, ibe_E1.s, ibe_E1.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
        for (int n = vt.GetLowerBound(b); n <= vt.GetUpperBound(b); ++n) {
          vt(b, TE::E1, n, k, j, i) = vf(b, TE::E1, n, k, j, i);
        }
      });

  const auto ibe_E2 = to->GetBoundsI(IndexDomain::entire,TE::E2);
  const auto jbe_E2 = to->GetBoundsJ(IndexDomain::entire,TE::E2);
  const auto kbe_E2 = to->GetBoundsK(IndexDomain::entire,TE::E2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DeepCopyConservedData_E2", parthenon::DevExecSpace(), 0,
      to->NumBlocks() - 1, kbe_E2.s, kbe_E2.e, jbe_E2.s, jbe_E2.e, ibe_E2.s, ibe_E2.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;   
	for (int n = vt.GetLowerBound(b); n <= vt.GetUpperBound(b); ++n) {
          vt(b, TE::E2, n, k, j, i) = vf(b, TE::E2, n, k, j, i);
        }         
      });

  const auto ibe_E3 = to->GetBoundsI(IndexDomain::entire,TE::E3);
  const auto jbe_E3 = to->GetBoundsJ(IndexDomain::entire,TE::E3);
  const auto kbe_E3 = to->GetBoundsK(IndexDomain::entire,TE::E3);
  parthenon::par_for( 
      DEFAULT_LOOP_PATTERN, "DeepCopyConservedData_E3", parthenon::DevExecSpace(), 0,
      to->NumBlocks() - 1, kbe_E3.s, kbe_E3.e, jbe_E3.s, jbe_E3.e, ibe_E3.s, ibe_E3.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;            
	for (int n = vt.GetLowerBound(b); n <= vt.GetUpperBound(b); ++n) {
          vt(b, TE::E3, n, k, j, i) = vf(b, TE::E3, n, k, j, i);
        }                   
      });
  return TaskStatus::complete;
}



//==============================================================================================
//! \fn  TaskStatus XMHD::CalculateEMF
//! \brief
template <Coordinates GEOM>
TaskStatus CalculateEdgeFluxes(MeshData<Real> *u0) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const int ndim = pm->ndim;
  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  auto Metadata_Bfield = Metadata::GetUserFlag("Bfield");
  auto Metadata_Efield = Metadata::GetUserFlag("Efield");
  auto Metadata_Jcurrent = Metadata::GetUserFlag("Jcurrent");

  std::vector<MetadataFlag> flagsm1({Metadata::Cell, Metadata::Derived, 
		  Metadata::GetUserFlag("Density")});                   
  static auto descm1 = MakePackDescriptor<any>(u0, flagsm1);             
  const auto vm1 = descm1.GetPack(u0);
  std::vector<MetadataFlag> flags0({Metadata::Cell, Metadata::Derived, 
		  Metadata::GetUserFlag("Velfield")});
  static auto desc0 = MakePackDescriptor<any>(u0, flags0); 
  const auto v0 = desc0.GetPack(u0);
  std::vector<MetadataFlag> flagsP({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Pressure")});
  static auto descP = MakePackDescriptor<any>(u0, flagsP);
  const auto v0_P = descP.GetPack(u0);
  // YH: (a) retrieve XMHD-related package
  std::vector<MetadataFlag> flags1({Metadata::Cell, Metadata::Derived, 
		  Metadata::GetUserFlag("electron")});                 
  static auto desc1 = MakePackDescriptor<any>(u0, flags1);       
  const auto v0_Pe = desc1.GetPack(u0);
  std::vector<MetadataFlag> flags2({Metadata::Face, Metadata::Conserved, Metadata::WithFluxes 
		  , Metadata_Bfield});
  static auto desc2 = MakePackDescriptor<any>(u0, flags2, {parthenon::PDOpt::WithFluxes});
  const auto v0_fcc = desc2.GetPack(u0);

  std::vector<MetadataFlag> flags3({Metadata::Edge, Metadata::Conserved, Metadata_Efield});
  static auto desc3 = MakePackDescriptor<any>(u0, flags3);
  const auto E0_ecc = desc3.GetPack(u0);
  std::vector<MetadataFlag> flags4({Metadata::Edge, Metadata::Conserved, Metadata_Jcurrent});
  static auto desc4 = MakePackDescriptor<any>(u0, flags4);
  const auto J0_ecc = desc4.GetPack(u0);

  // YH: (b) retrieve XMHD-flux 
  std::vector<MetadataFlag> flags5({Metadata::Edge, Metadata::GetUserFlag("Efield_flux")});
  static auto desc5 = MakePackDescriptor<any>(u0, flags5);
  const auto E0_flux = desc5.GetPack(u0);
  std::vector<MetadataFlag> flags5p5({Metadata::Cell, Metadata::GetUserFlag("Efield"), Metadata::WithFluxes});
  static auto desc5p5 = MakePackDescriptor<any>(u0, flags5p5, {parthenon::PDOpt::WithFluxes});
  const auto E0 = desc5p5.GetPack(u0);

  std::vector<MetadataFlag> flags6({Metadata::Node, Metadata_Jcurrent});
  static auto desc6 = MakePackDescriptor<any>(u0, flags6);
  const auto J0_flux = desc6.GetPack(u0);
  std::vector<MetadataFlag> flags7({Metadata::Cell, Metadata_Jcurrent, Metadata::WithFluxes});
  static auto desc7 = MakePackDescriptor<any>(u0, flags7, {parthenon::PDOpt::WithFluxes});
  const auto J0 = desc7.GetPack(u0);

 
  const auto ibE1 = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jbE1 = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kbE1 = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbE1.s-2*(ndim>2), kbE1.e+2*(ndim>2), jbE1.s-2*(ndim>1), jbE1.e+2*(ndim>1), ibE1.s-2, ibE1.e+2,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
	geometry::Coords<GEOM> coords(v0_fcc.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

	// YH: Compute fluxes for Efield 
	// YH: Limitation of using curl to compute is that it will lead to boundary issues as cells at the start or end will need boundary cells and outflow BC will render the gradient zero. Thus, need to extrapolate outside the boundaries!!!!.
	Real dBz_dy, dBy_dz;
	dBz_dy = -(v0_fcc(b, TE::F3, 0, k, j - (ndim > 1), i)
                        - v0_fcc(b, TE::F3, 0, k, j, i)) / dx[1];
        dBy_dz = -(v0_fcc(b, TE::F2, 0, k - (ndim > 2), j, i)
                        - v0_fcc(b, TE::F2, 0, k, j, i)) / dx[2];
        E0_flux(b, TE::E1, 0, k, j, i) = SQR(c_per_v) * (dBz_dy - dBy_dz);
      });

  const auto ibE2 = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jbE2 = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kbE2 = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbE2.s-2*(ndim>2), kbE2.e+2*(ndim>2), jbE2.s-2*(ndim>1), jbE2.e+2*(ndim>1), ibE2.s-2, ibE2.e+2,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_fcc.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBz_dx, dBx_dz;
        dBz_dx = -(v0_fcc(b, TE::F3, 0, k, j, i - 1)
                        - v0_fcc(b, TE::F3, 0, k, j, i)) / dx[0];
        dBx_dz = -(v0_fcc(b, TE::F1, 0, k - (ndim > 2), j, i)
                        - v0_fcc(b, TE::F1, 0, k, j, i)) / dx[2];
        E0_flux(b, TE::E2, 0, k, j, i) = SQR(c_per_v) * (-dBz_dx + dBx_dz);
      });

  const auto ibE3 = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jbE3 = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kbE3 = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbE3.s-2*(ndim>2), kbE3.e+2*(ndim>2), jbE3.s-2*(ndim>1), jbE3.e+2*(ndim>1), ibE3.s-2, ibE3.e+2,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_fcc.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBy_dx, dBx_dy;
        dBy_dx = -(v0_fcc(b, TE::F2, 0, k, j, i - 1)
                        - v0_fcc(b, TE::F2, 0, k, j, i)) / dx[0];
        dBx_dy = -(v0_fcc(b, TE::F1, 0, k, j - (ndim > 1), i)
                        - v0_fcc(b, TE::F1, 0, k, j, i)) / dx[1];
        E0_flux(b, TE::E3, 0, k, j, i) = SQR(c_per_v) * (dBy_dx - dBx_dy);
      });

  const auto ibnn = u0->GetBoundsI(IndexDomain::interior, TE::NN);
  const auto jbnn = u0->GetBoundsJ(IndexDomain::interior, TE::NN);
  const auto kbnn = u0->GetBoundsK(IndexDomain::interior, TE::NN);
  const int ibnn_s = ibnn.s-2, ibnn_e = ibnn.e+2;
  const int jbnn_s = jbnn.s-2*(ndim>1), jbnn_e = jbnn.e+2*(ndim>1);
  const int kbnn_s = kbnn.s-2*(ndim>2), kbnn_e = kbnn.e+2*(ndim>2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbnn_s, kbnn_e, jbnn_s, jbnn_e, ibnn_s, ibnn_e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_fcc.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	// YH : try flux from CC RS - which is better
        // 0=Fx, 1=Gx, 2=Hx
        J0_flux(b, TE::NN, 0, k, j, i) = 0.25 * (
		            J0.flux(b, X1DIR, 0, k                   , j                   , i)
		          + J0.flux(b, X1DIR, 0, k                   , j - (ndim>1 and j>0), i)
		          + J0.flux(b, X1DIR, 0, k - (ndim>2 and k>0), j                   , i)
		          + J0.flux(b, X1DIR, 0, k - (ndim>2 and k>0), j - (ndim>1 and j>0), i));
	//if (abs(J0.flux(b, X1DIR, 0, k, j, i))>1.e-6 and i>=20 and i<=27) 
	//  std::cout<<"At i = "<<i<<", Jflux at X1 face = "<<J0.flux(b, X1DIR, 0, k, j, i)<<std::endl;
        J0_flux(b, TE::NN, 3, k, j, i) = 0.25 * (
                            J0.flux(b, X1DIR, 1, k                   , j                   , i)
                          + J0.flux(b, X1DIR, 1, k                   , j - (ndim>1 and j>0), i)
                          + J0.flux(b, X1DIR, 1, k - (ndim>2 and k>0), j                   , i)
                          + J0.flux(b, X1DIR, 1, k - (ndim>2 and k>0), j - (ndim>1 and j>0), i));
        J0_flux(b, TE::NN, 6, k, j, i) = 0.25 * (
                            J0.flux(b, X1DIR, 2, k                   , j                   , i)
                          + J0.flux(b, X1DIR, 2, k                   , j - (ndim>1 and j>0), i)
                          + J0.flux(b, X1DIR, 2, k - (ndim>2 and k>0), j                   , i)
                          + J0.flux(b, X1DIR, 2, k - (ndim>2 and k>0), j - (ndim>1 and j>0), i));
        // 3=Fy, 4=Gy, 5=Hy
        J0_flux(b, TE::NN, 1, k, j, i) = ndim<1 ? 0. : 0.25 * (
		            J0.flux(b, X2DIR, 0, k                   , j, i        )
		          + J0.flux(b, X2DIR, 0, k                   , j, i - (i>0))
		          + J0.flux(b, X2DIR, 0, k - (ndim>2 and k>0), j, i        )
		          + J0.flux(b, X2DIR, 0, k - (ndim>2 and k>0), j, i - (i>0)));
        J0_flux(b, TE::NN, 4, k, j, i) = ndim<1 ? 0. : 0.25 * (
                            J0.flux(b, X2DIR, 1, k                   , j, i        )
                          + J0.flux(b, X2DIR, 1, k                   , j, i - (i>0))
                          + J0.flux(b, X2DIR, 1, k - (ndim>2 and k>0), j, i        )
                          + J0.flux(b, X2DIR, 1, k - (ndim>2 and k>0), j, i - (i>0)));
        J0_flux(b, TE::NN, 7, k, j, i) = ndim<1 ? 0. : 0.25 * (
                            J0.flux(b, X2DIR, 2, k                   , j, i        )
                          + J0.flux(b, X2DIR, 2, k                   , j, i - (i>0))
                          + J0.flux(b, X2DIR, 2, k - (ndim>2 and k>0), j, i        )
                          + J0.flux(b, X2DIR, 2, k - (ndim>2 and k>0), j, i - (i>0)));
        // 6=Fz, 7=Gz, 8=Hz
        J0_flux(b, TE::NN, 2, k, j, i) = ndim<2 ? 0. : 0.25 * (
		            J0.flux(b, X3DIR, 0, k, j                   , i        )
		          + J0.flux(b, X3DIR, 0, k, j                   , i - (i>0))
		          + J0.flux(b, X3DIR, 0, k, j - (ndim>1 and j>0), i        )
		          + J0.flux(b, X3DIR, 0, k, j - (ndim>1 and j>0), i - (i>0)));
        J0_flux(b, TE::NN, 5, k, j, i) = ndim<2 ? 0. : 0.25 * (
                            J0.flux(b, X3DIR, 1, k, j                   , i        )
                          + J0.flux(b, X3DIR, 1, k, j                   , i - (i>0))
                          + J0.flux(b, X3DIR, 1, k, j - (ndim>1 and j>0), i        )
                          + J0.flux(b, X3DIR, 1, k, j - (ndim>1 and j>0), i - (i>0)));
        J0_flux(b, TE::NN, 8, k, j, i) = ndim<2 ? 0. : 0.25 * (
                            J0.flux(b, X3DIR, 2, k, j                   , i        )
                          + J0.flux(b, X3DIR, 2, k, j                   , i - (i>0))
                          + J0.flux(b, X3DIR, 2, k, j - (ndim>1 and j>0), i        )
                          + J0.flux(b, X3DIR, 2, k, j - (ndim>1 and j>0), i - (i>0)));
	
  });

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::InitEfield
//! \brief
template <Coordinates GEOM>
TaskStatus InitEfield(MeshData<Real> *u0, MeshData<Real> *u1) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();

  // Packing and indexing
  std::vector<MetadataFlag> flags({Metadata::Cell, Metadata::Conserved, Metadata::WithFluxes});
  static auto desc = MakePackDescriptor<any>(u0, flags, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc.GetPack(u0);
  const auto ib = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kb = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;
  // YH: (a) edge Efield & J
  std::vector<MetadataFlag> flags_E({Metadata::Edge, Metadata::Conserved, Metadata::GetUserFlag("Efield")});
  static auto desc_E = MakePackDescriptor<any>(u0, flags_E);
  const auto v0_E = desc_E.GetPack(u0);
  static auto desc_E1 = MakePackDescriptor<any>(u1, flags_E);
  const auto v1_E = desc_E.GetPack(u1);

  int ib_1s = ib.s - 1,        ib_1e = ib.e + 1;
  int jb_1s = jb.s - (ndim>1), jb_1e = jb.e + (ndim>1);
  int kb_1s = kb.s - (ndim>2), kb_1e = kb.e + (ndim>2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "InitEdge_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_1s, kb_1e, jb_1s, jb_1e, ib_1s, ib_1e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
	const auto &dx = coords.GetCellWidths();
	v0_E(b, TE::E1, 0, k, j, i) = v0.flux(b, TE::E1, 0, k, j, i);
	v1_E(b, TE::E1, 0, k, j, i) = v0.flux(b, TE::E1, 0, k, j, i);
  });

  const auto iby = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jby = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kby = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  int ib_2s = iby.s - 1,        ib_2e = iby.e + 1;
  int jb_2s = jby.s - (ndim>1), jb_2e = jby.e + (ndim>1);
  int kb_2s = kby.s - (ndim>2), kb_2e = kby.e + (ndim>2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "InitEdge_y", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_2s, kb_2e, jb_2s, jb_2e, ib_2s, ib_2e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
	const auto &dx = coords.GetCellWidths();
	v0_E(b, TE::E2, 0, k, j, i) = v0.flux(b, TE::E2, 0, k, j, i);
	v1_E(b, TE::E2, 0, k, j, i) = v0.flux(b, TE::E2, 0, k, j, i);
  });

  const auto ibz = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jbz = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kbz = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  int ib_3s = ibz.s - 1,        ib_3e = ibz.e + 1;
  int jb_3s = jbz.s - (ndim>1), jb_3e = jbz.e + (ndim>1);
  int kb_3s = kbz.s - (ndim>2), kb_3e = kbz.e + (ndim>2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "InitEdge_z", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_3s, kb_3e, jb_3s, jb_3e, ib_3s, ib_3e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
	const auto &dx = coords.GetCellWidths();
	v0_E(b, TE::E3, 0, k, j, i) = v0.flux(b, TE::E3, 0, k, j, i);
	v1_E(b, TE::E3, 0, k, j, i) = v0.flux(b, TE::E3, 0, k, j, i);
      });
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::ApplyUpdateEdge
//! \brief
template <Coordinates GEOM>
TaskStatus ApplyUpdateEdge(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();

  // Extract integrator weights
  const Real gam0 = integrator->gam0[stage - 1];
  const Real gam1 = integrator->gam1[stage - 1];       
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;

  // Packing and indexing
  std::vector<MetadataFlag> flags({Metadata::Face, Metadata::Conserved, Metadata::WithFluxes});
  static auto desc = MakePackDescriptor<any>(u0, flags, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kb = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;
  // YH: (a) edge Efield & J
  std::vector<MetadataFlag> flags_E({Metadata::Edge, Metadata::Conserved, Metadata::GetUserFlag("Efield")});
  static auto desc_E = MakePackDescriptor<any>(u0, flags_E);
  const auto v0_E = desc_E.GetPack(u0);
  const auto v1_E = desc_E.GetPack(u1);
  std::vector<MetadataFlag> flags_J({Metadata::Edge, Metadata::Conserved, Metadata::GetUserFlag("Jcurrent")});
  static auto desc_J = MakePackDescriptor<any>(u0, flags_J);
  const auto v0_J = desc_J.GetPack(u0);
  const auto v1_J = desc_J.GetPack(u1);
  // YH: (b) E & J flux at node
  std::vector<MetadataFlag> flags_Ef({Metadata::Edge, Metadata::GetUserFlag("Efield_flux")});
  static auto desc_Ef = MakePackDescriptor<any>(u0, flags_Ef);
  const auto v0_Eflux = desc_Ef.GetPack(u0);
  std::vector<MetadataFlag> flags_Jf({Metadata::Node, Metadata::GetUserFlag("Jcurrent")});
  static auto desc_Jf = MakePackDescriptor<any>(u0, flags_Jf);
  const auto v0_Jflux = desc_Jf.GetPack(u0);

  int ib_1s = ib.s, ib_1e = ib.e;
  int jb_1s = jb.s, jb_1e = jb.e;
  int kb_1s = kb.s, kb_1e = kb.e;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Apply_expUpdateEdge_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_1s-(ndim>2), kb_1e+(ndim>2), jb_1s-(ndim>1), jb_1e+(ndim>1), ib_1s-1, ib_1e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
	const auto &dx = coords.GetCellWidths();

	// YH: (a) for Efield
	v0_E(b, TE::E1, 0, k, j, i) = gam0 * v0_E(b, TE::E1, 0, k, j, i) + gam1 * v1_E(b, TE::E1, 0, k, j, i) 
		+ v0_Eflux(b, TE::E1, 0, k, j, i) * beta_dt;
	// YH: (b) for J
	// YH: I thinkd div.F is 1D as like a 1D problem since for edge along x, it is only connected to two node x distance from each other. these two nodes does not vary along y and z thus, only flux from F but no G and H so is like a 1D problem.
	Real dFx = (v0_Jflux(b, TE::NN, 0, k, j, i + 1) - v0_Jflux(b, TE::NN, 0, k, j, i)) / dx[0];
	Real dGx = (v0_Jflux(b, TE::NN, 1, k, j + (ndim>1), i) - 
		    v0_Jflux(b, TE::NN, 1, k, j - (ndim>1), i)) / (2.*dx[1]);
	Real dGx_ip1 = (v0_Jflux(b, TE::NN, 1, k, j + (ndim>1), i + 1) -                         
			v0_Jflux(b, TE::NN, 1, k, j - (ndim>1), i + 1)) / (2.*dx[1]);
	Real dHx = (v0_Jflux(b, TE::NN, 2, k + (ndim>2), j, i) -                            
		    v0_Jflux(b, TE::NN, 2, k - (ndim>2), j, i)) / (2.*dx[2]);
	Real dHx_ip1 = (v0_Jflux(b, TE::NN, 2, k + (ndim>2), j, i + 1) -                    
		        v0_Jflux(b, TE::NN, 2, k - (ndim>2), j, i + 1)) / (2.*dx[2]);
	Real divf = - (dFx + 0.5*(dGx+dGx_ip1) + 0.5*(dHx+dHx_ip1));
	/*if (abs(divf)>1.e-6) {
	  std::cout << "i=" << i << " dFx=" << dFx << " dGx=" << dGx << "/" << dGx_ip1
    		    << " dHx=" << dHx << "/" << dHx_ip1 << " divf=" << divf << std::endl;
	  std::cout << "->" << " For dFx inputs: " << v0_Jflux(b, TE::NN, 0, k, j, i)
		    << "/" << v0_Jflux(b, TE::NN, 0, k, j, i + 1) << std::endl;
	}*/
        v0_J(b, TE::E1, 0, k, j, i) = gam0 * v0_J(b, TE::E1, 0, k, j, i) + gam1 * v1_J(b, TE::E1, 0, k, j, i) 
				+ divf * beta_dt;
  });

  const auto iby = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jby = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kby = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  int ib_2s = iby.s, ib_2e = iby.e;
  int jb_2s = jby.s, jb_2e = jby.e;
  int kb_2s = kby.s, kb_2e = kby.e;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdateEdge_y", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_2s-(ndim>2), kb_2e+(ndim>2), jb_2s-(ndim>1), jb_2e+(ndim>1), ib_2s-1, ib_2e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
	const auto &dx = coords.GetCellWidths();

	// YH: (a) for Efield
	v0_E(b, TE::E2, 0, k, j, i) = gam0 * v0_E(b, TE::E2, 0, k, j, i) + gam1 * v1_E(b, TE::E2, 0, k, j, i) 
		+ v0_Eflux(b, TE::E2, 0, k, j, i) * beta_dt;
	// YH: (b) for J
	Real dFy = (v0_Jflux(b, TE::NN, 3, k, j, i + 1) - 
		    v0_Jflux(b, TE::NN, 3, k, j, i - 1)) / (2.*dx[0]);
	Real dFy_jp1 = (v0_Jflux(b, TE::NN, 3, k, j + (ndim>1), i + 1) - 
		        v0_Jflux(b, TE::NN, 3, k, j + (ndim>1), i - 1)) / (2.*dx[0]);
	Real dGy = (v0_Jflux(b, TE::NN, 4, k, j + (ndim>1), i) - 
		    v0_Jflux(b, TE::NN, 4, k, j, i)) / dx[1];
	Real dHy = (v0_Jflux(b, TE::NN, 5, k + (ndim>2), j, i) - 
		    v0_Jflux(b, TE::NN, 5, k - (ndim>2), j, i)) / (2.*dx[2]);
	Real dHy_jp1 = (v0_Jflux(b, TE::NN, 5, k + (ndim>2), j + (ndim>1), i) - 
			v0_Jflux(b, TE::NN, 5, k - (ndim>2), j + (ndim>1), i)) / (2.*dx[2]);
	Real divf = - (0.5*(dFy+dFy_jp1) + dGy + 0.5*(dHy+dHy_jp1));
        v0_J(b, TE::E2, 0, k, j, i) = gam0 * v0_J(b, TE::E2, 0, k, j, i) + gam1 * v1_J(b, TE::E2, 0, k, j, i)  
				+ divf * beta_dt;
  });

  const auto ibz = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jbz = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kbz = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  int ib_3s = ibz.s, ib_3e = ibz.e;
  int jb_3s = jbz.s, jb_3e = jbz.e;
  int kb_3s = kbz.s, kb_3e = kbz.e;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdateEdge_z", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_3s-(ndim>2), kb_3e+(ndim>2), jb_3s-(ndim>1), jb_3e+(ndim>1), ib_3s-1, ib_3e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
	const auto &dx = coords.GetCellWidths();

	// YH: (a) for Efield
	v0_E(b, TE::E3, 0, k, j, i) = gam0 * v0_E(b, TE::E3, 0, k, j, i) + gam1 * v1_E(b, TE::E3, 0, k, j, i) 
		+ v0_Eflux(b, TE::E3, 0, k, j, i) * beta_dt;
	// YH: (b) for J
	Real dFz = (v0_Jflux(b, TE::NN, 6, k, j, i + 1) - 
		    v0_Jflux(b, TE::NN, 6, k, j, i - 1)) / (2.*dx[0]);
	Real dFz_kp1 = (v0_Jflux(b, TE::NN, 6, k + (ndim>2), j, i + 1) - 
			v0_Jflux(b, TE::NN, 6, k + (ndim>2), j, i - 1)) / (2.*dx[0]);
	Real dGz = (v0_Jflux(b, TE::NN, 7, k, j + (ndim>1), i) - 
		    v0_Jflux(b, TE::NN, 7, k, j - (ndim>1), i)) / (2.*dx[1]);
	Real dGz_kp1 = (v0_Jflux(b, TE::NN, 7, k + (ndim>2), j + (ndim>1), i) - 
			v0_Jflux(b, TE::NN, 7, k + (ndim>2), j - (ndim>1), i)) / (2.*dx[1]);
	Real dHz = (v0_Jflux(b, TE::NN, 8, k + (ndim>2), j, i) - 
		    v0_Jflux(b, TE::NN, 8, k, j, i)) / (dx[2]);
	Real divf = - (0.5*(dFz+dFz_kp1) + 0.5*(dGz+dGz_kp1) +dHz);
        v0_J(b, TE::E3, 0, k, j, i) = gam0 * v0_J(b, TE::E3, 0, k, j, i) + gam1 * v1_J(b, TE::E3, 0, k, j, i) 
				+ divf * beta_dt;
      });
  return TaskStatus::complete;
}


KOKKOS_INLINE_FUNCTION std::array<Real, 6>
LinearSolve3by3(const std::array<Real, 3> n_star, const Real dt, const std::array<Real, 3> eta,
		const std::array<Real, 3> &u_B, const std::array<Real, 3> &B_star,
		const std::array<Real, 3> &E_star, const std::array<Real, 3> &J_star) {
  const Real tau_x = std::sqrt(SQR(lambda_e)/(SQR(L0)*n_star[0]));
  const Real Tau_x = SQR(tau_x) / dt;
  const Real tau_y = std::sqrt(SQR(lambda_e)/(SQR(L0)*n_star[1]));
  const Real Tau_y = SQR(tau_y) / dt;
  const Real tau_z = std::sqrt(SQR(lambda_e)/(SQR(L0)*n_star[2]));
  const Real Tau_z = SQR(tau_z) / dt;
  const Real ampere = dt *SQR(c_per_v);
  const Real alpha_x = Tau_x + ampere + eta[0];
  const Real alpha_y = Tau_y + ampere + eta[1];
  const Real alpha_z = Tau_z + ampere + eta[2];
  const Real hall_x = lambda_ion / (L0*n_star[0]);
  const Real hall_y = lambda_ion / (L0*n_star[1]);
  const Real hall_z = lambda_ion / (L0*n_star[2]);

  const Real a_A=alpha_x; 
  const Real e_A=alpha_y; 
  const Real i_A=alpha_z;
  const Real b_A=hall_x*B_star[2];
  const Real c_A=hall_x*(-B_star[1]);
  const Real d_A=hall_y*(-B_star[2]);
  const Real f_A=hall_y*B_star[0];
  const Real g_A=hall_z*B_star[1];
  const Real h_A=hall_z*(-B_star[0]);

  const Real det_A = a_A*(e_A*i_A - f_A*h_A) - b_A*(d_A*i_A - f_A*g_A) + c_A*(d_A*h_A - e_A*g_A);
  if (det_A==0.) printf("YH: det_A = 0 at implicit solved!!!");
  // Inverse matrix
  std::array<std::array<Real, 3>, 3> invA;
  const Real a_iA=e_A*i_A - f_A*h_A;
  const Real b_iA=c_A*h_A - b_A*i_A;
  const Real c_iA=b_A*f_A - c_A*e_A;
  const Real d_iA=f_A*g_A - d_A*i_A;
  const Real e_iA=a_A*i_A - c_A*g_A;
  const Real f_iA=c_A*d_A - a_A*f_A;
  const Real g_iA=d_A*h_A - e_A*g_A;
  const Real h_iA=b_A*g_A - a_A*h_A;
  const Real i_iA=a_A*e_A - b_A*d_A;
  
  invA[0][0] = a_iA / det_A;
  invA[0][1] = b_iA / det_A;
  invA[0][2] = c_iA / det_A;
  invA[1][0] = d_iA / det_A;
  invA[1][1] = e_iA / det_A;
  invA[1][2] = f_iA / det_A;
  invA[2][0] = g_iA / det_A;
  invA[2][1] = h_iA / det_A;
  invA[2][2] = i_iA / det_A;

  // Construct RHS (u_B from MHD Riemann solver of CT)
  std::array<Real, 3> rhs ={
  	Tau_x * J_star[0] + E_star[0] - u_B[0], // YH: note MHD RS gives -u x B
  	Tau_y * J_star[1] + E_star[1] - u_B[1],
  	Tau_z * J_star[2] + E_star[2] - u_B[2]
  };

  // Final result: J_analytical = invA * rhs_A
  std::array<Real, 3> J{};
  for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
          J[i] += invA[i][j] * rhs[j];

  std::array<Real, 3> E{};
  for (int i=0; i<3; ++i) E[i] = E_star[i] - dt * SQR(c_per_v) * J[i];

  return {E[0], E[1], E[2], J[0], J[1], J[2]};
}

KOKKOS_INLINE_FUNCTION std::array<Real, 6>
LinearSolve3by3_LUpp(const std::array<Real, 3> n_star, const Real dt, const std::array<Real, 3> eta,
		const std::array<Real, 3> &u_B, const std::array<Real, 3> &B_star,
		const std::array<Real, 3> &E_star, const std::array<Real, 3> &J_star) {
  const Real tau_x = std::sqrt(SQR(lambda_e)/(SQR(L0)*n_star[0]));
  const Real Tau_x = SQR(tau_x) / dt;
  const Real tau_y = std::sqrt(SQR(lambda_e)/(SQR(L0)*n_star[1]));
  const Real Tau_y = SQR(tau_y) / dt;
  const Real tau_z = std::sqrt(SQR(lambda_e)/(SQR(L0)*n_star[2]));
  const Real Tau_z = SQR(tau_z) / dt;
  const Real ampere = dt *SQR(c_per_v);
  const Real alpha_x = Tau_x + ampere + eta[0];
  const Real alpha_y = Tau_y + ampere + eta[1];
  const Real alpha_z = Tau_z + ampere + eta[2];
  const Real hall_x = lambda_ion / (L0*n_star[0]);
  const Real hall_y = lambda_ion / (L0*n_star[1]);
  const Real hall_z = lambda_ion / (L0*n_star[2]);

  const Real a_A=alpha_x; 
  const Real e_A=alpha_y; 
  const Real i_A=alpha_z;
  const Real b_A=hall_x*B_star[2];
  const Real c_A=hall_x*(-B_star[1]);
  const Real d_A=hall_y*(-B_star[2]);
  const Real f_A=hall_y*B_star[0];
  const Real g_A=hall_z*B_star[1];
  const Real h_A=hall_z*(-B_star[0]);

  const Real det_A = a_A*(e_A*i_A - f_A*h_A) - b_A*(d_A*i_A - f_A*g_A) + c_A*(d_A*h_A - e_A*g_A);
  if (det_A==0.) printf("YH: det_A = 0 at implicit solved!!!");

  // Construct RHS (u_B from MHD Riemann solver of CT)
  std::array<Real, 3> rhs ={
  	Tau_x * J_star[0] + E_star[0] - u_B[0], // YH: note MHD RS gives -u x B
  	Tau_y * J_star[1] + E_star[1] - u_B[1],
  	Tau_z * J_star[2] + E_star[2] - u_B[2]
  };

  // Gaussian elimination with partial pivoting from Algo. 21.1 of Lloyd N. Trefethen
  // Original matrix to be solved
  std::array<std::array<Real,3>,3> M = {{
    {{ a_A, b_A, c_A }},
    {{ d_A, e_A, f_A }},
    {{ g_A, h_A, i_A }}
  }};

  // Need L & U matrix for LU factorization
  // (i) U initialized by copying from M
  // (ii) L initialized with identity matrix
  std::array<std::array<Real,3>,3> U;
  std::array<std::array<Real,3>,3> L;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      U[r][c] = M[r][c];             
      L[r][c] = (r == c) ? 1. : 0.;
    }
  }

  const Real tol = 1.e-16;
  for (int k = 0; k < 2; ++k) {
    // 1) Select i>=k to maximize |u_ik|
    int piv = k;
    Real maxabs = std::fabs(U[k][k]);
    for (int i = k + 1; i < 3; ++i) {
      Real a = std::fabs(U[i][k]);
      if (a > maxabs) { maxabs = a; piv = i; }
    }
    // 2) interchange entire rows of U (std::array supports this)
    if (piv != k) {
      std::swap(U[k], U[piv]);                 // swap row arrays
      // 3) swap multipliers (first k columns) in L
      for (int c = 0; c < k; ++c) std::swap(L[k][c], L[piv][c]);
      // 4) directly swap rhs entries instead of computing Permutation matrix
      std::swap(rhs[k], rhs[piv]);
    }
    if (std::fabs(U[k][k]) <= tol) {
      throw std::runtime_error("Near-zero pivot encountered (matrix may be singular).");
    }
    // 5) elimination
    for (int j = k + 1; j < 3; ++j) {
      Real f = U[j][k] / U[k][k];
      L[j][k] = f;
      for (int c = k; c < 3; ++c) {
        U[j][c] -= f * U[k][c];
      }
    }
  }
  // forward substitution L * y = rhs
  // https://mathweb.ucsd.edu/~dumitriu/chapter1p3.pdf = simple way to code
  std::array<Real,3> y = {0.,0.,0.};
  for (int i = 0; i < 3; ++i) {
    Real s = rhs[i];
    for (int j = 0; j < i; ++j) s -= L[i][j] * y[j];
    y[i] = s / L[i][i]; 
  }

  // back substitution U * J = y (similar to above but opposite)
  std::array<Real,3> J = {0.,0.,0.};
  for (int i = 2; i >= 0; --i) {
    Real s = y[i];
    for (int j = i + 1; j < 3; ++j) s -= U[i][j] * J[j];
    if (std::fabs(U[i][i]) <= tol) {
      throw std::runtime_error("Near-zero diagonal in U during back substitution.");
    }
    J[i] = s / U[i][i];
  }

  std::array<Real, 3> E{}; 
  for (int i=0; i<3; ++i) E[i] = E_star[i] - dt * SQR(c_per_v) * J[i];

  return {E[0], E[1], E[2], J[0], J[1], J[2]};
}


//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::ApplyEJSource
//! \brief (Implicit solver for E-field & J-current)
template <Coordinates GEOM>
TaskStatus ApplyEJSource_direcSplit(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;
  // Extract integrator weights
  const Real gam0 = integrator->gam0[stage - 1];
  const Real gam1 = integrator->gam1[stage - 1];
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;
  if (beta_dt==0.) return TaskStatus::complete;

  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  const auto eta_type = gas_pkg->template Param<EtaType>("eta_type");

  // YH: extract cc rho & vel
  std::vector<MetadataFlag> flags_rho({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Density")});
  static auto desc_rho = MakePackDescriptor<any>(u0, flags_rho);
  const auto v0_rho = desc_rho.GetPack(u0);
  std::vector<MetadataFlag> flags_P({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Pressure")});
  static auto desc_P = MakePackDescriptor<any>(u0, flags_P);
  const auto v0_P = desc_P.GetPack(u0);
  std::vector<MetadataFlag> flags_Ener({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Energy")});
  static auto desc_Ener = MakePackDescriptor<any>(u0, flags_Ener);
  const auto v0_Ener = desc_Ener.GetPack(u0);
  std::vector<MetadataFlag> flags_Se({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                  Metadata::GetUserFlag("electron")});
  static auto desc_Se = MakePackDescriptor<any>(u0, flags_Se);
  const auto v0_Se = desc_Se.GetPack(u0);
  std::vector<MetadataFlag> flags1({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("electron")});
  static auto desc1 = MakePackDescriptor<any>(u0, flags1);
  const auto v0_Pe = desc1.GetPack(u0);
  // YH: extract face Bfield      
  std::vector<MetadataFlag> flags_B({Metadata::Face, Metadata::Conserved, Metadata::WithFluxes,
		  Metadata::GetUserFlag("Bfield")});
  static auto desc_B = MakePackDescriptor<any>(u0, flags_B, {parthenon::PDOpt::WithFluxes});
  const auto v0_B = desc_B.GetPack(u0);
  // YH: extract edge Efield & J  
  std::vector<MetadataFlag> flags_E({Metadata::Edge, Metadata::Conserved, 
		  Metadata::GetUserFlag("Efield")});
  static auto desc_E = MakePackDescriptor<any>(u0, flags_E);
  const auto v0_E = desc_E.GetPack(u0);
  const auto v1_E = desc_E.GetPack(u1);
  std::vector<MetadataFlag> flags_J({Metadata::Edge, Metadata::Conserved, 
		  Metadata::GetUserFlag("Jcurrent")});
  static auto desc_J = MakePackDescriptor<any>(u0, flags_J);
  const auto v0_J = desc_J.GetPack(u0);
  const auto v1_J = desc_J.GetPack(u1);
  // Apply ion-electron energy exchange source term thus need ion & electron temperature
  std::vector<MetadataFlag> flagsTi({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Temp_ion")});
  static auto descTi = MakePackDescriptor<any>(u0, flagsTi);
  const auto vTi = descTi.GetPack(u0);
  std::vector<MetadataFlag> flagsTe({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Temp_elec")});
  static auto descTe = MakePackDescriptor<any>(u0, flagsTe);
  const auto vTe = descTe.GetPack(u0);

  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+(ndim>2), jb.s-2*(ndim>1), jb.e+(ndim>1), ib.s-2, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

	// YH: Interpolate cc vel & rho to edges
	const auto v0_vel = Null<Real>();
	const Real rho_e1 = INTERP::CC_2_E1_max(v0_rho, v0_vel, b, k, j, i, ndim);
	const Real rho_e2 = INTERP::CC_2_E2_max(v0_rho, v0_vel, b, k, j, i, ndim);
	const Real rho_e3 = INTERP::CC_2_E3_max(v0_rho, v0_vel, b, k, j, i, ndim);
	const Real n_star_e1 = Z_ion * rho_e1;
	const Real n_star_e2 = Z_ion * rho_e2;
	const Real n_star_e3 = Z_ion * rho_e3;
	const Real Pe_e1 = INTERP::CC_2_E1_maxrho(v0_Pe, v0_rho, b, k, j, i, ndim);
	const Real Pe_e2 = INTERP::CC_2_E2_maxrho(v0_Pe, v0_rho, b, k, j, i, ndim);
	const Real Pe_e3 = INTERP::CC_2_E3_maxrho(v0_Pe, v0_rho, b, k, j, i, ndim);
	const Real eta_e1 = ComputeEta(eta_type, n_star_e1, Pe_e1);
        const Real eta_e2 = ComputeEta(eta_type, n_star_e2, Pe_e2);
        const Real eta_e3 = ComputeEta(eta_type, n_star_e3, Pe_e3);
	// Get E & J
        Real Ex = v0_E(b,TE::E1,0,k,j,i);
        Real Ey = v0_E(b,TE::E2,0,k,j,i);
        Real Ez = v0_E(b,TE::E3,0,k,j,i);
        Real Jx = v0_J(b,TE::E1,0,k,j,i);
        Real Jy = v0_J(b,TE::E2,0,k,j,i);
        Real Jz = v0_J(b,TE::E3,0,k,j,i);

	// For x
	const Real Bx_e1 = INTERP::F1_2_E1(v0_B, v0_vel, b, k, j, i, ndim);
        const Real By_e1 = INTERP::F2_2_E1(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bz_e1 = INTERP::F3_2_E1(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Ey_e1 = INTERP::E2_2_E1(v0_E, v0_vel, b, k, j, i, ndim);
        const Real Ez_e1 = INTERP::E3_2_E1(v0_E, v0_vel, b, k, j, i, ndim);
        const Real Jy_e1 = INTERP::E2_2_E1(v0_J, v0_vel, b, k, j, i, ndim);
        const Real Jz_e1 = INTERP::E3_2_E1(v0_J, v0_vel, b, k, j, i, ndim);
	const Real u_By_e1 = 0.25 * (
             v0_B.flux(b, TE::E2, 0, k, j, i) + v0_B.flux(b, TE::E2, 0, k, j, i + 1)
           + v0_B.flux(b, TE::E2, 0, k, j - (ndim>1), i) + v0_B.flux(b, TE::E2, 0, k, j - (ndim>1), i + 1)
              );
	const Real u_Bz_e1 = 0.25 * (
             v0_B.flux(b, TE::E3, 0, k, j, i) + v0_B.flux(b, TE::E3, 0, k, j, i + 1)
           + v0_B.flux(b, TE::E3, 0, k - (ndim>2), j, i) + v0_B.flux(b, TE::E3, 0, k - (ndim>2), j, i + 1)
              );
	const std::array<Real, 3> B_star_e1{Bx_e1, By_e1, Bz_e1};
	const std::array<Real, 3> u_B_e1{v0_B.flux(b,TE::E1,0,k,j,i),u_By_e1,u_Bz_e1};
	const std::array<Real, 3> E_star_e1{Ex, Ey_e1, Ez_e1};
	const std::array<Real, 3> J_star_e1{Jx, Jy_e1, Jz_e1};
	const std::array<Real, 3> n_star_E1{n_star_e1, n_star_e1, n_star_e1};
	const std::array<Real, 3> eta_E1{eta_e1, eta_e1, eta_e1};
	const std::array<Real, 6> newEJ_x = 
		LinearSolve3by3_LUpp(n_star_E1,beta_dt,eta_E1,u_B_e1,B_star_e1,E_star_e1,J_star_e1);
	// For y
	const Real Bx_e2 = INTERP::F1_2_E2(v0_B, v0_vel, b, k, j, i, ndim);
        const Real By_e2 = INTERP::F2_2_E2(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bz_e2 = INTERP::F3_2_E2(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Ex_e2 = INTERP::E1_2_E2(v0_E, v0_vel, b, k, j, i, ndim);
        const Real Ez_e2 = INTERP::E3_2_E2(v0_E, v0_vel, b, k, j, i, ndim);
        const Real Jx_e2 = INTERP::E1_2_E2(v0_J, v0_vel, b, k, j, i, ndim);
        const Real Jz_e2 = INTERP::E3_2_E2(v0_J, v0_vel, b, k, j, i, ndim);
	const Real u_Bx_e2 = 0.25 * (
             v0_B.flux(b, TE::E1, 0, k, j, i) + v0_B.flux(b, TE::E1, 0, k, j + (ndim>1), i)
           + v0_B.flux(b, TE::E1, 0, k, j, i - 1) + v0_B.flux(b, TE::E1, 0, k, j + (ndim>1), i - 1)
              );
	const Real u_Bz_e2 = 0.25 * (
             v0_B.flux(b, TE::E3, 0, k, j, i) + v0_B.flux(b, TE::E3, 0, k, j + (ndim>1), i)
           + v0_B.flux(b, TE::E3, 0, k - (ndim>2), j, i) + v0_B.flux(b, TE::E3, 0, k - (ndim>2), j + (ndim>1), i - 1)
              );
        const std::array<Real, 3> B_star_e2{Bx_e2, By_e2, Bz_e2};
        const std::array<Real, 3> u_B_e2{u_Bx_e2,v0_B.flux(b,TE::E2,0,k,j,i),u_Bz_e2};
        const std::array<Real, 3> E_star_e2{Ex_e2, Ey, Ez_e2};
        const std::array<Real, 3> J_star_e2{Jx_e2, Jy, Jz_e2};
        const std::array<Real, 3> n_star_E2{n_star_e2, n_star_e2, n_star_e2};
        const std::array<Real, 3> eta_E2{eta_e2, eta_e2, eta_e2};
        const std::array<Real, 6> newEJ_y =
                LinearSolve3by3_LUpp(n_star_E2,beta_dt,eta_E2,u_B_e2,B_star_e2,E_star_e2,J_star_e2);
	// For z
        const Real Bx_e3 = INTERP::F1_2_E3(v0_B, v0_vel, b, k, j, i, ndim);
        const Real By_e3 = INTERP::F2_2_E3(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bz_e3 = INTERP::F3_2_E3(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Ex_e3 = INTERP::E1_2_E3(v0_E, v0_vel, b, k, j, i, ndim);
        const Real Ey_e3 = INTERP::E2_2_E3(v0_E, v0_vel, b, k, j, i, ndim);
        const Real Jx_e3 = INTERP::E1_2_E3(v0_J, v0_vel, b, k, j, i, ndim);
        const Real Jy_e3 = INTERP::E2_2_E3(v0_J, v0_vel, b, k, j, i, ndim);
	const Real u_Bx_e3 = 0.25 * (
             v0_B.flux(b, TE::E1, 0, k, j, i) + v0_B.flux(b, TE::E1, 0, k + (ndim>2), j, i)
           + v0_B.flux(b, TE::E1, 0, k, j, i - 1) + v0_B.flux(b, TE::E1, 0, k + (ndim>2), j, i - 1)
              );
	const Real u_By_e3 = 0.25 * (
             v0_B.flux(b, TE::E2, 0, k, j, i) + v0_B.flux(b, TE::E2, 0, k + (ndim>2), j, i)
           + v0_B.flux(b, TE::E2, 0, k, j - (ndim>1), i) + v0_B.flux(b, TE::E2, 0, k + (ndim>2), j - (ndim>1), i)
              );
        const std::array<Real, 3> B_star_e3{Bx_e3, By_e3, Bz_e3};
        const std::array<Real, 3> u_B_e3{u_Bx_e3,u_By_e3,v0_B.flux(b,TE::E3,0,k,j,i)};
        const std::array<Real, 3> E_star_e3{Ex_e3, Ey_e3, Ez};
        const std::array<Real, 3> J_star_e3{Jx_e3, Jy_e3, Jz};
        const std::array<Real, 3> n_star_E3{n_star_e3, n_star_e3, n_star_e3};
        const std::array<Real, 3> eta_E3{eta_e3, eta_e3, eta_e3};
        const std::array<Real, 6> newEJ_z =
                LinearSolve3by3_LUpp(n_star_E3,beta_dt,eta_E3,u_B_e3,B_star_e3,E_star_e3,J_star_e3);

	v0_E(b,TE::E1,0,k,j,i) = newEJ_x[0];
	v0_J(b,TE::E1,0,k,j,i) = newEJ_x[3];
	v0_E(b,TE::E2,0,k,j,i) = newEJ_y[1];
        v0_J(b,TE::E2,0,k,j,i) = newEJ_y[4];
	v0_E(b,TE::E3,0,k,j,i) = newEJ_z[2];
        v0_J(b,TE::E3,0,k,j,i) = newEJ_z[5];

      });

  // Apply implicit update to electron temperature through e-i energy exchange
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+2*(ndim>2), jb.s-2*(ndim>1), jb.e+2*(ndim>1), ib.s-2, ib.e+2,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

	Real ne = Z_ion * v0_rho(b, TE::CC, 0, k, j, i);
	Real Ti = vTi(b, TE::CC, 0, k, j, i);
        Real Te = vTe(b, TE::CC, 0, k, j, i);
        Real fac = (gamma-1.)*((kB*T0*eV_to_K)/(m_ion*SQR(char_speed))) *
                (me/(m_ion+me)) ;
	Real tau_ei = collision::tau_ei_FLASH(v0_rho(b, TE::CC, 0, k, j, i), Ti, Te) / t0;
	Real tmp = beta_dt*fac/tau_ei;
	Real cv_ratio = Z_ion;  // cv,e/cv,ion
	Real RHS = Te + tmp/(1+tmp*cv_ratio)*Ti;
	Real LHS = 1. + tmp - (tmp*tmp*cv_ratio)/(1+tmp*cv_ratio);
	Real Te_new = RHS / LHS;
	/*if (v0_Pe(b, TE::CC, 0, k, j, i) > v0_P(b, TE::CC, 0, k, j, i)) {
	  Real Pi_floor = 1.e-10;
	  Te_new = (v0_P(b, TE::CC, 0, k, j, i) - Pi_floor)/ne;
	}*/
	vTe(b, TE::CC, 0, k, j, i) = Te_new;
	v0_Pe(b, TE::CC, 0, k, j, i) = ne*Te_new;
	v0_Se(b, TE::CC, 0, k, j, i) = v0_Pe(b, TE::CC, 0, k, j, i)*pow(ne,1.-gamma);

	});

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::ApplyEJSource_NN
//! \brief (Implicit solver for E-field & J-current but done at nodes first)
template <Coordinates GEOM>
TaskStatus ApplyEJSource_NN(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;
  // Extract integrator weights
  const Real gam0 = integrator->gam0[stage - 1];
  const Real gam1 = integrator->gam1[stage - 1];
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;
  if (beta_dt==0.) return TaskStatus::complete;
  
  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  const auto eta_type = gas_pkg->template Param<EtaType>("eta_type");

  // YH: extract cc rho & vel
  std::vector<MetadataFlag> flags_rho({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Density")});
  static auto desc_rho = MakePackDescriptor<any>(u0, flags_rho);
  const auto v0_rho = desc_rho.GetPack(u0);
  std::vector<MetadataFlag> flags_P({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Pressure")});
  static auto desc_P = MakePackDescriptor<any>(u0, flags_P);
  const auto v0_P = desc_P.GetPack(u0);
  std::vector<MetadataFlag> flags_Se({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                  Metadata::GetUserFlag("electron")});
  static auto desc_Se = MakePackDescriptor<any>(u0, flags_Se);
  const auto v0_Se = desc_Se.GetPack(u0);
  std::vector<MetadataFlag> flags1({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("electron")});
  static auto desc1 = MakePackDescriptor<any>(u0, flags1);
  const auto v0_Pe = desc1.GetPack(u0);
  // YH: extract face Bfield      
  std::vector<MetadataFlag> flags_B({Metadata::Face, Metadata::Conserved, Metadata::WithFluxes,
		  Metadata::GetUserFlag("Bfield")});
  static auto desc_B = MakePackDescriptor<any>(u0, flags_B, {parthenon::PDOpt::WithFluxes});
  const auto v0_B = desc_B.GetPack(u0);
  // YH: extract edge Efield & J  
  std::vector<MetadataFlag> flags_E({Metadata::Edge, Metadata::Conserved, 
		  Metadata::GetUserFlag("Efield")});
  static auto desc_E = MakePackDescriptor<any>(u0, flags_E);
  const auto v0_E = desc_E.GetPack(u0);
  const auto v1_E = desc_E.GetPack(u1);
  std::vector<MetadataFlag> flags_J({Metadata::Edge, Metadata::Conserved, 
		  Metadata::GetUserFlag("Jcurrent")});
  static auto desc_J = MakePackDescriptor<any>(u0, flags_J);
  const auto v0_J = desc_J.GetPack(u0);
  const auto v1_J = desc_J.GetPack(u1);
  // YH: extract node for storing implicit source term initially
  std::vector<MetadataFlag> flags_EJ({Metadata::Node, Metadata::GetUserFlag("EJ_src")});
  static auto desc_EJ = MakePackDescriptor<any>(u0, flags_EJ);
  const auto v0_EJ = desc_EJ.GetPack(u0);
  const auto v1_EJ = desc_EJ.GetPack(u1);

  auto ib = u0->GetBoundsI(IndexDomain::interior, TE::NN);
  auto jb = u0->GetBoundsJ(IndexDomain::interior, TE::NN);
  auto kb = u0->GetBoundsK(IndexDomain::interior, TE::NN);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Compute_EJSource", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+(ndim>2), jb.s-2*(ndim>1), jb.e+(ndim>1), ib.s-2, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

	// YH: Interpolate cc vel & rho to edges
	const auto v0_vel = Null<Real>();
	// YH: using max helps!!!
	const Real rho_nn = INTERP::CC_2_NN_max(v0_rho, b, k, j, i, ndim);
	const Real n_star_nn = Z_ion * rho_nn;
	const Real Pe_nn = INTERP::CC_2_NN(v0_Pe, b, k, j, i, ndim);
        const Real eta_nn = ComputeEta(eta_type, n_star_nn, Pe_nn);
	// YH: Interpolate fcc Bfield to edge
	const Real Bx = INTERP::F1_2_NN(v0_B, b, k, j, i, ndim);
        const Real By = INTERP::F2_2_NN(v0_B, b, k, j, i, ndim);
        const Real Bz = INTERP::F3_2_NN(v0_B, b, k, j, i, ndim);
	// Get E & J (Must interpolate edge to node too)
	Real Ex = INTERP::E1_2_NN(v0_E, b, k, j, i, ndim);
	Real Ey = INTERP::E2_2_NN(v0_E, b, k, j, i, ndim);
	Real Ez = INTERP::E3_2_NN(v0_E, b, k, j, i, ndim);
	Real Jx = INTERP::E1_2_NN(v0_J, b, k, j, i, ndim);
        Real Jy = INTERP::E2_2_NN(v0_J, b, k, j, i, ndim);
        Real Jz = INTERP::E3_2_NN(v0_J, b, k, j, i, ndim);
	// Interpolate uxB to node too!
	const Real u_B_x = 0.5 * (v0_B.flux(b, TE::E1, 0, k, j, i) 
			 + v0_B.flux(b, TE::E1, 0, k, j, i - 1));
	const Real u_B_y = 0.5 * (v0_B.flux(b, TE::E2, 0, k, j, i)
 		 	 + v0_B.flux(b, TE::E2, 0, k, j - (ndim>1), i));
	const Real u_B_z = 0.5 * (v0_B.flux(b, TE::E3, 0, k, j, i)     
 			 + v0_B.flux(b, TE::E3, 0, k - (ndim>2), j, i));
	/*const Real u_B_x = (std::abs(v0_B.flux(b, TE::E1, 0, k, j, i)) >
                    std::abs(v0_B.flux(b, TE::E1, 0, k, j, i - 1)))
                   ? v0_B.flux(b, TE::E1, 0, k, j, i)
                   : v0_B.flux(b, TE::E1, 0, k, j, i - 1);
	const Real u_B_y = (std::abs(v0_B.flux(b, TE::E2, 0, k, j, i)) >
                    std::abs(v0_B.flux(b, TE::E2, 0, k, j - (ndim>1), i)))
                   ? v0_B.flux(b, TE::E2, 0, k, j, i)
                   : v0_B.flux(b, TE::E2, 0, k, j - (ndim>1), i);
	const Real u_B_z = (std::abs(v0_B.flux(b, TE::E3, 0, k, j, i)) >
                    std::abs(v0_B.flux(b, TE::E3, 0, k - (ndim>2), j, i)))
                   ? v0_B.flux(b, TE::E3, 0, k, j, i)
                   : v0_B.flux(b, TE::E3, 0, k - (ndim>2), j, i);*/

	const std::array<Real, 3> B_star{Bx, By, Bz};
	const std::array<Real, 3> u_B{u_B_x, u_B_y, u_B_z};
	const std::array<Real, 3> E_star{Ex, Ey, Ez};
	const std::array<Real, 3> J_star{Jx, Jy, Jz};
	const std::array<Real, 3> n_star{n_star_nn, n_star_nn, n_star_nn};
	const std::array<Real, 3> eta{eta_nn, eta_nn, eta_nn};
	const std::array<Real, 6> newEJ = 
		LinearSolve3by3_LUpp(n_star,beta_dt,eta,u_B,B_star,E_star,J_star);

	v0_EJ(b,TE::NN,0,k,j,i) = newEJ[0];
	v0_EJ(b,TE::NN,1,k,j,i) = newEJ[1];
	v0_EJ(b,TE::NN,2,k,j,i) = newEJ[2];
        v0_EJ(b,TE::NN,3,k,j,i) = newEJ[3];
	v0_EJ(b,TE::NN,4,k,j,i) = newEJ[4];
        v0_EJ(b,TE::NN,5,k,j,i) = newEJ[5];

      });

  ib = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  jb = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  kb = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+(ndim>2), jb.s-2*(ndim>1), jb.e+(ndim>1), ib.s-2, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
	const Real Jx_i = v0_EJ(b,TE::NN,3,k,j,i);
	const Real ne_i = Z_ion*INTERP::CC_2_NN(v0_rho, b, k, j, i, ndim);
	const Real vH_i = - (J0/(e_charge*n0*char_speed))*(Jx_i/ne_i);
	const Real Jx_ip1 = v0_EJ(b,TE::NN,3,k,j,i + 1);
        const Real ne_ip1 = Z_ion*INTERP::CC_2_NN(v0_rho, b, k, j, i+1, ndim);
        const Real vH_ip1 = - (J0/(e_charge*n0*char_speed))*(Jx_ip1/ne_ip1);
	const Real vH = 0.5*(vH_i + vH_ip1);
	v0_E(b,TE::E1,0,k,j,i) = INTERP::NN_2_E1_vec_upw(v0_EJ, b, k, j, i, ndim, 0, vH);
	v0_J(b,TE::E1,0,k,j,i) = INTERP::NN_2_E1_vec_upw(v0_EJ, b, k, j, i, ndim, 3, vH);
	//v0_E(b,TE::E1,0,k,j,i) = INTERP::NN_2_E1_vec(v0_EJ, b, k, j, i, ndim, 0);
        //v0_J(b,TE::E1,0,k,j,i) = INTERP::NN_2_E1_vec(v0_EJ, b, k, j, i, ndim, 3);
	/*const Real Jx = abs(Jx_i)<=abs(Jx_ip1) ? Jx_i : Jx_ip1;
	v0_J(b,TE::E1,0,k,j,i) = Jx * (std::min(ne_i,ne_ip1)/std::max(ne_i,ne_ip1));*/
  });
  ib = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  jb = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  kb = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+(ndim>2), jb.s-2*(ndim>1), jb.e+(ndim>1), ib.s-2, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
	const Real Jy_j = v0_EJ(b,TE::NN,4,k,j,i);
        const Real ne_j = Z_ion*INTERP::CC_2_NN(v0_rho, b, k, j, i, ndim);
        const Real vH_j = - (J0/(e_charge*n0*char_speed))*(Jy_j/ne_j);
        const Real Jy_jp1 = v0_EJ(b,TE::NN,4,k,j+(ndim>1),i);
        const Real ne_jp1 = Z_ion*INTERP::CC_2_NN(v0_rho, b, k, j+(ndim>1), i, ndim);
        const Real vH_jp1 = - (J0/(e_charge*n0*char_speed))*(Jy_jp1/ne_jp1);
        const Real vH = 0.5*(vH_j + vH_jp1);
        v0_E(b,TE::E2,0,k,j,i) = INTERP::NN_2_E2_vec_upw(v0_EJ, b, k, j, i, ndim, 1, vH);
        v0_J(b,TE::E2,0,k,j,i) = INTERP::NN_2_E2_vec_upw(v0_EJ, b, k, j, i, ndim, 4, vH);
	//v0_E(b,TE::E2,0,k,j,i) = INTERP::NN_2_E2_vec(v0_EJ, b, k, j, i, ndim, 1);
        //v0_J(b,TE::E2,0,k,j,i) = INTERP::NN_2_E2_vec(v0_EJ, b, k, j, i, ndim, 4);
	/*const Real Jy = abs(Jy_j)<=abs(Jy_jp1) ? Jy_j : Jy_jp1;
	v0_J(b,TE::E2,0,k,j,i) = Jy * (std::min(ne_j,ne_jp1)/std::max(ne_j,ne_jp1));*/
  });
  ib = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  jb = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  kb = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+(ndim>2), jb.s-2*(ndim>1), jb.e+(ndim>1), ib.s-2, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
	const Real Jz_k = v0_EJ(b,TE::NN,5,k,j,i);
        const Real ne_k = Z_ion*INTERP::CC_2_NN(v0_rho, b, k, j, i, ndim);
        const Real vH_k = - (J0/(e_charge*n0*char_speed))*(Jz_k/ne_k);
        const Real Jz_kp1 = v0_EJ(b,TE::NN,5,k+(ndim>2),j,i);
        const Real ne_kp1 = Z_ion*INTERP::CC_2_NN(v0_rho, b, k+(ndim>2), j, i, ndim);
        const Real vH_kp1 = - (J0/(e_charge*n0*char_speed))*(Jz_kp1/ne_kp1);
        const Real vH = 0.5*(vH_k + vH_kp1);
        v0_E(b,TE::E3,0,k,j,i) = INTERP::NN_2_E3_vec_upw(v0_EJ, b, k, j, i, ndim, 2, vH);
        v0_J(b,TE::E3,0,k,j,i) = INTERP::NN_2_E3_vec_upw(v0_EJ, b, k, j, i, ndim, 5, vH);
	//v0_E(b,TE::E3,0,k,j,i) = INTERP::NN_2_E3_vec(v0_EJ, b, k, j, i, ndim, 2);
        //v0_J(b,TE::E3,0,k,j,i) = INTERP::NN_2_E3_vec(v0_EJ, b, k, j, i, ndim, 5);
	/*const Real Jz = abs(Jz_k)<=abs(Jz_kp1) ? Jz_k : Jz_kp1;
	v0_J(b,TE::E3,0,k,j,i) = Jz * (std::min(ne_k,ne_kp1)/std::max(ne_k,ne_kp1));*/
  });

  return TaskStatus::complete;
}




//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::ApplyEJSource
//! \brief (Implicit solver for E-field & J-current)
template <Coordinates GEOM>
TaskStatus ApplyEJSource(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;
  // Extract integrator weights
  const Real gam0 = integrator->gam0[stage - 1];
  const Real gam1 = integrator->gam1[stage - 1];
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;
  if (beta_dt==0.) return TaskStatus::complete;
  
  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  const auto eta_type = gas_pkg->template Param<EtaType>("eta_type");

  // YH: extract cc rho & vel
  std::vector<MetadataFlag> flags_rho({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Density")});
  static auto desc_rho = MakePackDescriptor<any>(u0, flags_rho);
  const auto v0_rho = desc_rho.GetPack(u0);
  std::vector<MetadataFlag> flags_P({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Pressure")});
  static auto desc_P = MakePackDescriptor<any>(u0, flags_P);
  const auto v0_P = desc_P.GetPack(u0);
  std::vector<MetadataFlag> flags_Se({Metadata::Cell, Metadata::Conserved, Metadata::Independent,
                  Metadata::GetUserFlag("electron")});
  static auto desc_Se = MakePackDescriptor<any>(u0, flags_Se);
  const auto v0_Se = desc_Se.GetPack(u0);
  std::vector<MetadataFlag> flags1({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("electron")});
  static auto desc1 = MakePackDescriptor<any>(u0, flags1);
  const auto v0_Pe = desc1.GetPack(u0);
  // YH: extract face Bfield      
  std::vector<MetadataFlag> flags_B({Metadata::Face, Metadata::Conserved, Metadata::WithFluxes,
		  Metadata::GetUserFlag("Bfield")});
  static auto desc_B = MakePackDescriptor<any>(u0, flags_B, {parthenon::PDOpt::WithFluxes});
  const auto v0_B = desc_B.GetPack(u0);
  // YH: extract edge Efield & J  
  std::vector<MetadataFlag> flags_E({Metadata::Edge, Metadata::Conserved, 
		  Metadata::GetUserFlag("Efield")});
  static auto desc_E = MakePackDescriptor<any>(u0, flags_E);
  const auto v0_E = desc_E.GetPack(u0);
  const auto v1_E = desc_E.GetPack(u1);
  std::vector<MetadataFlag> flags_J({Metadata::Edge, Metadata::Conserved, 
		  Metadata::GetUserFlag("Jcurrent")});
  static auto desc_J = MakePackDescriptor<any>(u0, flags_J);
  const auto v0_J = desc_J.GetPack(u0);
  const auto v1_J = desc_J.GetPack(u1);

  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+(ndim>2), jb.s-2*(ndim>1), jb.e+(ndim>1), ib.s-2, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

	// YH: Interpolate cc vel & rho to edges
	const auto v0_vel = Null<Real>();
	const Real rho_e1 = INTERP::CC_2_E1_max(v0_rho, v0_vel, b, k, j, i, ndim);
	const Real rho_e2 = INTERP::CC_2_E2_max(v0_rho, v0_vel, b, k, j, i, ndim);
	const Real rho_e3 = INTERP::CC_2_E3_max(v0_rho, v0_vel, b, k, j, i, ndim);
	const Real n_star_e1 = Z_ion * rho_e1;
	const Real n_star_e2 = Z_ion * rho_e2;
	const Real n_star_e3 = Z_ion * rho_e3;
	const Real Pe_e1 = INTERP::CC_2_E1_maxrho(v0_Pe, v0_rho, b, k, j, i, ndim);
	const Real Pe_e2 = INTERP::CC_2_E2_maxrho(v0_Pe, v0_rho, b, k, j, i, ndim);
	const Real Pe_e3 = INTERP::CC_2_E3_maxrho(v0_Pe, v0_rho, b, k, j, i, ndim);
        const Real eta_e1 = ComputeEta(eta_type, n_star_e1, Pe_e1);
        const Real eta_e2 = ComputeEta(eta_type, n_star_e2, Pe_e2);
        const Real eta_e3 = ComputeEta(eta_type, n_star_e3, Pe_e3);
	// YH: Interpolate fcc Bfield to edge
	const Real Bx = INTERP::F1_2_E1(v0_B, v0_vel, b, k, j, i, ndim);
        const Real By = INTERP::F2_2_E2(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bz = INTERP::F3_2_E3(v0_B, v0_vel, b, k, j, i, ndim);
	// Get E & J
	Real Ex = v0_E(b,TE::E1,0,k,j,i);
	Real Ey = v0_E(b,TE::E2,0,k,j,i);
	Real Ez = v0_E(b,TE::E3,0,k,j,i);
	Real Jx = v0_J(b,TE::E1,0,k,j,i);
        Real Jy = v0_J(b,TE::E2,0,k,j,i);
        Real Jz = v0_J(b,TE::E3,0,k,j,i);

	const std::array<Real, 3> B_star{Bx, By, Bz};
	const std::array<Real, 3> u_B{v0_B.flux(b,TE::E1,0,k,j,i), 
		v0_B.flux(b,TE::E2,0,k,j,i), v0_B.flux(b,TE::E3,0,k,j,i)};
	const std::array<Real, 3> E_star{Ex, Ey, Ez};
	const std::array<Real, 3> J_star{Jx, Jy, Jz};
	const std::array<Real, 3> n_star{n_star_e1, n_star_e2, n_star_e3};
	const std::array<Real, 3> eta{eta_e1, eta_e2, eta_e3};
	const std::array<Real, 6> newEJ = 
		LinearSolve3by3_LUpp(n_star,beta_dt,eta,u_B,B_star,E_star,J_star);

	v0_E(b,TE::E1,0,k,j,i) = newEJ[0];
	v0_J(b,TE::E1,0,k,j,i) = newEJ[3];
	v0_E(b,TE::E2,0,k,j,i) = newEJ[1];
        v0_J(b,TE::E2,0,k,j,i) = newEJ[4];
	v0_E(b,TE::E3,0,k,j,i) = newEJ[2];
        v0_J(b,TE::E3,0,k,j,i) = newEJ[5];

      });

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::ApplyExplicitSource
//! \brief (Explicit source for other variables.)
template <Coordinates GEOM>
TaskStatus ComputeExplicitSource(MeshData<Real> *u0, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;
  const int nghosts = parthenon::Globals::nghost;

  // Extract integrator weights
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;
  if (beta_dt==0.) return TaskStatus::complete;

  // Retrieve required parameters
  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  const auto eta_type = gas_pkg->template Param<EtaType>("eta_type");

  // Collect CC package
  std::vector<MetadataFlag> flagsm1({Metadata::Cell, Metadata::Conserved,            
		  Metadata::GetUserFlag("Density"), Metadata::WithFluxes});       
  static auto descm1 = MakePackDescriptor<any>(u0, flagsm1, {parthenon::PDOpt::WithFluxes});    
  const auto v0_rho = descm1.GetPack(u0);
  std::vector<MetadataFlag> flags0({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Velfield")});
  static auto desc0 = MakePackDescriptor<any>(u0, flags0);
  const auto v0_mom = desc0.GetPack(u0);
  std::vector<MetadataFlag> flags0p({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Velfield")});
  static auto desc0p = MakePackDescriptor<any>(u0, flags0p);
  const auto v0_vel = desc0p.GetPack(u0);

  std::vector<MetadataFlag> flags1({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Energy")});
  static auto desc1 = MakePackDescriptor<any>(u0, flags1);
  const auto v0_ener = desc1.GetPack(u0);
  std::vector<MetadataFlag> flags2({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("electron")});
  static auto desc2 = MakePackDescriptor<any>(u0, flags2);
  const auto v0_Se = desc2.GetPack(u0);
  // Collect FCC package
  std::vector<MetadataFlag> flags_B({Metadata::Face, Metadata::Conserved, 
		  Metadata::WithFluxes, Metadata::GetUserFlag("Bfield")});
  static auto desc_B = MakePackDescriptor<any>(u0, flags_B, {parthenon::PDOpt::WithFluxes});
  const auto v0_B = desc_B.GetPack(u0);
  // Collect Edge package
  std::vector<MetadataFlag> flags_E({Metadata::Edge, Metadata::Conserved,
                  Metadata::GetUserFlag("Efield")});
  static auto desc_E = MakePackDescriptor<any>(u0, flags_E);
  const auto v0_E = desc_E.GetPack(u0);
  std::vector<MetadataFlag> flags_J({Metadata::Edge, Metadata::Conserved,
                  Metadata::GetUserFlag("Jcurrent")});
  static auto desc_J = MakePackDescriptor<any>(u0, flags_J);
  const auto v0_J = desc_J.GetPack(u0);
  // Collect source package
  std::vector<MetadataFlag> flags_NI({Metadata::Face, Metadata::GetUserFlag("NonIdeal")});
  static auto desc_NI = MakePackDescriptor<any>(u0, flags_NI);
  const auto v0_NI = desc_NI.GetPack(u0);
  std::vector<MetadataFlag> flags_NI_ed({Metadata::Edge, Metadata::GetUserFlag("NonIdeal")});
  static auto desc_NI_ed = MakePackDescriptor<any>(u0, flags_NI_ed);
  const auto v0_NI_ed = desc_NI_ed.GetPack(u0);

  std::vector<MetadataFlag> flags_NI1({Metadata::Face, Metadata::GetUserFlag("NonIdeal1")});
  static auto desc_NI1 = MakePackDescriptor<any>(u0, flags_NI1);
  const auto v0_NI1 = desc_NI1.GetPack(u0);
  std::vector<MetadataFlag> flags_NI1_ed({Metadata::Edge, Metadata::GetUserFlag("NonIdeal1")});
  static auto desc_NI1_ed = MakePackDescriptor<any>(u0, flags_NI1_ed);
  const auto v0_NI1_ed = desc_NI1_ed.GetPack(u0);

  std::vector<MetadataFlag> flags_rel({Metadata::Face, Metadata::GetUserFlag("Relativistic")});
  static auto desc_rel = MakePackDescriptor<any>(u0, flags_rel);
  const auto v0_rel = desc_rel.GetPack(u0);
  std::vector<MetadataFlag> flags_rel_ed({Metadata::Edge, Metadata::GetUserFlag("Relativistic")});
  static auto desc_rel_ed = MakePackDescriptor<any>(u0, flags_rel_ed);
  const auto v0_rel_ed = desc_rel_ed.GetPack(u0);
  // Compute source package to save
  std::vector<MetadataFlag> flags_SrcB({Metadata::Face, Metadata::GetUserFlag("Source_Bfield")});
  static auto desc_SrcB = MakePackDescriptor<any>(u0, flags_SrcB);
  const auto v0_SrcB = desc_SrcB.GetPack(u0);
  std::vector<MetadataFlag> flags_Srcmom({Metadata::Cell, Metadata::GetUserFlag("Source_mom")});
  static auto desc_Srcmom = MakePackDescriptor<any>(u0, flags_Srcmom);
  const auto v0_Srcmom = desc_Srcmom.GetPack(u0);
  std::vector<MetadataFlag> flags_Srcener({Metadata::Cell, Metadata::GetUserFlag("Source_ener")});
  static auto desc_Srcener = MakePackDescriptor<any>(u0, flags_Srcener);
  const auto v0_Srcener = desc_Srcener.GetPack(u0);
  std::vector<MetadataFlag> flags_SrcSe({Metadata::Cell, Metadata::GetUserFlag("Source_Se")});
  static auto desc_SrcSe = MakePackDescriptor<any>(u0, flags_SrcSe);
  const auto v0_SrcSe = desc_SrcSe.GetPack(u0);
  // Apply Upwind-biased non-ideal source term based on the idea of GS2004--------------------------
  std::vector<MetadataFlag> flagsB_cc({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Bfield"), Metadata::WithFluxes});
  static auto descB_cc = MakePackDescriptor<any>(u0, flagsB_cc, {parthenon::PDOpt::WithFluxes});
  const auto vB_cc = descB_cc.GetPack(u0);
  std::vector<MetadataFlag> flagsE_cc({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Efield")});
  static auto descE_cc = MakePackDescriptor<any>(u0, flagsE_cc);
  const auto vE_cc = descE_cc.GetPack(u0);
  // Apply ion-electron energy exchange source term thus need ion & electron temperature
  std::vector<MetadataFlag> flagsTi({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Temp_ion")});
  static auto descTi = MakePackDescriptor<any>(u0, flagsTi);
  const auto vTi = descTi.GetPack(u0);
  std::vector<MetadataFlag> flagsTe({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Temp_elec")});
  static auto descTe = MakePackDescriptor<any>(u0, flagsTe);
  const auto vTe = descTe.GetPack(u0);

  // Obtain edge component for relativistic source term
  const auto ibE1 = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jbE1 = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kbE1 = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbE1.s-(ndim>2), kbE1.e+(ndim>2), jbE1.s-(ndim>1), jbE1.e+(ndim>1), ibE1.s-1, ibE1.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rel_ed.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBz_dy = -(v0_B(b, TE::F3, 0, k, j - (ndim > 1 and j>0), i)
                        - v0_B(b, TE::F3, 0, k, j, i)) / dx[1];
        Real dBy_dz = -(v0_B(b, TE::F2, 0, k - (ndim > 2 and k>0), j, i)
                        - v0_B(b, TE::F2, 0, k, j, i)) / dx[2];
        v0_rel_ed(b, TE::E1, 0, k, j, i) = v0_J(b, TE::E1, 0, k, j, i) - (dBz_dy - dBy_dz);
	v0_NI_ed(b, TE::E1, 0, k, j, i) = v0_E(b, TE::E1, 0, k, j, i) - v0_B.flux(b, TE::E1, 0, k, j, i);
	// YH: add hyper-resisitivity
	const auto v0_vel = Null<Real>();
	const Real ne = Z_ion * INTERP::CC_2_E1(v0_rho, v0_vel, b, k, j, i, ndim);
	const Real Bx = INTERP::F1_2_E1(v0_B, v0_vel, b, k, j, i, ndim);
	const Real By = INTERP::F2_2_E1(v0_B, v0_vel, b, k, j, i, ndim);
	const Real Bz = INTERP::F3_2_E1(v0_B, v0_vel, b, k, j, i, ndim);
	const Real Bmag = SQR(Bx) + SQR(By) + SQR(Bz);
	const Real eta_H = lambda_ion*sqrt(Bmag)/(L0*ne);
	const Real Chyp = 8.;
	const Real eta_hyp = Chyp*eta_H;
	const Real D2JDx2 = (v0_J(b, TE::E1, 0, k, j, i + 1) - 2*v0_J(b, TE::E1, 0, k, j, i)
			  +  v0_J(b, TE::E1, 0, k, j, i - 1));
	Real D2JDy2 = 0.;
	//if (ndim>1) {}
	Real D2JDz2 = 0.;
	// if (ndim>2) {}
	//const Real f = 1. / (1.0 + exp((ne - 0.003) * 1000000.));	
	//if (f<1.) std::cout<<i<<": ne="<<ne<<", f="<<f<<std::endl;
	//v0_NI1_ed(b, TE::E1, 0, k, j, i) = v0_NI_ed(b, TE::E1, 0, k, j, i) - f*sigma0*mu0*eta_hyp* (D2JDx2 + D2JDy2 + D2JDz2);
	v0_NI1_ed(b, TE::E1, 0, k, j, i) = v0_NI_ed(b, TE::E1, 0, k, j, i) + sigma0*mu0*eta_hyp* (dBz_dy - dBy_dz);
	});

  const auto ibE2 = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jbE2 = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kbE2 = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbE2.s-(ndim>2), kbE2.e+(ndim>2), jbE2.s-(ndim>1), jbE2.e+(ndim>1), ibE2.s-1, ibE2.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rel_ed.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
        Real dBz_dx = -(v0_B(b, TE::F3, 0, k, j, i - (i>0))
                        - v0_B(b, TE::F3, 0, k, j, i)) / dx[0];
        Real dBx_dz = -(v0_B(b, TE::F1, 0, k - (ndim > 2 and k>0), j, i)
                        - v0_B(b, TE::F1, 0, k, j, i)) / dx[2];
        v0_rel_ed(b, TE::E2, 0, k, j, i) = v0_J(b, TE::E2, 0, k, j, i) - (-dBz_dx + dBx_dz);
	v0_NI_ed(b, TE::E2, 0, k, j, i) = v0_E(b, TE::E2, 0, k, j, i) - v0_B.flux(b, TE::E2, 0, k, j, i);
	
	// YH: add hyper-resisitivity 
        const auto v0_vel = Null<Real>();
        const Real ne = Z_ion * INTERP::CC_2_E2(v0_rho, v0_vel, b, k, j, i, ndim);
        const Real Bx = INTERP::F1_2_E2(v0_B, v0_vel, b, k, j, i, ndim);
        const Real By = INTERP::F2_2_E2(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bz = INTERP::F3_2_E2(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bmag = SQR(Bx) + SQR(By) + SQR(Bz);
        const Real eta_H = lambda_ion*sqrt(Bmag)/(L0*ne);
        const Real Chyp = 8.;
        const Real eta_hyp = Chyp*eta_H;
	const Real Jy_im1_e1 = INTERP::E2_2_E1(v0_J, v0_vel, b, k, j, i-1, ndim);
	const Real Jy_i_e1   = INTERP::E2_2_E1(v0_J, v0_vel, b, k, j, i  , ndim);
	const Real Jy_ip1_e1 = INTERP::E2_2_E1(v0_J, v0_vel, b, k, j, i+1, ndim);
        const Real D2JDx2 = (Jy_ip1_e1 -2*Jy_i_e1 + Jy_im1_e1);
        Real D2JDy2 = 0.;
        if (ndim>1) D2JDy2 = (v0_J(b, TE::E2, 0, k, j + 1, i) - 2*v0_J(b, TE::E2, 0, k, j, i)
                           +  v0_J(b, TE::E2, 0, k, j - 1, i));
        Real D2JDz2 = 0.;
        // if (ndim>2) {}
	//const Real f = 1. / (1.0 + exp((ne - 0.003) * 1000000.));
        //v0_NI1_ed(b, TE::E2, 0, k, j, i) = v0_NI_ed(b, TE::E2, 0, k, j, i) - f*sigma0*mu0*eta_hyp*(D2JDx2 + D2JDy2 + D2JDz2);
	v0_NI1_ed(b, TE::E2, 0, k, j, i) = v0_NI_ed(b, TE::E2, 0, k, j, i) + sigma0*mu0*eta_hyp* (-dBz_dx + dBx_dz);
	});
        
  const auto ibE3 = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jbE3 = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kbE3 = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbE3.s-(ndim>2), kbE3.e+(ndim>2), jbE3.s-(ndim>1), jbE3.e+(ndim>1), ibE3.s-1, ibE3.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rel_ed.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	Real dBy_dx = -(v0_B(b, TE::F2, 0, k, j, i - (i>0))
                        - v0_B(b, TE::F2, 0, k, j, i)) / dx[0];
        Real dBx_dy = -(v0_B(b, TE::F1, 0, k, j - (ndim > 1 and j>0), i)
                        - v0_B(b, TE::F1, 0, k, j, i)) / dx[1];
        v0_rel_ed(b, TE::E3, 0, k, j, i) = v0_J(b, TE::E3, 0, k, j, i) - (dBy_dx - dBx_dy);
	v0_NI_ed(b, TE::E3, 0, k, j, i) = v0_E(b, TE::E3, 0, k, j, i) - v0_B.flux(b, TE::E3, 0, k, j, i);
	// YH: add hyper-resisitivity
        const auto v0_vel = Null<Real>();
        const Real ne = Z_ion * INTERP::CC_2_E3(v0_rho, v0_vel, b, k, j, i, ndim);
        const Real Bx = INTERP::F1_2_E3(v0_B, v0_vel, b, k, j, i, ndim);
        const Real By = INTERP::F2_2_E3(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bz = INTERP::F3_2_E3(v0_B, v0_vel, b, k, j, i, ndim);
        const Real Bmag = SQR(Bx) + SQR(By) + SQR(Bz);
        const Real eta_H = lambda_ion*sqrt(Bmag)/(L0*ne);
        const Real Chyp = 8.;
        const Real eta_hyp = Chyp*eta_H;
        const Real Jz_im1_e1 = INTERP::E3_2_E1(v0_J, v0_vel, b, k, j, i-1, ndim);
        const Real Jz_i_e1   = INTERP::E3_2_E1(v0_J, v0_vel, b, k, j, i  , ndim);
        const Real Jz_ip1_e1 = INTERP::E3_2_E1(v0_J, v0_vel, b, k, j, i+1, ndim);
        const Real D2JDx2 = (Jz_ip1_e1 -2*Jz_i_e1 + Jz_im1_e1);
        Real D2JDy2 = 0.;
        //if (ndim>1) {}
        Real D2JDz2 = 0.;
        if (ndim>2) D2JDz2 = (v0_J(b, TE::E3, 0, k + 1, j, i) - 2*v0_J(b, TE::E3, 0, k, j, i)
                           +  v0_J(b, TE::E3, 0, k - 1, j, i));
	//const Real f = 1. / (1.0 + exp((ne - 0.003) * 1000000.));
        //v0_NI1_ed(b, TE::E3, 0, k, j, i) = v0_NI_ed(b, TE::E3, 0, k, j, i) - f*sigma0*mu0*eta_hyp*(D2JDx2 + D2JDy2 + D2JDz2);
	v0_NI1_ed(b, TE::E3, 0, k, j, i) = v0_NI_ed(b, TE::E3, 0, k, j, i) + sigma0*mu0*eta_hyp*(dBy_dx - dBx_dy);
	});
  
  // Obtain face components for relativistic & nonideal source terms
  // And fcc Bfield update
  const auto ibF1 = u0->GetBoundsI(IndexDomain::interior, TE::F1);
  const auto jbF1 = u0->GetBoundsJ(IndexDomain::interior, TE::F1);
  const auto kbF1 = u0->GetBoundsK(IndexDomain::interior, TE::F1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbF1.s-1*(ndim>2), kbF1.e+1*(ndim>2), jbF1.s-1*(ndim>1), jbF1.e+1*(ndim>1), ibF1.s-1, ibF1.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_B.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	const auto ax1 = coords.GetFaceAreaX1();
	const auto ax2 = coords.GetFaceAreaX2();
	const auto ax3 = coords.GetFaceAreaX3();

	// (a) For relativistic source term: $(J - \curl{B}) \times B$
	//     -> Interpolate edge J-\curl(B) to fcc
       	Real rel_f_y = INTERP::E2_2_F1(v0_rel_ed, v0_vel, b, k, j, i, ndim);
	Real rel_f_z = INTERP::E3_2_F1(v0_rel_ed, v0_vel, b, k, j, i, ndim);
	Real By_F1 = INTERP::F2_2_F1(v0_B, b, k, j, i, ndim);
	Real Bz_F1 = INTERP::F3_2_F1(v0_B, b, k, j, i, ndim);
	v0_rel(b, TE::F1, 0, k, j, i) = rel_f_y*Bz_F1 - rel_f_z*By_F1;
	// (b) For nonideal source term: $- \nabla \times (E + u \times B)$
        geometry::Coords<GEOM> coords_jp1(v0_B.GetCoordinates(b), k, j+(ndim>1), i);
        geometry::Coords<GEOM> coords_kp1(v0_B.GetCoordinates(b), k+(ndim>2), j, i);
        const auto e3len = coords.GetEdgeLengthX3();
        const auto e3len_jp1 = coords_jp1.GetEdgeLengthX3();
        const auto e2len = coords.GetEdgeLengthX2();
        const auto e2len_kp1 = coords_kp1.GetEdgeLengthX2();
        v0_NI(b, TE::F1, 0, k, j, i) = - (1./ax1[0]) *
                (e3len_jp1*v0_NI_ed(b, TE::E3, 0, k, j+(ndim>1), i) -
                 e3len*v0_NI_ed(b, TE::E3, 0, k, j, i));
        if (ndim>2) v0_NI(b, TE::F1, 0, k, j, i) += - (1./ax1[0]) * (
                 e2len*v0_NI_ed(b, TE::E2, 0, k, j, i) -
                 e2len_kp1*v0_NI_ed(b, TE::E2, 0, k+(ndim>2), j, i));

	v0_NI1(b, TE::F1, 0, k, j, i) = - (1./ax1[0]) *
                (e3len_jp1*v0_NI1_ed(b, TE::E3, 0, k, j+(ndim>1), i) -
                 e3len*v0_NI1_ed(b, TE::E3, 0, k, j, i));
        if (ndim>2) v0_NI1(b, TE::F1, 0, k, j, i) += - (1./ax1[0]) * (
                 e2len*v0_NI1_ed(b, TE::E2, 0, k, j, i) -
                 e2len_kp1*v0_NI1_ed(b, TE::E2, 0, k+(ndim>2), j, i));
	// (c) Update fcc Bfield with its source term
        v0_SrcB(b, TE::F1, 0, k, j, i) = v0_NI1(b, TE::F1, 0, k, j, i);
      });

  const auto ibF2 = u0->GetBoundsI(IndexDomain::interior, TE::F2);
  const auto jbF2 = u0->GetBoundsJ(IndexDomain::interior, TE::F2);
  const auto kbF2 = u0->GetBoundsK(IndexDomain::interior, TE::F2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbF2.s-1*(ndim>2), kbF2.e+1*(ndim>2), jbF2.s-1*(ndim>1), jbF2.e+1*(ndim>1), ibF2.s-1, ibF2.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_B.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = coords.GetFaceAreaX2();
        const auto ax3 = coords.GetFaceAreaX3();

        Real rel_f_x = INTERP::E1_2_F2(v0_rel_ed, v0_vel, b, k, j, i, ndim);
        Real rel_f_z = INTERP::E3_2_F2(v0_rel_ed, v0_vel, b, k, j, i, ndim);
	Real Bx_F2 = INTERP::F1_2_F2(v0_B, b, k, j, i, ndim);
        Real Bz_F2 = INTERP::F3_2_F2(v0_B, b, k, j, i, ndim);
        v0_rel(b, TE::F2, 0, k, j, i) = - rel_f_x*Bz_F2 + rel_f_z*Bx_F2;
	geometry::Coords<GEOM> coords_ip1(v0_B.GetCoordinates(b), k, j, i+1);
        geometry::Coords<GEOM> coords_kp1(v0_B.GetCoordinates(b), k+(ndim>2), j, i);
        const auto e3len_ip1 = coords_ip1.GetEdgeLengthX3();
	const auto e3len = coords.GetEdgeLengthX3();
        const auto e1len = coords.GetEdgeLengthX1();
        const auto e1len_kp1 = coords_kp1.GetEdgeLengthX1();
        v0_NI(b, TE::F2, 0, k, j, i) = - (1./ax2[0]) *
                (e3len*v0_NI_ed(b, TE::E3, 0, k, j, i) -
                 e3len_ip1*v0_NI_ed(b, TE::E3, 0, k, j, i+1));
        if (ndim>2) v0_NI(b, TE::F2, 0, k, j, i) +=  - (1./ax2[0]) * (
                 e1len_kp1*v0_NI_ed(b, TE::E1, 0, k+(ndim>2), j, i) -
                 e1len*v0_NI_ed(b, TE::E1, 0, k, j, i));

	v0_NI1(b, TE::F2, 0, k, j, i) = - (1./ax2[0]) *
                (e3len*v0_NI1_ed(b, TE::E3, 0, k, j, i) -
                 e3len_ip1*v0_NI1_ed(b, TE::E3, 0, k, j, i+1));
        if (ndim>2) v0_NI1(b, TE::F2, 0, k, j, i) +=  - (1./ax2[0]) * (
                 e1len_kp1*v0_NI1_ed(b, TE::E1, 0, k+(ndim>2), j, i) -
                 e1len*v0_NI1_ed(b, TE::E1, 0, k, j, i));

	v0_SrcB(b, TE::F2, 0, k, j, i) = v0_NI1(b, TE::F2, 0, k, j, i);
      });

  const auto ibF3 = u0->GetBoundsI(IndexDomain::interior, TE::F3);
  const auto jbF3 = u0->GetBoundsJ(IndexDomain::interior, TE::F3);
  const auto kbF3 = u0->GetBoundsK(IndexDomain::interior, TE::F3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbF3.s-1*(ndim>2), kbF3.e+1*(ndim>2), jbF3.s-1*(ndim>1), jbF3.e+1*(ndim>1), ibF3.s-1, ibF3.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_B.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = coords.GetFaceAreaX2();
        const auto ax3 = coords.GetFaceAreaX3();

        Real rel_f_x = INTERP::E1_2_F3(v0_rel_ed, v0_vel, b, k, j, i, ndim);
        Real rel_f_y = INTERP::E2_2_F3(v0_rel_ed, v0_vel, b, k, j, i, ndim);
	Real Bx_F3 = INTERP::F1_2_F3(v0_B, b, k, j, i, ndim);
        Real By_F3 = INTERP::F2_2_F3(v0_B, b, k, j, i, ndim);
        v0_rel(b, TE::F3, 0, k, j, i) = rel_f_x*By_F3 - rel_f_y*Bx_F3;
	geometry::Coords<GEOM> coords_ip1(v0_B.GetCoordinates(b), k, j, i+1);
        geometry::Coords<GEOM> coords_jp1(v0_B.GetCoordinates(b), k, j+(ndim>1), i);
        const auto e2len_ip1 = coords_ip1.GetEdgeLengthX2();
	const auto e2len = coords.GetEdgeLengthX2();
        const auto e1len_jp1 = coords_jp1.GetEdgeLengthX1();
	const auto e1len = coords.GetEdgeLengthX1();
        v0_NI(b, TE::F3, 0, k, j, i) = - (1./ax3[0]) *
                (e2len_ip1*v0_NI_ed(b, TE::E2, 0, k, j, i+1) -
                 e2len*v0_NI_ed(b, TE::E2, 0, k, j, i));
        if (ndim>1) v0_NI(b, TE::F3, 0, k, j, i) += - (1./ax3[0]) * (
                 e1len*v0_NI_ed(b, TE::E1, 0, k, j, i) -
                 e1len_jp1*v0_NI_ed(b, TE::E1, 0, k, j+(ndim>1), i));
	
	v0_NI1(b, TE::F3, 0, k, j, i) = - (1./ax3[0]) *
                (e2len_ip1*v0_NI1_ed(b, TE::E2, 0, k, j, i+1) -
                 e2len*v0_NI1_ed(b, TE::E2, 0, k, j, i));
        if (ndim>1) v0_NI1(b, TE::F3, 0, k, j, i) += - (1./ax3[0]) * (
                 e1len*v0_NI1_ed(b, TE::E1, 0, k, j, i) -
                 e1len_jp1*v0_NI1_ed(b, TE::E1, 0, k, j+(ndim>1), i));

	v0_SrcB(b, TE::F3, 0, k, j, i) = v0_NI1(b, TE::F3, 0, k, j, i);
        });
 
  // For cc source term update
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+2*(ndim>2), jb.s-2*(ndim>1), jb.e+2*(ndim>1), ib.s-2, ib.e+2,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

	// Interpolate variables to CC
	// 1. (a) Interpolate edge J to cc
	Real Jx = INTERP::E1_2_CC(v0_J, v0_vel, b, k, j, i, ndim);
        Real Jy = INTERP::E2_2_CC(v0_J, v0_vel, b, k, j, i, ndim);
        Real Jz = INTERP::E3_2_CC(v0_J, v0_vel, b, k, j, i, ndim);
	Real Jmag = SQR(Jx) + SQR(Jy) + SQR(Jz);
	// 1. (b) Interpolate relativistic source term to cc
	Kokkos::Array<Real, 3> RST{};
        RST[0] = INTERP::F1_2_CC(v0_rel, b, k, j, i, ndim);
        RST[1] = INTERP::F2_2_CC(v0_rel, b, k, j, i, ndim);
        RST[2] = INTERP::F3_2_CC(v0_rel, b, k, j, i, ndim);
	// 1. (c) Interpolate non-ideal source term to cc
	Kokkos::Array<Real, 3> NIT{};
        NIT[0] = INTERP::Two_F1_2_CC(v0_B, v0_NI, b, k, j, i, ndim);
        NIT[1] = INTERP::Two_F2_2_CC(v0_B, v0_NI, b, k, j, i, ndim);
        NIT[2] = INTERP::Two_F3_2_CC(v0_B, v0_NI, b, k, j, i, ndim);
	// Update cc variables
	// 2. (b) Update with Energy source term
	//        -> Requires u thus must update this before momentum source update!
	const Real ne = Z_ion * v0_rho(b, TE::CC, 0, k, j, i);
	const Real Pe = v0_Se(b, TE::CC, 0, k, j, i) * pow(ne, gamma-1.);
        const Real eta = ComputeEta(eta_type, ne, Pe);
	if (std::isnan(eta) or eta<0.) printf("YH: @update (%d,%d,%d) eta=%f & Pe=%f & ne=%f & Se=%f \n"
			,i,j,k,eta,Pe, ne,v0_Se(b, TE::CC, 0, k, j, i));
	v0_Srcener(b, TE::CC, 0, k, j, i) = 
		((NIT[0] + NIT[1] + NIT[2]) +
		 (v0_vel(b,TE::CC,0,k,j,i)*RST[0] + 
		  v0_vel(b,TE::CC,1,k,j,i)*RST[1] + 
		  v0_vel(b,TE::CC,2,k,j,i)*RST[2]) +
		 eta*Jmag);
	// 2. (a) Update with Momentum source term
        for (int vi=0; vi<3; ++vi)
          v0_Srcmom(b, TE::CC, vi, k, j, i) = RST[vi];
	// 2. (c) Uconst Real ne_roe = Z_ion*wroe_idn;pdate Se with source term
	Real tau_ei = collision::tau_ei_FLASH(v0_rho(b, TE::CC, 0, k, j, i), 
			vTi(b, TE::CC, 0, k, j, i), vTe(b, TE::CC, 0, k, j, i));
	// -> Assumes rate of momentum transfer same for e & i where both species are nearly thermal
	Real tau_ie = (m_ion/(Z_ion*me)) * tau_ei;
	Real zeta_ei = 0.5*((tau_ei*m_ion)/(tau_ie*me+tau_ei*m_ion) + tau_ie/(tau_ie+tau_ei));
	v0_SrcSe(b, TE::CC, 0, k, j, i) = (gamma-1.)*pow(ne,1.-gamma)*zeta_ei*eta*Jmag;
	});

  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  TaskStatus XMHD::ApplyExplicitSource
//! \brief (Explicit source for other variables.)
template <Coordinates GEOM>
TaskStatus ApplyExplicitSource(MeshData<Real> *u0, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;

  // Extract integrator weights
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;
  if (beta_dt==0.) return TaskStatus::complete;

  // Retrieve required parameters
  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  // Collect CC package
  std::vector<MetadataFlag> flagsm1({Metadata::Cell, Metadata::Conserved,            
		  Metadata::GetUserFlag("Density")});       
  static auto descm1 = MakePackDescriptor<any>(u0, flagsm1);    
  const auto v0_rho = descm1.GetPack(u0);
  std::vector<MetadataFlag> flags0({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Velfield")});
  static auto desc0 = MakePackDescriptor<any>(u0, flags0);
  const auto v0_mom = desc0.GetPack(u0);
  //std::vector<MetadataFlag> flags_Bcc({Metadata::Cell, Metadata::Conserved,
  //                Metadata::GetUserFlag("Bfield")});
  //static auto descBcc = MakePackDescriptor<any>(u0, flags_Bcc);
  //const auto v0_Bcc = descBcc.GetPack(u0);

  std::vector<MetadataFlag> flags1({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Energy")});
  static auto desc1 = MakePackDescriptor<any>(u0, flags1);
  const auto v0_ener = desc1.GetPack(u0);
  std::vector<MetadataFlag> flags2({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("electron")});
  static auto desc2 = MakePackDescriptor<any>(u0, flags2);
  const auto v0_Se = desc2.GetPack(u0);
  // Collect FCC package
  std::vector<MetadataFlag> flags_B({Metadata::Face, Metadata::Conserved, 
		  Metadata::WithFluxes, Metadata::GetUserFlag("Bfield")});
  static auto desc_B = MakePackDescriptor<any>(u0, flags_B, {parthenon::PDOpt::WithFluxes});
  const auto v0_B = desc_B.GetPack(u0);
  // Compute source package to update
  std::vector<MetadataFlag> flags_SrcB({Metadata::Face, Metadata::GetUserFlag("Source_Bfield")});
  static auto desc_SrcB = MakePackDescriptor<any>(u0, flags_SrcB);
  const auto v0_SrcB = desc_SrcB.GetPack(u0);
  std::vector<MetadataFlag> flags_Srcmom({Metadata::Cell, Metadata::GetUserFlag("Source_mom")});
  static auto desc_Srcmom = MakePackDescriptor<any>(u0, flags_Srcmom);
  const auto v0_Srcmom = desc_Srcmom.GetPack(u0);
  std::vector<MetadataFlag> flags_Srcener({Metadata::Cell, Metadata::GetUserFlag("Source_ener")});
  static auto desc_Srcener = MakePackDescriptor<any>(u0, flags_Srcener);
  const auto v0_Srcener = desc_Srcener.GetPack(u0);
  std::vector<MetadataFlag> flags_SrcSe({Metadata::Cell, Metadata::GetUserFlag("Source_Se")});
  static auto desc_SrcSe = MakePackDescriptor<any>(u0, flags_SrcSe);
  const auto v0_SrcSe = desc_SrcSe.GetPack(u0);

  // YH: CT's energy correction to improve positive pressure -> see if it helps with HLLD
  /*const auto ibcc = u0->GetBoundsI(IndexDomain::interior);
  const auto jbcc = u0->GetBoundsJ(IndexDomain::interior);
  const auto kbcc = u0->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbcc.s, kbcc.e, jbcc.s, jbcc.e, ibcc.s, ibcc.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	const Real Bfcc_x = INTERP::F1_2_CC(v0_B, b, k, j, i, ndim);
	const Real Bfcc_y = INTERP::F2_2_CC(v0_B, b, k, j, i, ndim);
	const Real Bfcc_z = INTERP::F3_2_CC(v0_B, b, k, j, i, ndim);
	const Real Bmag_fcc = SQR(Bfcc_x) + SQR(Bfcc_y) + SQR(Bfcc_z);
	const Real Bmag_cc = SQR(v0_Bcc(b, TE::CC, 0, k, j, i)) +
			     SQR(v0_Bcc(b, TE::CC, 1, k, j, i)) +
			     SQR(v0_Bcc(b, TE::CC, 2, k, j, i));
	v0_ener(b, TE::CC, 0, k, j, i) += 0.5*(Bmag_fcc - Bmag_cc);
	});*/
	

  // Obtain edge component for relativistic source term
  const auto ibf1 = u0->GetBoundsI(IndexDomain::interior, TE::F1);
  const auto jbf1 = u0->GetBoundsJ(IndexDomain::interior, TE::F1);
  const auto kbf1 = u0->GetBoundsK(IndexDomain::interior, TE::F1);
  // And fcc Bfield update
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbf1.s-(ndim>2), kbf1.e+(ndim>2), jbf1.s-(ndim>1), jbf1.e+(ndim>1), ibf1.s-1, ibf1.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
	// (c) Update fcc Bfield with its source term
	v0_B(b, TE::F1, 0, k, j, i) += beta_dt*v0_SrcB(b, TE::F1, 0, k, j, i);
        });
  const auto ibf2 = u0->GetBoundsI(IndexDomain::interior, TE::F2);
  const auto jbf2 = u0->GetBoundsJ(IndexDomain::interior, TE::F2);
  const auto kbf2 = u0->GetBoundsK(IndexDomain::interior, TE::F2);
  // And fcc Bfield update
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbf2.s-(ndim>2), kbf2.e+(ndim>2), jbf2.s-(ndim>1), jbf2.e+(ndim>1), ibf2.s-1, ibf2.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        v0_B(b, TE::F2, 0, k, j, i) += beta_dt*v0_SrcB(b, TE::F2, 0, k, j, i);
        });
  const auto ibf3 = u0->GetBoundsI(IndexDomain::interior, TE::F3);
  const auto jbf3 = u0->GetBoundsJ(IndexDomain::interior, TE::F3);
  const auto kbf3 = u0->GetBoundsK(IndexDomain::interior, TE::F3);
  // And fcc Bfield update
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbf3.s-(ndim>2), kbf3.e+(ndim>2), jbf3.s-(ndim>1), jbf3.e+(ndim>1), ibf3.s-1, ibf3.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        v0_B(b, TE::F3, 0, k, j, i) += beta_dt*v0_SrcB(b, TE::F3, 0, k, j, i);
        });

 
  // For cc source term update
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
	const Real norm = SQR(B0)/(n0*m_ion*SQR(char_speed)*mu0);
	// Did not include norm here as I need include it in my ideal MHD riemann solver too
	if (norm<0.99 or norm>1.) std::cout<<"Error! As norm = "<<norm<<std::endl;

	v0_ener(b, TE::CC, 0, k, j, i) += beta_dt*v0_Srcener(b, TE::CC, 0, k, j, i); 
	// 2. (a) Update with Momentum source term
        for (int vi=0; vi<3; ++vi)
          v0_mom(b, TE::CC, vi, k, j, i) += beta_dt*v0_Srcmom(b, TE::CC, vi, k, j, i);
	// 2. (c) Update Se with source term 
	v0_Se(b, TE::CC, 0, k, j, i) += beta_dt*v0_SrcSe(b, TE::CC, 0, k, j, i);
	
	});

  return TaskStatus::complete;
}


} // namespace xmhd

#endif // XMHD_XMHD_HPP_
