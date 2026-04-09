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
#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

#include "mhd/extended/interp.hpp"
#include "mhd/integrator/defs_int.hpp"

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace MHD {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::DeepCopyConservedFaceData
//! \brief
inline TaskStatus DeepCopyConservedFaceData(MeshData<Real> *to, MeshData<Real> *from) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;

  std::vector<MetadataFlag> flags({Metadata::Face,Metadata::Conserved});
  static auto desc = MakePackDescriptor<any>(to, flags);
  const auto vt = desc.GetPack(to);
  const auto vf = desc.GetPack(from);

  const auto ibe_F1 = to->GetBoundsI(IndexDomain::entire,TE::F1);
  const auto jbe_F1 = to->GetBoundsJ(IndexDomain::entire,TE::F1);
  const auto kbe_F1 = to->GetBoundsK(IndexDomain::entire,TE::F1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DeepCopyConservedData_F1", parthenon::DevExecSpace(), 0,
      to->NumBlocks() - 1, kbe_F1.s, kbe_F1.e, jbe_F1.s, jbe_F1.e, ibe_F1.s, ibe_F1.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;
        for (int n = vt.GetLowerBound(b); n <= vt.GetUpperBound(b); ++n) {
          vt(b, TE::F1, n, k, j, i) = vf(b, TE::F1, n, k, j, i);
        }
      });

  const auto ibe_F2 = to->GetBoundsI(IndexDomain::entire,TE::F2);
  const auto jbe_F2 = to->GetBoundsJ(IndexDomain::entire,TE::F2);
  const auto kbe_F2 = to->GetBoundsK(IndexDomain::entire,TE::F2);
  parthenon::par_for(                                                                                                                     
      DEFAULT_LOOP_PATTERN, "DeepCopyConservedData_F2", parthenon::DevExecSpace(), 0,
      to->NumBlocks() - 1, kbe_F2.s, kbe_F2.e, jbe_F2.s, jbe_F2.e, ibe_F2.s, ibe_F2.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;                                                                         
	for (int n = vt.GetLowerBound(b); n <= vt.GetUpperBound(b); ++n) {
          vt(b, TE::F2, n, k, j, i) = vf(b, TE::F2, n, k, j, i);
        }                                                                      
      });

  const auto ibe_F3 = to->GetBoundsI(IndexDomain::entire,TE::F3);
  const auto jbe_F3 = to->GetBoundsJ(IndexDomain::entire,TE::F3);
  const auto kbe_F3 = to->GetBoundsK(IndexDomain::entire,TE::F3);
  parthenon::par_for(                            
      DEFAULT_LOOP_PATTERN, "DeepCopyConservedData_F3", parthenon::DevExecSpace(), 0,
      to->NumBlocks() - 1, kbe_F3.s, kbe_F3.e, jbe_F3.s, jbe_F3.e, ibe_F3.s, ibe_F3.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        using parthenon::TopologicalElement;                                 
	for (int n = vt.GetLowerBound(b); n <= vt.GetUpperBound(b); ++n) {
          vt(b, TE::F3, n, k, j, i) = vf(b, TE::F3, n, k, j, i);
        }                                                           
      });
  return TaskStatus::complete;
}

	
//==============================================================================================
//! \fn  TaskStatus ArtemisUtils::CalculateEMF
//! \brief
template <Coordinates GEOM>
TaskStatus CalculateEMF(MeshData<Real> *u0) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const int ndim = pm->ndim;

  // YH: retrieve flux
  std::vector<MetadataFlag> flags2({Metadata::Cell, Metadata::Conserved, 
		  Metadata::GetUserFlag("Bfield"), Metadata::WithFluxes});
  static auto desc2 = MakePackDescriptor<any>(u0, flags2, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc2.GetPack(u0);

  std::vector<MetadataFlag> flags3({Metadata::Face, Metadata::Conserved, 
		  Metadata::GetUserFlag("Bfield"), Metadata::WithFluxes});
  static auto desc3 = MakePackDescriptor<any>(u0, flags3, {parthenon::PDOpt::WithFluxes});
  const auto v0_fcc = desc3.GetPack(u0);

  const auto ib_1 = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jb_1 = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kb_1 = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  int ib_1s = ib_1.s - 2, ib_1e = ib_1.e + 2;
  int jb_1s = jb_1.s, jb_1e = jb_1.e;
  if (ndim>1) {jb_1s -= 2; jb_1e += 2;}
  int kb_1s = kb_1.s, kb_1e = kb_1.e;
  if (ndim>2) {kb_1s -= 2; kb_1e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_1s, kb_1e, jb_1s, jb_1e, ib_1s, ib_1e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

        const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) =
            0.25*(v0.flux(b, X3DIR, n_cc-1, k, j-1, i) +
                        v0.flux(b, X3DIR, n_cc-1, k, j, i) -
                        v0.flux(b, X2DIR, n_cc, k-1, j, i) -
                        v0.flux(b, X2DIR, n_cc, k, j, i));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) = - v0.flux(b, X2DIR, n_cc, k, j, i);
        } else {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) = 0.;
        }
  });

  const auto ib_2 = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jb_2 = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kb_2 = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  int ib_2s = ib_2.s - 2, ib_2e = ib_2.e + 2;
  int jb_2s = jb_2.s, jb_2e = jb_2.e;
  if (ndim>1) {jb_2s -= 2; jb_2e += 2;}
  int kb_2s = kb_2.s, kb_2e = kb_2.e;
  if (ndim>2) {kb_2s -= 2; kb_2e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_y", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_2s, kb_2e, jb_2s, jb_2e, ib_2s, ib_2e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

        const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) =
            0.25*(v0.flux(b, X1DIR, n_cc, k-1, j, i) +
                        v0.flux(b, X1DIR, n_cc, k, j, i) -
                        v0.flux(b, X3DIR, n_cc-2, k, j, i-1) -
                        v0.flux(b, X3DIR, n_cc-2, k, j, i));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) = v0.flux(b, X1DIR, n_cc, k, j, i);
        } else {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) = v0.flux(b, X1DIR, n_cc, k, j, i);
        }
  });

  const auto ib_3 = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jb_3 = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kb_3 = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  int ib_3s = ib_3.s - 2, ib_3e = ib_3.e + 2;
  int jb_3s = jb_3.s, jb_3e = jb_3.e;
  if (ndim>1) {jb_3s -= 2; jb_3e += 2;}
  int kb_3s = kb_3.s, kb_3e = kb_3.e;
  if (ndim>2) {kb_3s -= 2; kb_3e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_z", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_3s, kb_3e, jb_3s, jb_3e, ib_3s, ib_3e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

        const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) =
            0.25*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1) +
                        v0.flux(b, X2DIR, n_cc-2, k, j, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j-1, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j, i));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) =
            0.25*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1) +
                        v0.flux(b, X2DIR, n_cc-2, k, j, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j-1, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j, i));
        } else {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) = - v0.flux(b, X1DIR, n_cc-1, k, j, i);
        }
  });

  return TaskStatus::complete;
}

//==============================================================================================
//! \fn  TaskStatus ArtemisUtils::CalculateEMF
//! \brief
template <Coordinates GEOM>
TaskStatus CalculateEMF_Mignone2020(MeshData<Real> *u0) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const int ndim = pm->ndim;

  // YH: retrieve flux
  std::vector<MetadataFlag> flags2({Metadata::Cell, Metadata::Conserved, 
		  Metadata::GetUserFlag("Bfield"), Metadata::WithFluxes});
  static auto desc2 = MakePackDescriptor<any>(u0, flags2, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc2.GetPack(u0);

  std::vector<MetadataFlag> flags3({Metadata::Face, Metadata::Conserved, 
		  Metadata::GetUserFlag("Bfield"), Metadata::WithFluxes});
  static auto desc3 = MakePackDescriptor<any>(u0, flags3, {parthenon::PDOpt::WithFluxes});
  const auto vBf = desc3.GetPack(u0);

  // For upw CT
  std::vector<MetadataFlag> flags4({Metadata::Face, Metadata::GetUserFlag("VelL")});
  static auto desc4 = MakePackDescriptor<any>(u0, flags4);
  const auto velL = desc4.GetPack(u0);
  std::vector<MetadataFlag> flags5({Metadata::Face, Metadata::GetUserFlag("VelR")});
  static auto desc5 = MakePackDescriptor<any>(u0, flags5);
  const auto velR = desc5.GetPack(u0);
  std::vector<MetadataFlag> flags6({Metadata::Face, Metadata::GetUserFlag("LambdaL")});
  static auto desc6 = MakePackDescriptor<any>(u0, flags6);
  const auto lambdaL = desc6.GetPack(u0);
  std::vector<MetadataFlag> flags7({Metadata::Face, Metadata::GetUserFlag("LambdaR")});
  static auto desc7 = MakePackDescriptor<any>(u0, flags7);
  const auto lambdaR = desc7.GetPack(u0);

  const auto ib_1 = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jb_1 = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kb_1 = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  int ib_1s = ib_1.s - 2, ib_1e = ib_1.e + 2;
  int jb_1s = jb_1.s, jb_1e = jb_1.e;
  if (ndim>1) {jb_1s -= 2; jb_1e += 2;}
  int kb_1s = kb_1.s, kb_1e = kb_1.e;
  if (ndim>2) {kb_1s -= 2; kb_1e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_1s, kb_1e, jb_1s, jb_1e, ib_1s, ib_1e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

	if (ndim>1) {
	  const Real a = 0.5;
	  Real alphay_m = - std::min(0.,std::min(
			lambdaL(b, TE::F2, 0, k-(ndim>2), j-1, i), lambdaL(b, TE::F2, 0, k, j-1, i)));
	  Real alphay_p = std::max(0.,std::max(
                        lambdaR(b, TE::F2, 0, k-(ndim>2), j-1, i), lambdaR(b, TE::F2, 0, k, j-1, i)));
	  const Real vy_W = (alphay_p*velL(b, TE::F2, 0, k, j - 1, i) + 
			     alphay_m*velR(b, TE::F2, 0, k, j - 1, i)) / (alphay_p + alphay_m);
	  alphay_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F2, 0, k-(ndim>2), j, i), lambdaL(b, TE::F2, 0, k, j, i)));
          alphay_p = std::max(0.,std::max(
                        lambdaR(b, TE::F2, 0, k-(ndim>2), j, i), lambdaR(b, TE::F2, 0, k, j, i)));
          const Real vy_E = (alphay_p*velL(b, TE::F2, 0, k, j, i) +
                             alphay_m*velR(b, TE::F2, 0, k, j, i)) / (alphay_p + alphay_m);
	  Real dL_jkm1 = std::abs(lambdaL(b, TE::F2, 0, k-(ndim>2), j, i))/2.;
          Real dL_jk = std::abs(lambdaL(b, TE::F2, 0, k, j, i))/2.;
          Real dy_W = 0.5*(dL_jkm1 + dL_jk);
          Real dR_jkm1 = std::abs(lambdaR(b, TE::F2, 0, k-(ndim>2), j, i))/2.;
          Real dR_jk = std::abs(lambdaR(b, TE::F2, 0, k, j, i))/2.;
          Real dy_E = 0.5*(dR_jkm1 + dR_jk);
	  vBf.flux(b, TE::E1, 0, k, j, i) = - (
		((a*vy_W*vBf(b, TE::F3, 0, k, j - 1, i)) + (a*vy_E*vBf(b, TE::F3, 0, k, j, i))) -
		((dy_E*vBf(b, TE::F3, 0, k, j, i)) - (dy_W*vBf(b, TE::F3, 0, k, j - 1, i))));
	  if (ndim>2) {
	    Real alphaz_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F3, 0, k-1, j-1, i), lambdaL(b, TE::F3, 0, k-1, j, i)));
            Real alphaz_p = std::max(0.,std::max(
                        lambdaR(b, TE::F3, 0, k-1, j-1, i), lambdaR(b, TE::F3, 0, k-1, j, i)));
	    const Real vz_S = (alphaz_p*velL(b, TE::F3, 0, k - 1, j, i) +
                               alphaz_m*velR(b, TE::F3, 0, k - 1, j, i)) / (alphaz_p + alphaz_m);
	    alphaz_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F3, 0, k, j-1, i), lambdaL(b, TE::F3, 0, k, j, i)));
            alphaz_p = std::max(0.,std::max(
                        lambdaR(b, TE::F3, 0, k, j-1, i), lambdaR(b, TE::F3, 0, k, j, i)));
            const Real vz_N = (alphaz_p*velL(b, TE::F3, 0, k, j, i) +
                               alphaz_m*velR(b, TE::F3, 0, k, j, i)) / (alphaz_p + alphaz_m);
	    Real dL_jm1k = std::abs(lambdaL(b, TE::F3, 0, k, j-1, i))/2.;
            dL_jk = std::abs(lambdaL(b, TE::F3, 0, k, j, i))/2.;
            Real dz_S = 0.5*(dL_jm1k + dL_jk);
            Real dR_jm1k = std::abs(lambdaR(b, TE::F3, 0, k, j-1, i))/2.;
            dR_jk = std::abs(lambdaR(b, TE::F3, 0, k, j, i))/2.;
            Real dz_N = 0.5*(dR_jm1k + dR_jk);
	    vBf.flux(b, TE::E1, 0, k, j, i) += (
                ((a*vz_S*vBf(b, TE::F2, 0, k - 1, j, i)) + (a*vz_N*vBf(b, TE::F2, 0, k, j, i))) -
                ((dz_N*vBf(b, TE::F2, 0, k, j, i)) - (dz_S*vBf(b, TE::F2, 0, k - 1, j, i))));
	  }
	}
  });

  const auto ib_2 = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jb_2 = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kb_2 = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  int ib_2s = ib_2.s - 2, ib_2e = ib_2.e + 2;
  int jb_2s = jb_2.s, jb_2e = jb_2.e;
  if (ndim>1) {jb_2s -= 2; jb_2e += 2;}
  int kb_2s = kb_2.s, kb_2e = kb_2.e;
  if (ndim>2) {kb_2s -= 2; kb_2e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_y", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_2s, kb_2e, jb_2s, jb_2e, ib_2s, ib_2e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

	const Real a = 0.5;
	Real alphax_m = - std::min(0.,std::min(
			lambdaL(b, TE::F1, 0, k-(ndim>2), j, i-1), lambdaL(b, TE::F1, 0, k, j, i-1)));
	Real alphax_p = std::max(0.,std::max(
                        lambdaR(b, TE::F1, 0, k-(ndim>2), j, i-1), lambdaR(b, TE::F1, 0, k, j, i-1)));
	const Real vx_S = (alphax_p*velL(b, TE::F1, 0, k, j, i - 1) + 
			   alphax_m*velR(b, TE::F1, 0, k, j, i - 1)) / (alphax_p + alphax_m);
	alphax_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F1, 0, k-(ndim>2), j, i), lambdaL(b, TE::F1, 0, k, j, i)));
        alphax_p = std::max(0.,std::max(
                        lambdaR(b, TE::F1, 0, k-(ndim>2), j, i), lambdaR(b, TE::F1, 0, k, j, i)));
        const Real vx_N = (alphax_p*velL(b, TE::F1, 0, k, j, i) +
                           alphax_m*velR(b, TE::F1, 0, k, j, i)) / (alphax_p + alphax_m);
	Real dL_ikm1 = std::abs(lambdaL(b, TE::F1, 0, k-(ndim>2), j, i))/2.;
        Real dL_ik = std::abs(lambdaL(b, TE::F1, 0, k, j, i))/2.;
        Real dx_S = 0.5*(dL_ikm1 + dL_ik);
        Real dR_ikm1 = std::abs(lambdaR(b, TE::F1, 0, k-(ndim>2), j, i))/2.;
        Real dR_ik = std::abs(lambdaR(b, TE::F1, 0, k, j, i))/2.;
        Real dx_N = 0.5*(dR_ikm1 + dR_ik);
	vBf.flux(b, TE::E2, 0, k, j, i) = (
		((a*vx_S*vBf(b, TE::F3, 0, k, j, i - 1)) + (a*vx_N*vBf(b, TE::F3, 0, k, j, i))) -
		((dx_N*vBf(b, TE::F3, 0, k, j, i)) - (dx_S*vBf(b, TE::F3, 0, k, j, i - 1))));
	if (ndim>2) {
	  Real alphaz_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F3, 0, k-1, j, i-1), lambdaL(b, TE::F3, 0, k-1, j, i)));
          Real alphaz_p = std::max(0.,std::max(
                        lambdaR(b, TE::F3, 0, k-1, j, i-1), lambdaR(b, TE::F3, 0, k-1, j, i)));
	  const Real vz_W = (alphaz_p*velL(b, TE::F3, 0, k - 1, j, i) +
                             alphaz_m*velR(b, TE::F3, 0, k - 1, j, i)) / (alphaz_p + alphaz_m);
	  alphaz_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F3, 0, k, j, i-1), lambdaL(b, TE::F3, 0, k, j, i)));
          alphaz_p = std::max(0.,std::max(
                        lambdaR(b, TE::F3, 0, k, j, i-1), lambdaR(b, TE::F3, 0, k, j, i)));
          const Real vz_E = (alphaz_p*velL(b, TE::F3, 0, k, j, i) +
                             alphaz_m*velR(b, TE::F3, 0, k, j, i)) / (alphaz_p + alphaz_m);
	  Real dL_im1k = std::abs(lambdaL(b, TE::F3, 0, k, j, i-1))/2.;
          dL_ik = std::abs(lambdaL(b, TE::F3, 0, k, j, i))/2.;
          Real dz_W = 0.5*(dL_im1k + dL_ik);
          Real dR_im1k = std::abs(lambdaR(b, TE::F3, 0, k, j, i-1))/2.;
          Real dR_ik = std::abs(lambdaR(b, TE::F3, 0, k, j, i))/2.;
          Real dz_E = 0.5*(dR_im1k + dR_ik);
	  vBf.flux(b, TE::E2, 0, k, j, i) += - (
                ((a*vz_W*vBf(b, TE::F1, 0, k - 1, j, i)) + (a*vz_E*vBf(b, TE::F1, 0, k, j, i))) -
                ((dz_E*vBf(b, TE::F1, 0, k, j, i)) - (dz_W*vBf(b, TE::F1, 0, k - 1, j, i))));
	}
  });

  const auto ib_3 = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jb_3 = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kb_3 = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  int ib_3s = ib_3.s - 2, ib_3e = ib_3.e + 2;
  int jb_3s = jb_3.s, jb_3e = jb_3.e;
  if (ndim>1) {jb_3s -= 2; jb_3e += 2;}
  int kb_3s = kb_3.s, kb_3e = kb_3.e;
  if (ndim>2) {kb_3s -= 2; kb_3e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_z", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_3s, kb_3e, jb_3s, jb_3e, ib_3s, ib_3e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

	const Real a = 0.5;
	Real alphax_m = - std::min(0., std::min(
			lambdaL(b, TE::F1, 0, k, j-(ndim>1), i-1), lambdaL(b, TE::F1, 0, k, j, i-1)));
	Real alphax_p = std::max(0., std::max(
                        lambdaR(b, TE::F1, 0, k, j-(ndim>1), i-1), lambdaR(b, TE::F1, 0, k, j, i-1)));
	const Real vx_W = (alphax_p*velL(b, TE::F1, 0, k, j, i - 1) + 
			   alphax_m*velR(b, TE::F1, 0, k, j, i - 1)) / (alphax_p + alphax_m);
	alphax_m = - std::min(0., std::min(
                        lambdaL(b, TE::F1, 0, k, j-(ndim>1), i), lambdaL(b, TE::F1, 0, k, j, i)));
        alphax_p = std::max(0., std::max(
                        lambdaR(b, TE::F1, 0, k, j-(ndim>1), i), lambdaR(b, TE::F1, 0, k, j, i)));
        const Real vx_E = (alphax_p*velL(b, TE::F1, 0, k, j, i) +
                           alphax_m*velR(b, TE::F1, 0, k, j, i)) / (alphax_p + alphax_m);
	Real dL_ijm1 = std::abs(lambdaL(b, TE::F1, 0, k, j - (ndim>1), i))/2.;
	Real dL_ij = std::abs(lambdaL(b, TE::F1, 0, k, j, i))/2.;
	Real dx_W = 0.5*(dL_ijm1 + dL_ij);
	Real dR_ijm1 = std::abs(lambdaR(b, TE::F1, 0, k, j - (ndim>1), i))/2.;
        Real dR_ij = std::abs(lambdaR(b, TE::F1, 0, k, j, i))/2.;
        Real dx_E = 0.5*(dR_ijm1 + dR_ij);
	vBf.flux(b, TE::E3, 0, k, j, i) = - (
		((a*vx_W*vBf(b, TE::F2, 0, k, j, i - 1)) + (a*vx_E*vBf(b, TE::F2, 0, k, j, i))) -
		((dx_E*vBf(b, TE::F2, 0, k, j, i)) - (dx_W*vBf(b, TE::F2, 0, k, j, i - 1))));
	if (ndim>1) {
	  Real alphay_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F2, 0, k, j-1, i-1), lambdaL(b, TE::F2, 0, k, j-1, i)));
          Real alphay_p = std::max(0.,std::max(
                        lambdaR(b, TE::F2, 0, k, j-1, i-1), lambdaR(b, TE::F2, 0, k, j-1, i)));
	  const Real vy_S = (alphay_p*velL(b, TE::F2, 0, k, j - 1, i) +
                             alphay_m*velR(b, TE::F2, 0, k, j - 1, i)) / (alphay_p + alphay_m);
	  alphay_m = - std::min(0.,std::min(
                        lambdaL(b, TE::F2, 0, k, j, i-1), lambdaL(b, TE::F2, 0, k, j, i)));
          alphay_p = std::max(0.,std::max(
                        lambdaR(b, TE::F2, 0, k, j, i-1), lambdaR(b, TE::F2, 0, k, j, i)));
          const Real vy_N = (alphay_p*velL(b, TE::F2, 0, k, j, i) +
                             alphay_m*velR(b, TE::F2, 0, k, j, i)) / (alphay_p + alphay_m);
	  Real dL_im1j = std::abs(lambdaL(b, TE::F2, 0, k, j, i - 1))/2.;
          dL_ij = std::abs(lambdaL(b, TE::F2, 0, k, j, i))/2.;
          Real dy_S = 0.5*(dL_im1j + dL_ij);
          Real dR_im1j = std::abs(lambdaR(b, TE::F2, 0, k, j, i - 1))/2.;
          dR_ij = std::abs(lambdaR(b, TE::F2, 0, k, j, i))/2.;
          Real dy_N = 0.5*(dR_im1j + dR_ij);
	  vBf.flux(b, TE::E3, 0, k, j, i) += (
                ((a*vy_S*vBf(b, TE::F1, 0, k, j - 1, i)) + (a*vy_N*vBf(b, TE::F1, 0, k, j, i))) -
                ((dy_N*vBf(b, TE::F1, 0, k, j, i)) - (dy_S*vBf(b, TE::F1, 0, k, j - 1, i))));
	}

  });

  return TaskStatus::complete;
}


//==============================================================================================
//! \fn  TaskStatus ArtemisUtils::CalculateEMF
//! \brief CT-Contact scheme from Gardiner & Stone 2004
template <Coordinates GEOM>
TaskStatus CalculateEMF_GS2004(MeshData<Real> *u0) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const int ndim = pm->ndim;

  // YH: retrieve flux
  std::vector<MetadataFlag> flags0({Metadata::Cell, Metadata::Conserved,
                  Metadata::GetUserFlag("Density"), Metadata::WithFluxes});
  static auto desc0 = MakePackDescriptor<any>(u0, flags0, {parthenon::PDOpt::WithFluxes});
  const auto v0_rho = desc0.GetPack(u0);
  std::vector<MetadataFlag> flags1({Metadata::Cell, Metadata::Derived, 
		  Metadata::GetUserFlag("Velfield")});   
  static auto desc1 = MakePackDescriptor<any>(u0, flags1);
  const auto v0_vel = desc1.GetPack(u0);
  std::vector<MetadataFlag> flags2({Metadata::Cell, Metadata::Conserved, 
		  Metadata::GetUserFlag("Bfield"), Metadata::WithFluxes});
  static auto desc2 = MakePackDescriptor<any>(u0, flags2, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc2.GetPack(u0);

  std::vector<MetadataFlag> flags3({Metadata::Face, Metadata::Conserved, 
		  Metadata::GetUserFlag("Bfield"), Metadata::WithFluxes});
  static auto desc3 = MakePackDescriptor<any>(u0, flags3, {parthenon::PDOpt::WithFluxes});
  const auto v0_fcc = desc3.GetPack(u0);

  const auto ib_1 = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jb_1 = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kb_1 = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  int ib_1s = ib_1.s - 2, ib_1e = ib_1.e + 2;
  int jb_1s = jb_1.s, jb_1e = jb_1.e;
  if (ndim>1) {jb_1s -= 2; jb_1e += 2;}
  int kb_1s = kb_1.s, kb_1e = kb_1.e;
  if (ndim>2) {kb_1s -= 2; kb_1e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_1s, kb_1e, jb_1s, jb_1e, ib_1s, ib_1e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
	geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

        const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) =
            0.25*(v0.flux(b, X3DIR, n_cc-1, k, j-1, i) +
                        v0.flux(b, X3DIR, n_cc-1, k, j, i) -
                        v0.flux(b, X2DIR, n_cc, k-1, j, i) -
                        v0.flux(b, X2DIR, n_cc, k, j, i));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) = - v0.flux(b, X2DIR, n_cc, k, j, i);
        } else {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) = 0.;
        }

	// Compute upwind correction for Ex (E1)
	if (ndim>1) {
    	  // Fy(Bz) at j-1 and j: Fyc_Bz_jm1, Fyc_Bz
    	  Real Fyc_Bz_km1 = v0(b, 2, k-(ndim>2), j, i) * v0_vel(b, 1, k-(ndim>2), j, i) - 
	    	      	    v0(b, 1, k-(ndim>2), j, i) * v0_vel(b, 2, k-(ndim>2), j, i);
    	  Real Fyc_Bz = v0(b, 2, k, j, i) * v0_vel(b, 1, k, j, i) - 
	    	  	v0(b, 1, k, j, i) * v0_vel(b, 2, k, j, i);
	  Real szf;
	  if (ndim<=2) { // No reconstruction so can just use CC values
	    szf = v0_vel(b, 2, k, j, i)<0. ? -1 : (v0_vel(b, 2, k, j, i)==0. ? 0. : 1.);
	  } else {
    	    szf = v0_rho.flux(b, X3DIR, 0, k-1, j, i) < 0 ? -1. : 
		 (v0_rho.flux(b, X3DIR, 0, k-1, j, i) == 0 ? 0. : 1.);
	  }
    	  Real dExdy_N = (1.+szf) * (v0.flux(b, X2DIR, n_cc, k-(ndim>2), j, i) - Fyc_Bz_km1) / dx[1]
                       + (1.-szf) * (v0.flux(b, X2DIR, n_cc, k         , j, i) - Fyc_Bz    ) / dx[1];
	  
	  Fyc_Bz_km1 = v0(b, 2, k-(ndim>2), j-1, i) * v0_vel(b, 1, k-(ndim>2), j-1, i) -
                       v0(b, 1, k-(ndim>2), j-1, i) * v0_vel(b, 2, k-(ndim>2), j-1, i);
          Fyc_Bz = v0(b, 2, k, j-1, i) * v0_vel(b, 1, k, j-1, i) -
                   v0(b, 1, k, j-1, i) * v0_vel(b, 2, k, j-1, i);
   	  Real dExdy_S = -(1.+szf) * (v0.flux(b, X2DIR, n_cc, k-(ndim>2), j, i) - Fyc_Bz_km1) / dx[1]
                       + -(1.-szf) * (v0.flux(b, X2DIR, n_cc, k         , j, i) - Fyc_Bz    ) / dx[1];
	  Real upw_corr_E1 = 0.125*dx[1]*(dExdy_S - dExdy_N) *2.;

    	  if (ndim>2) {
    	    Real Fzc_By_jm1 = v0(b, 1, k, j-1, i) * v0_vel(b, 2, k, j-1, i) 
		  	    - v0(b, 2, k, j-1, i) * v0_vel(b, 1, k, j-1, i);
    	    Real Fzc_By = v0(b, 1, k, j, i) * v0_vel(b, 2, k, j, i) - 
		          v0(b, 2, k, j, i) * v0_vel(b, 1, k, j, i);
    	    Real syf = v0_rho.flux(b, X2DIR, 0, k, j-1, i) < 0. ? -1. : 
		      (v0_rho.flux(b, X2DIR, 0, k, j-1, i) == 0. ? 0. : 1.);
    	    Real dExdz_E = -(1.+syf) * (v0.flux(b, X3DIR, n_cc-1, k, j-1, i) - Fzc_By_jm1) / dx[2]
                 	 + -(1.-syf) * (v0.flux(b, X3DIR, n_cc-1, k, j,   i) - Fzc_By    ) / dx[2];

	    Fzc_By_jm1 = v0(b, 1, k-1, j-1, i) * v0_vel(b, 2, k-1, j-1, i)  
		       - v0(b, 2, k-1, j-1, i) * v0_vel(b, 1, k-1, j-1, i);      
	    Fzc_By = v0(b, 1, k-1, j, i) * v0_vel(b, 2, k-1, j, i) 
		   - v0(b, 2, k-1, j, i) * v0_vel(b, 1, k-1, j, i);
    	    Real dExdz_W = (1.+syf) * (v0.flux(b, X3DIR, n_cc-1, k, j-1, i) - Fzc_By_jm1) / dx[2]
                 	 + (1.-syf) * (v0.flux(b, X3DIR, n_cc-1, k, j,   i) - Fzc_By    ) / dx[2];

    	    upw_corr_E1 = upw_corr_E1/2. + 0.125*dx[2]*(dExdz_W - dExdz_E);
	  }
    	  v0_fcc.flux(b, TE::E1, 0, k, j, i) += upw_corr_E1;
    	}
  });

  const auto ib_2 = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jb_2 = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kb_2 = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  int ib_2s = ib_2.s - 2, ib_2e = ib_2.e + 2;
  int jb_2s = jb_2.s, jb_2e = jb_2.e;
  if (ndim>1) {jb_2s -= 2; jb_2e += 2;}
  int kb_2s = kb_2.s, kb_2e = kb_2.e;
  if (ndim>2) {kb_2s -= 2; kb_2e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_y", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_2s, kb_2e, jb_2s, jb_2e, ib_2s, ib_2e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
	geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();

        const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) =
            0.25*(v0.flux(b, X1DIR, n_cc, k-1, j, i) +
                        v0.flux(b, X1DIR, n_cc, k, j, i) -
                        v0.flux(b, X3DIR, n_cc-2, k, j, i-1) -
                        v0.flux(b, X3DIR, n_cc-2, k, j, i));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) = v0.flux(b, X1DIR, n_cc, k, j, i);
        } else {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) = v0.flux(b, X1DIR, n_cc, k, j, i);
        }

	// Compute upwind correction for Ey (E2)
    	Real Fxc_Bz_km1 = v0(b, 2, k-(ndim>2), j, i) * v0_vel(b, 0, k-(ndim>2), j, i) -
                    	  v0(b, 0, k-(ndim>2), j, i) * v0_vel(b, 2, k-(ndim>2), j, i);
    	Real Fxc_Bz = v0(b, 2, k, j, i) * v0_vel(b, 0, k, j, i) -
		      v0(b, 0, k, j, i) * v0_vel(b, 2, k, j, i);
	Real szf;
	if (ndim<=2) {
	  szf = v0_vel(b, 2, k, j, i)<0. ? -1. : (v0_vel(b, 2, k, j, i)==0. ? 0. : 1.);
	} else {
    	  szf = v0_rho.flux(b, X3DIR, 0, k-1, j, i) < 0 ? -1. : 
	       (v0_rho.flux(b, X3DIR, 0, k-1, j, i) == 0 ? 0. : 1.);
	}
    	Real dEydx_E = -(1.+szf) * (v0.flux(b, X1DIR, n_cc, k-(ndim>2), j, i) - Fxc_Bz_km1) / dx[0]
                     + -(1.-szf) * (v0.flux(b, X1DIR, n_cc, k         , j, i) - Fxc_Bz    ) / dx[0];

	Fxc_Bz_km1 = v0(b, 2, k-(ndim>2), j, i-1) * v0_vel(b, 0, k-(ndim>2), j, i-1)
                   - v0(b, 0, k-(ndim>2), j, i-1) * v0_vel(b, 2, k-(ndim>2), j, i-1);
        Fxc_Bz = v0(b, 2, k, j, i-1) * v0_vel(b, 0, k, j, i-1)
               - v0(b, 0, k, j, i-1) * v0_vel(b, 2, k, j, i-1);
    	Real dEydx_W = (1.+szf) * (v0.flux(b, X1DIR, n_cc, k-(ndim>2), j, i) - Fxc_Bz_km1) / dx[0]
                     + (1.-szf) * (v0.flux(b, X1DIR, n_cc, k         , j, i) - Fxc_Bz    ) / dx[0];
	Real upw_corr_E2 = 0.125*dx[0]*(dEydx_W - dEydx_E) *2.;

    	if (ndim>2) {
	  Real Fzc_Bx_im1 = v0(b, 0, k, j, i-1) * v0_vel(b, 2, k, j, i-1) 
		  	  - v0(b, 2, k, j, i-1) * v0_vel(b, 0, k, j, i-1);
      	  Real Fzc_Bx = v0(b, 0, k, j, i) * v0_vel(b, 2, k, j, i) 
		      - v0(b, 2, k, j, i) * v0_vel(b, 0, k, j, i);
    	  Real sxf = v0_rho.flux(b, X1DIR, 0, k, j, i-1) < 0. ? -1. : 
		    (v0_rho.flux(b, X1DIR, 0, k, j, i-1) == 0. ? 0. : 1.);
    	  Real dEydz_N = (1.+sxf) * (v0.flux(b, X3DIR, n_cc-2, k, j, i-1) - Fzc_Bx_im1) / dx[2]
                       + (1.-sxf) * (v0.flux(b, X3DIR, n_cc-2, k, j, i  ) - Fzc_Bx    ) / dx[2];
	
	  Fzc_Bx_im1 = v0(b, 0, k-1, j, i-1) * v0_vel(b, 2, k-1, j, i-1) 
		     - v0(b, 2, k-1, j, i-1) * v0_vel(b, 0, k-1, j, i-1);
          Fzc_Bx = v0(b, 0, k-1, j, i) * v0_vel(b, 2, k-1, j, i)
                 - v0(b, 2, k-1, j, i) * v0_vel(b, 0, k-1, j, i);
    	  Real dEydz_S = -(1.+sxf) * (v0.flux(b, X3DIR, n_cc-2, k, j, i-1) - Fzc_Bx_im1) / dx[2]
                       + -(1.-sxf) * (v0.flux(b, X3DIR, n_cc-2, k, j, i  ) - Fzc_Bx    ) / dx[2];
	  upw_corr_E2  = upw_corr_E2/2. + 0.125*dx[2]*(dEydz_S - dEydz_N);
	}
    	v0_fcc.flux(b, TE::E2, 0, k, j, i) += upw_corr_E2;
  });

  const auto ib_3 = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jb_3 = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kb_3 = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  int ib_3s = ib_3.s - 2, ib_3e = ib_3.e + 2;
  int jb_3s = jb_3.s, jb_3e = jb_3.e;
  if (ndim>1) {jb_3s -= 2; jb_3e += 2;}
  int kb_3s = kb_3.s, kb_3e = kb_3.e;
  if (ndim>2) {kb_3s -= 2; kb_3e += 2;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_z", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_3s, kb_3e, jb_3s, jb_3e, ib_3s, ib_3e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
	geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
	const auto &dx = coords.GetCellWidths();

        const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) =
            0.25*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1) +
                        v0.flux(b, X2DIR, n_cc-2, k, j, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j-1, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j, i));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) =
            0.25*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1) +
                        v0.flux(b, X2DIR, n_cc-2, k, j, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j-1, i) -
                        v0.flux(b, X1DIR, n_cc-1, k, j, i));
        } else {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) = - v0.flux(b, X1DIR, n_cc-1, k, j, i);
        }
	// Compute upwind correction
	Real Fxc_By_jm1 = v0(b, 1, k, j-(ndim>1), i) * v0_vel(b, 0, k, j-(ndim>1), i) - 
			  v0(b, 0, k, j-(ndim>1), i) * v0_vel(b, 1, k, j-(ndim>1), i);
        Real Fxc_By = v0(b, 1, k, j, i) * v0_vel(b, 0, k, j, i) - 
		      v0(b, 0, k, j, i) * v0_vel(b, 1, k, j, i);
        Real syf; 
	if (ndim<=1) {
	  syf = v0_vel(b, 1, k, j, i)<0. ? -1. : (v0_vel(b, 1, k, j, i)==0. ? 0. : 1.);
	} else {
	  syf = v0_rho.flux(b, X2DIR, 0, k, j-1, i)<0. ? -1. : 
	       (v0_rho.flux(b, X2DIR, 0, k, j-1, i)==0 ? 0. : 1.);
	}
        Real dEzdx_E = (1.+syf)*(v0.flux(b, X1DIR, n_cc-1, k, j-(ndim>1), i)-Fxc_By_jm1)/dx[0]
                     + (1.-syf)*(v0.flux(b, X1DIR, n_cc-1, k, j         , i)-Fxc_By    )/dx[0];

        Fxc_By_jm1 = v0(b, 1, k, j-(ndim>1), i-1) * v0_vel(b, 0, k, j-(ndim>1), i-1) - 
		     v0(b, 0, k, j-(ndim>1), i-1) * v0_vel(b, 1, k, j-(ndim>1), i-1);
        Fxc_By = v0(b, 1, k, j, i-1) * v0_vel(b, 0, k, j, i-1) - 
		 v0(b, 0, k, j, i-1) * v0_vel(b, 1, k, j, i-1);
        Real dEzdx_W = - (1.+syf)*(v0.flux(b, X1DIR, n_cc-1, k, j-(ndim>1), i)-Fxc_By_jm1)/dx[0]
                       - (1.-syf)*(v0.flux(b, X1DIR, n_cc-1, k, j, i)-Fxc_By)/dx[0];
	Real upw_corr = 0.125*dx[0]*(dEzdx_W-dEzdx_E) *2.;

	if (ndim>1) {
	  Real Fyc_Bx_im1 = v0(b, 0, k, j, i-1) * v0_vel(b, 1, k, j, i-1) - 
		  	    v0(b, 1, k, j, i-1) * v0_vel(b, 0, k, j, i-1);
	  Real Fyc_Bx = v0(b, 0, k, j, i) * v0_vel(b, 1, k, j, i) - 
		        v0(b, 1, k, j, i) * v0_vel(b, 0, k, j, i);
	  Real sxf = v0_rho.flux(b, X1DIR, 0, k, j, i-1)<0 ? -1. : 
		    (v0_rho.flux(b, X1DIR, 0, k, j, i-1)==0 ? 0. : 1.);
	  Real dEzdy_N = - (1.+sxf)*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1)-Fyc_Bx_im1)/dx[1] 
		       + - (1.-sxf)*(v0.flux(b, X2DIR, n_cc-2, k, j, i)-Fyc_Bx)/dx[1];
	  Fyc_Bx_im1 = v0(b, 0, k, j-1, i-1) * v0_vel(b, 1, k, j-1, i-1) - 
		       v0(b, 1, k, j-1, i-1) * v0_vel(b, 0, k, j-1, i-1);
	  Fyc_Bx = v0(b, 0, k, j-1, i) * v0_vel(b, 1, k, j-1, i) - 
		   v0(b, 1, k, j-1, i) * v0_vel(b, 0, k, j-1, i);
	  Real dEzdy_S = (1.+sxf)*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1)-Fyc_Bx_im1)/dx[1]
                       + (1.-sxf)*(v0.flux(b, X2DIR, n_cc-2, k, j, i)-Fyc_Bx)/dx[1];
	  upw_corr = upw_corr/2. + 0.125*dx[1]*(dEzdy_S-dEzdy_N);
	}
	v0_fcc.flux(b, TE::E3, 0, k, j, i) += upw_corr; 
  });

  return TaskStatus::complete;
}


//==============================================================================================
//! \fn  TaskStatus ArtemisUtils::CalculateEMF_v2
//! \brief: Variant of CT scheme with upwinding at magnetosonic shock
template <Coordinates GEOM>
TaskStatus CalculateEMF_v2(MeshData<Real> *u0) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  const int ndim = pm->ndim;
  auto &gas_pkg = pm->packages.Get("gas");
  const Real gamma = gas_pkg->template Param<Real>("adiabatic_index");

  // YH: retrieve flux
  std::vector<MetadataFlag> flags2({Metadata::Cell, Metadata::Conserved, Metadata::WithFluxes});
  static auto desc2 = MakePackDescriptor<any>(u0, flags2, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc2.GetPack(u0);

  std::vector<MetadataFlag> flags3({Metadata::Face, Metadata::Conserved, Metadata::WithFluxes});
  static auto desc3 = MakePackDescriptor<any>(u0, flags3, {parthenon::PDOpt::WithFluxes});
  const auto v0_fcc = desc3.GetPack(u0);

  std::vector<MetadataFlag> flagsm1({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Density")});
  static auto descm1 = MakePackDescriptor<any>(u0, flagsm1);
  const auto v0_rho = descm1.GetPack(u0);
  std::vector<MetadataFlag> flagsP({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Pressure")});
  static auto descP = MakePackDescriptor<any>(u0, flagsP);
  const auto v0_P = descP.GetPack(u0);
  std::vector<MetadataFlag> flags0p({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Velfield")});
  static auto desc0p = MakePackDescriptor<any>(u0, flags0p);
  const auto v0_vel = desc0p.GetPack(u0);
  std::vector<MetadataFlag> flagsB({Metadata::Cell, Metadata::Derived,
		  Metadata::GetUserFlag("Bfield")});
  static auto descB = MakePackDescriptor<any>(u0, flagsB);
  const auto v0_B = descB.GetPack(u0);

  const Real beta = 0.5;
  const Real delta = 0.1;

  const auto ib_1 = u0->GetBoundsI(IndexDomain::interior, TE::E1);
  const auto jb_1 = u0->GetBoundsJ(IndexDomain::interior, TE::E1);
  const auto kb_1 = u0->GetBoundsK(IndexDomain::interior, TE::E1);
  int ib_1s = ib_1.s - 1, ib_1e = ib_1.e + 1;
  int jb_1s = jb_1.s, jb_1e = jb_1.e;
  if (ndim>1) {jb_1s -= 1; jb_1e += 1;}
  int kb_1s = kb_1.s, kb_1e = kb_1.e;
  if (ndim>2) {kb_1s -= 1; kb_1e += 1;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_1s, kb_1e, jb_1s, jb_1e, ib_1s, ib_1e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

	const Real DPz = INTERP::CC_2_E1_grad3(v0_P, b, k, j, i, ndim);
	const Real DPy = INTERP::CC_2_E1_grad2(v0_P, b, k, j, i, ndim);
	const Real divV = INTERP::CCvel_2_E1_divV(v0_vel, b, k, j, i, ndim);
	const Real Bijk = SQR(v0_B(b, TE::CC, 0, k, j, i)) + SQR(v0_B(b, TE::CC, 1, k, j, i)) 
			+ SQR(v0_B(b, TE::CC, 2, k, j, i));
	const Real Cijk = std::sqrt((gamma*v0_P(b, k, j, i)/v0_rho(b, k, j, i)) + 
			  	    (Bijk/v0_rho(b, k, j, i)));
	const Real Bijkm1 = SQR(v0_B(b, TE::CC, 0, k-(ndim>2), j, i)) + SQR(v0_B(b, TE::CC, 1, k-(ndim>2), j, i))
			  + SQR(v0_B(b, TE::CC, 2, k-(ndim>2), j, i));
        const Real Cijkm1 = std::sqrt((gamma*v0_P(b, k-(ndim>2), j, i)/v0_rho(b, k-(ndim>2), j, i)) +
                                     (Bijk/v0_rho(b, k-(ndim>2), j, i)));
	const Real Bijm1k = SQR(v0_B(b, TE::CC, 0, k, j-(ndim>1), i)) + SQR(v0_B(b, TE::CC, 1, k, j-(ndim>1), i))
			  + SQR(v0_B(b, TE::CC, 2, k, j-(ndim>1), i));
        const Real Cijm1k = std::sqrt((gamma*v0_P(b, k, j-(ndim>1), i)/v0_rho(b, k, j-(ndim>1), i)) +
                                      (Bijk/v0_rho(b, k, j-(ndim>1), i)));
	const Real Bijm1km1 = SQR(v0_B(b, TE::CC, 0, k-(ndim>2), j-(ndim>1), i)) + SQR(v0_B(b, TE::CC, 1, k-(ndim>2), j-(ndim>1), i))
			    + SQR(v0_B(b, TE::CC, 2, k-(ndim>2), j-(ndim>1), i));
        const Real Cijm1km1 = std::sqrt((gamma*v0_P(b, k-(ndim>2), j-(ndim>1), i)/v0_rho(b, k-(ndim>2), j-(ndim>1), i)) +
                                       (Bijk/v0_rho(b, k-(ndim>2), j-(ndim>1), i)));

	const Real minP = INTERP::CC_2_E1_min(v0_P, b, k, j, i, ndim);
	const bool SW1 = ((abs(DPz) + abs(DPy)) > (beta*minP));
	const Real minC1 = std::min(Cijm1km1, Cijm1k);
	const Real minC2 = std::min(Cijkm1, Cijk);
	const bool SW2 = ((-delta*std::min(minC1, minC2)) > divV);
	Real phi = 0.5;
	if (SW1 and SW2) phi = abs(DPy) / (abs(DPz) + abs(DPy));

	const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) =
            0.5*((1.-phi)*(v0.flux(b, X3DIR, n_cc-1, k, j-1, i) +
                        v0.flux(b, X3DIR, n_cc-1, k, j, i)) +
                   phi*(- v0.flux(b, X2DIR, n_cc, k-1, j, i) 
                        - v0.flux(b, X2DIR, n_cc, k, j, i)));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) = - v0.flux(b, X2DIR, n_cc, k, j, i);
        } else {
          v0_fcc.flux(b, TE::E1, 0, k, j, i) = 0.;
        }
  });

  const auto ib_2 = u0->GetBoundsI(IndexDomain::interior, TE::E2);
  const auto jb_2 = u0->GetBoundsJ(IndexDomain::interior, TE::E2);
  const auto kb_2 = u0->GetBoundsK(IndexDomain::interior, TE::E2);
  int ib_2s = ib_2.s - 1, ib_2e = ib_2.e + 1;
  int jb_2s = jb_2.s, jb_2e = jb_2.e;
  if (ndim>1) {jb_2s -= 1; jb_2e += 1;}
  int kb_2s = kb_2.s, kb_2e = kb_2.e;
  if (ndim>2) {kb_2s -= 1; kb_2e += 1;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_y", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_2s, kb_2e, jb_2s, jb_2e, ib_2s, ib_2e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

	const Real DPx = INTERP::CC_2_E2_grad1(v0_P, b, k, j, i, ndim);
	const Real DPz = INTERP::CC_2_E2_grad3(v0_P, b, k, j, i, ndim);
	const Real divV = INTERP::CCvel_2_E2_divV(v0_vel, b, k, j, i, ndim);
	const Real Bijk = SQR(v0_B(b, TE::CC, 0, k, j, i)) + SQR(v0_B(b, TE::CC, 1, k, j, i)) 
			+ SQR(v0_B(b, TE::CC, 2, k, j, i));
	const Real Cijk = std::sqrt((gamma*v0_P(b, k, j, i)/v0_rho(b, k, j, i)) + 
			  	    (Bijk/v0_rho(b, k, j, i)));
	const Real Bim1jk = SQR(v0_B(b, TE::CC, 0, k, j, i-1)) + SQR(v0_B(b, TE::CC, 1, k, j, i-1))
			  + SQR(v0_B(b, TE::CC, 2, k, j, i-1));
        const Real Cim1jk = std::sqrt((gamma*v0_P(b, k, j, i-1)/v0_rho(b, k, j, i-1)) +
                                      (Bijk/v0_rho(b, k, j, i-1)));
	const Real Bijkm1 = SQR(v0_B(b, TE::CC, 0, k-(ndim>2), j, i)) + SQR(v0_B(b, TE::CC, 1, k-(ndim>2), j, i))
			  + SQR(v0_B(b, TE::CC, 2, k-(ndim>2), j, i));
        const Real Cijkm1 = std::sqrt((gamma*v0_P(b, k-(ndim>2), j, i)/v0_rho(b, k-(ndim>2), j, i)) +
                                      (Bijk/v0_rho(b, k-(ndim>2), j, i)));
	const Real Bim1jkm1 = SQR(v0_B(b, TE::CC, 0, k-(ndim>2), j, i-1)) + SQR(v0_B(b, TE::CC, 1, k-(ndim>2), j, i-1))
			    + SQR(v0_B(b, TE::CC, 2, k-(ndim>2), j, i-1));
        const Real Cim1jkm1 = std::sqrt((gamma*v0_P(b, k-(ndim>2), j, i-1)/v0_rho(b, k-(ndim>2), j, i-1)) +
                                         (Bijk/v0_rho(b, k-(ndim>2), j, i-1)));

	const Real minP = INTERP::CC_2_E2_min(v0_P, b, k, j, i, ndim);
	const bool SW1 = ((abs(DPx) + abs(DPz)) > (beta*minP));
	const Real minC1 = std::min(Cim1jkm1, Cijkm1);
	const Real minC2 = std::min(Cim1jk, Cijk);
	const bool SW2 = ((-delta*std::min(minC1, minC2)) > divV);
	Real phi = 0.5;
	if (SW1 and SW2) phi = abs(DPx) / (abs(DPx) + abs(DPz));

	const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) =
            0.5*(phi*(v0.flux(b, X1DIR, n_cc, k-1, j, i) +
                        v0.flux(b, X1DIR, n_cc, k, j, i)) + 
              (1.-phi)*(- v0.flux(b, X3DIR, n_cc-2, k, j, i-1) 
                        - v0.flux(b, X3DIR, n_cc-2, k, j, i)));
        } else if (ndim > 1) {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) = v0.flux(b, X1DIR, n_cc, k, j, i);
        } else {
          v0_fcc.flux(b, TE::E2, 0, k, j, i) = v0.flux(b, X1DIR, n_cc, k, j, i);
        }
  });

  const auto ib_3 = u0->GetBoundsI(IndexDomain::interior, TE::E3);
  const auto jb_3 = u0->GetBoundsJ(IndexDomain::interior, TE::E3);
  const auto kb_3 = u0->GetBoundsK(IndexDomain::interior, TE::E3);
  int ib_3s = ib_3.s - 1, ib_3e = ib_3.e + 1;
  int jb_3s = jb_3.s, jb_3e = jb_3.e;
  if (ndim>1) {jb_3s -= 1; jb_3e += 1;}
  int kb_3s = kb_3.s, kb_3e = kb_3.e;
  if (ndim>2) {kb_3s -= 1; kb_3e += 1;}
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateEMF_z", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb_3s, kb_3e, jb_3s, jb_3e, ib_3s, ib_3e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;

	const Real DPx = INTERP::CC_2_E3_grad1(v0_P, b, k, j, i, ndim);
	const Real DPy = INTERP::CC_2_E3_grad2(v0_P, b, k, j, i, ndim);
	const Real divV = INTERP::CCvel_2_E3_divV(v0_vel, b, k, j, i, ndim);
	const Real Bijk = SQR(v0_B(b, TE::CC, 0, k, j, i)) + SQR(v0_B(b, TE::CC, 1, k, j, i)) 
			+ SQR(v0_B(b, TE::CC, 2, k, j, i));
	const Real Cijk = std::sqrt((gamma*v0_P(b, k, j, i)/v0_rho(b, k, j, i)) + 
			  	    (Bijk/v0_rho(b, k, j, i)));
	const Real Bim1jk = SQR(v0_B(b, TE::CC, 0, k, j, i-1)) + SQR(v0_B(b, TE::CC, 1, k, j, i-1))
			  + SQR(v0_B(b, TE::CC, 2, k, j, i-1));
        const Real Cim1jk = std::sqrt((gamma*v0_P(b, k, j, i-1)/v0_rho(b, k, j, i-1)) +
                                      (Bijk/v0_rho(b, k, j, i-1)));
	const Real Bijm1k = SQR(v0_B(b, TE::CC, 0, k, j-(ndim>1), i)) + SQR(v0_B(b, TE::CC, 1, k, j-(ndim>1), i))
			  + SQR(v0_B(b, TE::CC, 2, k, j-(ndim>1), i));
        const Real Cijm1k = std::sqrt((gamma*v0_P(b, k, j-(ndim>1), i)/v0_rho(b, k, j-(ndim>1), i)) +
                                      (Bijk/v0_rho(b, k, j-(ndim>1), i)));
	const Real Bim1jm1k = SQR(v0_B(b, TE::CC, 0, k, j-(ndim>1), i-1)) + SQR(v0_B(b, TE::CC, 1, k, j-(ndim>1), i-1))
			    + SQR(v0_B(b, TE::CC, 2, k, j-(ndim>1), i-1));
        const Real Cim1jm1k = std::sqrt((gamma*v0_P(b, k, j-(ndim>1), i-1)/v0_rho(b, k, j-(ndim>1), i-1)) +
                                         (Bijk/v0_rho(b, k, j-(ndim>1), i-1)));

	const Real minP = INTERP::CC_2_E3_min(v0_P, b, k, j, i, ndim);	
	const bool SW1 = ((abs(DPx) + abs(DPy)) > (beta*minP));
	const Real minC1 = std::min(Cim1jm1k, Cijm1k);
	const Real minC2 = std::min(Cim1jk, Cijk);
	const bool SW2 = ((-delta*std::min(minC1, minC2)) > divV);
	Real phi = 0.5;
	if (SW1 and SW2) phi = abs(DPx) / (abs(DPx) + abs(DPy));

	const int n_cc = 2;
        if (ndim > 2) {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) =
            0.5*((1.-phi)*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1) +
                        v0.flux(b, X2DIR, n_cc-2, k, j, i)) + 
                   phi*(- v0.flux(b, X1DIR, n_cc-1, k, j-1, i) 
                        - v0.flux(b, X1DIR, n_cc-1, k, j, i)));
        } else if (ndim > 1) {
	  v0_fcc.flux(b, TE::E3, 0, k, j, i) =
            0.5*((1.-phi)*(v0.flux(b, X2DIR, n_cc-2, k, j, i-1) +
                        v0.flux(b, X2DIR, n_cc-2, k, j, i)) + 
                   phi*(- v0.flux(b, X1DIR, n_cc-1, k, j-1, i)     
                        - v0.flux(b, X1DIR, n_cc-1, k, j, i)));
        } else {
          v0_fcc.flux(b, TE::E3, 0, k, j, i) = - v0.flux(b, X1DIR, n_cc-1, k, j, i);
        }
  });
  return TaskStatus::complete;
}




//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::ApplyUpdateFCC
//! \brief
template <Coordinates GEOM>
TaskStatus ApplyUpdateFCC(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();

  // Extract integrator weights
  const Real gam0 = integrator->gam0[stage - 1];
  const Real gam1 = integrator->gam1[stage - 1];
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;

  // Packing and indexing
  std::vector<MetadataFlag> flags({Metadata::Face, Metadata::Conserved, Metadata::WithFluxes,
		  Metadata::GetUserFlag("Bfield")});
  static auto desc = MakePackDescriptor<any>(u0, flags, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior, TE::F1);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior, TE::F1);
  const auto kb = u0->GetBoundsK(IndexDomain::interior, TE::F1);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;

  if (ndim>1) { // YH: as j=j+1 for 1D
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdateFCC_x", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e+(ndim>2), jb.s, jb.e+(ndim>1), ib.s, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
        geometry::Coords<GEOM> coords_jp1(v0.GetCoordinates(b), k, j+1, i);

        const auto ax1 = coords.GetFaceAreaX1();
        const auto e3len = coords.GetEdgeLengthX3();
        const auto e3len_jp1 = coords_jp1.GetEdgeLengthX3();

        v0(b, TE::F1, 0, k, j, i) = gam0 * v0(b, TE::F1, 0, k, j, i) + gam1 * v1(b, TE::F1, 0, k, j, i)
           - (beta_dt/ax1[0]) *
             (e3len_jp1*v0.flux(b, TE::E3, 0, k, j+1, i) -
              e3len*v0.flux(b, TE::E3, 0, k, j, i));
        if (ndim>2) {
          geometry::Coords<GEOM> coords_kp1(v0.GetCoordinates(b), k+1, j, i);
          const auto e2len = coords.GetEdgeLengthX2();
          const auto e2len_kp1 = coords_kp1.GetEdgeLengthX2();
          v0(b, TE::F1, 0, k, j, i) += - (beta_dt/ax1[0]) *
             (e2len*v0.flux(b, TE::E2, 0, k, j, i) -
              e2len_kp1*v0.flux(b, TE::E2, 0, k+1, j, i));
        }
  });
 }

  const auto iby = u0->GetBoundsI(IndexDomain::interior, TE::F2);
  const auto jby = u0->GetBoundsJ(IndexDomain::interior, TE::F2);
  const auto kby = u0->GetBoundsK(IndexDomain::interior, TE::F2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdateFCC_y", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kby.s, kby.e+(ndim>2), jby.s, jby.e+(ndim>1), iby.s, iby.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
        geometry::Coords<GEOM> coords_ip1(v0.GetCoordinates(b), k, j, i+1);

        const auto ax2 = coords.GetFaceAreaX2();
        const auto e3len = coords.GetEdgeLengthX3();
        const auto e3len_ip1 = coords_ip1.GetEdgeLengthX3();

        v0(b, TE::F2, 0, k, j, i) = gam0 * v0(b, TE::F2, 0, k, j, i) + gam1 * v1(b, TE::F2, 0, k, j, i)
                - (beta_dt/ax2[0]) *
                  (e3len*v0.flux(b, TE::E3, 0, k, j, i) -
                   e3len_ip1*v0.flux(b, TE::E3, 0, k, j, i+1));
        if (ndim > 2) {
          geometry::Coords<GEOM> coords_kp1(v0.GetCoordinates(b), k+1, j, i);
          const auto e1len = coords.GetEdgeLengthX1();
          const auto e1len_kp1 = coords_kp1.GetEdgeLengthX1();
          v0(b, TE::F2, 0, k, j, i) += - (beta_dt/ax2[0]) *
                  (e1len_kp1*v0.flux(b, TE::E1, 0, k+1, j, i) -
                   e1len*v0.flux(b, TE::E1, 0, k, j, i));
        }
  });

  const auto ibz = u0->GetBoundsI(IndexDomain::interior, TE::F3);
  const auto jbz = u0->GetBoundsJ(IndexDomain::interior, TE::F3);
  const auto kbz = u0->GetBoundsK(IndexDomain::interior, TE::F3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdateFCC_z", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kbz.s, kbz.e+(ndim>2), jbz.s, jbz.e+(ndim>1), ibz.s, ibz.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);
        geometry::Coords<GEOM> coords_ip1(v0.GetCoordinates(b), k, j, i+1);

        const auto ax3 = coords.GetFaceAreaX3();
        const auto e2len = coords.GetEdgeLengthX2();
        const auto e2len_ip1 = coords_ip1.GetEdgeLengthX2();

        v0(b, TE::F3, 0, k, j, i) = gam0 * v0(b, TE::F3, 0, k, j, i) + gam1 * v1(b, TE::F3, 0, k, j, i)
                - (beta_dt/ax3[0]) *
                  (e2len_ip1*v0.flux(b, TE::E2, 0, k, j, i+1) -
                   e2len*v0.flux(b, TE::E2, 0, k, j, i));
        if (ndim>1) {
          geometry::Coords<GEOM> coords_jp1(v0.GetCoordinates(b), k, j+1, i);
          const auto e1len = coords.GetEdgeLengthX1();
          const auto e1len_jp1 = coords_jp1.GetEdgeLengthX1();
          v0(b, TE::F3, 0, k, j, i) += - (beta_dt/ax3[0]) *
                  (e1len*v0.flux(b, TE::E1, 0, k, j, i) -
                   e1len_jp1*v0.flux(b, TE::E1, 0, k, j+1, i));
        }
      });

  return TaskStatus::complete;
}



//----------------------------------------------------------------------------------------
//! \fn  TaskStatus ArtemisUtils::ApplyUpdate
//! \brief
template <Coordinates GEOM>
TaskStatus ApplyUpdateCC(MeshData<Real> *u0, MeshData<Real> *u1, const int stage,
                       parthenon::LowStorageIntegrator *integrator) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();

  // Extract integrator weights
  const Real gam0 = integrator->gam0[stage - 1];
  const Real gam1 = integrator->gam1[stage - 1];
  const Real beta_dt = integrator->beta[stage - 1] * integrator->dt;

  // Packing and indexing
  std::vector<MetadataFlag> flags({Metadata::Cell, Metadata::Conserved, Metadata::WithFluxes});
  static auto desc = MakePackDescriptor<any>(u0, flags, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);

        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = (multi_d) ? coords.GetFaceAreaX2() : NewArray<Real, 2>(0.0);
        const auto ax3 = (three_d) ? coords.GetFaceAreaX3() : NewArray<Real, 2>(0.0);

        const Real vol = coords.Volume();
        for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
          // compute flux divergence
          Real divf = (ax1[0] * v0.flux(b, X1DIR, n, k, j, i) -
                       ax1[1] * v0.flux(b, X1DIR, n, k, j, i + 1));
          if (multi_d)
            divf += (ax2[0] * v0.flux(b, X2DIR, n, k, j, i) -
                     ax2[1] * v0.flux(b, X2DIR, n, k, j + 1, i));
          if (three_d)
            divf += (ax3[0] * v0.flux(b, X3DIR, n, k, j, i) -
                     ax3[1] * v0.flux(b, X3DIR, n, k + 1, j, i));

          // Apply update for cc values
          v0(b, n, k, j, i) =
              gam0 * v0(b, n, k, j, i) + gam1 * v1(b, n, k, j, i) + divf * beta_dt / vol;
        }
      });

  return TaskStatus::complete;
}


} // namespace mhd

#endif // MHD_MHD_HPP_
