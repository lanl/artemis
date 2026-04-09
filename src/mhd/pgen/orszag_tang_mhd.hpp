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
#ifndef PGEN_ORSZAG_TANG_MHD_HPP_
#define PGEN_ORSZAG_TANG_MHD_HPP_
//! \file orszag_tang_mhd.hpp
//! \brief
//!

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace orszag_tang_mhd {

struct ORSZAG_TANG_MHD_Params {
};

//----------------------------------------------------------------------------------------
//! \fn void Init_ORSZAG_TANG_MHD_Params
//! \brief Extracts orszag_tang_mhd parameters from ParameterInput.
inline void Init_ORSZAG_TANG_MHD_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("orszag_tang_mhd_params"))) {
    ORSZAG_TANG_MHD_Params orszag_tang_mhd_params;
    params.Add("orszag_tang_mhd_params", orszag_tang_mhd_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential
KOKKOS_INLINE_FUNCTION Real A3(const Real x1, const Real x2, const Real Binit) {
  return (Binit/(4.0*M_PI))*(std::cos(4.0*M_PI*x1) + 2.0*std::cos(2.0*M_PI*x2));
}


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ORSZAG_TANG_MHD()
//! \brief Sets initial conditions for orszag_tang_mhd problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The orszag_tang mhd problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The orszag_tang mhd problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",1.66666666666667);
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

  // ORSZAG_TANG_MHD parameters
  auto shkp = artemis_pkg->Param<ORSZAG_TANG_MHD_Params>("orszag_tang_mhd_params");
  const Real Binit = 1./sqrt(4.*M_PI);

  // Setup orszag_tang_mhd state
  pmb->par_for(
      "orszag_tang_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const Real rho = 25./(36.*M_PI);
        const Real P = 5./(12.*M_PI);
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = -sin(2.*M_PI*xi[1]);
        v(0, gas::prim::velocity(1), k, j, i) = sin(2.*M_PI*xi[0]);
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
      });

  // YH: Use vector potential===========================================================
  // 1) Initialize components of vector potential
  const int nghost = pin->GetInteger("parthenon/mesh", "nghost");
  const int nx1 = pin->GetInteger("parthenon/mesh", "nx1") + 2*nghost;
  const int nx2 = pin->GetInteger("parthenon/mesh", "nx2") + 2*nghost;
  const int nx3 = 2;
  ParArrayND<Real> a3("a3", nx3, nx2, nx1);
  // Initialize components of vector potential
  pmb->par_for(
      "pgen_linwave1", kb.s, kb.e, jb.s, jb.e+1, ib.s, ib.e+1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &x3e = coords.GetEdgeCenterX3();
        a3(k, j, i) = A3(x3e[0], x3e[1], Binit);
        });
  //====================================================================================

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes},{parthenon::PDOpt::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "orszag_tang_mhd", kb.s, kb.e, jb.s, jb.e+1, ib.s, ib.e+1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	//vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = -Binit * sin(2.*M_PI*xf[1]);
        const auto e2len = coords.GetEdgeLengthX2();
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = 
		(a3(k, j+1, i) - a3(k, j, i))/e2len;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "orszag_tang_mhd", kb.s, kb.e, jb.s, jb.e+1, ib.s, ib.e+1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
        //vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = Binit*sin(4.*M_PI*xf[0]);
	const auto e1len = coords.GetEdgeLengthX1();
	vmag(0, TE::F2, gas::face::bfield(0), k, j, i) =
		(a3(k, j, i) - a3(k, j, i+1))/e1len;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "orszag_tang_mhd", kb.s, kb.e, jb.s, jb.e+1, ib.s, ib.e+1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = 0.;
      });

}

} // namespace orszag_tang_mhd
#endif // PGEN_ORSZAG_TANG_MHD_HPP_
