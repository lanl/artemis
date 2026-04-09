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
#ifndef PGEN_CURRENT_SHEET_MHD_HPP_
#define PGEN_CURRENT_SHEET_MHD_HPP_
//! \file current_sheet_mhd.hpp
//! \brief
//!

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace current_sheet_mhd {

struct CURRENT_SHEET_MHD_Params {
  Real rhol, vxl, vyl, pl, bxl, byl, bzl;
  Real rhor, vxr, vyr, pr, bxr, byr, bzr;
  Real xstart;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_CURRENT_SHEET_MHD_Params
//! \brief Extracts current_sheet_mhd parameters from ParameterInput.
inline void Init_CURRENT_SHEET_MHD_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("current_sheet_mhd_params"))) {
    CURRENT_SHEET_MHD_Params current_sheet_mhd_params;
    // Left and right region
    current_sheet_mhd_params.rhol = pin->GetOrAddReal("problem", "rhol", 1.0);
    current_sheet_mhd_params.vxl = pin->GetOrAddReal("problem", "vxl", 0.1);
    current_sheet_mhd_params.vyl = pin->GetOrAddReal("problem", "vyl", 0.0);
    current_sheet_mhd_params.pl = pin->GetOrAddReal("problem", "pl", 0.1);
    current_sheet_mhd_params.bxl = pin->GetOrAddReal("problem", "bxl", 0);
    current_sheet_mhd_params.byl = pin->GetOrAddReal("problem", "byl", 1.0);
    current_sheet_mhd_params.bzl = pin->GetOrAddReal("problem", "bzl", 0.0);
	
    // Middle region 
    current_sheet_mhd_params.rhor = pin->GetOrAddReal("problem", "rhor", 1.0);
    current_sheet_mhd_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.1);
    current_sheet_mhd_params.vyr = pin->GetOrAddReal("problem", "vyr", 0.0);
    current_sheet_mhd_params.pr = pin->GetOrAddReal("problem", "pr", 0.1);
    current_sheet_mhd_params.bxr = pin->GetOrAddReal("problem", "bxr", 0);
    current_sheet_mhd_params.byr = pin->GetOrAddReal("problem", "byr", -1.0);
    current_sheet_mhd_params.bzr = pin->GetOrAddReal("problem", "bzr", 0.0);
    current_sheet_mhd_params.xstart = pin->GetOrAddReal("problem", "xstart", 0.5);
    params.Add("current_sheet_mhd_params", current_sheet_mhd_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::CURRENT_SHEET_MHD()
//! \brief Sets initial conditions for current_sheet_mhd problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The current_sheet mhd problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The current_sheet mhd problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",2.0);
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

  // CURRENT_SHEET_MHD parameters
  auto shkp = artemis_pkg->Param<CURRENT_SHEET_MHD_Params>("current_sheet_mhd_params");
  const auto xend = shkp.xstart + 1.0;

  // Setup current_sheet_mhd state
  pmb->par_for(
      "current_sheet_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const bool middle = (xi[0] >= shkp.xstart and xi[0] <= xend);
        const Real rho = middle ? shkp.rhol : shkp.rhor;
        const Real vx = middle ? shkp.vxl*std::sin(2.*M_PI*xi[1]) : shkp.vxr*std::sin(2.*M_PI*xi[1]);
	const Real vy = middle ? shkp.vyl : shkp.vyr;
        const Real P = middle ? shkp.pl : shkp.pr;
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = vy;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes},{parthenon::PDOpt::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); // YH: why F2 here? - change to F1 instead
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "current_sheet_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	const bool middle = (xf[0] >= shkp.xstart and xf[0] <= xend);
	const Real bx = middle ? shkp.bxl : shkp.bxr;
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = bx;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "current_sheet_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
	const bool middle = (xf[0] >= shkp.xstart and xf[0] <= xend);
        const Real by = middle ? shkp.byl : shkp.byr;
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = by;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "current_sheet_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
	const bool middle = (xf[0] >= shkp.xstart and xf[0] <= xend);
        const Real bz = middle ? shkp.bzl : shkp.bzr;
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = bz;
      });

}

} // namespace current_sheet_mhd
#endif // PGEN_CURRENT_SHEET_MHD_HPP_
