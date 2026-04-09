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
#ifndef PGEN_BLAST3D_MHD_HPP_
#define PGEN_BLAST3D_MHD_HPP_
//! \file blast3D_mhd.hpp
//! \brief
//!

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace blast3D_mhd {

struct BLAST3D_MHD_Params {
  Real rhol, vxl, vyl, pl, bxl, byl, bzl;
  Real rhor, vxr, vyr, pr, bxr, byr, bzr;
  Real rdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_BLAST3D_MHD_Params
//! \brief Extracts blast3D_mhd parameters from ParameterInput.
inline void Init_BLAST3D_MHD_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("blast3D_mhd_params"))) {
    BLAST3D_MHD_Params blast3D_mhd_params;
    blast3D_mhd_params.rhol = pin->GetOrAddReal("problem", "rhol", 1.0);
    blast3D_mhd_params.vxl = pin->GetOrAddReal("problem", "vxl", 0.0);
    blast3D_mhd_params.vyl = pin->GetOrAddReal("problem", "vyl", 0.0);
    blast3D_mhd_params.pl = pin->GetOrAddReal("problem", "pl", 1000.0);
    blast3D_mhd_params.bxl = pin->GetOrAddReal("problem", "bxl", 0.0);
    blast3D_mhd_params.byl = pin->GetOrAddReal("problem", "byl", 14.10473959);
    blast3D_mhd_params.bzl = pin->GetOrAddReal("problem", "bzl", 28.20947918);
    blast3D_mhd_params.rhor = pin->GetOrAddReal("problem", "rhor", 1.0);
    blast3D_mhd_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.0);
    blast3D_mhd_params.vyr = pin->GetOrAddReal("problem", "vyr", 0.0);
    blast3D_mhd_params.pr = pin->GetOrAddReal("problem", "pr", 0.1);
    blast3D_mhd_params.bxr = pin->GetOrAddReal("problem", "bxr", 0.0);
    blast3D_mhd_params.byr = pin->GetOrAddReal("problem", "byr", 14.10473959);
    blast3D_mhd_params.bzr = pin->GetOrAddReal("problem", "bzr", 28.20947918);
    blast3D_mhd_params.rdisc = pin->GetOrAddReal("problem", "rdisc", 0.1);
    params.Add("blast3D_mhd_params", blast3D_mhd_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BLAST3D_MHD()
//! \brief Sets initial conditions for blast3D_mhd problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The blast3D mhd problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The blast3D mhd problem does not permit dust hydrodynamics!");
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

  // BLAST3D_MHD parameters
  auto shkp = artemis_pkg->Param<BLAST3D_MHD_Params>("blast3D_mhd_params");

  // Setup blast3D_mhd state
  pmb->par_for(
      "blast3D_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
	Real radius = sqrt(SQR(xi[0]) + SQR(xi[1]) + SQR(xi[2]));
        const bool upwind = (radius <= shkp.rdisc);
        const Real rho = upwind ? shkp.rhol : shkp.rhor;
        const Real vx = upwind ? shkp.vxl : shkp.vxr;
	const Real vy = upwind ? shkp.vyl : shkp.vyr;
        const Real P = upwind ? shkp.pl : shkp.pr;
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = vy;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
	// YH: add magnetic field -> removed after fcc work
        const Real bx = upwind ? shkp.bxl : shkp.bxr;
        const Real by = upwind ? shkp.byl : shkp.byr;
        const Real bz = upwind ? shkp.bzl : shkp.bzr;
        v(0, gas::prim::Bfield(0), k, j, i) = bx;
        v(0, gas::prim::Bfield(1), k, j, i) = by;
        v(0, gas::prim::Bfield(2), k, j, i) = bz;
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes},{parthenon::PDOpt::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); // YH: why F2 here? - change to F1 instead
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "blast3D_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	Real radiusf = sqrt(SQR(xf[0]) + SQR(xf[1]) + SQR(xf[2]));
	const bool upwind = (radiusf <= shkp.rdisc);
	// YH: add magnetic field - refer to problem gen from fine_adv
	const Real bx = upwind ? shkp.bxl : shkp.bxr;
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = bx;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "blast3D_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
	Real radiusf = sqrt(SQR(xf[0]) + SQR(xf[1]) + SQR(xf[2]));
        const bool upwind = (radiusf <= shkp.rdisc);
        const Real by = upwind ? shkp.byl : shkp.byr;
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = by;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "blast3D_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
	Real radiusf = sqrt(SQR(xf[0]) + SQR(xf[1]) + SQR(xf[2]));
        const bool upwind = (radiusf <= shkp.rdisc);
        const Real bz = upwind ? shkp.bzl : shkp.bzr;
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = bz;
      });

}

} // namespace blast3D_mhd
#endif // PGEN_BLAST3D_MHD_HPP_
