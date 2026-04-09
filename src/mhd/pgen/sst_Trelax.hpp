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
#ifndef PGEN_SST_Trelax_HPP_
#define PGEN_SST_Trelax_HPP_
//! \file sst_Trelax.hpp
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

using ArtemisUtils::EOS;

namespace sst_Trelax {

struct SST_TrelaxParams {
  Real rhol, vxl, pl, pel;
  Real rhor, vxr, pr, per;
  Real xdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void InitSST_TrelaxParams
//! \brief Extracts sst_Trelax parameters from ParameterInput.
inline void InitSST_TrelaxParams(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("sst_Trelax_params"))) {
    SST_TrelaxParams sst_Trelax_params;
    sst_Trelax_params.rhol = pin->GetOrAddReal("problem", "rhol", 1.0);
    sst_Trelax_params.vxl = pin->GetOrAddReal("problem", "vxl", 0.0);
    sst_Trelax_params.pl = pin->GetOrAddReal("problem", "pl", 1.0);
    sst_Trelax_params.pel = pin->GetOrAddReal("problem", "pel", 0.5);
    sst_Trelax_params.rhor = pin->GetOrAddReal("problem", "rhor", 0.125);
    sst_Trelax_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.0);
    sst_Trelax_params.pr = pin->GetOrAddReal("problem", "pr", 0.1);
    sst_Trelax_params.per = pin->GetOrAddReal("problem", "per", 0.05);
    sst_Trelax_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.5);
    params.Add("sst_Trelax_params", sst_Trelax_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SST_Trelax()
//! \brief Sets initial conditions for sst_Trelax problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  PARTHENON_REQUIRE(do_gas, "The sst_Trelax problem requires gas hydrodynamics!");
  PARTHENON_REQUIRE(!(do_dust), "The sst_Trelax problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",1.666666666667);
  Real gm1 = gamma - 1.;
  const int ndim = ProblemDimension(pin);

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
	  gas::prim::Pe, gas::prim::Te, gas::prim::Ti,
          gas::prim::pressure,gas::prim::Bfield>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // SST_Trelax parameters
  auto shkp = artemis_pkg->Param<SST_TrelaxParams>("sst_Trelax_params");

  // Setup sst_Trelax state
  pmb->par_for(
      "sst_Trelax", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const bool upwind = (xi[0] <= shkp.xdisc);
        const Real rho = upwind ? shkp.rhol : shkp.rhor;
        const Real vx = upwind ? shkp.vxl : shkp.vxr;
        const Real P = upwind ? shkp.pl : shkp.pr;
        const Real Pe = upwind ? shkp.pel : shkp.per;
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
	v(0, gas::prim::pressure(0), k, j, i) = P;
        v(0, gas::prim::Bfield(0), k, j, i) = 0.;
        v(0, gas::prim::Bfield(1), k, j, i) = 0.;
        v(0, gas::prim::Bfield(2), k, j, i) = 0.;
        v(0, gas::prim::Pe(0), k, j, i) = Pe;

	// Need compute temperature for electron & ion energy exchange
        v(0, gas::prim::Ti(0), k, j, i) = (v(0, gas::prim::pressure(0), k, j, i) -
                        v(0, gas::prim::Pe(0), k, j, i)) / v(0, gas::prim::density(0), k, j, i);
        v(0, gas::prim::Te(0), k, j, i) = v(0, gas::prim::Pe(0), k, j, i) / (Z_ion * v(0, gas::prim::density(0), k, j, i));
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "brio_wu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        // YH: add magnetic field - refer to problem gen from fine_adv
        vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = 0.;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "brio_wu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = 0.;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "brio_wu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = 0.;
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
        vEJ(0, TE::E1, gas::edge::J(), k, j, i) = 0.;
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
        vEJ(0, TE::E2, gas::edge::J(), k, j, i) = 0.;
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
        vEJ(0, TE::E3, gas::edge::J(), k, j, i) = 0.;
      });
}

} // namespace sst_Trelax
#endif // PGEN_SST_Trelax_HPP_
