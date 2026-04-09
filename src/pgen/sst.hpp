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
#ifndef PGEN_SST_HPP_
#define PGEN_SST_HPP_
//! \file sst.hpp
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

namespace sst {

struct SSTParams {
  Real rhol, vxl, pl;
  Real rhor, vxr, pr;
  Real xdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void InitSSTParams
//! \brief Extracts sst parameters from ParameterInput.
inline void InitSSTParams(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("sst_params"))) {
    SSTParams sst_params;
    sst_params.rhol = pin->GetOrAddReal("problem", "rhol", 1.0);
    sst_params.vxl = pin->GetOrAddReal("problem", "vxl", 0.0);
    sst_params.pl = pin->GetOrAddReal("problem", "pl", 1.0);
    sst_params.rhor = pin->GetOrAddReal("problem", "rhor", 0.125);
    sst_params.vxr = pin->GetOrAddReal("problem", "vxr", 0.0);
    sst_params.pr = pin->GetOrAddReal("problem", "pr", 0.1);
    sst_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.5);
    params.Add("sst_params", sst_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SST()
//! \brief Sets initial conditions for sst problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  PARTHENON_REQUIRE(do_gas, "The sst problem requires gas hydrodynamics!");
  PARTHENON_REQUIRE(!(do_dust), "The sst problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",1.4);
  Real gm1 = gamma - 1.;

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // SST parameters
  auto shkp = artemis_pkg->Param<SSTParams>("sst_params");

  // Setup sst state
  pmb->par_for(
      "sst", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
        const bool upwind = (xi[0] <= shkp.xdisc);
        const Real rho = upwind ? shkp.rhol : shkp.rhor;
        const Real vx = upwind ? shkp.vxl : shkp.vxr;
        const Real P = upwind ? shkp.pl : shkp.pr;
        v(0, gas::prim::density(0), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P/(rho*gm1);
      });

}

} // namespace sst
#endif // PGEN_SST_HPP_
