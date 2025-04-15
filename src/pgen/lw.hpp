//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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
#ifndef PGEN_LW_HPP_
#define PGEN_LW_HPP_

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace {

struct LWParams {
  Real rho0, rho1;
  Real pres0, pres1;
  Real sie0, sie1;
  Real y0;
};

} // end anonymous namespace

namespace lw {

static LWParams lw_params;

template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");

  lw_params.y0 = pin->GetOrAddReal("problem", "y0", 0.15);
  lw_params.rho0 = pin->GetOrAddReal("problem", "rho0", 0.125);
  lw_params.pres0 = pin->GetOrAddReal("problem", "pres0", 0.14);
  lw_params.rho1 = pin->GetOrAddReal("problem", "rho1", 1.0);
  lw_params.pres1 = pin->GetOrAddReal("problem", "pres1", 1.0);

  const auto gm1 = gas_pkg->Param<Real>("adiabatic_index");

  lw_params.sie0 = lw_params.pres0/(lw_params.rho0 * gm1);
  lw_params.sie1 = lw_params.pres1/(lw_params.rho1 * gm1);

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
  auto pars = lw_params;

  const int ndim = ProblemDimension(pin);


  // setup uniform ambient medium
  pmb->par_for(
      "lw", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto bbox = geometry::BBox(pco,k,j,i);
        const std::array<Real,4> px{ bbox.x1[0],bbox.x1[1], bbox.x1[1],bbox.x1[0]};
        const std::array<Real,4> py{ bbox.x2[0],bbox.x2[0], bbox.x2[1],bbox.x2[1]};
        const Real isqrt2 = std::sqrt(1./2.);
        const Real vf1 = ArtemisUtils::CutCell2D(px, py, {0.5*pars.y0, 0.5*pars.y0}, {isqrt2, isqrt2});
        const Real vf2 = 1.0 - vf1;
        
        const Real dens = vf1 * pars.rho1 + vf2 * pars.rho0;
        const Real pres = vf1 * pars.pres1 + vf2 * pars.pres0;
        
        v(0,gas::prim::density(0),k,j,i) = dens;
        v(0,gas::prim::sie(0),k,j,i) = pres/(dens * gm1); 
        v(0,gas::prim::velocity(0),k,j,i) = 0.0;
        v(0,gas::prim::velocity(1),k,j,i) = 0.0;
        v(0,gas::prim::velocity(2),k,j,i) = 0.0;
  });
}

} // namespace lw
#endif // PGEN_LW_HPP_
