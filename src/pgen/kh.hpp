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
#ifndef PGEN_KH_HPP_
#define PGEN_KH_HPP_

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace {

struct KHParams {
  Real rho0, rho1;
  Real pres0;
  Real y1, y2;
  Real a, sigma, amp, u;
};

} // end anonymous namespace

namespace kh {

static KHParams KH_params;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::KH
//! \brief
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  auto gas_pkg = pmb->packages.Get("gas");
  const auto &eos = gas_pkg->Param<ArtemisUtils::EOS>("eos_d");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");

  KH_params.y1 = pin->GetOrAddReal("problem", "y1", 0.5);
  KH_params.y2 = pin->GetOrAddReal("problem", "y2", 1.5);
  KH_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  KH_params.pres0 = pin->GetOrAddReal("problem", "pres0", 10.0);
  KH_params.rho1 = pin->GetOrAddReal("problem", "rho1", 2.0);
  KH_params.amp = pin->GetOrAddReal("problem", "amplitude", 0.01);
  KH_params.a = pin->GetOrAddReal("problem", "a", 0.05);
  KH_params.sigma = pin->GetOrAddReal("problem", "sigma", 0.2);
  KH_params.u = pin->GetOrAddReal("problem", "uflow", 1.0);

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
  auto pars = KH_params;

  const int ndim = ProblemDimension(pin);

  PARTHENON_REQUIRE(ndim > 1, "KH problem requires ndim >= 2");
  const bool three_d = (ndim == 3);

  // setup uniform ambient medium
  pmb->par_for(
      "KH", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto bbox = geometry::BBox(pco, k, j, i);
        const Real xc = 0.5 * (bbox.x1[0] + bbox.x1[1]);
        const Real zc =
            (three_d) ? 0.5 * (bbox.x3[0] + bbox.x3[1]) : 0.5 * (bbox.x2[0] + bbox.x2[1]);

        const Real dens = 1.0 + 0.5 * (pars.rho1 / pars.rho0 - 1.0) *
                                    (std::tanh((zc - pars.y1) / pars.a) -
                                     std::tanh((zc - pars.y2) / pars.a));
        const Real vx = pars.u * (std::tanh((zc - pars.y1) / pars.a) -
                                  std::tanh((zc - pars.y2) / pars.a) - 1.0);
        const Real vz = pars.amp * std::sin(2 * M_PI * xc) *
                        (std::exp(-SQR((zc - pars.y1) / pars.sigma)) +
                         std::exp(-SQR((zc - pars.y2) / pars.sigma)));

        v(0, gas::prim::density(0), k, j, i) = dens;
        v(0, gas::prim::sie(0), k, j, i) = ArtemisUtils::EofPR(eos, pars.pres0, dens);
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = (!three_d) * vz;
        v(0, gas::prim::velocity(2), k, j, i) = three_d * vz;
      });
}

} // namespace kh
#endif // PGEN_KH_HPP_
