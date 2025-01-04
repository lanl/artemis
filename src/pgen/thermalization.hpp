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
#ifndef PGEN_THERMALIZATION_HPP_
#define PGEN_THERMALIZATION_HPP_
//! \file thermalization.hpp
//! \brief

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

// jaybenne includes
#include "jaybenne.hpp"

using ArtemisUtils::EOS;

namespace thermalization {

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Thermalization()
//! \brief Sets initial conditions for thermal relaxation problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_radiation = artemis_pkg->Param<bool>("do_radiation");
  const bool do_imc = artemis_pkg->Param<bool>("do_imc");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  PARTHENON_REQUIRE(do_gas, "Thermalization problem requires gas!");
  PARTHENON_REQUIRE(!(do_dust), "Thermalization problem does not permit dust!");
  auto gas_pkg = pmb->packages.Get("gas");

  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  const Real tgas = pin->GetOrAddReal("problem", "tgas", 2.0);
  const Real trad = pin->GetOrAddReal("problem", "trad", 1.0);

  const auto eos = gas_pkg->Param<EOS>("eos_d");
  Real ar = Null<Real>();
  if (do_moment) {
    auto rad_pkg = pmb->packages.Get("radiation");
    ar = rad_pkg->Param<Real>("arad");
  }

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         rad::prim::energy, rad::prim::flux>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // Set state vector to initialize radiation field via trad
  if (do_imc) {
    pmb->par_for(
        "thermalization::trad", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          v(0, gas::prim::density(), k, j, i) = rho;
          v(0, gas::prim::sie(), k, j, i) =
              eos.InternalEnergyFromDensityTemperature(rho, trad);
        });

    jaybenne::InitializeRadiation(md.get(), true);
  }

  // Now reset fluid state out of thermal equilibrium via tgas
  pmb->par_for(
      "thermalization::tgas", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(0, gas::prim::density(), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = vx;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(), k, j, i) =
            eos.InternalEnergyFromDensityTemperature(rho, tgas);
        ;
        if (do_moment) {
          v(0, rad::prim::energy(), k, j, i) = ar * SQR(SQR(trad));
          v(0, rad::prim::flux(0), k, j, i) = 0.0;
          v(0, rad::prim::flux(1), k, j, i) = 0.0;
          v(0, rad::prim::flux(2), k, j, i) = 0.0;
        }
        printf("%lg %lg\n",
               v(0, gas::prim::sie(), k, j, i) * v(0, gas::prim::density(), k, j, i),
               v(0, rad::prim::energy(), k, j, i));
      });
}

} // namespace thermalization
#endif // PGEN_THERMALIZATION_HPP_
