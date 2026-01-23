//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#ifndef PGEN_CROOKED_PIPE_HPP_
#define PGEN_CROOKED_PIPE_HPP_
//! \file crooked_pipe.hpp
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

namespace crooked_pipe {

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Crooked_Pipe()
//! \brief Sets initial conditions for crooked pipe radiation flow problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_radiation = artemis_pkg->Param<bool>("do_radiation");
  const bool do_imc = artemis_pkg->Param<bool>("do_imc");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");
  PARTHENON_REQUIRE(do_gas, "Crooked pipe problem requires gas!");
  PARTHENON_REQUIRE(!(do_dust), "Crooked pipe problem does not permit dust!");
  //PARTHENON_REQUIRE(!(do_gas), "Crooked pipe problem does not permit gas!");
  auto gas_pkg = pmb->packages.Get("gas");
  const auto eos = gas_pkg->Param<EOS>("eos_d");
  Real ar = Null<Real>();
  if (do_moment) {
    auto rad_pkg = pmb->packages.Get("moments");
    ar = rad_pkg->Param<Real>("arad");
  }

  // Initial conditions
  const Real rho_thin = pin->GetOrAddReal("problem", "rho_thin", 1.0);
  const Real rho_thick = pin->GetOrAddReal("problem", "rho_thick", 1000.0);
  const Real t_source = pin->GetOrAddReal("problem", "t_source", 0.3); // put in K?
  const Real t_init = pin->GetOrAddReal("problem", "t_init", 0.01); // put in K?

  // Allocate sparse
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  // packing and capture variables for kernel
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         rad::prim::energy, rad::prim::flux>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  std::array<std::array<double, 4>, 7> thick_regions = {{{3.0,4.0, -1.0,1.0},
                                                      {-2.0,2.5, -2.0, -0.5}, // extended xl to -2 for thick region above source
                                                      {-2.0, 2.5, 0.5, 2.0}, // extended xl to -2 for thick region above source
                                                      {4.5, 7.0, -2.0, -0.5},
                                                      {4.5, 7.0, 0.5, 2.0},
                                                      {2.5, 4.5, -2.0, -1.5},
                                                      {2.5, 4.5, 1.5, 2.5}}};

  std::array<std::array<double, 4>, 1> thin_source_regions= {{{-2.0,0.0,-0.5,0.5}}};

  const auto &cpars =
      pmb->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");
  auto &pco = pmb->coords;

  static auto desc_g = MakePackDescriptor<geom::vol, geom::x1v, geom::x2v, geom::x3v>(
      (pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());

  // Set state vector to initialize:
  // * source radiation and material temperatureradiation field via t_source
  // * density of thin regions using rho_thin
  // * density of thick regions using rho_thick
  // * initial material and radiation temperature of all cell using t_init
  if (do_imc) {
    pmb->par_for(
      "crooked_pipe::trad", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {

      geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
      const auto &xv = coords.GetCellCenter(vg, 0, k, j, i);

      const Real xl = xv[0];
      const Real xu = xv[0];
      const Real yl = xv[1];
      const Real yu = xv[1];

      // default thin cell
      v(0, gas::prim::density(), k, j, i) = rho_thin;
      v(0, gas::prim::sie(), k, j, i) =
          eos.InternalEnergyFromDensityTemperature(rho_thin, t_init);

      for( const auto &iregion : thick_regions) {
        if (xl >= iregion[0] && xu <= iregion[1] && yl >= iregion[2] && yu <= iregion[3]) {
          v(0, gas::prim::density(), k, j, i) = rho_thick;
          v(0, gas::prim::sie(), k, j, i) =
              eos.InternalEnergyFromDensityTemperature(rho_thick, t_init);
        }
      }
      for( const auto &iregion : thin_source_regions) {
        if (xl >= iregion[0] && xu <= iregion[1] && yl >= iregion[2] && yu <= iregion[3]) {
          v(0, gas::prim::sie(), k, j, i) =
              eos.InternalEnergyFromDensityTemperature(rho_thin, t_source);
        }
      }
    });
    jaybenne::InitializeRadiation(md.get(), true);
  }

  /*
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

        if (do_moment) {
          v(0, rad::prim::energy(), k, j, i) = ar * SQR(SQR(trad));
          v(0, rad::prim::flux(0), k, j, i) = 0.0;
          v(0, rad::prim::flux(1), k, j, i) = 0.0;
          v(0, rad::prim::flux(2), k, j, i) = 0.0;
        }
      });
  */
}

} // namespace crooked_pipe
#endif // PGEN_CROOKED_PIPE_HPP_

