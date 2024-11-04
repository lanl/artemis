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
#ifndef PGEN_CONSTANT_HPP_
#define PGEN_CONSTANT_HPP_
//! \file constant.hpp
//! \brief Constant initial conditions

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace {
//----------------------------------------------------------------------------------------
//! \struct ConstantParams
//! \brief container for variables associated with constant state
struct ConstantParams {
  Real g_rho;
  Real g_vx1;
  Real g_vx2;
  Real g_vx3;
  Real g_temp;
  Real d_rho;
  Real d_vx1;
  Real d_vx2;
  Real d_vx3;
  Coordinates input_system;
  Coordinates system;
};

} // end anonymous namespace

namespace constant {

static ConstantParams constant_params;

template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  EOS eos_d;
  if (do_gas) {
    auto gas_pkg = pmb->packages.Get("gas");
    PARTHENON_REQUIRE(gas_pkg->Param<int>("nspecies") == 1,
                      "Constant pgen requires a single gas species.")
    eos_d = gas_pkg->Param<EOS>("eos_d");
    constant_params.g_rho = pin->GetOrAddReal("problem", "gas_rho", 1.0);
    constant_params.g_vx1 = pin->GetOrAddReal("problem", "gas_vx1", 0.0);
    constant_params.g_vx2 = pin->GetOrAddReal("problem", "gas_vx2", 0.0);
    constant_params.g_vx3 = pin->GetOrAddReal("problem", "gas_vx3", 0.0);
    constant_params.g_temp = pin->GetOrAddReal("problem", "gas_temp", 1.0);
  }
  if (do_dust) {
    auto dust_pkg = pmb->packages.Get("dust");
    constant_params.d_rho = pin->GetOrAddReal("problem", "dust_rho", 1.0);
    constant_params.d_vx1 = pin->GetOrAddReal("problem", "dust_vx1", 0.0);
    constant_params.d_vx2 = pin->GetOrAddReal("problem", "dust_vx2", 0.0);
    constant_params.d_vx3 = pin->GetOrAddReal("problem", "dust_vx3", 0.0);
  }

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  auto pars = constant_params;

  const int ndim = ProblemDimension(pin);
  std::string input_system = pin->GetString("problem", "system");
  std::string system = pin->GetString("artemis", "coordinates");
  constant_params.input_system = geometry::CoordSelect(input_system, ndim);
  constant_params.system = geometry::CoordSelect(system, ndim);

  // setup uniform ambient medium
  pmb->par_for(
      "constant", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();

        auto xo = NewArray<Real, 3>();
        auto ex1 = NewArray<Real, 3>();
        auto ex2 = NewArray<Real, 3>();
        auto ex3 = NewArray<Real, 3>();
        if (pars.input_system == Coordinates::cartesian) {
          const auto &[xo_, ex1_, ex2_, ex3_] = coords.ConvertToCartWithVec(xi);
          xo = xo_;
          ex1 = ex1_;
          ex2 = ex2_;
          ex3 = ex3_;
        } else if (pars.input_system == Coordinates::axisymmetric) {
          const auto &[xo_, ex1_, ex2_, ex3_] = coords.ConvertToAxiWithVec(xi);
          xo = xo_;
          ex1 = ex1_;
          ex2 = ex2_;
          ex3 = ex3_;
        } else if (geometry::is_spherical(pars.input_system)) {
          const auto &[xo_, ex1_, ex2_, ex3_] = coords.ConvertToSphWithVec(xi);
          xo = xo_;
          ex1 = ex1_;
          ex2 = ex2_;
          ex3 = ex3_;
        } else if (pars.input_system == Coordinates::cylindrical) {
          const auto &[xo_, ex1_, ex2_, ex3_] = coords.ConvertToCylWithVec(xi);
          xo = xo_;
          ex1 = ex1_;
          ex2 = ex2_;
          ex3 = ex3_;
        }

        if (do_gas) {
          v(0, gas::prim::density(0), k, j, i) = pars.g_rho;
          v(0, gas::prim::velocity(0), k, j, i) =
              (pars.g_vx1 * ex1[0] + pars.g_vx2 * ex1[1] + pars.g_vx3 * ex1[2]);
          v(0, gas::prim::velocity(1), k, j, i) =
              (pars.g_vx1 * ex2[0] + pars.g_vx2 * ex2[1] + pars.g_vx3 * ex2[2]);
          v(0, gas::prim::velocity(2), k, j, i) =
              (pars.g_vx1 * ex3[0] + pars.g_vx2 * ex3[1] + pars.g_vx3 * ex3[2]);
          v(0, gas::prim::sie(0), k, j, i) =
              eos_d.InternalEnergyFromDensityTemperature(pars.g_rho, pars.g_temp);
        }
        if (do_dust) {
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            v(0, dust::prim::density(n), k, j, i) = pars.d_rho;
            v(0, dust::prim::velocity(n * 3 + 0), k, j, i) =
                (pars.d_vx1 * ex1[0] + pars.d_vx2 * ex1[1] + pars.d_vx3 * ex1[2]);
            v(0, dust::prim::velocity(n * 3 + 1), k, j, i) =
                (pars.d_vx1 * ex2[0] + pars.d_vx2 * ex2[1] + pars.d_vx3 * ex2[2]);
            v(0, dust::prim::velocity(n * 3 + 2), k, j, i) =
                (pars.d_vx1 * ex3[0] + pars.d_vx2 * ex3[1] + pars.d_vx3 * ex3[2]);
          }
        }
      });
}

} // namespace constant
#endif // PGEN_CONSTANT_HPP_
