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
#ifndef PGEN_GAUSSIAN_BUMP_HPP_
#define PGEN_GAUSSIAN_BUMP_HPP_

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace gaussian_bump {

struct BumpParams {
  Real g_rho;
  Real g_vx1;
  Real g_vx2;
  Real g_vx3;
  Real g_pres;
  Real d_rho;
  Real d_vx1;
  Real d_vx2;
  Real d_vx3;
  Coordinates input_system;
  Coordinates system;
  Real xc_bump[3];
  Real sig_bump;
  Real dfac, tfac, ufac, vfac, wfac;
};

static BumpParams bump_params;

template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  auto pm = pmb->pmy_mesh;
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  EOS eos_d;

  bump_params.xc_bump[0] = pin->GetOrAddReal("problem", "x1c", 0.0);
  bump_params.xc_bump[1] = pin->GetOrAddReal("problem", "x2c", 0.0);
  bump_params.xc_bump[2] = pin->GetOrAddReal("problem", "x3c", 0.0);
  bump_params.sig_bump = pin->GetReal("problem", "sigma");
  bump_params.dfac = pin->GetOrAddReal("problem", "density_bump", 0.0);
  bump_params.tfac = pin->GetOrAddReal("problem", "temperature_bump", 0.0);
  bump_params.ufac = pin->GetOrAddReal("problem", "vx1_bump", 0.0);
  bump_params.vfac = pin->GetOrAddReal("problem", "vx2_bump", 0.0);
  bump_params.wfac = pin->GetOrAddReal("problem", "vx3_bump", 0.0);

  if (do_gas) {
    auto gas_pkg = pmb->packages.Get("gas");
    PARTHENON_REQUIRE((gas_pkg->Param<int>("nspecies") == 1),
                      "Gaussian bump pgen requires a single gas species.")
    eos_d = gas_pkg->Param<EOS>("eos_d");
    bump_params.g_rho = pin->GetOrAddReal("problem", "gas_rho", 1.0);
    bump_params.g_vx1 = pin->GetOrAddReal("problem", "gas_vx1", 0.0);
    bump_params.g_vx2 = pin->GetOrAddReal("problem", "gas_vx2", 0.0);
    bump_params.g_vx3 = pin->GetOrAddReal("problem", "gas_vx3", 0.0);
    bump_params.g_pres = pin->GetOrAddReal("problem", "gas_pres", 1.0);
  }
  if (do_dust) {
    auto dust_pkg = pmb->packages.Get("dust");
    PARTHENON_REQUIRE((dust_pkg->Param<int>("nspecies") == 1),
                      "Gaussian bump pgen requires a single dust species.")
    bump_params.d_rho = pin->GetOrAddReal("problem", "dust_rho", 1.0);
    bump_params.d_vx1 = pin->GetOrAddReal("problem", "dust_vx1", 0.0);
    bump_params.d_vx2 = pin->GetOrAddReal("problem", "dust_vx2", 0.0);
    bump_params.d_vx3 = pin->GetOrAddReal("problem", "dust_vx3", 0.0);
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

  std::string input_system = pin->GetString("problem", "system");
  std::string system = pin->GetString("artemis", "coordinates");
  bump_params.input_system = geometry::CoordSelect(input_system);
  bump_params.system = geometry::CoordSelect(system);

  auto &pco = pmb->coords;
  auto &pars = bump_params;

  const bool multi_d = (pm->ndim >= 2);
  const bool three_d = (pm->ndim == 3);
  const Real gamma = pin->GetReal("gas", "gamma");
  const Real cv = 1.0 / (gamma - 1.);
  pmb->par_for(
      "constant", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        Real xi[3] = {Null<Real>()};
        coords.GetCellCenter(xi);

        Real xo[3] = {Null<Real>()};
        Real ex1[3] = {Null<Real>()};
        Real ex2[3] = {Null<Real>()};
        Real ex3[3] = {Null<Real>()};
        if (pars.input_system == Coordinates::cartesian) {
          coords.ConvertToCartWithVec(xi, xo, ex1, ex2, ex3);
        } else if (pars.input_system == Coordinates::axisymmetric) {
          coords.ConvertToAxiWithVec(xi, xo, ex1, ex2, ex3);
        } else if (pars.input_system == Coordinates::spherical) {
          coords.ConvertToSphWithVec(xi, xo, ex1, ex2, ex3);
        } else if (pars.input_system == Coordinates::cylindrical) {
          coords.ConvertToCylWithVec(xi, xo, ex1, ex2, ex3);
        }

        Real xc[3] = {Null<Real>()};
        coords.ConvertToCart(xi, xc);

        const Real dx2 = SQR(xc[0] - pars.xc_bump[0]) +
                         SQR(xc[1] - pars.xc_bump[1]) * multi_d +
                         SQR(xc[2] - pars.xc_bump[2]) * three_d;
        const Real bump = std::exp(-dx2 / (2.0 * SQR(pars.sig_bump)));

        if (do_gas) {
          const Real vx1 =
              (pars.g_vx1 * ex1[0] + pars.g_vx2 * ex1[1] + pars.g_vx3 * ex1[2]);
          const Real vx2 =
              (pars.g_vx1 * ex2[0] + pars.g_vx2 * ex2[1] + pars.g_vx3 * ex2[2]);
          const Real vx3 =
              (pars.g_vx1 * ex3[0] + pars.g_vx2 * ex3[1] + pars.g_vx3 * ex3[2]);
          v(0, gas::prim::velocity(0), k, j, i) = vx1 + pars.ufac * bump;
          v(0, gas::prim::velocity(1), k, j, i) = vx2 + pars.vfac * bump;
          v(0, gas::prim::velocity(2), k, j, i) = vx3 + pars.wfac * bump;
          if (pars.tfac > 0.0) {
            // P = const = rho e (gamma-1);  T0*(1 + f)
            const Real sie0 = pars.g_pres / (pars.g_rho * (gamma - 1.0));
            const Real sie = sie0 * (1. + pars.tfac * bump);
            v(0, gas::prim::density(0), k, j, i) = pars.g_pres / (sie * (gamma - 1.0));
            v(0, gas::prim::sie(0), k, j, i) = sie;
          } else {
            const Real dens = pars.g_rho * (1. + pars.dfac * bump);
            v(0, gas::prim::density(0), k, j, i) = dens;
            v(0, gas::prim::sie(0), k, j, i) = pars.g_pres / ((gamma - 1.0) * dens);
          }
        }
        if (do_dust) {
          v(0, dust::prim::density(0), k, j, i) = pars.d_rho * (1. + pars.dfac * bump);
          v(0, dust::prim::velocity(0), k, j, i) =
              (pars.d_vx1 * ex1[0] + pars.d_vx2 * ex1[1] + pars.d_vx3 * ex1[2]);
          v(0, dust::prim::velocity(1), k, j, i) =
              (pars.d_vx1 * ex2[0] + pars.d_vx2 * ex2[1] + pars.d_vx3 * ex2[2]);
          v(0, dust::prim::velocity(2), k, j, i) =
              (pars.d_vx1 * ex3[0] + pars.d_vx2 * ex3[1] + pars.d_vx3 * ex3[2]);
        }
      });
}

} // namespace gaussian_bump
#endif // PGEN_GAUSSIAN_BUMP_HPP_
