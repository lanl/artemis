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
#ifndef PGEN_CONDUCTION_HPP_
#define PGEN_CONDUCTION_HPP_
//! \file conduction.hpp
//! \brief Conduction initial conditions

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/diffusion/diffusion_coeff.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;

namespace cond {
//----------------------------------------------------------------------------------------
//! \struct CondParams
//! \brief container for variables associated with constant state
struct CondParams {
  Real g_rho;
  Real g_vx1;
  Real g_vx2;
  Real g_vx3;
  Real g_temp;
  Real flux;
};

//----------------------------------------------------------------------------------------
//! \fn void InitCondParams
//! \brief Extracts cond parameters from ParameterInput.
//! NOTE(PDM): In order for our user-defined BCs to be compatible with restarts, we must
//! reset the CondParams struct upon initialization.
inline void InitCondParams(MeshBlock *pmb, ParameterInput *pin) {
  Params &params = pmb->packages.Get("artemis")->AllParams();
  if (!(params.hasKey("cond_pgen_params"))) {
    CondParams cond_params;
    cond_params.g_rho = pin->GetOrAddReal("problem", "gas_rho", 1.0);
    cond_params.g_vx1 = pin->GetOrAddReal("problem", "gas_vx1", 0.0);
    cond_params.g_vx2 = pin->GetOrAddReal("problem", "gas_vx2", 0.0);
    cond_params.g_vx3 = pin->GetOrAddReal("problem", "gas_vx3", 0.0);
    cond_params.g_temp = pin->GetOrAddReal("problem", "gas_temp", 1.0);
    cond_params.flux = pin->GetOrAddReal("problem", "flux", 0.0);
    params.Add("cond_pgen_params", cond_params);
  }
}

template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_gravity = artemis_pkg->Param<bool>("do_gravity");
  auto cond_params = artemis_pkg->Param<CondParams>("cond_pgen_params");
  EOS eos_d;
  Real gx1 = 0.;
  if (do_gas) {
    auto gas_pkg = pmb->packages.Get("gas");
    PARTHENON_REQUIRE(gas_pkg->Param<int>("nspecies") == 1,
                      "Cond pgen requires a single gas species.")
    eos_d = gas_pkg->Param<EOS>("eos_d");
    if (do_gravity) {
      auto grav_pkg = pmb->packages.Get("gravity");
      if (grav_pkg->Param<Gravity::GravityType>("type") ==
          Gravity::GravityType::uniform) {
        gx1 = grav_pkg->Param<Real>("gx1");
      }
    }
  }

  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
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
  auto pars = cond_params;

  pmb->par_for(
      "conduction", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (do_gas) {
          geometry::Coords<GEOM> coords(pco, k, j, i);
          const auto &xv = coords.GetCellCenter();

          const Real P0 = eos_d.PressureFromDensityTemperature(pars.g_rho, pars.g_temp);
          const Real Rgas = P0 / (pars.g_rho * pars.g_temp);
          const Real P = P0 * std::exp(gx1 * pars.g_rho / P0 * (xv[0] - x1min));
          const Real dens = P / (Rgas * pars.g_temp);

          v(0, gas::prim::density(0), k, j, i) = dens;
          v(0, gas::prim::velocity(0), k, j, i) = pars.g_vx1;
          v(0, gas::prim::velocity(1), k, j, i) = pars.g_vx2;
          v(0, gas::prim::velocity(2), k, j, i) = pars.g_vx3;
          v(0, gas::prim::sie(0), k, j, i) =
              eos_d.InternalEnergyFromDensityTemperature(dens, pars.g_temp);
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void Disk::CondBoundaryImpl()
//! \brief Sets inner X1 boundary condition to the initial condition
template <Coordinates GEOM, IndexDomain BDY, Diffusion::DiffType DTYP>
void CondBoundaryImpl(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {

  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  auto &gas_pkg = pmb->packages.Get("gas");
  auto cond_params = artemis_pkg->Param<CondParams>("cond_pgen_params");
  auto diff_params = gas_pkg->Param<Diffusion::DiffCoeffParams>("cond_params");

  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  const bool fine = false;

  static auto descriptors =
      ArtemisUtils::GetBoundaryPackDescriptorMap<gas::prim::density, gas::prim::velocity,
                                                 gas::prim::sie>(mbd);

  auto v = descriptors[coarse].GetPack(mbd.get());

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  auto &dp = cond_params;
  auto &dcp = diff_params;
  const auto nb = IndexRange{0, 0};

  // Boundary index arithmetic
  int is = Null<int>(), ie = Null<int>();
  int js = Null<int>(), je = Null<int>();
  int ks = Null<int>(), ke = Null<int>();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  if constexpr (BDY == IndexDomain::inner_x1) {
    is = bounds.GetBoundsI(IndexDomain::interior, TE::CC).s;
  } else if constexpr (BDY == IndexDomain::outer_x1) {
    ie = bounds.GetBoundsI(IndexDomain::interior, TE::CC).e;
  } else if constexpr (BDY == IndexDomain::inner_x2) {
    js = bounds.GetBoundsJ(IndexDomain::interior, TE::CC).s;
  } else if constexpr (BDY == IndexDomain::outer_x2) {
    je = bounds.GetBoundsJ(IndexDomain::interior, TE::CC).e;
  } else if constexpr (BDY == IndexDomain::inner_x3) {
    ks = bounds.GetBoundsK(IndexDomain::interior, TE::CC).s;
  } else if constexpr (BDY == IndexDomain::outer_x3) {
    ke = bounds.GetBoundsK(IndexDomain::interior, TE::CC).e;
  }
  constexpr bool INNER = (BDY == IndexDomain::inner_x1) ||
                         (BDY == IndexDomain::inner_x2) || (BDY == IndexDomain::inner_x3);
  constexpr bool x1dir =
      ((BDY == IndexDomain::inner_x1) || (BDY == IndexDomain::outer_x1));
  constexpr bool x2dir =
      ((BDY == IndexDomain::inner_x2) || (BDY == IndexDomain::outer_x2));
  constexpr bool x3dir =
      ((BDY == IndexDomain::inner_x3) || (BDY == IndexDomain::outer_x3));
  constexpr int ix1 = x1dir ? 0 : (x2dir ? 1 : 2);
  constexpr int ix2 = (ix1 + 1) % 3;
  constexpr int ix3 = (ix1 + 2) % 3;

  Real gx1 = 0.;
  auto do_gravity = artemis_pkg->template Param<bool>("do_gravity");
  auto do_gas = artemis_pkg->template Param<bool>("do_gas");
  if (do_gravity) {
    auto grav_pkg = pmb->packages.Get("gravity");
    Gravity::GravityType gtype = grav_pkg->template Param<Gravity::GravityType>("type");
    if (gtype == Gravity::GravityType::uniform) {
      if (x1dir) {
        gx1 = grav_pkg->template Param<Real>("gx1");
      } else if (x2dir) {
        gx1 = grav_pkg->template Param<Real>("gx2");
      } else {
        gx1 = grav_pkg->template Param<Real>("gx3");
      }
    }
  }

  pmb->par_for_bndry(
      "CondBC", nb, BDY, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // We are extrapolating into the ghost zone. Extrapolation is done on cylinders.
        //   dP/dx = -rho g
        //   F = -K dT/dx
        const int ia[3] = {x3dir ? ((BDY == IndexDomain::inner_x3) ? ks : ke) : k,
                           x2dir ? ((BDY == IndexDomain::inner_x2) ? js : je) : j,
                           x1dir ? ((BDY == IndexDomain::inner_x1) ? is : ie) : i};

        // Extract coordinates at k, j, i
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xv = coords.GetCellCenter();

        // Extract coordinates at ia, im, ic
        geometry::Coords<GEOM> ca(pco, ia[0], ia[1], ia[2]);
        const auto &xva = ca.GetCellCenter();

        const Real xma = (INNER ? -1. : 1.) * coords.Distance(xv, xva);

        if (do_gas) {
          Diffusion::DiffusionCoeff<DTYP, GEOM, Fluid::gas> dcoeff;

          // active zone
          const Real da = v(0, gas::prim::density(0), ia[0], ia[1], ia[2]);
          const Real siea = v(0, gas::prim::sie(0), ia[0], ia[1], ia[2]);
          const Real Ta = eos_d.TemperatureFromDensityInternalEnergy(da, siea);
          const Real Pa = eos_d.PressureFromDensityInternalEnergy(da, siea);

          const Real ka = dcoeff.Get(dcp, ca, da, siea, eos_d);
          Real Tg = dp.g_temp;
          if (INNER) {
            Tg = Ta - dp.flux * xma / ka;
          }

          // Density from dP/dx = - rho g
          const Real densg = da * (Ta - 0.5 * gx1 * xma) / (Tg + 0.5 * gx1 * xma);
          const Real sieg = eos_d.InternalEnergyFromDensityTemperature(densg, Tg);

          // Extrapolate gas velocity
          Real gva[3] = {v(0, gas::prim::velocity(0), ia[0], ia[1], ia[2]),
                         v(0, gas::prim::velocity(1), ia[0], ia[1], ia[2]),
                         v(0, gas::prim::velocity(2), ia[0], ia[1], ia[2])};

          // Set extrapolated values
          v(0, gas::prim::density(0), k, j, i) = densg;
          v(0, gas::prim::sie(0), k, j, i) = sieg;
          v(0, gas::prim::velocity(ix1), k, j, i) = gva[ix1];
          v(0, gas::prim::velocity(ix2), k, j, i) = gva[ix2];
          v(0, gas::prim::velocity(ix3), k, j, i) = gva[ix3];
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void CondBoundary
//! \brief
template <Coordinates GEOM, IndexDomain BDY>
inline void CondBoundary(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto artemis_pkg = pmb->packages.Get("artemis");
  auto &pkg = pmb->packages.Get("gas");

  const auto dcp = pkg->template Param<Diffusion::DiffCoeffParams>("cond_params");

  if constexpr (BDY == IndexDomain::inner_x1) {
    if (dcp.type == Diffusion::DiffType::conductivity_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::inner_x1,
                              Diffusion::DiffType::conductivity_plaw>(mbd, coarse);
    } else if (dcp.type == Diffusion::DiffType::thermaldiff_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::inner_x1,
                              Diffusion::DiffType::thermaldiff_plaw>(mbd, coarse);
    } else {
      PARTHENON_FAIL(
          "Chosen conductivity type is not compatible with conductivity boundaries");
    }
  } else if constexpr (BDY == IndexDomain::outer_x1) {
    if (dcp.type == Diffusion::DiffType::conductivity_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::outer_x1,
                              Diffusion::DiffType::conductivity_plaw>(mbd, coarse);
    } else if (dcp.type == Diffusion::DiffType::thermaldiff_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::outer_x1,
                              Diffusion::DiffType::thermaldiff_plaw>(mbd, coarse);
    } else {
      PARTHENON_FAIL(
          "Chosen conductivity type is not compatible with conductivity boundaries");
    }
  } else if constexpr (BDY == IndexDomain::inner_x2) {
    if (dcp.type == Diffusion::DiffType::conductivity_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::inner_x2,
                              Diffusion::DiffType::conductivity_plaw>(mbd, coarse);
    } else if (dcp.type == Diffusion::DiffType::thermaldiff_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::inner_x2,
                              Diffusion::DiffType::thermaldiff_plaw>(mbd, coarse);
    } else {
      PARTHENON_FAIL(
          "Chosen conductivity type is not compatible with conductivity boundaries");
    }
  } else if constexpr (BDY == IndexDomain::outer_x2) {
    if (dcp.type == Diffusion::DiffType::conductivity_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::outer_x2,
                              Diffusion::DiffType::conductivity_plaw>(mbd, coarse);
    } else if (dcp.type == Diffusion::DiffType::thermaldiff_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::outer_x2,
                              Diffusion::DiffType::thermaldiff_plaw>(mbd, coarse);
    } else {
      PARTHENON_FAIL(
          "Chosen conductivity type is not compatible with conductivity boundaries");
    }
  } else if constexpr (BDY == IndexDomain::inner_x3) {
    if (dcp.type == Diffusion::DiffType::conductivity_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::inner_x3,
                              Diffusion::DiffType::conductivity_plaw>(mbd, coarse);
    } else if (dcp.type == Diffusion::DiffType::thermaldiff_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::inner_x3,
                              Diffusion::DiffType::thermaldiff_plaw>(mbd, coarse);
    } else {
      PARTHENON_FAIL(
          "Chosen conductivity type is not compatible with conductivity boundaries");
    }
  } else if constexpr (BDY == IndexDomain::outer_x3) {
    if (dcp.type == Diffusion::DiffType::conductivity_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::outer_x3,
                              Diffusion::DiffType::conductivity_plaw>(mbd, coarse);
    } else if (dcp.type == Diffusion::DiffType::thermaldiff_plaw) {
      return CondBoundaryImpl<GEOM, IndexDomain::outer_x3,
                              Diffusion::DiffType::thermaldiff_plaw>(mbd, coarse);
    } else {
      PARTHENON_FAIL(
          "Chosen conductivity type is not compatible with conductivity boundaries");
    }
  }
}

} // namespace cond
#endif // PGEN_CONDUCTION_HPP_
