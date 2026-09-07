//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef PGEN_FIELD_HPP_
#define PGEN_FIELD_HPP_
//! \file FIELD.hpp
//! \brief Problem generator that initializes primitive fields from a
//! user-provided Spiner file.

// C/C++
#include <memory>
#include <string>
#include <vector>

#ifdef SPINER_USE_HDF
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#include <ports-of-call/portability.hpp>
#include <spiner/databox.hpp>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;
using DataBox = Spiner::DataBox<Real>;

namespace field {

struct DataBoxes {
  Spiner::DataBox<Real> gas_rho;
  Spiner::DataBox<Real> gas_temp;
  Spiner::DataBox<Real> gas_vx;
  Spiner::DataBox<Real> gas_vy;
  Spiner::DataBox<Real> gas_vz;
  Spiner::DataBox<Real> dust_rho;
  Spiner::DataBox<Real> dust_vx;
  Spiner::DataBox<Real> dust_vy;
  Spiner::DataBox<Real> dust_vz;
  Spiner::DataBox<Real> rad_e;
  Spiner::DataBox<Real> rad_fx;
  Spiner::DataBox<Real> rad_fy;
  Spiner::DataBox<Real> rad_fz;

  bool do_gas = false, do_dust = false, do_moment = false;
  bool use_temp = false, use_pres = false, use_sie = false;
};

//----------------------------------------------------------------------------------------
//! \fn void InitFieldParams
//! \brief Loads the Spiner file once per rank
inline void InitFieldParams(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (params.hasKey("field_params")) return;
  DataBoxes state;

  state.do_gas = params.Get<bool>("do_gas");
  state.do_dust = params.Get<bool>("do_dust");
  state.do_moment = params.Get<bool>("do_moment");

  std::string filename = pin->GetString("problem", "data_file");
  PARTHENON_REQUIRE(
      !filename.empty(),
      "field pgen: data_file parameter must be provided for FIELD problem.");
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (state.do_gas) {
    state.gas_rho.loadHDF(file, "gas_density");
    auto var = pin->GetString("problem", "gas_variable");
    char c = var[0];
    switch (std::toupper(c)) {
    case 'T':
      state.use_temp = true;
      state.gas_temp.loadHDF(file, "gas_temperature");
      break;
    case 'P':
      state.use_pres = true;
      state.gas_temp.loadHDF(file, "gas_pressure");
      break;
    case 'S':
      state.use_sie = true;
      state.gas_temp.loadHDF(file, "gas_sie");
      break;
    default:
      PARTHENON_FAIL("FIELD pgen: unrecognized gas variable " + var);
    }
    state.gas_vx.loadHDF(file, "gas_vx");
    state.gas_vy.loadHDF(file, "gas_vy");
    state.gas_vz.loadHDF(file, "gas_vz");
  }

  if (state.do_dust) {
    state.dust_rho.loadHDF(file, "dust_density");
    state.dust_vx.loadHDF(file, "dust_vx");
    state.dust_vy.loadHDF(file, "dust_vy");
    state.dust_vz.loadHDF(file, "dust_vz");
  }
  if (state.do_moment) {
    state.rad_e.loadHDF(file, "rad_energy");
    state.rad_fx.loadHDF(file, "rad_fx");
    state.rad_fy.loadHDF(file, "rad_fy");
    state.rad_fz.loadHDF(file, "rad_fz");
  }
  H5Fclose(file);

  params.Add("field_params", state);
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator
//! \brief Writes primitives by sampling the loaded fields at each cell center.
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;

  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto &field_params = artemis_pkg->Param<DataBoxes>("field_params");

  EOS eos_d;
  if (field_params.do_gas) {
    eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  }

  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity, rad::prim::energy,
                         rad::prim::flux>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  const auto &sp = field_params;
  pmb->par_for(
      "FIELD", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto &xv = coords.GetCellCenter();

        if (sp.do_gas) {
          const int nspecies = v.GetSize(0, gas::prim::density());
          for (int n = 0; n < nspecies; ++n) {
            const auto id = v(0, gas::prim::density(n)).sparse_id;
            const Real rho = sp.gas_rho.interpToReal(xv[2], xv[1], xv[0], id);
            const Real q = sp.gas_temp.interpToReal(xv[2], xv[1], xv[0], id);
            if (sp.use_temp) {
              v(0, gas::prim::sie(n), k, j, i) =
                  eos_d.InternalEnergyFromDensityTemperature(rho, q);
            } else if (sp.use_pres) {
              v(0, gas::prim::sie(n), k, j, i) = ArtemisUtils::EofPR(eos_d, q, rho);
            } else if (sp.use_sie) {
              v(0, gas::prim::sie(n), k, j, i) = q;
            }
            v(0, gas::prim::density(n), k, j, i) = rho;
            v(0, gas::prim::velocity(VI(n, 0)), k, j, i) =
                sp.gas_vx.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, gas::prim::velocity(VI(n, 1)), k, j, i) =
                sp.gas_vy.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, gas::prim::velocity(VI(n, 2)), k, j, i) =
                sp.gas_vz.interpToReal(xv[2], xv[1], xv[0], id);
          }
        }
        if (sp.do_dust) {
          const int nspecies = v.GetSize(0, dust::prim::density());
          for (int n = 0; n < nspecies; ++n) {
            const auto id = v(0, dust::prim::density(n)).sparse_id;
            const Real rho = sp.dust_rho.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, dust::prim::density(n), k, j, i) = rho;
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) =
                sp.dust_vx.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) =
                sp.dust_vy.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) =
                sp.dust_vz.interpToReal(xv[2], xv[1], xv[0], id);
          }
        }

        if (sp.do_moment) {
          const int nspecies = v.GetSize(0, rad::prim::energy());
          for (int n = 0; n < nspecies; ++n) {
            const auto id = v(0, rad::prim::energy(n)).sparse_id;
            v(0, rad::prim::energy(n), k, j, i) =
                sp.rad_e.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, rad::prim::flux(VI(n, 0)), k, j, i) =
                sp.rad_fx.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, rad::prim::flux(VI(n, 1)), k, j, i) =
                sp.rad_fy.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, rad::prim::flux(VI(n, 2)), k, j, i) =
                sp.rad_fz.interpToReal(xv[2], xv[1], xv[0], id);
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void field::FieldBoundaryIC()
//! \brief Sets inner or outer boundary condition to the initial condition
template <Coordinates GEOM, IndexDomain BDY>
void FieldBoundaryIC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  PARTHENON_INSTRUMENT
  auto pmb = mbd->GetBlockPointer();

  auto artemis_pkg = pmb->packages.Get("artemis");
  const auto &field_params = artemis_pkg->Param<DataBoxes>("field_params");

  auto &gas_pkg = pmb->packages.Get("gas");
  auto eos_d = gas_pkg->template Param<EOS>("eos_d");

  static auto descriptors = ArtemisUtils::GetBoundaryPackDescriptorMap<
      gas::prim::density, gas::prim::velocity, gas::prim::sie, dust::prim::density,
      dust::prim::velocity, rad::prim::energy, rad::prim::flux>(mbd);
  auto v = descriptors[coarse].GetPack(mbd.get());
  if (v.GetMaxNumberOfVars() == 0) return;

  const auto &pco = (coarse) ? pmb->pmr->GetCoarseCoords() : pmb->coords;
  const auto &sp = field_params;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  pmb->par_for_bndry(
      "DiskInnerX1", nb, BDY, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto &xv = coords.GetCellCenter();

        if (sp.do_gas) {
          const int nspecies = v.GetSize(0, gas::prim::density());
          for (int n = 0; n < nspecies; ++n) {
            const auto id = v(0, gas::prim::density(n)).sparse_id;
            const Real rho = sp.gas_rho.interpToReal(xv[2], xv[1], xv[0], id);
            const Real q = sp.gas_temp.interpToReal(xv[2], xv[1], xv[0], id);
            if (sp.use_temp) {
              v(0, gas::prim::sie(n), k, j, i) =
                  eos_d.InternalEnergyFromDensityTemperature(rho, q);
            } else if (sp.use_pres) {
              v(0, gas::prim::sie(n), k, j, i) = ArtemisUtils::EofPR(eos_d, q, rho);
            } else if (sp.use_sie) {
              v(0, gas::prim::sie(n), k, j, i) = q;
            }
            v(0, gas::prim::density(n), k, j, i) = rho;
            v(0, gas::prim::velocity(VI(n, 0)), k, j, i) =
                sp.gas_vx.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, gas::prim::velocity(VI(n, 1)), k, j, i) =
                sp.gas_vy.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, gas::prim::velocity(VI(n, 2)), k, j, i) =
                sp.gas_vz.interpToReal(xv[2], xv[1], xv[0], id);
          }
        }
        if (sp.do_dust) {
          const int nspecies = v.GetSize(0, dust::prim::density());
          for (int n = 0; n < nspecies; ++n) {
            const auto id = v(0, dust::prim::density(n)).sparse_id;
            const Real rho = sp.dust_rho.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, dust::prim::density(n), k, j, i) = rho;
            v(0, dust::prim::velocity(VI(n, 0)), k, j, i) =
                sp.dust_vx.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, dust::prim::velocity(VI(n, 1)), k, j, i) =
                sp.dust_vy.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, dust::prim::velocity(VI(n, 2)), k, j, i) =
                sp.dust_vz.interpToReal(xv[2], xv[1], xv[0], id);
          }
        }
        if (sp.do_moment) {
          const int nspecies = v.GetSize(0, rad::prim::energy());
          for (int n = 0; n < nspecies; ++n) {
            const auto id = v(0, rad::prim::energy(n)).sparse_id;
            v(0, rad::prim::energy(n), k, j, i) =
                sp.rad_e.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, rad::prim::flux(VI(n, 0)), k, j, i) =
                sp.rad_fx.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, rad::prim::flux(VI(n, 1)), k, j, i) =
                sp.rad_fy.interpToReal(xv[2], xv[1], xv[0], id);
            v(0, rad::prim::flux(VI(n, 2)), k, j, i) =
                sp.rad_fz.interpToReal(xv[2], xv[1], xv[0], id);
          }
        }
      });
}

} // namespace field

#endif // PGEN_FIELD_HPP_
