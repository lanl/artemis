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

// C++ headers
#include <limits>
#include <vector>

// Parthenon includes
#include <globals.hpp>

// Artemis includes
#include "artemis.hpp"
#include "sts.hpp"
#include "utils/artemis_utils.hpp"


namespace STS{
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor STS ::Initialize
//! \brief Adds intialization function for STS package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {

  // Extract artemis package
  artemis_pkg = pm->packages.Get("artemis").get();

  // initial check of getting the physics needed for the sts integrator
  do_viscosity = artemis_pkg->template Param<bool>("do_viscosity");
  do_conduction = artemis_pkg->template Param<bool>("do_conduction");
  bool do_diffusion = do_viscosity || do_conduction;

  if (!do_diffusion) {
    PARTHENON_FAIL("STS integrator requires diffusion to be enabled!");
  }

  // Getting the integrator & time ratio between hyperbolic and parabolic terms
  std::string sts_integrator = pin->GetOrAddString("sts", "integrator", "none");
  Real sts_max_dt_ratio  = pin->GetOrAddReal("sts","sts_max_dt_ratio", -1.0);

  STSInt Diff_integrator = STSInt::null;
  if (sts_integrator == "rkl1") {
    Diff_integrator = STSInt::rkl1;
  } else if (sts_integrator == "rkl2") {
    PARTHENON_FAIL("rkl2 STS integrator not implemented!");
    Diff_integrator = STSInt::rkl2;
  } else {
    PARTHENON_FAIL("STS integrator not recognized!");
  }

  artemis_pkg->AddParam("sts_integrator", sts_integrator);
  artemis_pkg->AddParam("sts_max_dt_ratio", sts_max_dt_ratio);
  
}

//----------------------------------------------------------------------------------------
//! \fn STSRKL1
//! \brief Assembles the tasks for the STS RKL1 integrator
// comment: Maybe it should be moved back to artemis_driver.cpp
void  STSRKL1( Mesh *pmesh, const Real time, Real dt, int nstages) {
  using namespace ::parthenon::Update;
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // Deep copy u0 into u1 for integrator logic
  auto &init_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = init_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
    tl.AddTask(none, ArtemisUtils::DeepCopyConservedData, u1.get(), u0.get());
  }

  for (int stage = 1; stage <= nstages; stage++) {
    // Set up the STS stage coefficients
    // v0 = Y_{j-1}
    // v1 = Y_{j-2}
    // gam1 = muj = (2.*j - 1.)/j;
    // gam0 = nuj = (1. - j)/j;
    // beta_dt/dt = muj_tilde = pm->muj*2./(std::pow(s, 2.) + s);
    Real muj = (1. - stage)/stage;
    Real nuj = (2.*stage - 1.)/stage;
    Real muj_tilde = (2.*stage - 1.)/stage * 2./(std::pow(nstages, 2.) + nstages);

    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);

      // Start looking for incoming messages (including for flux correction)
      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
      auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, u0);

      // Compute (gas) diffusive fluxes
      TaskID diff_flx = none;
      if (do_diffusion && do_gas) {
        auto zf = tl.AddTask(none, Gas::ZeroDiffusionFlux, u0.get());
        TaskID vflx = zf, tflx = zf;
        if (do_viscosity) vflx = tl.AddTask(zf, Gas::ViscousFlux<GEOM>, u0.get());
        if (do_conduction) tflx = tl.AddTask(zf | vflx, Gas::ThermalFlux<GEOM>, u0.get());
        diff_flx = vflx | tflx;
      }

      // TODO Dust diffusion fluxes

      // Communicate and set fluxes
      auto send_flx =
          tl.AddTask(diff_flx,
                    parthenon::SendBoundBufs<parthenon::BoundaryType::flxcor_send>, u0);
      auto recv_flx = tl.AddTask(start_flx_recv, parthenon::ReceiveFluxCorrections, u0);
      auto set_flx = tl.AddTask(recv_flx, parthenon::SetFluxCorrections, u0);

      // Apply flux divergence, STS need stage to 0 for the sts ceofficients
      auto update =
          tl.AddTask(set_flx, RKL1FluxUpadte,
                     u0.get(), u1.get(),
                     dt, muj, nuj, muj_tilde);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn STSRKL2FirstStage
//! \brief Assembles the tasks for first stage of the STS RKL2 integrator
void STSRKL2FirstStage( Mesh *pm, const Real time, Real dt, int nstages) {
  // TODO: Implement RKL2 STS integration
}



//----------------------------------------------------------------------------------------
//! \fn STSRKL2SecondStage
//! \brief Assembles the tasks for first stage of the STS RKL2 integrator
void STSRKL2SecondStage( Mesh *pm, const Real time, Real dt, int nstages) {
  // TODO: Implement RKL2 STS integration
}


//----------------------------------------------------------------------------------------
//! \fn RKL1FluxUpadte
//! \brief Applies the STS RKL1 update to the conserved variables
template <Coordinates GEOM>
TaskStatus RKL1FluxUpadte(MeshData<Real> *u0, MeshData<Real> *u1, const Real dt,
                        const Real muj, const Real nuj, const Real muj_tilde) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();

  // Packing and indexing
  std::vector<MetadataFlag> flags({Metadata::Conserved, Metadata::WithFluxes});
  static auto desc = MakePackDescriptor<any>(u0, flags, {parthenon::PDOpt::WithFluxes});
  const auto v0 = desc.GetPack(u0);
  const auto v1 = desc.GetPack(u1);
  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RKL1FluxUpadte", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0.GetCoordinates(b), k, j, i);

        const auto ax1 = coords.GetFaceAreaX1();
        const auto ax2 = (multi_d) ? coords.GetFaceAreaX2() : NewArray<Real, 2>(0.0);
        const auto ax3 = (three_d) ? coords.GetFaceAreaX3() : NewArray<Real, 2>(0.0);

        const Real vol = coords.Volume();

        for (int n = v0.GetLowerBound(b); n <= v0.GetUpperBound(b); ++n) {
          // compute flux divergence
          Real divf = (ax1[0] * v0.flux(b, X1DIR, n, k, j, i) -
                       ax1[1] * v0.flux(b, X1DIR, n, k, j, i + 1));
          if (multi_d)
            divf += (ax2[0] * v0.flux(b, X2DIR, n, k, j, i) -
                     ax2[1] * v0.flux(b, X2DIR, n, k, j + 1, i));
          if (three_d)
            divf += (ax3[0] * v0.flux(b, X3DIR, n, k, j, i) -
                     ax3[1] * v0.flux(b, X3DIR, n, k + 1, j, i));

          //----------------------------------------------------------------------------------------
          // Apply STS RKL1 update
          // Y_{m} = nuj*Y_{j-2} + muj*Y_{j-1} + dt_sts*muj_tilde*M(Y_{j-1})
          //       = nuj*Y_{j-2} + muj*Y_{j-1} + dt_sts*muj_tilde*(divf/vol)
          // v0 = Y_{j-1}, v1 = Y_{j-2}
          Real Y_jm2 = v0(b, n, k, j, i);
          v0(b, n, k, j, i) =
              nuj * v1(b, n, k, j, i) + muj * v0(b, n, k, j, i) + divf * dt * muj_tilde/ vol;
          
          // ----------------------------------------------------------------------------------------
          // Rearrange the variables for the next step
          v1(b, n, k, j, i) = Y_jm2;
        }
      });
  return TaskStatus::complete;
}

}