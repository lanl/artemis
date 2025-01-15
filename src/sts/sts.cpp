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
#include "gas/gas.hpp"
#include "sts.hpp"
#include "utils/artemis_utils.hpp"

namespace STS{
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor STS ::Initialize
//! \brief Adds intialization function for STS package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {

  auto STS = std::make_shared<StateDescriptor>("STS");
  Params &params = STS->AllParams();

  // initial check of getting the physics needed for the sts integrator
  // Determine input file specified physics
  const bool do_gas = pin->GetOrAddBoolean("physics", "gas", true);
  const bool do_viscosity = pin->GetOrAddBoolean("physics", "viscosity", false);
  const bool do_conduction = pin->GetOrAddBoolean("physics", "conduction", false);
  const bool do_sts = pin->GetOrAddBoolean("physics", "sts", false);
  const bool do_diffusion = do_conduction || do_viscosity;

  params.Add("do_sts", do_sts);
  params.Add("do_gas", do_gas);
  params.Add("do_viscosity", do_viscosity);
  params.Add("do_conduction", do_conduction);
  params.Add("do_diffusion", do_diffusion);

  if (!do_diffusion) {
    PARTHENON_FAIL("STS integrator requires diffusion to be enabled!");
  }

  // Getting the integrator & time ratio between hyperbolic and parabolic terms
  std::string sts_integrator = pin->GetOrAddString("sts", "integrator", "none");
  Real sts_max_dt_ratio  = pin->GetOrAddReal("sts","sts_max_dt_ratio", -1.0);

  STSInt sts_integrator_param = STSInt::null;
  if (sts_integrator == "rkl1") {
    sts_integrator_param = STSInt::rkl1;
  } else if (sts_integrator == "rkl2") {
    PARTHENON_FAIL("rkl2 STS integrator not implemented!");
    sts_integrator_param = STSInt::rkl2;
  } else {
    PARTHENON_FAIL("STS integrator not recognized!");
  }

  params.Add("sts_integrator", sts_integrator_param);
  params.Add("sts_max_dt_ratio", sts_max_dt_ratio);
  
}

//----------------------------------------------------------------------------------------
//! \fn PreStepSTSTasks
//! \brief Executes the pre-step tasks for the STS integrator
template <Coordinates GEOM>
void PreStepSTSTasks(Mesh *pmesh, const Real time, Real dt, int nstages) {

  // Getting the integrator & time ratio between hyperbolic and parabolic terms
  auto &sts = pmesh->packages.Get("STS");
  const STSInt sts_integrator = sts->Param<STSInt>("sts_integrator");

  // Check if the integrator is set
  if (sts_integrator == STSInt::null) {
    PARTHENON_FAIL("STS integrator not set!");
  }

  // Execute the integrator tasks
  if (sts_integrator == STSInt::rkl1) {
    // (TODO) RKL1 : Full timestep dt_sts
    STSRKL1<GEOM>(pmesh, time, dt, nstages);
  } else if (sts_integrator == STSInt::rkl2) {
    // (TODO) RKL2 : // eq (21) using half hyperbolic timestep 
    // due to Strang split
    //STSRKL2FirstStage<GEOM>(pmesh, time, 0.5*dt, nstages);
  }
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

//----------------------------------------------------------------------------------------
//! \fn STSRKL2FirstStage
//! \brief Assembles the tasks for first stage of the STS RKL2 integrator
template <Coordinates GEOM>
void STSRKL2FirstStage( Mesh *pm, const Real time, Real dt, int nstages) {
  // TODO: Implement RKL2 STS integration
}

//----------------------------------------------------------------------------------------
//! \fn STSRKL2SecondStage
//! \brief Assembles the tasks for first stage of the STS RKL2 integrator
template <Coordinates GEOM>
void STSRKL2SecondStage( Mesh *pm, const Real time, Real dt, int nstages) {
  // TODO: Implement RKL2 STS integration
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Mesh M;
//RK1 template instantiations
template void STSRKL1<Coordinates::cartesian>(M *m, const Real time, Real dt, int nstages);
template void STSRKL1<Coordinates::cylindrical>( M *m, const Real time, Real dt, int nstages);
template void STSRKL1<Coordinates::spherical1D>( M *m, const Real time, Real dt, int nstages);
template void STSRKL1<Coordinates::spherical2D>( M *m, const Real time, Real dt, int nstages);
template void STSRKL1<Coordinates::spherical3D>( M *m, const Real time, Real dt, int nstages);
template void STSRKL1<Coordinates::axisymmetric>( M *m, const Real time, Real dt, int nstages);
//RK2 first stage template instantiations
template void STSRKL2FirstStage<Coordinates::cartesian>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<Coordinates::cylindrical>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<Coordinates::spherical1D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<Coordinates::spherical2D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<Coordinates::spherical3D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2FirstStage<Coordinates::axisymmetric>(M *m, const Real time, Real dt, int nstages);
//RK2 second stage template instantiations
template void STSRKL2SecondStage<Coordinates::cartesian>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<Coordinates::cylindrical>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<Coordinates::spherical1D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<Coordinates::spherical2D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<Coordinates::spherical3D>(M *m, const Real time, Real dt, int nstages);
template void STSRKL2SecondStage<Coordinates::axisymmetric>(M *m, const Real time, Real dt, int nstages);
//PreStepSTSTasks template instantiations
template void PreStepSTSTasks<Coordinates::cartesian>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<Coordinates::cylindrical>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<Coordinates::spherical1D>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<Coordinates::spherical2D>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<Coordinates::spherical3D>(M *m, const Real time, Real dt, int nstages);
template void PreStepSTSTasks<Coordinates::axisymmetric>(M *m,const Real time, Real dt, int nstages);
} // namespace STS