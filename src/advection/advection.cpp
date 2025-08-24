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
// Artemis includes
#include "advection.hpp"
#include "artemis.hpp"
#include "utils/artemis_utils.hpp"

namespace Advection {

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RotatingFrame::Initialize
//! \brief Adds intialization function for advection package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto adv_pkg = std::make_shared<StateDescriptor>("advection");
  Params &params = adv_pkg->AllParams();

  const Real omega = pin->GetReal("advection", "omega");
  const Real qshear = pin->GetOrAddReal("advection", "qshear", 0.0);
  PARTHENON_REQUIRE(omega != 0.0, "advection/omega cannot be zero! To disable, set "
                                  "physics/advection = false");
  if (qshear != 0) {
    const std::string sys = pin->GetString("artemis", "coordinates");
    PARTHENON_REQUIRE(
        sys == "cartesian",
        "advection/qshear must be zero for non-Cartesian coordinate systems!");
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                      "Rotating frame advection step requires at least 2 ghost cells.");
  }
  params.Add("omega", omega);
  params.Add("qshear", qshear);

  // Linear advection timestep controls
  const Real cfl = pin->GetOrAddReal("advection", "cfl", 0.9);
  const Real dt_ratio = pin->GetOrAddReal("advection", "dt_ratio", 100.0);
  params.Add("cfl", cfl);
  params.Add("dt_ratio", dt_ratio);

  // Reconstruction algorithm for remap
  ReconstructionMethod recon_method = ReconstructionMethod::null;
  const std::string recon = pin->GetOrAddString("advection", "reconstruct", "plm");
  recon_method = ArtemisUtils::ChooseReconMethod(recon);
  params.Add("recon", recon_method);

  // Coordinates
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);
  params.Add("coords", coords);

  // Rotating frame timestep (if do_shear)
  if (coords == Coordinates::cartesian) {
    adv_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cartesian>;
  } else if (coords == Coordinates::spherical1D) {
    adv_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical1D>;
  } else if (coords == Coordinates::spherical2D) {
    adv_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical2D>;
  } else if (coords == Coordinates::spherical3D) {
    adv_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical3D>;
  } else if (coords == Coordinates::cylindrical) {
    adv_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cylindrical>;
  } else if (coords == Coordinates::axisymmetric) {
    adv_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::axisymmetric>;
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }

  return adv_pkg;
}
//----------------------------------------------------------------------------------------
//! \fn Real RotatingFrame::EstimateTimestep
//! \brief Compute multiple of linear advection timestep (if do_shear)
template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md) {
  auto pmesh = md->GetParentPointer();
  const bool do_shear = pmesh->packages.Get("artemis")->template Param<bool>("do_shear");
  if (!(do_shear)) return Big<Real>();

  auto &adv_pkg = pmesh->packages.Get("advection");
  const Real &dt_ratio = adv_pkg->template Param<Real>("dt_ratio");
  return EstimateTimestep(pmesh, dt_ratio);
}

//----------------------------------------------------------------------------------------
//! \fn Real RotatingFrame::EstimateTimeStep
//! \brief Not enrolled in parthenon's determination for global dt
Real EstimateTimestep(parthenon::Mesh *pmesh, const Real dt_ratio) {
  // Extract rotating frame params
  auto &adv_pkg = pmesh->packages.Get("advection");
  const Real &om0 = adv_pkg->Param<Real>("omega");
  const Real &qshear = adv_pkg->Param<Real>("qshear");
  const Real &cfl = adv_pkg->template Param<Real>("cfl");

  // Compute linear advection timestep to sub-cycle
  Real min_dt = Big<Real>();
  for (auto const &pmb : pmesh->block_list) {
    const auto &reg = pmb->block_size;
    const auto wp = BackgroundVelocity<Coordinates::cartesian>(qshear, om0, reg.xmax_[0]);
    const auto wm = BackgroundVelocity<Coordinates::cartesian>(qshear, om0, reg.xmin_[0]);
    const Real dx2 = (reg.xmax_[1] - reg.xmin_[1]) / reg.nx_[1];
    min_dt = std::min(min_dt, dx2 / std::max(std::abs(wp[1]), std::abs(wm[1])));
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &min_dt, 1, MPI_PARTHENON_REAL, MPI_MIN,
                                    MPI_COMM_WORLD));
#endif
  return cfl * min_dt * dt_ratio;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef MeshData<Real> MD;
template Real EstimateTimestepMesh<Coordinates::cartesian>(MD *md);
template Real EstimateTimestepMesh<Coordinates::cylindrical>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical1D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical2D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::spherical3D>(MD *md);
template Real EstimateTimestepMesh<Coordinates::axisymmetric>(MD *md);

} // namespace Advection