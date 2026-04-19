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

// Artemis includes
#include "rotating_frame.hpp"
#include "artemis.hpp"
#include "rotating_frame_impl.hpp"
#include "utils/artemis_utils.hpp"

namespace RotatingFrame {

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RotatingFrame::Initialize
//! \brief Adds intialization function for rotating frame package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto rframe_pkg = std::make_shared<StateDescriptor>("rotating_frame");
  Params &params = rframe_pkg->AllParams();

  const Real omega = pin->GetOrAddReal("rotating_frame", "omega", 0.0);
  const Real qshear = pin->GetOrAddReal("rotating_frame", "qshear", 0.0);
  const bool do_rotating_frame = pin->GetOrAddBoolean("physics", "rotating_frame", false);
  if (do_rotating_frame) {
    PARTHENON_REQUIRE(omega != 0.0,
                      "rotating_frame/omega cannot be zero! To disable, set "
                      "physics/rotating_frame = false");
  }

  // Coordinates
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);

  // qshear is only valid for cartesian (shearing box)
  if (qshear != 0) {
    PARTHENON_REQUIRE(
        sys == "cartesian",
        "rotating_frame/qshear must be zero for non-Cartesian coordinate systems!");
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                      "Rotating frame advection step requires at least 2 ghost cells.");
  }

  // GM for orbital advection in curvilinear coordinates
  Real gm = pin->GetOrAddReal("rotating_frame", "gm", 0.0);
  const bool do_orbital_advection =
      pin->GetOrAddBoolean("physics", "orbital_advection", false);
  if (do_orbital_advection) {
    if (geometry::is_cartesian(coords)) {
      PARTHENON_REQUIRE(qshear > 0.0, "Cartesian orbital advection requires qshear > 0!");
      PARTHENON_REQUIRE(do_rotating_frame,
                        "Cartesian orbital advection requires rotating_frame = true!");
    } else {
      PARTHENON_REQUIRE(
          coords != Coordinates::spherical1D,
          "Orbital advection cannot be used with 1D spherical coordinates!");
      PARTHENON_REQUIRE(gm > 0.0,
                        "Orbital advection in curvilinear coordinates requires gm > 0!");
    }
  }

  params.Add("omega", omega);
  params.Add("qshear", qshear);
  params.Add("gm", gm);
  params.Add("do_orbital_advection", do_orbital_advection);

  // Linear advection timestep controls
  const Real cfl = pin->GetOrAddReal("rotating_frame", "cfl", 0.9);
  const Real dt_ratio = pin->GetOrAddReal("rotating_frame", "dt_ratio", 100.0);
  params.Add("cfl", cfl);
  params.Add("dt_ratio", dt_ratio);

  // Reconstruction algorithm for remap
  ReconstructionMethod recon_method = ReconstructionMethod::null;
  const std::string recon = pin->GetOrAddString("rotating_frame", "reconstruct", "plm");
  recon_method = ArtemisUtils::ChooseReconMethod(recon);
  params.Add("recon", recon_method);

  params.Add("coords", coords);

  // Rotating frame / orbital advection timestep
  if (coords == Coordinates::cartesian) {
    rframe_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cartesian>;
  } else if (coords == Coordinates::spherical1D) {
    rframe_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical1D>;
  } else if (coords == Coordinates::spherical2D) {
    rframe_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical2D>;
  } else if (coords == Coordinates::spherical3D) {
    rframe_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::spherical3D>;
  } else if (coords == Coordinates::cylindrical) {
    rframe_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::cylindrical>;
  } else if (coords == Coordinates::axisymmetric) {
    rframe_pkg->EstimateTimestepMesh = EstimateTimestepMesh<Coordinates::axisymmetric>;
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }

  return rframe_pkg;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus RotatingFrame::RotatingFrameForce
//! \brief
TaskStatus RotatingFrameForce(MeshData<Real> *md, const Real time, const Real dt) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const auto coords = artemis_pkg->Param<Coordinates>("coords");

  auto &rframe_pkg = pm->packages.Get("rotating_frame");
  const Real om0 = rframe_pkg->Param<Real>("omega");
  const Real qshear = rframe_pkg->Param<Real>("qshear");

  // Switch for the different implementations based on coordinate system
  if (coords == Coordinates::cartesian) {
    return ShearingBoxImpl(md, om0, qshear, do_gas, do_dust, dt);
  } else if (coords == Coordinates::axisymmetric) {
    return RotatingFrameImpl<Coordinates::axisymmetric>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::spherical1D) {
    return RotatingFrameImpl<Coordinates::spherical1D>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::spherical2D) {
    return RotatingFrameImpl<Coordinates::spherical2D>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::spherical3D) {
    return RotatingFrameImpl<Coordinates::spherical3D>(md, om0, do_gas, do_dust, dt);
  } else if (coords == Coordinates::cylindrical) {
    return RotatingFrameImpl<Coordinates::cylindrical>(md, om0, do_gas, do_dust, dt);
  } else {
    PARTHENON_FAIL("Rotating frame is not consistent with this coordinate system");
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Real RotatingFrame::EstimateTimestep
//! \brief Compute multiple of linear advection timestep (if do_orbital_advection)
template <Coordinates GEOM>
Real EstimateTimestepMesh(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  auto pmesh = md->GetParentPointer();
  const bool do_oa =
      pmesh->packages.Get("artemis")->template Param<bool>("do_orbital_advection");
  if (!do_oa) return Big<Real>();

  auto &rframe_pkg = pmesh->packages.Get("rotating_frame");
  const Real &dt_ratio = rframe_pkg->template Param<Real>("dt_ratio");
  return EstimateTimestep<GEOM>(pmesh, dt_ratio);
}

//----------------------------------------------------------------------------------------
//! \fn Real RotatingFrame::EstimateTimeStep
//! \brief Not enrolled in parthenon's determination for global dt
template <Coordinates GEOM>
Real EstimateTimestep(parthenon::Mesh *pmesh, const Real dt_ratio) {
  PARTHENON_INSTRUMENT
  // Extract rotating frame params
  auto &rframe_pkg = pmesh->packages.Get("rotating_frame");
  const Real &om0 = rframe_pkg->Param<Real>("omega");
  const Real &qshear = rframe_pkg->Param<Real>("qshear");
  const Real &gm = rframe_pkg->Param<Real>("gm");
  const Real &cfl = rframe_pkg->template Param<Real>("cfl");

  // Compute linear advection timestep to sub-cycle
  Real min_dt = Big<Real>();
  for (auto const &pmb : pmesh->block_list) {
    [[maybe_unused]] const auto &reg = pmb->block_size;
    if constexpr (GEOM == Coordinates::cartesian) {
      const auto wp = BackgroundVelocity<GEOM>(qshear, om0, gm, {reg.xmax_[0], 0.0, 0.0});
      const auto wm = BackgroundVelocity<GEOM>(qshear, om0, gm, {reg.xmin_[0], 0.0, 0.0});
      const Real dx2 = (reg.xmax_[1] - reg.xmin_[1]) / reg.nx_[1];
      min_dt = std::min(min_dt, dx2 / std::max(std::abs(wp[1]), std::abs(wm[1])));
    } else if constexpr (GEOM == Coordinates::cylindrical) {
      const Real Rmin = reg.xmin_[0];
      const Real dphi = (reg.xmax_[1] - reg.xmin_[1]) / reg.nx_[1];
      const Real omega_max = OmegaKep(gm, Rmin) - om0;
      min_dt = std::min(min_dt, dphi / std::abs(omega_max));
    } else if constexpr (GEOM == Coordinates::spherical3D) {
      const Real rmin = reg.xmin_[0];
      const Real dphi = (reg.xmax_[2] - reg.xmin_[2]) / reg.nx_[2];
      const Real omega_max = OmegaKep(gm, rmin) - om0;
      min_dt = std::min(min_dt, dphi / std::abs(omega_max));
    }
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

template Real EstimateTimestep<Coordinates::cartesian>(parthenon::Mesh *, const Real);
template Real EstimateTimestep<Coordinates::cylindrical>(parthenon::Mesh *, const Real);
template Real EstimateTimestep<Coordinates::spherical1D>(parthenon::Mesh *, const Real);
template Real EstimateTimestep<Coordinates::spherical2D>(parthenon::Mesh *, const Real);
template Real EstimateTimestep<Coordinates::spherical3D>(parthenon::Mesh *, const Real);
template Real EstimateTimestep<Coordinates::axisymmetric>(parthenon::Mesh *, const Real);

} // namespace RotatingFrame
