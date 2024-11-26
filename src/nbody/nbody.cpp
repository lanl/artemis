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

// Parthenon includes
#include <globals.hpp>

// REBOUND includes
extern "C" {
#define restrict __restrict__
#include "rebound.h"
}

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "nbody/nbody.hpp"
#include "nbody/nbody_utils.hpp"

using parthenon::MetadataFlag;

namespace NBody {

namespace RebAttrs {
Real PN;
Real c;
int include_pn2;
bool extras;
bool merge_on_collision;
} // namespace RebAttrs

void UserWorkBeforeRestartOutputMesh(Mesh *pmesh, ParameterInput *, SimTime &,
                                     OutputParameters *);

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor NBody::Initialize
//! \brief Adds intialization function for NBody package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto nbody = std::make_shared<StateDescriptor>("nbody");
  Params &params = nbody->AllParams();
  PARTHENON_REQUIRE(
      (std::is_same<Real, double>::value),
      "Must be using double precision in Parthenon for consistency with REBOUND");

  // Coordinates
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);

  PARTHENON_REQUIRE(!(geometry::is_axisymmetric(coords)),
                    "<nbody> does not work with axisymmetric coordinates");

  params.Add("coords", coords);

  // Rebound integrator
  std::string integrator = pin->GetOrAddString("nbody", "integrator", "ias15");
  params.Add("integrator", integrator);

  // Rebound timestep
  Real dt_reb = pin->GetOrAddReal("nbody", "dt", Big<Real>());
  if (integrator == "none") dt_reb = Big<Real>();
  params.Add("dt_reb", dt_reb);

  // Unit system for gravity
  const Real GM = pin->GetReal("gravity", "gm");
  params.Add("GM", GM);
  params.Add("mscale", pin->GetOrAddReal("nbody", "mscale", 1.0));

  // Frame specification
  // NOTE(ADM): Shearing box has Rf = {R0, 0,0}, shearing box has Vf = {0, Om0*R0,0},
  // NOTE(ADM): Shearing box global frame --> Omf^2 = GM/R0^3
  const bool global_frame = (pin->GetOrAddString("nbody", "frame", "global") == "global");
  const Real Omf = pin->GetOrAddReal("rotating_frame", "omega", 0.0);
  const Real qshear = pin->GetOrAddReal("rotating_frame", "qshear", 0.0);
  Real Rf[3] = {0.0};
  Real Vf[3] = {0.0};
  if (global_frame && (Omf != 0.0) && (qshear != 0.0)) {
    const Real R0 = std::pow(SQR(Omf) / GM, 1.0 / 3.0);
    Rf[0] = R0;
    Vf[1] = R0 * Omf;
  }
  params.Add("frame_correction", global_frame);

  // Extra forces
  RebAttrs::PN = pin->GetOrAddReal("nbody", "pn", 0);
  RebAttrs::include_pn2 = pin->GetOrAddInteger("nbody", "pn2_corr", 1);
  RebAttrs::extras = (RebAttrs::PN > 0);
  RebAttrs::c = (RebAttrs::extras)
                    ? pin->GetReal("nbody", "light_speed")
                    : pin->GetOrAddReal("nbody", "light_speed", Big<Real>());
  RebAttrs::merge_on_collision =
      pin->GetOrAddBoolean("nbody", "merge_on_collision", true);

  // Read the parameter file for the particles
  std::vector<int> particle_id;
  std::vector<Particle> particles_v;
  NBodySetup(pin, GM, Rf, Vf, particle_id, particles_v);
  const int npart = static_cast<int>(particles_v.size());
  params.Add("npart", npart);
  params.Add("particle_id", particle_id);

  // Create particles ParArray and initialize
  ParArray1D<Particle> particles("particles", npart);
  auto particles_h = particles.GetHostMirror();
  for (int i = 0; i < npart; i++) {
    particles_h(i) = particles_v[i];
  }
  particles.DeepCopy(particles_h);
  params.Add("particles", particles);

  // Create ParArrays for particle forces
  ParArray2D<Real> particle_force("particle_force", npart, 7);
  ParArray2D<Real> particle_force_step("particle_force_step", npart, 7);
  ParArray2D<Real> particle_force_tot("particle_force_tot", npart, 7);
  params.Add("particle_force", particle_force);
  params.Add("particle_force_step", particle_force_step);
  params.Add("particle_force_tot", particle_force_tot);

  // Create vector for Rebound restart
  std::vector<char> reb_sim_restart;
  params.Add("reb_sim_buffer", reb_sim_restart, Params::Mutability::Restart);

  // Output parameters
  int output_count = 0;
  ParArray2D<int> orbit_output_count("orbit_output_count", npart, npart);
  auto orbit_output_count_h = orbit_output_count.GetHostMirror();
  for (int i = 0; i < npart; i++) {
    for (int j = 0; j < npart; j++) {
      orbit_output_count_h(i, j) = 0;
    }
  }
  orbit_output_count.DeepCopy(orbit_output_count_h);
  params.Add("orbit_output_count", orbit_output_count, Params::Mutability::Restart);
  params.Add("output_base", pin->GetString("parthenon/job", "problem_id"));
  params.Add("tnext", 0.0, Params::Mutability::Restart);
  params.Add("dt_output", pin->GetOrAddReal("nbody", "dt_output", Big<Real>()));
  params.Add("output_count", output_count, Params::Mutability::Restart);
  params.Add("disable_outputs", pin->GetOrAddBoolean("nbody", "disable_outputs", false));

  // Build the rebound sim
  const Real box_size = pin->GetOrAddReal("nbody", "box_size", Big<Real>());
  struct reb_simulation *reb_sim = nullptr;
  reb_sim = reb_simulation_create();
  if (parthenon::Globals::my_rank == 0) {
    for (int i = 0; i < npart; i++) {
      struct reb_particle pl = {0};
      pl.hash = i + 1;
      pl.r = particles_v[i].radius;
      pl.m = particles_v[i].GM;
      pl.x = particles_v[i].pos[0];
      pl.y = particles_v[i].pos[1];
      pl.z = particles_v[i].pos[2];
      pl.vx = particles_v[i].vel[0];
      pl.vy = particles_v[i].vel[1];
      pl.vz = particles_v[i].vel[2];
      reb_simulation_add(reb_sim, pl);

      // Verify that what we added still lives
      struct reb_particle *pl2 = reb_simulation_particle_by_hash(reb_sim, i + 1);
      PARTHENON_REQUIRE(pl2->r == particles_v[i].radius,
                        "Particle radius is inconsistent at setup!");
      PARTHENON_REQUIRE(pl2->m == particles_v[i].GM,
                        "Particle mass is inconsistent at setup!");
      PARTHENON_REQUIRE(pl2->x == particles_v[i].pos[0],
                        "Particle x is inconsistent at setup!");
      PARTHENON_REQUIRE(pl2->y == particles_v[i].pos[1],
                        "Particle y is inconsistent at setup!");
      PARTHENON_REQUIRE(pl2->z == particles_v[i].pos[2],
                        "Particle z is inconsistent at setup!");
      PARTHENON_REQUIRE(pl2->vx == particles_v[i].vel[0],
                        "Particle vx is inconsistent at setup!");
      PARTHENON_REQUIRE(pl2->vy == particles_v[i].vel[1],
                        "Particle vy is inconsistent at setup!");
      PARTHENON_REQUIRE(pl2->vz == particles_v[i].vel[2],
                        "Particle vz is inconsistent at setup!");
    }

    reb_simulation_configure_box(reb_sim, box_size, 1, 1, 1);
    reb_sim->boundary = reb_simulation::REB_BOUNDARY_OPEN;
    reb_sim->collision = reb_simulation::REB_COLLISION_LINE;
    reb_sim->dt = dt_reb;
    if (RebAttrs::extras) reb_sim->force_is_velocity_dependent = 1;
    if (integrator == "whfast") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_WHFAST;
    } else if (integrator == "leapfrog") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_LEAPFROG;
    } else if (integrator == "janus") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_JANUS;
    } else if (integrator == "mercurius") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_MERCURIUS;
    } else if (integrator == "saba") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_SABA;
    } else if (integrator == "bs") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_BS;
    } else if (integrator == "ias15") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_IAS15;
    } else if (integrator == "none") {
      reb_sim->integrator = reb_simulation::REB_INTEGRATOR_NONE;
    } else {
      std::stringstream msg;
      msg << integrator << " is an invalid REBOUND integrator!";
      PARTHENON_FAIL(msg);
    }

    // Set the function pointers
    SetReboundPtrs(reb_sim);
  }
  params.Add("reb_sim", reb_sim, true);

  // Hydro integrator
  LowStorageIntegrator hydro_integ(pin);
  nbody->AddParam<LowStorageIntegrator>("hydro_integ", hydro_integ);

  // NBody timestep
  nbody->EstimateTimestepMesh = EstimateTimestepMesh;

  // NBody refinement criterion
  const std::string refine_type = pin->GetOrAddString("nbody", "refine_type", "none");
  params.Add("derefine_factor", pin->GetOrAddReal("nbody", "derefine_factor", 2.0));
  if (refine_type != "none") {
    const bool refine_distance = (refine_type == "distance");
    PARTHENON_REQUIRE(refine_distance,
                      "Only distance based criterion currently supported!");
    if (refine_distance) {
      if (coords == Coordinates::cartesian) {
        nbody->CheckRefinementBlock = DistanceRefinement<Coordinates::cartesian>;
      } else if (coords == Coordinates::spherical3D) {
        nbody->CheckRefinementBlock = DistanceRefinement<Coordinates::spherical3D>;
      } else if (coords == Coordinates::cylindrical) {
        nbody->CheckRefinementBlock = DistanceRefinement<Coordinates::cylindrical>;
      }
    }
  }

  // UserWorkBeforeRestartOutput
  nbody->UserWorkBeforeRestartOutputMesh = NBody::UserWorkBeforeRestartOutputMesh;

  return nbody;
}

//----------------------------------------------------------------------------------------
//! \fn  Real NBody::EstimateTimestepMesh
//! \brief Compute NBody timestep
Real EstimateTimestepMesh(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();

  auto &nbody_pkg = pm->packages.Get("nbody");
  auto &params = nbody_pkg->AllParams();
  return params.Get<Real>("dt_reb");
}

//----------------------------------------------------------------------------------------
//! \fn  AmrTag NBody::DistanceRefinement
//! \brief Distance based refinement criterion
template <Coordinates GEOM>
AmrTag DistanceRefinement(MeshBlockData<Real> *md) {
  auto pmb = md->GetBlockPointer();
  auto pm = pmb->pmy_mesh;
  auto &pco = pmb->coords;

  auto &nbody_pkg = pm->packages.Get("nbody");
  auto particles = nbody_pkg->Param<ParArray1D<NBody::Particle>>("particles");
  const auto npart = static_cast<int>(particles.size());
  const auto derefine_factor = nbody_pkg->Param<Real>("derefine_factor");

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  Real min_dist = Big<Real>();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &ldist) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &x = coords.GetCellCenter();

        const auto &xcart = coords.ConvertToCart(x);

        // Each particle returns the distance normalized to it's target radius
        for (int n = 0; n < npart; n++) {
          ldist = std::min(ldist, particles(n).refine_distance(xcart));
        }
      },
      Kokkos::Min<Real>(min_dist));

  if (min_dist <= 1.0) return parthenon::AmrTag::refine;
  if (min_dist <= derefine_factor) return parthenon::AmrTag::same;
  return parthenon::AmrTag::derefine;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::UserWorkBeforeRestartOutputMesh
//! \brief Create REBOUND restart file and store in Params to reuse as Parthenon restart
void UserWorkBeforeRestartOutputMesh(Mesh *pmesh, ParameterInput *, SimTime &,
                                     OutputParameters *) {
  auto &artemis_pkg = pmesh->packages.Get("artemis");
  if (!(artemis_pkg->Param<bool>("do_nbody"))) return;

  // Extract Rebound simulation
  auto &nbody_pkg = pmesh->packages.Get("nbody");
  auto reb_sim = nbody_pkg->Param<struct reb_simulation *>("reb_sim");

  // Write native Rebound restart
  if (Globals::my_rank == 0) {
    // Delete temporary REBOUND output if it exists
    if (FILE *file = std::fopen(NBody::rebound_filename.c_str(), "r")) {
      std::fclose(file);
      if (std::remove(NBody::rebound_filename.c_str()) != 0) {
        PARTHENON_FAIL("Unable to delete temporary REBOUND file!");
      }
    }
    reb_simulation_save_to_file(reb_sim, NBody::rebound_filename.c_str());
  }

#ifdef MPI_PARALLEL
  // Ensure the file is available for all ranks
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Read Rebound restart back into string
  std::ifstream file(NBody::rebound_filename, std::ios::binary);
  PARTHENON_REQUIRE(file.is_open(), "Error opening temporary rebound output file!");
  std::vector<char> reb_sim_buffer((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
  file.close();

  // Store current rebound output as restartable parameter.  Every rank must store a
  // matching buffer parameter or else I/O will hang
  nbody_pkg->UpdateParam<std::vector<char>>("reb_sim_buffer", reb_sim_buffer);
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::InitializeFromRestart
//! \brief On restart event, load REBOUND state using stored save file
void InitializeFromRestart(Mesh *pm) {
  auto &artemis_pkg = pm->packages.Get("artemis");
  if (!(artemis_pkg->Param<bool>("do_nbody"))) return;

  // Extract Rebound parameters
  auto &nbody_pkg = pm->packages.Get("nbody");
  auto reb_sim = nbody_pkg->Param<struct reb_simulation *>("reb_sim");
  auto particle_id = nbody_pkg->Param<std::vector<int>>("particle_id");
  auto particles = nbody_pkg->Param<ParArray1D<NBody::Particle>>("particles");

  // Initialize rebound state on rank 0
  if (Globals::my_rank == 0) {
    // Create rebound save file from stored buffer
    auto reb_sim_buffer = nbody_pkg->Param<std::vector<char>>("reb_sim_buffer");
    std::ofstream outfile(NBody::rebound_filename.c_str(), std::ios::binary);
    outfile.write(reb_sim_buffer.data(), reb_sim_buffer.size());
    outfile.close();

    // Create rebound simulation from new save file
    char *reb_filename = new char[NBody::rebound_filename.size() + 1];
    std::strcpy(reb_filename, NBody::rebound_filename.c_str());
    auto new_reb_sim = reb_simulation_create_from_file(reb_filename, -1);
    reb_simulation_free(reb_sim);
    SetReboundPtrs(new_reb_sim);
    nbody_pkg->UpdateParam<struct reb_simulation *>("reb_sim", new_reb_sim);
  }

  // Send restarted rebound particles to all nodes
  SyncWithRebound(reb_sim, particle_id, particles);
}

//----------------------------------------------------------------------------------------
//! template instantiations
template AmrTag DistanceRefinement<Coordinates::cartesian>(MeshBlockData<Real> *md);
template AmrTag DistanceRefinement<Coordinates::cylindrical>(MeshBlockData<Real> *md);
template AmrTag DistanceRefinement<Coordinates::spherical3D>(MeshBlockData<Real> *md);

} // namespace NBody
