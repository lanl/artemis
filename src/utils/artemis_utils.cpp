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
#include "artemis_utils.hpp"
#include "nbody/nbody_utils.hpp"

namespace ArtemisUtils {

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::PrintArtemisConfiguration
//! \brief
void PrintArtemisConfiguration(Packages_t &packages) {
  if (parthenon::Globals::my_rank == 0) {
    Params &params = packages.Get("artemis")->AllParams();
    std::string hfill(21, ' ');
    const auto nx = params.Get<std::array<int, 3>>("prob_dim");
    const int nd = (nx[0] > 1) + (nx[1] > 1) + (nx[2] > 1);
    const auto nb = params.Get<std::array<int, 3>>("mb_dim");
    std::string msg = "";
    if (params.Get<bool>("do_gas")) msg += "Gas\n";
    if (params.Get<bool>("do_dust")) msg += hfill + "Dust\n";
    if (params.Get<bool>("do_gravity")) msg += hfill + "Gravity\n";
    if (params.Get<bool>("do_rotating_frame")) msg += hfill + "Rotating frame\n";
    if (params.Get<bool>("do_cooling")) msg += hfill + "Cooling\n";
    if (params.Get<bool>("do_conduction")) msg += hfill + "Conduction\n";
    if (params.Get<bool>("do_viscosity")) msg += hfill + "Viscosity\n";
    if (params.Get<bool>("do_drag")) msg += hfill + "Drag\n";
    if (params.Get<bool>("do_nbody")) msg += hfill + "N-body\n";
    if (params.Get<bool>("do_radiation")) msg += hfill + "IMC radiation\n";
    printf("\n=======================================================\n");
    printf("  ARTEMIS\n");
    printf("    name:            %s\n", params.Get<std::string>("job_name").c_str());
    printf("    problem:         %s\n", params.Get<std::string>("pgen_name").c_str());
    printf("    coordinates:     %dD %s\n", nd,
           params.Get<std::string>("coord_sys").c_str());
    printf("    MPI ranks:       %d\n", parthenon::Globals::nranks);
    printf("    dimensions:      %dx%dx%d\n", nx[0], nx[1], nx[2]);
    printf("    meshblock:       %dx%dx%d\n", nb[0], nb[1], nb[2]);
    printf("    Active physics:  %s", msg.c_str());

    if (params.Get<bool>("do_nbody")) {

      auto nbody_pkg = packages.Get("nbody");
      auto particles = nbody_pkg->Param<ParArray1D<NBody::Particle>>("particles");
      auto particles_h = particles.GetHostMirrorAndCopy();
      auto npart = particles_h.size();
      printf("      %d NBody particle(s)\n", npart);
      printf("      |_\n");
      for (int n = 0; n < npart; n++) {
        auto &part = particles_h(n);
        printf("        Particle      %2d:\n", part.id);
        printf("        |            mass: %.2e\n", part.GM);
        printf("        |         coupled: %s\n", part.couple == 1 ? "yes" : "no");
        printf("        |            live: %s\n", part.live == 1 ? "yes" : "no");
        printf("        |       softening: %s\n",
               part.spline == 1 ? "spline" : "plummer");
        printf("        |          radius: %.2e\n", part.rs);
        printf("        | accretion rates: gamma=%.2e\n", part.gamma);
        printf("        |                   beta=%.2e\n", part.beta);
        printf("        |          radius: %.2e\n", part.racc);
        printf("        |        position: (%.2e,%.2e,%.2e)\n", part.pos[0], part.pos[1],
               part.pos[2]);
        printf("        |        velocity: (%.2e,%.2e,%.2e)\n", part.vel[0], part.vel[1],
               part.vel[2]);
        printf("        -----------------------------------------------\n");
      }
    }
    printf("=======================================================\n\n");
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::EnrollArtemisRefinementOps
//! \brief Registers custom prolongation and restriction operators on provided Metadata
void EnrollArtemisRefinementOps(parthenon::Metadata &m, Coordinates coords) {
  typedef Coordinates C;
  if (coords == C::cartesian) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::cartesian>,
                            ArtemisUtils::RestrictAverage<C::cartesian>>();
  } else if (coords == C::spherical1D) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::spherical1D>,
                            ArtemisUtils::RestrictAverage<C::spherical1D>>();
  } else if (coords == C::spherical2D) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::spherical2D>,
                            ArtemisUtils::RestrictAverage<C::spherical2D>>();
  } else if (coords == C::spherical3D) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::spherical3D>,
                            ArtemisUtils::RestrictAverage<C::spherical3D>>();
  } else if (coords == C::cylindrical) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::cylindrical>,
                            ArtemisUtils::RestrictAverage<C::cylindrical>>();
  } else if (coords == C::axisymmetric) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::axisymmetric>,
                            ArtemisUtils::RestrictAverage<C::axisymmetric>>();
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<std::vector<Real>> NBody::loadtxt
//! \brief
std::vector<std::vector<Real>> loadtxt(std::string fname) {
  // Open File
  std::fstream file{fname};
  if (!(file)) {
    std::stringstream msg;
    msg << "Cannot read file \"" << fname << "\"!";
    PARTHENON_FAIL(msg);
  }

  // Read and store in vector
  std::vector<std::vector<Real>> table;
  std::fstream ifs;
  ifs.open(fname);
  while (true) {
    std::string line;
    Real buf;
    getline(ifs, line);
    std::stringstream ss(line,
                         std::ios_base::out | std::ios_base::in | std::ios_base::binary);
    if (!ifs) break;
    if (line[0] == '#' || line.empty()) continue;
    std::vector<Real> row;
    while (ss >> buf) {
      row.push_back(buf);
    }
    table.push_back(row);
  }
  ifs.close();
  return table;
}

} // namespace ArtemisUtils
