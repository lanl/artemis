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

// C++ headers
#include "artemis_utils.hpp"
#include "nbody/nbody_utils.hpp"
#include "units.hpp"

namespace ArtemisUtils {

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::PrintArtemisConfiguration
//! \brief
void PrintArtemisConfiguration(Packages_t &packages) {
  // Generate and print splash screen
  if (parthenon::Globals::my_rank == 0) {
    Params &params = packages.Get("artemis")->AllParams();

    // Extract select params
    const auto nx = params.Get<std::array<int, 3>>("prob_dim");
    const auto nb = params.Get<std::array<int, 3>>("mb_dim");
    const auto units = params.Get<Units>("units");
    const int nd = (nx[0] > 1) + (nx[1] > 1) + (nx[2] > 1);

    // NOTE(@pdmullen:) Below (and only below...), we permit line length violations so
    // that we can better see the format of the splash screen...
    // clang-format off
    std::string hfill(21, ' ');
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
    if (params.Get<bool>("do_imc")) msg += hfill + "IMC radiation\n";
    if (params.Get<bool>("do_moment")) msg += hfill + "Moment radiation\n";
    printf("\n=====================================================\n");
    printf("  ARTEMIS\n");
    printf("    name:            %s\n", params.Get<std::string>("job_name").c_str());
    printf("    problem:         %s\n", params.Get<std::string>("pgen_name").c_str());
    printf("    coordinates:     %dD %s\n", nd, params.Get<std::string>("coord_sys").c_str());
    printf("    integrator:      %s\n", params.Get<std::string>("integrator").c_str());
    printf("    MPI ranks:       %d\n", parthenon::Globals::nranks);
    printf("    dimensions:      %dx%dx%d\n", nx[0], nx[1], nx[2]);
    printf("    meshblock:       %dx%dx%d\n", nb[0], nb[1], nb[2]);
    printf("    Unit System:  %s\n", units.GetSystemName().c_str());
    printf("                  [L] = %.2e\n", units.GetLengthCodeToPhysical());
    printf("                  [M] = %.2e\n", units.GetMassCodeToPhysical());
    printf("                  [T] = %.2e\n", units.GetTimeCodeToPhysical());
    printf("                  [K] = %.2e\n", units.GetTemperatureCodeToPhysical());
    printf("    Active physics:  %s", msg.c_str());
    if (params.Get<bool>("do_nbody")) {
      auto nbody_pkg = packages.Get("nbody");
      auto particles = nbody_pkg->Param<ParArray1D<NBody::Particle>>("particles");
      auto particles_h = particles.GetHostMirrorAndCopy();
      auto npart = particles_h.size();
      printf("      %d NBody particle(s)\n", static_cast<int>(npart));
      printf("      |_\n");
      for (int n = 0; n < npart; n++) {
        auto &part = particles_h(n);
        printf("        Particle      %2d:\n", part.id);
        printf("        |            mass: %.2e\n", part.GM);
        printf("        |         coupled: %s\n", part.couple == 1 ? "yes" : "no");
        printf("        |            live: %s\n", part.live == 1 ? "yes" : "no");
        printf("        |       softening: %s\n", part.spline == 1 ? "spline" : "plummer");
        printf("        |          radius: %.2e\n", part.rs);
        printf("        | accretion rates: gamma=%.2e\n", part.gamma);
        printf("        |                   beta=%.2e\n", part.beta);
        printf("        |          radius: %.2e\n", part.racc);
        printf("        |        position: (%.2e,%.2e,%.2e)\n", part.pos[0], part.pos[1], part.pos[2]);
        printf("        |        velocity: (%.2e,%.2e,%.2e)\n", part.vel[0], part.vel[1], part.vel[2]);
        printf("        -----------------------------------------------\n");
      }
    }
    printf("=======================================================\n\n");
    // clang-format on
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::EnrollArtemisRefinementOps
//! \brief Registers custom prolongation and restriction operators on provided Metadata
void EnrollArtemisRefinementOps(parthenon::Metadata &m, Coordinates coords,
                                const bool log) {
  typedef Coordinates G;

  // log space
  if (coords == G::cartesian) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::cartesian, false>,
                            ArtemisUtils::RestrictAverage<G::cartesian, false>>();
  } else if (coords == G::spherical1D) {
    if (log) {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::spherical1D, true>,
                              ArtemisUtils::RestrictAverage<G::spherical1D, true>>();
    } else {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::spherical1D, false>,
                              ArtemisUtils::RestrictAverage<G::spherical1D, false>>();
    }
  } else if (coords == G::spherical2D) {
    if (log) {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::spherical2D, true>,
                              ArtemisUtils::RestrictAverage<G::spherical2D, true>>();
    } else {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::spherical2D, false>,
                              ArtemisUtils::RestrictAverage<G::spherical2D, false>>();
    }
  } else if (coords == G::spherical3D) {
    if (log) {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::spherical3D, true>,
                              ArtemisUtils::RestrictAverage<G::spherical3D, true>>();
    } else {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::spherical3D, false>,
                              ArtemisUtils::RestrictAverage<G::spherical3D, false>>();
    }
  } else if (coords == G::cylindrical) {
    if (log) {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::cylindrical, true>,
                              ArtemisUtils::RestrictAverage<G::cylindrical, true>>();
    } else {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::cylindrical, false>,
                              ArtemisUtils::RestrictAverage<G::cylindrical, false>>();
    }
  } else if (coords == G::axisymmetric) {
    if (log) {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<G::axisymmetric, true>,
                              ArtemisUtils::RestrictAverage<G::axisymmetric, true>>();
    } else {
      m.RegisterRefinementOps<
          ArtemisUtils::ProlongateSharedMinMod<G::axisymmetric, false>,
          ArtemisUtils::RestrictAverage<G::axisymmetric, false>>();
    }
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

ReconstructionMethod ChooseReconMethod(std::string recon) {
  if (recon.compare("pcm") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 1,
                      "PCM requires at least 1 ghost cell.");
    return ReconstructionMethod::pcm;
  } else if (recon.compare("plm") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 2,
                      "PLM requires at least 2 ghost cells.");
    return ReconstructionMethod::plm;
  } else if (recon.compare("ppm") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 3,
                      "PPM requires at least 3 ghost cells.");
    return ReconstructionMethod::ppm;
  }
  PARTHENON_FAIL("Reconstruction method not recognized.");
  return ReconstructionMethod::pcm;
}

bool FineNeighbor(MeshBlock *pmb) {
  bool has_finer = false;
  const int mylevel = pmb->loc.level();
  for (const auto &nb : pmb->neighbors) {
    if (nb.origin_loc.level() > mylevel) {
      has_finer = true;
      break;
    }
  }
  return has_finer;
}

} // namespace ArtemisUtils
