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
#include "geometry/geometry.hpp"
#include "nbody/nbody_utils.hpp"
#include "units.hpp"

namespace ArtemisUtils {

namespace {

template <Coordinates GEOM>
Real ComputeLocalMaxAbsFaceDivBImpl(const BlockList_t &blocks,
                                    const geometry::CoordParams &cpars, const int ndim) {
  using TE = TopologicalElement;
  constexpr int f1 = static_cast<int>(TE::F1) % 3;
  constexpr int f2 = static_cast<int>(TE::F2) % 3;
  constexpr int f3 = static_cast<int>(TE::F3) % 3;
  constexpr Real tiny = 1.0e-300;

  Real max_divb = 0.0;
  for (const auto &pmb : blocks) {
    auto &md = pmb->meshblock_data.Get();
    if (!md->HasVariable("field.face.B")) continue;

    auto var = md->GetVarPtr("field.face.B");
    if (!var->IsAllocated()) continue;

    auto b = var->data;
    auto pco = pmb->coords;
    const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    Real block_max_divb = 0.0;
    if (ndim == 3) {
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), kb.s,
          kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmax_divb) {
            geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
            const Real vol = coords.Volume();
            const auto ax1 = coords.GetFaceAreaX1();
            const auto ax2 = coords.GetFaceAreaX2();
            const auto ax3 = coords.GetFaceAreaX3();
            const Real divb = ((ax1[1] * b(f1, 0, 0, 0, k, j, i + 1) -
                                ax1[0] * b(f1, 0, 0, 0, k, j, i)) +
                               (ax2[1] * b(f2, 0, 0, 0, k, j + 1, i) -
                                ax2[0] * b(f2, 0, 0, 0, k, j, i)) +
                               (ax3[1] * b(f3, 0, 0, 0, k + 1, j, i) -
                                ax3[0] * b(f3, 0, 0, 0, k, j, i))) /
                              (vol + tiny);
            const Real abs_divb = fabs(divb);
            if (abs_divb > lmax_divb) lmax_divb = abs_divb;
          },
          Kokkos::Max<Real>(block_max_divb));
    } else if (ndim == 2) {
      const int k = kb.s;
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), jb.s,
          jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int j, const int i, Real &lmax_divb) {
            geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
            const Real vol = coords.Volume();
            const auto ax1 = coords.GetFaceAreaX1();
            const auto ax2 = coords.GetFaceAreaX2();
            const Real divb = ((ax1[1] * b(f1, 0, 0, 0, k, j, i + 1) -
                                ax1[0] * b(f1, 0, 0, 0, k, j, i)) +
                               (ax2[1] * b(f2, 0, 0, 0, k, j + 1, i) -
                                ax2[0] * b(f2, 0, 0, 0, k, j, i))) /
                              (vol + tiny);
            const Real abs_divb = fabs(divb);
            if (abs_divb > lmax_divb) lmax_divb = abs_divb;
          },
          Kokkos::Max<Real>(block_max_divb));
    } else {
      const int k = kb.s;
      const int j = jb.s;
      parthenon::par_reduce(
          parthenon::loop_pattern_flatrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
          ib.s, ib.e,
          KOKKOS_LAMBDA(const int i, Real &lmax_divb) {
            geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
            const Real vol = coords.Volume();
            const auto ax1 = coords.GetFaceAreaX1();
            const Real divb = (ax1[1] * b(f1, 0, 0, 0, k, j, i + 1) -
                               ax1[0] * b(f1, 0, 0, 0, k, j, i)) /
                              (vol + tiny);
            const Real abs_divb = fabs(divb);
            if (abs_divb > lmax_divb) lmax_divb = abs_divb;
          },
          Kokkos::Max<Real>(block_max_divb));
    }

    max_divb = std::max(max_divb, block_max_divb);
  }

  return max_divb;
}

void PrintMeshMaxAbsDivB(Mesh *pm, const char *label) {
  auto artemis = pm->packages.Get("artemis");
  if (!artemis->Param<bool>("do_mhd")) return;

  const auto coords = artemis->Param<Coordinates>("coords");
  const auto &cpars = artemis->Param<geometry::CoordParams>("coord_params");
  const int ndim = artemis->Param<int>("ndim");

  Real max_divb = 0.0;
  if (coords == Coordinates::cartesian) {
    max_divb = ComputeLocalMaxAbsFaceDivBImpl<Coordinates::cartesian>(pm->block_list,
                                                                      cpars, ndim);
  } else if (coords == Coordinates::spherical1D) {
    max_divb = ComputeLocalMaxAbsFaceDivBImpl<Coordinates::spherical1D>(pm->block_list,
                                                                        cpars, ndim);
  } else if (coords == Coordinates::spherical2D) {
    max_divb = ComputeLocalMaxAbsFaceDivBImpl<Coordinates::spherical2D>(pm->block_list,
                                                                        cpars, ndim);
  } else if (coords == Coordinates::spherical3D) {
    max_divb = ComputeLocalMaxAbsFaceDivBImpl<Coordinates::spherical3D>(pm->block_list,
                                                                        cpars, ndim);
  } else if (coords == Coordinates::cylindrical) {
    max_divb = ComputeLocalMaxAbsFaceDivBImpl<Coordinates::cylindrical>(pm->block_list,
                                                                        cpars, ndim);
  } else if (coords == Coordinates::axisymmetric) {
    max_divb = ComputeLocalMaxAbsFaceDivBImpl<Coordinates::axisymmetric>(pm->block_list,
                                                                         cpars, ndim);
  } else {
    PARTHENON_FAIL("Invalid Artemis coordinate system for divB diagnostic.");
  }

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &max_divb, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif
  if (parthenon::Globals::my_rank == 0 && max_divb > 0.0) {
    std::cout << "AMR remesh: max |divB| [" << label << "] = " << std::scientific
              << max_divb << std::defaultfloat << std::endl;
  }
}

} // namespace

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
    if (!params.Get<bool>("update_fluxes")) msg += "(without flux updates)\n";
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

void PreStepDiagnosticsRemeshDivB(SimTime const &simtime, MeshData<Real> *rc) {
  auto pm = rc->GetMeshPointer();
  if ((simtime.ncycle > 0) && pm->modified) {
    PrintMeshMaxAbsDivB(pm, "post-remesh");
  }
}

void PostStepDiagnosticsRemeshDivB(SimTime const &, MeshData<Real> *rc) {
  PrintMeshMaxAbsDivB(rc->GetMeshPointer(), "pre-remesh");
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::EnrollArtemisRefinementOps
//! \brief Registers custom prolongation and restriction operators on provided Metadata
void EnrollArtemisRefinementOps(parthenon::Metadata &m, Coordinates coords,
                                const bool log, const bool use_minmod_slope) {
  typedef Coordinates G;

  if (coords == G::cartesian) {
    if (use_minmod_slope) {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateShared<G::cartesian, false, true>,
                              ArtemisUtils::RestrictAverage<G::cartesian, false>>();
    } else {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateShared<G::cartesian, false, false>,
                              ArtemisUtils::RestrictAverage<G::cartesian, false>>();
    }
  } else if (coords == G::spherical1D) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, true, true>,
            ArtemisUtils::RestrictAverage<G::spherical1D, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, true, false>,
            ArtemisUtils::RestrictAverage<G::spherical1D, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, false, true>,
            ArtemisUtils::RestrictAverage<G::spherical1D, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, false, false>,
            ArtemisUtils::RestrictAverage<G::spherical1D, false>>();
      }
    }
  } else if (coords == G::spherical2D) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, true, true>,
            ArtemisUtils::RestrictAverage<G::spherical2D, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, true, false>,
            ArtemisUtils::RestrictAverage<G::spherical2D, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, false, true>,
            ArtemisUtils::RestrictAverage<G::spherical2D, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, false, false>,
            ArtemisUtils::RestrictAverage<G::spherical2D, false>>();
      }
    }
  } else if (coords == G::spherical3D) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, true, true>,
            ArtemisUtils::RestrictAverage<G::spherical3D, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, true, false>,
            ArtemisUtils::RestrictAverage<G::spherical3D, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, false, true>,
            ArtemisUtils::RestrictAverage<G::spherical3D, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, false, false>,
            ArtemisUtils::RestrictAverage<G::spherical3D, false>>();
      }
    }
  } else if (coords == G::cylindrical) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, true, true>,
            ArtemisUtils::RestrictAverage<G::cylindrical, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, true, false>,
            ArtemisUtils::RestrictAverage<G::cylindrical, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, false, true>,
            ArtemisUtils::RestrictAverage<G::cylindrical, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, false, false>,
            ArtemisUtils::RestrictAverage<G::cylindrical, false>>();
      }
    }
  } else if (coords == G::axisymmetric) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, true, true>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, true, false>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, false, true>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, false, false>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, false>>();
      }
    }
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::EnrollArtemisFaceRefinementOps
//! \brief Registers custom face-centered prolongation and restriction operators.
void EnrollArtemisFaceRefinementOps(parthenon::Metadata &m, Coordinates coords,
                                    const bool log, const bool use_minmod_slope) {
  typedef Coordinates G;

  if (coords == G::cartesian) {
    if (use_minmod_slope) {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateShared<G::cartesian, false, true>,
                              ArtemisUtils::RestrictAverage<G::cartesian, false>,
                              ArtemisUtils::ProlongateTothAndRoe<G::cartesian, false>>();
    } else {
      m.RegisterRefinementOps<ArtemisUtils::ProlongateShared<G::cartesian, false, false>,
                              ArtemisUtils::RestrictAverage<G::cartesian, false>,
                              ArtemisUtils::ProlongateTothAndRoe<G::cartesian, false>>();
    }
  } else if (coords == G::spherical1D) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, true, true>,
            ArtemisUtils::RestrictAverage<G::spherical1D, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical1D, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, true, false>,
            ArtemisUtils::RestrictAverage<G::spherical1D, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical1D, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, false, true>,
            ArtemisUtils::RestrictAverage<G::spherical1D, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical1D, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical1D, false, false>,
            ArtemisUtils::RestrictAverage<G::spherical1D, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical1D, false>>();
      }
    }
  } else if (coords == G::spherical2D) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, true, true>,
            ArtemisUtils::RestrictAverage<G::spherical2D, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical2D, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, true, false>,
            ArtemisUtils::RestrictAverage<G::spherical2D, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical2D, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, false, true>,
            ArtemisUtils::RestrictAverage<G::spherical2D, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical2D, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical2D, false, false>,
            ArtemisUtils::RestrictAverage<G::spherical2D, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical2D, false>>();
      }
    }
  } else if (coords == G::spherical3D) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, true, true>,
            ArtemisUtils::RestrictAverage<G::spherical3D, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical3D, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, true, false>,
            ArtemisUtils::RestrictAverage<G::spherical3D, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical3D, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, false, true>,
            ArtemisUtils::RestrictAverage<G::spherical3D, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical3D, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::spherical3D, false, false>,
            ArtemisUtils::RestrictAverage<G::spherical3D, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::spherical3D, false>>();
      }
    }
  } else if (coords == G::cylindrical) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, true, true>,
            ArtemisUtils::RestrictAverage<G::cylindrical, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::cylindrical, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, true, false>,
            ArtemisUtils::RestrictAverage<G::cylindrical, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::cylindrical, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, false, true>,
            ArtemisUtils::RestrictAverage<G::cylindrical, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::cylindrical, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::cylindrical, false, false>,
            ArtemisUtils::RestrictAverage<G::cylindrical, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::cylindrical, false>>();
      }
    }
  } else if (coords == G::axisymmetric) {
    if (log) {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, true, true>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::axisymmetric, true>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, true, false>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, true>,
            ArtemisUtils::ProlongateTothAndRoe<G::axisymmetric, true>>();
      }
    } else {
      if (use_minmod_slope) {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, false, true>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::axisymmetric, false>>();
      } else {
        m.RegisterRefinementOps<
            ArtemisUtils::ProlongateShared<G::axisymmetric, false, false>,
            ArtemisUtils::RestrictAverage<G::axisymmetric, false>,
            ArtemisUtils::ProlongateTothAndRoe<G::axisymmetric, false>>();
      }
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
  } else if (recon.compare("wenoz") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 3,
                      "WENO-Z requires at least 3 ghost cells.");
    return ReconstructionMethod::wenoz;
  } else if (recon.compare("wenomz") == 0) {
    PARTHENON_REQUIRE(parthenon::Globals::nghost >= 3,
                      "WENO-MZ requires at least 3 ghost cells.");
    return ReconstructionMethod::wenomz;
  }
  PARTHENON_FAIL("Reconstruction method not recognized.");
  return ReconstructionMethod::pcm;
}

} // namespace ArtemisUtils
