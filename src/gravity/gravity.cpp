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

// Parthenon includes
#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/solver_utils.hpp>
#include <solvers/tridiag_solver.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "gravity/gravity.hpp"
#include "gravity/nbody_gravity.hpp"
#include "gravity/poisson_equation.hpp"

namespace Gravity {

using namespace parthenon::BoundaryFunction;

struct any_poisson : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_poisson(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "grav[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto Zero() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_poisson>(rc, coarse, 0.0);
  };
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Gravity::Initialize
//! \brief Adds intialization function for gravity package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            const ArtemisUtils::Constants &constants,
                                            const Packages_t &packages) {
  auto gravity = std::make_shared<StateDescriptor>("gravity");
  Params &params = gravity->AllParams();

  // Problem dimensionality
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);

  // Start and stop time for gravity (if applicable)
  params.Add("tstart",
             pin->GetOrAddReal("gravity", "tstart", std::numeric_limits<Real>::lowest()));
  params.Add("tstop", pin->GetOrAddReal("gravity", "tstop", Big<Real>()));

  // Find which gravity type is requested
  // Note that we do not use if-elif statements to detect multiple gravity blocks
  int count = 0;
  GravityType gtype = GravityType::null;
  std::string block_name = "none";

  // params specific to the gravity type
  if (pin->DoesBlockExist("gravity/uniform")) {
    count++;
    block_name = "gravity/uniform";
    gtype = GravityType::uniform;
    params.Add("gx1", pin->GetReal(block_name, "gx1"));
    params.Add("gx2", pin->GetReal(block_name, "gx2"));
    params.Add("gx3", pin->GetReal(block_name, "gx3"));
  }
  if (pin->DoesBlockExist("gravity/point")) {
    count++;
    gtype = GravityType::point;
    block_name = "gravity/point";
    const Real m = pin->GetReal(block_name, "mass");
    params.Add("mass", m);
    params.Add("gm", constants.GetGCode() * m);
    params.Add("soft", pin->GetOrAddReal(block_name, "soft", 0.0));
    params.Add("sink", pin->GetOrAddReal(block_name, "sink", 0.0));
    params.Add("sink_rate", pin->GetOrAddReal(block_name, "sink_rate", 0.0));
    const Real x = pin->GetOrAddReal(block_name, "x", 0.0);
    const Real y = pin->GetOrAddReal(block_name, "y", 0.0);
    const Real z = pin->GetOrAddReal(block_name, "z", 0.0);

    if (geometry::is_axisymmetric(coords)) {
      PARTHENON_REQUIRE(
          (x == 0.0) && (y == 0.0) && (z == 0.0),
          "In axisymmetric coordinates, the point mass must be at the origin!");
    }
    params.Add("x", x);
    params.Add("y", y);
    params.Add("z", z);
  }
  if (pin->DoesBlockExist("gravity/binary")) {
    count++;
    gtype = GravityType::binary;
    block_name = "gravity/binary";

    PARTHENON_REQUIRE(!geometry::is_axisymmetric(coords),
                      "Binary gravity is not compatable with axisymmetric coordinates!");

    const Real m = pin->GetReal(block_name, "mass");
    params.Add("mass", m);
    const Real gm = constants.GetGCode() * m;
    params.Add("gm", gm);
    params.Add("soft1", pin->GetOrAddReal(block_name, "soft1", 0.0));
    params.Add("soft2", pin->GetOrAddReal(block_name, "soft2", 0.0));
    params.Add("sink1", pin->GetOrAddReal(block_name, "sink1", 0.0));
    params.Add("sink2", pin->GetOrAddReal(block_name, "sink2", 0.0));
    params.Add("sink_rate1", pin->GetOrAddReal(block_name, "sink_rate1", 0.0));
    params.Add("sink_rate2", pin->GetOrAddReal(block_name, "sink_rate2", 0.0));
    params.Add("x", pin->GetOrAddReal(block_name, "x", 0.0));
    params.Add("y", pin->GetOrAddReal(block_name, "y", 0.0));
    params.Add("z", pin->GetOrAddReal(block_name, "z", 0.0));

    const Real qbin = pin->GetReal(block_name, "q");
    const Real abin = pin->GetReal(block_name, "a");
    const Real ebin = pin->GetOrAddReal(block_name, "e", 0.0);
    const Real ibin = pin->GetOrAddReal(block_name, "i", 0.0) * M_PI / 180.;
    const Real obin = pin->GetOrAddReal(block_name, "omega", 0.0) * M_PI / 180.;
    const Real Obin = pin->GetOrAddReal(block_name, "Omega", 0.0) * M_PI / 180.;
    const Real fbin = pin->GetOrAddReal(block_name, "f", 180.0) * M_PI / 180.;
    Orbit orb(gm, abin, ebin, ibin, obin, Obin, fbin);
    params.Add("q", qbin);
    params.Add("orb", orb);
  }
  if (pin->DoesBlockExist("gravity/nbody")) {
    count++;
    gtype = GravityType::nbody;
    block_name = "gravity/nbody";
    const bool do_nbody = pin->GetBoolean("physics", "nbody");
    PARTHENON_REQUIRE(do_nbody, "You have <gravity/nbody> but not physics/nbody = true!");
    auto &nbody_pkg = packages.Get("nbody");
    params.Add("gm", nbody_pkg->Param<Real>("gm"));
  }
  if (pin->DoesBlockExist("gravity/self")) {
    count++;
    gtype = GravityType::self;
    block_name = "gravity/self";
    PARTHENON_REQUIRE(coords == Coordinates::cartesian,
                      "Self-gravity currently only supports Cartesian coordinates");

    // Units and constants
    const Real gcode = constants.GetGCode();
    Real four_pi_G = 4.0 * M_PI * gcode;
    if (pin->GetOrAddBoolean(block_name, "units_override", false)) {
      four_pi_G = pin->GetOrAddReal(block_name, "four_pi_G", 1.0);
    }
    params.Add("four_pi_G", four_pi_G);

    // Jeans swindle
    bool pi1 = (pin->GetOrAddString("parthenon/mesh", "ix1_bc", "outflow") == "periodic");
    bool po1 = (pin->GetOrAddString("parthenon/mesh", "ox1_bc", "outflow") == "periodic");
    bool pi2 = (pin->GetOrAddString("parthenon/mesh", "ix2_bc", "outflow") == "periodic");
    bool po2 = (pin->GetOrAddString("parthenon/mesh", "ox2_bc", "outflow") == "periodic");
    bool pi3 = (pin->GetOrAddString("parthenon/mesh", "ix3_bc", "outflow") == "periodic");
    bool po3 = (pin->GetOrAddString("parthenon/mesh", "ox3_bc", "outflow") == "periodic");
    const bool needs_swindle = (pi1 && pi2 && pi3 && po1 && po2 && po3);
    const bool swindle = pin->GetOrAddBoolean(block_name, "use_swindle", needs_swindle);
    params.Add("use_swindle", swindle);
    PARTHENON_REQUIRE(swindle || !(needs_swindle),
                      "Fully periodic BCs require Jeans swindle!");
    if (swindle && !(needs_swindle)) {
      PARTHENON_WARN("Invoking Jeans swindle when not mandated by BCs");
    }

    // Gravity Package FillDerived function
    gravity->FillDerivedMesh = FillPoissonRHS<Coordinates::cartesian>;

    // Enroll ZeroBC
    using BF = parthenon::BoundaryFace;
    constexpr auto IN = BCSide::Inner;
    constexpr auto OUT = BCSide::Outer;
    const bool zi1 = (pin->GetOrAddString(block_name, "ix1_bc", "default") == "zero");
    const bool zo1 = (pin->GetOrAddString(block_name, "ox1_bc", "default") == "zero");
    const bool zi2 = (pin->GetOrAddString(block_name, "ix2_bc", "default") == "zero");
    const bool zo2 = (pin->GetOrAddString(block_name, "ox2_bc", "default") == "zero");
    const bool zi3 = (pin->GetOrAddString(block_name, "ix3_bc", "default") == "zero");
    const bool zo3 = (pin->GetOrAddString(block_name, "ox3_bc", "default") == "zero");
    if (zi1) gravity->UserBoundaryFunctions[BF::inner_x1].push_back(Zero<X1DIR, IN>());
    if (zo1) gravity->UserBoundaryFunctions[BF::inner_x2].push_back(Zero<X2DIR, IN>());
    if (zi2) gravity->UserBoundaryFunctions[BF::inner_x3].push_back(Zero<X3DIR, IN>());
    if (zo2) gravity->UserBoundaryFunctions[BF::outer_x1].push_back(Zero<X1DIR, OUT>());
    if (zi3) gravity->UserBoundaryFunctions[BF::outer_x2].push_back(Zero<X2DIR, OUT>());
    if (zo3) gravity->UserBoundaryFunctions[BF::outer_x3].push_back(Zero<X3DIR, OUT>());

    // Gravitational potential
    using namespace parthenon::refinement_ops;
    std::vector<MetadataFlag> flags{Metadata::Cell,        Metadata::Independent,
                                    Metadata::FillGhost,   Metadata::WithFluxes,
                                    Metadata::GMGRestrict, Metadata::GMGProlongate};
    Metadata m = Metadata(flags);
    m.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    gravity->AddField<grav::phi>(m);

    // 4piG * \Sum rho
    auto mrhs = Metadata(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    gravity->AddField<grav::rhs>(mrhs);

    // Multigrid
    using PoissEq = PoissonEquation<grav::phi>;
    PoissEq eq(pin, "poisson");
    params.Add("poisson_equation", eq, parthenon::Params::Mutability::Mutable);

    std::shared_ptr<parthenon::solvers::SolverBase> psolver;
    using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
    using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
    psolver =
        std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            "base", "phi", "rhs", pin, block_name, PoissEq(pin, block_name));
    // psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
    //     "base", "phi", "rhs", pin, block_name, PoissEq(pin, block_name));

    params.Add("solver_pointer", psolver);
  }

  PARTHENON_REQUIRE((count > 0) && (gtype != GravityType::null), "Unknown gravity node!");
  PARTHENON_REQUIRE(count == 1, "artemis only supports 1 gravity type at this time");

  params.Add("type", gtype);

  return gravity;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::ExternalGravity
//! \brief Wrapper function for external gravity options
template <Coordinates GEOM>
TaskStatus ExternalGravity(MeshData<Real> *md, const Real time, const Real dt) {
  auto pm = md->GetParentPointer();

  auto &pkg = pm->packages.Get("gravity");
  GravityType gtype = pkg->template Param<GravityType>("type");
  const Real tstart = pkg->template Param<Real>("tstart");
  const Real tstop = pkg->template Param<Real>("tstop");

  if ((time >= tstart) && (time < tstop)) {
    if (gtype == GravityType::uniform) {
      return UniformGravity<GEOM>(md, time, dt);
    } else if (gtype == GravityType::point) {
      return PointMassGravity<GEOM>(md, time, dt);
    } else if (gtype == GravityType::binary) {
      if constexpr (!geometry::is_axisymmetric<GEOM>()) {
        return BinaryMassGravity<GEOM>(md, time, dt);
      }
    } else if (gtype == GravityType::nbody) {
      if constexpr (!geometry::is_axisymmetric<GEOM>()) {
        auto &pkg = pm->packages.Get("nbody");
        if (pkg->template Param<int>("npart") > 0)
          return NBodyGravity<GEOM>(md, time, dt);
      }
    } else if (gtype == GravityType::self) {
      return SelfGravity<GEOM>(md, time, dt);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::ExternalGravity
//! \brief Wrapper function for external gravity options
template <Coordinates GEOM>
void FillPoissonRHS(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Extract artemis parameters
  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::prim::density, dust::prim::density, grav::rhs>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const int nblocks = md->NumBlocks();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);

  // Compute mean density
  Real grav_mean_rho = 0.0;
  auto &grav_pkg = pm->packages.Get("gravity");
  const bool use_swindle = grav_pkg->template Param<bool>("use_swindle");
  if (use_swindle) {
    Real total_mass = 0.0;
    Real total_volume = artemis_pkg->template Param<Real>("domain_volume");
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Gravity::TotalMass",
        parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      Real &mtot) {
          geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
          const Real vv = coords.Volume();
          for (int n = 0; n < do_gas * vmesh.GetSize(b, gas::prim::density()); ++n) {
            mtot += vmesh(b, gas::prim::density(n), k, j, i) * vv;
          }
          for (int n = 0; n < do_dust * vmesh.GetSize(b, dust::prim::density()); ++n) {
            mtot += vmesh(b, dust::prim::density(n), k, j, i) * vv;
          }
        },
        Kokkos::Sum<Real>(total_mass));
    Kokkos::fence();

#ifdef MPI_PARALLEL
    // NOTE(@pdmullen): This reduction only works because we require pack_size==-1...
    MPI_Allreduce(MPI_IN_PLACE, &total_mass, 1, MPI_PARTHENON_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    grav_mean_rho = total_mass / total_volume;
  }

  // Set RHS
  const Real four_pi_G = grav_pkg->template Param<Real>("four_pi_G");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetRHS", parthenon::DevExecSpace(), 0, nblocks - 1, kbe.s,
      kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        Real &rhs = vmesh(b, grav::rhs(), k, j, i) = 0.0;
        for (int n = 0; n < do_gas * vmesh.GetSize(b, gas::prim::density()); ++n) {
          rhs += vmesh(b, gas::prim::density(n), k, j, i);
        }
        for (int n = 0; n < do_dust * vmesh.GetSize(b, dust::prim::density()); ++n) {
          rhs += vmesh(b, dust::prim::density(n), k, j, i);
        }
        rhs -= use_swindle * grav_mean_rho;
        rhs *= four_pi_G;
      });
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef MeshData<Real> MD;
template TaskStatus ExternalGravity<G::cartesian>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<G::cylindrical>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<G::spherical1D>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<G::spherical2D>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<G::spherical3D>(MD *m, const Real t, const Real dt);
template TaskStatus ExternalGravity<G::axisymmetric>(MD *m, const Real t, const Real dt);
template void FillPoissonRHS<G::cartesian>(MD *m);
template void FillPoissonRHS<G::cylindrical>(MD *m);
template void FillPoissonRHS<G::spherical1D>(MD *m);
template void FillPoissonRHS<G::spherical2D>(MD *m);
template void FillPoissonRHS<G::spherical3D>(MD *m);
template void FillPoissonRHS<G::axisymmetric>(MD *m);

} // namespace Gravity
