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

#include <vector>

// Parthenon includes
#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/solver_utils.hpp>

// Artemis includes
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "self_gravity/poisson_equation.hpp"
#include "self_gravity/self_gravity.hpp"
#include "utils/artemis_utils.hpp"

namespace SelfGravity {

using namespace parthenon::BoundaryFunction;
using namespace parthenon::package::prelude;
using ArtemisUtils::VI;

struct any_poisson : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_poisson(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "grav[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto DirZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_poisson>(rc, coarse, 0.0);
  };
}

template <CoordinateDirection DIR, BCSide SIDE>
auto NeuZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::Outflow, any_poisson>(rc, coarse, 0.0);
  };
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Gravity::Initialize
//! \brief Adds intialization function for the self-gravity package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            const ArtemisUtils::Constants &constants,
                                            const Packages_t &packages) {
  auto self_gravity = std::make_shared<StateDescriptor>("self_gravity");
  Params &params = self_gravity->AllParams();
  std::string block_name = "self_gravity";

  // Check coordinates compatibility
  const int ndim = ProblemDimension(pin);
  std::string sys = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  Coordinates coords = geometry::CoordSelect(sys, ndim);
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
  bool pi2 = (pin->GetOrAddString("parthenon/mesh", "ix2_bc", "outflow") == "periodic");
  bool pi3 = (pin->GetOrAddString("parthenon/mesh", "ix3_bc", "outflow") == "periodic");
  bool po1 = (pin->GetOrAddString("parthenon/mesh", "ox1_bc", "outflow") == "periodic");
  bool po2 = (pin->GetOrAddString("parthenon/mesh", "ox2_bc", "outflow") == "periodic");
  bool po3 = (pin->GetOrAddString("parthenon/mesh", "ox3_bc", "outflow") == "periodic");
  const bool needs_swindle = (pi1 && pi2 && pi3 && po1 && po2 && po3);
  const bool swindle = pin->GetOrAddBoolean(block_name, "use_swindle", needs_swindle);
  params.Add("use_swindle", swindle);
  PARTHENON_REQUIRE(swindle || !(needs_swindle),
                    "Fully periodic BCs require Jeans swindle!");
  if (swindle && !(needs_swindle)) {
    PARTHENON_WARN("Invoking Jeans swindle when not mandated by BCs");
  }

  // self_gravity Package FillDerived function
  self_gravity->FillDerivedMesh = FillPoissonRHS<Coordinates::cartesian>;

  // Enroll DirZ or NeuZ BCs if specified
  using BF = parthenon::BoundaryFace;
  constexpr auto LL = BCSide::Inner;
  constexpr auto RR = BCSide::Outer;
  const std::string grav_bci1 = pin->GetOrAddString(block_name, "ix1_bc", "default");
  const std::string grav_bci2 = pin->GetOrAddString(block_name, "ix2_bc", "default");
  const std::string grav_bci3 = pin->GetOrAddString(block_name, "ix3_bc", "default");
  const std::string grav_bco1 = pin->GetOrAddString(block_name, "ox1_bc", "default");
  const std::string grav_bco2 = pin->GetOrAddString(block_name, "ox2_bc", "default");
  const std::string grav_bco3 = pin->GetOrAddString(block_name, "ox3_bc", "default");

  // Check validity
  auto valid_grav_bc = [](const std::string &s) {
    return s == "default" || s == "zero" || s == "neumann";
  };
  PARTHENON_REQUIRE(valid_grav_bc(grav_bci1), "Unsupported IX1 Poisson BC: " + grav_bci1);
  PARTHENON_REQUIRE(valid_grav_bc(grav_bci2), "Unsupported IX2 Poisson BC: " + grav_bci2);
  PARTHENON_REQUIRE(valid_grav_bc(grav_bci3), "Unsupported IX3 Poisson BC: " + grav_bci3);
  PARTHENON_REQUIRE(valid_grav_bc(grav_bco1), "Unsupported OX1 Poisson BC: " + grav_bco1);
  PARTHENON_REQUIRE(valid_grav_bc(grav_bco2), "Unsupported OX2 Poisson BC: " + grav_bco2);
  PARTHENON_REQUIRE(valid_grav_bc(grav_bco3), "Unsupported OX3 Poisson BC: " + grav_bco3);

  // Dirchlet (phi = 0) BC enrollment
  const bool di1 = (grav_bci1 == "zero");
  const bool do1 = (grav_bco1 == "zero");
  const bool di2 = (grav_bci2 == "zero");
  const bool do2 = (grav_bco2 == "zero");
  const bool di3 = (grav_bci3 == "zero");
  const bool do3 = (grav_bco3 == "zero");
  if (di1) self_gravity->UserBoundaryFunctions[BF::inner_x1].push_back(DirZ<X1DIR, LL>());
  if (di2) self_gravity->UserBoundaryFunctions[BF::inner_x2].push_back(DirZ<X2DIR, LL>());
  if (di3) self_gravity->UserBoundaryFunctions[BF::inner_x3].push_back(DirZ<X3DIR, LL>());
  if (do1) self_gravity->UserBoundaryFunctions[BF::outer_x1].push_back(DirZ<X1DIR, RR>());
  if (do2) self_gravity->UserBoundaryFunctions[BF::outer_x2].push_back(DirZ<X2DIR, RR>());
  if (do3) self_gravity->UserBoundaryFunctions[BF::outer_x3].push_back(DirZ<X3DIR, RR>());

  // Neumann (dphi/dn = 0) BC enrollment
  const bool ni1 = (grav_bci1 == "neumann");
  const bool no1 = (grav_bco1 == "neumann");
  const bool ni2 = (grav_bci2 == "neumann");
  const bool no2 = (grav_bco2 == "neumann");
  const bool ni3 = (grav_bci3 == "neumann");
  const bool no3 = (grav_bco3 == "neumann");
  if (ni1) self_gravity->UserBoundaryFunctions[BF::inner_x1].push_back(NeuZ<X1DIR, LL>());
  if (ni2) self_gravity->UserBoundaryFunctions[BF::inner_x2].push_back(NeuZ<X2DIR, LL>());
  if (ni3) self_gravity->UserBoundaryFunctions[BF::inner_x3].push_back(NeuZ<X3DIR, LL>());
  if (no1) self_gravity->UserBoundaryFunctions[BF::outer_x1].push_back(NeuZ<X1DIR, RR>());
  if (no2) self_gravity->UserBoundaryFunctions[BF::outer_x2].push_back(NeuZ<X2DIR, RR>());
  if (no3) self_gravity->UserBoundaryFunctions[BF::outer_x3].push_back(NeuZ<X3DIR, RR>());

  // Gravitational potential
  using namespace parthenon::refinement_ops;
  std::vector<MetadataFlag> flags{Metadata::Cell,        Metadata::Independent,
                                  Metadata::FillGhost,   Metadata::WithFluxes,
                                  Metadata::GMGRestrict, Metadata::GMGProlongate};
  Metadata m = Metadata(flags);
  ArtemisUtils::EnrollArtemisRefinementOps(m, coords, false, false);
  self_gravity->AddField<grav::phi>(m);

  // 4piG * \Sum rho
  auto mrhs = Metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
  ArtemisUtils::EnrollArtemisRefinementOps(mrhs, coords, false, false);
  self_gravity->AddField<grav::rhs>(mrhs);

  // Solvers
  using PoissEq = PoissonEquation<Coordinates::cartesian, grav::phi>;
  PoissEq eq(pin, "poisson");
  params.Add("poisson_equation", eq, parthenon::Params::Mutability::Mutable);
  std::shared_ptr<parthenon::solvers::SolverBase> psolver;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
  psolver =
      std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
          "base", "phi", "rhs", pin, block_name, PoissEq(pin, block_name));

  params.Add("solver_pointer", psolver);

  // Enroll in tstart/tstop machinery
  ArtemisUtils::AddPackageTimeParams(params, block_name, pin);

  return self_gravity;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus SelfGravity::FillPoissonRHS
//! \brief Sets RHS of Poisson (accounting for Jeans swindle if fully periodic)
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
  auto &grav_pkg = pm->packages.Get("self_gravity");
  const bool use_swindle = grav_pkg->template Param<bool>("use_swindle");
  if (use_swindle) {
    Real total_mass = 0.0;
    Real total_volume = 0.0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "Gravity::TotalMass",
        parthenon::DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &lmtot,
                      Real &lvtot) {
          geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
          const Real vv = coords.Volume();
          lvtot += vv;
          for (int n = 0; n < do_gas * vmesh.GetSize(b, gas::prim::density()); ++n) {
            lmtot += vmesh(b, gas::prim::density(n), k, j, i) * vv;
          }
          for (int n = 0; n < do_dust * vmesh.GetSize(b, dust::prim::density()); ++n) {
            lmtot += vmesh(b, dust::prim::density(n), k, j, i) * vv;
          }
        },
        Kokkos::Sum<Real>(total_mass), Kokkos::Sum<Real>(total_volume));
    Kokkos::fence();

#ifdef MPI_PARALLEL
    // NOTE(@pdmullen): This reduction only works because we require pack_size==-1...
    Real values[2] = {total_mass, total_volume};
    MPI_Allreduce(MPI_IN_PLACE, values, 2, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD);
    total_mass = values[0];
    total_volume = values[1];
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
        rhs = four_pi_G * (rhs - grav_mean_rho); // grav_mean_rho = 0 when !use_swindle
      });
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Gravity::SelfGravity
//! \brief Applies accelerations due to a constant g
template <Coordinates GEOM>
TaskStatus SelfGravity(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  static auto desc =
      MakePackDescriptor<gas::cons::momentum, gas::cons::total_energy,
                         dust::cons::momentum, gas::prim::density, dust::prim::density,
                         grav::phi>(resolved_pkgs.get());
  static auto descf = MakePackDescriptor<gas::cons::density>(
      resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto vmesh = desc.GetPack(md);
  auto vflux = descf.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int d1 = X1DIR;
  const int d2 = d1 + multi_d;
  const int d3 = d2 + three_d;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SelfGravity", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vmesh.GetCoordinates(b), k, j, i);
        const auto &dx = coords.GetCellWidths();
        const Real hdtodx1 = (0.5 * dt / dx[0]);
        const Real hdtodx2 = multi_d * (0.5 * dt / dx[1]);
        const Real hdtodx3 = three_d * (0.5 * dt / dx[2]);

        // Potential differences
        const Real &phic = vmesh(b, grav::phi(), k, j, i);
        const Real dpl1 = -(phic - vmesh(b, grav::phi(), k, j, i - 1));
        const Real dpr1 = -(vmesh(b, grav::phi(), k, j, i + 1) - phic);
        const Real dpl2 = -(phic - vmesh(b, grav::phi(), k, j - multi_d, i));
        const Real dpr2 = -(vmesh(b, grav::phi(), k, j + multi_d, i) - phic);
        const Real dpl3 = -(phic - vmesh(b, grav::phi(), k - three_d, j, i));
        const Real dpr3 = -(vmesh(b, grav::phi(), k + three_d, j, i) - phic);

        // Shared b/w gas and dust
        const Real wdt1 = hdtodx1 * (dpl1 + dpr1);
        const Real wdt2 = hdtodx2 * (dpl2 + dpr2);
        const Real wdt3 = hdtodx3 * (dpl3 + dpr3);

        if (do_gas) {
          // Gravitational acceleration and energy release
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            const Real &rr = vmesh(b, gas::prim::density(n), k, j, i);
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += rr * wdt1;
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += rr * wdt2;
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += rr * wdt3;
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                hdtodx1 * (vflux.flux(b, d1, gas::cons::density(n), k, j, i) * dpl1 +
                           vflux.flux(b, d1, gas::cons::density(n), k, j, i + 1) * dpr1);
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                hdtodx2 *
                (vflux.flux(b, d2, gas::cons::density(n), k, j, i) * dpl2 +
                 vflux.flux(b, d2, gas::cons::density(n), k, j + multi_d, i) * dpr2);
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                hdtodx3 *
                (vflux.flux(b, d3, gas::cons::density(n), k, j, i) * dpl3 +
                 vflux.flux(b, d3, gas::cons::density(n), k + three_d, j, i) * dpr3);
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Gravitational acceleration
            const Real &rr = vmesh(b, dust::prim::density(n), k, j, i);
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) += rr * wdt1;
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) += rr * wdt2;
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) += rr * wdt3;
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates G;
typedef MeshData<Real> MD;
template void FillPoissonRHS<G::cartesian>(MD *m);
template void FillPoissonRHS<G::cylindrical>(MD *m);
template void FillPoissonRHS<G::spherical1D>(MD *m);
template void FillPoissonRHS<G::spherical2D>(MD *m);
template void FillPoissonRHS<G::spherical3D>(MD *m);
template void FillPoissonRHS<G::axisymmetric>(MD *m);
template TaskStatus SelfGravity<G::cartesian>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::cylindrical>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::spherical1D>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::spherical2D>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::spherical3D>(MD *m, const Real t, const Real d);
template TaskStatus SelfGravity<G::axisymmetric>(MD *m, const Real t, const Real d);

} // namespace SelfGravity
