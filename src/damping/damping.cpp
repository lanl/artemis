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

// Artemis includes
#include "damping/damping.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

using ArtemisUtils::VI;

namespace Damping {
//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor Damping::Initialize
//! \brief Adds intialization function for damping package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto damping = std::make_shared<StateDescriptor>("damping");
  Params &params = damping->AllParams();

  // Below we store the bounds of the Mesh and parameters that define the bounds of the
  // damping zones and the damping rates
  //
  // We damp in the X* (* = 1,2,3) direction between
  //          x*min <= x* <= inner_x*, at the rate inner_x*_rate
  // and   outer_x* <= x* <=    x*max, at the rate outer_x*_rate
  params.Add("x1min", pin->GetReal("parthenon/mesh", "x1min"));
  params.Add("x2min", pin->GetReal("parthenon/mesh", "x2min"));
  params.Add("x3min", pin->GetReal("parthenon/mesh", "x3min"));
  params.Add("x1max", pin->GetReal("parthenon/mesh", "x1max"));
  params.Add("x2max", pin->GetReal("parthenon/mesh", "x2max"));
  params.Add("x3max", pin->GetReal("parthenon/mesh", "x3max"));

  const Real inner_x1 = pin->GetOrAddReal("damping", "inner_x1", -Big<Real>());
  const Real outer_x1 = pin->GetOrAddReal("damping", "outer_x1", Big<Real>());
  const Real inner_x1_rate = pin->GetOrAddReal("damping", "inner_x1_rate", 0.);
  const Real outer_x1_rate = pin->GetOrAddReal("damping", "outer_x1_rate", 0.);
  params.Add("inner_x1", inner_x1);
  params.Add("inner_x1_rate", inner_x1_rate);
  params.Add("outer_x1", outer_x1);
  params.Add("outer_x1_rate", outer_x1_rate);

  const Real inner_x2 = pin->GetOrAddReal("damping", "inner_x2", -Big<Real>());
  const Real outer_x2 = pin->GetOrAddReal("damping", "outer_x2", Big<Real>());
  const Real inner_x2_rate = pin->GetOrAddReal("damping", "inner_x2_rate", 0.);
  const Real outer_x2_rate = pin->GetOrAddReal("damping", "outer_x2_rate", 0.);
  params.Add("inner_x2", inner_x2);
  params.Add("inner_x2_rate", inner_x2_rate);
  params.Add("outer_x2", outer_x2);
  params.Add("outer_x2_rate", outer_x2_rate);

  const Real inner_x3 = pin->GetOrAddReal("damping", "inner_x3", -Big<Real>());
  const Real outer_x3 = pin->GetOrAddReal("damping", "outer_x3", Big<Real>());
  const Real inner_x3_rate = pin->GetOrAddReal("damping", "inner_x3_rate", 0.);
  const Real outer_x3_rate = pin->GetOrAddReal("damping", "outer_x3_rate", 0.);
  params.Add("inner_x3", inner_x3);
  params.Add("outer_x3", outer_x3);
  params.Add("inner_x3_rate", inner_x3_rate);
  params.Add("outer_x3_rate", outer_x3_rate);

  // Various checks
  PARTHENON_REQUIRE(inner_x1_rate >= 0.0,
                    "The damping rate in the x1 direction must be >= 0");
  PARTHENON_REQUIRE(inner_x1 <= outer_x1,
                    "The damping bounds must have inner_x1 <= outer_x1");
  PARTHENON_REQUIRE(inner_x2_rate >= 0.0,
                    "The damping rate in the x2 direction must be >= 0");
  PARTHENON_REQUIRE(inner_x2 <= outer_x2,
                    "The damping bounds must have inner_x2 <= outer_x2");
  PARTHENON_REQUIRE(inner_x3_rate >= 0.0,
                    "The damping rate in the x3 direction must be >= 0");
  PARTHENON_REQUIRE(inner_x3 <= outer_x3,
                    "The damping bounds must have inner_x3 <= outer_x3");

  return damping;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus Damping::DampingSource
//! \brief Wrapper function for external damping options
//!
//!   Backward Euler step
//!     d(v)/dt = - v_x/td
//!     vp = v - dt/td * vp
//!     vp = v/(1 + dt/td)
//!     vp - v = - dt v / ( td + dt)
//!
//!   Because density doesn't change, the momentum change is just rho * (vp-v)
//!
//!   The damping regions for a 2D domain would look like this
//!    +----------------+
//!    |xxxxxxxxxxxxxxxx|
//!    |xx+----------+xx|
//!    |xx|          |xx|
//!    |xx+----------+xx|
//!    |xxxxxxxxxxxxxxxx|
//!    +----------------+
template <Coordinates GEOM>
TaskStatus DampingSource(MeshData<Real> *md, const Real time, const Real dt) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");

  auto &damping_pkg = pm->packages.Get("damping");
  const int ndim = pm->ndim;
  const Real x1min = damping_pkg->template Param<Real>("x1min");
  const Real x1max = damping_pkg->template Param<Real>("x1max");
  const Real ix1 = damping_pkg->template Param<Real>("inner_x1");
  const Real ox1 = damping_pkg->template Param<Real>("outer_x1");
  const Real ix1_rate = damping_pkg->template Param<Real>("inner_x1_rate");
  const Real ox1_rate = damping_pkg->template Param<Real>("outer_x1_rate");

  const Real x2min = damping_pkg->template Param<Real>("x2min");
  const Real x2max = damping_pkg->template Param<Real>("x2max");
  const Real ix2 = damping_pkg->template Param<Real>("inner_x2");
  const Real ox2 = damping_pkg->template Param<Real>("outer_x2");
  const Real ix2_rate = (ndim >= 2) * damping_pkg->template Param<Real>("inner_x2_rate");
  const Real ox2_rate = (ndim >= 2) * damping_pkg->template Param<Real>("outer_x2_rate");

  const Real x3min = damping_pkg->template Param<Real>("x3min");
  const Real x3max = damping_pkg->template Param<Real>("x3max");
  const Real ix3 = damping_pkg->template Param<Real>("inner_x3");
  const Real ox3 = damping_pkg->template Param<Real>("outer_x3");
  const Real ix3_rate = (ndim == 3) * damping_pkg->template Param<Real>("inner_x3_rate");
  const Real ox3_rate = (ndim == 3) * damping_pkg->template Param<Real>("outer_x3_rate");

  static auto desc =
      MakePackDescriptor<gas::cons::total_energy, gas::cons::momentum, gas::prim::density,
                         gas::prim::velocity, dust::cons::momentum, dust::prim::density,
                         dust::prim::velocity>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "VelocityDamping", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        Real xv[3] = {Null<Real>()};
        Real hx[3] = {Null<Real>()};
        coords.GetCellCenter(xv);
        coords.GetScaleFactors(hx);

        // Compute the ramp for this cell
        // Ramps are quadratic, eg. the left regions is SQR( (X - ix)/(ix - xmin) )
        const Real fx1 =
            dt * (ix1_rate * ((xv[0] < ix1) * SQR((xv[0] - ix1) / (ix1 - x1min))) +
                  ox1_rate * ((xv[0] > ox1) * SQR((xv[0] - ox1) / (ox1 - x1max))));
        const Real fx2 =
            dt * (ix2_rate * ((xv[1] < ix2) * SQR((xv[1] - ix2) / (ix2 - x2min))) +
                  ox2_rate * ((xv[1] > ox2) * SQR((xv[1] - ox2) / (ox2 - x2max))));
        const Real fx3 =
            dt * (ix3_rate * ((xv[2] < ix3) * SQR((xv[2] - ix3) / (ix3 - x3min))) +
                  ox3_rate * ((xv[2] > ox3) * SQR((xv[2] - ox3) / (ox3 - x3max))));
        if (do_gas) {
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            const Real &dens = vmesh(b, gas::prim::density(n), k, j, i);
            const Real v[3] = {vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i),
                               vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i),
                               vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i)};

            // Ep - E = 0.5 d ( vp^2 - v^2 )
            //  (vp-v) . (vp + v) = dv . (2v + dv) =  2 dv.v + dv.dv
            const Real dv1 = -fx1 * v[0] / (1.0 + fx1);
            const Real dv2 = -fx2 * v[1] / (1.0 + fx2);
            const Real dv3 = -fx3 * v[2] / (1.0 + fx3);
            vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) += dv1 * dens * hx[0];
            vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) += dv2 * dens * hx[1];
            vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) += dv3 * dens * hx[2];
            vmesh(b, gas::cons::total_energy(n), k, j, i) +=
                dens * (0.5 * (SQR(dv1) + SQR(dv2) + SQR(dv3)) +
                        (dv1 * v[0] + dv2 * v[1] + dv3 * v[2]));
          }
        }
        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            const Real &dens = vmesh(b, dust::prim::density(n), k, j, i);
            const Real v[3] = {vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i),
                               vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i),
                               vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i)};

            const Real dv1 = -fx1 * v[0] / (1.0 + fx1);
            const Real dv2 = -fx2 * v[1] / (1.0 + fx2);
            const Real dv3 = -fx3 * v[2] / (1.0 + fx3);
            vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) += dv1 * dens * hx[0];
            vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) += dv2 * dens * hx[1];
            vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) += dv3 * dens * hx[2];
          }
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshData<Real> MD;
template TaskStatus DampingSource<C::cartesian>(MD *md, const Real tt, const Real dt);
template TaskStatus DampingSource<C::cylindrical>(MD *md, const Real tt, const Real dt);
template TaskStatus DampingSource<C::spherical>(MD *md, const Real tt, const Real dt);
template TaskStatus DampingSource<C::axisymmetric>(MD *md, const Real tt, const Real dt);

} // namespace Damping
