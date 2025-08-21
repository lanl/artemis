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
#include "artemis.hpp"

#include "geometry.hpp"

namespace geometry {

// helper macro
#define ADD_FIELD(name)                                                                  \
  {                                                                                      \
    const auto shape = coords.template shape<name>();                                    \
    pkg->AddField<name>(Metadata({Metadata::None, Metadata::OneCopy, Metadata::Restart}, \
                                 std::vector<int>({shape[0] * shape[1] * shape[2]})));   \
  }

template <Coordinates GEOM>
void EnrollFields(StateDescriptor *pkg, CoordParams &cpars) {

  Coords<GEOM> coords(cpars);

  ADD_FIELD(geom::x1v);
  ADD_FIELD(geom::x2v);
  ADD_FIELD(geom::x3v);
  ADD_FIELD(geom::hx1v);
  ADD_FIELD(geom::hx2v);
  ADD_FIELD(geom::hx3v);
  ADD_FIELD(geom::hx1f1);
  ADD_FIELD(geom::hx2f1);
  ADD_FIELD(geom::hx3f1);
  ADD_FIELD(geom::hx1f2);
  ADD_FIELD(geom::hx2f2);
  ADD_FIELD(geom::hx3f2);
  ADD_FIELD(geom::hx1f3);
  ADD_FIELD(geom::hx2f3);
  ADD_FIELD(geom::hx3f3);
  ADD_FIELD(geom::dx1);
  ADD_FIELD(geom::dx2);
  ADD_FIELD(geom::dx3);
  ADD_FIELD(geom::vol);
  ADD_FIELD(geom::ax1);
  ADD_FIELD(geom::ax2);
  ADD_FIELD(geom::ax3);
  ADD_FIELD(geom::dh1dx1);
  ADD_FIELD(geom::dh2dx1);
  ADD_FIELD(geom::dh3dx1);
  ADD_FIELD(geom::dh1dx2);
  ADD_FIELD(geom::dh2dx2);
  ADD_FIELD(geom::dh3dx2);
  ADD_FIELD(geom::dh1dx3);
  ADD_FIELD(geom::dh2dx3);
  ADD_FIELD(geom::dh3dx3);
  ADD_FIELD(geom::rfw1m);
  ADD_FIELD(geom::rfw1p);
  ADD_FIELD(geom::rfw2m);
  ADD_FIELD(geom::rfw2p);
  ADD_FIELD(geom::rfw3m);
  ADD_FIELD(geom::rfw3p);
}

//----------------------------------------------------------------------------------------
//! \fn  StateDescriptor RotatingFrame::Initialize
//! \brief Adds intialization function for rotating frame package
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto geom = std::make_shared<StateDescriptor>("geometry");
  Params &params = geom->AllParams();

  CoordParams cpars(pin);

  if (cpars.sys == Coordinates::cartesian) {
    EnrollFields<Coordinates::cartesian>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::axisymmetric) {
    EnrollFields<Coordinates::axisymmetric>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::cylindrical) {
    EnrollFields<Coordinates::cylindrical>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::spherical1D) {
    EnrollFields<Coordinates::spherical1D>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::spherical2D) {
    EnrollFields<Coordinates::spherical2D>(geom.get(), cpars);
  } else if (cpars.sys == Coordinates::spherical3D) {
    EnrollFields<Coordinates::spherical3D>(geom.get(), cpars);
  }

  return geom;
}

template <Coordinates GEOM>
void InitBlockGeom(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  auto pm = pmb->pmy_mesh;
  const int ndim = pm->ndim;
  auto &md = pmb->meshblock_data.Get();
  auto &artemis_pkg = pmb->packages.Get("artemis");
  auto &pco = pmb->coords;
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  static auto desc_g = MakePackDescriptor<
      geom::x1v, geom::x2v, geom::x3v, geom::dx1, geom::dx2, geom::dx3, geom::vol,
      geom::ax1, geom::ax2, geom::ax3, geom::hx1v, geom::hx2v, geom::hx3v, geom::dh1dx1,
      geom::dh2dx1, geom::dh3dx1, geom::dh1dx2, geom::dh2dx2, geom::dh3dx2, geom::dh1dx3,
      geom::dh2dx3, geom::dh3dx3, geom::rfw1m, geom::rfw1p, geom::rfw2m, geom::rfw2p,
      geom::rfw3m, geom::rfw3p, geom::hx1f1, geom::hx1f2, geom::hx1f3, geom::hx2f1,
      geom::hx2f2, geom::hx2f3, geom::hx3f1, geom::hx3f2, geom::hx3f3>(
      (pm->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  const bool x1dep_ = x1dep<GEOM>();
  const bool x2dep_ = x1dep<GEOM>() && (ndim > 1);
  const bool x3dep_ = x1dep<GEOM>() && (ndim > 2);
  const int b = 0;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry::InitBlock", parthenon::DevExecSpace(), kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto xv = coords.GetCellCenter();
        Real &x1v = vg(b, geom::x1v())(coords.template index<geom::x1v>(k, j, i));
        Real &x2v = vg(b, geom::x2v())(coords.template index<geom::x2v>(k, j, i));
        Real &x3v = vg(b, geom::x3v())(coords.template index<geom::x3v>(k, j, i));
        Kokkos::atomic_store(&x1v, xv[0]);
        Kokkos::atomic_store(&x2v, xv[1]);
        Kokkos::atomic_store(&x3v, xv[2]);

        const auto dx = coords.GetCellWidths();
        Real &dx1 = vg(b, geom::dx1())(coords.template index<geom::dx1>(k, j, i));
        Real &dx2 = vg(b, geom::dx2())(coords.template index<geom::dx2>(k, j, i));
        Real &dx3 = vg(b, geom::dx3())(coords.template index<geom::dx3>(k, j, i));
        Kokkos::atomic_store(&dx1, dx[0]);
        Kokkos::atomic_store(&dx2, dx[1]);
        Kokkos::atomic_store(&dx3, dx[2]);

        const auto hx = coords.GetScaleFactors();
        Real &hx1 = vg(b, geom::hx1v())(coords.template index<geom::hx1v>(k, j, i));
        Real &hx2 = vg(b, geom::hx2v())(coords.template index<geom::hx2v>(k, j, i));
        Real &hx3 = vg(b, geom::hx3v())(coords.template index<geom::hx3v>(k, j, i));
        Kokkos::atomic_store(&hx1, hx[0]);
        Kokkos::atomic_store(&hx2, hx[1]);
        Kokkos::atomic_store(&hx3, hx[2]);

        Real &vol = vg(b, geom::vol())(coords.template index<geom::vol>(k, j, i));
        Kokkos::atomic_store(&vol, coords.Volume());

        const auto &[rfw1, rfw2, rfw3] = coords.RFWeights();
        Real &rfw1m = vg(b, geom::rfw1m())(coords.template index<geom::rfw1m>(k, j, i));
        Real &rfw1p = vg(b, geom::rfw1p())(coords.template index<geom::rfw1p>(k, j, i));
        Real &rfw2m = vg(b, geom::rfw2m())(coords.template index<geom::rfw2m>(k, j, i));
        Real &rfw2p = vg(b, geom::rfw2p())(coords.template index<geom::rfw2p>(k, j, i));
        Real &rfw3m = vg(b, geom::rfw3m())(coords.template index<geom::rfw3m>(k, j, i));
        Real &rfw3p = vg(b, geom::rfw3p())(coords.template index<geom::rfw3p>(k, j, i));

        Kokkos::atomic_store(&rfw1m, rfw1[0]);
        Kokkos::atomic_store(&rfw1p, rfw1[1]);
        Kokkos::atomic_store(&rfw2m, (ndim > 1) * rfw2[0]);
        Kokkos::atomic_store(&rfw2p, (ndim > 1) * rfw2[1]);
        Kokkos::atomic_store(&rfw3m, (ndim > 2) * rfw3[0]);
        Kokkos::atomic_store(&rfw3p, (ndim > 2) * rfw3[1]);

        // Face quantities
        // Extra scope for simpler variable names
        {
          auto ax = coords.GetFaceAreaX1();
          Real &ax1 = vg(b, geom::ax1())(coords.template index<geom::ax1>(k, j, i));
          Kokkos::atomic_store(&ax1, ax[0]);
          auto xf = coords.FaceCenX1(CellFace::lower);
          Real &hx1f = vg(b, geom::hx1f1())(coords.template index<geom::hx1f1>(k, j, i));
          Real &hx2f = vg(b, geom::hx2f1())(coords.template index<geom::hx2f1>(k, j, i));
          Real &hx3f = vg(b, geom::hx3f1())(coords.template index<geom::hx3f1>(k, j, i));
          Kokkos::atomic_store(&hx1f, coords.hx1(xf[0], xf[1], xf[2]));
          Kokkos::atomic_store(&hx2f, coords.hx2(xf[0], xf[1], xf[2]));
          Kokkos::atomic_store(&hx3f, coords.hx3(xf[0], xf[1], xf[2]));

          if (i == ib.e) {
            Real &ax1 = vg(b, geom::ax1())(coords.template index<geom::ax1>(k, j, i + 1));
            Kokkos::atomic_store(&ax1, ax[1]);
            xf = coords.FaceCenX1(CellFace::upper);
            Real &hx1f =
                vg(b, geom::hx1f1())(coords.template index<geom::hx1f1>(k, j, i + 1));
            Real &hx2f =
                vg(b, geom::hx2f1())(coords.template index<geom::hx2f1>(k, j, i + 1));
            Real &hx3f =
                vg(b, geom::hx3f1())(coords.template index<geom::hx3f1>(k, j, i + 1));
            Kokkos::atomic_store(&hx1f, coords.hx1(xf[0], xf[1], xf[2]));
            Kokkos::atomic_store(&hx2f, coords.hx2(xf[0], xf[1], xf[2]));
            Kokkos::atomic_store(&hx3f, coords.hx3(xf[0], xf[1], xf[2]));
          }
        }
        {
          auto ax = coords.GetFaceAreaX2();
          Real &ax2 = vg(b, geom::ax2())(coords.template index<geom::ax2>(k, j, i));
          Kokkos::atomic_store(&ax2, (ndim > 1) * ax[0]);
          auto xf = coords.FaceCenX2(CellFace::lower);
          Real &hx1f = vg(b, geom::hx1f2())(coords.template index<geom::hx1f2>(k, j, i));
          Real &hx2f = vg(b, geom::hx2f2())(coords.template index<geom::hx2f2>(k, j, i));
          Real &hx3f = vg(b, geom::hx3f2())(coords.template index<geom::hx3f2>(k, j, i));
          Kokkos::atomic_store(&hx1f, coords.hx1(xf[0], xf[1], xf[2]));
          Kokkos::atomic_store(&hx2f, coords.hx2(xf[0], xf[1], xf[2]));
          Kokkos::atomic_store(&hx3f, coords.hx3(xf[0], xf[1], xf[2]));
          if ((j == jb.e) && (ndim > 1)) {
            Real &ax2 = vg(b, geom::ax2())(coords.template index<geom::ax2>(k, j + 1, i));
            Kokkos::atomic_store(&ax2, (ndim > 1) * ax[1]);
            xf = coords.FaceCenX2(CellFace::upper);
            Real &hx1f =
                vg(b, geom::hx1f2())(coords.template index<geom::hx1f2>(k, j + 1, i));
            Real &hx2f =
                vg(b, geom::hx2f2())(coords.template index<geom::hx2f2>(k, j + 1, i));
            Real &hx3f =
                vg(b, geom::hx3f2())(coords.template index<geom::hx3f2>(k, j + 1, i));
            Kokkos::atomic_store(&hx1f, coords.hx1(xf[0], xf[1], xf[2]));
            Kokkos::atomic_store(&hx2f, coords.hx2(xf[0], xf[1], xf[2]));
            Kokkos::atomic_store(&hx3f, coords.hx3(xf[0], xf[1], xf[2]));
          }
        }
        {
          auto ax = coords.GetFaceAreaX3();
          Real &ax3 = vg(b, geom::ax3())(coords.template index<geom::ax3>(k, j, i));
          Kokkos::atomic_store(&ax3, (ndim > 2) * ax[0]);
          auto xf = coords.FaceCenX3(CellFace::lower);
          Real &hx1f = vg(b, geom::hx1f3())(coords.template index<geom::hx1f3>(k, j, i));
          Real &hx2f = vg(b, geom::hx2f3())(coords.template index<geom::hx2f3>(k, j, i));
          Real &hx3f = vg(b, geom::hx3f3())(coords.template index<geom::hx3f3>(k, j, i));
          Kokkos::atomic_store(&hx1f, coords.hx1(xf[0], xf[1], xf[2]));
          Kokkos::atomic_store(&hx2f, coords.hx2(xf[0], xf[1], xf[2]));
          Kokkos::atomic_store(&hx3f, coords.hx3(xf[0], xf[1], xf[2]));
          if ((k == kb.e) && (ndim > 2)) {
            Real &ax3 = vg(b, geom::ax3())(coords.template index<geom::ax3>(k + 1, j, i));
            Kokkos::atomic_store(&ax3, (ndim > 2) * ax[1]);
            xf = coords.FaceCenX3(CellFace::upper);
            Real &hx1f =
                vg(b, geom::hx1f3())(coords.template index<geom::hx1f3>(k + 1, j, i));
            Real &hx2f =
                vg(b, geom::hx2f3())(coords.template index<geom::hx2f3>(k + 1, j, i));
            Real &hx3f =
                vg(b, geom::hx3f3())(coords.template index<geom::hx3f3>(k + 1, j, i));
            Kokkos::atomic_store(&hx1f, coords.hx1(xf[0], xf[1], xf[2]));
            Kokkos::atomic_store(&hx2f, coords.hx2(xf[0], xf[1], xf[2]));
            Kokkos::atomic_store(&hx3f, coords.hx3(xf[0], xf[1], xf[2]));
          }
        }

        // connection coeffs
        {
          auto dh = coords.GetConnX1();
          Real &dh1 = vg(b, geom::dh1dx1())(coords.template index<geom::dh1dx1>(k, j, i));
          Real &dh2 = vg(b, geom::dh2dx1())(coords.template index<geom::dh2dx1>(k, j, i));
          Real &dh3 = vg(b, geom::dh3dx1())(coords.template index<geom::dh3dx1>(k, j, i));
          Kokkos::atomic_store(&dh1, x1dep_ * dh[0]);
          Kokkos::atomic_store(&dh2, x1dep_ * dh[1]);
          Kokkos::atomic_store(&dh3, x1dep_ * dh[2]);
        }
        {
          auto dh = coords.GetConnX2();
          Real &dh1 = vg(b, geom::dh1dx2())(coords.template index<geom::dh1dx2>(k, j, i));
          Real &dh2 = vg(b, geom::dh2dx2())(coords.template index<geom::dh2dx2>(k, j, i));
          Real &dh3 = vg(b, geom::dh3dx2())(coords.template index<geom::dh3dx2>(k, j, i));
          Kokkos::atomic_store(&dh1, x2dep_ * dh[0]);
          Kokkos::atomic_store(&dh2, x2dep_ * dh[1]);
          Kokkos::atomic_store(&dh3, x2dep_ * dh[2]);
        }
        {
          auto dh = coords.GetConnX3();
          Real &dh1 = vg(b, geom::dh1dx3())(coords.template index<geom::dh1dx3>(k, j, i));
          Real &dh2 = vg(b, geom::dh2dx3())(coords.template index<geom::dh2dx3>(k, j, i));
          Real &dh3 = vg(b, geom::dh3dx3())(coords.template index<geom::dh3dx3>(k, j, i));
          Kokkos::atomic_store(&dh1, x3dep_ * dh[0]);
          Kokkos::atomic_store(&dh2, x3dep_ * dh[1]);
          Kokkos::atomic_store(&dh3, x3dep_ * dh[2]);
        }
      });
}

template void InitBlockGeom<Coordinates::cartesian>(MeshBlock *pmb, ParameterInput *pin);
template void InitBlockGeom<Coordinates::spherical1D>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::spherical2D>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::spherical3D>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::cylindrical>(MeshBlock *pmb,
                                                      ParameterInput *pin);
template void InitBlockGeom<Coordinates::axisymmetric>(MeshBlock *pmb,
                                                       ParameterInput *pin);

template void EnrollFields<Coordinates::cartesian>(StateDescriptor *pkg,
                                                   CoordParams &cpars);
template void EnrollFields<Coordinates::axisymmetric>(StateDescriptor *pkg,
                                                      CoordParams &cpars);
template void EnrollFields<Coordinates::cylindrical>(StateDescriptor *pkg,
                                                     CoordParams &cpars);
template void EnrollFields<Coordinates::spherical1D>(StateDescriptor *pkg,
                                                     CoordParams &cpars);
template void EnrollFields<Coordinates::spherical2D>(StateDescriptor *pkg,
                                                     CoordParams &cpars);
template void EnrollFields<Coordinates::spherical3D>(StateDescriptor *pkg,
                                                     CoordParams &cpars);

} // namespace geometry
