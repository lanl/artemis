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
    pkg->AddField<name>(Metadata({Metadata::None, Metadata::OneCopy},                    \
                                 std::vector<int>({shape[0] * shape[1] * shape[2]})));   \
  }

template <Coordinates GEOM>
void EnrollFields(StateDescriptor *pkg, CoordParams &cpars) {

  Coords<GEOM> coords(cpars);

  ADD_FIELD(geom::x1v);
  ADD_FIELD(geom::x2v);
  ADD_FIELD(geom::x3v);
  ADD_FIELD(geom::vol);
  ADD_FIELD(geom::ax1);
  ADD_FIELD(geom::ax2);
  ADD_FIELD(geom::ax3);
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

  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::vol, geom::ax1, geom::ax2,
                         geom::ax3>((pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  const int b = 0;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry::InitBlock", parthenon::DevExecSpace(), kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto xv = coords.GetCellCenter();
        const int idx = coords.template index<geom::x1v>(k, j, i);
        vg(b, geom::x1v())(coords.template index<geom::x1v>(k, j, i)) = xv[0];
        vg(b, geom::x2v())(coords.template index<geom::x2v>(k, j, i)) = xv[1];
        vg(b, geom::x3v())(coords.template index<geom::x3v>(k, j, i)) = xv[2];

        vg(b, geom::vol())(coords.template index<geom::vol>(k, j, i)) = coords.Volume();

        // Face quantities
        auto ax = coords.GetFaceAreaX1();
        vg(b, geom::ax1())(coords.template index<geom::ax1>(k, j, i)) = ax[0];
        if (i == ib.e) {
          vg(b, geom::ax1())(coords.template index<geom::ax1>(k, j, i + 1)) = ax[1];
        }
        ax = coords.GetFaceAreaX2();
        vg(b, geom::ax2())(coords.template index<geom::ax2>(k, j, i)) = ax[0];
        if ((j == jb.e) && (ndim > 1)) {
          vg(b, geom::ax2())(coords.template index<geom::ax2>(k, j + 1, i)) = ax[1];
        }
        ax = coords.GetFaceAreaX3();
        vg(b, geom::ax3())(coords.template index<geom::ax3>(k, j, i)) = ax[0];
        if ((k == kb.e) && (ndim > 2)) {
          vg(b, geom::ax3())(coords.template index<geom::ax3>(k + 1, j, i)) = ax[1];
        }
      });
}

template <Coordinates GEOM>
parthenon::TaskStatus UpdateGeom(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  const int ndim = pm->ndim;
  auto &artemis_pkg = pm->packages.Get("artemis");
  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v, geom::vol, geom::ax1, geom::ax2,
                         geom::ax3>((pm->resolved_packages).get());
  auto vg = desc_g.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry::UpdateGeom", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(cpars, vg.GetCoordinates(b), k, j, i);
        const auto xv = coords.GetCellCenter();
        vg(b, geom::x1v())(coords.template index<geom::x1v>(k, j, i)) = xv[0];
        vg(b, geom::x2v())(coords.template index<geom::x2v>(k, j, i)) = xv[1];
        vg(b, geom::x3v())(coords.template index<geom::x3v>(k, j, i)) = xv[2];

        vg(b, geom::vol())(coords.template index<geom::vol>(k, j, i)) = coords.Volume();

        // Face quantities
        auto ax = coords.GetFaceAreaX1();
        vg(b, geom::ax1())(coords.template index<geom::ax1>(k, j, i)) = ax[0];
        if (i == ib.e) {
          vg(b, geom::ax1())(coords.template index<geom::ax1>(k, j, i + 1)) = ax[1];
        }
        ax = coords.GetFaceAreaX2();
        vg(b, geom::ax2())(coords.template index<geom::ax2>(k, j, i)) = ax[0];
        if ((j == jb.e) && (ndim > 1)) {
          vg(b, geom::ax2())(coords.template index<geom::ax2>(k, j + 1, i)) = ax[1];
        }
        ax = coords.GetFaceAreaX3();
        vg(b, geom::ax3())(coords.template index<geom::ax3>(k, j, i)) = ax[0];
        if ((k == kb.e) && (ndim > 2)) {
          vg(b, geom::ax3())(coords.template index<geom::ax3>(k + 1, j, i)) = ax[1];
        }
      });

  return TaskStatus::complete;
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

template parthenon::TaskStatus
UpdateGeom<Coordinates::cartesian>(parthenon::MeshData<Real> *md);
template parthenon::TaskStatus
UpdateGeom<Coordinates::axisymmetric>(parthenon::MeshData<Real> *md);
template parthenon::TaskStatus
UpdateGeom<Coordinates::cylindrical>(parthenon::MeshData<Real> *md);
template parthenon::TaskStatus
UpdateGeom<Coordinates::spherical1D>(parthenon::MeshData<Real> *md);
template parthenon::TaskStatus
UpdateGeom<Coordinates::spherical2D>(parthenon::MeshData<Real> *md);
template parthenon::TaskStatus
UpdateGeom<Coordinates::spherical3D>(parthenon::MeshData<Real> *md);

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
