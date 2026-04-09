#include "artemis.hpp"
#include "utils/units.hpp"

namespace B_CT_package {
using namespace parthenon::package::prelude;

template <class var, class flux_var>
TaskStatus CalculateVectorFluxes(parthenon::TopologicalElement edge,
                                 parthenon::CellLevel cl, Real fac, MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  using recon = Conserved::recon;
  using recon_f = Conserved::recon_f;

  int ndim = md->GetParentPointer()->ndim;
  static auto desc =
      parthenon::MakePackDescriptor<var, flux_var, advection_package::Conserved::recon,
                                    advection_package::Conserved::recon_f>(
          md, {}, {parthenon::PDOpt::WithFluxes});
  auto pack = desc.GetPack(md);

  // 1. Reconstruct the component of the flux field pointing in the direction of edge in
  // the quartants of the chosen edge
  TE fe;
  if (edge == TE::E1) fe = TE::F1;
  if (edge == TE::E2) fe = TE::F2;
  if (edge == TE::E3) fe = TE::F3;
  IndexRange ib = md->GetBoundsI(cl, IndexDomain::interior, TE::CC);
  IndexRange jb = md->GetBoundsJ(cl, IndexDomain::interior, TE::CC);
  IndexRange kb = md->GetBoundsK(cl, IndexDomain::interior, TE::CC);
  int koff = edge == TE::E3 ? ndim > 2 : 0;
  int joff = edge == TE::E2 ? ndim > 1 : 0;
  int ioff = edge == TE::E1 ? ndim > 0 : 0;
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s - (ndim > 2),
      kb.e + (ndim > 2), jb.s - (ndim > 1), jb.e + (ndim > 1), ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Piecewise linear in the orthogonal directions, could do something better here
        pack(b, TE::CC, recon(0), k, j, i) =
            0.5 * (pack(b, fe, flux_var(), k, j, i) +
                   pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
        pack(b, TE::CC, recon(1), k, j, i) =
            0.5 * (pack(b, fe, flux_var(), k, j, i) +
                   pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
        pack(b, TE::CC, recon(2), k, j, i) =
            0.5 * (pack(b, fe, flux_var(), k, j, i) +
                   pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
        pack(b, TE::CC, recon(3), k, j, i) =
            0.5 * (pack(b, fe, flux_var(), k, j, i) +
                   pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
      });

  // 2. Calculate the quartant averaged flux
  koff = edge != TE::E3 ? ndim > 2 : 0;
  joff = edge != TE::E2 ? ndim > 1 : 0;
  ioff = edge != TE::E1 ? ndim > 0 : 0;
  Real npoints = (koff + 1) * (joff + 1) * (ioff + 1);
  ib = md->GetBoundsI(cl, IndexDomain::interior, edge);
  jb = md->GetBoundsJ(cl, IndexDomain::interior, edge);
  kb = md->GetBoundsK(cl, IndexDomain::interior, edge);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack.flux(b, edge, var(), k, j, i) = 0.0;
        for (int dk = -koff; dk < 1; ++dk)
          for (int dj = -joff; dj < 1; ++dj)
            for (int di = -ioff; di < 1; ++di) {
              // TODO(LFR): Pick out the correct component of the reconstructed flux,
              // current version is not an issue for piecewise constant flux though.
              pack.flux(b, edge, var(), k, j, i) +=
                  pack(b, TE::CC, recon(0), k + dk, j + dj, i + di);
            }
        pack.flux(b, edge, var(), k, j, i) /= npoints;
        pack.flux(b, edge, var(), k, j, i) *= fac;
      });

  // 3. Reconstruct the transverse components of the advected field at the edge
  std::vector<TE> faces{TE::F2, TE::F3};
  if (edge == TE::E2) faces = {TE::F3, TE::F1};
  if (edge == TE::E3) faces = {TE::F1, TE::F2};
  for (auto f : faces) {
    ib = md->GetBoundsI(cl, IndexDomain::interior, f);
    jb = md->GetBoundsJ(cl, IndexDomain::interior, f);
    kb = md->GetBoundsK(cl, IndexDomain::interior, f);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s - (ndim > 2),
        kb.e + (ndim > 2), jb.s - (ndim > 1), jb.e + (ndim > 1), ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          // Piecewise linear in the orthogonal directions, could do something better here
          pack(b, f, recon_f(0), k, j, i) = pack(b, f, var(), k, j, i);
          pack(b, f, recon_f(1), k, j, i) = pack(b, f, var(), k, j, i);
        });
  }

  // 4. Add the diffusive piece to the numerical flux, which is proportional to the curl
  ib = md->GetBoundsI(cl, IndexDomain::interior, edge);
  jb = md->GetBoundsJ(cl, IndexDomain::interior, edge);
  kb = md->GetBoundsK(cl, IndexDomain::interior, edge);
  TE f1 = faces[0];
  TE f2 = faces[1];
  std::array<int, 3> d1{ndim > 0, ndim > 1, ndim > 2};
  std::array<int, 3> d2 = d1;
  d1[static_cast<int>(edge) % 3] = 0;
  d1[static_cast<int>(f1) % 3] = 0;
  d2[static_cast<int>(edge) % 3] = 0;
  d2[static_cast<int>(f2) % 3] = 0;
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack.flux(b, edge, var(), k, j, i) +=
            0.5 * (pack(b, f1, recon_f(0), k, j, i) -
                   pack(b, f1, recon_f(1), k - d1[2], j - d1[1], i - d1[0]));
        pack.flux(b, edge, var(), k, j, i) -=
            0.5 * (pack(b, f2, recon_f(0), k, j, i) -
                   pack(b, f2, recon_f(1), k - d2[2], j - d2[1], i - d2[0]));
      });

  // Add in the diffusive component
  return TaskStatus::complete;
}


} // namespace B_CT_package

