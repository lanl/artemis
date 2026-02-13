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
#ifndef UTILS_ARTEMIS_UTILS_HPP_
#define UTILS_ARTEMIS_UTILS_HPP_

// Artemis includes
#include "artemis.hpp"
#include "utils/refinement/prolongation.hpp"
#include "utils/refinement/restriction.hpp"

namespace ArtemisUtils {

ReconstructionMethod ChooseReconMethod(std::string recon);
//----------------------------------------------------------------------------------------
//! \fn int ArtemisUtils::VI
//! \brief Returns vector index associated with species n for vector element
KOKKOS_FORCEINLINE_FUNCTION
static int VI(const int n, const int d) { return n * 3 + d; }

//----------------------------------------------------------------------------------------
//! \fn Real ArtemisUtils::VDot(const Real a[3], const Real b[3])
//! \brief Returns dot product of input vectors a and b
template <typename V1, typename V2>
KOKKOS_FORCEINLINE_FUNCTION Real VDot(const V1 &a, const V2 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

//----------------------------------------------------------------------------------------
//! \fn Real ArtemisUtils::DualEnergySIE(vmesh, const int b, const int n, const int k,
//!                                      const int j, const int i, const Real de_switch,
//!                                      const Real hx[3])
//! \brief Returns appropriate specific internal energy variable based on de_switch
//! NOTE(@pdmullen): Floors should be handled outside this function call
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION Real DualEnergySIE(T &vmesh, const int b, const int n,
                                               const int k, const int j, const int i,
                                               const Real de_switch,
                                               const std::array<Real, 3> &hx) {
  // Extract state vector
  const Real invd = 1.0 / vmesh(b, gas::cons::density(n), k, j, i);
  const Real &rv1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) / hx[0];
  const Real &rv2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) / hx[1];
  const Real &rv3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) / hx[2];
  const Real &u_e = vmesh(b, gas::cons::total_energy(n), k, j, i);
  const Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);
  const Real ke = 0.5 * invd * (SQR(rv1) + SQR(rv2) + SQR(rv3));

  // Calculate conserved representation of internal energy
  const Real ut_sie = invd * (u_e - ke);
  const bool dual_switch = (ut_sie > invd * de_switch * u_e);
  return (dual_switch)*ut_sie + (!dual_switch) * invd * u_u;
}

//----------------------------------------------------------------------------------------
//! \fn int ArtemisUtils::GetBoundaryPackDescriptorMap
//! \brief Returns a map of pack descriptors to be used with boundary conditions. This is
//! straight from parthenon.
template <class... var_ts>
using map_bc_pack_descriptor_t =
    std::unordered_map<bool, typename SparsePack<var_ts...>::Descriptor>;

template <class... var_ts>
map_bc_pack_descriptor_t<var_ts...>
GetBoundaryPackDescriptorMap(std::shared_ptr<MeshBlockData<Real>> &rc) {
  map_bc_pack_descriptor_t<var_ts...> my_map;
  std::vector<parthenon::MetadataFlag> flags{parthenon::Metadata::FillGhost};
  std::set<PDOpt> opts{PDOpt::Coarse};
  my_map.emplace(
      std::make_pair(true, MakePackDescriptor<var_ts...>(rc.get(), flags, opts)));
  my_map.emplace(std::make_pair(false, MakePackDescriptor<var_ts...>(rc.get(), flags)));
  return my_map;
}
//----------------------------------------------------------------------------------------
//! \struct ArtemisUtils::array_type
//! NOTE(PDM): The following is copied from the open-source Kokkos Custom Reduction Wiki
//! and adapted for Parthenon/Artemis by PDM on 10/09/23
template <class ScalarType, int N>
struct array_type {
  ScalarType myArray[N];

  KOKKOS_FORCEINLINE_FUNCTION
  array_type() { init(); }

  KOKKOS_FORCEINLINE_FUNCTION
  array_type(const array_type &rhs) {
    for (int i = 0; i < N; i++) {
      myArray[i] = rhs.myArray[i];
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION // initialize myArray to 0
      void
      init() {
    for (int i = 0; i < N; i++) {
      myArray[i] = 0;
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION
  array_type &operator+=(const array_type &src) {
    for (int i = 0; i < N; i++) {
      myArray[i] += src.myArray[i];
    }
    return *this;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void operator+=(const volatile array_type &src) volatile {
    for (int i = 0; i < N; i++) {
      myArray[i] += src.myArray[i];
    }
  }
};

//----------------------------------------------------------------------------------------
//! struct for special summation ops for arrays in e.g., linear wave regression
template <class T, class Space, int N>
struct SumMyArray {
 public:
  // Required
  typedef SumMyArray reducer;
  typedef array_type<T, N> value_type;
  typedef Kokkos::View<value_type *, Space, Kokkos::MemoryUnmanaged> result_view_type;

 private:
  value_type &value;

 public:
  KOKKOS_FORCEINLINE_FUNCTION
  SumMyArray(value_type &value_) : value(value_) {}

  // Required
  KOKKOS_FORCEINLINE_FUNCTION
  void join(value_type &dest, const value_type &src) const { dest += src; }

  KOKKOS_FORCEINLINE_FUNCTION
  void join(volatile value_type &dest, const volatile value_type &src) const {
    dest += src;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void init(value_type &val) const { val.init(); }

  KOKKOS_FORCEINLINE_FUNCTION
  value_type &reference() const { return value; }

  KOKKOS_FORCEINLINE_FUNCTION
  result_view_type view() const { return result_view_type(&value, 1); }

  KOKKOS_FORCEINLINE_FUNCTION
  bool references_scalar() const { return true; }
};

//----------------------------------------------------------------------------------------
//! Defined in artemis_utils.cpp
//! NOTE(@pdmullen): We should likely move everything above to implementation file too...
void PrintArtemisConfiguration(Packages_t &packages);
void EnrollArtemisRefinementOps(parthenon::Metadata &m, Coordinates coords,
                                const bool log);
std::vector<std::vector<Real>> loadtxt(std::string fname);

// 4D  outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(OuterLoopPatternTeams, const std::string &name,
                          DevExecSpace exec_space, size_t scratch_size_in_bytes,
                          const int scratch_level, const int nl, const int nu,
                          const int kl, const int ku, const int jl, const int ju,
                          const int il, const int iu, const Function &function) {
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  const int NkNjNi = Nk * Nj * Ni;
  const int NnNkNjNi = Nn * Nk * Nj * Ni;

  team_policy policy(exec_space, NnNkNjNi, Kokkos::AUTO);

  Kokkos::parallel_for(
      name,
      policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        int n = team_member.league_rank() / NkNjNi;
        int k = (team_member.league_rank() - n * NkNjNi) / NjNi;
        int j = (team_member.league_rank() - n * NkNjNi - k * NjNi) / Ni;
        const int i = team_member.league_rank() - n * NkNjNi - k * NjNi - j * Ni + il;
        n += nl;
        k += kl;
        j += jl;
        function(team_member, n, k, j, i);
      });
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<std::vector<Real>> NBody::loadtxt
//! \brief Cuts a 2D rectangle with the given plane
//!        The volume is computed using the divergence theorem, V = \int div(x) dV
KOKKOS_INLINE_FUNCTION
Real CutCell2D(const std::array<Real, 4> &x, const std::array<Real, 4> &y,
               const std::array<Real, 2> &xc, const std::array<Real, 2> &nx) {
  // Cuts a 2D rectangle with the given plane
  // The volume is computed using the divergence theorem, V = \int div(x) dV

  auto plane_distance = [&xc, &nx](const Real px, const Real py) {
    return nx[0] * (px - xc[0]) + nx[1] * (py - xc[1]);
  };
  const Real x0 = x[0];
  const Real y0 = y[0];
  auto contrib = [&x0, &y0](const Real xi, const Real yi, const Real xj, const Real yj) {
    return 0.5 * ((xi - x0) * (yj - y0) - (xj - x0) * (yi - y0));
  };
  Real vol_inside = 0.0;
  Real vol = 0.0;

  // Loop through the edges of the quad
  for (int i = 0; i < 4; i++) {
    const int j = (i + 1) % 4;
    vol += contrib(x[i], y[i], x[j], y[j]);

    // distance to the plane
    const Real di = plane_distance(x[i], y[i]);
    const Real dj = plane_distance(x[j], y[j]);

    // are we removing the point
    const int clipi = (di < 0.0);
    const int clipj = (dj < 0.0);

    // intersection point
    const Real xp =
        (std::abs(di) * x[j] + std::abs(dj) * x[i]) / (std::abs(di) + std::abs(dj));
    const Real yp =
        (std::abs(di) * y[j] + std::abs(dj) * y[i]) / (std::abs(di) + std::abs(dj));

    const Real x1 = (clipi) ? xp : x[i];
    const Real y1 = (clipi) ? yp : y[i];
    const Real x2 = (clipj) ? xp : x[j];
    const Real y2 = (clipj) ? yp : y[j];

    vol_inside += ((clipi + clipj) <= 1) * contrib(x1, y1, x2, y2);
  }
  return vol_inside / vol;
}

bool CoarseNeighbor(MeshBlock *pmb);

} // namespace ArtemisUtils

#endif // UTILS_ARTEMIS_UTILS_HPP_
