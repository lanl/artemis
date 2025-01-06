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
#ifndef UTILS_ARTEMIS_UTILS_HPP_
#define UTILS_ARTEMIS_UTILS_HPP_

// Artemis includes
#include "artemis.hpp"
#include "utils/refinement/prolongation.hpp"
#include "utils/refinement/restriction.hpp"

namespace ArtemisUtils {
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
//! \fn int ArtemisUtils::ProblemSourceTerm
//! \brief Wrapper function for user-defined source term, checking for nullptr
static TaskStatus ProblemSourceTerm(MeshData<Real> *md, const Real time, const Real dt) {
  if (artemis::ProblemGeneratorSourceTerm != nullptr) {
    return artemis::ProblemGeneratorSourceTerm(md, time, dt);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Real ArtemisUtils::GetSpecificInternalEnergy(vmesh, const int b, const int n,
//!              const int k, const int j, const int i, const Real de_switch,
//!              const Real dflr, const Real sieflr, const Real hx[3])
//! \brief Returns appropriate specific internal energy variable based on de_switch
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION Real
GetSpecificInternalEnergy(T &vmesh, const int b, const int n, const int k, const int j,
                          const int i, const Real de_switch, const Real dflr,
                          const Real sieflr, const std::array<Real, 3> &hx) {
  // Calculate kinetic energy
  const Real u_d = std::max(vmesh(b, gas::cons::density(n), k, j, i), dflr);
  const Real &rv1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) / hx[0];
  const Real &rv2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) / hx[1];
  const Real &rv3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) / hx[2];
  const Real ke = 0.5 * (SQR(rv1) + SQR(rv2) + SQR(rv3)) / u_d;

  // Calculate conserved representation of
  // internal energy
  const Real e_cons = vmesh(b, gas::cons::total_energy(n), k, j, i);
  const Real ue_cons = e_cons - ke;
  const Real sie = (ue_cons > de_switch * e_cons)
                       ? ue_cons / u_d
                       : vmesh(b, gas::cons::internal_energy(n), k, j, i) / u_d;

  return std::max(sie, sieflr);
}

//----------------------------------------------------------------------------------------
//! \fn Real ArtemisUtils::GetSpecificInternalEnergy(vmesh, const int b, const int n,
//!              const int k, const int j, const int i, const Real de_switch,
//!              const Real dflr, const Real sieflr)
//! \brief Returns appropriate specific internal energy variable based on de_switch
template <Coordinates GEOM, typename T>
KOKKOS_FORCEINLINE_FUNCTION Real GetSpecificInternalEnergy(
    T &vmesh, const int b, const int n, const int k, const int j, const int i,
    const Real de_switch, const Real dflr, const Real sieflr) {
  // Get scale factors
  geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
  const auto &hx = coords.GetScaleFactors();

  return GetSpecificInternalEnergy(vmesh, b, n, k, j, i, de_switch, dflr, sieflr, hx);
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
void EnrollArtemisRefinementOps(parthenon::Metadata &m, Coordinates coords);
std::vector<std::vector<Real>> loadtxt(std::string fname);

} // namespace ArtemisUtils

#endif // UTILS_ARTEMIS_UTILS_HPP_
