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

#include "artemis.hpp"

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

template <class... var_ts>
using map_bc_pack_descriptor_t =
    std::unordered_map<bool, typename SparsePack<var_ts...>::Descriptor>;

//----------------------------------------------------------------------------------------
//! \fn int ArtemisUtils::GetBoundaryPackDescriptorMap
//! \brief Returns a map of pack descriptors to be used with boundary conditions. This is
//! straight from parthenon.
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
//! \fn void ArtemisUtils::EnrollArtemisRefinementOps
//! \brief Registers custom prolongation and restriction operators on provided Metadata
static void EnrollArtemisRefinementOps(parthenon::Metadata &m, Coordinates coords) {
  typedef Coordinates C;
  if (coords == C::cartesian) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::cartesian>,
                            ArtemisUtils::RestrictAverage<C::cartesian>>();
  } else if (coords == C::spherical) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::spherical>,
                            ArtemisUtils::RestrictAverage<C::spherical>>();
  } else if (coords == C::cylindrical) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::cylindrical>,
                            ArtemisUtils::RestrictAverage<C::cylindrical>>();
  } else if (coords == C::axisymmetric) {
    m.RegisterRefinementOps<ArtemisUtils::ProlongateSharedMinMod<C::axisymmetric>,
                            ArtemisUtils::RestrictAverage<C::axisymmetric>>();
  } else {
    PARTHENON_FAIL("Invalid artemis/coordinate system!");
  }
}

//----------------------------------------------------------------------------------------
//! \fn ArtemisBC ArtemisUtils::BCChoice
//! \brief Translates a named physical boundary condition to its associated enum class
//! entry for Artemis-supported BCs
static ArtemisBC BCChoice(std::string bc) {
  // Parthenon defaults
  if (bc == "periodic") {
    return ArtemisBC::periodic;
  } else if (bc == "outflow") {
    return ArtemisBC::outflow;
  } else if (bc == "reflecting") {
    return ArtemisBC::reflect;
  } else if (bc == "user") {
    return ArtemisBC::user;
    // Artemis custom BCs
  } else if (bc == "extrapolate") {
    return ArtemisBC::extrap;
  } else if (bc == "ic") {
    return ArtemisBC::ic;
  } else if (bc == "conductive") {
    return ArtemisBC::conduct;
  } else if (bc == "viscous") {
    return ArtemisBC::visc;
  } else if (bc == "inflow") {
    return ArtemisBC::inflow;
  } else if (bc == "none") {
    return ArtemisBC::none;
  } else {
    std::stringstream msg;
    msg << bc << " is not a valid BC choice!";
    PARTHENON_FAIL(msg);
  }
  return ArtemisBC::none;
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisUtils::ArtemisBoundaryCheck
//! \brief This hack permits Artemis to specify a named, user-defined boundary conditions
//! under the <parthenon/mesh> block in the input file.  Under the hood, the
//! function catches
//!
//!      <parthenon/mesh>
//!      ix1_bc = ic
//!
//! prior to boundary condition enrollment in Parthenon and translates it to
//!
//!      <parthenon/mesh>
//!      ix1_bc = user
//!
//!      <problem>
//!      ix1_bc = ic
static void ArtemisBoundaryCheck(ParameterInput *pin) {
  std::string pgen = pin->GetString("artemis", "problem");
  for (auto bc : {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"}) {
    const std::string choice = pin->GetString("parthenon/mesh", bc);
    const bool artemis_bc = (choice != "periodic" && choice != "outflow" &&
                             choice != "reflecting" && choice != "user");
    if (artemis_bc) {
      pin->SetString("parthenon/mesh", bc, "user");
      pin->SetString("problem", bc, choice);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real ArtemisUtils::VDot(const Real a[3], const Real b[3])
//! \brief Returns dot product of input vectors a and b
KOKKOS_FORCEINLINE_FUNCTION
Real VDot(const Real a[3], const Real b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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

} // namespace ArtemisUtils

#endif // UTILS_ARTEMIS_UTILS_HPP_
