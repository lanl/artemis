#ifndef SLOPE_LIMITER_HPP_
#define SLOPE_LIMITER_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace SlopeLimiter {

KOKKOS_INLINE_FUNCTION
Real original(const Real &a, const Real &b) { // Van Leer
  if (a*b <= 0.0) return 0.0;
  Real dq2 = a * b;
  Real dqm = dq2 / (a + b);
  return 2. * dqm;
}

KOKKOS_INLINE_FUNCTION
Real van_albada(const Real &a, const Real &b) {
  if (a*b <= 0.0) return 0.0;
  return (a*b*(a+b))/(a*a + b*b);
}

KOKKOS_INLINE_FUNCTION
Real minmod(const Real &a, const Real &b) {
  if (a*b <= 0.0) return 0.0;
  return (fabs(a) < fabs(b)) ? a : b;
}

KOKKOS_INLINE_FUNCTION
Real MC(const Real &a, const Real &b) {
  Real c = 0.5 * (a + b);
  return minmod(c, minmod(2.0*a, 2.0*b));
}

KOKKOS_INLINE_FUNCTION
Real superbee(const Real &a, const Real &b) {
  Real s1 = minmod(2.0*a, b);
  Real s2 = minmod(a, 2.0*b);
  return (fabs(s1) > fabs(s2)) ? s1 : s2;
}

} // SlopeLimiter

KOKKOS_INLINE_FUNCTION
Real ApplyTVD(const TVDType tvd, const Real &a, const Real &b) {
  switch (tvd) {
    case TVDType::Original: // Van Leer
      return SlopeLimiter::original(a, b);
    case TVDType::VanAlbada:
      return SlopeLimiter::van_albada(a,b);
    case TVDType::Minmod:
      return SlopeLimiter::minmod(a, b);
    case TVDType::MC:
      return SlopeLimiter::MC(a, b);
    case TVDType::Superbee:
      return SlopeLimiter::superbee(a,b);
  }
  return 0.0;
}

#endif // SLOPE_LIMITER_HPP_
