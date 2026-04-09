#ifndef COLLISION_COLLISION_HPP_
#define COLLISION_COLLISION_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

#include "defs.hpp"

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

// Note here return the dimensional tau_ei so rmb to non-dimensionalize with t0 in other parts of the code!!!
namespace collision {

KOKKOS_INLINE_FUNCTION Real tau_ei_FLASH(const Real rho, const Real Ti, const Real Te) {
  Real Ti_dim = Ti * T0 * eV_to_K;
  Real Te_dim = Te * T0 * eV_to_K;
  Real ni_dim = rho * n0;
  // FLASH model which was originally written in CGS units
  Real tau_ei = ((3.*pow(kB,1.5))/(8.*sqrt(2.*pi)*pow(e_charge,4))) *
                (pow(m_ion*Te_dim+me*Ti_dim,1.5) /
                (sqrt(me*m_ion)*pow(Z_ion,2)*ni_dim*Coulomb_log));
  tau_ei *= pow(4.*pi*eps0,2); // this factor to convert from CGS -> SI units
  return tau_ei;
}

}

#endif // COLLISION_COLLISION_HPP_
