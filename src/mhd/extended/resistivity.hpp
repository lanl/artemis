#ifndef RESIST_RESIST_HPP_
#define RESIST_RESIST_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

#include "defs.hpp"

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace resistivity {

KOKKOS_INLINE_FUNCTION Real None(const Real ne, const Real Pe) {
  return 0.;
}

// For ideal MHD limit: https://pure.mpg.de/rest/items/item_1560326_3/component/file_1560325/content
//                    : https://academic.oup.com/mnras/article/394/4/1727/1198578?login=false
//                    -> Also: Increasing the resolution can help to increase the maximum value of the resistivity which 
//                       can be handled, but since this gain is only linear with the number of gridpoints aiming for 
//                       higher conductivities results impractical.
// YH: this paper set \eta_ideal=1.e-8 -> https://m3dc1.pppl.gov/Papers/ferraro_PP_10.pdf
KOKKOS_INLINE_FUNCTION Real idealMHD(const Real ne, const Real Pe) {
  return 1.e-8*sigma0;
}

KOKKOS_INLINE_FUNCTION Real Spitzer(const Real ne, const Real Pe) {	
  const Real eta0 = (SQR(e_charge)*sqrt(me)/SQR(eps0)) * 
	  	1./pow(m_ion*char_speed*char_speed,1.5);
  Real eta = (4.*sqrt(2.*pi)/3.) * (Z_ion*Coulomb_log/pow(4.*pi,2)) * 
	  	pow(ne/Pe,1.5);
  if (eta > 1.3e-3) { // 1.3e-3
    eta = 1.3e-3; // use smaller cause used in Se-evolve source term -> bad for stability 
  } else if (eta < 1.3e-7) {
    eta = 1.3e-7;
  }
  if (std::isnan(eta) or eta<0.) printf("YH: eta=%f & ne=%f & Pe=%f & eta0=%f \n",eta,ne,Pe,eta0);
  return eta*sigma0; 
}

} // resitivity

KOKKOS_INLINE_FUNCTION
Real ComputeEta(EtaType eta, Real ne, Real Pe) {
  switch (eta) {
    case EtaType::None:
      return resistivity::None(ne, Pe);
    case EtaType::IdealMHD:
      return resistivity::idealMHD(ne, Pe);
    case EtaType::Spitzer:
      return resistivity::Spitzer(ne, Pe);
  }
  return 0.0;
}

#endif // RESIST_RESIST_HPP_
