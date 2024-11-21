//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_OPACITY_OPACITY_HPP_
#define UTILS_OPACITY_OPACITY_HPP_

// Singularity-opac includes
#include <singularity-opac/photons/opac_photons.hpp>
#include <singularity-opac/photons/s_opac_photons.hpp>

namespace ArtemisUtils {

// Reduced absorption variant for this codebase
using Opacity = singularity::photons::impl::Variant<singularity::photons::Gray,
                                                    singularity::photons::EPBremss>;

// Reduced scattering variant for this codebase
using Scattering = singularity::photons::impl::S_Variant<singularity::photons::GrayS,
                                                         singularity::photons::ThomsonS>;

} // namespace ArtemisUtils

#endif // UTILS_OPACITY_OPACITY_HPP_
