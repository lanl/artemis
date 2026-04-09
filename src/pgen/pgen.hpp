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
#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_

// Parthenon includes
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Artemis includes
#include "advection.hpp"
#include "blast.hpp"
#include "conduction.hpp"
#include "constant.hpp"
#include "disk.hpp"
#include "gaussian_bump.hpp"
#include "kh.hpp"
#include "linear_wave.hpp"
#include "lw.hpp"
#include "rt.hpp"
#include "shock.hpp"
#include "strat.hpp"
#include "thermalization.hpp"
#include "sst.hpp"
#include "../mhd/pgen/brio_wu.hpp"
#include "../mhd/pgen/blast_mhd.hpp"
#include "../mhd/pgen/rotor_mhd.hpp"
#include "../mhd/pgen/current_sheet_mhd.hpp"
#include "../mhd/pgen/blast3D_mhd.hpp"
#include "../mhd/pgen/plasmoid_mhd.hpp"
#include "../mhd/pgen/orszag_tang_mhd.hpp"
#include "../mhd/pgen/linear_wave_mhd.hpp"
#include "../mhd/pgen/fast_shock.hpp" // Both my code and Athena++ still show contact discontinuity so bad case
#include "../mhd/pgen/huba_hall.hpp"
#include "../mhd/pgen/field_diffusion.hpp"
#include "../mhd/pgen/brio_wu_reversed.hpp"
#include "../mhd/pgen/brio_wu_y.hpp"
#include "../mhd/pgen/brio_wu_2sides.hpp"
#include "../mhd/pgen/EMwave.hpp"
#include "../mhd/pgen/sst_Trelax.hpp"
#include "../mhd/pgen/Zpinch_2D.hpp"
#include "../mhd/pgen/planar_foil.hpp"
#include "../mhd/pgen/one_planar_foil.hpp"
#include "../mhd/pgen/whistler_wave.hpp"

using namespace parthenon::package::prelude;

namespace artemis {
//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator
//! \brief
template <Coordinates T>
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  std::string name = pin->GetString("artemis", "problem");
  if (name == "advection") {
    advection::ProblemGenerator<T>(pmb, pin);
  } else if (name == "blast") {
    blast::ProblemGenerator<T>(pmb, pin);
  } else if (name == "conduction") {
    cond::ProblemGenerator<T>(pmb, pin);
  } else if (name == "constant") {
    constant::ProblemGenerator<T>(pmb, pin);
  } else if (name == "disk") {
    disk::ProblemGenerator<T>(pmb, pin);
  } else if (name == "gaussian_bump") {
    gaussian_bump::ProblemGenerator<T>(pmb, pin);
  } else if (name == "linear_wave") {
    linear_wave::ProblemGenerator<T>(pmb, pin);
  } else if (name == "lw") {
    lw::ProblemGenerator<T>(pmb, pin);
  } else if (name == "kh") {
    kh::ProblemGenerator<T>(pmb, pin);
  } else if (name == "rt") {
    rt::ProblemGenerator<T>(pmb, pin);
  } else if (name == "shock") {
    shock::ProblemGenerator<T>(pmb, pin);
  } else if (name == "strat") {
    strat::ProblemGenerator<T>(pmb, pin);
  } else if (name == "thermalization") {
    thermalization::ProblemGenerator<T>(pmb, pin);
  } else if (name == "sst") {
    sst::ProblemGenerator<T>(pmb, pin);
  } else if (name == "brio_wu") {
    brio_wu::ProblemGenerator<T>(pmb, pin);
  } else if (name == "brio_wu_2sides") {
    brio_wu_2sides::ProblemGenerator<T>(pmb, pin);
  } else if (name == "brio_wu_reversed") {
    brio_wu_reversed::ProblemGenerator<T>(pmb, pin);
  } else if (name == "brio_wu_y") {
    brio_wu_y::ProblemGenerator<T>(pmb, pin);
  } else if (name == "blast_mhd") {
    blast_mhd::ProblemGenerator<T>(pmb, pin);
  } else if (name == "rotor_mhd") {
    rotor_mhd::ProblemGenerator<T>(pmb, pin);
  } else if  (name == "current_sheet_mhd") {
    current_sheet_mhd::ProblemGenerator<T>(pmb, pin);
  } else if (name == "blast3D_mhd") {
    blast3D_mhd::ProblemGenerator<T>(pmb, pin);
  } else if (name == "plasmoid_mhd") {
    plasmoid_mhd::ProblemGenerator<T>(pmb, pin);
  } else if (name == "orszag_tang_mhd") {
    orszag_tang_mhd::ProblemGenerator<T>(pmb, pin);
  } else if (name == "linear_wave_mhd") {
    linear_wave_mhd::ProblemGenerator<T>(pmb, pin);
  } else if (name == "fast_shock") {
    fast_shock::ProblemGenerator<T>(pmb, pin);
  } else if (name == "huba_hall") {
    huba_hall::ProblemGenerator<T>(pmb, pin);
  } else if (name == "field_diffusion") {
    field_diffusion::ProblemGenerator<T>(pmb, pin);
  } else if (name == "EMwave") {
    EMwave::ProblemGenerator<T>(pmb, pin);
  } else if (name == "sst_Trelax") {
    sst_Trelax::ProblemGenerator<T>(pmb, pin);
  } else if (name == "Zpinch_2D") {
    Zpinch_2D::ProblemGenerator<T>(pmb, pin);
  } else if (name == "planar_foil") {
    planar_foil::ProblemGenerator<T>(pmb, pin);
  } else if (name == "one_planar_foil") {
    one_planar_foil::ProblemGenerator<T>(pmb, pin);
  } else if (name == "whistler_wave") {
    whistler_wave::ProblemGenerator<T>(pmb, pin);
  } else {
    PARTHENON_FAIL("Invalid problem name!");
  }
}

} // namespace artemis

#endif // PGEN_PGEN_HPP_
