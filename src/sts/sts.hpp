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
#ifndef STS_STS_HPP_
#define STS_STS_HPP_

#include "artemis.hpp"
#include "utils/units.hpp"

namespace STS {
  void STSRKL1( Mesh *pm, const Real time, Real dt, int nstages);
  void STSRKL2FirstStage( Mesh *pm, const Real time, Real dt, int nstages);
  void STSRKL2SecondStage( Mesh *pm, const Real time, Real dt, int nstages);
}