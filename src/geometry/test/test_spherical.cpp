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

#include "../geometry.hpp"
#include "artemis.hpp"
#include "unit_test_utils.hpp"
#include "utils/robust.hpp"

int main() {
  auto coords = geometry::Coords<Coordinates::spherical3D>();

  artemis::test::UnitTester test(__FILE__);

  int nfail = 0;

  // Dimensionality
  COMPARE(coords.x1dep(), true);
  COMPARE(coords.x2dep(), true);
  COMPARE(coords.x3dep(), false);

  // Geometry at a point
  const Real x[3] = {1.5, 1.7, 1.9};
  COMPARE(coords.hx1(x[0], x[1], x[2]), 1.);
  COMPARE(coords.hx2(x[0], x[1], x[2]), x[0]);
  COMPARE(coords.hx3(x[0], x[1], x[2]), x[0] * std::sin(x[1]));

  /*
  // Test of ConvertVecToCart<>
  {auto coords = ... const Real x[] = COMPARE(...)}
          .
          .

          // Text of ConvertCoordstoSph<>
          .. */

  // Indicate whether the test passed
  return test.return_code();
}
