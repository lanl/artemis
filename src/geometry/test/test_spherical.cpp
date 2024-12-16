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
#include "utils/robust.hpp"

class TestRecorder {
 public:
  TestRecorder() : n_fail_(0) {}

  void operator()(bool pass) {
    if (!pass) {
      n_fail_++;
    }
  }

  int get_n_fail() const { return n_fail_; }

 private:
  int n_fail_;
};

using parthenon::robust::SoftEquiv;

int main() {
  auto coords = geometry::Coords<Coordinates::spherical3D>();

  TestRecorder rec;

  int nfail = 0;

  // Dimensionality
  rec(coords.x1dep() == true);
  rec(coords.x2dep() == true);
  rec(coords.x3dep() == false);

  // Geometry at a point
  const Real x[3] = {1.5, 1.7, 1.9};
  rec(SoftEquiv(coords.hx1(x[0], x[1], x[2]), 1.));
  rec(SoftEquiv(coords.hx2(x[0], x[1], x[2]), x[0]));
  rec(SoftEquiv(coords.hx3(x[0], x[1], x[2]), x[0] * std::sin(x[1])));

  // Indicate whether the test passed
  return rec.get_n_fail();
}
