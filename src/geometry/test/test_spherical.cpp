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

int main() {
  auto coords = geometry::Coords<Coordinates::spherical3D>();

  TestRecorder rec;

  int nfail = 0;

  // coords.x1dep() == false ? nfail++;
  rec(coords.x1dep() == true);

  // Indicate whether the test passed
  return rec.get_n_fail();
}
