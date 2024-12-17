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

// This file was created in part by one of OpenAI's generative AI models

#include "utils/robust.hpp"
#include <iostream>
#include <string>
#include <type_traits>

namespace artemis {
namespace test {

#define COMPARE(a, b)                                                                    \
  { test.compare(a, b, #a, #b, __FILE__, __LINE__); }

class UnitTester {
 public:
  UnitTester() = delete;
  UnitTester(const char *name) : n_fail_(0) {
    std::string path(name);
    const std::string target = "/src/"; // Delimiter to find the src directory
    size_t first_pos = path.find(target);

    // If "src" appears only once in the path, remove leading absolute path from name
    if (first_pos != std::string::npos &&
        path.find(target, first_pos + 1) == std::string::npos) {
      // Return the substring starting from "src"
      test_name_ = path.substr(first_pos + 1); // +1 to remove leading '/'
    } else {
      // Return the original path if "src" doesn't appear once
      test_name_ = path;
    }
  }

  // Expect for integral types
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value>::type
  compare(T a, T b, const char *a_expr, const char *b_expr, const char *file, int line) {
    if (a != b) {
      std::cout << "Test failed at " << file << ":" << line << "\n  " << a_expr
                << " != " << b_expr << "\n";
      n_fail_++;
    }
  }

  // Expect for floating-point types
  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value>::type
  compare(T a, T b, const char *a_expr, const char *b_expr, const char *file, int line) {
    if (!parthenon::robust::SoftEquiv(a, b)) {
      std::cout << "Test failed at " << file << ":" << line << "\n  " << a_expr
                << " ~= " << b_expr << "\n";
      n_fail_++;
    }
  }

  int return_code() const {
    if (n_fail_ == 0) {
      std::cout << "Test " << test_name_ << " PASSED\n";
    } else {
      std::cout << "Test " << test_name_ << " FAILED\n";
    }
    return n_fail_;
  }

 private:
  std::string test_name_;
  int n_fail_;
};

} // namespace test
} // namespace artemis
