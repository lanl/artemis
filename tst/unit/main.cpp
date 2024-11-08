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

// This code was generated in whole or in part by one of OpenAI's generative AI models.

#include <functional>
#include <iostream>
#include <map>
#include <string>

// Temporary testing
void test_function1() {}
void test_function2() {}

int main(int argc, char *argv[]) {
  // Map of test names to test functions
  std::map<std::string, std::function<void()>> tests = {
      {"test1", test_function1}, {"test2", test_function2},
      // Add more test mappings here
  };

  if (argc == 1) {
    // No test name provided, run all tests
    for (const auto &test : tests) {
      std::cout << "Running " << test.first << "..." << std::endl;
      test.second();
    }
  } else {
    // Run specified test(s)
    for (int i = 1; i < argc; ++i) {
      std::string test_name = argv[i];
      auto it = tests.find(test_name);
      if (it != tests.end()) {
        std::cout << "Running " << test_name << "..." << std::endl;
        it->second();
      } else {
        std::cerr << "Test \"" << test_name << "\" not found." << std::endl;
      }
    }
  }

  return 0;
}
