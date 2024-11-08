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

// This file was generated in part or in whole by one of OpenAI's generative AI models.

#include <functional>
#include <iostream>
#include <map>
#include <string>

struct UnitTestResult {
  bool success;
  std::string message;
};

// Temporary testing
UnitTestResult test_function1() {
  UnitTestResult result;
  result.success = true;
  return result;
}
UnitTestResult test_function2() {
  UnitTestResult result;
  result.success = false;
  return result;
}

UnitTestResult run_test(std::pair<std::string, std::function<UnitTestResult()>> test) {
  auto test_name = test.first;
  std::cout << "Running " << test_name << "..." << std::endl;
  auto result = test.second();
  if (result.success == true) {
    printf("Test succeeded!\n");
  } else {
    printf("Test FAILED!\n");
  }
  return result;
}

int main(int argc, char *argv[]) {
  // Map of test names to test functions
  std::map<std::string, std::function<UnitTestResult()>> tests = {
      {"test1", test_function1}, {"test2", test_function2},
      // Add more test mappings here
  };

  std::map<std::string, UnitTestResult> results;
  bool all_tests_succeeded = true;

  if (argc == 1) {
    // No test name provided, run all tests
    for (const auto &test : tests) {
      results[test.first] = run_test(test);
      if (!results.at(test.first).success) {
        all_tests_succeeded = false;
      }
    }
  } else {
    // Run specified test(s)
    for (int i = 1; i < argc; ++i) {
      std::string test_name = argv[i];
      auto it = tests.find(test_name);
      if (it != tests.end()) {
        results[(*it).first] = run_test(*it);
        if (!results.at((*it).first).success) {
          all_tests_succeeded = false;
        }
      } else {
        std::cerr << "Test \"" << test_name << "\" not found." << std::endl;
      }
    }
  }

  printf("\n");
  if (results.size() == 0) {
    printf("No tests found!\n");
  } else {
    if (all_tests_succeeded) {
      printf("All tests SUCCEEDED!\n");
    } else {
      printf("Some tests FAILED!\n");
    }
    for (auto result : results) {
      printf("  %s: %s\n", result.first.c_str(),
             result.second.success ? "SUCCESS" : "FAILURE");
    }
  }

  return 0;
}
