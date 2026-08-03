# ========================================================================================
#  (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# This file was created in part or in whole by generative AI

# Unit tests using Catch2

# Helper function to add unit tests
function(add_unit_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  
  target_link_libraries(${TEST_NAME}
    PRIVATE
      Catch2::Catch2WithMain
      parthenon
      singularity-eos::singularity-eos
      Kokkos::kokkos
  )
  
  target_include_directories(${TEST_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../../src
      ${ARTEMIS_SINGULARITY_INCLUDE_PATHS}
  )
  
  # Discover and add tests to CTest
  catch_discover_tests(${TEST_NAME}
    PROPERTIES LABELS "unit")
endfunction()

# Include Catch2's module for test discovery
include(Catch)

# Add individual unit tests here
add_unit_test(test_ideal_gas test_ideal_gas.cpp)
add_unit_test(test_ideal_hhe test_ideal_hhe.cpp)
add_unit_test(test_geometry test_geometry.cpp)
add_unit_test(test_collision_integrals test_collision_integrals.cpp)
