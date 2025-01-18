# ========================================================================================
#  (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

# Boiler plate
cmake_minimum_required(VERSION 3.13)
project(artemis LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

# Options
option(ARTEMIS_ENABLE_CUDA "Enable cuda for artemis and all dependencies" OFF)
option(ARTEMIS_ENABLE_HDF5 "Enable HDF5 for artemis and all dependencies" ON)
option(ARTEMIS_ENABLE_MPI "Enable MPI for artemis and all dependencies" ON)
option(ARTEMIS_ENABLE_OPENMP "Enable OpenMP for artemis and parthenon" OFF)
option(ARTEMIS_ENABLE_COMPILE_TIMING "Enable timing of compilation of artemis" ON)
option(ARTEMIS_ENABLE_ASAN "Enable AddressSanitizer to detect memory errors" OFF)

# Timing of compilation
if (ARTEMIS_ENABLE_COMPILE_TIMING)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE
  "${CMAKE_CURRENT_SOURCE_DIR}/env/compile_timing.sh ${PROJECT_BINARY_DIR}/compile_timing.log")
endif()

# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR
    "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
    "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# If the user doesn't specify a build type, prefer RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

# append to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Aggressive vectorization
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
    "Kokkos aggressive vectorization" FORCE)

# CUDA
if(ARTEMIS_ENABLE_CUDA)
  set(SINGULARITY_USE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
endif()

# HDF5
if(ARTEMIS_ENABLE_HDF5)
  set(HDF5_PREFER_PARALLEL ${ARTEMIS_ENABLE_MPI})
  find_package(HDF5 COMPONENTS C HL)
  if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 is required but couldn't be found. "
      "If you want to build artemis without HDF5, please rerun "
      "CMake with -DARTEMIS_ENABLE_HDF5=OFF")
  endif()
  if (ARTEMIS_ENABLE_MPI AND (NOT HDF5_IS_PARALLEL))
    message(FATAL_ERROR "Both MPI and HDF5 are enabled "
      "but only a serial version of HDF5 was found. Please install "
      "a parallel version of HDF5 (or point CMake to it by adding its path "
      "to the CMAKE_PREFIX_PATH environment variable), or disable either MPI "
      "or HDF5 by rerunning CMake with -DARTEMIS_ENABLE_MPI=OFF or "
      "-DARTEMIS_ENABLE_HDF5=OFF")
  endif()
  set(SINGULARITY_USE_HDF5 ON CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 OFF CACHE BOOL "" FORCE)
else()
  set(SINGULARITY_USE_HDF5 OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_HDF5 ON CACHE BOOL "" FORCE)
endif()

if(ARTEMIS_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "" FORCE)
endif()

if(ARTEMIS_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  set(PAR_LOOP_LAYOUT MDRANGE_LOOP CACHE STRING "" FORCE)
else()
  set(PARTHENON_DISABLE_OPENMP ON CACHE BOOL "" FORCE)
endif()

if(ARTEMIS_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fsanitize=undefined)
  add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

# NOTE(@jonahm): For some reason, order still matters for including
# parthenon and singularity. Likely has to do with project
# includes other than Kokkos. MPI and OpenMP likely culprits.

# Configure parthenon if the target does not already exist in the build
if (NOT TARGET parthenon)
  message("Configuring Parthenon")
  set(PARTHENON_ENABLE_INIT_PACKING ON CACHE BOOL "" FORCE)
  set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "" FORCE)
  set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  add_subdirectory(external/parthenon parthenon)
else()
  message(STATUS "Parthenon already configured in this build, skipping.")
endif()

# singularity common
message("Configuring singularity")
set(SINGULARITY_USE_KOKKOS ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_FORTRAN OFF CACHE BOOL "" FORCE)
set(SINGULARITY_SUBMODULE_MODE ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_KOKKOSKERNELS OFF CACHE BOOL "" FORCE)
set(SINGULARITY_BUILD_CLOSURE OFF CACHE BOOL "" FORCE)

# Kill cmake's package registry because it can interfere
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON CACHE BOOL "" FORCE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON CACHE BOOL "" FORCE)

# singularity eos
message("Configuring singularity-eos")
add_subdirectory(external/singularity-eos singularity-eos)

# REBOUND
message("Configuring REBOUND")
# Define the source and destination directories
set(REBOUND_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/rebound")
set(REBOUND_BUILD_DIR "${CMAKE_BINARY_DIR}/rebound")

# Copy rebound source tree to build directory to avoid in-source patch/build
file(COPY ${REBOUND_SOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})

# Patch rebound for nvcc by removing `-Wno-unknown-pragmas`
set(TARGET_FILE "${REBOUND_BUILD_DIR}/src/Makefile.defs")
file(READ ${TARGET_FILE} FILE_CONTENTS)
string(REGEX REPLACE "-Wno-unknown-pragmas" "" MODIFIED_FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE ${TARGET_FILE} "${MODIFIED_FILE_CONTENTS}")

# Compile rebound with its own makefile
file(GLOB rebound_src
     "${REBOUND_BUILD_DIR}/src/*.c"
     "${REBOUND_BUILD_DIR}/src/*.h")
add_custom_command(OUTPUT ${REBOUND_BUILD_DIR}/librebound.so
                   COMMAND make
                   WORKING_DIRECTORY ${REBOUND_BUILD_DIR}
                   DEPENDS ${rebound_src})
add_custom_target(rebound ALL DEPENDS ${REBOUND_BUILD_DIR}/librebound.so)

# Set Artemis/Singularity paths
set(ARTEMIS_SINGULARITY_INCLUDE_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos
    ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-opac
    ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/utils/ports-of-call
    ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/utils/spiner
    ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/utils/variant/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-opac/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

# Set Jaybenne config options and paths
# NOTE(@pdmullen): For the life of us, we are still not sure why we can't treat
# singularity-opac on the same cmake footing as singularity-eos when interfacing with
# Jaybenne.  Notice that we have directly hardcoded paths below, which is far from ideal.
# Work is likely needed in singularity-opac to accomplish the cleaner cmake
set(JAYBENNE_STANDALONE_MODE OFF CACHE BOOL "Use Jaybenne as a library only" FORCE)
set(JAYBENNE_EOS_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/eos/eos.hpp")
set(JAYBENNE_EOS_TYPE "ArtemisUtils::EOS")
set(JAYBENNE_OPACITY_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/opacity/opacity.hpp")
set(JAYBENNE_ABSORPTION_OPACITY_TYPE "ArtemisUtils::Opacity")
set(JAYBENNE_SCATTERING_OPACITY_TYPE "ArtemisUtils::Scattering")
set(JAYBENNE_SINGULARITY_INCLUDE_PATHS ${ARTEMIS_SINGULARITY_INCLUDE_PATHS})

# Point Jaybenne at the correct Artemis variables
set(JAYBENNE_HOST_VARIABLE_HEADER "src/artemis.hpp")
set(JAYBENNE_HOST_DENSITY_VARIABLE "gas::prim::density")
set(JAYBENNE_HOST_SPECIFIC_INTERNAL_ENERGY_VARIABLE "gas::prim::sie")
set(JAYBENNE_HOST_UPDATE_ENERGY_VARIABLE "gas::cons::internal_energy")

# Add jaybenne
message(STATUS "Adding jaybenne and dependencies")
add_subdirectory(external/jaybenne)

# Add artemis
message("\nConfiguring src")
add_subdirectory(src)
