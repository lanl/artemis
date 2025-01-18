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

# File list in src/
set (SRC_LIST
  artemis_driver.cpp
  artemis_driver.hpp
  artemis.cpp
  artemis.hpp

  drag/drag.cpp
  drag/drag.hpp


  derived/fill_derived.cpp
  derived/fill_derived.hpp

  dust/dust.cpp
  dust/dust.hpp

  gas/gas.cpp
  gas/gas.hpp
  gas/cooling/beta_cooling.cpp
  gas/cooling/cooling.cpp
  gas/cooling/cooling.hpp

  geometry/geometry.hpp
  geometry/spherical.hpp
  geometry/cylindrical.hpp
  geometry/axisymmetric.hpp

  gravity/binary_mass.cpp
  gravity/gravity.cpp
  gravity/nbody_gravity.hpp
  gravity/point_mass.cpp
  gravity/uniform.cpp
  gravity/gravity.hpp

  rotating_frame/rotating_frame.cpp
  rotating_frame/rotating_frame_impl.hpp
  rotating_frame/rotating_frame.hpp

  nbody/nbody.cpp
  nbody/nbody.hpp
  nbody/nbody_advance.cpp
  nbody/nbody_extras.cpp
  nbody/nbody_outputs.cpp
  nbody/nbody_setup.cpp
  nbody/nbody_utils.hpp
  nbody/particle_base.hpp

  pgen/advection.hpp
  pgen/blast.hpp
  pgen/conduction.hpp
  pgen/constant.hpp
  pgen/disk.hpp
  pgen/gaussian_bump.hpp
  pgen/linear_wave.hpp
  pgen/pgen.hpp
  pgen/problem_modifier.hpp
  pgen/shock.hpp
  pgen/strat.hpp
  pgen/thermalization.hpp

  radiation/imc/imc.hpp

  utils/artemis_utils.cpp
  utils/artemis_utils.hpp
  utils/history.hpp
  utils/units.cpp
  utils/units.hpp

  utils/diffusion/diffusion.hpp
  utils/diffusion/diffusion_coeff.hpp
  utils/diffusion/momentum_diffusion.hpp
  utils/diffusion/thermal_diffusion.hpp

  utils/fluxes/fluid_fluxes.hpp
  utils/fluxes/reconstruction/reconstruction.hpp
  utils/fluxes/reconstruction/pcm.hpp
  utils/fluxes/reconstruction/plm.hpp
  utils/fluxes/reconstruction/ppm.hpp
  utils/fluxes/riemann/hllc.hpp
  utils/fluxes/riemann/hlle.hpp
  utils/fluxes/riemann/llf.hpp
  utils/fluxes/riemann/riemann.hpp

  utils/integrators/artemis_integrator.hpp

  utils/eos/eos.hpp

  utils/opacity/opacity.hpp

  utils/refinement/amr_criteria.hpp
  utils/refinement/prolongation.hpp
  utils/refinement/restriction.hpp
)

# Generate library
add_library(artemislib ${SRC_LIST})

# Include directories
# NOTE(@pdmullen): Again, we are hardcoding paths here... we should be able to do better
target_include_directories(artemislib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  ${CMAKE_BINARY_DIR}/rebound/src
  ${ARTEMIS_SINGULARITY_INCLUDE_PATHS}
)

# Compile options
target_compile_features(artemislib PUBLIC cxx_std_17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "XL")
  target_compile_options(artemislib PUBLIC -std=c++1y -qxflag=disable__cplusplusOverride)
endif()

# CUDA options
if (Kokkos_ENABLE_CUDA)
  target_compile_options(artemislib PUBLIC --expt-relaxed-constexpr)
endif()

# Link libraries
target_link_libraries(artemislib PUBLIC parthenon
  singularity-eos
  ${CMAKE_BINARY_DIR}/rebound/librebound.so
  lib_jaybenne_package
)

# Dependencies
add_dependencies(artemislib rebound)

# Disable executable automatically for builds with downstream codes
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   message(STATUS "Standalone mode. Adding artemis executable")
   add_executable(artemis main.cpp)
   target_link_libraries(artemis PRIVATE artemislib)

   # Bake in rpath to rebound on mac
   if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    add_custom_command(TARGET artemis POST_BUILD
      COMMAND install_name_tool -add_rpath @executable_path/../rebound $<TARGET_FILE:artemis>
      COMMAND install_name_tool -change librebound.so @rpath/librebound.so $<TARGET_FILE:artemis>
    )
   endif()
endif()
