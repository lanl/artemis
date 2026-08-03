# Artemis Unit Tests

  <!-- This file was created in part or in whole by generative AI -->

This directory contains unit tests for Artemis using the Catch2 testing framework.

## Building and Running Tests

To build with unit tests enabled:

```bash
cmake --preset cpu-debug  # or your preferred preset
cmake --build --preset cpu-debug
```

To run all unit tests:

```bash
cd build
ctest
```

Or run tests with verbose output:

```bash
ctest --verbose
```

To run only unit tests (excluding regression tests):

```bash
ctest -L unit
```

To run the regression test suite:

```bash
ctest -L regression
```

To exclude regression tests:

```bash
ctest -LE regression
```

To run a specific unit test directly:

```bash
./tst/unit/test_ideal_gas
```

## Regression Tests

Artemis includes regression tests that run full simulations. These are automatically integrated into CTest:

- **CPU builds**: Runs `regression.suite` (serial + parallel tests)
- **GPU builds** (CUDA/HIP): Runs `gpu.suite` (serial + GPU-specific tests)

The regression tests use `tst/run_tests.py` and can take significantly longer than unit tests. Test suites are defined in `tst/suites/`.

## Adding New Tests

1. Create a new `.cpp` file in `tst/unit/`
2. Include Catch2 headers: `#include <catch2/catch_test_macros.hpp>`
3. Write your tests using Catch2's `TEST_CASE` macro
4. Add your test to `tst/unit/CMakeLists.txt` using `add_unit_test()`

Example:
```cpp
#include <catch2/catch_test_macros.hpp>

TEST_CASE("My test description", "[tag]") {
  REQUIRE(1 + 1 == 2);
}
```

## Current Tests

### test_ideal_gas.cpp
Tests for the ideal gas equation of state (IdealGas EOS):
- Gamma parameter recovery
- Pressure, density, and temperature relationships
- Specific internal energy calculations
- UnitSystem wrapper functionality with different unit systems

### test_ideal_hhe.cpp
Tests for the hydrogen-helium mixture equation of state (IdealHHe EOS):
- Temperature recovery from density and internal energy
- Pressure calculations
- Internal energy monotonicity with temperature
- Different H-He compositions (solar, H-rich, He-rich)
- UnitSystem wrapper with CGS and scaled units

### test_geometry.cpp
Tests for coordinate system implementations in `src/geometry/`:
- **Cartesian** (x, y, z): Volume, centroids, face areas, unit scale factors, no metric dependence
- **Spherical 3D** (r, θ, φ): Volume, centroids, face areas
- **Spherical 2D** (r, θ): Axisymmetric volume and centroid calculations
- **Spherical 1D** (r): Spherically symmetric volume, radial centroid, face areas, thin shell approximation
- **Cylindrical** (R, φ, z): Volume, radial centroid, face areas, scale factors
- **Axisymmetric** (R, z, φ): Volume, radial centroid, face areas, scale factors
- **Edge cases**: Poles, axis, thin shells

Each test validates:
- Volume calculations using analytical integration formulas
- Volume-weighted centroids (x1v, x2v, x3v)
- Face areas (AreaX1, AreaX2, AreaX3)
- Metric scale factors (hx1v, hx2v, hx3v)
- Coordinate dependencies (x1dep, x2dep, x3dep)

## Test Organization

Tests should be organized by component:
- `test_ideal_gas.cpp` - Tests for ideal gas equation of state
- Add more tests as needed for other components

## Catch2 Resources

- [Catch2 Tutorial](https://github.com/catchorg/Catch2/blob/devel/docs/tutorial.md)
- [Assertion Macros](https://github.com/catchorg/Catch2/blob/devel/docs/assertions.md)
