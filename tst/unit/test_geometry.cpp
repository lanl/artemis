//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
//  license in this material to reproduce, prepare derivative works, distribute copies to
//  the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// This file was created in part or in whole by generative AI

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "geometry/spherical.hpp"
#include "geometry/cylindrical.hpp"
#include "geometry/axisymmetric.hpp"

using Catch::Approx;
using namespace geometry;

// Helper to create a simple BBox for testing
struct TestBBox {
  Real x1[2];
  Real x2[2];
  Real x3[2];
};

TEST_CASE("Spherical 3D coordinate system", "[geometry][spherical]") {
  // Create a test cell in spherical coordinates
  // r: [1, 2], theta: [pi/4, pi/2], phi: [0, pi/2]
  
  SECTION("Volume calculation") {
    // Analytical volume: ∫∫∫ r² sin(θ) dr dθ dφ
    // = [r³/3]₁² × [-cos(θ)]_{π/4}^{π/2} × [φ]₀^{π/2}
    // = (8/3 - 1/3) × (0 - (-√2/2)) × π/2
    // = 7/3 × √2/2 × π/2
    
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = M_PI / 4.0;
    coords.bnds.x2[1] = M_PI / 2.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI / 2.0;
    
    const Real expected_volume = (7.0 / 3.0) * (std::sqrt(2.0) / 2.0) * (M_PI / 2.0);
    const Real computed_volume = coords.Volume();
    
    REQUIRE(computed_volume == Approx(expected_volume).epsilon(1e-10));
  }
  
  SECTION("Radial centroid (x1v)") {
    // For spherical coordinates, the radial centroid is volume-weighted:
    // <r> = ∫ r³ dr / ∫ r² dr
    // For r ∈ [1, 2]: <r> = [r⁴/4]₁² / [r³/3]₁²
    // = (16/4 - 1/4) / (8/3 - 1/3) = (15/4) / (7/3) = 45/28
    
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 2.0 * M_PI;
    
    const Real expected_x1v = 45.0 / 28.0;
    const Real computed_x1v = coords.x1v();
    
    REQUIRE(computed_x1v == Approx(expected_x1v).epsilon(1e-10));
  }
  
  SECTION("Theta centroid (x2v)") {
    // Volume-weighted theta: ∫ θ sin(θ) dθ / ∫ sin(θ) dθ
    // = [sin(θ) - θ cos(θ)] / [-cos(θ)]
    
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = M_PI / 4.0;
    coords.bnds.x2[1] = 3.0 * M_PI / 4.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI;
    
    const Real computed_x2v = coords.x2v();
    
    // Should be close to pi/2 by symmetry
    REQUIRE(computed_x2v == Approx(M_PI / 2.0).epsilon(1e-6));
  }
  
  SECTION("X1 face area") {
    // Area of radial face: ∫∫ r² sin(θ) dθ dφ
    // At r = 1.5, θ ∈ [0, π/2], φ ∈ [0, π]
    // = 1.5² × [-cos(θ)]₀^{π/2} × π
    // = 2.25 × 1 × π
    
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI / 2.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI;
    
    const Real r = 1.5;
    const Real expected_area = r * r * 1.0 * M_PI;
    const Real computed_area = coords.AreaX1(r);
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("X2 face area") {
    // Area of theta face: ∫∫ r sin(θ) dr dφ
    // For r ∈ [1, 2], θ = π/4, φ ∈ [0, π]
    // = [r²/2]₁² × sin(π/4) × π
    
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = M_PI / 6.0;
    coords.bnds.x2[1] = M_PI / 3.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI;
    
    const Real theta = M_PI / 4.0;
    const Real expected_area = 1.5 * 1.0 * std::sin(theta) * M_PI;
    const Real computed_area = coords.AreaX2(theta);
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("X3 face area") {
    // Area of phi face: ∫∫ r dr dθ = [r²/2] × Δθ
    
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI / 2.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI;
    
    const Real expected_area = 1.5 * 1.0 * (M_PI / 2.0);
    const Real computed_area = coords.AreaX3(0.0); // phi value doesn't matter
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
}

TEST_CASE("Spherical 2D coordinate system", "[geometry][spherical]") {
  // 2D spherical (r, theta) with axisymmetry in phi
  
  SECTION("Volume calculation") {
    // Volume: ∫∫ r² sin(θ) dr dθ (no explicit phi integration in 2D)
    // The 2D implementation doesn't include the 2π factor
    
    Coords<Coordinates::spherical2D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = M_PI / 4.0;
    coords.bnds.x2[1] = M_PI / 2.0;
    
    // dx1 * rfac * |cos(x2[0]) - cos(x2[1])|
    const Real dx1 = 1.0;
    const Real rfac = (1.0 + 2.0 + 4.0) / 3.0; // (r₀² + r₀r₁ + r₁²) / 3
    const Real dx2 = std::abs(std::cos(M_PI / 4.0) - std::cos(M_PI / 2.0));
    const Real expected_volume = rfac * dx1 * dx2;
    const Real computed_volume = coords.Volume();
    
    REQUIRE(computed_volume == Approx(expected_volume).epsilon(1e-10));
  }
  
  SECTION("Centroid calculations") {
    Coords<Coordinates::spherical2D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI;
    
    const Real expected_x1v = 45.0 / 28.0;
    REQUIRE(coords.x1v() == Approx(expected_x1v).epsilon(1e-10));
  }
}

TEST_CASE("Cylindrical coordinate system", "[geometry][cylindrical]") {
  // Cylindrical coordinates: (R, phi, z)
  
  SECTION("Volume calculation") {
    // Volume: ∫∫∫ R dR dφ dz = [R²/2] × Δφ × Δz
    // For R ∈ [1, 2], φ ∈ [0, π], z ∈ [0, 3]
    
    Coords<Coordinates::cylindrical> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 3.0;
    
    // <R> = (R₀ + R₁) / 2 = 1.5
    // Volume = <R> × ΔR × Δφ × Δz = 1.5 × 1.0 × π × 3.0
    const Real expected_volume = 1.5 * 1.0 * M_PI * 3.0;
    const Real computed_volume = coords.Volume();
    
    REQUIRE(computed_volume == Approx(expected_volume).epsilon(1e-10));
  }
  
  SECTION("Radial centroid (x1v)") {
    // For cylindrical, <R> = ∫ R² dR / ∫ R dR
    // = [R³/3]₁² / [R²/2]₁²
    // = (8/3 - 1/3) / (4/2 - 1/2) = (7/3) / (3/2) = 14/9
    
    Coords<Coordinates::cylindrical> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 2.0 * M_PI;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 1.0;
    
    const Real expected_x1v = 14.0 / 9.0;
    const Real computed_x1v = coords.x1v();
    
    REQUIRE(computed_x1v == Approx(expected_x1v).epsilon(1e-10));
  }
  
  SECTION("X1 face area (radial face)") {
    // Area: ∫∫ R dφ dz = R × Δφ × Δz
    
    Coords<Coordinates::cylindrical> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI / 2.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 2.0;
    
    const Real R = 1.5;
    const Real expected_area = R * (M_PI / 2.0) * 2.0;
    const Real computed_area = coords.AreaX1(R);
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("X3 face area (z face)") {
    // Area: ∫∫ R dR dφ = [R²/2] × Δφ = <R> × ΔR × Δφ
    
    Coords<Coordinates::cylindrical> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 1.0;
    
    const Real expected_area = 1.5 * 1.0 * M_PI;
    const Real computed_area = coords.AreaX3(0.5); // z value doesn't matter
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("Scale factor hx2v") {
    // In cylindrical coords, h_phi = R
    
    Coords<Coordinates::cylindrical> coords;
    coords.bnds.x1[0] = 2.0;
    coords.bnds.x1[1] = 4.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 1.0;
    
    const Real expected_hx2v = coords.x1v();
    const Real computed_hx2v = coords.hx2v();
    
    REQUIRE(computed_hx2v == Approx(expected_hx2v).epsilon(1e-10));
  }
}

TEST_CASE("Axisymmetric coordinate system", "[geometry][axisymmetric]") {
  // Axisymmetric coordinates: (R, z, phi) with phi as azimuthal direction
  
  SECTION("Volume calculation") {
    // Volume: ∫∫∫ R dR dz dφ = [R²/2] × Δz × Δφ
    // Similar to cylindrical but with different ordering
    
    Coords<Coordinates::axisymmetric> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 3.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI;
    
    // <R> = (R₀ + R₁) / 2 = 1.5
    // Volume = <R> × ΔR × Δz × Δφ
    const Real expected_volume = 1.5 * 1.0 * 3.0 * M_PI;
    const Real computed_volume = coords.Volume();
    
    REQUIRE(computed_volume == Approx(expected_volume).epsilon(1e-10));
  }
  
  SECTION("Radial centroid (x1v)") {
    // Same as cylindrical
    
    Coords<Coordinates::axisymmetric> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 1.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 2.0 * M_PI;
    
    const Real expected_x1v = 14.0 / 9.0;
    const Real computed_x1v = coords.x1v();
    
    REQUIRE(computed_x1v == Approx(expected_x1v).epsilon(1e-10));
  }
  
  SECTION("X1 face area (radial face)") {
    // Area: ∫∫ R dz dφ = R × Δz × Δφ
    
    Coords<Coordinates::axisymmetric> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 2.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI / 2.0;
    
    const Real R = 1.5;
    const Real expected_area = R * 2.0 * (M_PI / 2.0);
    const Real computed_area = coords.AreaX1(R);
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("X2 face area (z face)") {
    // Area: ∫∫ R dR dφ = [R²/2] × Δφ = <R> × ΔR × Δφ
    
    Coords<Coordinates::axisymmetric> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 1.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = M_PI;
    
    const Real expected_area = 1.5 * 1.0 * M_PI;
    const Real computed_area = coords.AreaX2(0.5); // z value doesn't matter
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("Scale factor hx3v") {
    // In axisymmetric coords, h_phi = R
    
    Coords<Coordinates::axisymmetric> coords;
    coords.bnds.x1[0] = 2.0;
    coords.bnds.x1[1] = 4.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 1.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 2.0 * M_PI;
    
    const Real expected_hx3v = coords.x1v();
    const Real computed_hx3v = coords.hx3v();
    
    REQUIRE(computed_hx3v == Approx(expected_hx3v).epsilon(1e-10));
  }
}

TEST_CASE("Cartesian coordinate system", "[geometry][cartesian]") {
  // Cartesian coordinates: (x, y, z)
  
  SECTION("Volume calculation") {
    // Volume: ∫∫∫ dx dy dz = Δx × Δy × Δz
    
    Coords<Coordinates::cartesian> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 3.0;
    coords.bnds.x2[0] = 2.0;
    coords.bnds.x2[1] = 5.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 4.0;
    
    const Real expected_volume = 2.0 * 3.0 * 4.0;
    const Real computed_volume = coords.Volume();
    
    REQUIRE(computed_volume == Approx(expected_volume).epsilon(1e-10));
  }
  
  SECTION("Centroids are cell centers") {
    // In Cartesian coords, centroids are simple arithmetic means
    
    Coords<Coordinates::cartesian> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 3.0;
    coords.bnds.x2[0] = 2.0;
    coords.bnds.x2[1] = 6.0;
    coords.bnds.x3[0] = -1.0;
    coords.bnds.x3[1] = 1.0;
    
    REQUIRE(coords.x1v() == Approx(2.0).epsilon(1e-10));
    REQUIRE(coords.x2v() == Approx(4.0).epsilon(1e-10));
    REQUIRE(coords.x3v() == Approx(0.0).epsilon(1e-10));
  }
  
  SECTION("Face areas") {
    Coords<Coordinates::cartesian> coords;
    coords.bnds.x1[0] = 0.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 3.0;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 5.0;
    
    // X1 face: dy × dz
    const Real expected_area_x1 = 3.0 * 5.0;
    REQUIRE(coords.AreaX1(1.0) == Approx(expected_area_x1).epsilon(1e-10));
    
    // X2 face: dx × dz
    const Real expected_area_x2 = 2.0 * 5.0;
    REQUIRE(coords.AreaX2(1.5) == Approx(expected_area_x2).epsilon(1e-10));
    
    // X3 face: dx × dy
    const Real expected_area_x3 = 2.0 * 3.0;
    REQUIRE(coords.AreaX3(2.5) == Approx(expected_area_x3).epsilon(1e-10));
  }
  
  SECTION("Scale factors are unity") {
    // In Cartesian coordinates, all scale factors h_i = 1
    
    Coords<Coordinates::cartesian> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 3.0;
    coords.bnds.x2[1] = 4.0;
    coords.bnds.x3[0] = 5.0;
    coords.bnds.x3[1] = 6.0;
    
    REQUIRE(coords.hx1v() == 1.0);
    REQUIRE(coords.hx2v() == 1.0);
    REQUIRE(coords.hx3v() == 1.0);
    
    REQUIRE(coords.hx1(1.5, 3.5, 5.5) == 1.0);
    REQUIRE(coords.hx2(1.5, 3.5, 5.5) == 1.0);
    REQUIRE(coords.hx3(1.5, 3.5, 5.5) == 1.0);
  }
  
  SECTION("No coordinate dependence") {
    Coords<Coordinates::cartesian> coords;
    
    REQUIRE(coords.x1dep() == false);
    REQUIRE(coords.x2dep() == false);
    REQUIRE(coords.x3dep() == false);
  }
}

TEST_CASE("Spherical 1D coordinate system", "[geometry][spherical]") {
  // 1D spherical (r only) with spherical symmetry
  
  SECTION("Volume calculation") {
    // Volume: ∫ r² dr × 4π (full solid angle)
    // = [r³/3] × 4π (but implementation doesn't include 4π explicitly)
    
    Coords<Coordinates::spherical1D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    
    // dx1 * rfac where rfac = (r₀² + r₀r₁ + r₁²) / 3
    const Real dx1 = 1.0;
    const Real rfac = (1.0 + 2.0 + 4.0) / 3.0;
    const Real expected_volume = rfac * dx1;
    const Real computed_volume = coords.Volume();
    
    REQUIRE(computed_volume == Approx(expected_volume).epsilon(1e-10));
  }
  
  SECTION("Radial centroid (x1v)") {
    // Volume-weighted radial centroid
    // Same formula as 3D spherical
    
    Coords<Coordinates::spherical1D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    
    const Real expected_x1v = 45.0 / 28.0;
    const Real computed_x1v = coords.x1v();
    
    REQUIRE(computed_x1v == Approx(expected_x1v).epsilon(1e-10));
  }
  
  SECTION("X1 face area (radial face)") {
    // Area: 4π r² (but implementation returns just r²)
    
    Coords<Coordinates::spherical1D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    
    const Real r = 1.5;
    const Real expected_area = r * r;
    const Real computed_area = coords.AreaX1(r);
    
    REQUIRE(computed_area == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("X2 and X3 face areas") {
    // Theta and phi faces: [r²/2] × Δr
    
    Coords<Coordinates::spherical1D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    
    const Real dx1 = 1.0;
    const Real expected_area = 0.5 * (1.0 + 2.0) * dx1;
    
    REQUIRE(coords.AreaX2(0.0) == Approx(expected_area).epsilon(1e-10));
    REQUIRE(coords.AreaX3(0.0) == Approx(expected_area).epsilon(1e-10));
  }
  
  SECTION("Coordinate dependence") {
    Coords<Coordinates::spherical1D> coords;
    
    // Only x1 (r) dependent
    REQUIRE(coords.x1dep() == true);
    REQUIRE(coords.x2dep() == false);
    REQUIRE(coords.x3dep() == false);
  }
  
  SECTION("Thin shell approximation") {
    // For a thin shell, volume ≈ 4πr² × Δr
    
    Coords<Coordinates::spherical1D> coords;
    coords.bnds.x1[0] = 100.0;
    coords.bnds.x1[1] = 100.1;
    
    const Real volume = coords.Volume();
    const Real dr = 0.1;
    const Real r_mid = 100.05;
    const Real expected_volume = (r_mid * r_mid + r_mid * r_mid / 3.0 * (dr * dr) / (r_mid * r_mid)) * dr;
    
    // For thin shell: V ≈ r² × dr
    REQUIRE(volume == Approx(r_mid * r_mid * dr).epsilon(0.01));
  }
}

TEST_CASE("Coordinate system edge cases", "[geometry]") {
  
  SECTION("Spherical coordinate at pole (theta = 0)") {
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 1.0;
    coords.bnds.x1[1] = 2.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 1e-6; // Near pole
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 2.0 * M_PI;
    
    const Real volume = coords.Volume();
    
    // Should be very small but positive
    REQUIRE(volume > 0.0);
    REQUIRE(volume < 1e-4);
  }
  
  SECTION("Cylindrical coordinate on axis (R = 0)") {
    Coords<Coordinates::cylindrical> coords;
    coords.bnds.x1[0] = 0.0;
    coords.bnds.x1[1] = 1.0;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = 2.0 * M_PI;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 1.0;
    
    const Real volume = coords.Volume();
    
    // Volume should still be computed correctly
    const Real expected_volume = 0.5 * 1.0 * 2.0 * M_PI * 1.0;
    REQUIRE(volume == Approx(expected_volume).epsilon(1e-10));
  }
  
  SECTION("Very thin spherical shell") {
    Coords<Coordinates::spherical3D> coords;
    coords.bnds.x1[0] = 10.0;
    coords.bnds.x1[1] = 10.01;
    coords.bnds.x2[0] = 0.0;
    coords.bnds.x2[1] = M_PI;
    coords.bnds.x3[0] = 0.0;
    coords.bnds.x3[1] = 2.0 * M_PI;
    
    const Real volume = coords.Volume();
    const Real surface_area = 4.0 * M_PI * 10.0 * 10.0;
    const Real dr = 0.01;
    
    // For thin shell: V ≈ surface_area × dr
    REQUIRE(volume == Approx(surface_area * dr).epsilon(0.01));
  }
}
