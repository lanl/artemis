// Try to construct appropriate upwind-based interpolation here

#ifndef INTERP_INTERP_HPP_
#define INTERP_INTERP_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace INTERP {

//==============================================================================================
//! \fn  TaskStatus INTERP::CC_2_E?
//! \brief - Interpolate from cell-center to edges
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real CC_2_E1(const PackInterp &v0,  const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const Real results = 0.25 * (
                v0(b, kk , jj , i) +
                v0(b, km1, jj , i) +
                v0(b, kk , jm1, i) +
                v0(b, km1, jm1, i));
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real CC_2_E1_max(const PackInterp &v0,  const PackVel &vel,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const Real v1 = v0(b, kk , jj , i);
  const Real v2 = v0(b, km1, jj , i);
  const Real v3 = v0(b, kk , jm1, i);
  const Real v4 = v0(b, km1, jm1, i);
  Real result = v1;
  if (v2 > result) result = v2;
  if (v3 > result) result = v3;
  if (v4 > result) result = v4;
  return result;
}
template <typename PackInterp, typename PackDen>
KOKKOS_INLINE_FUNCTION
Real CC_2_E1_maxrho(const PackInterp &v0,  const PackDen &vrho,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const Real v1 = vrho(b, kk , jj , i);
  const Real v2 = vrho(b, km1, jj , i);
  const Real v3 = vrho(b, kk , jm1, i);
  const Real v4 = vrho(b, km1, jm1, i);
  Real result = v0(b, kk , jj , i);
  Real max_rho = v1;
  if (v2 > max_rho) { max_rho = v2; result = v0(b, km1, jj , i); }
  if (v3 > max_rho) { max_rho = v3; result = v0(b, kk , jm1, i); }
  if (v4 > max_rho) { max_rho = v4; result = v0(b, km1, jm1, i); }
  // YH: initially I set below which is wrong but seem to produce similar result (i didnt run all though but at least past the problematic part) to just using averaging. Using maxrho seems to be unstable so I will skip maxrho.
  /*if (v2 > v1) result = v0(b, km1, jj , i);
  if (v3 > v1) result = v0(b, kk , jm1, i);
  if (v4 > v1) result = v0(b, km1, jm1, i);*/
  return result;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real CC_2_E2(const PackInterp &v0,  const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) { 
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int ii  = i;
  const int im1 = i - 1;
  const Real results = 0.25 * (
                v0(b, kk , j, ii ) +
                v0(b, km1, j, ii ) +
                v0(b, kk , j, im1) +
                v0(b, km1, j, im1));
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real CC_2_E2_max(const PackInterp &v0,  const PackVel &vel,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int ii  = i;
  const int im1 = i - 1;
  const Real v1 = v0(b, kk , j, ii );
  const Real v2 = v0(b, km1, j, ii );
  const Real v3 = v0(b, kk , j, im1);
  const Real v4 = v0(b, km1, j, im1);
  Real result = v1;
  if (v2 > result) result = v2;
  if (v3 > result) result = v3;
  if (v4 > result) result = v4;
  return result;
}
template <typename PackInterp, typename PackDen>
KOKKOS_INLINE_FUNCTION
Real CC_2_E2_maxrho(const PackInterp &v0,  const PackDen &vrho,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int ii  = i;
  const int im1 = i - 1;
  const Real v1 = vrho(b, kk , j, ii );
  const Real v2 = vrho(b, km1, j, ii );
  const Real v3 = vrho(b, kk , j, im1);
  const Real v4 = vrho(b, km1, j, im1);
  Real result = v0(b, kk , j, ii );
  Real max_rho = v1;
  if (v2 > max_rho) {max_rho = v2; result = v0(b, km1, j, ii );}
  if (v3 > max_rho) {max_rho = v3; result = v0(b, kk , j, im1);}
  if (v4 > max_rho) {max_rho = v4; result = v0(b, km1, j, im1);}
  /*if (v2 > v1) result = v0(b, km1, j, ii );
  if (v3 > v1) result = v0(b, kk , j, im1);
  if (v4 > v1) result = v0(b, km1, j, im1);*/
  return result;
}

template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real CC_2_E3(const PackInterp &v0,  const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1;
  const Real results = 0.25 * (
                v0(b, k, jj , ii ) +
                v0(b, k, jj , im1) +
                v0(b, k, jm1, ii ) +
                v0(b, k, jm1, im1));
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real CC_2_E3_max(const PackInterp &v0,  const PackVel &vel,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1;
  const Real v1 = v0(b, k, jj , ii );
  const Real v2 = v0(b, k, jj , im1);
  const Real v3 = v0(b, k, jm1, ii );
  const Real v4 = v0(b, k, jm1, im1);
  Real result = v1;
  if (v2 > result) result = v2;
  if (v3 > result) result = v3;
  if (v4 > result) result = v4;
  return result;
}
template <typename PackInterp, typename PackDen>
KOKKOS_INLINE_FUNCTION
Real CC_2_E3_maxrho(const PackInterp &v0,  const PackDen &vrho,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1;
  const Real v1 = vrho(b, k, jj , ii );
  const Real v2 = vrho(b, k, jj , im1);
  const Real v3 = vrho(b, k, jm1, ii );
  const Real v4 = vrho(b, k, jm1, im1);
  Real result = v0(b, k, jj , ii );
  Real max_rho = v1;
  if (v2 > max_rho) {max_rho = v2; result = v0(b, k, jj , im1);}
  if (v3 > max_rho) {max_rho = v3; result = v0(b, k, jm1, ii );}
  if (v4 > max_rho) {max_rho = v4; result = v0(b, k, jm1, im1);}
  /*if (v2 > v1) result = v0(b, k, jj , im1);
  if (v3 > v1) result = v0(b, k, jm1, ii );
  if (v4 > v1) result = v0(b, k, jm1, im1);*/
  return result;
}

//==============================================================================================
//! \fn  TaskStatus INTERP::CC_2_E?_min
//! \brief - Take minimum from cell-center to edges
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E1_min(const PackInterp &v0,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = std::min(
       std::min(v0(b, k           , j           , i) ,
                v0(b, k - (ndim>2), j           , i)),
       std::min(v0(b, k           , j - (ndim>1), i) , 
                v0(b, k - (ndim>2), j - (ndim>1), i)));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E2_min(const PackInterp &v0,
                 const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = std::min(
       std::min(v0(b, k           , j, i    ) ,
                v0(b, k - (ndim>2), j, i    )),
       std::min(v0(b, k           , j, i - 1) ,
                v0(b, k - (ndim>2), j, i - 1)));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E3_min(const PackInterp &v0,  
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = std::min(
       std::min(v0(b, k, j           , i    ) ,
                v0(b, k, j           , i - 1)),
       std::min(v0(b, k, j - (ndim>1), i    ) ,
                v0(b, k, j - (ndim>1), i - 1)));
  return results;
}


//==============================================================================================
//! \fn  TaskStatus INTERP::CC_2_E?_grad
//! \brief - Gradient (1=x, 2=y, 3=z) from cell-center to edges
//         -> Note gradient only in 2D if 3D problem as along plane
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E1_grad2(const PackInterp &v0,  
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 0.5 * (
                  v0(b, k           , j           , i) +
                  v0(b, k - (ndim>2), j           , i) +
                - v0(b, k           , j - (ndim>1), i) +
                - v0(b, k - (ndim>2), j - (ndim>1), i));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E1_grad3(const PackInterp &v0, 
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 0.5 * (
                  v0(b, k           , j           , i) +
                - v0(b, k - (ndim>2), j           , i) +
                  v0(b, k           , j - (ndim>1), i) +
                - v0(b, k - (ndim>2), j - (ndim>1), i));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E2_grad1(const PackInterp &v0, 
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 0.5 * (
                  v0(b, k           , j, i    ) +
                  v0(b, k - (ndim>2), j, i    ) +
                - v0(b, k           , j, i - 1) +
                - v0(b, k - (ndim>2), j, i - 1));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E2_grad3(const PackInterp &v0, 
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 0.5 * (
                  v0(b, k           , j, i    ) +
                - v0(b, k - (ndim>2), j, i    ) +
                  v0(b, k           , j, i - 1) +
                - v0(b, k - (ndim>2), j, i - 1));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E3_grad1(const PackInterp &v0,  
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 0.5 * (
                  v0(b, k, j           , i    ) +
                - v0(b, k, j           , i - 1) +
                  v0(b, k, j - (ndim>1), i    ) +
                - v0(b, k, j - (ndim>1), i - 1));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_E3_grad2(const PackInterp &v0,
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 0.5 * (
                  v0(b, k, j           , i    ) +
                  v0(b, k, j           , i - 1) +
                - v0(b, k, j - (ndim>1), i    ) +
                - v0(b, k, j - (ndim>1), i - 1));
  return results;
}


//==============================================================================================
//! \fn  TaskStatus INTERP::CC_2_E?_divV
//! \brief - divV from cell-center to edges
//         -> Note divV only in 2D if 3D problem as along plane
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CCvel_2_E1_divV(const PackInterp &v0,
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 
	   0.5 * (v0(b, 1, k           , j           , i) +
                  v0(b, 1, k - (ndim>2), j           , i) +
                - v0(b, 1, k           , j - (ndim>1), i) +
                - v0(b, 1, k - (ndim>2), j - (ndim>1), i)) +
           0.5 * (v0(b, 2, k           , j           , i) +
                - v0(b, 2, k - (ndim>2), j           , i) +
                  v0(b, 2, k           , j - (ndim>1), i) +
                - v0(b, 2, k - (ndim>2), j - (ndim>1), i));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CCvel_2_E2_divV(const PackInterp &v0,
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 
	   0.5 * (v0(b, 0, k           , j, i    ) +
                  v0(b, 0, k - (ndim>2), j, i    ) +
                - v0(b, 0, k           , j, i - 1) +
                - v0(b, 0, k - (ndim>2), j, i - 1)) +
  	   0.5 * (v0(b, 2, k           , j, i    ) +
                - v0(b, 2, k - (ndim>2), j, i    ) +
                  v0(b, 2, k           , j, i - 1) +
                - v0(b, 2, k - (ndim>2), j, i - 1));
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CCvel_2_E3_divV(const PackInterp &v0,
                   const int &b, const int &k, const int &j, const int &i, int ndim) {
  const Real results = 
	   0.5 * (v0(b, 0, k, j           , i    ) +
                - v0(b, 0, k, j           , i - 1) +
                  v0(b, 0, k, j - (ndim>1), i    ) +
                - v0(b, 0, k, j - (ndim>1), i - 1)) + 
	   0.5 * (v0(b, 1, k, j           , i    ) +
                  v0(b, 1, k, j           , i - 1) +
                - v0(b, 1, k, j - (ndim>1), i    ) +
                - v0(b, 1, k, j - (ndim>1), i - 1));
  return results;
}


//==============================================================================================
//! \fn  TaskStatus INTERP::F?_2_E?
//! \brief - Interpolate from face-center to edges
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F1_2_E1(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  // YH: Unfortunately interpolation will need boundary cells at left end of interior cells, thus extrapolate the gradient otherwise will cause bad BCs issue for 2D Brio-Wu
  // --> But this is not good as some ghost cells should lie in interior domain from other blocks.
  const int ii  = i;
  const int ip1 = i + 1;
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const Real results = 0.125 * (
                  v0(b, TE::F1,0, kk , jj, ii ) + v0(b, TE::F1,0, kk , jm1, ii ) // i-1/2
                + v0(b, TE::F1,0, km1, jj, ii ) + v0(b, TE::F1,0, km1, jm1, ii )
                + v0(b, TE::F1,0, kk , jj, ip1) + v0(b, TE::F1,0, kk , jm1, ip1) // i+1/2
                + v0(b, TE::F1,0, km1, jj, ip1) + v0(b, TE::F1,0, km1, jm1, ip1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F2_2_E2(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int jj  = j;
  const int jp1 = j + (ndim > 1);
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int ii  = i;
  const int im1 = i - 1;
  const Real results = 0.125 * (
                  v0(b, TE::F2,0, kk, jj , ii ) + v0(b, TE::F2,0, km1, jj , ii ) // j-1/2
                + v0(b, TE::F2,0, kk, jj , im1) + v0(b, TE::F2,0, km1, jj , im1)
                + v0(b, TE::F2,0, kk, jp1, ii ) + v0(b, TE::F2,0, km1, jp1, ii ) // j+1/2
                + v0(b, TE::F2,0, kk, jp1, im1) + v0(b, TE::F2,0, km1, jp1, im1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F3_2_E3(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int kk  = k;
  const int kp1 = k + (ndim > 2);
  const int ii  = i;
  const int im1 = i - 1;
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const Real results = 0.125 * (
                  v0(b, TE::F3,0, kk , jj , ii) + v0(b, TE::F3,0, kk , jj , im1) // k-1/2
                + v0(b, TE::F3,0, kk , jm1, ii) + v0(b, TE::F3,0, kk , jm1, im1)
                + v0(b, TE::F3,0, kp1, jj , ii) + v0(b, TE::F3,0, kp1, jj , im1) // k+1/2
                + v0(b, TE::F3,0, kp1, jm1, ii) + v0(b, TE::F3,0, kp1, jm1, im1)
                );
  return results;
}

//===================Interpolation for directional splitting of implicit update=================
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E2_2_E1(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
                  v0(b, TE::E2, 0, k, j, i) + v0(b, TE::E2, 0, k, j, i + 1) 
                + v0(b, TE::E2, 0, k, j - (ndim>1), i) + v0(b, TE::E2, 0, k, j - (ndim>1), i + 1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E3_2_E1(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
                  v0(b, TE::E3, 0, k, j, i) + v0(b, TE::E3, 0, k, j, i + 1) 
                + v0(b, TE::E3, 0, k - (ndim>2), j, i) + v0(b, TE::E3, 0, k - (ndim>2), j, i + 1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E1_2_E2(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
                  v0(b, TE::E1, 0, k, j, i) + v0(b, TE::E1, 0, k, j + (ndim>1), i) 
                + v0(b, TE::E1, 0, k, j, i - 1) + v0(b, TE::E1, 0, k, j + (ndim>1), i - 1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E3_2_E2(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
                  v0(b, TE::E3, 0, k, j, i) + v0(b, TE::E3, 0, k, j + (ndim>1), i) 
                + v0(b, TE::E3, 0, k - (ndim>2), j, i) + v0(b, TE::E3, 0, k - (ndim>2), j + (ndim>1), i - 1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E1_2_E3(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
                  v0(b, TE::E1, 0, k, j, i) + v0(b, TE::E1, 0, k + (ndim>2), j, i) 
                + v0(b, TE::E1, 0, k, j, i - 1) + v0(b, TE::E1, 0, k + (ndim>2), j, i - 1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E2_2_E3(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
                  v0(b, TE::E2, 0, k, j, i) + v0(b, TE::E2, 0, k + (ndim>2), j, i) 
                + v0(b, TE::E2, 0, k, j - (ndim>1), i) + v0(b, TE::E2, 0, k + (ndim>2), j - (ndim>1), i)
                );
  return results;
}

template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F1_2_E2(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                  v0(b, TE::F1, 0, k, j, i) + v0(b, TE::F1, 0, k - (ndim>2), j, i) 
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F1_2_E3(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                  v0(b, TE::F1, 0, k, j, i) + v0(b, TE::F1, 0, k, j - (ndim>1), i) 
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F2_2_E1(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                  v0(b, TE::F2, 0, k, j, i) + v0(b, TE::F2, 0, k - (ndim>2), j, i) 
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F2_2_E3(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                  v0(b, TE::F2, 0, k, j, i) + v0(b, TE::F2, 0, k, j, i - 1) 
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F3_2_E1(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                  v0(b, TE::F3, 0, k, j, i) + v0(b, TE::F3, 0, k, j - (ndim>1), i) 
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real F3_2_E2(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                  v0(b, TE::F3, 0, k, j, i) + v0(b, TE::F3, 0, k, j, i - 1) 
                );
  return results;
}
//====================End interpolation for directional splitting of implicit update============

//==============================================================================================
//! \fn  TaskStatus INTERP::E?_2_F?
//! \brief - Interpolate from edges to face-center
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E1_2_F1(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.125 * ( 
	  	    v0(b, TE::E1, 0, k,   j,   i-1) +
                    v0(b, TE::E1, 0, k,   j+(ndim>1), i-1) +
                    v0(b, TE::E1, 0, k+(ndim>2), j,   i-1) +
                    v0(b, TE::E1, 0, k+(ndim>2), j+(ndim>1), i-1) +
                    v0(b, TE::E1, 0, k,   j,   i  ) +
                    v0(b, TE::E1, 0, k,   j+(ndim>1), i  ) +
                    v0(b, TE::E1, 0, k+(ndim>2), j,   i  ) +
                    v0(b, TE::E1, 0, k+(ndim>2), j+(ndim>1), i  )
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E1_2_F2(const PackInterp &v0,  const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                v0(b, TE::E1, 0, k,            j, i) + 
                v0(b, TE::E1, 0, k + (ndim>2), j, i)     
              );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E1_2_F3(const PackInterp &v0,  const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
                v0(b, TE::E1, 0, k, j,            i) +  
                v0(b, TE::E1, 0, k, j + (ndim>1), i)  
              );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E2_2_F2(const PackInterp &v0,  const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.125 * (
                    v0(b, TE::E2, 0, k,   j-(ndim>1), i  ) +
                    v0(b, TE::E2, 0, k,   j-(ndim>1), i+1) +
                    v0(b, TE::E2, 0, k+(ndim>2), j-(ndim>1), i  ) +
                    v0(b, TE::E2, 0, k+(ndim>2), j-(ndim>1), i+1) +
                    v0(b, TE::E2, 0, k,   j,   i  ) +
                    v0(b, TE::E2, 0, k,   j,   i+1) +
                    v0(b, TE::E2, 0, k+(ndim>2), j,   i  ) +
                    v0(b, TE::E2, 0, k+(ndim>2), j,   i+1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E2_2_F1(const PackInterp &v0, const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
        v0(b, TE::E2, 0, k,            j, i) +
        v0(b, TE::E2, 0, k + (ndim>2), j, i)
      );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E2_2_F3(const PackInterp &v0, const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
        v0(b, TE::E2, 0, k, j, i) +
        v0(b, TE::E2, 0, k, j, i + 1)
      );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E3_2_F3(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.125 * (
                    v0(b, TE::E3, 0, k-(ndim>2), j,   i  ) +
                    v0(b, TE::E3, 0, k-(ndim>2), j+(ndim>1), i  ) +
                    v0(b, TE::E3, 0, k-(ndim>2), j,   i+1) +
                    v0(b, TE::E3, 0, k-(ndim>2), j+(ndim>1), i+1) +
                    v0(b, TE::E3, 0, k,   j,   i  ) +
                    v0(b, TE::E3, 0, k,   j+(ndim>1), i  ) +
                    v0(b, TE::E3, 0, k,   j,   i+1) +
                    v0(b, TE::E3, 0, k,   j+(ndim>1), i+1)
                );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E3_2_F1(const PackInterp &v0, const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
        v0(b, TE::E3, 0, k, j,            i) +
        v0(b, TE::E3, 0, k, j + (ndim>1), i)
      );
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E3_2_F2(const PackInterp &v0, const PackVel &vel,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.5 * (
        v0(b, TE::E3, 0, k, j, i) +
        v0(b, TE::E3, 0, k, j, i + 1)
      );
  return results;
}

//==============================================================================================
//! \fn  TaskStatus INTERP::F?_2_F?
//! \brief - Interpolate from face to other faces
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F2_2_F1(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
        v0(b, TE::F2, 0, k, j,            i-1) +
        v0(b, TE::F2, 0, k, j + (ndim>1), i-1) +
        v0(b, TE::F2, 0, k, j,            i  ) +
        v0(b, TE::F2, 0, k, j + (ndim>1), i  )
      );
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F3_2_F1(const PackInterp &v0, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
        v0(b, TE::F3, 0, k,             j, i-1) +
        v0(b, TE::F3, 0, k + (ndim>2),  j, i-1) +
        v0(b, TE::F3, 0, k,             j, i  ) +
        v0(b, TE::F3, 0, k + (ndim>2),  j, i  )
      );
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F1_2_F2(const PackInterp &v0, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
        v0(b, TE::F1, 0, k, j,            i+1) +
        v0(b, TE::F1, 0, k, j - (ndim>1), i+1) +
        v0(b, TE::F1, 0, k, j,            i  ) +
        v0(b, TE::F1, 0, k, j - (ndim>1), i  )
      );
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F3_2_F2(const PackInterp &v0, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
        v0(b, TE::F3, 0, k,            j,            i) +
        v0(b, TE::F3, 0, k + (ndim>2), j,            i) +
        v0(b, TE::F3, 0, k,            j - (ndim>1), i) +
        v0(b, TE::F3, 0, k + (ndim>2), j - (ndim>1), i)
      );
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F1_2_F3(const PackInterp &v0, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
        v0(b, TE::F1, 0, k,            j, i+1) +  
        v0(b, TE::F1, 0, k - (ndim>2), j, i+1) +  
        v0(b, TE::F1, 0, k,            j, i  ) +  
        v0(b, TE::F1, 0, k - (ndim>2), j, i  )   
      );
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F2_2_F3(const PackInterp &v0, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (
        v0(b, TE::F2, 0, k,            j, i) +  
        v0(b, TE::F2, 0, k - (ndim>2), j, i) +  
        v0(b, TE::F2, 0, k,            j + (ndim>1), i) +  
        v0(b, TE::F2, 0, k - (ndim>2), j + (ndim>1), i)   
      );
  return results;
}

//==============================================================================================
//! \fn  TaskStatus INTERP::E?_2_CC
//! \brief - Interpolate from edges to cell-center
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E1_2_CC(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (v0(b, TE::E1, 0, k, j, i) +
                          v0(b, TE::E1, 0, k, j + (ndim > 1), i) +
                          v0(b, TE::E1, 0, k + (ndim > 2), j, i) +
                          v0(b, TE::E1, 0, k + (ndim > 2), j + (ndim > 1), i));
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E2_2_CC(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (v0(b, TE::E2, 0, k, j, i) +
                          v0(b, TE::E2, 0, k, j, i + 1) +
                          v0(b, TE::E2, 0, k + (ndim > 2), j, i) +
                          v0(b, TE::E2, 0, k + (ndim > 2), j, i + 1));
  return results;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E3_2_CC(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real results = 0.25 * (v0(b, TE::E3, 0, k, j, i) +
                          v0(b, TE::E3, 0, k, j, i + 1) +
                          v0(b, TE::E3, 0, k, j + (ndim > 1), i) +
                          v0(b, TE::E3, 0, k, j + (ndim > 1), i + 1));
  return results;
}


//==============================================================================================
//! \fn  TaskStatus INTERP::F?_2_CC
//! \brief - Interpolate from edges to cell-center
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F1_2_CC(const PackInterp &v0,  
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v0_dx = v0(b, TE::F1, 0, k, j, i + 1) - v0(b, TE::F1, 0, k, j, i);
  const Real results = v0(b, TE::F1, 0, k, j, i) + 0.5 * v0_dx;
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F2_2_CC(const PackInterp &v0,  
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v0_dy = v0(b, TE::F2, 0, k, j + (ndim>1), i) - v0(b, TE::F2, 0, k, j, i);
  const Real results = v0(b, TE::F2, 0, k, j, i) + 0.5 * v0_dy;
  return results;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F3_2_CC(const PackInterp &v0,  
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v0_dz = v0(b, TE::F3, 0, k + (ndim>2), j, i) - v0(b, TE::F3, 0, k, j, i);
  const Real results = v0(b, TE::F3, 0, k, j, i) + 0.5 * v0_dz;
  return results;
}


//=============================================================================================================
//! \fn  TaskStatus INTERP::Two_F?_2_CC
//! \brief - Interpolate from edges to cell-center
template <typename PackInterp, typename PackInterp2>
KOKKOS_INLINE_FUNCTION
Real Two_F1_2_CC(const PackInterp &v0, const PackInterp2 &v1,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v0_dx = v0(b, TE::F1, 0, k, j, i + 1) * v1(b, TE::F1, 0, k, j, i + 1)
	  	   - v0(b, TE::F1, 0, k, j, i) * v1(b, TE::F1, 0, k, j, i);
  const Real results = v0(b, TE::F1, 0, k, j, i) * v1(b, TE::F1, 0, k, j, i)
	  	     + 0.5 * v0_dx;
  return results;
}
template <typename PackInterp, typename PackInterp2>
KOKKOS_INLINE_FUNCTION
Real Two_F2_2_CC(const PackInterp &v0, const PackInterp2 &v1,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v0_dy = v0(b, TE::F2, 0, k, j + (ndim>1), i) * v1(b, TE::F2, 0, k, j + (ndim>1), i)
	  	   - v0(b, TE::F2, 0, k, j, i) * v1(b, TE::F2, 0, k, j, i);
  Real results = v0(b, TE::F2, 0, k, j, i) * v1(b, TE::F2, 0, k, j, i)
	  	     + 0.5 * v0_dy;
  return results;
}
template <typename PackInterp, typename PackInterp2>
KOKKOS_INLINE_FUNCTION
Real Two_F3_2_CC(const PackInterp &v0, const PackInterp2 &v1,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v0_dz = v0(b, TE::F3, 0, k + (ndim>2), j, i) * v1(b, TE::F3, 0, k + (ndim>2), j, i)
	  	   - v0(b, TE::F3, 0, k, j, i) * v1(b, TE::F3, 0, k, j, i);
  Real results = v0(b, TE::F3, 0, k, j, i) * v1(b, TE::F3, 0, k, j, i)
	  	     + 0.5 * v0_dz;
  return results;
}

// ========================================================================================
// Interpolate to nodes for implicit EJ source term update
// ========================================================================================
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_NN(const PackInterp &v0, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1;  
  return 0.125 * (
        v0(b, kk , jj , ii ) +
        v0(b, kk , jj , im1) +
        v0(b, kk , jm1, ii ) +
        v0(b, kk , jm1, im1) +
        v0(b, km1, jj , ii ) +
        v0(b, km1, jj , im1) +
        v0(b, km1, jm1, ii ) +
        v0(b, km1, jm1, im1)
    );
}

template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_NN_harmonic(const PackInterp &v0, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1; 
  return 8.0 / (
      1.0 / v0(b, kk , jj , ii ) +
      1.0 / v0(b, kk , jj , im1) +
      1.0 / v0(b, kk , jm1, ii ) +
      1.0 / v0(b, kk , jm1, im1) +
      1.0 / v0(b, km1, jj , ii ) +
      1.0 / v0(b, km1, jj , im1) +
      1.0 / v0(b, km1, jm1, ii ) +
      1.0 / v0(b, km1, jm1, im1)
  ); 
}

template <typename PackInterp, typename PackDen>
KOKKOS_INLINE_FUNCTION
Real CC_2_NN_maxrho(const PackInterp &v0, const PackDen &vrho, 
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1;
  // Initialize with first point
  Real result  = v0(b, kk, jj, ii);
  Real max_rho = vrho(b, kk, jj, ii);
  // Compare all neighbors and update
  if (vrho(b, kk, jj, im1) > max_rho) {
    max_rho = vrho(b, kk, jj, im1);
    result  = v0(b, kk, jj, im1);
  }
  if (vrho(b, kk, jm1, ii) > max_rho) {
    max_rho = vrho(b, kk, jm1, ii);
    result  = v0(b, kk, jm1, ii);
  }
  if (vrho(b, kk, jm1, im1) > max_rho) {
    max_rho = vrho(b, kk, jm1, im1);
    result  = v0(b, kk, jm1, im1);
  }
  if (vrho(b, km1, jj, ii) > max_rho) {
    max_rho = vrho(b, km1, jj, ii);
    result  = v0(b, km1, jj, ii);
  }
  if (vrho(b, km1, jj, im1) > max_rho) {
    max_rho = vrho(b, km1, jj, im1);
    result  = v0(b, km1, jj, im1);
  }
  if (vrho(b, km1, jm1, ii) > max_rho) {
    max_rho = vrho(b, km1, jm1, ii);
    result  = v0(b, km1, jm1, ii);
  }
  if (vrho(b, km1, jm1, im1) > max_rho) {
    max_rho = vrho(b, km1, jm1, im1);
    result  = v0(b, km1, jm1, im1);
  }
  return result; 
}

template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_NN_max(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1;
  return fmax(
    fmax(
      fmax(v0(b, kk , jj , ii ), v0(b, kk , jj , im1)),
      fmax(v0(b, kk , jm1, ii ), v0(b, kk , jm1, im1))
    ),
    fmax(
      fmax(v0(b, km1, jj , ii ), v0(b, km1, jj , im1)),
      fmax(v0(b, km1, jm1, ii ), v0(b, km1, jm1, im1))
    )
  );
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real CC_2_NN_min(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  const int kk  = k;
  const int km1 = k - (ndim > 2);
  const int jj  = j;
  const int jm1 = j - (ndim > 1);
  const int ii  = i;
  const int im1 = i - 1;
  return fmin(
    fmin(
      fmin(v0(b, kk , jj , ii ), v0(b, kk , jj , im1)),
      fmin(v0(b, kk , jm1, ii ), v0(b, kk , jm1, im1))
    ),
    fmin(
      fmin(v0(b, km1, jj , ii ), v0(b, km1, jj , im1)),
      fmin(v0(b, km1, jm1, ii ), v0(b, km1, jm1, im1))
    )
  );
}

template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F1_2_NN(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;	
  const int km1 = k - (ndim > 2);
  const int jm1 = j - (ndim > 1);
  return 0.25 * (
      v0(b, TE::F1, 0, k  , j  , i) +
      v0(b, TE::F1, 0, k  , jm1, i) +
      v0(b, TE::F1, 0, km1, j  , i) +
      v0(b, TE::F1, 0, km1, jm1, i)
  );
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F2_2_NN(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;	
  const int km1 = k - (ndim > 2);
  const int im1 = i - 1;
  return 0.25 * (
      v0(b, TE::F2, 0, k  , j, i  ) +
      v0(b, TE::F2, 0, k  , j, im1) +
      v0(b, TE::F2, 0, km1, j, i  ) +
      v0(b, TE::F2, 0, km1, j, im1)
  );
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F3_2_NN(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;	
  const int jm1 = j - (ndim > 1);
  const int im1 = i - 1;
  return 0.25 * (
      v0(b, TE::F3, 0, k, j  , i  ) +
      v0(b, TE::F3, 0, k, jm1, i  ) +
      v0(b, TE::F3, 0, k, j  , im1) +
      v0(b, TE::F3, 0, k, jm1, im1)
  );
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E1_2_NN(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int im1 = i - 1;
  return 0.5 * (v0(b, TE::E1, 0, k, j, i  ) + v0(b, TE::E1, 0, k, j, im1));
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E2_2_NN(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int jm1 = j - (ndim > 1);
  return 0.5 * (v0(b, TE::E2, 0, k, j  , i) + v0(b, TE::E2, 0, k, jm1, i));
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E3_2_NN(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int km1 = k - (ndim > 2);
  return 0.5 * (v0(b, TE::E3, 0, k  , j, i) + v0(b, TE::E3, 0, km1, j, i));
}

template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E1_vec(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, int iv) {
  const int ip1 = i + 1;
  return 0.5 * (v0(b, TE::NN, iv, k, j, i   ) + v0(b, TE::NN, iv, k, j, ip1 ));
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E2_vec(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, int iv) {
  const int jp1 = j + (ndim > 1);
  return 0.5 * (v0(b, TE::NN, iv, k, j   , i) + v0(b, TE::NN, iv, k, jp1 , i));
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E3_vec(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, int iv) {
  const int kp1 = k + (ndim > 2);
  return 0.5 * (v0(b, TE::NN, iv, k   , j, i) + v0(b, TE::NN, iv, kp1 , j, i));
}

template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E1_vec_upw(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, 
	     const int iv, const Real vel) {
  const int ip1 = i + 1;
  return (vel>0.) ? v0(b, TE::NN, iv, k, j, i) : v0(b, TE::NN, iv, k, j, ip1);
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E2_vec_upw(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, 
	     const int iv, const Real vel) {
  const int jp1 = j + (ndim > 1);
  return (vel>0.) ? v0(b, TE::NN, iv, k, j, i) : v0(b, TE::NN, iv, k, jp1 , i);
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E3_vec_upw(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim,
	     const int iv, const Real vel) {
  const int kp1 = k + (ndim > 2);
  return (vel>0.) ? v0(b, TE::NN, iv, k, j, i) : v0(b, TE::NN, iv, kp1 , j, i);
}

//===============Testing========================================================
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E1_2_NN_min(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int im1 = i - 1;
  const Real v_a = v0(b, TE::E1, 0, k, j, i);
  const Real v_b = v0(b, TE::E1, 0, k, j, im1);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E2_2_NN_min(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int jm1 = j - (ndim > 1);
  const Real v_a = v0(b, TE::E2, 0, k, j  , i);
  const Real v_b = v0(b, TE::E2, 0, k, jm1, i);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E3_2_NN_min(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int km1 = k - (ndim > 2);
  const Real v_a = v0(b, TE::E3, 0, k  , j, i);
  const Real v_b = v0(b, TE::E3, 0, km1, j, i);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp> // harmonic <= geometric <= arithmetic
KOKKOS_INLINE_FUNCTION // Harmonic average which bias toward smaller magnitude value
Real NN_2_E1_vec_harmonic(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, int iv) {
  const int ip1 = i + 1;
  const Real va = v0(b, TE::NN, iv, k, j, i   );
  const Real vb = v0(b, TE::NN, iv, k, j, ip1 );
  return 2.*va*vb/(va+vb);
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E2_vec_harmonic(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, int iv) {
  const int jp1 = j + (ndim > 1);
  const Real va = v0(b, TE::NN, iv, k, j   , i);
  const Real vb = v0(b, TE::NN, iv, k, jp1 , i);
  return 2.*va*vb/(va+vb);
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real NN_2_E3_vec_harmonic(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim, int iv) {
  const int kp1 = k + (ndim > 2);
  const Real va = v0(b, TE::NN, iv, k   , j, i);
  const Real vb = v0(b, TE::NN, iv, kp1 , j, i);
  return 2.*va*vb/(va+vb);
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E1_2_CC_max(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v1 = v0(b, TE::E1, 0, k, j, i);
  const Real v2 = v0(b, TE::E1, 0, k, j + (ndim > 1), i);
  const Real v3 = v0(b, TE::E1, 0, k + (ndim > 2), j, i);
  const Real v4 = v0(b, TE::E1, 0, k + (ndim > 2), j + (ndim > 1), i);
  Real result = v1;
  if (std::abs(v2) > std::abs(result)) result = v2;
  if (std::abs(v3) > std::abs(result)) result = v3;
  if (std::abs(v4) > std::abs(result)) result = v4;
  return result;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E2_2_CC_max(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v1 = v0(b, TE::E2, 0, k, j, i);
  const Real v2 = v0(b, TE::E2, 0, k, j, i + 1);
  const Real v3 = v0(b, TE::E2, 0, k + (ndim > 2), j, i);
  const Real v4 = v0(b, TE::E2, 0, k + (ndim > 2), j, i + 1);
  Real result = v1;
  if (std::abs(v2) > std::abs(result)) result = v2;
  if (std::abs(v3) > std::abs(result)) result = v3;
  if (std::abs(v4) > std::abs(result)) result = v4;
  return result;
}
template <typename PackInterp, typename PackVel>
KOKKOS_INLINE_FUNCTION
Real E3_2_CC_max(const PackInterp &v0,  const PackVel &vel,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real v1 = v0(b, TE::E3, 0, k, j, i);
  const Real v2 = v0(b, TE::E3, 0, k, j + (ndim > 1), i);
  const Real v3 = v0(b, TE::E3, 0, k, j, i + 1);
  const Real v4 = v0(b, TE::E3, 0, k, j + (ndim > 1), i + 1);
  Real result = v1;
  if (std::abs(v2) > std::abs(result)) result = v2;
  if (std::abs(v3) > std::abs(result)) result = v3;
  if (std::abs(v4) > std::abs(result)) result = v4;
  return result;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F1_2_CC_min(const PackInterp &v0,  
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int ip1 = i + 1;
  const Real v_a = v0(b, TE::F1, 0, k, j, i);
  const Real v_b = v0(b, TE::F1, 0, k, j, ip1);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F2_2_CC_min(const PackInterp &v0,  
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int jp1 = j + (ndim>1);
  const Real v_a = v0(b, TE::F2, 0, k, j, i);
  const Real v_b = v0(b, TE::F2, 0, k, jp1, i);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F3_2_CC_min(const PackInterp &v0,  
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int kp1 = k + (ndim>2);
  const Real v_a = v0(b, TE::F3, 0, k, j, i);
  const Real v_b = v0(b, TE::F3, 0, kp1, j, i);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp, typename PackInterp2>
KOKKOS_INLINE_FUNCTION
Real Two_F1_2_CC_min(const PackInterp &v0, const PackInterp2 &v1,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real left = v0(b, TE::F1, 0, k, j, i)
                  * v1(b, TE::F1, 0, k, j, i);
  const Real right = v0(b, TE::F1, 0, k, j, i + 1)
                   * v1(b, TE::F1, 0, k, j, i + 1);
  return (std::abs(left) <= std::abs(right)) ? left : right;
}
template <typename PackInterp, typename PackInterp2>
KOKKOS_INLINE_FUNCTION
Real Two_F2_2_CC_min(const PackInterp &v0, const PackInterp2 &v1,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real left = v0(b, TE::F2, 0, k, j, i)
                  * v1(b, TE::F2, 0, k, j, i);
  const Real right = v0(b, TE::F2, 0, k, j + (ndim>1), i)
                   * v1(b, TE::F2, 0, k, j + (ndim>1), i);
  return (std::abs(left) <= std::abs(right)) ? left : right;
}
template <typename PackInterp, typename PackInterp2>
KOKKOS_INLINE_FUNCTION
Real Two_F3_2_CC_min(const PackInterp &v0, const PackInterp2 &v1,
                        const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const Real left = v0(b, TE::F3, 0, k, j, i)
                  * v1(b, TE::F3, 0, k, j, i);
  const Real right = v0(b, TE::F3, 0, k + (ndim>2), j, i)
                   * v1(b, TE::F3, 0, k + (ndim>2), j, i);
  return (std::abs(left) <= std::abs(right)) ? left : right;
}

// Temp here as just anyhow modify to see if there is a bound
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E1_2_NN_mod(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int im1 = i - 1;
  const Real v_a = v0(b, TE::E1, 0, k, j, i);
  const Real v_b = v0(b, TE::E1, 0, k, j, im1);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E2_2_NN_mod(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int jm1 = j - (ndim > 1);
  const Real v_a = v0(b, TE::E2, 0, k, j  , i);
  const Real v_b = v0(b, TE::E2, 0, k, jm1, i);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real E3_2_NN_mod(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;
  const int km1 = k - (ndim > 2);
  const Real v_a = v0(b, TE::E3, 0, k  , j, i);
  const Real v_b = v0(b, TE::E3, 0, km1, j, i);
  return (std::abs(v_a) < std::abs(v_b)) ? v_a : v_b;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F1_2_NN_mod(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;	
  const int km1 = k - (ndim > 2);
  const int jm1 = j - (ndim > 1);
  const Real va = v0(b, TE::F1, 0, k  , j  , i);
  const Real vb = v0(b, TE::F1, 0, k  , jm1, i);
  const Real vc = v0(b, TE::F1, 0, km1, j  , i);
  const Real vd = v0(b, TE::F1, 0, km1, jm1, i);
  Real m = va;
  if (std::abs(vb) < std::abs(m)) m = vb;
  if (std::abs(vc) < std::abs(m)) m = vc;
  if (std::abs(vd) < std::abs(m)) m = vd;
  return m;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F2_2_NN_mod(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;	
  const int km1 = k - (ndim > 2);
  const int im1 = i - 1;
  const Real va = v0(b, TE::F2, 0, k  , j, i  );
  const Real vb = v0(b, TE::F2, 0, k  , j, im1);
  const Real vc = v0(b, TE::F2, 0, km1, j, i  );
  const Real vd = v0(b, TE::F2, 0, km1, j, im1);
  Real m = va;
  if (std::abs(vb) < std::abs(m)) m = vb;
  if (std::abs(vc) < std::abs(m)) m = vc;
  if (std::abs(vd) < std::abs(m)) m = vd;
  return m;
}
template <typename PackInterp>
KOKKOS_INLINE_FUNCTION
Real F3_2_NN_mod(const PackInterp &v0,
             const int &b, const int &k, const int &j, const int &i, int ndim) {
  using parthenon::TopologicalElement;	
  const int jm1 = j - (ndim > 1);
  const int im1 = i - 1;
  const Real va = v0(b, TE::F3, 0, k, j  , i  );
  const Real vb = v0(b, TE::F3, 0, k, jm1, i  );
  const Real vc = v0(b, TE::F3, 0, k, j  , im1);
  const Real vd = v0(b, TE::F3, 0, k, jm1, im1);
  Real m = va;
  if (std::abs(vb) < std::abs(m)) m = vb;
  if (std::abs(vc) < std::abs(m)) m = vc;
  if (std::abs(vd) < std::abs(m)) m = vd;
  return m;
}



} // namespace INTERP

#endif // INTERP_INTERP_HPP_
                          
