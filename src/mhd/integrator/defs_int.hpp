#ifndef XMHD_DEFS_INT_HPP_
#define XMHD_DEFS_INT_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace XMHD_INT{
/*
constexpr int nstages = 1; // 1st-order 
constexpr Kokkos::Array<Real, nstages> xi_exp{1.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_exp{0.};     // gam0
constexpr Kokkos::Array<Real, nstages> xi_imp{1.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_imp{0.};     // gam0
constexpr Kokkos::Array<Real, nstages> A_coef{1.};      // For implicit EJ
constexpr Kokkos::Array<Real, nstages> Aexp_coef{0.};    // For explicit EJ
constexpr Kokkos::Array<Real, nstages> B_coef{1.};      // For explicit flux
constexpr Kokkos::Array<Real, nstages> C_coef{1.};      // For explicit source
*/

// 2nd - order Strang splitting (From EF Toro)
// Source that show one using both explicit & implicit for Strang splitting:
// https://www.witpress.com/Secure/elibrary/papers/EURO99/EURO99105FU2.pdf
// https://ir.cwi.nl/pub/1429/1429D.pdf
// For 2nd-order 3-splitting method beyond Strang splitting: https://arxiv.org/pdf/2302.08034
/*
constexpr int nstages = 3;
constexpr Kokkos::Array<Real, nstages> xi_exp{0., 0., 0.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_exp{1., 1., 1.};     // gam0
constexpr Kokkos::Array<Real, nstages> xi_imp{0., 0., 0.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_imp{1., 1., 1.};     // gam0
constexpr Kokkos::Array<Real, nstages> Aexp_coef{0., 0., 0.};    // For explicit EJ
constexpr Kokkos::Array<Real, nstages> A_coef{0.5, 0.5, 0.0};      // For implicit EJ
constexpr Kokkos::Array<Real, nstages> B_coef{0.5, 0.0, 0.5};      // For explicit flux
constexpr Kokkos::Array<Real, nstages> C_coef{1.0, 0.0, 0.0};      // For explicit source
constexpr int nsubstages = 3;
constexpr Kokkos::Array<Real, nsubstages> gam0{0., 0.25, 2./3.};
constexpr Kokkos::Array<Real, nsubstages> gam1{1., 0.75, 1./3.};
constexpr Kokkos::Array<Real, nsubstages> beta{1., 0.25, 2./3.};
*/


// AK 3-2 (ii) 
/*
constexpr Kokkos::Array<Real, nstages> xi_exp{0., 0., 0.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_exp{1., 1., 1.};     // gam0
constexpr Kokkos::Array<Real, nstages> xi_imp{0., 0., 0.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_imp{1., 1., 1.};     // gam0
constexpr Kokkos::Array<Real, nstages> Aexp_coef{0., 0., 0.};    // For explicit EJ
constexpr Kokkos::Array<Real, nstages> A_coef{0.273890572734778059,
		0.438287559165397521, 0.287821868099824420};      // For implicit EJ
constexpr Kokkos::Array<Real, nstages> B_coef{0.662265355057626845, 
		0.0664399910533392230, 0.271294653889033932};      // For explicit flux
constexpr Kokkos::Array<Real, nstages> C_coef{0.316620935432115636,
		-0.0303736077786568570, 0.713752672346541221};      // For explicit source
*/

constexpr int nstages = 2;

// When explicit source and flux updates are unsplit
//constexpr int nstages = 2; // 2nd-order 
// (-) Broadwell model
/*
constexpr Real mu = (2.-sqrt(2.))/2.; // Original is 1./3.
constexpr Kokkos::Array<Real, nstages> xi_exp{0., 
				  2.*mu*mu-2.*mu+1.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_exp{1., 
				      -2.*mu*(mu-1.)};     // gam0
constexpr Kokkos::Array<Real, nstages> xi_imp{0., 
				  2.*mu*mu-2.*mu+1.};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_imp{1., 
				      -2.*mu*(mu-1.)};     // gam0
constexpr Kokkos::Array<Real, nstages> A_coef{
			(2.*mu-1.)/(2.*(mu-1.)), mu};      // For implicit EJ
constexpr Kokkos::Array<Real, nstages> Aexp_coef{
	- (2.*SQR(mu)-2.*mu+1.) / (2.*mu*(mu-1.)), 0.};    // For explicit EJ
constexpr Kokkos::Array<Real, nstages> B_coef{
		       1./(2.*mu), -1./(2.*(mu-1.))};      // For explicit flux
constexpr Kokkos::Array<Real, nstages> C_coef{
		       1./(2.*mu), -1./(2.*(mu-1.))};      // For explicit source
*/		   

// (a) Heun's method
constexpr Kokkos::Array<Real, nstages> xi_exp{1., 0.5};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_exp{0., 0.5};     // gam0
constexpr Kokkos::Array<Real, nstages> xi_imp{1., 0.5};      // gam1
constexpr Kokkos::Array<Real, nstages> eta_imp{0., 0.5};     // gam0
constexpr Kokkos::Array<Real, nstages> A_coef{1., 0.5};      // For implicit EJ
constexpr Kokkos::Array<Real, nstages> Aexp_coef{0., 0.};    // For explicit EJ
constexpr Kokkos::Array<Real, nstages> B_coef{1., 0.5};      // For explicit flux
constexpr Kokkos::Array<Real, nstages> C_coef{1., 0.5};      // For explicit source

// (b) Ralston's method: minimizes truncation error
// -> Tested implicit must be placed before explicit at least for large timestep
/*
constexpr Kokkos::Array<Real, nstages> xi_exp{1., 5./8.};    // gam1
constexpr Kokkos::Array<Real, nstages> eta_exp{0., 3./8.};   // gam0
constexpr Kokkos::Array<Real, nstages> xi_imp{1., 5./8.};    // gam1
constexpr Kokkos::Array<Real, nstages> eta_imp{0., 3./8.};   // gam0
constexpr Kokkos::Array<Real, nstages> A_coef{2./3., 3./4.}; // For implicit EJ
constexpr Kokkos::Array<Real, nstages> Aexp_coef{0., 0.};    // For explicit EJ
constexpr Kokkos::Array<Real, nstages> B_coef{2./3., 3./4.}; // For explicit flux
constexpr Kokkos::Array<Real, nstages> C_coef{2./3., 3./4.}; // For explicit source
*/

// (c) DIRK
/*
constexpr Real gamma = (2.-std::sqrt(2.))/2.;
constexpr Real delta = 1.-1./(2.*gamma);
constexpr Kokkos::Array<Real, nstages> xi_exp{1., 1.-delta/gamma};    // gam1
constexpr Kokkos::Array<Real, nstages> eta_exp{0., delta/gamma};      // gam0
constexpr Kokkos::Array<Real, nstages> xi_imp{1., 1./gamma};          // gam1
constexpr Kokkos::Array<Real, nstages> eta_imp{0., (1.-gamma)/gamma}; // gam0
constexpr Kokkos::Array<Real, nstages> A_coef{gamma, gamma};          // For implicit EJ
constexpr Kokkos::Array<Real, nstages> Aexp_coef{0., 1.-gamma};       // For explicit EJ
constexpr Kokkos::Array<Real, nstages> B_coef{gamma, 1.-delta};       // For explicit flux
constexpr Kokkos::Array<Real, nstages> C_coef{gamma, 1.-delta};       // For explicit source
*/


// From Athenak: https://github.com/IAS-Astrophysics/athenak/blob/main/src/driver/driver.cpp
// IMEX-SSP2(3,2,2): Pareschi & Russo (2005) Table III.
// two-stage explicit, three-stage implicit, second-order ImEx
// Note explicit steps identical to RK2
/*nimp_stages = 3;
nexp_stages = 2;
cfl_limit = 1.0;
gam0[0] = 1.0;
gam1[0] = 0.0;
beta[0] = 1.0;

gam0[1] = 0.5;
gam1[1] = 0.5;
beta[1] = 0.5;

a_twid[0][0] = -1.0;
a_twid[0][1] = 0.0;
a_twid[0][2] = 0.0;

a_twid[1][0] = 0.5;
a_twid[1][1] = 0.0;
a_twid[1][2] = 0.0;

a_twid[2][0] = 0.0;
a_twid[2][1] = 0.25;
a_twid[2][2] = 0.25;
a_impl = 0.5;*/
}

#endif // XMHD_DEFS_INT_HPP_
