#ifndef XMHD_DEFS_HPP_
#define XMHD_DEFS_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

// Dimensional normalization factors (Cannot be changed as reduces the governing equations coefficients mostly to 1)
constexpr Real n0 = 6.e28; //2.41e26; //6.e28;   //3.e25;   //1.e12;     //1.e18; 
constexpr Real t0 = 100.e-9; //100.e-8; //1.e0;   
constexpr Real L0 = 1.e-3;   //1.e-3;   //1.e5;      //1.;   
constexpr Real B0 = 580.; //36.84;   //580.;    //1.2996;  //2.373e-5;  //0.4568;  // in Tesla

// Material property - e.g., Aluminium
constexpr Real m_Al = 27.;   //1.;   //- switch to hydrogen 1st
constexpr int Z_Al =  3.;    //1.;   

// Fundamental constants
constexpr Real pi = 3.14159265358979323846;
constexpr Real amu = 1.66053906892e-27;   // Atomic mass unit
constexpr Real me = 9.10938e-31;          // Electron mass
constexpr Real e_charge = 1.60217663e-19; // Elementary charge in Coulombs
constexpr Real c_light = 299792458.;      // Speed of light
constexpr Real eps0 = 8.8541878188e-12;   // Vacuum permittivity
constexpr Real mu0 = pi*4.e-7;            // Vacuum permeability
constexpr Real kB = 1.380649e-23;         // Boltzmann's constant
constexpr Real Coulomb_log = 5.;          // Coulomb logarithm - usually 5-15    
constexpr Real eV_to_K = 11600.;           // Convert 1eV to K

// Ion property
constexpr Real m_ion = m_Al * amu;
constexpr Real Z_ion = Z_Al;

// Derived constants
constexpr Real mu = Z_ion*me/m_ion;
constexpr Real char_speed = L0/t0;         // Characteristic speed (avoid using v from paper)
constexpr Real c_per_v = c_light / char_speed;
constexpr Real reduction_factor = 30.;     // Paper recommend 15-30 for HED problems
constexpr Real reduced_c = c_light/reduction_factor;
constexpr Real lambda_ion = std::sqrt(m_ion/(n0*SQR(e_charge)*mu0)); 
constexpr Real lambda_e = std::sqrt(me/(n0*SQR(e_charge)*mu0));
constexpr Real sigma0 = 1./(mu0*L0*char_speed);
constexpr Real E0 = char_speed * B0;
constexpr Real J0 = eps0*SQR(c_light)*B0/L0;
constexpr Real Press0 = n0*m_ion*SQR(char_speed);
constexpr Real T0 = Press0/(n0*kB*eV_to_K); // in eV 
constexpr Real vA_ion = B0/std::sqrt(mu0*n0*m_ion);  // Ion-Alfven speed (set equal to char_speed)
constexpr Real dens0 = n0*(me + m_ion/Z_ion);
constexpr Real Pe_coef = (m_ion/me)*(L0 / lambda_ion);

#endif // XMHD_DEFS_HPP_
