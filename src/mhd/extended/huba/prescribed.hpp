#ifndef PRESCRIBED_PRESCRIBED_HPP_
#define PRESCRIBED_PRESCRIBED_HPP_

// Parthenon includes
#include <parthenon/package.hpp>

// Artemis includes
#include "artemis.hpp"

#include "defs.hpp"

using namespace parthenon::package::prelude;
using parthenon::MetadataFlag;

namespace prescribed {

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus prescribed::init_cond
//! \brief Prescribed initial conditions for rho & T/P 
template <Coordinates GEOM>
TaskStatus init_cond(MeshData<Real> *u0) {
  using parthenon::MakePackDescriptor;
  using parthenon::variable_names::any;
  auto pm = u0->GetParentPointer();
  auto &artemis_pkg = pm->packages.Get("artemis");
  const Real At = 20.; // Atwood number
  const Real B0norm = 0.1/B0;
  const Real deltaB = 0.001/B0;
  const Real plasma_beta = 0.0001;
  const Real press = plasma_beta*SQR(0.1)/(2.*mu0);
  const Real P0 = n0*kB*T0*eV_to_K; // Note T0 is in eV!
  const Real Pfac = press/P0;
  const Real Ly = 0.2;

  const bool multi_d = (pm->ndim > 1);
  const bool three_d = (pm->ndim > 2);
  const int ndim = pm->ndim;

  // YH: extract cc rho & vel
  std::vector<MetadataFlag> flags_rho({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Density")});
  static auto desc_rho = MakePackDescriptor<any>(u0, flags_rho);
  const auto v0_rho = desc_rho.GetPack(u0);
  std::vector<MetadataFlag> flags_P({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("Pressure")});
  static auto desc_P = MakePackDescriptor<any>(u0, flags_P);
  const auto v0_P = desc_P.GetPack(u0);
  std::vector<MetadataFlag> flags_Pe({Metadata::Cell, Metadata::Derived,
                  Metadata::GetUserFlag("electron")});
  static auto desc_Pe = MakePackDescriptor<any>(u0, flags_Pe);
  const auto v0_Pe = desc_Pe.GetPack(u0);
  std::vector<MetadataFlag> flags_vel({Metadata::Cell, Metadata::Derived, 
                  Metadata::GetUserFlag("Velfield")});
  static auto desc_vel = MakePackDescriptor<any>(u0, flags_vel);
  const auto v0_vel = desc_vel.GetPack(u0);

  const auto ib = u0->GetBoundsI(IndexDomain::interior);
  const auto jb = u0->GetBoundsJ(IndexDomain::interior);
  const auto kb = u0->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplySourceUpdate", parthenon::DevExecSpace(), 0,
      u0->NumBlocks() - 1, kb.s-2*(ndim>2), kb.e+(ndim>2), jb.s-2*(ndim>1), jb.e+(ndim>1), ib.s-2, ib.e+1,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        using parthenon::TopologicalElement;
        geometry::Coords<GEOM> coords(v0_rho.GetCoordinates(b), k, j, i);
	const auto &xi = coords.GetCellCenter();
        const auto &dx = coords.GetCellWidths();
	const Real left_right = xi[0]>=0. ? 1. : -1.;
        const Real n0_fac = 1.e18 / n0;
	const Real nx = 0.5*n0_fac*((1.+At) + 
			(1.-At)*std::tanh(left_right*(xi[0]-left_right*0.1)/0.03));      
	const Real nx_x0 = 0.5*n0_fac*((1.+At) + 
			(1.-At)*std::tanh(left_right*(0.-left_right*0.1)/dx[0])); 
	const Real rho = nx;   
        const Real P = Pfac*nx/nx_x0;
	v0_vel(b, 0, k, j, i) = 0.;
	//v0_vel(b, 1, k, j, i) = 0.;
	//v0_rho(b, k, j, i) = rho;
	//v0_P(b, k, j, i) = P;
	//v0_Pe(b, k, j, i) = 0.5*P;
	});

  return TaskStatus::complete; 
}

} // prescribed

#endif // PRESCRIBED_PRESCRIBED_HPP_
