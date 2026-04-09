//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

// Artemis includes
#include "fill_derived.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"

#include "mhd/extended/defs.hpp"

using ArtemisUtils::EOS;
using ArtemisUtils::VI;

namespace ArtemisDerived {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus ArtemisDerived::SetAuxillaryFields(MeshData<Real> *md)
//! \brief Sets auxillary fields over IndexDomain::interior after an integration stage
//! NOTE(PDM): Note that this function is not called during remeshing.
template <Coordinates GEOM>
TaskStatus SetAuxillaryFields(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  if (!(do_gas)) return TaskStatus::complete;
  const bool do_mhd = artemis_pkg->template Param<bool>("do_mhd");

  // Extract gas parameters
  const Real dflr_gas = pm->packages.Get("gas").get()->template Param<Real>("dfloor");
  const Real sieflr_gas = pm->packages.Get("gas").get()->template Param<Real>("siefloor");
  const Real de_switch = pm->packages.Get("gas").get()->template Param<Real>("de_switch");

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy,
			 gas::cons::Bfield>(resolved_pkgs.get()); // YH: add mhd
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetAuxillaryFields", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        for (int n = 0; n < vmesh.GetSize(b, gas::cons::density()); ++n) {
          // Extract conserved density
          Real u_d = vmesh(b, gas::cons::density(n), k, j, i);
          u_d = (u_d > dflr_gas) ? u_d : dflr_gas;

          // Sync the internal energy with the total energy
          Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);
          u_u = ArtemisUtils::GetSpecificInternalEnergy<GEOM>(
                    vmesh, b, n, k, j, i, de_switch, dflr_gas, sieflr_gas, do_mhd) *
                u_d;

          // Apply internal energy floor
          const Real uflr_gas = sieflr_gas * u_d;
          u_u = (u_u > uflr_gas) ? u_u : uflr_gas;
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisDerived::ConsToPrim(MeshData<Real> *md)
//! \brief Exectues C2P over IndexDomain::interior after an integration stage
//! or remeshing event in preparation for FillGhost
template <Coordinates GEOM>
void ConsToPrim(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  const int ndim = md->GetMeshPointer()->ndim;

  // Extract gas parameters
  Real dflr_gas = Null<Real>();
  Real sieflr_gas = Null<Real>();
  Real gamma = Null<Real>();
  Real Peflr_mhd = Null<Real>();
  EOS eos_d;
  if (do_gas) {
    auto &gas_pkg = pm->packages.Get("gas");
    dflr_gas = gas_pkg->template Param<Real>("dfloor");
    sieflr_gas = gas_pkg->template Param<Real>("siefloor");
    gamma = gas_pkg->template Param<Real>("adiabatic_index");
    if (do_mhd) Peflr_mhd = gas_pkg->template Param<Real>("Pefloor");
  }

  // Extract dust parameters
  Real dflr_dust = Null<Real>();
  if (do_dust) {
    dflr_dust = pm->packages.Get("dust").get()->template Param<Real>("dfloor");
  }

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum,
                         gas::cons::internal_energy, gas::cons::Bfield,
			 gas::cons::Efield, gas::cons::J, gas::cons::Se,
			 gas::face::bfield, // YH: add fcc bfield
			 gas::edge::Efield, gas::edge::J,
			 gas::prim::Efield, gas::prim::J, gas::prim::Pe,
			 gas::prim::Bfield, gas::prim::density,
                         gas::prim::velocity, gas::prim::sie, gas::prim::pressure,
			 gas::prim::Ti, gas::prim::Te,
			 dust::cons::density,
                         dust::cons::momentum, dust::prim::density, dust::prim::velocity>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  const int nblocks = md->NumBlocks();
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  IndexRange ib_int = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb_int = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb_int = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConsToPrim", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &hx = coords.GetScaleFactors();

        if (do_gas) {
	  Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            // Set primitive density
            const Real u_d = vmesh(b, gas::cons::density(n), k, j, i);
            Real &w_d = vmesh(b, gas::prim::density(n), k, j, i);
	    //const bool woolstrum2022_stability = false;//u_d <= 1.01*dflr_gas ? true : false; 
            w_d = (u_d > dflr_gas) ? u_d : dflr_gas; // YH: enesure density does not go to negative by setting density floor

	    // Sync primtive sie, pressure, and conserved internal energy
            Real &w_s = vmesh(b, gas::prim::sie(n), k, j, i);
            Real &w_p = vmesh(b, gas::prim::pressure(n), k, j, i);
            Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);
            w_p = eos_d.PressureFromDensityInternalEnergy(w_d, w_s, lambda);
	    //printf("P2C: (%d,%d,%d): Press = %.6f, sie=%.6f, u = %.6f \n",i,j,k,w_p,w_s,u_u);

            // Set primitive velocity
            Real &vel1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            Real &vel2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            Real &vel3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            vel1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i) / (w_d * hx[0]);
            vel2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i) / (w_d * hx[1]);
            vel3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i) / (w_d * hx[2]);
	    /*if (woolstrum2022_stability) {
	      vel1 = 0.; vel2 = 0.; vel3 = 0.;
	    }*/

            // Set primitive specific internal energy
            const Real w_ie = vmesh(b, gas::cons::internal_energy(n), k, j, i) / w_d;	    
            vmesh(b, gas::prim::sie(n), k, j, i) = (w_ie > sieflr_gas) ? w_ie : sieflr_gas;

	    // YH: account for fcc magnetic field from mhd
	    if (do_mhd) {
	      vmesh(b, gas::prim::Bfield(0), k, j, i) = 
		      0.5 * (vmesh(b, TE::F1, gas::face::bfield(), k, j, i) +
			     vmesh(b, TE::F1, gas::face::bfield(), k, j, i + (ndim > 0)));
	      vmesh(b, gas::prim::Bfield(1), k, j, i) = 
		      0.5 * (vmesh(b, TE::F2, gas::face::bfield(), k, j, i) +
			     vmesh(b, TE::F2, gas::face::bfield(), k, j + (ndim > 1), i));
	      vmesh(b, gas::prim::Bfield(2), k, j, i) = 
		      0.5 * (vmesh(b, TE::F3, gas::face::bfield(), k, j, i) +
			     vmesh(b, TE::F3, gas::face::bfield(), k + (ndim > 2), j, i));

	      // YH: compute cell-centered Efield
	      vmesh(b, gas::prim::Efield(0), k, j, i) =
                      0.25 * (vmesh(b, TE::E1, gas::edge::Efield(), k, j, i) +
                             vmesh(b, TE::E1, gas::edge::Efield(), k, j + (ndim > 1), i) +
			     vmesh(b, TE::E1, gas::edge::Efield(), k + (ndim > 2), j, i) +
			     vmesh(b, TE::E1, gas::edge::Efield(), k + (ndim > 2), j + (ndim > 1), i));
	      vmesh(b, gas::prim::Efield(1), k, j, i) =
                      0.25 * (vmesh(b, TE::E2, gas::edge::Efield(), k, j, i) +
                             vmesh(b, TE::E2, gas::edge::Efield(), k, j, i + 1) +
                             vmesh(b, TE::E2, gas::edge::Efield(), k + (ndim > 2), j, i) +
                             vmesh(b, TE::E2, gas::edge::Efield(), k + (ndim > 2), j, i + 1));
	      vmesh(b, gas::prim::Efield(2), k, j, i) =
                      0.25 * (vmesh(b, TE::E3, gas::edge::Efield(), k, j, i) +
                             vmesh(b, TE::E3, gas::edge::Efield(), k, j, i + 1) +
                             vmesh(b, TE::E3, gas::edge::Efield(), k, j + (ndim > 1), i) +
                             vmesh(b, TE::E3, gas::edge::Efield(), k, j + (ndim > 1), i + 1));
	      // YH: compute cell-centered J
	      vmesh(b, gas::prim::J(0), k, j, i) =
                      0.25 * (vmesh(b, TE::E1, gas::edge::J(), k, j, i) +
                             vmesh(b, TE::E1, gas::edge::J(), k, j + (ndim > 1), i) +
                             vmesh(b, TE::E1, gas::edge::J(), k + (ndim > 2), j, i) +
                             vmesh(b, TE::E1, gas::edge::J(), k + (ndim > 2), j + (ndim > 1), i));
              vmesh(b, gas::prim::J(1), k, j, i) =
                      0.25 * (vmesh(b, TE::E2, gas::edge::J(), k, j, i) +
                             vmesh(b, TE::E2, gas::edge::J(), k, j, i + 1) +
                             vmesh(b, TE::E2, gas::edge::J(), k + (ndim > 2), j, i) +
                             vmesh(b, TE::E2, gas::edge::J(), k + (ndim > 2), j, i + 1));
              vmesh(b, gas::prim::J(2), k, j, i) =
                      0.25 * (vmesh(b, TE::E3, gas::edge::J(), k, j, i) +
                             vmesh(b, TE::E3, gas::edge::J(), k, j, i + 1) +
                             vmesh(b, TE::E3, gas::edge::J(), k, j + (ndim > 1), i) +
                             vmesh(b, TE::E3, gas::edge::J(), k, j + (ndim > 1), i + 1));
	      /*if (woolstrum2022_stability) { // YH: how apply for J at edge???
	         for (int ji=0; ji<3; ji++) vmesh(b, gas::prim::J(ji), k, j, i) = 0.;
	      }*/

	      // YH: compute electron Pressure
	      const Real ne = Z_ion * w_d;
	      Real w_Pe = vmesh(b, gas::cons::Se(), k, j, i)*pow(ne,gamma-1);
	      vmesh(b, gas::prim::Pe(), k, j, i) = (w_Pe > Peflr_mhd) ? w_Pe : Peflr_mhd;

	      // Compute cell-centered temperature (WHY IS w_p zero here???)
              /*vmesh(b, gas::prim::Ti(), k, j, i) = (vmesh(b, gas::prim::pressure(n), k, j, i)-vmesh(b, gas::prim::Pe(), k, j, i))/w_d;
	      if (vmesh(b, gas::prim::Ti(), k, j, i)<0.) {
		 printf("(%d,%d,%d): P=%.6f, Pe=%.6f, rho=%.6f \n",i,j,k,vmesh(b, gas::prim::pressure(n), k, j, i),vmesh(b, gas::prim::Pe(), k, j, i),w_d);
	      }
              vmesh(b, gas::prim::Te(), k, j, i) = vmesh(b, gas::prim::Pe(), k, j, i)/(Z_ion*w_d);*/

	    }
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Set primitive density
            const Real u_d = vmesh(b, dust::cons::density(n), k, j, i);
            Real &w_d = vmesh(b, dust::prim::density(n), k, j, i);
            w_d = (u_d > dflr_dust) ? u_d : dflr_dust;

            // set primitive velocity
            Real &vel1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
            Real &vel2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
            Real &vel3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);
            vel1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i) / (w_d * hx[0]);
            vel2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i) / (w_d * hx[1]);
            vel3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i) / (w_d * hx[2]);
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void ArtemisDerived::PrimToCons(MeshData<Real> *md)
//! \brief Executes P2C following integrator updates and/or remeshing events
template <typename T, Coordinates GEOM>
void PrimToCons(T *md) {
  using parthenon::MakePackDescriptor;
  using TE = parthenon::TopologicalElement;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  auto &artemis_pkg = pm->packages.Get("artemis");
  const bool do_gas = artemis_pkg->template Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->template Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->template Param<bool>("do_mhd");
  const int ndim = md->GetMeshPointer()->ndim;

  // Extract gas parameters
  Real dflr_gas = Null<Real>();
  Real sieflr_gas = Null<Real>();
  Real gamma = Null<Real>();
  Real Peflr_mhd = Null<Real>();
  EOS eos_d;
  if (do_gas) {
    auto &gas_pkg = pm->packages.Get("gas");
    dflr_gas = gas_pkg->template Param<Real>("dfloor");
    sieflr_gas = gas_pkg->template Param<Real>("siefloor");
    eos_d = gas_pkg->template Param<EOS>("eos_d");
    gamma = gas_pkg->template Param<Real>("adiabatic_index");
    Peflr_mhd = gas_pkg->template Param<Real>("Pefloor");
  }

  // Extract dust parameters
  Real dflr_dust = Null<Real>();
  if (do_dust) {
    dflr_dust = pm->packages.Get("dust").get()->template Param<Real>("dfloor");
  }

  // Packing and indexing
  static auto desc =
      MakePackDescriptor<gas::cons::density, gas::cons::momentum, gas::cons::total_energy,
                         gas::cons::internal_energy, gas::cons::Bfield,
			 gas::cons::divB, gas::face::bfield, // YH: for fcc bfield
			 gas::cons::divE, // YH: check for quasi-neutrality 
			 gas::cons::Efield, gas::cons::J, gas::cons::Se,
			 gas::edge::Efield, gas::edge::J,
			 gas::prim::Efield, gas::prim::J, gas::prim::Pe,
			 gas::prim::Ti, gas::prim::Te,
			 gas::prim::density, gas::prim::Bfield,
                         gas::prim::velocity, gas::prim::pressure, gas::prim::sie,
                         dust::cons::density, dust::cons::momentum, dust::prim::density,
                         dust::prim::velocity>(resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrimToCons", parthenon::DevExecSpace(), 0,
      vmesh.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Extract coordinates
        geometry::Coords<GEOM> coords(vmesh.GetCoordinates(b), k, j, i);
        const auto &xv = coords.GetCellCenter();
        const auto &hx = coords.GetScaleFactors();

        if (do_gas) {
          Real lambda[ArtemisUtils::lambda_max_vals] = {Null<Real>()};
          for (int n = 0; n < vmesh.GetSize(b, gas::prim::density()); ++n) {
            // Sync conserved and primitive density
            Real &w_d = vmesh(b, gas::prim::density(n), k, j, i);
            Real &u_d = vmesh(b, gas::cons::density(n), k, j, i);
	    //const bool woolstrum2022_stability = false;//w_d <= 1.01*dflr_gas ? true : false;
            w_d = (w_d > dflr_gas) ? w_d : dflr_gas;
            u_d = w_d;

            // Sync conserved momenta and primitive velocity
            const Real vel1 = vmesh(b, gas::prim::velocity(VI(n, 0)), k, j, i);
            const Real vel2 = vmesh(b, gas::prim::velocity(VI(n, 1)), k, j, i);
            const Real vel3 = vmesh(b, gas::prim::velocity(VI(n, 2)), k, j, i);
            Real &mom1 = vmesh(b, gas::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, gas::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, gas::cons::momentum(VI(n, 2)), k, j, i);
            mom1 = w_d * vel1 * hx[0];
            mom2 = w_d * vel2 * hx[1];
            mom3 = w_d * vel3 * hx[2];
	    /*if (woolstrum2022_stability) {
	      mom1 = 0.; mom2 = 0.; mom3 = 0.;
	    }*/

            // Sync primtive sie, pressure, and conserved internal energy
            Real &w_s = vmesh(b, gas::prim::sie(n), k, j, i);
            Real &w_p = vmesh(b, gas::prim::pressure(n), k, j, i);
            Real &u_u = vmesh(b, gas::cons::internal_energy(n), k, j, i);
            w_s = (w_s > sieflr_gas) ? w_s : sieflr_gas;
            u_u = w_s * u_d;
            w_p = eos_d.PressureFromDensityInternalEnergy(w_d, w_s, lambda);
	    //printf("C2P: (%d,%d,%d): Press = %.6f, sie=%.6f, u = %.6f \n",i,j,k,w_p,w_s,u_u);

            // Sync conserved total energy
            const Real ke = 0.5 * w_d * (SQR(vel1) + SQR(vel2) + SQR(vel3));
            Real &u_e = vmesh(b, gas::cons::total_energy(n), k, j, i);
            u_e = u_u + ke;

	    // YH: account for fcc magnetic energy for mhd
	    if (do_mhd) {
	      vmesh(b, gas::prim::Bfield(0), k, j, i) = 
		      0.5 * (vmesh(b, TE::F1, gas::face::bfield(), k, j, i) + 
			     vmesh(b, TE::F1, gas::face::bfield(), k, j, i + (ndim > 0)));
              vmesh(b, gas::prim::Bfield(1), k, j, i) = 
		      0.5 * (vmesh(b, TE::F2, gas::face::bfield(), k, j, i) +
			     vmesh(b, TE::F2, gas::face::bfield(), k, j + (ndim > 1), i));
              vmesh(b, gas::prim::Bfield(2), k, j, i) = 
		      0.5 * (vmesh(b, TE::F3, gas::face::bfield(), k, j, i) +
			     vmesh(b, TE::F3, gas::face::bfield(), k + (ndim>2), j, i));
	      const Real Bmag = 0.5 * (SQR(vmesh(b, gas::prim::Bfield(0), k, j, i)) 
			      + SQR(vmesh(b, gas::prim::Bfield(1), k, j, i)) 
			      + SQR(vmesh(b, gas::prim::Bfield(2), k, j, i)));
	      u_e += Bmag;
	      vmesh(b, gas::cons::Bfield(0), k, j, i) = vmesh(b, gas::prim::Bfield(0), k, j, i);
	      vmesh(b, gas::cons::Bfield(1), k, j, i) = vmesh(b, gas::prim::Bfield(1), k, j, i);
	      vmesh(b, gas::cons::Bfield(2), k, j, i) = vmesh(b, gas::prim::Bfield(2), k, j, i);

	      // YH: check divB too
	      const auto &dx = coords.GetCellWidths();
	      vmesh(b, gas::cons::divB(), k, j, i) = 
		      (vmesh(b, TE::F1, gas::face::bfield(), k, j, i + (ndim > 0)) - 
		       vmesh(b, TE::F1, gas::face::bfield(), k, j, i)) / dx[0] +
		      (vmesh(b, TE::F2, gas::face::bfield(), k, j + (ndim > 1), i) -
		       vmesh(b, TE::F2, gas::face::bfield(), k, j, i)) / dx[1] +
		      (vmesh(b, TE::F3, gas::face::bfield(), k + (ndim > 2), j, i) -
		       vmesh(b, TE::F3, gas::face::bfield(), k, j, i)) / dx[2];
	      // YH: check divE
	      vmesh(b, gas::cons::divE(), k, j, i) =
                      (vmesh(b, TE::E1, gas::edge::Efield(), k, j, i + (ndim > 0)) -
                       vmesh(b, TE::E1, gas::edge::Efield(), k, j, i)) / dx[0] +
                      (vmesh(b, TE::E2, gas::edge::Efield(), k, j + (ndim > 1), i) -
                       vmesh(b, TE::E2, gas::edge::Efield(), k, j, i)) / dx[1] +
                      (vmesh(b, TE::E3, gas::edge::Efield(), k + (ndim > 2), j, i) -
                       vmesh(b, TE::E3, gas::edge::Efield(), k, j, i)) / dx[2];

	      // YH: compute cell-centered Efield
              vmesh(b, gas::prim::Efield(0), k, j, i) =
                      0.25 * (vmesh(b, TE::E1, gas::edge::Efield(), k, j, i) +
                             vmesh(b, TE::E1, gas::edge::Efield(), k, j + (ndim > 1), i) +
                             vmesh(b, TE::E1, gas::edge::Efield(), k + (ndim > 2), j, i) +
                             vmesh(b, TE::E1, gas::edge::Efield(), k + (ndim > 2), j + (ndim > 1), i));
              vmesh(b, gas::prim::Efield(1), k, j, i) =
                      0.25 * (vmesh(b, TE::E2, gas::edge::Efield(), k, j, i) +
                             vmesh(b, TE::E2, gas::edge::Efield(), k, j, i + 1) +
                             vmesh(b, TE::E2, gas::edge::Efield(), k + (ndim > 2), j, i) +
                             vmesh(b, TE::E2, gas::edge::Efield(), k + (ndim > 2), j, i + 1));
              vmesh(b, gas::prim::Efield(2), k, j, i) =
                      0.25 * (vmesh(b, TE::E3, gas::edge::Efield(), k, j, i) +
                             vmesh(b, TE::E3, gas::edge::Efield(), k, j, i + 1) +
                             vmesh(b, TE::E3, gas::edge::Efield(), k, j + (ndim > 1), i) +
                             vmesh(b, TE::E3, gas::edge::Efield(), k, j + (ndim > 1), i + 1));
	      vmesh(b, gas::cons::Efield(0), k, j, i) = vmesh(b, gas::prim::Efield(0), k, j, i);
	      vmesh(b, gas::cons::Efield(1), k, j, i) = vmesh(b, gas::prim::Efield(1), k, j, i);
	      vmesh(b, gas::cons::Efield(2), k, j, i) = vmesh(b, gas::prim::Efield(2), k, j, i);
              // YH: compute cell-centered J
              vmesh(b, gas::prim::J(0), k, j, i) =
                      0.25 * (vmesh(b, TE::E1, gas::edge::J(), k, j, i) +
                             vmesh(b, TE::E1, gas::edge::J(), k, j + (ndim > 1), i) +
                             vmesh(b, TE::E1, gas::edge::J(), k + (ndim > 2), j, i) +
                             vmesh(b, TE::E1, gas::edge::J(), k + (ndim > 2), j + (ndim > 1), i));
              vmesh(b, gas::prim::J(1), k, j, i) =
                      0.25 * (vmesh(b, TE::E2, gas::edge::J(), k, j, i) +
                             vmesh(b, TE::E2, gas::edge::J(), k, j, i + 1) +
                             vmesh(b, TE::E2, gas::edge::J(), k + (ndim > 2), j, i) +
                             vmesh(b, TE::E2, gas::edge::J(), k + (ndim > 2), j, i + 1));
              vmesh(b, gas::prim::J(2), k, j, i) =
                      0.25 * (vmesh(b, TE::E3, gas::edge::J(), k, j, i) +
                             vmesh(b, TE::E3, gas::edge::J(), k, j, i + 1) +
                             vmesh(b, TE::E3, gas::edge::J(), k, j + (ndim > 1), i) +
                             vmesh(b, TE::E3, gas::edge::J(), k, j + (ndim > 1), i + 1));
	      vmesh(b, gas::cons::J(0), k, j, i) = vmesh(b, gas::prim::J(0), k, j, i);
	      vmesh(b, gas::cons::J(1), k, j, i) = vmesh(b, gas::prim::J(1), k, j, i);
  	      vmesh(b, gas::cons::J(2), k, j, i) = vmesh(b, gas::prim::J(2), k, j, i);
	      /*if (woolstrum2022_stability) { // YH: how apply for J at edge???
	        for (int ji=0; ji<3; ji++) {
		  vmesh(b, gas::prim::J(ji), k, j, i) = 0.;
		  vmesh(b, gas::cons::J(ji), k, j, i) = 0.;
		}
	      }*/

	      // YH: compute electron entropy density
              const Real ne = Z_ion * vmesh(b, gas::prim::density(n), k, j, i);
	      Real Pe = (vmesh(b, gas::prim::Pe(), k, j, i) > Peflr_mhd) ? vmesh(b, gas::prim::Pe(), k, j, i) : Peflr_mhd;
	      vmesh(b, gas::cons::Se(), k, j, i) = Pe / pow(ne,gamma-1);

	      // Compute cell-centered temperature
              vmesh(b, gas::prim::Ti(), k, j, i) = (vmesh(b, gas::prim::pressure(n), k, j, i)-vmesh(b, gas::prim::Pe(), k, j, i))/w_d;
              vmesh(b, gas::prim::Te(), k, j, i) = vmesh(b, gas::prim::Pe(), k, j, i)/(Z_ion*w_d);
	    }
          }
        }

        if (do_dust) {
          for (int n = 0; n < vmesh.GetSize(b, dust::prim::density()); ++n) {
            // Sync conserved and primitive density
            Real &w_d = vmesh(b, dust::prim::density(n), k, j, i);
            Real &u_d = vmesh(b, dust::cons::density(n), k, j, i);
            w_d = (w_d > dflr_dust) ? w_d : dflr_dust;
            u_d = w_d;

            // Sync conserved momenta and primitive velocity
            const Real vel1 = vmesh(b, dust::prim::velocity(VI(n, 0)), k, j, i);
            const Real vel2 = vmesh(b, dust::prim::velocity(VI(n, 1)), k, j, i);
            const Real vel3 = vmesh(b, dust::prim::velocity(VI(n, 2)), k, j, i);
            Real &mom1 = vmesh(b, dust::cons::momentum(VI(n, 0)), k, j, i);
            Real &mom2 = vmesh(b, dust::cons::momentum(VI(n, 1)), k, j, i);
            Real &mom3 = vmesh(b, dust::cons::momentum(VI(n, 2)), k, j, i);
            mom1 = w_d * vel1 * hx[0];
            mom2 = w_d * vel2 * hx[1];
            mom3 = w_d * vel3 * hx[2];
          }
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ArtemisDerived::PostInitialization(MeshBlock *pmb, ParameterInput *pin)
//! \brief Post-initialization hook which calls PrimToCons after ProblemGenerator
//! but before PreCommFillDerived
template <Coordinates GEOM>
void PostInitialization(MeshBlock *pmb, ParameterInput *pin) {
  auto &md = pmb->meshblock_data.Get();
  PrimToCons<MeshBlockData<Real>, GEOM>(md.get());
}

//----------------------------------------------------------------------------------------
//! \fn TaskCollection ArtemisDerived::SyncFields
//! \brief Syncs fields following an operator split update
template <Coordinates GEOM>
TaskCollection SyncFields(Mesh *pmesh, const Real time, const Real dt) {
  using namespace ::parthenon::Update;
  TaskCollection tc;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;

  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &post_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = post_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
    auto c2p = tl.AddTask(start_recv, PreCommFillDerived<MeshData<Real>>, u0.get());
    auto bcs = parthenon::AddBoundaryExchangeTasks(c2p, tl, u0, pmesh->multilevel);
    auto p2c = tl.AddTask(bcs, FillDerived<MeshData<Real>>, u0.get());
  }

  return tc;
}

//----------------------------------------------------------------------------------------
//! template instantiations
typedef Coordinates C;
typedef MeshBlock MB;
typedef MeshData<Real> MD;
typedef MeshBlockData<Real> MBD;
typedef ParameterInput PI;
template void ConsToPrim<C::cartesian>(MD *md);
template void ConsToPrim<C::cylindrical>(MD *md);
template void ConsToPrim<C::spherical1D>(MD *md);
template void ConsToPrim<C::spherical2D>(MD *md);
template void ConsToPrim<C::spherical3D>(MD *md);
template void ConsToPrim<C::axisymmetric>(MD *md);
template void PrimToCons<MBD, C::cartesian>(MBD *mbd);
template void PrimToCons<MBD, C::cylindrical>(MBD *mbd);
template void PrimToCons<MBD, C::spherical1D>(MBD *mbd);
template void PrimToCons<MBD, C::spherical2D>(MBD *mbd);
template void PrimToCons<MBD, C::spherical3D>(MBD *mbd);
template void PrimToCons<MBD, C::axisymmetric>(MBD *mbd);
template void PrimToCons<MD, C::cartesian>(MD *md);
template void PrimToCons<MD, C::cylindrical>(MD *md);
template void PrimToCons<MD, C::spherical1D>(MD *md);
template void PrimToCons<MD, C::spherical2D>(MD *md);
template void PrimToCons<MD, C::spherical3D>(MD *md);
template void PrimToCons<MD, C::axisymmetric>(MD *md);
template void PostInitialization<C::cartesian>(MB *pmb, PI *pin);
template void PostInitialization<C::cylindrical>(MB *pmb, PI *pin);
template void PostInitialization<C::spherical1D>(MB *pmb, PI *pin);
template void PostInitialization<C::spherical2D>(MB *pmb, PI *pin);
template void PostInitialization<C::spherical3D>(MB *pmb, PI *pin);
template void PostInitialization<C::axisymmetric>(MB *pmb, PI *pin);
template TaskStatus SetAuxillaryFields<C::cartesian>(MD *md);
template TaskStatus SetAuxillaryFields<C::cylindrical>(MD *md);
template TaskStatus SetAuxillaryFields<C::spherical1D>(MD *md);
template TaskStatus SetAuxillaryFields<C::spherical2D>(MD *md);
template TaskStatus SetAuxillaryFields<C::spherical3D>(MD *md);
template TaskStatus SetAuxillaryFields<C::axisymmetric>(MD *md);
template TaskCollection SyncFields<C::cartesian>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<C::cylindrical>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<C::spherical1D>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<C::spherical2D>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<C::spherical3D>(Mesh *m, const Real t, const Real dt);
template TaskCollection SyncFields<C::axisymmetric>(Mesh *m, const Real t, const Real dt);

} // namespace ArtemisDerived
