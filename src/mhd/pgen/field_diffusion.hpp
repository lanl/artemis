//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef PGEN_FIELD_DIFFUSION_HPP_
#define PGEN_FIELD_DIFFUSION_HPP_
//! \file field_diffusion.hpp
//! \brief
//!

// artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "pgen/pgen.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "mhd/extended/defs.hpp"
#include "utils/integrators/artemis_integrator.hpp"

using ArtemisUtils::EOS;

namespace field_diffusion {

struct FIELD_DIFFUSION_Params {
  Real rho0, Amp, eta;
  Real xdisc;
};

//----------------------------------------------------------------------------------------
//! \fn void Init_FIELD_DIFFUSION_Params
//! \brief Extracts field_diffusion parameters from ParameterInput.
inline void Init_FIELD_DIFFUSION_Params(MeshBlock *pmb, ParameterInput *pin) {
  auto &artemis_pkg = pmb->packages.Get("artemis");
  Params &params = artemis_pkg->AllParams();
  if (!(params.hasKey("field_diffusion_params"))) {
    FIELD_DIFFUSION_Params field_diffusion_params;
    field_diffusion_params.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    field_diffusion_params.Amp = pin->GetOrAddReal("problem", "Amp", 1.e-6);
    // YH: for now just match my eta to my resistivity
    field_diffusion_params.eta = pin->GetOrAddReal("problem", "eta", 1.e-8);
    field_diffusion_params.xdisc = pin->GetOrAddReal("problem", "xdisc", 0.5);
    params.Add("field_diffusion_params", field_diffusion_params);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::FIELD_DIFFUSION()
//! \brief Sets initial conditions for field_diffusion problem
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MakePackDescriptor;
  int nghosts = parthenon::Globals::nghost;

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_mhd = artemis_pkg->Param<bool>("do_mhd");
  PARTHENON_REQUIRE(do_gas || do_mhd, "The field diffusion problem requires both gas & mhd!");
  PARTHENON_REQUIRE(!(do_dust), "The field diffusion problem does not permit dust hydrodynamics!");
  auto eos_d = pmb->packages.Get("gas")->Param<EOS>("eos_d");
  auto gamma = pin->GetOrAddReal("gas","gamma",2.0);
  Real gm1 = gamma - 1.;
  const int ndim = ProblemDimension(pin);

  // packing and capture variables for kernel
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }
  // -> Init prim.B & press so that BCs whichneed these will not run into segfault 
  //    during initialization
  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
	  gas::prim::Pe,
	  gas::prim::pressure,gas::prim::Bfield>(
          (pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto &pco = pmb->coords;

  // FIELD_DIFFUSION parameters
  auto shkp = artemis_pkg->Param<FIELD_DIFFUSION_Params>("field_diffusion_params");
  const Real rho0 = shkp.rho0;
  const Real Amp = shkp.Amp;
  const Real t0 = 0.5;
  //const Real D = 0.25; // Ambipolar diffusion: not included in my GOL so ignore
  const Real eta = shkp.eta;

  // Setup field_diffusion state
  pmb->par_for(
      "field_diffusion", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xi = coords.GetCellCenter();
	const Real rho = 1.;
	const Real cs0 = 1.;
	const Real P = rho*SQR(cs0);
	const Real Pe = 0.5*P;
	const Real x2 = SQR(xi[0]);
	const Real y2 = ndim>1 ? SQR(xi[1]) : 0.;
	const Real z2 = ndim>2 ? SQR(xi[2]) : 0.;
	const Real by = (Amp/pow(std::sqrt(4.*M_PI*eta*t0),ndim)) *
			std::exp(-(x2+y2+z2)/(4.*eta*t0));
        v(0, gas::prim::density(0), k, j, i) = rho*rho0;
        v(0, gas::prim::velocity(0), k, j, i) = 0.0;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(0), k, j, i) = P*rho0/(rho*rho0*gm1);
	v(0, gas::prim::pressure(0), k, j, i) = P*rho0;
	v(0, gas::prim::Bfield(0), k, j, i) = 0.;
	v(0, gas::prim::Bfield(1), k, j, i) = by*sqrt(rho0);
	v(0, gas::prim::Bfield(2), k, j, i) = 0.;
	v(0, gas::prim::Pe(0), k, j, i) = Pe*rho0;
      });

  static auto desc_mag =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get(),
        {parthenon::Metadata::WithFluxes});
  auto vmag = desc_mag.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F1); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F1);
  pmb->par_for(
      "field_diffusion", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
	const auto &xf = coords.GetFaceCenterX1();
	const bool upwind = (xf[0] <= shkp.xdisc);
	// YH: add magnetic field - refer to problem gen from fine_adv
	vmag(0, TE::F1, gas::face::bfield(0), k, j, i) = 0.;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F2); 
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F2);
  pmb->par_for(
      "field_diffusion", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX2();
	const Real x2 = SQR(xf[0]);
        const Real y2 = ndim>1 ? SQR(xf[1]) : 0.;
        const Real z2 = ndim>2 ? SQR(xf[2]) : 0.;
        const Real by = (Amp/pow(std::sqrt(4.*M_PI*eta*t0),ndim)) *
                        std::exp(-(x2+y2+z2)/(4.*eta*t0));
        vmag(0, TE::F2, gas::face::bfield(0), k, j, i) = by*sqrt(rho0);
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::F3);       
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::F3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::F3);
  pmb->par_for(
      "field_diffusion", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &xf = coords.GetFaceCenterX3();
        vmag(0, TE::F3, gas::face::bfield(0), k, j, i) = 0.;
      });

  static auto desc_EJ =
      MakePackDescriptor<gas::edge::Efield,
      			 gas::edge::J>((pmb->resolved_packages).get());
  auto vEJ = desc_EJ.GetPack(md.get());
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E1);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E1);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E1);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E1, gas::edge::Efield(), k, j, i) = 0.;
	Real dBz_dy = -(vmag(0, TE::F3, 0, k, j - (ndim > 1), i)
                        - vmag(0, TE::F3, 0, k, j, i)) / dx[1];
        Real dBy_dz = -(vmag(0, TE::F2, 0, k - (ndim > 2), j, i)
                        - vmag(0, TE::F2, 0, k, j, i)) / dx[2];
        vEJ(0, TE::E1, gas::edge::J(), k, j, i) = dBz_dy - dBy_dz;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E2);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E2);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E2);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E2, gas::edge::Efield(), k, j, i) = 0.;
	Real dBz_dx = -(vmag(0, TE::F3, 0, k, j, i - (i>0))
                        - vmag(0, TE::F3, 0, k, j, i)) / dx[0];
        Real dBx_dz = -(vmag(0, TE::F1, 0, k - (ndim > 2), j, i)
                        - vmag(0, TE::F1, 0, k, j, i)) / dx[2];
        vEJ(0, TE::E2, gas::edge::J(), k, j, i) = -dBz_dx + dBx_dz;
      });
  ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, TE::E3);
  jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, TE::E3);
  kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, TE::E3);
  pmb->par_for(
      "blast_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(pco, k, j, i);
        const auto &dx = coords.GetCellWidths();
	vEJ(0, TE::E3, gas::edge::Efield(), k, j, i) = 0.;
	Real dBy_dx = -(vmag(0, TE::F2, 0, k, j, i - (i>0))
                        - vmag(0, TE::F2, 0, k, j, i)) / dx[0];
        Real dBx_dy = -(vmag(0, TE::F1, 0, k, j - (ndim > 1), i)
                        - vmag(0, TE::F1, 0, k, j, i)) / dx[1];
        vEJ(0, TE::E3, gas::edge::J(), k, j, i) = dBy_dx - dBx_dy;
      });

}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Computes errors in field diffusion mhd solution by subtracting current solution from
//! ICs, and outputting errors to file. Problem must be run for an integer number of wave
//! periods.
template <Coordinates GEOM>
inline void UserWorkAfterLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  using parthenon::MakePackDescriptor;
  const int nhydro = 5;
  const int nvars = nhydro + 3 + 7;

  // packing and capture variables for kernel
  auto &md = pmesh->mesh_data.GetOrAdd("base", 0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  static auto desc =
      MakePackDescriptor<gas::face::bfield>((pmb->resolved_packages).get());
  auto vmag = desc.GetPack(md.get());
  const int ndim = ProblemDimension(pin);

  // Extract parameters from packages
  auto artemis_pkg = pmb->packages.Get("artemis");
  // FIELD_DIFFUSION parameters
  auto shkp = artemis_pkg->Param<FIELD_DIFFUSION_Params>("field_diffusion_params");
  const Real rho0 = shkp.rho0;
  const Real Amp = shkp.Amp;
  const Real t0 = 0.5;
  const Real eta = shkp.eta;


  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, TE::F2);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, TE::F2);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, TE::F2);
  ArtemisUtils::array_type<Real, 1> l1_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "field_diffusion", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    ArtemisUtils::array_type<Real, 1> &lsum) {
        // Capture coordinates this Meshblock
        geometry::Coords<GEOM> coords(vmag.GetCoordinates(b), k, j, i);
	const auto &xf = coords.GetFaceCenterX2();
        const auto &xv = coords.GetCellCenter();
        Real x1v = xv[0];
        Real x2v = xv[1];
        Real x3v = xv[2];
        Real vol = coords.Volume();
	const auto &dx = coords.GetCellWidths();
	const Real x2 = SQR(xf[0]);
        const Real y2 = ndim>1 ? SQR(xf[1]) : 0.;
        const Real z2 = ndim>2 ? SQR(xf[2]) : 0.;
        const Real by = (Amp/pow(std::sqrt(4.*M_PI*eta*(t0+tm.time)),ndim)) *
                        std::exp(-(x2+y2+z2)/(4.*eta*(t0+tm.time)));
        lsum.myArray[0] += abs(vmag(0, TE::F2, gas::face::bfield(0), k, j, i) - by*sqrt(rho0))*dx[0];
      },
      ArtemisUtils::SumMyArray<Real, Kokkos::HostSpace, 1>(l1_err));
  Kokkos::fence();

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &(l1_err.myArray[0]), 1, MPI_PARTHENON_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  Real rms_err = l1_err.myArray[0];

  // root process opens output file and writes out errors
  if (parthenon::Globals::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("parthenon/job", "problem_id"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        PARTHENON_FAIL("Error output file could not be opened");
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1         ");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmesh->mesh_size.nx(X1DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %04d", pmesh->mesh_size.nx(X3DIR));
    std::fprintf(pfile, "  %05d  %e ", tm.ncycle, rms_err);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}


} // namespace field_diffusion
#endif // PGEN_FIELD_DIFFUSION_HPP_
