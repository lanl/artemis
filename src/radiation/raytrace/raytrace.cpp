// Artemis includes
#include "raytrace.hpp"
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"
#include "utils/eos/eos.hpp"
#include "utils/opacity/opacity.hpp"
#include "utils/units.hpp"

namespace RT {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin,
                                            ArtemisUtils::Units &units,
                                            ArtemisUtils::Constants &constants) {
  using namespace singularity::photons;

  auto rt = std::make_shared<StateDescriptor>("raytrace");
  Params &params = rt->AllParams();

  // Opacity models
  const Real time = units.GetTimeCodeToPhysical();
  const Real mass = units.GetMassCodeToPhysical();
  const Real length = units.GetLengthCodeToPhysical();
  const Real temp = units.GetTemperatureCodeToPhysical();

  std::string band = pin->GetOrAddString("radiation/raytrace", "band", "uv");
  params.Add("band", band);

  std::string block_name = "gas/opacity/" + band + "/absorption";

  // Absorption opacity model
  std::string opacity_model_name =
      pin->GetOrAddString(block_name, "opacity_model", "constant");
  ArtemisUtils::Opacity opacity;
  if (opacity_model_name == "constant") {
    const Real kappa_a = pin->GetOrAddReal(block_name, "kappa_a", 0.0);
    opacity = NonCGSUnits<Gray>(Gray(kappa_a), time, mass, length, temp);
  } else {
    PARTHENON_FAIL("Opacity model not recognized!");
  }
  params.Add("opacity_h", opacity);
  params.Add("opacity_d", opacity.GetOnDevice());

  const Real stellar_temp = pin->GetReal("radiation/raytrace", "temperature_cgs");
  Real stellar_radius = pin->GetReal("radiation/raytrace", "radius_solar");
  stellar_radius *= constants.GetRsolarPhysical();
  const Real sb = 0.25 * constants.GetCPhysical() * constants.GetARPhysical();
  Real luminosity_cgs = 4 * M_PI * SQR(stellar_radius) * sb * SQR(SQR(stellar_temp));
  params.Add("luminosity", luminosity_cgs * units.GetLuminosityPhysicalToCode());
  params.Add("luminosity_cgs", luminosity_cgs);
  params.Add("stellar_temp", stellar_temp * units.GetTemperaturePhysicalToCode());
  params.Add("stellar_radius", stellar_radius * units.GetLengthPhysicalToCode());
  params.Add("max_iterations",
             pin->GetOrAddInteger("radiation/raytrace", "max_iterations", 1000));

  params.Add("efloor", pin->GetOrAddReal("radiation/raytrace", "efloor", 1e-10));

  // Note that these pull the real x1 values, not the ones from the artemis package
  params.Add("x1min", pin->GetReal("parthenon/mesh", "x1min"));
  params.Add("x1max", pin->GetReal("parthenon/mesh", "x1max"));
  params.Add("x2min", pin->GetReal("parthenon/mesh", "x2min"));
  params.Add("x2max", pin->GetReal("parthenon/mesh", "x2max"));
  params.Add("nx2", pin->GetInteger("parthenon/mesh", "nx2"));
  params.Add("x3min", pin->GetReal("parthenon/mesh", "x3min"));
  params.Add("x3max", pin->GetReal("parthenon/mesh", "x3max"));
  params.Add("nx3", pin->GetInteger("parthenon/mesh", "nx3"));

  // Swarm and swarm variables
  parthenon::Metadata swarm_metadata({Metadata::None});
  rt->AddSwarm("star", swarm_metadata);
  parthenon::Metadata mreal({parthenon::Metadata::Real});
  rt->AddSwarmValue(rad::part::flux::name(), "star", mreal);
  parthenon::Metadata mintv({Metadata::Integer, Metadata::Vector}, std::vector<int>{3});
  rt->AddSwarmValue(rad::part::ijk::name(), "star", mintv);

  Metadata m = Metadata({Metadata::Cell});
  rt->AddField<rad::opac::cross_section>(m);
  rt->AddField<gas::src::energy>(m);

  return rt;
}

// Taken from jaybenne
TaskStatus MeshResetCommunication(MeshData<Real> *md) {
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->ResetCommunication();
  }

  return TaskStatus::complete;
}

TaskStatus MeshSend(MeshData<Real> *md) {
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    sc->Send(BoundaryCommSubset::all);
  }

  return TaskStatus::complete;
}

TaskStatus MeshReceive(MeshData<Real> *md) {
  TaskStatus status = TaskStatus::complete;
  const int nblocks = md->NumBlocks();
  for (int n = 0; n < nblocks; n++) {
    auto &mbd = md->GetBlockData(n);
    auto &sc = mbd->GetSwarmData();
    auto local_status = sc->Receive(BoundaryCommSubset::all);
    if (local_status == TaskStatus::incomplete) {
      status = TaskStatus::incomplete;
    }
  }

  return status;
}

TaskStatus SourceParticles(MeshData<Real> *md, const ParticleWeights &pwght) {
  auto pm = md->GetParentPointer();
  auto &artemis_pkg = pm->packages.Get("artemis");
  auto geom = artemis_pkg->Param<Coordinates>("coords");
  auto cpars = artemis_pkg->Param<geometry::CoordParams>("coord_params");
  switch (geom) {
  case Coordinates::spherical1D: {
    if (cpars.log) {
      return SourceParticlesImpl<Coordinates::spherical1D, true>(md, pwght);
    } else {
      return SourceParticlesImpl<Coordinates::spherical1D, false>(md, pwght);
    }
  }
  case Coordinates::spherical2D: {
    if (cpars.log) {
      return SourceParticlesImpl<Coordinates::spherical2D, true>(md, pwght);
    } else {
      return SourceParticlesImpl<Coordinates::spherical2D, false>(md, pwght);
    }
  }
  case Coordinates::spherical3D: {
    if (cpars.log) {
      return SourceParticlesImpl<Coordinates::spherical3D, true>(md, pwght);
    } else {
      return SourceParticlesImpl<Coordinates::spherical3D, false>(md, pwght);
    }
  }
  default:
    PARTHENON_FAIL("Unsupported geometry in raytracing");
  }
  return TaskStatus::complete;
}

TaskStatus PushParticles(MeshData<Real> *md) {

  auto pm = md->GetParentPointer();
  auto &artemis_pkg = pm->packages.Get("artemis");
  auto geom = artemis_pkg->Param<Coordinates>("coords");
  auto cpars = artemis_pkg->Param<geometry::CoordParams>("coord_params");
  switch (geom) {
  case Coordinates::spherical1D: {
    if (cpars.log) {
      return PushParticlesImpl<Coordinates::spherical1D, true>(md);
    } else {
      return PushParticlesImpl<Coordinates::spherical1D, false>(md);
    }
  }
  case Coordinates::spherical2D: {
    if (cpars.log) {
      return PushParticlesImpl<Coordinates::spherical2D, true>(md);
    } else {
      return PushParticlesImpl<Coordinates::spherical2D, false>(md);
    }
  }
  case Coordinates::spherical3D: {
    if (cpars.log) {
      return PushParticlesImpl<Coordinates::spherical3D, true>(md);
    } else {
      return PushParticlesImpl<Coordinates::spherical3D, false>(md);
    }
  }
  default:
    PARTHENON_FAIL("Unsupported geometry in raytracing");
  }
  return TaskStatus::complete;
}

TaskStatus RemoveParticles(MeshData<Real> *md) {

  for (int b = 0; b < md->NumBlocks(); ++b) {
    md->GetSwarmData(b)->Get("star")->RemoveMarkedParticles();
  }

  return TaskStatus::complete;
}

TaskStatus CheckCompletion(MeshData<Real> *md) {
  // Taken from jaybenne
  auto pm = md->GetParentPointer();
  // Create SwarmPacks
  static auto pdesc_r =
      MakeSwarmPackDescriptor<swarm_position::x, rad::part::flux>("star");
  auto ppack_r = pdesc_r.GetPack(md);
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();

  auto &rt_pkg = pm->packages.Get("raytrace");
  auto x1max = rt_pkg->Param<Real>("x1max");

  int num_unfinished = 0;
  parthenon::par_reduce(
      "CheckCompletion", 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx, int &num_unfinished) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          const Real &xp = ppack_r(b, swarm_position::x(), n);
          const bool alive = ppack_r(b, rad::part::flux(), n) > 0.0;
          const bool outside =
              (xp >= x1max) ||
              (std::abs(xp - x1max) < 10 * std::numeric_limits<Real>::epsilon());
          num_unfinished += (alive && !outside);
        }
      },
      Kokkos::Sum<int>(num_unfinished));

  const bool keep_going = (num_unfinished > 0);
  return (keep_going) ? TaskStatus::iterate : TaskStatus::complete;
}

TaskStatus EvalOpac(MeshData<Real> *md) {
  using parthenon::MakePackDescriptor;
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;
  auto &rt_pkg = pm->packages.Get("raytrace");
  auto &gas_pkg = pm->packages.Get("gas");

  ArtemisUtils::EOS eos_d = gas_pkg->template Param<ArtemisUtils::EOS>("eos_d");
  ArtemisUtils::Opacity opacity_d =
      rt_pkg->template Param<ArtemisUtils::Opacity>("opacity_d");

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::sie, rad::opac::cross_section>(
          resolved_pkgs.get());
  auto vmesh = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  // Set opacities
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "EvalOpac", parthenon::DevExecSpace(), 0, md->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const Real &rho = vmesh(b, gas::prim::density(), k, j, i);
        const Real &sie = vmesh(b, gas::prim::sie(), k, j, i);
        //%%%%%%%%%%%%%%%%
        // Evaluated at T*
        //%%%%%%%%%%%%%%%%
        const Real temp = eos_d.TemperatureFromDensityInternalEnergy(rho, sie);
        const Real kappa = opacity_d.AbsorptionCoefficient(rho, temp, 1.0);
        vmesh(b, rad::opac::cross_section(), k, j, i) = kappa;
      });

  return TaskStatus::complete;
}

TaskCollection RaytraceDriverTasks(Mesh *pmesh, const ParticleWeights &pwght) {
  using TQ = TaskQualifier;
  auto &rt_pkg = pmesh->packages.Get("raytrace");

  const auto max_iterations = rt_pkg->template Param<int>("max_iterations");

  TaskCollection tc;
  TaskID none(0);

  const int num_partitions = pmesh->DefaultNumPartitions();
  PARTHENON_REQUIRE(
      num_partitions == 1,
      "Iterative tasking may not support multiple partitions per rank as of 2024/5/14")
  auto &reg = tc.AddRegion(num_partitions);

  for (int i = 0; i < num_partitions; i++) {
    auto &tl = reg[i];
    // Get base register for particle
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);

    // prepare for iterative transport loop
    auto source = tl.AddTask(none, SourceParticles, base.get(), pwght);
    auto opac = tl.AddTask(none, EvalOpac, base.get());

    // keep pushing particles until there are none left
    auto [itl, push] = tl.AddSublist(source | opac, {1, max_iterations});
    auto transport = itl.AddTask(none, PushParticles, base.get());
    auto reset_comms = itl.AddTask(transport, MeshResetCommunication, base.get());
    auto send = itl.AddTask(reset_comms, MeshSend, base.get());
    auto receive = itl.AddTask(transport | send, MeshReceive, base.get());

    auto complete = itl.AddTask(TQ::once_per_region | TQ::global_sync | TQ::completion,
                                receive, CheckCompletion, base.get());
    auto remove = tl.AddTask(push, RemoveParticles, base.get());
  }
  // Delete particles

  return tc;
}

TaskListStatus RaytraceDriver(Mesh *pmesh) {
  auto &artemis_pkg = pmesh->packages.Get("artemis");
  auto geom = artemis_pkg->Param<Coordinates>("coords");
  // What is the minimum dtheta, dphi
  auto &rt_pkg = pmesh->packages.Get("raytrace");

  const auto x2min = rt_pkg->Param<Real>("x2min");
  const auto x2max = rt_pkg->Param<Real>("x2max");
  const auto nx2 = rt_pkg->Param<int>("nx2");

  const auto x3min = rt_pkg->Param<Real>("x3min");
  const auto x3max = rt_pkg->Param<Real>("x3max");
  const auto nx3 = rt_pkg->Param<int>("nx3");

  // Max level to find the minimum x2 and x3 spacings
  int max_level = pmesh->GetCurrentLevel();
  const int fac = 1 << max_level;

  // defaults to 1D
  ParticleWeights pwght(x2min, x2max, x3min, x3max);
  pwght.mult = fac;
  pwght.max_level = max_level;
  int reduc = 1;
  if (nx2 > 1) {
    pwght.dx2 = (x2max - x2min) / (nx2 * fac);
    reduc *= nx2 * fac;
  }
  if (nx3 > 1) {
    pwght.dx3 = (x3max - x3min) / (nx3 * fac);
    reduc *= nx3 * fac;
  }

  return RaytraceDriverTasks(pmesh, pwght).Execute();
}

} // namespace RT
