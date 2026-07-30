//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was created in part by generative AI

#include "function_init.hpp"

#include <algorithm>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <globals.hpp>
#include <rummy/full_deck.hpp>

namespace artemis {
namespace function_init {
namespace {

struct DeviceModuleStorage {
  ParArray1D<std::uint8_t> code;
  ParArray1D<pips::device::DeviceValue> constants;
  ParArray1D<pips::device::DeviceFunction> functions;
  ParArray1D<pips::device::DeviceValue> globals;
};

struct StorageRegistry {
  std::vector<DeviceModuleStorage> modules;
};

template <typename T>
ParArray1D<T> Upload(const std::string &label, const T *data, const std::uint32_t count) {
  if (count == 0) return {};
  ParArray1D<T> view(label, count);
  auto host = view.GetHostMirror();
  for (std::uint32_t n = 0; n < count; ++n)
    host(n) = data[n];
  view.DeepCopy(host);
  return view;
}

std::string RequireFunctionName(ParameterInput *pin, const std::string &block,
                                const std::string &field) {
  PARTHENON_REQUIRE(pin->DoesParameterExist(block, field),
                    "Initialization block <" + block + "> requires field '" + field +
                        "'.");
  const auto name = pin->GetString(block, field);
  PARTHENON_REQUIRE(!name.empty(), "Initialization field " + block + "/" + field +
                                       " must name a function.");
  return name;
}

Coordinates ReadCoordinateSystem(ParameterInput *pin, const std::string &block) {
  const auto default_system = pin->GetOrAddString("artemis", "coordinates", "cartesian");
  const auto system = pin->GetOrAddString(block, "system", default_system);
  const auto coordinates = geometry::CoordSelect(system, ProblemDimension(pin));
  PARTHENON_REQUIRE(coordinates != Coordinates::null,
                    "Invalid coordinate system '" + system + "' in <" + block + ">.");
  return coordinates;
}

class FunctionPacker {
 public:
  FunctionPacker(Rummy::FullDeck *deck, std::shared_ptr<StorageRegistry> storage,
                 std::vector<std::string> &names)
      : deck_(deck), storage_(std::move(storage)), names_(names) {}

  DeviceCallable Pack(const std::string &name, const int expected_arity) {
    const auto cached = callables_.find(name);
    if (cached != callables_.end()) {
      PARTHENON_REQUIRE(arities_.at(name) == expected_arity,
                        "Callable initializer function '" + name + "' has arity " +
                            std::to_string(arities_.at(name)) + ", expected " +
                            std::to_string(expected_arity) + ".");
      return cached->second;
    }

    std::string error;
    PARTHENON_REQUIRE(deck_->PackDeviceFunction(name, error),
                      "Unable to pack callable initializer function '" + name +
                          "' for the device: " + error);
    const auto host = deck_->GetDeviceFunction(name);
    PARTHENON_REQUIRE(host.entry_id < host.module.function_count,
                      "Packed callable initializer function '" + name +
                          "' has an invalid entry id.");
    const auto arity = host.module.functions[host.entry_id].arity;
    PARTHENON_REQUIRE(static_cast<int>(arity) == expected_arity,
                      "Callable initializer function '" + name + "' has arity " +
                          std::to_string(arity) + ", expected " +
                          std::to_string(expected_arity) + ".");

    const int id = static_cast<int>(names_.size());
    names_.push_back(name);
    DeviceModuleStorage owner;
    const auto prefix = "pips initializer " + std::to_string(id) + " ";
    owner.code = Upload(prefix + "code", host.module.code, host.module.code_size);
    owner.constants =
        Upload(prefix + "constants", host.module.constants, host.module.constant_count);
    owner.functions =
        Upload(prefix + "functions", host.module.functions, host.module.function_count);
    owner.globals =
        Upload(prefix + "globals", host.module.globals, host.module.global_count);

    DeviceCallable callable;
    callable.module = host.module;
    callable.module.code = owner.code.data();
    callable.module.constants = owner.constants.data();
    callable.module.functions = owner.functions.data();
    callable.module.globals = owner.globals.data();
    callable.entry_id = host.entry_id;
    callable.function_id = id;
    storage_->modules.push_back(std::move(owner));
    callables_.emplace(name, callable);
    arities_.emplace(name, static_cast<int>(arity));
    return callable;
  }

 private:
  Rummy::FullDeck *deck_;
  std::shared_ptr<StorageRegistry> storage_;
  std::vector<std::string> &names_;
  std::unordered_map<std::string, DeviceCallable> callables_;
  std::unordered_map<std::string, int> arities_;
};

void ConfigureGas(ParameterInput *pin, FunctionPacker &packer,
                  FunctionInitConfig &config) {
  constexpr const char *block = "gas/initialize";
  config.gas.enabled = true;
  config.gas.input_system = ReadCoordinateSystem(pin, block);
  config.gas.density = packer.Pack(RequireFunctionName(pin, block, "density"), 4);
  for (int d = 0; d < 3; ++d) {
    const auto field = "velocity_x" + std::to_string(d + 1);
    config.gas.velocity[d] = packer.Pack(RequireFunctionName(pin, block, field), 4);
  }

  const std::array<std::pair<const char *, ThermodynamicInput>, 3> options = {
      std::pair{"temperature", ThermodynamicInput::temperature},
      std::pair{"pressure", ThermodynamicInput::pressure},
      std::pair{"internal_energy", ThermodynamicInput::internal_energy}};
  int count = 0;
  for (const auto &[field, type] : options) {
    if (pin->DoesParameterExist(block, field)) {
      ++count;
      config.gas.thermodynamic_input = type;
      config.gas.thermodynamic = packer.Pack(RequireFunctionName(pin, block, field), 4);
    }
  }
  PARTHENON_REQUIRE(count == 1,
                    "<gas/initialize> requires exactly one of temperature, pressure, or "
                    "internal_energy.");
}

void ConfigureDust(ParameterInput *pin, FunctionPacker &packer,
                   FunctionInitConfig &config) {
  constexpr const char *block = "dust/initialize";
  config.dust.enabled = true;
  config.dust.input_system = ReadCoordinateSystem(pin, block);
  config.dust.density = packer.Pack(RequireFunctionName(pin, block, "density"), 5);
  for (int d = 0; d < 3; ++d) {
    const auto field = "velocity_x" + std::to_string(d + 1);
    config.dust.velocity[d] = packer.Pack(RequireFunctionName(pin, block, field), 5);
  }
}

void ConfigureMoment(ParameterInput *pin, FunctionPacker &packer,
                     FunctionInitConfig &config) {
  constexpr const char *block = "radiation/moment/initialize";
  config.moment.enabled = true;
  config.moment.input_system = ReadCoordinateSystem(pin, block);
  config.moment.energy = packer.Pack(RequireFunctionName(pin, block, "energy"), 4);
  for (int d = 0; d < 3; ++d) {
    const auto field = "reduced_flux_x" + std::to_string(d + 1);
    config.moment.reduced_flux[d] =
        packer.Pack(RequireFunctionName(pin, block, field), 4);
  }
}

} // namespace

void Configure(ParameterInput *pin, Rummy::FullDeck *deck, Packages_t &packages) {
  if (Globals::is_restart) return;

  const bool has_gas = pin->DoesBlockExist("gas/initialize");
  const bool has_dust = pin->DoesBlockExist("dust/initialize");
  const bool has_moment = pin->DoesBlockExist("radiation/moment/initialize");
  const auto problem = pin->GetOrAddString("artemis", "problem", "unset");

  const auto artemis_pkg = packages.Get("artemis");
  const bool do_gas = artemis_pkg->Param<bool>("do_gas");
  const bool do_dust = artemis_pkg->Param<bool>("do_dust");
  const bool do_moment = artemis_pkg->Param<bool>("do_moment");

  if (problem == "unset") {
    PARTHENON_REQUIRE(!do_gas || has_gas,
                      "artemis/problem=unset requires <gas/initialize> when gas is "
                      "enabled.");
    PARTHENON_REQUIRE(!do_dust || has_dust,
                      "artemis/problem=unset requires <dust/initialize> when dust is "
                      "enabled.");
    PARTHENON_REQUIRE(
        !do_moment || has_moment,
        "artemis/problem=unset requires <radiation/moment/initialize> when moment "
        "radiation is enabled.");
  }

  if (!(has_gas || has_dust || has_moment)) return;
  PARTHENON_REQUIRE(deck != nullptr,
                    "Callable initialization requires a Rummy FullDeck input parser.");

  auto storage = std::make_shared<StorageRegistry>();
  std::vector<std::string> names;
  FunctionPacker packer(deck, storage, names);
  FunctionInitConfig config;
  config.time = pin->GetOrAddReal("parthenon/time", "start_time", 0.0);
  if (has_gas) ConfigureGas(pin, packer, config);
  if (has_dust) ConfigureDust(pin, packer, config);
  if (has_moment) ConfigureMoment(pin, packer, config);

  artemis_pkg->AddParam("function_init_config", config);
  artemis_pkg->AddParam("function_init_names", names);
  artemis_pkg->AddParam("function_init_storage",
                        std::static_pointer_cast<void>(std::move(storage)));
}

template <Coordinates GEOM>
void Initialize(MeshBlock *pmb, ParameterInput *pin) {
  auto artemis_pkg = pmb->packages.Get("artemis");
  if (!artemis_pkg->AllParams().hasKey("function_init_config")) return;

  const auto config = artemis_pkg->Param<FunctionInitConfig>("function_init_config");
  if (!config.Enabled()) return;

  using parthenon::MakePackDescriptor;
  auto &md = pmb->meshblock_data.Get();
  for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
  }

  static auto desc =
      MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie,
                         dust::prim::density, dust::prim::velocity, rad::prim::energy,
                         rad::prim::flux>((pmb->resolved_packages).get());
  auto v = desc.GetPack(md.get());
  static auto desc_g =
      MakePackDescriptor<geom::x1v, geom::x2v, geom::x3v>((pmb->resolved_packages).get());
  auto vg = desc_g.GetPack(md.get());

  ArtemisUtils::EOS eos_d;
  if (config.gas.enabled) {
    eos_d = pmb->packages.Get("gas")->Param<ArtemisUtils::EOS>("eos_d");
  }
  ParArray1D<Real> dust_sizes;
  if (config.dust.enabled) {
    dust_sizes = pmb->packages.Get("dust")->Param<ParArray1D<Real>>("sizes");
  }

  ParArray1D<int> error("function initializer error", 6);
  auto h_error = error.GetHostMirror();
  for (int n = 0; n < 6; ++n)
    h_error(n) = (n == 0) ? 0 : -1;
  error.DeepCopy(h_error);

  const auto &cpars = artemis_pkg->template Param<geometry::CoordParams>("coord_params");
  auto &pco = pmb->coords;
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "callable primitive initialization", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto xi = coords.GetCellCenter(vg, 0, k, j, i);
        pips::device::DeviceVM vm;

        if (config.gas.enabled) {
          std::array<Real, 3> xo, ex1, ex2, ex3;
          impl::ConvertCoordinates(coords, xi, config.gas.input_system, xo, ex1, ex2,
                                   ex3);
          pips::device::DeviceValue args[4] = {
              pips::device::dv_number(xo[0]), pips::device::dv_number(xo[1]),
              pips::device::dv_number(xo[2]), pips::device::dv_number(config.time)};
          Real rho, thermo;
          std::array<Real, 3> velocity;
          if (!impl::Call(vm, config.gas.density, args, 4, rho, error, k, j, i) ||
              !impl::RequirePositive(rho, config.gas.density, error, k, j, i) ||
              !impl::Call(vm, config.gas.thermodynamic, args, 4, thermo, error, k, j,
                          i) ||
              !impl::RequirePositive(thermo, config.gas.thermodynamic, error, k, j, i)) {
            return;
          }
          for (int d = 0; d < 3; ++d) {
            if (!impl::Call(vm, config.gas.velocity[d], args, 4, velocity[d], error, k, j,
                            i)) {
              return;
            }
          }
          const auto projected = impl::ProjectVector(velocity, ex1, ex2, ex3);
          Real sie = thermo;
          if (config.gas.thermodynamic_input == ThermodynamicInput::temperature) {
            sie = eos_d.InternalEnergyFromDensityTemperature(rho, thermo);
          } else if (config.gas.thermodynamic_input == ThermodynamicInput::pressure) {
            eos_d.InternalEnergyFromDensityPressure(rho, thermo, sie);
          }
          if (!std::isfinite(sie) || sie <= 0.0) {
            impl::RecordError(error, config.gas.thermodynamic.function_id,
                              !std::isfinite(sie)
                                  ? static_cast<int>(impl::ErrorCode::non_finite)
                                  : static_cast<int>(impl::ErrorCode::non_positive),
                              k, j, i);
            return;
          }
          for (int n = 0; n < v.GetSize(0, gas::prim::density()); ++n) {
            v(0, gas::prim::density(n), k, j, i) = rho;
            v(0, gas::prim::sie(n), k, j, i) = sie;
            for (int d = 0; d < 3; ++d) {
              v(0, gas::prim::velocity(ArtemisUtils::VI(n, d)), k, j, i) = projected[d];
            }
          }
        }

        if (config.dust.enabled) {
          std::array<Real, 3> xo, ex1, ex2, ex3;
          impl::ConvertCoordinates(coords, xi, config.dust.input_system, xo, ex1, ex2,
                                   ex3);
          for (int n = 0; n < v.GetSize(0, dust::prim::density()); ++n) {
            pips::device::DeviceValue args[5] = {
                pips::device::dv_number(xo[0]), pips::device::dv_number(xo[1]),
                pips::device::dv_number(xo[2]), pips::device::dv_number(config.time),
                pips::device::dv_number(dust_sizes(n))};
            Real rho;
            std::array<Real, 3> velocity;
            if (!impl::Call(vm, config.dust.density, args, 5, rho, error, k, j, i) ||
                !impl::RequirePositive(rho, config.dust.density, error, k, j, i)) {
              return;
            }
            for (int d = 0; d < 3; ++d) {
              if (!impl::Call(vm, config.dust.velocity[d], args, 5, velocity[d], error, k,
                              j, i)) {
                return;
              }
            }
            const auto projected = impl::ProjectVector(velocity, ex1, ex2, ex3);
            v(0, dust::prim::density(n), k, j, i) = rho;
            for (int d = 0; d < 3; ++d) {
              v(0, dust::prim::velocity(ArtemisUtils::VI(n, d)), k, j, i) = projected[d];
            }
          }
        }

        if (config.moment.enabled) {
          std::array<Real, 3> xo, ex1, ex2, ex3;
          impl::ConvertCoordinates(coords, xi, config.moment.input_system, xo, ex1, ex2,
                                   ex3);
          pips::device::DeviceValue args[4] = {
              pips::device::dv_number(xo[0]), pips::device::dv_number(xo[1]),
              pips::device::dv_number(xo[2]), pips::device::dv_number(config.time)};
          Real energy;
          std::array<Real, 3> reduced_flux;
          if (!impl::Call(vm, config.moment.energy, args, 4, energy, error, k, j, i) ||
              !impl::RequirePositive(energy, config.moment.energy, error, k, j, i)) {
            return;
          }
          for (int d = 0; d < 3; ++d) {
            if (!impl::Call(vm, config.moment.reduced_flux[d], args, 4, reduced_flux[d],
                            error, k, j, i)) {
              return;
            }
          }
          const auto projected = impl::ProjectVector(reduced_flux, ex1, ex2, ex3);
          for (int n = 0; n < v.GetSize(0, rad::prim::energy()); ++n) {
            v(0, rad::prim::energy(n), k, j, i) = energy;
            for (int d = 0; d < 3; ++d) {
              v(0, rad::prim::flux(ArtemisUtils::VI(n, d)), k, j, i) = projected[d];
            }
          }
        }
      });

  h_error = error.GetHostMirrorAndCopy();
  if (h_error(0) != 0) {
    const auto &names =
        artemis_pkg->Param<std::vector<std::string>>("function_init_names");
    const int function_id = h_error(1);
    const std::string function_name =
        (function_id >= 0 && function_id < static_cast<int>(names.size()))
            ? names[function_id]
            : "<unknown>";
    std::ostringstream msg;
    msg << "Callable initializer function '" << function_name << "' "
        << impl::ErrorDescription(h_error(2)) << " at cell (k,j,i)=(" << h_error(3) << ","
        << h_error(4) << "," << h_error(5) << ").";
    PARTHENON_FAIL(msg.str());
  }
}

template void Initialize<Coordinates::cartesian>(MeshBlock *pmb, ParameterInput *pin);
template void Initialize<Coordinates::axisymmetric>(MeshBlock *pmb, ParameterInput *pin);
template void Initialize<Coordinates::cylindrical>(MeshBlock *pmb, ParameterInput *pin);
template void Initialize<Coordinates::spherical1D>(MeshBlock *pmb, ParameterInput *pin);
template void Initialize<Coordinates::spherical2D>(MeshBlock *pmb, ParameterInput *pin);
template void Initialize<Coordinates::spherical3D>(MeshBlock *pmb, ParameterInput *pin);

} // namespace function_init
} // namespace artemis
