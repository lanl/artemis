# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Regression test for the implicit radiation-matter coupling solver, exercised on the
# ray-traced disk problem (inputs/radiation/raytrace.in).

# Modules
import logging
import os

import h5py
import numpy as np

import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to avoid issues with DISPLAY not being set
import matplotlib.pyplot as plt

# Plotting style
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Commands
_nranks = 1
# problem_id for each run.  The first uses the raytrace.in default; the second overrides
# problem_id and forces the unsplit moment path (radiation/moment/split=false).
_file_ids = ["raytrace", "raytrace_unsplit"]

# Inputs
_gamma = 1.4

# Thresholds
# Global energy-budget closure.  The coupling law itself holds to the solver's
# outer-iteration tolerance (1e-12); this residual is set by transport flux through the
# domain boundary plus O(dt^2) gravity/coordinate-source work, and is a geometric ratio
# that does not shrink with dt.  A measured ~2.2e-3 on the reference build sits well below
# this bound, while a broken coupling operator would violate conservation by O(1).
_thr_budget = 5.0e-3
# Hard invariants the solver must satisfy exactly (up to a few ULP of roundoff).
_roundoff_tol = 512.0 * np.finfo(float).eps


def _scalar(dataset):
    """Return a scalar field as [block, k, j, i], squeezing a singleton component axis."""
    field = np.asarray(dataset)
    if field.ndim == 5 and field.shape[1] == 1:
        field = field[:, 0, ...]
    if field.ndim != 4:
        raise ValueError(
            "Expected a four-dimensional scalar field, got {}".format(field.shape)
        )
    return field


def _cell_volumes(xf, yf, zf):
    """Cell volumes for spherical-polar coordinates with logarithmic radius.

    Locations/{x,y,z} are the per-block face coordinates; the radial coordinate is stored
    as log(r), so the radial volume element is (exp(3 x)_hi - exp(3 x)_lo) / 3.
    """
    radial = (np.exp(3.0 * xf[:, 1:]) - np.exp(3.0 * xf[:, :-1])) / 3.0
    polar = np.cos(yf[:, :-1]) - np.cos(yf[:, 1:])
    azimuthal = zf[:, 1:] - zf[:, :-1]
    return (
        azimuthal[:, :, None, None] * polar[:, None, :, None] * radial[:, None, None, :]
    )


def _gas_total_energy(pressure, density, velocity):
    """Ideal-gas total energy density: internal (p / (gamma-1)) plus kinetic."""
    kinetic = 0.5 * density * np.sum(velocity * velocity, axis=1)
    return pressure / (_gamma - 1.0) + kinetic


# Run Artemis
def run(**kwargs):
    logger.debug("Running test " + __name__)
    # Primary configuration lives entirely in inputs/radiation/raytrace.in.
    artemis.run(_nranks, "radiation/raytrace.in", [])
    # Second configuration: identical problem forced onto the unsplit moment path.
    artemis.run(
        _nranks,
        "radiation/raytrace.in",
        [
            "parthenon/time/nlim=100",
            "parthenon/job/problem_id=" + _file_ids[1],
            "radiation/moment/split=false",
        ],
    )


def _analyze_one(file_id):
    """Check the invariants and the global energy budget for a single run."""
    data_dir = artemis.get_data_dir()
    status = True

    with h5py.File(
        os.path.join(data_dir, "{}.out1.00000.phdf".format(file_id)), "r"
    ) as f0:
        rho0 = _scalar(f0["gas.prim.density_0"][...])
        p0 = _scalar(f0["gas.prim.pressure_0"][...])
        vel0 = np.asarray(f0["gas.prim.velocity_0"][...])
        erad0 = _scalar(f0["rad.prim.energy_0"][...])
        has_temperature = "gas.prim.temperature_0" in f0
        t0 = _scalar(f0["gas.prim.temperature_0"][...]) if has_temperature else None
        time0 = float(f0["Info"].attrs["Time"])

    with h5py.File(
        os.path.join(data_dir, "{}.out1.final.phdf".format(file_id)), "r"
    ) as f1:
        rhof = _scalar(f1["gas.prim.density_0"][...])
        pf = _scalar(f1["gas.prim.pressure_0"][...])
        velf = np.asarray(f1["gas.prim.velocity_0"][...])
        eradf = _scalar(f1["rad.prim.energy_0"][...])
        reduced_flux = np.asarray(f1["rad.prim.flux_0"][...])
        src_energy = _scalar(f1["gas.src.energy"][...])
        time1 = float(f1["Info"].attrs["Time"])
        xf = np.asarray(f1["Locations/x"][...])
        yf = np.asarray(f1["Locations/y"][...])
        zf = np.asarray(f1["Locations/z"][...])
        efloor = float(f1["Params"].attrs["moments/efloor"])
        tfloor = float(f1["Params"].attrs["moments/tfloor"])
        chat = float(f1["Params"].attrs["moments/chat"])
        light_speed = float(f1["Params"].attrs["moments/c"])

    dt = time1 - time0
    volumes = _cell_volumes(xf, yf, zf)

    # --- Sanity: finiteness and positivity ------------------------------------------
    fields = {
        "final density": rhof,
        "final pressure": pf,
        "final radiation energy": eradf,
        "final velocity": velf,
        "final reduced flux": reduced_flux,
        "final gas source energy": src_energy,
    }
    for name, field in fields.items():
        if not np.isfinite(field).all():
            logger.warning("[{}] Non-finite values in {}.".format(file_id, name))
            status = False
    if not status:
        return False

    if np.any(rhof <= 0.0):
        logger.warning("[{}] Non-positive gas density found.".format(file_id))
        status = False
    if np.any(pf <= 0.0):
        logger.warning("[{}] Non-positive gas pressure found.".format(file_id))
        status = False

    # --- Hard invariants the solver guarantees exactly ------------------------------
    if np.any(eradf < efloor * (1.0 - _roundoff_tol)):
        logger.warning(
            "[{}] Radiation energy floor violated: min(E)={:.16e}, floor={:.16e}.".format(
                file_id, np.min(eradf), efloor
            )
        )
        status = False

    max_reduced_flux = np.max(np.sqrt(np.sum(reduced_flux * reduced_flux, axis=1)))
    if max_reduced_flux > 1.0 + _roundoff_tol:
        logger.warning(
            "[{}] Radiation realizability violated: max(|F|/E)={:.16e}.".format(
                file_id, max_reduced_flux
            )
        )
        status = False

    max_velocity_fraction = np.max(np.sqrt(np.sum(velf * velf, axis=1))) / light_speed
    if max_velocity_fraction >= 1.0:
        logger.warning(
            "[{}] Subluminal gas-velocity bound violated: max(|v|/c)={:.16e}.".format(
                file_id, max_velocity_fraction
            )
        )
        status = False

    # --- Precondition: the material temperature floor must be initially inactive ----
    # The coupling operator is conservative except for the one-time correction that
    # lifts cells starting below tfloor up to the floor (matter_coupling.hpp:1048).  If
    # any cell starts below tfloor that injection breaks the budget below, so guard the
    # assumption explicitly rather than mis-attributing the resulting residual.
    if has_temperature and np.any(t0 < tfloor * (1.0 - _roundoff_tol)):
        logger.warning(
            "[{}] Initial gas temperature below floor (min={:.16e}, tfloor={:.16e}); "
            "non-conservative floor injection would corrupt the energy budget.".format(
                file_id, np.min(t0), tfloor
            )
        )
        status = False

    # --- Global reduced-c total-energy budget ---------------------------------------
    # Per cell the coupling operator enforces
    #     dE_gas_total + (c / chat) dE_radiation = dt * src_energy.
    # Summed over the whole domain, conservative gas/radiation transport telescopes to a
    # small boundary flux and gravity/coordinate-source work is O(dt^2), so the equality
    # holds to the boundary-flux level over a single short step.
    delta_gas = _gas_total_energy(pf, rhof, velf) - _gas_total_energy(p0, rho0, vel0)
    delta_rad = (light_speed / chat) * (eradf - erad0)
    coupled = np.sum(volumes * (delta_gas + delta_rad))
    injected = np.sum(volumes * dt * src_energy)

    if injected <= 0.0:
        logger.warning(
            "[{}] No raytraced energy injected (injected={:.16e}).".format(
                file_id, injected
            )
        )
        return False

    budget_residual = abs(coupled - injected) / abs(injected)
    if budget_residual > _thr_budget:
        logger.warning(
            "[{}] Energy-budget closure exceeds threshold: coupled={:.16e}, "
            "injected={:.16e}, relative residual={:.16e} > {:.16e}.".format(
                file_id, coupled, injected, budget_residual, _thr_budget
            )
        )
        status = False

    # --- Diagnostic figure ----------------------------------------------------------
    # Per-cell coupled energy vs injected energy for illuminated cells, with the 1:1
    # line.  Deviations from the line localize where the coupling law is not satisfied.
    cell_injected = (volumes * dt * src_energy).ravel()
    cell_coupled = (volumes * (delta_gas + delta_rad)).ravel()
    illuminated = cell_injected > 1.0e-6 * np.max(cell_injected)

    os.makedirs(artemis.get_fig_dir(), exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(
        cell_injected[illuminated],
        cell_coupled[illuminated],
        s=8,
        alpha=0.4,
        color=colors[0],
        label="illuminated cells",
    )
    lo = np.min(cell_injected[illuminated])
    hi = np.max(cell_injected[illuminated])
    ax1.plot([lo, hi], [lo, hi], lw=2, color="black", ls="--", label="1:1")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"injected $\int Q \, dV$ (per cell)")
    ax1.set_ylabel(
        r"coupled $\int (\Delta E_g + (c/\hat{c})\Delta E_r)\, dV$ (per cell)"
    )
    ax1.set_title(
        "{}: global residual = {:.2e} (thr {:.0e})".format(
            file_id, budget_residual, _thr_budget
        )
    )
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.5)
    ax1.tick_params(which="both", direction="in")
    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(os.path.join(artemis.get_fig_dir(), "{}.png".format(file_id)))
    plt.close(fig)

    logger.info(
        "[{}] coupling budget: injected={:.16e}, coupled={:.16e}, relative "
        "residual={:.16e}, max|F|/E={:.16e}, max|v|/c={:.16e}.".format(
            file_id,
            injected,
            coupled,
            budget_residual,
            max_reduced_flux,
            max_velocity_fraction,
        )
    )
    return status


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    for file_id in _file_ids:
        analyze_status = _analyze_one(file_id) and analyze_status
    return analyze_status
