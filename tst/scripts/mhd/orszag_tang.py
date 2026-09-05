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

# This file was created in part or in whole by one of OpenAI's generative AI models

# Coarse, multi-MeshBlock HLLD Orszag-Tang regression. This checks the conserved global
# quantities and the local total-energy split without depending on a fragile pointwise
# reference through the shock network.

import glob
import logging
import os

import h5py
import numpy as np

import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])

_nranks = 1
_file_id = "mhd_orszag_tang"
_gamma = 1.66666666667
_rho0 = 0.22104853207207686
_p0 = 0.13262911924324611
_v0 = 1.0
_b0 = 0.28209479177387814
_initial_total_energy = _p0 / (_gamma - 1.0) + 0.5 * _rho0 * _v0**2 + 0.5 * _b0**2


def run(**kwargs):
    logger.debug("Running test " + __name__)
    npx = int(np.sqrt(_nranks))
    npy = _nranks / npx
    if npx * npy != _nranks:
        raise RuntimeError("Number of processsors is wrong")

    artemis.run(
        _nranks,
        "orszag_tang/orszag_tang.in",
        [
            "parthenon/job/problem_id=" + _file_id,
            "gas/riemann=hlld",
            "parthenon/mesh/nx1=64",
            "parthenon/mesh/nx2=64",
            "parthenon/meshblock/nx1={:d}".format(int(64 / npx)),
            "parthenon/meshblock/nx2={:d}".format(int(64 / npy)),
            "parthenon/output1/dt=0.1",
            "parthenon/time/ncycle_out=100",
        ],
    )


def analyze():
    logger.debug("Analyzing test " + __name__)
    snapshot = _load_snapshot(_latest_file(_file_id))
    status = True

    for name, values in snapshot.items():
        if not np.all(np.isfinite(values)):
            logger.warning("%s contains non-finite values", name)
            status = False

    density = snapshot["density"]
    pressure = snapshot["pressure"]
    velocity = snapshot["velocity"]
    magnetic = snapshot["magnetic"]
    total_energy = snapshot["total_energy"]
    internal_energy = snapshot["internal_energy"]
    magnetic_energy = snapshot["magnetic_energy"]

    if np.min(density) <= 0.0 or np.min(pressure) <= 0.0:
        logger.warning("Orszag-Tang has non-positive density or pressure")
        status = False
    if np.max(np.abs(snapshot["divB"])) > 1.0e-10:
        logger.warning(
            "Orszag-Tang max|divB| is too large: %.8e", np.max(np.abs(snapshot["divB"]))
        )
        status = False
    if not np.isclose(np.mean(density), _rho0, rtol=1.0e-12, atol=0.0):
        logger.warning("Orszag-Tang mean density changed: %.16e", np.mean(density))
        status = False

    momentum = np.mean(density[:, None, ...] * velocity, axis=(0, 2, 3, 4))
    if np.max(np.abs(momentum)) > 1.0e-10:
        logger.warning("Orszag-Tang mean momentum is too large: %s", momentum)
        status = False
    if not np.isclose(
        np.mean(total_energy), _initial_total_energy, rtol=1.0e-8, atol=0.0
    ):
        logger.warning(
            "Orszag-Tang mean total energy changed: %.16e", np.mean(total_energy)
        )
        status = False

    kinetic_energy = 0.5 * density * np.sum(velocity**2, axis=1)
    residual = total_energy - internal_energy - kinetic_energy - magnetic_energy
    if np.max(np.abs(residual)) > 1.0e-10:
        logger.warning(
            "Orszag-Tang energy decomposition residual is %.8e",
            np.max(np.abs(residual)),
        )
        status = False
    if np.mean(internal_energy) < _p0 / (_gamma - 1.0) + 1.0e-3 * _initial_total_energy:
        logger.warning("Orszag-Tang did not produce the expected shock heating")
        status = False

    return status


def _latest_file(base):
    files = sorted(
        glob.glob(os.path.join(artemis.get_data_dir(), base + ".out1.final.phdf"))
    )
    if not files:
        raise RuntimeError("No output files found for " + base)
    return files[-1]


def _load_snapshot(filename):
    with h5py.File(filename, "r") as f:
        return {
            "density": f["gas.prim.density_0"][...],
            "pressure": f["gas.prim.pressure_0"][...],
            "velocity": f["gas.prim.velocity_0"][...],
            "magnetic": f["field.cell.B"][...],
            "total_energy": f["gas.cons.total_energy_0"][...],
            "internal_energy": f["gas.cons.internal_energy_0"][...],
            "magnetic_energy": f["field.cell.energy"][...],
            "divB": f["field.cell.divB"][...],
        }
