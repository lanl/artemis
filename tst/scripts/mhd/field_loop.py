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

"""Regression test for planar and tilted constrained-transport field-loop advection."""

import glob
import logging
import os

import h5py
import numpy as np

import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])

_nranks = 1
_file_prefix = "mhd_field_loop"
_max_divb = 1.0e-10
_max_axial_field = 1.0e-10
_max_fine_shape_error = 0.6
_min_fine_energy_retention = 0.5

_cases = (
    {
        "name": "planar",
        "resolutions": (64, 128),
        "extra_args": ["problem/loop_type=planar", "problem/v3=1.0"],
    },
    {
        "name": "tilted",
        "resolutions": (16, 32),
        "extra_args": [
            "problem/loop_type=tilted",
            "problem/v3=0.0",
            "parthenon/mesh/x1min=-0.5",
            "parthenon/mesh/x1max=0.5",
            "parthenon/mesh/x2min=-0.5",
            "parthenon/mesh/x2max=0.5",
            "parthenon/mesh/x3min=-0.5",
            "parthenon/mesh/x3max=0.5",
        ],
    },
)


def _file_id(case, resolution):
    return "{}_{}_{}".format(_file_prefix, case["name"], resolution)


def _remove_outputs(file_id):
    for path in glob.glob(os.path.join(artemis.get_data_dir(), file_id + ".out1*.phdf")):
        os.remove(path)


def run(**kwargs):
    logger.debug("Running test " + __name__)
    for case in _cases:
        for resolution in case["resolutions"]:
            file_id = _file_id(case, resolution)
            _remove_outputs(file_id)
            if case["name"] == "planar":
                nx = (resolution, resolution // 2, 1)
                nmb = (resolution // 4, resolution // 4, 1)
            else:
                nx = (resolution, resolution, resolution)
                nmb = (resolution // 2, resolution // 2, resolution // 2)
            artemis.run(
                _nranks,
                "field_loop/field_loop.in",
                [
                    "parthenon/job/problem_id=" + file_id,
                    "parthenon/time/nlim=10000",
                    "gas/cfl=0.4",
                    "gas/reconstruct=plm",
                    "gas/riemann=hlld",
                    "parthenon/mesh/nx1={:d}".format(nx[0]),
                    "parthenon/mesh/nx2={:d}".format(nx[1]),
                    "parthenon/mesh/nx3={:d}".format(nx[2]),
                    "parthenon/meshblock/nx1={:d}".format(nmb[0]),
                    "parthenon/meshblock/nx2={:d}".format(nmb[1]),
                    "parthenon/meshblock/nx3={:d}".format(nmb[2]),
                ]
                + case["extra_args"],
            )


def _snapshot(file_id, output):
    path = os.path.join(artemis.get_data_dir(), "{}.out1.{}.phdf".format(file_id, output))
    with h5py.File(path, "r") as f:
        return {
            "density": f["gas.prim.density_0"][...],
            "pressure": f["gas.prim.pressure_0"][...],
            "momentum": f["gas.cons.momentum_0"][...],
            "total_energy": f["gas.cons.total_energy_0"][...],
            "B": f["field.cell.B"][...],
            "magnetic_energy": f["field.cell.energy"][...],
            "divB": f["field.cell.divB"][...],
            "x": f["Locations/x"][...],
            "y": f["Locations/y"][...],
            "z": f["Locations/z"][...],
        }


def _min_cell_width(snapshot):
    return min(
        np.min(np.diff(snapshot["x"], axis=-1)),
        np.min(np.diff(snapshot["y"], axis=-1)),
        np.min(np.diff(snapshot["z"], axis=-1)),
    )


def _metrics(initial, final, tilted):
    b_initial = initial["B"]
    b_final = final["B"]
    b_magnitude = np.linalg.norm(b_initial, axis=1)
    b_scale = np.max(b_magnitude)
    shape_error = np.mean(np.linalg.norm(b_final - b_initial, axis=1)) / np.mean(b_magnitude)
    retention = np.mean(final["magnetic_energy"]) / np.mean(initial["magnetic_energy"])
    relative_divb = np.max(np.abs(final["divB"])) * _min_cell_width(final) / b_scale
    if tilted:
        axis_component = (-b_final[:, 0] + b_final[:, 2]) / np.sqrt(2.0)
    else:
        axis_component = b_final[:, 2]
    axial_field = np.max(np.abs(axis_component)) / b_scale
    return shape_error, retention, relative_divb, axial_field


def _conserved(initial, final, name):
    old = np.mean(initial[name], axis=tuple(range(0, initial[name].ndim)))
    new = np.mean(final[name], axis=tuple(range(0, final[name].ndim)))
    return np.max(np.abs(new - old)) / max(np.max(np.abs(old)), 1.0)


def analyze():
    logger.debug("Analyzing test " + __name__)
    status = True
    for case in _cases:
        metrics = []
        for resolution in case["resolutions"]:
            file_id = _file_id(case, resolution)
            initial = _snapshot(file_id, "00000")
            final = _snapshot(file_id, "final")
            for name, values in final.items():
                if name not in ("x", "y", "z") and not np.all(np.isfinite(values)):
                    logger.warning("%s contains non-finite %s", file_id, name)
                    status = False
            if np.min(final["density"]) <= 0.0 or np.min(final["pressure"]) <= 0.0:
                logger.warning("%s has non-positive density or pressure", file_id)
                status = False
            for name in ("density", "momentum", "total_energy"):
                error = _conserved(initial, final, name)
                if error > 1.0e-10:
                    logger.warning("%s does not conserve %s: %.8e", file_id, name, error)
                    status = False
            metrics.append(_metrics(initial, final, case["name"] == "tilted"))

        coarse, fine = metrics
        if fine[0] > _max_fine_shape_error:
            logger.warning("%s fine loop-shape error %.8e exceeds %.8e", case["name"], fine[0], _max_fine_shape_error)
            status = False
        if fine[0] > 0.9 * coarse[0]:
            logger.warning("%s loop-shape error does not improve enough: %.8e -> %.8e", case["name"], coarse[0], fine[0])
            status = False
        if fine[1] < _min_fine_energy_retention or fine[1] < coarse[1]:
            logger.warning("%s magnetic-energy retention is inadequate: %.8e -> %.8e", case["name"], coarse[1], fine[1])
            status = False
        for resolution, metric in zip(case["resolutions"], metrics):
            if metric[2] > _max_divb:
                logger.warning("%s_%d relative divB %.8e exceeds %.8e", case["name"], resolution, metric[2], _max_divb)
                status = False
            if metric[3] > _max_axial_field:
                logger.warning("%s_%d axial field %.8e exceeds %.8e", case["name"], resolution, metric[3], _max_axial_field)
                status = False
    return status
