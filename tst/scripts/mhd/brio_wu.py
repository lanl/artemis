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

# MHD shock-tube regression against a conservative restriction of an Artemis
# 4096-zone reference solution. The 512-zone test is split across four MeshBlocks to
# exercise block-boundary reconstruction and constrained transport.

import glob
import logging
import os

import h5py
import numpy as np
from scipy.interpolate import interp1d

import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])

_nranks = 1
_file_id = "mhd_brio_wu"
_reference_file = os.path.join(os.path.dirname(__file__), "brio_wu.std")
_fields = ("density", "pressure", "velocity_x", "velocity_y", "magnetic_y")
_profile_tolerance = 3.0e-2
_solvers = ["llf", "hlle", "hlld"]


def run(**kwargs):
    logger.debug("Running test " + __name__)
    for r in _solvers:
        args = [
            _nranks,
            "shock/brio_wu.in",
            [
                "parthenon/job/problem_id={}_{}".format(_file_id, r),
                "gas/riemann={}".format(r),
                "gas/scr_level=1",
                "parthenon/mesh/nx1=512",
                "parthenon/meshblock/nx1=512",
                "parthenon/time/ncycle_out=1",
            ],
        ]
        print("artemis -i " + args[1] + " " + " ".join(args[2]))
        artemis.run(*args)


def analyze():
    logger.debug("Analyzing test " + __name__)
    status = True
    reference = np.loadtxt(_reference_file, comments="#", ndmin=2)

    def test_one(x, v, column):
        names = ["density", "pressure", "vx", "vy", "vz", "Bx", "By", "Bz"]
        dynamic_range = np.ptp(reference[:, column])
        f = interp1d(
            reference[:, 0],
            reference[:, column],
            bounds_error=False,
            fill_value="extrapolate",
        )

        l1 = np.mean(np.abs(v - f(x))) / dynamic_range
        if l1 > _profile_tolerance:
            logger.warning(
                "%s normalized L1 error %.8e exceeds %.8e",
                names[column - 1],
                l1,
                _profile_tolerance,
            )
            return False
        return True

    for r in _solvers:
        filename = os.path.join(
            artemis.get_data_dir(), "{}_{}.out1.final.phdf".format(_file_id, r)
        )
        with h5py.File(filename, "r") as f:
            B = f["field.cell.B"][...][0, :, 0, 0, :]
            d = f["gas.prim.density_0"][...][0, 0, 0, :]
            xm = f["Locations/x"][...][0, :]
            x = 0.5 * (xm[1:] + xm[:-1])
            u = f["gas.prim.velocity_0"][0, :, 0, 0, :]
            p = f["gas.prim.pressure_0"][0, 0, 0, :]
            divB = f["field.cell.divB"][0, 0, 0, :]

        if not np.all(np.isfinite(np.vstack((d, p, u, B, divB)))):
            logger.warning("Brio-Wu output contains non-finite values")
            status = False

        if np.min(d) <= 0.0:
            logger.warning("Brio-Wu density is not positive")
            status = False
        if np.min(p) <= 0.0:
            logger.warning("Brio-Wu pressure is not positive")
            status = False
        if np.max(np.abs(B[0, :] - 0.75)) > 1.0e-12:
            logger.warning("Brio-Wu Bx is not constant at 0.75")
            status = False
        if np.max(np.abs(divB)) > 1.0e-10:
            logger.warning("Brio-Wu max|divB| is too large: %.8e", np.max(np.abs(divB)))
            status = False
        if max(np.max(np.abs(u[2, :])), np.max(np.abs(B[2, :]))) > 1.0e-12:
            logger.warning("Brio-Wu developed an out-of-plane component")
            status = False

        status = status and test_one(x, d, 1)
        status = status and test_one(x, p, 2)
        status = status and test_one(x, u[0, :], 3)
        status = status and test_one(x, u[1, :], 4)
        status = status and test_one(x, u[2, :], 5)
        status = status and test_one(x, B[1, :], 7)

    return status
