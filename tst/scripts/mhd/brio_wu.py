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
import scripts.utils.artemis as artemis
from scipy.interpolate import interp1d

logger = logging.getLogger("artemis" + __name__[7:])
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to avoid issues with DISPLAY not being set
import matplotlib.pyplot as plt

_nranks = 1
_file_id = "mhd_brio_wu"
_reference_file = os.path.join(os.path.dirname(__file__), "brio_wu.std")
_fields = ("density", "pressure", "velocity_x", "velocity_y", "magnetic_y")
_profile_tolerance = 3.8e-3
_solvers = ["llf", "hlle", "hlld"]
_nx = 512


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
                "parthenon/mesh/nx1={:d}".format(_nx),
                "parthenon/meshblock/nx1={:d}".format(int(_nx / _nranks)),
                "parthenon/time/ncycle_out=100",
            ],
        ]
        print("artemis -i " + args[1] + " " + " ".join(args[2]))
        artemis.run(*args)


def analyze():
    logger.debug("Analyzing test " + __name__)
    os.makedirs(artemis.get_fig_dir(), exist_ok=True)
    status = True
    reference = np.loadtxt(_reference_file, comments="#", ndmin=2)
    ath = np.loadtxt(os.path.join(os.path.dirname(__file__), "athena_bw.std"))

    def test_one(x, v, column, solver):
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
                "{} normalized L1 error {:.8e} exceeds {:.8e} in {}".format(
                    names[column - 1], l1, _profile_tolerance, solver
                )
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
            logger.warning("Brio-Wu output contains non-finite values in " + r)
            status = False

        if np.min(d) <= 0.0:
            logger.warning("Brio-Wu density is not positive in " + r)
            status = False
        if np.min(p) <= 0.0:
            logger.warning("Brio-Wu pressure is not positive in " + r)
            status = False
        if np.max(np.abs(B[0, :] - 0.75)) > 1.0e-12:
            logger.warning("Brio-Wu Bx is not constant at 0.75 in " + r)
            status = False
        if np.max(np.abs(divB)) > 1.0e-10:
            logger.warning(
                "Brio-Wu max|divB| is too large: {:.8e} in {}".format(
                    np.max(np.abs(divB)), r
                )
            )
            status = False
        if max(np.max(np.abs(u[2, :])), np.max(np.abs(B[2, :]))) > 1.0e-12:
            logger.warning("Brio-Wu developed an out-of-plane component in " + r)
            status = False

        status = status and test_one(x, d, 1, r)
        status = status and test_one(x, p, 2, r)
        status = status and test_one(x, u[0, :], 3, r)
        status = status and test_one(x, u[1, :], 4, r)
        status = status and test_one(x, B[1, :], 7, r)

        fig, axes = plt.subplots(2, 3, figsize=(4 * 3, 4 * 2))
        axes[0, 0].plot(ath[:, 1], ath[:, 2], "-k", alpha=0.3, lw=2)
        axes[0, 0].plot(reference[:, 0], reference[:, 1], "-k")
        axes[0, 0].plot(x, d)

        axes[0, 1].plot(ath[:, 1], ath[:, 3], "-k", alpha=0.3, lw=2)
        axes[0, 1].plot(reference[:, 0], reference[:, 2], "-k")
        axes[0, 1].plot(x, p)

        axes[0, 2].plot(ath[:, 1], ath[:, 4], "-k", alpha=0.3, lw=2)
        axes[0, 2].plot(reference[:, 0], reference[:, 3], "-k")
        axes[0, 2].plot(x, u[0, :])

        axes[1, 0].plot(ath[:, 1], ath[:, 5], "-k", alpha=0.3, lw=2)
        axes[1, 0].plot(reference[:, 0], reference[:, 4], "-k")
        axes[1, 0].plot(x, u[1, :])

        axes[1, 1].plot(ath[:, 1], ath[:, 7], "-k", alpha=0.3, lw=2)
        axes[1, 1].plot(reference[:, 0], reference[:, 6], "-k")
        axes[1, 1].plot(x, B[0, :])

        axes[1, 2].plot(ath[:, 1], ath[:, 8], "-k", alpha=0.3, lw=2)
        axes[1, 2].plot(reference[:, 0], reference[:, 7], "-k")
        axes[1, 2].plot(x, B[1, :])

        axes[0, 0].set_ylabel("$\\rho$", fontsize=14)
        axes[0, 1].set_ylabel("$P$", fontsize=14)
        axes[0, 2].set_ylabel("$v_x$", fontsize=14)
        axes[1, 0].set_ylabel("$v_y$", fontsize=14)
        axes[1, 1].set_ylabel("$B_x$", fontsize=14)
        axes[1, 2].set_ylabel("$B_y$", fontsize=14)
        for ax in axes.flatten():
            ax.tick_params(labelsize=12)
            ax.set_xlabel("$x$", fontsize=14)
            ax.minorticks_on()
        fig.tight_layout()
        fig.savefig(
            os.path.join(artemis.get_fig_dir(), _file_id + "_{}.png".format(r)),
            bbox_inches="tight",
        )

    return status
