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

# This file was created in part or in whole by generative AI

# Regression test based on the self-gravitating slab advection problem

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name

_mag_thr = 0.01
_conv_thr = 0.3
_nranks = 1
_file_id = "grav_slab"

# "Golden" reference L1 errors from grav_slab-errs.dat
# [RMS-L1, d_L1, E_L1] for Nx=16 and Nx=32
_ref_errors = [
    [1.850706e-01, 7.195540e-02, 1.516752e-01],  # Nx=16
    [4.414474e-02, 1.904644e-02, 3.310753e-02],  # Nx=32
]


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    for res in (16, 32):
        args = [
            "parthenon/job/problem_id=" + _file_id,
            "parthenon/mesh/nx1=" + repr(res),
            "parthenon/mesh/nx2=" + repr(res / 2),
            "parthenon/mesh/nx3=" + repr(res / 2),
            "parthenon/meshblock/nx1=" + repr(res / 4),
            "parthenon/meshblock/nx2=" + repr(res / 4),
            "parthenon/meshblock/nx3=" + repr(res / 4),
        ]
        artemis.run(_nranks, "self-gravity/grav_slab.in", args)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  We check the magnitude of the error and convergence rates.
    logger.debug("Analyzing test " + __name__)
    data = np.loadtxt(
        os.path.join(artemis.get_data_dir(), _file_id + "-errs.dat"),
        dtype=np.float64,
        ndmin=2,
    )

    analyze_status = True
    if np.isnan(data).any():
        logger.warning("NaN encountered")
        analyze_status = False
        raise FloatingPointError("NaN encountered")

    # Check error magnitude
    for i, row in enumerate(data[:2]):
        Nx = int(row[0])
        computed = row[[4, 5, 9]]  # RMS-L1, d_L1, E_L1
        ref = _ref_errors[i]
        rel_err = np.abs(computed - ref) / ref
        for j, val in enumerate(rel_err):
            if val > _mag_thr:
                logger.warning(
                    "Resolution Nx={0}: L1 error too high "
                    "(computed {1:.3e}, reference {2:.3e}, rel {3:.2f})".format(
                        Nx, computed[j], ref[j], val
                    )
                )
                analyze_status = False

    # Check error convergence
    l1_n16 = data[0, [4, 5, 9]]
    l1_n32 = data[1, [4, 5, 9]]
    rate = l1_n32 / l1_n16
    if not np.all(rate < _conv_thr):
        logger.warning(
            "L1 errors not converging, "
            "conv: {0} threshold: {1}".format(rate, _conv_thr)
        )
        analyze_status = False

    return analyze_status
