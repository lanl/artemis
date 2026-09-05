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

"""Regression test for convergence of all seven ideal-MHD linear waves."""

import logging
import os

import numpy as np

import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])

_nranks = 1
_file_id = "mhd_linear_wave"
_amp = 1.0e-6
_waves = (
    "L-fast",
    "L-Alfven",
    "L-slow",
    "entropy",
    "R-slow",
    "R-Alfven",
    "R-fast",
)
_resolutions = (16, 32)
_max_normalized_error = 0.5
_max_convergence_ratio = 0.35
_max_divb = 1.0e-12


def run(**kwargs):
    logger.debug("Running test " + __name__)
    for res in _resolutions:
        common = [
            "parthenon/job/problem_id=" + _file_id,
            "parthenon/time/nlim=1000",
            "physics/mhd=true",
            "gas/cfl=0.4",
            "parthenon/mesh/nx1={:d}".format(res),
            "parthenon/mesh/nx2={:d}".format(res // 2),
            "parthenon/mesh/nx3={:d}".format(res // 2),
            "parthenon/meshblock/nx1={:d}".format(res // 4),
            "parthenon/meshblock/nx2={:d}".format(res // 4),
            "parthenon/meshblock/nx3={:d}".format(res // 4),
            "problem/amp={:.16e}".format(_amp),
            "gas/reconstruct=plm",
            "gas/riemann=hlld",
        ]
        for wave_flag in range(len(_waves)):
            vflow = 1.0 if wave_flag == 3 else 0.0
            artemis.run(
                _nranks,
                "linwave/linear_wave.in",
                common
                + [
                    "problem/wave_flag={:d}".format(wave_flag),
                    "problem/vflow={:.1f}".format(vflow),
                ],
            )


def analyze():
    logger.debug("Analyzing test " + __name__)
    path = os.path.join(artemis.get_data_dir(), _file_id + "-errs.dat")
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    expected_rows = len(_resolutions) * len(_waves)
    if data.shape != (expected_rows, 14):
        logger.warning("Unexpected MHD linear-wave error table shape: %s", data.shape)
        return False
    data = data.reshape((len(_resolutions), len(_waves), data.shape[-1]))

    status = True
    if not np.all(np.isfinite(data)):
        logger.warning("MHD linear-wave errors contain non-finite values")
        status = False

    for wave_index, wave_name in enumerate(_waves):
        coarse = data[0, wave_index, 4]
        fine = data[1, wave_index, 4]
        if fine / _amp > _max_normalized_error:
            logger.warning(
                "%s normalized error %.8e exceeds %.8e",
                wave_name,
                fine / _amp,
                _max_normalized_error,
            )
            status = False
        if fine / coarse > _max_convergence_ratio:
            logger.warning(
                "%s convergence ratio %.8e exceeds %.8e",
                wave_name,
                fine / coarse,
                _max_convergence_ratio,
            )
            status = False
        if np.max(data[:, wave_index, 13]) > _max_divb:
            logger.warning(
                "%s max|divB| %.8e exceeds %.8e",
                wave_name,
                np.max(data[:, wave_index, 13]),
                _max_divb,
            )
            status = False

    for left, right in ((0, 6), (1, 5), (2, 4)):
        if not np.isclose(
            data[1, left, 4],
            data[1, right, 4],
            rtol=1.0e-10,
            atol=1.0e-14 * _amp,
        ):
            logger.warning(
                "%s/%s errors are asymmetric: %.16e versus %.16e",
                _waves[left],
                _waves[right],
                data[1, left, 4],
                data[1, right, 4],
            )
            status = False

    return status
