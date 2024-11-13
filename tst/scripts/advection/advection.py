# ========================================================================================
#  (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

# Regression test based on a gas + dust advection convergence problem.
# NOTE(@pdmullen): The following is largely borrowed from the open-source Athena++/AthenaK
# softwares.

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
import sys

sys.path.append(os.path.join(artemis.artemis_dir, "analysis"))
from ahistory import ahistory

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name

_int = ["rk2"]
_recon = ["plm"]
_flux = ["hlle", "llf"]
_species = ["gas", "dust1", "dust2"]
_nranks = 1
_file_id = "advection"


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    for iv in _int:
        for rv in _recon:
            for fv in _flux:
                for res in (16, 32):
                    arguments = [
                        "parthenon/job/problem_id=" + _file_id,
                        "problem/nperiod=1",
                        "parthenon/time/nlim=1000",
                        "parthenon/time/integrator=" + iv,
                        "parthenon/mesh/nghost=4",
                        "parthenon/mesh/nx1=" + repr(res),
                        "parthenon/mesh/nx2=" + repr(res / 2),
                        "parthenon/mesh/nx3=" + repr(res / 2),
                        "parthenon/meshblock/nx1=" + repr(res / 4),
                        "parthenon/meshblock/nx2=" + repr(res / 4),
                        "parthenon/meshblock/nx3=" + repr(res / 4),
                        "parthenon/mesh/x1min=0.0",
                        "parthenon/mesh/x1max=3.0",
                        "parthenon/mesh/x2min=0.0",
                        "parthenon/mesh/x2max=1.5",
                        "parthenon/mesh/x3min=0.0",
                        "parthenon/mesh/x3max=1.5",
                        "problem/amp=1.0e-6",
                        "parthenon/output1/dt=-1.0",
                        "gas/reconstruct=" + rv,
                        "dust/reconstruct=" + rv,
                        "gas/riemann=" + fv,
                        "dust/riemann=" + fv,
                    ]
                    artemis.run(_nranks, "advection/advection.in", arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the magnitude of the error,
    # error convergence rates, and error identicality between L- and R-going
    # advection.
    logger.debug("Analyzing test " + __name__)
    data = np.loadtxt(
        os.path.join(artemis.get_run_directory(), _file_id + "-errs.dat"),
        dtype=np.float64,
        ndmin=2,
    )
    history = ahistory(
        os.path.join(artemis.get_run_directory(), _file_id + ".out0.hst")
    )
    analyze_status = True
    if np.isnan(data).any():
        logger.warning("NaN encountered")
        analyze_status = False
        raise FloatingPointError("NaN encountered")

    def history_equiv(a, b, tol=1.0e-4):
        if 2.0 * (np.fabs(a - b)) / (np.fabs(a) + np.fabs(b)) > tol:
            return False
        else:
            return True

    history_expected = {
        "time": 1.0,
        "dt": 1.11612e-02,
        "cycle": 56,
        "nbtotal": 16,
        "gas_mass_0": 6.75,
        "gas_momentum_x1_0": 2.25,
        "gas_momentum_x2_0": 4.5,
        "gas_momentum_x3_0": 4.5,
        "gas_energy_0": 9.45,
        "gas_internal_energy_0": 6.075,
        "dust_mass_0": 6.75,
        "dust_mass_1": 6.75,
        "dust_momentum_x1_0": 2.25,
        "dust_momentum_x1_1": -2.25,
        "dust_momentum_x2_0": 4.5,
        "dust_momentum_x2_1": -4.5,
        "dust_momentum_x3_0": 4.5,
        "dust_momentum_x3_1": -4.5,
    }

    for key in history_expected.keys():
        values = history.Get(key)
        if len(values) != 11:
            analyze_status = False
        for value in values:
            if np.isnan(value):
                logger.warning("NaN encountered")
                analyze_status = False
                raise FloatingPointError("NaN encountered")
        if not history_equiv(values[-1], history_expected[key]):
            print(
                f"History entry {key} = {values[-1]} does not match expectation = {history_expected[key]}!"
            )
            analyze_status = False

    data = data.reshape([len(_int), len(_recon), len(_flux), 2, data.shape[-1]])
    for ii, iv in enumerate(_int):
        for ri, rv in enumerate(_recon):
            error_threshold = [0.0] * len(_species)
            conv_threshold = [0.0] * len(_species)
            if rv == "plm":
                error_threshold[0] = error_threshold[1] = error_threshold[2] = 2.21e-7
                conv_threshold[0] = conv_threshold[1] = conv_threshold[2] = 0.30
            else:  # if rv == "ppm"
                error_threshold[0] = error_threshold[1] = error_threshold[2] = 9.0e-8
                conv_threshold[0] = conv_threshold[1] = conv_threshold[2] = 0.42
            for fi, fv in enumerate(_flux):
                for si, sv in enumerate(_species):
                    l1_rms_n16 = data[ii][ri][fi][0][si + 4]
                    l1_rms_n32 = data[ii][ri][fi][1][si + 4]
                    if l1_rms_n32 > error_threshold[si]:
                        logger.warning(
                            "{0} wave error too large for {1}+"
                            "{2}+{3} configuration, "
                            "error: {4:g} threshold: {5:g}".format(
                                sv, iv, rv, fv, l1_rms_n32, error_threshold[si]
                            )
                        )
                        analyze_status = False
                    if l1_rms_n32 / l1_rms_n16 > conv_threshold[si]:
                        logger.warning(
                            "{0} wave not converging for {1}+"
                            "{2}+{3} configuration, "
                            "conv: {4:g} threshold: {5:g}".format(
                                sv,
                                iv,
                                rv,
                                fv,
                                l1_rms_n32 / l1_rms_n16,
                                conv_threshold[si],
                            )
                        )
                        analyze_status = False
                l1_rms_l = data[ii][ri][fi][1][5]
                l1_rms_r = data[ii][ri][fi][1][6]
                if l1_rms_l != l1_rms_r:
                    logger.warning(
                        "Errors in L/R-going dust advection not "
                        "equal for {0}+{1}+{2} configuration, "
                        "{3:g} {4:g}".format(iv, rv, fv, l1_rms_l, l1_rms_r)
                    )
                    analyze_status = False

    return analyze_status
