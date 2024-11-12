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
    history = np.loadtxt(
        os.path.join(artemis.get_run_directory(), _file_id + ".out0.hst")
    )
    analyze_status = True
    if np.isnan(data).any() or np.isnan(history).any():
        logger.warning("NaN encountered")
        analyze_status = False
        raise FloatingPointError("NaN encountered")
    if history.shape != (44, 18):
        analyze_status = False
    history_line = history[-1]

    def history_equiv(a, b, tol=1.0e-4):
        if 2.0 * (np.fabs(a - b)) / (np.fabs(a) + np.fabs(b)) > tol:
            return False
        else:
            return True

    history_expected = [
        1.00000e00,
        1.11612e-02,
        5.60000e01,
        1.60000e01,
        6.75000e00,
        2.25000e00,
        4.50000e00,
        4.50000e00,
        9.45000e00,
        6.07500e00,
        6.75000e00,
        6.75000e00,
        2.25000e00,
        -2.25000e00,
        4.50000e00,
        -4.50000e00,
        4.50000e00,
        -4.50000e00,
    ]
    if len(history_line) != len(history_expected):
        print(
            f"Number of history rows ({len(history_line)}) do not equal expectation ({len(history_expected)})!"
        )
        analyze_status = False
    for n, val in enumerate(history_expected):
        if not history_equiv(history_line[n], val):
            print(
                f"History entry {n} = {history_line[n]} does not match expectation = {val}!"
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
