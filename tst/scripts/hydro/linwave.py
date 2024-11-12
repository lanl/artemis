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

# Regression test based on the Newtonian hydro linear wave convergence problem.
# NOTE(@pdmullen): The following is largely borrowed from the open-source Athena++/AthenaK
# softwares.

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name

_int = ["rk2"]
_recon = ["plm", "ppm"]
_flux = ["hllc", "hlle", "llf"]
_wave = ["L-sound", "R-sound", "entropy"]
_nranks = 1
_file_id = "linear_wave"


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
                        "gas/riemann=" + fv,
                    ]
                    # L-going sound wave
                    args_l = arguments + ["problem/wave_flag=0", "problem/vflow=0.0"]
                    artemis.run(_nranks, "linwave/linear_wave.in", args_l)
                    # R-going sound wave
                    args_r = arguments + ["problem/wave_flag=4", "problem/vflow=0.0"]
                    artemis.run(_nranks, "linwave/linear_wave.in", args_r)
                    # entropy wave
                    args_entr = arguments + ["problem/wave_flag=3", "problem/vflow=1.0"]
                    artemis.run(_nranks, "linwave/linear_wave.in", args_entr)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the magnitude of the error,
    # error convergence rates, and error identicality between L- and R-going
    # sound waves.
    logger.debug("Analyzing test " + __name__)
    data = np.loadtxt(
        os.path.join(artemis.get_run_directory(), _file_id + "-errs.dat"),
        dtype=np.float64,
        ndmin=2,
    )
    analyze_status = True
    if np.isnan(data).any():
        logger.warning("NaN encountered")
        analyze_status = False
        raise FloatingPointError("NaN encountered")
    data = data.reshape(
        [len(_int), len(_recon), len(_flux), 2, len(_wave), data.shape[-1]]
    )
    for ii, iv in enumerate(_int):
        for ri, rv in enumerate(_recon):
            error_threshold = [0.0] * len(_wave)
            conv_threshold = [0.0] * len(_wave)
            if rv == "plm":
                error_threshold[0] = error_threshold[1] = 2.23e-7  # sound
                error_threshold[2] = 2.21e-7  # entropy
                conv_threshold[0] = conv_threshold[1] = 0.29  # sound
                conv_threshold[2] = 0.30  # entropy
            else:  # if rv == "ppm"
                error_threshold[0] = error_threshold[1] = 1.75e-7  # sound
                error_threshold[2] = 1.11e-7  # entropy
                conv_threshold[0] = conv_threshold[1] = 0.44  # sound
                conv_threshold[2] = 0.42  # entropy
            for fi, fv in enumerate(_flux):
                for wi, wv in enumerate(_wave):
                    l1_rms_n16 = data[ii][ri][fi][0][wi][4]
                    l1_rms_n32 = data[ii][ri][fi][1][wi][4]
                    if l1_rms_n32 > error_threshold[wi]:
                        logger.warning(
                            "{0} wave error too large for {1}+"
                            "{2}+{3} configuration, "
                            "error: {4:g} threshold: {5:g}".format(
                                wv, iv, rv, fv, l1_rms_n32, error_threshold[wi]
                            )
                        )
                        analyze_status = False
                    if l1_rms_n32 / l1_rms_n16 > conv_threshold[wi]:
                        logger.warning(
                            "{0} wave not converging for {1}+"
                            "{2}+{3} configuration, "
                            "conv: {4:g} threshold: {5:g}".format(
                                wv,
                                iv,
                                rv,
                                fv,
                                l1_rms_n32 / l1_rms_n16,
                                conv_threshold[wi],
                            )
                        )
                        analyze_status = False
                l1_rms_l = data[ii][ri][fi][1][_wave.index("L-sound")][4]
                l1_rms_r = data[ii][ri][fi][1][_wave.index("R-sound")][4]
                if l1_rms_l != l1_rms_r:
                    logger.warning(
                        "Errors in L/R-going sound waves not "
                        "equal for {0}+{1}+{2} configuration, "
                        "{3:g} {4:g}".format(iv, rv, fv, l1_rms_l, l1_rms_r)
                    )
                    analyze_status = False

    return analyze_status
