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

# Regression to test collisions with REBOUND particles
# We test that the correct the numer of particles were removed
# and that total mass was conserved

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
from scipy.interpolate import interp1d


logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to avoid issues with DISPLAY not being set
import matplotlib.colors as colors
import matplotlib.pyplot as plt

_nranks = 1
_file_id = "collisions"


# Run Artemis
def run(**kwargs):
    input_path = "../../" + artemis.artemis_rel_path + "inputs/"
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id={}_{:d}".format(_file_id, _nranks),
        "nbody/planets/input_file=" + input_path + "planet_inputs/n20_sys.txt",
    ]
    artemis.run(
        _nranks,
        "disk/disk_collision.in",
        arguments + ["parthenon/time/tlim={:.16f}".format(1.0)],
    )

    artemis.run(
        _nranks,
        "disk/disk_collision.in",
        arguments + ["parthenon/time/tlim={:.16f}".format(2.5)],
        restart="{}_{:d}.out2.final.rhdf".format(_file_id, _nranks),
    )


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)

    fname = os.path.join(
        artemis.get_data_dir(), "{}_{:d}.reb".format(_file_id, _nranks)
    )
    logger.debug("Reading" + fname)
    d = np.loadtxt(fname)
    with open(fname, "r") as f:
        N = int(f.readline().split("=")[1].strip())
        header = f.readline()
        cols = np.array([_.split("=")[1] for _ in header[1:].strip().split()])

    rdata = d.reshape(d.shape[0] // N, N, d.shape[1])
    ind = np.argwhere(cols == "active")[0][0]
    mind = np.argwhere(cols == "mass")[0][0]
    Nstart = int(rdata[0, :, ind].sum())
    Nend = int(rdata[-1, :, ind].sum())
    logger.debug(
        "For {:d} objects, started with {:d}, ended with {:d}".format(N, Nstart, Nend)
    )
    analyze_status = (Nstart == N) & (Nend == N - 2)
    # mass conservation
    mass_err = (rdata[-1, :, mind] * rdata[-1, :, ind]).sum() - (
        rdata[0, :, mind]
    ).sum()
    logger.debug("Mass error {:.8e}".format(mass_err))
    analyze_status &= abs(mass_err) <= 1e-10
    return analyze_status
