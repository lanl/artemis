# ========================================================================================
#  (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

# Regression to test dust coagulation for both surface-density and volume-density

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
from scipy.interpolate import interp1d


logger = logging.getLogger("artemis" + __name__[7:])  # set logger name

_nranks = 1
_file_id = "coag"
_massunit = ["3.976832e28", "3.17305460032e29"]
_surfden = ["true", "false"]
_tlim = 3768.0


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    for ii, im in enumerate(_massunit):
        arguments = [
            "artemis/mass=" + im,
            "parthenon/job/problem_id=" + _file_id,
            "parthenon/time/tlim={:.8f}".format(_tlim),
            "dust/surface_density_flag=" + _surfden[ii],
        ]
        artemis.run(_nranks, "dust/dust_coagulation.in", arguments)


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)

    # Grab referenece solution
    dat_sden = np.loadtxt(
        os.path.join(
            artemis.get_artemis_dir(), "tst/scripts/coagulation/coag_info_sden.dat"
        ),
        unpack=True,
    )
    dat_den = np.loadtxt(
        os.path.join(
            artemis.get_artemis_dir(), "tst/scripts/coagulation/coag_info_den.dat"
        ),
        unpack=True,
    )

    data_ref = np.hstack([dat_sden[2:5, :], dat_den[2:5, :]])

    fname = os.path.join(artemis.get_data_dir(), _file_id + "_info.dat")
    data_tst = np.loadtxt(
        fname,
        unpack=True,
    )

    errs = data_ref - data_tst[2:5, :]
    errors = np.array(errs).ravel()
    fail = np.any(errors > 0)
    return not fail
