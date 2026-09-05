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

"""Four-rank wrapper for the MHD linear-wave regression."""

import importlib
import logging

import scripts.mhd.linwave as linwave

logger = logging.getLogger("artemis" + __name__[7:])

importlib.reload(linwave)
linwave._nranks = 4
linwave._file_id = "mhd_linear_wave_mpi"


def run(**kwargs):
    return linwave.run(**kwargs)


def analyze():
    return linwave.analyze()
