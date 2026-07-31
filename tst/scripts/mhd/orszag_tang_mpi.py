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

# Regression to test Orszag-Tang vortex

# Modules
import importlib
import logging
import scripts.mhd.orszag_tang as orszag_tang

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name

importlib.reload(orszag_tang)
orszag_tang._nranks = 4
orszag_tang._file_id = "orszag_tang_mpi"


# Run Artemis
def run(**kwargs):
    return orszag_tang.run(**kwargs)


# Analyze outputs
def analyze():
    return orszag_tang.analyze()
