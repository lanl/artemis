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

# Regression to test Brio-Wu shock tube

# Modules
import importlib
import logging
import scripts.mhd.brio_wu as brio_wu

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name

importlib.reload(brio_wu)
brio_wu._nranks = 4
brio_wu._file_id = "brio_wu_mpi"


# Run Artemis
def run(**kwargs):
    return brio_wu.run(**kwargs)


# Analyze outputs
def analyze():
    return brio_wu.analyze()
