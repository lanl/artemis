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

# Regression to test steady-state conduction.

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
import scripts.diffusion.thermal_diffusion as thermal_diffusion

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

thermal_diffusion._nranks = 2
thermal_diffusion._file_id = "thermal_diffusion_mpi"


# Run Artemis
def run(**kwargs):
    return thermal_diffusion.run(**kwargs)


# Analyze outputs
def analyze():
    return thermal_diffusion.analyze()
