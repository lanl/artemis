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

# Regression to test a 3D disk with REBOUND gravity
# Tests for stability across restarts

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
from scipy.interpolate import interp1d


logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid issues with DISPLAY not being set

_nranks = 1
_file_id = "disk_nbody"

_geom = ["cyl"]
_gamma = [1.0, 1.4]
_bc = ["ic", "extrap"]

_tol = 5e-3


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/time/nlim=5",
    ]
    arguments_rst = ["parthenon/time/nlim=10"]
    directions = dict(cart=[], axi=["x1", "x2"], sph=["x1", "x2"], cyl=["x1", "x3"])
    for b in _bc:
        for g in _geom:
            bc_args = []
            for d in directions[g]:
                bc_args.append("parthenon/mesh/i{}_bc={}".format(d, b))
                bc_args.append("parthenon/mesh/o{}_bc={}".format(d, b))
            for gam in _gamma:
                artemis.run(
                    _nranks,
                    "disk/disk_nbody_{}.in".format(g),
                    arguments
                    + [
                        "parthenon/job/problem_id=disk_nbody_{}_{:d}_{}".format(
                            g, int(10 * gam), b
                        ),
                        "problem/polytropic_index={:.2f}".format(gam),
                    ],
                )
                artemis.run(
                    _nranks,
                    "disk/disk_nbody_{}.in".format(g),
                    arguments_rst
                    + [
                        "parthenon/job/problem_id=disk_nbody_{}_{:d}_{}".format(
                            g, int(10 * gam), b
                        ),
                        "problem/polytropic_index={:.2f}".format(gam),
                    ],
                    restart="disk_nbody_{}_{:d}_{}.out2.final.rhdf".format(
                        g, int(10 * gam), b
                    ),
                )


# Analyze outputs
def analyze():
    bad = False
    for b in _bc:
        for g in _geom:
            for gam in _gamma:
                logger.debug("Analyzing test {}_{}".format(__name__, g))
                logger.debug(
                    os.path.join(
                        artemis.get_data_dir(),
                        "disk_nbody_{}_{:d}_{}.out1".format(g, int(10 * gam), b),
                    )
                )
                _, (x, y, z), (d0, _, _, _, _), sys, _ = loadf(
                    0,
                    base=os.path.join(
                        artemis.get_data_dir(),
                        "/disk_nbody_{}_{:d}_{}.out1".format(g, int(10 * gam), b),
                    ),
                )
                time, (x, y, z), (d, T, u, v, w), sys, dt = loadf(
                    "final",
                    base=os.path.join(
                        artemis.get_data_dir(),
                        "disk_nbody_{}_{:d}_{}.out1".format(g, int(10 * gam), b),
                    ),
                )
                mybad = False
                mybad |= np.any(np.isnan(d))
                mybad |= np.any(np.isnan(T))
                mybad |= np.any(np.isnan(u))
                mybad |= np.any(np.isnan(v))
                mybad |= np.any(np.isnan(w))
                mybad |= np.any(d <= 0.0)
                mybad |= np.any(T <= 0.0)
                mybad |= (dt <= 1e-4) | (dt >= 3e-2)
                err = d - d0
                err = np.sqrt((d0 * err**2).sum()) / (d0.sum())
                logger.debug("Error: {:.2e}".format(err))
                mybad |= err > _tol
                if mybad:
                    logger.debug(
                        "disk_nbody_{}_{:d}_{} FAILED".format(g, int(10 * gam), b)
                    )
                bad |= mybad

    return not bad


def loadf(n, base="disk_nbody.out1"):
    try:
        fname = "{}.{:05d}.phdf".format(base, n)
    except:
        fname = "{}.{}.phdf".format(base, n)

    with h5py.File(fname, "r") as f:
        d = f["gas.prim.density_0"][...]
        T = f["gas.prim.pressure_0"][...] / f["gas.prim.density_0"][...]
        u = f["gas.prim.velocity_0"][...][:, 0, :, :, :]
        v = f["gas.prim.velocity_0"][...][:, 1, :, :, :]
        w = f["gas.prim.velocity_0"][...][:, 2, :, :, :]
        x = f["Locations/x"][...]
        y = f["Locations/y"][...]
        z = f["Locations/z"][...]
        sys = f["Params"].attrs["artemis/coord_sys"]
        time = f["Info"].attrs["Time"]
        dt = f["Info"].attrs["dt"]
    return time, (x, y, z), (d, T, u, v, w), sys, dt
