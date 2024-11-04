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

# Regression to test a 3D disk
# This simply tests if the ICs are stable

# Modules
import logging
import numpy as np
import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

_nranks = 1
_file_id = "disk"

_geom = ["axi", "cart", "cyl", "sph"]
_gamma = [1.0, 1.4]
_bc = ["ic", "extrap"]

_tol = 6e-3

_dt_low = 1e-4
_dt_high = 3e-2


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
                    "disk/disk_{}.in".format(g),
                    arguments
                    + [
                        "parthenon/job/problem_id=disk_{}_{:d}_{}".format(
                            g, int(10 * gam), b
                        ),
                        "problem/polytropic_index={:.2f}".format(gam),
                    ],
                )
                artemis.run(
                    _nranks,
                    "disk/disk_{}.in".format(g),
                    arguments_rst
                    + [
                        "parthenon/job/problem_id=disk_{}_{:d}_{}".format(
                            g, int(10 * gam), b
                        ),
                        "problem/polytropic_index={:.2f}".format(gam),
                    ],
                    restart="disk_{}_{:d}_{}.out2.final.rhdf".format(
                        g, int(10 * gam), b
                    ),
                )


# Analyze outputs
def analyze():
    from scipy.interpolate import interp1d
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    bad = False
    for b in _bc:
        for g in _geom:
            for gam in _gamma:
                logger.debug("Analyzing test {}_{}".format(__name__, g))
                logger.debug(
                    "build/src/disk_{}_{:d}_{}.out1".format(g, int(10 * gam), b)
                )
                _, (x, y, z), (d0, _, _, _, _), sys, _ = loadf(
                    0,
                    base="build/src/disk_{}_{:d}_{}.out1".format(g, int(10 * gam), b),
                )
                time, (x, y, z), (d, T, u, v, w), sys, dt = loadf(
                    "final",
                    base="build/src/disk_{}_{:d}_{}.out1".format(g, int(10 * gam), b),
                )
                mybad = False
                mybad |= np.any(np.isnan(d))
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: Density is NaN".format(
                            g, int(10 * gam), b
                        )
                    )
                mybad |= np.any(np.isnan(T))
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: Temperature is NaN".format(
                            g, int(10 * gam), b
                        )
                    )
                mybad |= np.any(np.isnan(u))
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: vx1 is NaN".format(g, int(10 * gam), b)
                    )
                mybad |= np.any(np.isnan(v))
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: vx2 is NaN".format(g, int(10 * gam), b)
                    )
                mybad |= np.any(np.isnan(w))
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: vx3 is NaN".format(g, int(10 * gam), b)
                    )
                mybad |= np.any(d <= 0.0)
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: Density is negative".format(
                            g, int(10 * gam), b
                        )
                    )
                mybad |= np.any(T <= 0.0)
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: Temperature is negative".format(
                            g, int(10 * gam), b
                        )
                    )
                mybad |= (dt <= _dt_low) | (dt >= _dt_high)
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: Timestep is out of range [{:.1e}, {:.1e}]".format(
                            g, int(10 * gam), b, _dt_low, _dt_high
                        )
                    )
                err = d - d0
                err = np.sqrt((d0 * err**2).sum()) / (d0.sum())
                mybad |= err > _tol
                if mybad:
                    logger.debug(
                        "disk_{}_{:d}_{} FAILED: Density relative error of {:.2e} is too large".format(
                            g, int(10 * gam), b, err
                        )
                    )
                bad |= mybad

    return not bad


def loadf(n, base="disk.out1"):
    import h5py

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
