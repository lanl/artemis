# ========================================================================================
#  (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

# Regression test based on the Lowrie&Edwards, Mach=3 Steady Radiation Shock
# Exact solutions are generated via the open source, Quokka script here:
# https://github.com/quokka-astro/quokka/blob/development/extern/LowrieEdwards/radshock.py

# Modules
import logging
import numpy as np
import h5py
import os
from scipy.interpolate import interp1d
import scripts.utils.artemis as artemis
import sys

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to avoid issues with DISPLAY not being set
import matplotlib.pyplot as plt

# Plotting style
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Commands
_nranks = 1
_file_id = "shock_cgs"


# Thresholds
# NOTE(@pdmullen): The particle count in this test is too low to get good statistics of
# the radiation energy density via a tally.  Future extensions of this test may get at the
# radiation temperature via a different means so that we can lower the trad threshold...
_thr_gas = 0.004
_thr_rad = 0.002
# The shift in position to the exact solution
_dx = 0.000262


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + _file_id,
        "parthenon/mesh/nx1=256",
        "parthenon/meshblock/nx1=256",
    ]
    artemis.run(_nranks, "radiation/rad_shock_cgs.in", arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check that the correct equilibrium temperature
    # is obtained after reaching the final time.  Future extensions of this test could
    # test other jaybenne/dt to test how artemis fares for varying dt / (c rho kappa)^-1
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    with h5py.File(
        os.path.join(artemis.get_data_dir(), "{}.out1.final.phdf".format(_file_id)), "r"
    ) as f:
        cv = f["Params"].attrs["gas/cv"]
        ar = f["Params"].attrs["radiation/arad"]
        xm = f["Locations/x"][...].ravel()
        xc = 0.5 * (xm[:-1] + xm[1:])
        sie = f["gas.prim.sie_0"][...].ravel()
        er = f["rad.prim.energy_0"][...].ravel()
        tgas = sie / cv
        trad = (er / ar) ** (0.25)

    # Grab exact solution
    exact = np.loadtxt(
        os.path.join(artemis.get_artemis_dir(), "tst/scripts/radiation/rad_shock.dat")
    )
    int_tg = interp1d(exact[:, 0] - _dx, exact[:, 1], kind="linear")
    int_tr = interp1d(exact[:, 0] - _dx, exact[:, 2], kind="linear")
    tgas_exact = int_tg(xc)
    trad_exact = int_tr(xc)

    # Plot results
    os.makedirs(artemis.get_fig_dir(), exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(xc, tgas, label="$T_\\mathrm{gas}$", lw=4, alpha=0.25, color=colors[0])
    ax1.scatter(xc, trad, label="$T_\\mathrm{rad}$", lw=4, alpha=0.25, color=colors[1])
    ax1.plot(xc, tgas_exact, label="$T_\\mathrm{gas}$", lw=2, color=colors[0])
    ax1.plot(xc, trad_exact, label="$T_\\mathrm{rad}$", lw=2, color=colors[1])
    ax1.set_xlabel("$x \\; (\\mathrm{cm})$")
    ax1.set_ylabel("$T \\; (\\mathrm{K})$")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.5)
    ax1.xaxis.set_ticks_position("both")
    ax1.yaxis.set_ticks_position("both")
    ax1.tick_params(which="both", direction="in")
    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(os.path.join(artemis.get_fig_dir(), "rad_shock_cgs.png"))

    # Check if solution errors are above threshold.  See notes above regarding thresholds.
    l2_tgas = np.sqrt(
        np.trapz(((tgas - tgas_exact) / tgas_exact) ** 2, x=xc) / (xm[-1] - xm[0])
    )
    l2_trad = np.sqrt(
        np.trapz(((trad - trad_exact) / trad_exact) ** 2, x=xc) / (xm[-1] - xm[0])
    )
    print("l2_tgas: ", l2_tgas, "l2_trad: ", l2_trad)
    if l2_tgas > _thr_gas:
        logger.warning(
            "Error in gas temperature solution is greater than threshold: "
            "l2_tgas: {0} thr: {1}".format(l2_tgas, _thr_gas)
        )
        analyze_status = False
    if l2_trad > _thr_rad:
        logger.warning(
            "Error in radiation temperature solution is greater than threshold: "
            "l2_trad: {0} thr: {1} ".format(l2_trad, _thr_rad)
        )
        analyze_status = False

    return analyze_status
