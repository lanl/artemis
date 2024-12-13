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

# Regression of alpha disk

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
from scipy.interpolate import interp1d


logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import scripts.utils.analysis as analysis
import matplotlib.colors as colors
import matplotlib.pyplot as plt

_nranks = 1
_file_id = "alpha_disk"

_alpha = 0.1
_h = 0.1
_tlim = 8e3
_nx = 64
_tol = 2e-3


# Run Artemis
def run(**kwargs):
    d = 1
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id={}_{:d}d".format(_file_id, d),
        "parthenon/output1/dt=1000.",
        "parthenon/time/ncycle_out=10000",
        "parthenon/time/tlim={:.8e}".format(_tlim),
        "parthenon/mesh/x1max=2.0",
        "physics/viscosity=true",
        "gas/viscosity/alpha={:.8e}".format(_alpha),
        "cooling/tcyl={:.8e}".format(_h**2),
        "cooling/cyl_plaw=-1.0",
        "problem/mdot={:.8e}".format(_alpha * _h**2 * 3 * np.pi),
        "problem/quiet_start=true",
        "problem/h0={:.8e}".format(_h),
        "problem/dslope=0.0",
        "problem/flare=0.0",
    ]
    if d == 1:
        arguments += [
            "artemis/coordinates=axisymmetric",
            "parthenon/mesh/nx1={:d}".format(_nx),
            "parthenon/meshblock/nx1={:d}".format(_nx // _nranks),
            "parthenon/mesh/nx2=1",
            "parthenon/meshblock/nx2=1",
            "parthenon/mesh/nx3=1",
            "parthenon/meshblock/nx3=1",
            "parthenon/mesh/x2min=-0.5",
            "parthenon/mesh/x2max=0.5",
        ]

    artemis.run(_nranks, "diffusion/alpha_disk.in", arguments)


# Analyze outputs
def analyze():
    errors = []
    d = 1
    base = "{}_{:d}d".format(_file_id, d)
    logger.debug("Analyzing test " + __name__ + " {:d}D".format(d))
    os.makedirs(artemis.get_fig_dir(), exist_ok=True)

    time, x, y, z, [dens, u, v, w, T] = analysis.load_level(
        "final", dir=artemis.get_data_dir(), base=base + ".out1"
    )
    r = 0.5 * (x[1:] + x[:-1])

    dens = dens[0, :].mean(axis=0)
    u = u[0, :].mean(axis=0)
    mdot = -2 * np.pi * r * dens * u

    nu = _alpha * _h**2 * np.sqrt(r)
    dens_ans = 1.0 / np.sqrt(r)
    u_ans = -1.5 * nu / r
    mdot_ans = 3 * np.pi * _alpha * _h**2

    fig, axes = plt.subplots(1, 3, figsize=(8 * 3, 6))
    axes[0].plot(r, dens_ans, "--k", lw=3)
    axes[0].plot(r, dens)
    axes[0].set_ylabel("$\\rho$", fontsize=20)

    axes[1].plot(r, u_ans, "--k", lw=3)
    axes[1].plot(r, u)
    axes[1].set_ylabel("$v_R$", fontsize=20)

    axes[2].axhline(mdot_ans, c="k", ls="--", zorder=0, lw=3)
    axes[2].plot(r, mdot)
    axes[2].set_ylabel("$\\dot{M}$", fontsize=20)

    for ax in axes:
        ax.set_xlabel("$R/R_0$", fontsize=20)
        ax.tick_params(labelsize=14)
        ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(
        os.path.join(artemis.get_fig_dir(), base + "_res.png"), bbox_inches="tight"
    )

    errors = [
        abs((dens_ans - dens) / dens_ans).mean(),
        abs((mdot_ans - mdot) / mdot_ans).mean(),
    ]

    print(errors)
    analyze_status = all([err <= _tol for err in errors])
    return analyze_status
