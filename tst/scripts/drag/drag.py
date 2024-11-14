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

# Regression to test drag coupling between gas and dust

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

_nranks = 1
_file_id = "drag"
_tol = 3e-3
_tlim = 10.0


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + _file_id,
        "parthenon/time/ncycle_out=10000",
        "parthenon/time/tlim={:.8f}".format(_tlim),
    ]
    artemis.run(_nranks, "drag/simple_drag.in", arguments)


# Analyze outputs
def analyze():
    import h5py
    from scipy.interpolate import interp1d
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    logger.debug("Analyzing test " + __name__)
    os.makedirs(artemis.artemis_fig_dir, exist_ok=True)
    analyze_status = True

    dv0 = -1.0
    c = 0.01 / 10.0
    ans = lambda tc, t: np.exp(-(1.0 + c) * t / tc) * dv0
    tau = [1e-2, 0.1, 1.0, 10.0]
    times = []
    vsep = []
    mom_tot = []
    errors = []
    for n in range(1, int(_tlim / 0.05)):
        fname = os.path.join(
            artemis.get_run_directory(), "{}.out1.{:05d}.phdf".format(_file_id, n)
        )
        with h5py.File(fname, "r") as f:
            t = f["Info"].attrs["Time"]
            vg = f["gas.prim.velocity_0"][...][:, 0, :].ravel()
            dg = f["gas.prim.density_0"][...].ravel()
            mom = (dg * vg).sum()
            vdiff = []
            err = []
            dust_mom = []
            for d in range(4):
                vd_ = f["dust.prim.velocity_{:d}".format(d)][...][:, 0, :].ravel()
                dd_ = f["dust.prim.density_{:d}".format(d)][...].ravel()
                val = vd_ - vg

                err_ = abs(val.mean() - ans(tau[d], t))
                err.append(err_)
                vdiff.append(val.mean())
                mom += (vd_ * dd_).sum()
            errors.append(err)
            vsep.append(vdiff)
            times.append(t)
            mom_tot.append(mom)
    mom_tot = np.array(mom_tot)
    fig, axes = plt.subplots(1, 3, figsize=(3 * 8, 6))
    for d in range(4):
        (l,) = axes[0].plot(times, [ans(tau[d], t) for t in times])
        axes[0].plot(
            times,
            [vd[d] for vd in vsep],
            ".",
            c=l._color,
            label="$\\tau={:.2f}$".format(tau[d]),
        )
        axes[1].plot(
            times,
            [err[d] for err in errors],
            ".",
            c=l._color,
            label="$\\tau={:.2f}$".format(tau[d]),
        )

    mom_err = abs(mom_tot / mom_tot[0] - 1)
    axes[2].plot(times, mom_err, "-k")
    for ax in axes:
        ax.set_xlabel("Time", fontsize=18)
        ax.tick_params(labelsize=14)
        ax.set_xscale("log")
        ax.minorticks_on()
    axes[1].set_yscale("log")
    axes[1].legend(loc="best", fontsize=16)
    axes[0].set_ylabel("$v_d - v_g$", fontsize=20)
    axes[1].set_ylabel("$v_d - v_g$ Error", fontsize=18)
    axes[2].set_ylabel("Momentum Error", fontsize=18)
    fig.tight_layout()
    fig.savefig(artemis.artemis_fig_dir + _file_id + "_drag.png", bbox_inches="tight")

    errors = np.array(errors).ravel()
    fail = np.any(errors > _tol)
    fail |= np.max(mom_err) > 1e-13
    return not fail
