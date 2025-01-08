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

# Regression to test a binary in a 2D disk with rebound
# This tests three things:
#  1. Planet spiral is at the correct azimuthal location using the
#     Ogilvie & Lubow (2002) results.
#  2. Cooling fixes the temperature profile correctly
#  3. Damping removes the radial velocity correctly

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
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to avoid issues with DISPLAY not being set

_nranks = 1
_file_id = "nbody"


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = ["parthenon/job/problem_id=" + _file_id]
    artemis.run(
        _nranks,
        "disk/binary_nbody_cyl.in",
        arguments + ["parthenon/time/tlim={:.16f}".format(1.0 * np.pi)],
    )

    artemis.run(
        _nranks,
        "disk/binary_nbody_cyl.in",
        arguments + ["parthenon/time/tlim={:.16f}".format(2.0 * np.pi)],
        restart="{}.out2.final.rhdf".format(_file_id),
    )


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    os.makedirs(artemis.get_fig_dir(), exist_ok=True)
    analyze_status = True

    time, r, phi, z, [d, u, v, w, T] = analysis.load_level(
        "final", base="{}.out1".format(_file_id), dir=artemis.get_data_dir()
    )
    rc = 0.5 * (r[1:] + r[:-1])
    pc = 0.5 * (phi[1:] + phi[:-1])

    h = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sig = d[0, :] - d.mean(axis=1)[0, :]

    # Plot the 2D density and overplot the analytic spiral
    norm = colors.Normalize()
    axes[0].pcolormesh(rc, pc, sig, norm=norm)
    ri = np.linspace(r.min(), 1 - 2.0 / 3 * h, 50)
    ro = np.linspace(1 + 2.0 / 3 * h, r.max(), 50)
    axes[0].plot(ri, [analysis.spiral_pos(x) for x in ri], "--w")
    axes[0].plot(ro, [analysis.spiral_pos(x) for x in ro], "--w")

    axes[0].set_xlim(0.6, 1.4)
    axes[0].set_ylim(np.pi - 0.8, np.pi + 0.8)

    # Indices for the inner and outer evalulation rings
    ii = np.argwhere(rc >= 1 - 0.1)[0][0]
    io = np.argwhere(rc >= 1 + 0.1)[0][0]

    # the azimuthal locations of the spirals approximated as
    # where the max occurs
    pi = pc[np.argwhere(sig[:, ii] == sig[:, ii].max())[0][0]]
    po = pc[np.argwhere(sig[:, io] == sig[:, io].max())[0][0]]

    # the analytic answers
    p0i = analysis.spiral_pos(1 - 0.1)
    p0o = analysis.spiral_pos(1 + 0.1)

    # the errors
    names = ["Inner location", "Outer Location"]
    tols = [0.04, 0.04]
    errs = [abs(pi - p0i) / p0i, abs(po - p0o) / p0o]

    # Plot the result
    (li,) = axes[1].plot(pc, sig[:, ii], label="Artemis: r=0.9")
    (lo,) = axes[1].plot(pc, sig[:, io], label="Artemis: r=1.1")
    axes[1].axvline(p0i, c=li._color, label="Analytic: r=0.9")
    axes[1].axvline(p0o, c=lo._color, label="Analytic: r=1.1")
    for ax in axes.flatten():
        ax.set_xlabel("$R$", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.minorticks_on()
    axes[1].legend(loc="best", fontsize=12)
    axes[1].set_ylabel("$\\Sigma - \\langle \\Sigma \\rangle$", fontsize=20)
    axes[0].set_ylabel("$\\phi$", fontsize=20)
    analysis.create_colorbar(axes[0], norm=norm)
    fig.tight_layout()
    fig.savefig(
        os.path.join(artemis.get_fig_dir(), _file_id + "_spiral.png"),
        bbox_inches="tight",
    )

    # Cooling check
    Tavg = T.mean(axis=1)[0, :]
    fit = np.polyfit(np.log(rc), np.log(Tavg), 1)
    plaw_ans = -1.0
    tnorm_ans = 0.0025
    plaw = fit[0]
    tnorm = np.exp(fit[1])

    names.append("Temp plaw")
    tols.append(2e-4)
    errs.append(abs(plaw - plaw_ans))
    names.append("Temp norm")
    tols.append(5e-3)
    errs.append(abs(tnorm - tnorm_ans) / tnorm_ans)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rc, Tavg, label="Artemis")
    ax.plot(rc, tnorm_ans * rc**plaw_ans, "--k", label="Correct")
    ax.set_xlabel("$R$", fontsize=20)
    ax.set_ylabel("$\\langle T \\rangle$", fontsize=20)
    ax.minorticks_on()
    ax.tick_params(labelsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        os.path.join(artemis.get_fig_dir(), _file_id + "_temp.png"),
        bbox_inches="tight",
    )

    # Check failure criterion
    for name, res, tol in zip(names, errs, tols):
        if res >= tol:
            logger.warning(
                "Error too large for {} configuration, "
                "{} err = {:.8e} threshold: {:.8e} ".format(_file_id, name, res, tol)
            )
            analyze_status = False

    return analyze_status
