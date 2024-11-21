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

# Regression to test a planet in shearing sheet
# This tests that the planet spiral is at the correct azimuthal location using the
# Ogilvie & Lubow (2002) results.

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
import scripts.binary.binary as binary  # loads the

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

_nranks = 1
_file_id = "ssheet"


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + _file_id,
        "parthenon/time/tlim={:.16f}".format(2.0 * np.pi),
    ]
    artemis.run(_nranks, "ssheet/ssheet.in", arguments)


# Analyze outputs
def analyze():
    from scipy.interpolate import interp1d
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    logger.debug("Analyzing test " + __name__)
    os.makedirs(artemis.artemis_fig_dir, exist_ok=True)
    analyze_status = True

    time, x, y, z, [d, u, v, w, T] = binary.load_level(
        "final", dir="build/src", base="{}.out1".format(_file_id)
    )
    xc = 0.5 * (x[1:] + x[:-1])
    yc = 0.5 * (y[1:] + y[:-1])

    h = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sig = d[0, :] - d.mean(axis=1)[0, :]

    # Plot the 2D density and overplot the analytic spiral
    norm = colors.Normalize()
    axes[0].pcolormesh(xc, yc, sig, norm=norm)
    ri = np.linspace(-0.4, -2.0 / 3 * h, 20)
    ro = np.linspace(2.0 / 3 * h, 0.4, 20)
    axes[0].plot(ri, [spiral_pos(dx) for dx in ri], "--w")
    axes[0].plot(ro, [spiral_pos(dx) for dx in ro], "--w")

    axes[0].set_xlim(-0.4, 0.4)
    axes[0].set_ylim(-0.4, 0.4)

    # Indices for the inner and outer evalulation rings
    ii = np.argwhere(x <= -0.1)[-1][0]
    io = np.argwhere(xc >= 0.1)[0][0]

    # the azimuthal locations of the spirals approximated as
    # where the max occurs
    pi = yc[np.argwhere(sig[:, ii] == sig[:, ii].max())[0][0]]
    po = yc[np.argwhere(sig[:, io] == sig[:, io].max())[0][0]]

    # the analytic answers
    p0i = spiral_pos(-0.1)
    p0o = spiral_pos(0.1)

    # the errors
    names = ["Inner location", "Outer Location"]
    tols = [0.03, 0.03]
    errs = [abs(pi - p0i), abs(po - p0o)]

    # Plot the result
    (li,) = axes[1].plot(yc, sig[:, ii], label="Artemis: x=-0.1")
    (lo,) = axes[1].plot(yc, sig[:, io], label="Artemis: x=+0.1")
    axes[1].axvline(p0i, c=li._color, label="Analytic: x=-0.1")
    axes[1].axvline(p0o, c=lo._color, label="Analytic: x=+0.1")
    for ax in axes.flatten():
        ax.set_xlabel("$x$", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.minorticks_on()
    axes[1].legend(loc="best", fontsize=12)
    axes[1].set_ylabel("$\\Sigma - \\langle \\Sigma \\rangle$", fontsize=20)
    axes[0].set_ylabel("$y$", fontsize=20)
    binary.create_colorbar(axes[0], norm=norm)
    fig.tight_layout()
    fig.savefig(artemis.artemis_fig_dir + _file_id + "_spiral.png", bbox_inches="tight")

    # Check failure criterion
    for name, res, tol in zip(names, errs, tols):
        if res >= tol:
            logger.warning(
                "Error too large for {} configuration, "
                "{} err = {:.8e} threshold: {:.8e} ".format(_file_id, name, res, tol)
            )
            analyze_status = False

    return analyze_status


def spiral_pos(x, h=0.05):
    # Analytic spiral position from Ogilvie & Lubow (2002)
    # 3/4 x^2 / h

    if x > 0:
        return -0.75 * x**2 / h
    if x < 0:
        return 0.75 * x**2 / h
    return p0
