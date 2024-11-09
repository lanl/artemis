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

# Regression to test a viscous_diffusion of a gaussian bump

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
_file_id = "viscous_diffusion"

_nd = [1, 2]
_nu = 0.25
_t0 = 0.5
_eps = 1e-6
_tlim = 2.0
_nx = 64
_tol = 1e-8


# Run Artemis
def run(**kwargs):
    # v(t) = v0  (4 pi nu t0)^(-d/2)  exp(-x^2/(4 pi nu t0))

    sig2 = 2.0 * _nu * _t0
    for d in _nd:
        logger.debug("Runnning test " + __name__)
        arguments = [
            "parthenon/job/problem_id={}_{:d}d".format(_file_id, d),
            "physics/viscosity=true",
            "physics/conduction=false",
            "gas/viscosity/nu={:.8e}".format(_nu),
            "problem/vx3_bump={:.16e}".format(
                _eps * (2.0 * np.pi * sig2) ** (-0.5 * d)
            ),
            "problem/sigma={:.8e}".format(np.sqrt(sig2)),
            "problem/temperature_bump=0.0",
            "parthenon/time/tlim={:.8e}".format(_tlim),
        ]
        if d == 1:
            arguments += [
                "parthenon/mesh/nx1={:d}".format(_nx),
                "parthenon/meshblock/nx1={:d}".format(_nx // _nranks),
                "parthenon/mesh/nx2=1",
                "parthenon/meshblock/nx2=1",
                "parthenon/mesh/nx3=1",
                "parthenon/meshblock/nx3=1",
            ]
        elif d == 2:
            arguments += [
                "parthenon/mesh/nx1={:d}".format(_nx),
                "parthenon/meshblock/nx1={:d}".format(_nx // 2),
                "parthenon/mesh/nx2={:d}".format(_nx),
                "parthenon/meshblock/nx2={:d}".format(_nx // 2),
                "parthenon/mesh/nx3=1",
                "parthenon/meshblock/nx3=1",
            ]

        artemis.run(_nranks, "diffusion/gaussian_bump.in", arguments)


# Analyze outputs
def analyze():
    from scipy.interpolate import interp1d
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    errors = []
    for d in _nd:
        base = "{}_{:d}d".format(_file_id, d)
        logger.debug("Analyzing test " + __name__ + " {:d}D".format(d))
        os.makedirs(artemis.artemis_fig_dir, exist_ok=True)

        time, x, y, z, [dens, u, v, w, T] = binary.load_level(
            "final", dir=artemis.get_run_directory(), base=base + ".out1"
        )
        xc = 0.5 * (x[1:] + x[:-1])
        yc = 0.5 * (y[1:] + y[:-1])

        time0, x0, y0, z0, [dens0, u0, v0, w0, T0] = binary.load_level(
            0, dir="build/src", base=base + ".out1"
        )

        vx3 = w[0, :]
        vx30 = w0[0, :]
        time += _t0
        sig2 = 2.0 * _nu * time

        fig = None
        err = 1e99
        if d == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ans = (
                _eps
                * (2.0 * np.pi * sig2) ** (-0.5 * d)
                * np.exp(-(xc**2) / (2.0 * sig2))
            )
            ax.plot(xc, ans, "-", lw=4, alpha=0.2)
            ax.plot(xc, vx30[0, :], "--k")
            ax.plot(xc, vx3[0, :], "-k")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$v_3$")
            err = np.abs(ans - vx3[0, :]).mean()

        elif d == 2:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            norm = colors.Normalize(vmin=vx3.min(), vmax=vx3.max())
            axes[0].pcolormesh(xc, yc, vx3, norm=norm)
            yy, xx = np.meshgrid(yc, xc)
            dr2 = xx**2 + yy**2
            ans = (
                _eps * (2.0 * np.pi * sig2) ** (-0.5 * d) * np.exp(-dr2 / (2.0 * sig2))
            )
            err = (np.abs(ans.ravel() - vx3.ravel())).mean()
            axes[1].plot(np.sqrt(dr2).ravel(), ans.ravel(), ".", ms=1)
            axes[1].plot(np.sqrt(dr2).ravel(), vx30.ravel(), ".r", ms=1)
            axes[1].plot(np.sqrt(dr2).ravel(), vx3.ravel(), ".k", ms=1)
            axes[1].set_xlabel("Radius", fontsize=16)
            axes[1].set_ylabel("$V_3$", fontsize=18)
            axes[0].set_xlabel("x", fontsize=16)
            axes[0].set_ylabel("y", fontsize=16)
            create_colorbar(axes[0], norm=norm)

        fig.tight_layout()
        fig.savefig(artemis.artemis_fig_dir + base + "_vx3.png", bbox_inches="tight")
        errors.append((d, err))

    print(errors)
    analyze_status = all([err[1] <= _tol for err in errors])
    return analyze_status


def create_colorbar(ax, norm, where="top", cax=None, cmap="viridis", **kargs):
    import matplotlib
    import matplotlib.cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    labelsize = kargs.pop("labelsize", 14)
    labelsize = kargs.pop("fontsize", 14)

    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(where, size="3%", pad=0.05)

    cmap = matplotlib.cm.get_cmap(cmap)
    cb = matplotlib.colorbar.ColorbarBase(
        ax=cax, cmap=cmap, norm=norm, orientation="horizontal", **kargs
    )
    cb.ax.xaxis.set_ticks_position(where)
    cb.ax.xaxis.set_label_position(where)
    cb.ax.tick_params(labelsize=labelsize)

    return cb
