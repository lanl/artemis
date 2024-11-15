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

# Regression to test a binary in a 2D disk
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

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

_nranks = 1
_file_id = "binary"


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + _file_id,
        "parthenon/time/tlim={:.16f}".format(2.0 * np.pi),
    ]
    artemis.run(_nranks, "disk/binary_cyl.in", arguments)


# Analyze outputs
def analyze():
    from scipy.interpolate import interp1d
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    logger.debug("Analyzing test " + __name__)
    os.makedirs(artemis.artemis_fig_dir, exist_ok=True)
    analyze_status = True

    time, r, phi, z, [d, u, v, w, T] = load_level(
        "final", base="{}.out1".format(_file_id), dir=artemis.get_run_directory()
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
    axes[0].plot(ri, [spiral_pos(x) for x in ri], "--w")
    axes[0].plot(ro, [spiral_pos(x) for x in ro], "--w")

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
    p0i = spiral_pos(1 - 0.1)
    p0o = spiral_pos(1 + 0.1)

    # the errors
    names = ["Inner location", "Outer Location"]
    tols = [0.03, 0.03]
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
    create_colorbar(axes[0], norm=norm)
    fig.tight_layout()
    fig.savefig(
        os.path.join(artemis.artemis_fig_dir, _file_id + "_spiral.png"),
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
    print("FIGURE ", artemis.artemis_fig_dir)
    fig.savefig(
        os.path.join(artemis.artemis_fig_dir, _file_id + "_temp.png"),
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


def spiral_pos(r, r0=1.0, p0=np.pi, h=0.05):
    # Analytic spiral position from Ogilvie & Lubow (2002)
    def mod_2pi(p):
        while p > 2 * np.pi:
            p -= 2 * np.pi
        while p < 0.0:
            p += 2 * np.pi
        return p

    if r > r0:
        return mod_2pi(
            p0 - mod_2pi(2.0 / (3 * h) * (r ** (1.5) - 1.5 * np.log(r) - 1.0))
        )
    if r < r0:
        return mod_2pi(
            p0 + mod_2pi(2.0 / (3 * h) * (r ** (1.5) - 1.5 * np.log(r) - 1.0))
        )
    return p0


def load_level(n, base="binary.out1", dir="./", off=0):
    # Loads a snapshot and combines the blocks on a single level
    # This assumes levels are nested properly
    import h5py

    try:
        fname = "{}/{}.{:05d}.phdf".format(dir, base, n)
    except:
        fname = "{}/{}.{}.phdf".format(dir, base, n)

    with h5py.File(fname, "r") as f:
        time = f["Info"].attrs["Time"]

        lvl = f["Levels"][...]
        ind = lvl == max(lvl.max() - off, 0)

        offset = f["LogicalLocations"][...][ind, :]
        x = f["Locations/x"][...][ind, :]
        y = f["Locations/y"][...][ind, :]
        z = f["Locations/z"][...][ind, :]
        db = f["gas.prim.density_0"][...][ind, :]
        pb = f["gas.prim.pressure_0"][...][ind, :]
        ub = f["gas.prim.velocity_0"][...][ind, 0, :]
        vb = f["gas.prim.velocity_0"][...][ind, 1, :]
        wb = f["gas.prim.velocity_0"][...][ind, 2, :]
        sys = f["Params"].attrs["artemis/coord_sys"]
        time = f["Info"].attrs["Time"]

    nb, bsz, bsy, bsx = db.shape

    offset[:, 0] -= offset[:, 0].min()
    offset[:, 1] -= offset[:, 1].min()
    offset[:, 2] -= offset[:, 2].min()

    nx = bsx * (offset[:, 0].max() + 1)
    ny = bsy * (offset[:, 1].max() + 1)
    nz = bsz * (offset[:, 2].max() + 1)

    d = np.zeros((nz, ny, nx))
    p = np.zeros((nz, ny, nx))
    u = np.zeros((nz, ny, nx))
    v = np.zeros((nz, ny, nx))
    w = np.zeros((nz, ny, nx))
    for n in range(nb):
        ix = offset[n, 0]
        iy = offset[n, 1]
        iz = offset[n, 2]
        d[
            iz * bsz : (iz + 1) * bsz,
            iy * bsy : (iy + 1) * bsy,
            ix * bsx : (ix + 1) * bsx,
        ] = db[n, :, :, :]
        p[
            iz * bsz : (iz + 1) * bsz,
            iy * bsy : (iy + 1) * bsy,
            ix * bsx : (ix + 1) * bsx,
        ] = pb[n, :, :, :]
        u[
            iz * bsz : (iz + 1) * bsz,
            iy * bsy : (iy + 1) * bsy,
            ix * bsx : (ix + 1) * bsx,
        ] = ub[n, :, :, :]
        v[
            iz * bsz : (iz + 1) * bsz,
            iy * bsy : (iy + 1) * bsy,
            ix * bsx : (ix + 1) * bsx,
        ] = vb[n, :, :, :]
        w[
            iz * bsz : (iz + 1) * bsz,
            iy * bsy : (iy + 1) * bsy,
            ix * bsx : (ix + 1) * bsx,
        ] = wb[n, :, :, :]

    x = np.linspace(x.min(), x.max(), nx + 1)
    y = np.linspace(y.min(), y.max(), ny + 1)
    z = np.linspace(z.min(), z.max(), nz + 1)
    return time, x, y, z, [d, u, v, w, p / d]


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
