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

# Regression to test the sphericity of the blast wave in all coordinate systems

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

_nranks = 1
_file_id = "blast"
_flux = ["hlle"]  # , "hlle", "llf"]
_recon = ["plm"]  # , "ppm"]
_geom = ["sph", "cyl", "cart", "axi"]
_args = {}
_args["cart"] = []
_args["axi"] = [
    "artemis/coordinates=axisymmetric",
    "parthenon/time/ncycle_out=100",
    "parthenon/mesh/x1min=0.0",
    "parthenon/mesh/x1max=2.0",
    "parthenon/mesh/x2min=-1.0",
    "parthenon/mesh/x2max=1.0",
    "parthenon/mesh/x3min=-0.5",
    "parthenon/mesh/x3max=0.5",
    "parthenon/mesh/ix1_bc=reflecting",
    "problem/symmetry=spherical",
]
_args["cyl"] = [
    "artemis/coordinates=axisymmetric",
    "parthenon/time/ncycle_out=100",
    "parthenon/mesh/x1min=0.0",
    "parthenon/mesh/x1max=1.0",
    "parthenon/mesh/x2min=-0.5",
    "parthenon/mesh/x2max=0.5",
    "parthenon/mesh/nx1=1024",
    "parthenon/mesh/nx2=1",
    "parthenon/mesh/x3min=-0.5",
    "parthenon/mesh/x3max=0.5",
    "parthenon/meshblock/nx2=1",
    "problem/symmetry=cylindrical",
    "problem/samples=0",
]
_args["sph"] = [
    "artemis/coordinates=spherical",
    "parthenon/time/ncycle_out=100",
    "parthenon/mesh/x1min=0.0",
    "parthenon/mesh/x1max=1.0",
    "parthenon/mesh/x2min=0.0",
    "parthenon/mesh/x2max={:.16f}".format(np.pi),
    "parthenon/mesh/nx1=1024",
    "parthenon/mesh/nx2=1",
    "parthenon/mesh/x3min=-0.5",
    "parthenon/mesh/x3max=0.5",
    "parthenon/mesh/ix1_bc=reflecting",
    "parthenon/meshblock/nx2=1",
    "problem/symmetry=spherical",
    "problem/samples=0",
]


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    for g in _geom:
        for rv in _recon:
            for fv in _flux:
                arguments = _args[g] + [
                    "parthenon/job/problem_id=" + _file_id + "_{}{:d}".format(g, 2),
                    "gas/reconstruct=" + rv,
                    "gas/riemann=" + fv,
                ]
                logger.debug("{}, {}, {}".format(g, rv, fv))
                artemis.run(_nranks, "blast/blast.in", arguments)


# Analyze outputs
def analyze():
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt

    logger.debug("Analyzing test " + __name__)
    os.makedirs(artemis.artemis_fig_dir, exist_ok=True)
    analyze_status = True

    dat2 = np.loadtxt(
        os.path.join(
            artemis.get_source_directory(), "tst", "scripts", "coords", "sedov2d.dat"
        ),
        comments="#",
    )
    dat3 = np.loadtxt(
        os.path.join(
            artemis.get_source_directory(), "tst", "scripts", "coords", "sedov3d.dat"
        ),
        comments="#",
    )
    tol = 1.0
    fig, axes = plt.subplots(2, 3, figsize=(8 * 3, 6 * 2))
    axes[0, 0].plot(dat2[:, 0], dat2[:, 1], "-k")
    axes[0, 1].plot(dat2[:, 0], dat2[:, 2], "-k")
    axes[0, 2].plot(dat2[:, 0], dat2[:, 3], "-k")
    axes[1, 0].plot(dat3[:, 0], dat3[:, 1], "-k")
    axes[1, 1].plot(dat3[:, 0], dat3[:, 2], "-k")
    axes[1, 2].plot(dat3[:, 0], dat3[:, 3], "-k")
    L2 = []
    for g in _geom:
        pf = (
            interp1d(dat2[:, 0], dat2[:, 3])
            if g in ["cart", "cyl"]
            else interp1d(dat3[:, 0], dat3[:, 3])
        )
        res = load_snap(
            os.path.join(
                artemis.get_run_directory(),
                _file_id + "_{}{:d}.out1.final.phdf".format(g, 2),
            )
        )
        pres = res[4][-1]
        xc = 0.5 * (res[1][:, 1:] + res[1][:, :-1])
        if g in ["axi", "cart"]:
            yc = 0.5 * (res[2][:, 1:] + res[2][:, :-1])
            dx, dy, dr, vr = transform_2d(xc, yc, res[4][1], res[4][2])
            pans = pf(dr)
            ind = 0 if g == "cart" else 1
            axes[ind, 0].plot(dr.ravel(), res[4][0].ravel(), ".", ms=1, label=g)
            axes[ind, 1].plot(dr.ravel(), vr.ravel(), ".", ms=1, label=g)
            axes[ind, 2].plot(dr.ravel(), res[4][-1].ravel(), ".", ms=2, label=g)
        else:
            pans = pf(xc)
            ind = 0 if g == "cyl" else 1
            axes[ind, 0].plot(xc.ravel(), res[4][0].ravel(), ".", ms=1, label=g)
            axes[ind, 1].plot(xc.ravel(), res[4][1].ravel(), ".", ms=1, label=g)
            axes[ind, 2].plot(xc.ravel(), res[4][-1].ravel(), ".", ms=2, label=g)
        # L2 norm on pressure
        L2v = np.sqrt(((pres - pans) ** 2).mean())
        L2.append([g, L2v])
    axes[0, 0].set_ylabel("$\\rho$", fontsize=20)
    axes[0, 1].set_ylabel("$v_R$", fontsize=20)
    axes[0, 2].set_ylabel("$P$", fontsize=20)
    axes[0, 0].set_ylabel("$\\rho$", fontsize=20)
    axes[0, 1].set_ylabel("$v_r$", fontsize=20)
    axes[0, 2].set_ylabel("$P$", fontsize=20)
    axes[0, 0].legend(loc="best", fontsize=14)
    axes[1, 0].legend(loc="best", fontsize=14)
    for ax in axes.flatten():
        ax.set_xlabel("$R$", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_xlim(0, 0.6)
        ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(
        os.path.join(artemis.artemis_fig_dir, _file_id + ".png"), bbox_inches="tight"
    )

    # Check failure criterion
    for res in L2:
        if res[1] >= tol:
            logger.warning(
                "Error too large for {0} configuration, "
                "pres L2 err = {1:.8e} threshold: {2:.8e} ".format(res[0], res[1], tol)
            )
            analyze_status = False

    return analyze_status


def transform_2d(xc, yc, vx, vy, pos=(0, 0)):
    # xc are (nb,nx)
    rr = np.zeros(vx.shape)
    xx = np.zeros(rr.shape)
    yy = np.zeros(rr.shape)
    for b in range(xx.shape[0]):
        for j in range(yc.shape[1]):
            xx[b, 0, j, :] = xc[b, :] - pos[0]
            yy[b, 0, j, :] = yc[b, j] - pos[1]
            rr[b, 0, j, :] = np.sqrt(
                (xc[b, :] - pos[0]) ** 2 + (yc[b, j] - pos[1]) ** 2
            )
    vr = xx / rr * vx + yy / rr * vy
    return xx, yy, rr, vr


def transform_3d(xc, yc, zc, vx, vy, vz, pos=(0, 0, 0)):
    rr = np.zeros((zc.shape[0], yc.shape[0], xc.shape[0]))
    xx = np.zeros(rr.shape)
    yy = np.zeros(rr.shape)
    zz = np.zeros(rr.shape)
    for k in range(zc.shape[0]):
        for j in range(yc.shape[0]):
            xx[k, j, :] = xc - pos[0]
            yy[k, j, :] = yc[j] - pos[1]
            zz[k, j, :] = zc[k] - pos[2]
            rr[k, j, :] = np.sqrt(
                (xc - pos[0]) ** 2 + (yc[j] - pos[1]) ** 2 + (zc[k] - pos[2]) ** 2
            )
    vr = xx / rr * vx + yy / rr * vy + zz / rr * vz
    return xx, yy, zz, rr, vr


def load_snap(fname):
    import h5py

    with h5py.File(fname, "r") as f:
        time = f["Info"].attrs["Time"]
        x = f["Locations/x"][...]
        y = f["Locations/y"][...]
        z = f["Locations/z"][...]
        d = f["gas.prim.density_0"][...]
        p = f["gas.prim.pressure_0"][...]
        u = f["gas.prim.velocity_0"][...][:, 0, :]
        v = f["gas.prim.velocity_0"][...][:, 1, :]
        w = f["gas.prim.velocity_0"][...][:, 2, :]
    return time, x, y, z, (d, u, v, w, p)


def sedov_cyl(fcyl_1d, fcart_2d, savefig=None):
    import matplotlib.pyplot as plt

    dat = np.loadtxt("sedov2d.dat")
    fig, axes = plt.subplots(1, 3, figsize=(8 * 3, 6))

    axes[0].plot(dat[:, 0], dat[:, 1], "-k", label="Exact", lw=2)
    axes[1].plot(dat[:, 0], dat[:, 2], "-k", label="Exact", lw=2)
    axes[2].plot(dat[:, 0], dat[:, 3], "-k", label="Exact", lw=2)

    res = load_snap(fcart_2d)
    for b in range(res[1].shape[0]):
        xc = 0.5 * (res[1][b, 1:] + res[1][b, :-1])
        yc = 0.5 * (res[2][b, 1:] + res[2][b, :-1])
        x, y, r, vr = transform_2d(xc, yc, res[4][1][b, :], res[4][2][b, :])
        axes[0].plot(
            r.ravel(),
            res[4][0][b, :].ravel(),
            ".b",
            ms=1,
            alpha=0.5,
            label="Cartesian-2D" if b == 0 else None,
        )
        axes[1].plot(
            r.ravel(),
            vr.ravel(),
            ".b",
            ms=1,
            alpha=0.5,
            label="Cartesian-2D" if b == 0 else None,
        )
        axes[2].plot(
            r.ravel(),
            res[4][-1][b, :].ravel(),
            ".b",
            ms=1,
            alpha=0.5,
            label="Cartesian-2D" if b == 0 else None,
        )

    res = load_snap(fcyl_1d)
    for b in range(res[1].shape[0]):
        xc = 0.5 * (res[1][b, 1:] + res[1][b, :-1])
        axes[0].plot(
            xc.ravel(),
            res[4][0][b, :].ravel(),
            ".r",
            label="Cylindrical-1D" if b == 0 else None,
        )
        axes[1].plot(
            xc.ravel(),
            res[4][1][b, :].ravel(),
            ".r",
            label="Cylindrical-1D" if b == 0 else None,
        )
        axes[2].plot(
            xc.ravel(),
            res[4][-1][b, :].ravel(),
            ".r",
            label="Cylindrical-1D" if b == 0 else None,
        )

    axes[0].set_ylabel("$\\rho$", fontsize=20)
    axes[1].set_ylabel("$v_R$", fontsize=20)
    axes[2].set_ylabel("$P$", fontsize=20)
    axes[0].legend(loc="best", fontsize=14)
    axes[0].text(
        0.98,
        0.98,
        "Cylindrical Blast",
        fontsize=14,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )
    for ax in axes:
        ax.set_xlabel("$R$", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_xlim(0, 0.6)
        ax.minorticks_on()
    fig.tight_layout()
    if savefig is not None:
        os.makedirs(artemis.artemis_fig_dir, exist_ok=True)
        fig.savefig(os.path.join(artemis.artemis_fig_dir, savefig), bbox_inches="tight")


def sedov_sph(fsph_1d, faxi_2d, fcart_3d, savefig=None):
    dat = np.loadtxt("sedov3d.dat")
    fig, axes = plt.subplots(1, 3, figsize=(8 * 3, 6))

    axes[0].plot(dat[:, 0], dat[:, 1], "-k", label="Exact", lw=2)
    axes[1].plot(dat[:, 0], dat[:, 2], "-k", label="Exact", lw=2)
    axes[2].plot(dat[:, 0], dat[:, 3], "-k", label="Exact", lw=2)

    res = load_snap(fcart_3d)
    for b in range(res[1].shape[0]):
        xc = 0.5 * (res[1][b, 1:] + res[1][b, :-1])
        yc = 0.5 * (res[2][b, 1:] + res[2][b, :-1])
        zc = 0.5 * (res[3][b, 1:] + res[3][b, :-1])
        x, y, z, r, vr = transform_3d(
            xc, yc, zc, res[4][1][b, :], res[4][2][b, :], res[4][3][b, :]
        )
        axes[0].plot(
            r.ravel(),
            res[4][0][b, :].ravel(),
            ".b",
            ms=1,
            alpha=0.3,
            label="Cartesian-3D" if b == 0 else None,
        )
        axes[1].plot(
            r.ravel(),
            vr.ravel(),
            ".b",
            ms=1,
            alpha=0.3,
            label="Cartesian-3D" if b == 0 else None,
        )
        axes[2].plot(
            r.ravel(),
            res[4][-1][b, :].ravel(),
            ".b",
            ms=1,
            alpha=0.3,
            label="Cartesian-3D" if b == 0 else None,
        )

    res = load_snap(faxi_2d)
    for b in range(res[1].shape[0]):
        xc = 0.5 * (res[1][b, 1:] + res[1][b, :-1])
        yc = 0.5 * (res[2][b, 1:] + res[2][b, :-1])
        x, y, r, vr = transform_2d(xc, yc, res[4][1][b, :], res[4][2][b, :])
        axes[0].plot(
            r.ravel(),
            res[4][0][b, :].ravel(),
            ".",
            c="magenta",
            ms=1,
            alpha=0.5,
            label="Axisymmetric-2D" if b == 0 else None,
        )
        axes[1].plot(
            r.ravel(),
            vr.ravel(),
            ".",
            c="magenta",
            ms=1,
            alpha=0.5,
            label="Axisymmetric-2D" if b == 0 else None,
        )
        axes[2].plot(
            r.ravel(),
            res[4][-1][b, :].ravel(),
            ".",
            c="magenta",
            ms=1,
            alpha=0.5,
            label="Axisymmetric-2D" if b == 0 else None,
        )

    res = load_snap(fsph_1d)
    for b in range(res[1].shape[0]):
        xc = 0.5 * (res[1][b, 1:] + res[1][b, :-1])
        axes[0].plot(
            xc.ravel(),
            res[4][0][b, :].ravel(),
            ".r",
            label="Spherical-1D" if b == 0 else None,
        )
        axes[1].plot(
            xc.ravel(),
            res[4][1][b, :].ravel(),
            ".r",
            label="Spherical-1D" if b == 0 else None,
        )
        axes[2].plot(
            xc.ravel(),
            res[4][-1][b, :].ravel(),
            ".r",
            label="Spherical-1D" if b == 0 else None,
        )

    axes[0].set_ylabel("$\\rho$", fontsize=20)
    axes[1].set_ylabel("$v_r$", fontsize=20)
    axes[2].set_ylabel("$P$", fontsize=20)
    axes[0].legend(loc="best", fontsize=14)
    axes[0].text(
        0.98,
        0.98,
        "Spherical Blast",
        fontsize=14,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )
    for ax in axes:
        ax.set_xlabel("$r$", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_xlim(0, 0.6)
        ax.minorticks_on()
    fig.tight_layout()
    if savefig is not None:
        os.makedirs(artemis.artemis_fig_dir, exist_ok=True)
        fig.savefig(os.path.join(artemis.artemis_fig_dir, savefig), bbox_inches="tight")
