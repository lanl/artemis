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

# Functions for test analyses

# Modules
import h5py
import logging
import subprocess
import matplotlib
import matplotlib.cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Function for reading Artemis data
def load_level(n, base, dir="./", off=0):
    # Loads a snapshot and combines the blocks on a single level
    # This assumes levels are nested properly

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


# Associated functions for plotting Artemis data
def create_colorbar(ax, norm, where="top", cax=None, cmap="viridis", **kargs):
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


# Shared function to compute analytic Ogilvie & Lubow (2002) spiral positions
def spiral_pos(r, r0=1.0, p0=np.pi, h=0.05):
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


# General exception class for these functions
class ArtemisError(RuntimeError):
    pass
