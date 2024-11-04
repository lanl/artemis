#!/usr/bin/env python3
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

import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import ahdf


# Plot a 2D slice of a variable to a provided axis.
def plot(
    fig, ax, filename, variable_name, draw_meshblocks, slice, vmin, vmax, coords, scale
):
    dump = ahdf.ahdf(filename)

    # Coordinate-dependent defaults
    if dump.coordinates == "cartesian":
        if slice is None:
            slice = "xz"
    assert slice is not None, "No default slice option available for this mesh!"

    variable = dump.Get(variable_name)
    assert variable is not None, f"Variable {variable_name} does not exist!"
    if scale == "log":
        variable = np.log10(variable)

    # Check if each Meshblock is included in slice
    plot_meshblocks = np.zeros(dump.NumBlocks)
    slice_index = np.zeros(dump.NumBlocks)
    for b in range(dump.NumBlocks):
        if slice == "xy":
            z_slice = 0.0
            if (
                dump.BlockBounds[b][4] + dump.DX3[b] / 10.0 < z_slice
                and dump.BlockBounds[b][5] >= z_slice
            ):
                plot_meshblocks[b] = True
                slice_index[b] = (z_slice - dump.BlockBounds[b][4]) / dump.DX3[b]
        elif slice == "xz":
            y_slice = 0.0  # +x half-plane only
            if dump.BlockBounds[b][2] < y_slice and dump.BlockBounds[b][3] >= y_slice:
                plot_meshblocks[b] = True
                slice_index[b] = (y_slice - dump.BlockBounds[b][2]) / dump.DX2[b]
        else:
            assert False, f'slice "{slice}" unrecognized!'
    assert sum(plot_meshblocks) > 0, "No meshblocks within slice!"

    # Plot all meshblocks within slice
    for b in range(dump.NumBlocks):
        if plot_meshblocks[b]:
            if slice == "xy":
                idx = int(slice_index[b])
                assert idx >= 0 and idx < dump.NX3, "Slice index is out of bounds!"
                if coords == "cartesian":
                    im = ax.pcolormesh(
                        dump.x[b, idx, :, :],
                        dump.y[b, idx, :, :],
                        variable[b, idx, :, :],
                        vmin=vmin,
                        vmax=vmax,
                    )
                elif coords == "code":
                    im = ax.pcolormesh(
                        dump.X1[b, idx, :, :],
                        dump.X2[b, idx, :, :],
                        variable[b, idx, :, :],
                        vmin=vmin,
                        vmax=vmax,
                    )
            elif slice == "xz":
                idx = int(slice_index[b])
                assert idx >= 0 and idx < dump.NX2, "Slice index is out of bounds!"
                if coords == "cartesian":
                    im = ax.pcolormesh(
                        dump.x[b, :, idx, :],
                        dump.z[b, :, idx, :],
                        variable[b, :, idx, :],
                        vmin=vmin,
                        vmax=vmax,
                    )
                elif coords == "code":
                    im = ax.pcolormesh(
                        dump.X1[b, :, idx, :],
                        dump.X3[b, :, idx, :],
                        variable[b, :, idx, :],
                        vmin=vmin,
                        vmax=vmax,
                    )
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Label axes
    ax.set_title(f"{variable_name}")
    if coords == "cartesian":
        if slice == "xy":
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif slice == "xz":
            ax.set_xlabel("x")
            ax.set_ylabel("z")
    elif coords == "code":
        if slice == "xy":
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
        elif slice == "xz":
            ax.set_xlabel("X1")
            ax.set_ylabel("X3")

    # Draw meshblock boundaries
    if draw_meshblocks:
        lw = 0.25
        alpha = 0.1
        for b in range(dump.NumBlocks):
            if plot_meshblocks[b]:
                if slice == "xy":
                    if coords == "cartesian":
                        ax.plot(
                            dump.x[b, 0, 0, :],
                            dump.y[b, 0, 0, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.x[b, 0, :, 0],
                            dump.y[b, 0, :, 0],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.x[b, 0, -1, :],
                            dump.y[b, 0, -1, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.x[b, 0, :, -1],
                            dump.y[b, 0, :, -1],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                    elif coords == "code":
                        ax.plot(
                            dump.X1[b, 0, 0, :],
                            dump.X2[b, 0, 0, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.X1[b, 0, :, 0],
                            dump.X2[b, 0, :, 0],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.X1[b, 0, -1, :],
                            dump.X2[b, 0, -1, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.X1[b, 0, :, -1],
                            dump.X2[b, 0, :, -1],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                elif slice == "xz":
                    if coords == "cartesian":
                        ax.plot(
                            dump.x[b, 0, 0, :],
                            dump.z[b, 0, 0, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.x[b, :, 0, 0],
                            dump.z[b, :, 0, 0],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.x[b, -1, 0, :],
                            dump.z[b, -1, 0, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.x[b, :, 0, -1],
                            dump.z[b, :, 0, -1],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                    elif coords == "code":
                        ax.plot(
                            dump.X1[b, 0, 0, :],
                            dump.X3[b, 0, 0, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.X1[b, :, 0, 0],
                            dump.X3[b, :, 0, 0],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.X1[b, -1, 0, :],
                            dump.X3[b, -1, 0, :],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )
                        ax.plot(
                            dump.X1[b, :, 0, -1],
                            dump.X3[b, :, 0, -1],
                            color="k",
                            linewidth=lw,
                            alpha=alpha,
                        )

    # Uniform aspect ratio for Cartesian coordinates
    if coords == "cartesian":
        ax.set_aspect("equal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Artemis output")
    parser.add_argument("filename", type=str, help="Artemis output file to plot")
    parser.add_argument(
        "variable", type=str, default="gas.prim.density_0", help="Variable name to plot"
    )
    parser.add_argument(
        "--meshblocks", action="store_true", help="Draw meshblock boundaries"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        choices=["xy", "xz"],
        help="Slice of 3D data to make.",
    )
    parser.add_argument(
        "--vmin", type=float, default=-5, help="Minimum value of colorbar"
    )
    parser.add_argument(
        "--vmax", type=float, default=0, help="Maximum value of colorbar"
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="log",
        choices=["linear", "log"],
        help="Scale in which to plot variable.",
    )
    parser.add_argument(
        "--coords",
        type=str,
        default="cartesian",
        help="Coordinates to plot. Choices are: [cartesian code]",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1)

    plot(
        fig,
        ax,
        args.filename,
        args.variable,
        args.meshblocks,
        args.slice,
        args.vmin,
        args.vmax,
        args.coords,
        args.scale,
    )

    savename = os.path.basename(args.filename)[:-5] + ".png"
    print(f"Saving plot as {savename}")
    plt.savefig(savename, dpi=300, bbox_inches="tight")
