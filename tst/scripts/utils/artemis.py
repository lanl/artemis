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

# NOTE(@pdmullen): The following is largely borrowed from the open-source Athena++/AthenaK
# softwares.

# Functions for interfacing with Artemis during testing

# Modules
import logging
import numpy as np
import os
import subprocess
import datetime
from timeit import default_timer as timer
from .log_pipe import LogPipe

# Global variables
artemis_dir = os.path.abspath(os.path.join(__file__, "../../../../"))
current_dir = "dev/null"
artemis_exe_dir = "/dev/null"
artemis_inputs_dir = "/dev/null"
artemis_data_dir = "/dev/null"
artemis_fig_dir = "/dev/null"
artemis_log_dir = "/dev/null"
use_mpi_oversubscribe = False
use_supplied_exe = False


# Optionally invoke MPI oversubscribe
def set_mpi_oversubscribe(use_oversubscribe):
    global use_mpi_oversubscribe
    use_mpi_oversubscribe = use_oversubscribe


# Invoke custom executable
def set_supplied_exe(use_exe):
    global use_supplied_exe
    use_supplied_exe = use_exe
    custom_msg = "use pre-existing" if use_exe else "build"
    print("...Regression will " + custom_msg + " executable...\n")


# Optionally set custom path for executable, and update other variables related to where
# we run the code
def set_paths(executable_dir, output_dir):
    global current_dir
    global artemis_exe_dir
    global artemis_inputs_dir
    global artemis_data_dir
    global artemis_fig_dir
    global artemis_log_dir
    current_dir = os.getcwd()
    artemis_exe_dir = executable_dir
    artemis_inputs_dir = os.path.join(artemis_dir, "inputs")
    artemis_data_dir = os.path.join(output_dir, "data")
    artemis_fig_dir = os.path.join(output_dir, "figs")
    artemis_log_dir = os.path.join(output_dir, "logs")
    print("Artemis Regression Paths:")
    print("   Artemis directory:    ", artemis_dir)
    print("   Current directory:    ", current_dir)
    print("   Executable directory: ", artemis_exe_dir)
    print("   Inputs directory:     ", artemis_inputs_dir)
    print("   Data directory:       ", artemis_data_dir)
    print("   Figures directory:    ", artemis_fig_dir)
    print("   Logs directory:       ", artemis_log_dir)
    print("\n")


get_artemis_dir = lambda: artemis_dir
get_current_dir = lambda: current_dir
get_exe_dir = lambda: artemis_exe_dir
get_inputs_dir = lambda: artemis_inputs_dir
get_data_dir = lambda: artemis_data_dir
get_fig_dir = lambda: artemis_fig_dir
get_log_dir = lambda: artemis_log_dir
get_supplied_exe = lambda: use_supplied_exe


# Function for compiling Artemis
def make(cmake_args, make_nproc):
    logger = logging.getLogger("artemis.make")
    out_log = LogPipe("artemis.make", logging.INFO)
    current_dir = os.getcwd()
    try:
        subprocess.check_call(["mkdir", "build"], stdout=out_log)
        build_dir = current_dir + "/build/"
        os.chdir(build_dir)
        cmake_command = ["cmake", artemis_dir] + cmake_args
        make_command = ["make", "-j" + str(make_nproc)]
        try:
            t0 = timer()
            logger.debug("Executing: " + " ".join(cmake_command))
            subprocess.check_call(cmake_command, stdout=out_log)
            logger.debug("Executing: " + " ".join(make_command))
            subprocess.check_call(make_command, stdout=out_log)
            logger.debug("Build took {0:.3g} seconds.".format(timer() - t0))
        except subprocess.CalledProcessError as err:
            logger.error("Something bad happened", exc_info=True)
            raise ArtemisError(
                "Return code {0} from command '{1}'".format(
                    err.returncode, " ".join(err.cmd)
                )
            )
    finally:
        out_log.close()
        os.chdir(current_dir)


# Function for running Artemis (with MPI)
def run(nproc, input_filename, arguments, restart=None):
    # global run_directory
    out_log = LogPipe("artemis.run", logging.INFO)

    # Build the run command
    run_command = ["mpiexec"]
    if use_mpi_oversubscribe:
        run_command += ["--oversubscribe"]
    run_command += ["-n", str(nproc), os.path.join(artemis_exe_dir, "artemis")]
    if restart is not None:
        run_command += ["-r", os.path.join(artemis_data_dir, restart)]
    run_command += ["-i", os.path.join(artemis_inputs_dir, input_filename)]
    run_command += ["-d", artemis_data_dir]

    try:
        os.chdir(artemis_exe_dir)
        cmd = run_command + arguments
        logging.getLogger("artemis.run").debug("Executing: " + " ".join(cmd))
        subprocess.check_call(cmd, stdout=out_log)
    except subprocess.CalledProcessError as err:
        raise ArtemisError(
            "Return code {0} from command '{1}'".format(
                err.returncode, " ".join(err.cmd)
            )
        )
    finally:
        out_log.close()
        os.chdir(current_dir)


# Function for reading Artemis data
def load_level(n, base, dir="./", off=0):
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


# Associated functions for plotting Artemis data
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
