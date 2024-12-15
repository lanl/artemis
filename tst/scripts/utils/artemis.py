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
import os
import subprocess
from timeit import default_timer as timer
from .log_pipe import LogPipe

# Global variables
artemis_dir = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".."))
current_dir = os.getcwd()
artemis_exe_dir = "/dev/null"
artemis_outputs_dir = "/dev/null"
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
    global artemis_exe_dir
    global artemis_outputs_dir
    global artemis_inputs_dir
    global artemis_data_dir
    global artemis_fig_dir
    global artemis_log_dir
    artemis_exe_dir = executable_dir
    artemis_outputs_dir = output_dir
    artemis_inputs_dir = os.path.join(artemis_dir, "inputs")
    artemis_data_dir = os.path.join(output_dir, "data")
    artemis_fig_dir = os.path.join(output_dir, "figs")
    artemis_log_dir = os.path.join(output_dir, "logs")
    print("Artemis Regression Paths:")
    print("   Artemis directory:    ", artemis_dir)
    print("   Current directory:    ", current_dir)
    print("   Executable directory: ", artemis_exe_dir)
    print("   Outputs directory:    ", artemis_outputs_dir)
    print("   Inputs directory:     ", artemis_inputs_dir)
    print("   Data directory:       ", artemis_data_dir)
    print("   Figures directory:    ", artemis_fig_dir)
    print("   Logs directory:       ", artemis_log_dir)
    print("\n")


get_artemis_dir = lambda: artemis_dir
get_current_dir = lambda: current_dir
get_exe_dir = lambda: artemis_exe_dir
get_outputs_dir = lambda: artemis_outputs_dir
get_inputs_dir = lambda: artemis_inputs_dir
get_data_dir = lambda: artemis_data_dir
get_fig_dir = lambda: artemis_fig_dir
get_log_dir = lambda: artemis_log_dir
get_supplied_exe = lambda: use_supplied_exe


# Function for compiling Artemis
def make(cmake_args, make_nproc):
    logger = logging.getLogger("artemis.make")
    out_log = LogPipe("artemis.make", logging.INFO)
    build_dir = os.path.join(artemis_dir, "tst/build")
    build_src_dir = os.path.join(build_dir, "src")
    try:
        cmake_command = ["cmake", "-B", build_dir] + cmake_args
        make_command = ["make", "-j" + str(make_nproc), "-C", build_src_dir]
        try:
            t0 = timer()
            os.chdir(artemis_dir)
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
        os.chdir(artemis_outputs_dir)
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


# General exception class for these functions
class ArtemisError(RuntimeError):
    pass
