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
import datetime
from timeit import default_timer as timer
from .log_pipe import LogPipe

# Global variables
current_dir = os.getcwd()
artemis_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
artemis_executable = os.path.join(artemis_dir, "tst", "build", "src", "artemis")
artemis_inputs_dir = os.path.join(artemis_dir, "inputs")
# artemis_fig_dir = "./figs/"
artemis_run_dir = os.path.join(artemis_dir, "tst", "build", "src", "tst")
artemis_fig_dir = os.path.join(artemis_dir, "tst", "figs")
artemis_log_dir = os.path.join(artemis_dir, "tst")
custom_exe = False


# Optionally set custom path for executable, and update other variables related to where
# we run the code
def set_executable(executable_path):
    global artemis_executable
    global artemis_run_dir
    global artemis_fig_dir
    global artemis_log_dir
    global custom_exe
    artemis_executable = executable_path
    artemis_run_dir = os.path.join(os.path.dirname(artemis_executable), "tst")
    artemis_fig_dir = os.path.join(current_dir, "figs")
    artemis_log_dir = current_dir
    custom_exe = True


# Function for returning the path to the run directory for this set of tests
def get_run_directory():
    return artemis_run_dir


# Provide base directory of artemis source tree
def get_source_directory():
    return artemis_dir


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
    os.makedirs(artemis_run_dir, exist_ok=True)

    # Build the run command
    run_command = ["mpiexec", "--oversubscribe", "-n", str(nproc), artemis_executable]
    if restart is not None:
        run_command += ["-r", restart]
    input_filename_full = os.path.join(artemis_inputs_dir, input_filename)
    run_command += ["-i", input_filename_full]

    try:
        # os.chdir(run_directory)
        os.chdir(artemis_run_dir)
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
