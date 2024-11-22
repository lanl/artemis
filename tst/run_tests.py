#!/usr/bin/env python
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

# Regression test script for Artemis.

# Usage: From this directory, call this script with python:
#        python run_tests.py

# Notes:
#   - Requires Python 3+.
#   - This file should not be modified when adding new scripts.
#   - This file is largely borrowed from the open-source Athena++/AthenaK CI regression,
#     adapted to work with LANL Darwin runners
#   - This file was created in part by one of OpenAI's generative AI models

# Modules
import argparse
import os
import logging
import logging.config
import shutil
from collections import OrderedDict
from importlib import reload
from pkgutil import iter_modules
from timeit import default_timer as timer

# Prevent generation of .pyc files
# This should be set before importing any user modules
import sys

sys.dont_write_bytecode = True

# Artemis modules
import scripts.utils.artemis as artemis  # noqa

# Artemis logger
logger = logging.getLogger("artemis")


def process_suite(filename):
    tests = []
    aname = os.path.join(artemis.get_artemis_dir(), "tst")
    fname = os.path.join(aname, "suites", filename)

    with open(fname, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                if line[0] != "#":
                    test_name = line.split("#")[0]
                    if test_name[-1] == "/":
                        test_name = test_name[:-1]
                    if ".suite" in test_name:
                        tests += process_suite(test_name)
                    elif "/" in test_name:
                        tests.append(test_name.replace("/", "."))
                    else:
                        dir_test_names = [
                            name
                            for _, name, _ in iter_modules(
                                path=[os.path.join(aname, "scripts", test_name)],
                                prefix=test_name + ".",
                            )
                        ]
                        tests += dir_test_names
    return tests


# Main function
def main(**kwargs):
    # Make list of tests to run
    scripts_path = os.path.join(artemis.get_artemis_dir(), "tst/scripts/")
    tests = kwargs.pop("tests")
    test_names = []
    if len(tests) == 0:  # run all tests
        for _, directory, ispkg in iter_modules(path=[scripts_path]):
            if ispkg and directory != "utils":
                dir_test_names = [
                    name
                    for _, name, _ in iter_modules(
                        path=[os.path.join(scripts_path, directory)],
                        prefix=directory + ".",
                    )
                ]
                test_names.extend(dir_test_names)
    else:  # run selected tests
        for test in tests:
            if ".suite" in test:
                test_names += process_suite(test)
            if test[-1] == "/":
                test = test[:-1]  # remove trailing slash
            if "/" in test:  # specific test specified
                test_names.append(test.replace("/", "."))
            else:  # test suite specified
                dir_test_names = [
                    name
                    for _, name, _ in iter_modules(
                        path=[os.path.join(scripts_path, test)],
                        prefix=test + ".",
                    )
                ]
                test_names.extend(dir_test_names)

    # Remove duplicate test entries while preserving the original order
    test_names = list(OrderedDict.fromkeys(test_names))
    logger.info("Running: " + ", ".join(test_names))

    # Run tests
    build_dir = os.path.join(artemis.get_artemis_dir(), "tst")
    test_times = []
    test_results = []
    test_errors = []

    # Extract arguments
    try:
        # Check that required modules are installed for all test dependencies
        deps_installed = True
        for name in test_names:
            try:
                name_full = "scripts." + name
                module = __import__(
                    name_full, globals(), locals(), fromlist=["run", "analyze"]
                )
            except ImportError as e:
                if sys.version_info >= (3, 6, 0):  # ModuleNotFoundError
                    missing_module = e.name
                else:
                    missing_module = e.message.split(" ")[-1]
                logger.warning("Unable to " 'import "{:}".'.format(missing_module))
                deps_installed = False
        if not deps_installed:
            logger.warning("WARNING! Not all required Python modules are available")

        # Build Artemis
        if not artemis.get_supplied_exe() and not kwargs.pop("reuse_build"):
            try:
                os.system("rm -rf {0}/build".format(build_dir))
                # insert arguments for artemis.make()
                artemis_cmake_args = kwargs.pop("cmake")
                artemis_make_nproc = kwargs.pop("make_nproc")
                artemis.make(artemis_cmake_args, artemis_make_nproc)
            except Exception:
                logger.error("Exception occurred", exc_info=True)
                test_errors.append("make()")
                raise TestError("Unable to build Artemis")

        # Build working directory and copy librebound.so into working directory
        os.makedirs(artemis.get_outputs_dir(), exist_ok=True)
        reb_path = os.path.join(artemis.get_exe_dir(), "librebound.so")
        if not os.path.exists(reb_path):
            raise TestError(f'librebound.so not found at "{reb_path}"!')
        shutil.copy(reb_path, artemis.get_outputs_dir())

        # Run each test
        for name in test_names:
            t0 = timer()
            try:
                name_full = "scripts." + name
                module = __import__(
                    name_full, globals(), locals(), fromlist=["run", "analyze"]
                )
                reload(module)
                try:
                    run_ret = module.run()
                except Exception:
                    logger.error("Exception occurred", exc_info=True)
                    test_errors.append("run()")
                    raise TestError(name_full.replace(".", "/") + ".py")
                try:
                    result = module.analyze()
                except Exception:
                    logger.error("Exception occurred", exc_info=True)
                    test_errors.append("analyze()")
                    raise TestError(name_full.replace(".", "/") + ".py")
            except TestError as err:
                test_results.append(False)
                logger.error("---> Error in " + str(err))
                # do not measure runtime for failed/incomplete tests
                test_times.append(None)
            else:
                test_times.append(timer() - t0)
                msg = "Test {0} took {1:.3g} seconds to complete."
                msg = msg.format(name, test_times[-1])
                logging.getLogger("artemis.tests." + name).debug(msg)
                test_results.append(result)
                test_errors.append(None)
            # For CI, print after every individual test has finished
            logger.info("{} test: run(), analyze() finished".format(name))
    finally:
        if not kwargs.pop("save_build") and not artemis.get_supplied_exe():
            os.system("rm -rf {0}/build".format(build_dir))

    # Report test results
    logger.info("\nResults:")
    for name, result, error, time in zip(
        test_names, test_results, test_errors, test_times
    ):
        result_string = "passed" if result else "failed"
        error_string = (
            " -- unexpected failure in {0} stage".format(error)
            if error is not None
            else "; time elapsed: {0:.3g} s".format(time)
        )
        logger.info("    {0}: {1}{2}".format(name, result_string, error_string))
    logger.info("")
    num_tests = len(test_results)
    num_passed = test_results.count(True)
    test_string = "test" if num_tests == 1 else "tests"
    logger.info(
        "Summary: {0} out of {1} {2} "
        "passed\n".format(num_passed, num_tests, test_string)
    )
    # For CI calling scripts, explicitly raise error if not all tests passed
    if num_passed == num_tests:
        return 0
    else:
        raise TestError()


# Exception for unexpected behavior by individual tests
class TestError(RuntimeError):
    pass


# Filter out critical exceptions
class CriticalExceptionFilter(logging.Filter):
    def filter(self, record):
        return not record.exc_info or record.levelno != logging.CRITICAL


# Initialize log
def log_init(args):
    kwargs = vars(args)
    logging.basicConfig(level=0)  # setting to 0 gives output cntrl to handler
    logger.propagate = False  # don't use default handler
    c_handler = logging.StreamHandler()  # console/terminal handler
    c_handler.setLevel(logging.INFO)
    c_handler.addFilter(CriticalExceptionFilter())  # stderr errors to screen
    c_handler.setFormatter(logging.Formatter("%(message)s"))  # only show msg
    logger.addHandler(c_handler)
    # setup log_file
    log_fn = kwargs.pop("log_file")
    if log_fn:
        os.makedirs(artemis.get_log_dir(), exist_ok=True)
        f_handler = logging.FileHandler(os.path.join(artemis.get_log_dir(), log_fn))
        f_handler.setLevel(0)  # log everything
        f_format = logging.Formatter(
            "%(asctime)s|%(levelname)s" ":%(name)s: %(message)s"
        )
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    logger.debug("Starting Artemis regression tests")


# Reads Artemis CMakeCache.txt to find executable
def read_cmakecache(filename):
    with open(filename, "r") as f:
        for line in f.readlines():
            if "artemis_BINARY_DIR" in line:
                return os.path.join(line.split("=")[-1].strip(), "src", "artemis")


# Sets global variables
def set_globals(args):
    kwargs = vars(args)

    # Set MPI oversubscribe
    if kwargs.pop("use_oversubscribe"):
        artemis.set_mpi_oversubscribe(True)

    # Check for executable path and output directory args
    out_dir = kwargs.pop("output_dir")
    out_dir = os.path.join(out_dir, "testing") if out_dir is not None else None
    exe_path = kwargs.pop("exe")
    cwd = os.getcwd()

    # Set the correct paths
    if exe_path is not None:
        out_dir = (
            os.path.join(artemis_dir, "tst/testing") if out_dir is None else out_dir
        )
        reb_path = os.path.join(os.path.dirname(exe_path), "librebound.so")

        if not (os.path.exists(exe_path) and os.access(exe_path, os.X_OK)):
            raise TestError(f'Provided exe "{exe_path}" not found or cannot be run!')

        if not os.path.exists(reb_path):
            raise TestError(f'librebound.so not found at "{reb_path}"!')

        abs_out_dir = os.path.abspath(out_dir)
        abs_exe_dir = os.path.abspath(os.path.dirname(exe_path))
        artemis.set_paths(abs_exe_dir, abs_out_dir)
        artemis.set_supplied_exe(True)
    else:
        # If we are in a directory with an executable, default to using that
        local_path = os.path.join(cwd, "artemis")
        if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
            exe_path = local_path
            out_dir = os.path.join(cwd, "testing") if out_dir is None else out_dir
            reb_path = os.path.join(os.path.dirname(exe_path), "librebound.so")

            if not os.path.exists(reb_path):
                raise TestError(f'librebound.so not found at "{reb_path}"!')

            abs_out_dir = os.path.abspath(out_dir)
            abs_exe_dir = os.path.abspath(os.path.dirname(exe_path))
            artemis.set_paths(abs_exe_dir, abs_out_dir)
            artemis.set_supplied_exe(True)
        else:
            # Check if we are one level up from the executable
            local_path = os.path.join(cwd, "CMakeCache.txt")
            if os.path.exists(local_path) and os.access(local_path, os.R_OK):
                exe_path = read_cmakecache(local_path)
                out_dir = os.path.join(cwd, "testing") if out_dir is None else out_dir
                reb_path = os.path.join(os.path.dirname(exe_path), "librebound.so")

                if not (os.path.exists(exe_path) and os.access(exe_path, os.X_OK)):
                    raise TestError(f'No exe in "{exe_path}" or cannot be run!')

                if not os.path.exists(reb_path):
                    raise TestError(f'librebound.so not found at "{reb_path}"!')

                abs_out_dir = os.path.abspath(out_dir)
                abs_exe_dir = os.path.abspath(os.path.dirname(exe_path))
                artemis.set_paths(abs_exe_dir, abs_out_dir)
                artemis.set_supplied_exe(True)
            else:
                adir = os.path.join(artemis.get_artemis_dir(), "tst")
                exe_path = os.path.join(adir, "build/src/artemis")
                out_dir = os.path.join(adir, "testing") if out_dir is None else out_dir
                abs_out_dir = os.path.abspath(out_dir)
                abs_exe_dir = os.path.abspath(os.path.dirname(exe_path))
                artemis.set_paths(abs_exe_dir, abs_out_dir)
                artemis.set_supplied_exe(False)


# Execute main function
if __name__ == "__main__":
    help_msg = "names of tests or suites to run, relative to scripts/ or suites/"
    parser = argparse.ArgumentParser()
    parser.add_argument("tests", type=str, default=None, nargs="*", help=help_msg)

    parser.add_argument(
        "--cmake",
        default=[],
        action="append",
        help="architecture specific args to pass to cmake",
    )

    parser.add_argument(
        "--make_nproc", type=int, default=8, help="set nproc N for make -jN"
    )

    parser.add_argument(
        "--use_oversubscribe",
        action="store_true",
        help="use MPI oversubscribe",
    )

    parser.add_argument(
        "--log_file", type=str, default=None, help="set filename of logfile"
    )

    parser.add_argument(
        "--save_build",
        action="store_true",
        help="save build directory following regression",
    )

    parser.add_argument(
        "--reuse_build",
        action="store_true",
        help="do not recompile the code and reuse the build directory.",
    )

    parser.add_argument(
        "--exe",
        type=str,
        default=None,
        help="path to pre-built executable",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="path to output directory",
    )

    args = parser.parse_args()
    set_globals(args)
    log_init(args)

    try:
        logger.debug("args: " + str(vars(args)))
        main(**vars(args))
    except Exception:
        logger.critical("", exc_info=True)
        raise
