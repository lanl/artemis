# Artemis

Contributors: Adam Dempsey, Shengtai Li, Alex Long, Patrick Mullen, Ben Ryan, Ryan
Wollaeger

Software release number O4811

# Documentation

The latest documentation is hosted [here](https://lanl.github.io/artemis/index.html)

To build the documentation locally,

    cd artemis/doc/
    make html

If you encounter errors you may need to install additional python packages:

    pip install sphinx pyyaml sphinx_rtd_theme

Once the documentation is built, if you have built the docs on a remote server (e.g. darwin), to view
the documentation in your browser on your local machine, from a terminal on your local machine:

    ssh -L 8080:localhost:8080 brryan@darwin-fe.lanl.gov
    cd /path/to/artemis/doc/_build/html
    python3 -m http.server 8080

Then again on your local machine navigate your browser to

    http://localhost:8080

and the documentation should display.

# Required dependencies

* CMake 3.13 or greater
* C++ compiler (C++17 support)
* MPI
* OpenMP
* HDF5
* Parthenon
* singularity-eos

# Environment

On deployed platforms, the environment with required dependencies can be set up via

    source env/bash

Currently supported computers/partitions are:

## Darwin

    skylake-gold
    volta-x86 (gpu)

## Chicoma

    cpu
    gpu

## Venado

    gg (cpu)
    gh (gpu)

# Installation

    git submodule update --init --recursive
    mkdir build
    cd build
    cmake ../
    make -j install

## Submodules

The most reliable way to update submodule versions is to simply remove the `external`
folder:

    rm -rf external/

and re-initialize the submodules

    git submodule update --init --recursive

## Formatting the software

Any contributions to the Artemis software must be compliant with the C++ clang formatter and
the Python black formatter.  Linting contributions can be done via an automated formatting
script:
`CFM=clang-format-12 ./style/format.sh`

## Testing

There is a suite of tests in the `tst/` directory. Tests are run with the included `run_tests.py`
script. This script can be run in three ways:

1. With default arguments, where the current version of the source will be built. The resulting
executable can be saved for reuse with `--save_build`, and if saved can be reused in subsequent test
runs with `--reuse_build`. Note that `--save_build` must continue to be supplied as well to avoid
the reused build being deleted after the tests are run.
2. If the `run_tests.py` script is called from a directory with a valid `artemis` executable, that
executable will be used for testing and will not be cleaned up afterwards.
3. If the path to an `artemis` executable is provided to the `--exe` option of `run_tests.py`, that
executable will be used for testing and will not be cleaned up afterwards.

In all cases, the tests will be run from a `tst` directory created in the same folder as the
executable being used. Figures will be created in `artemis/tst/figs` and the log file in
`artemis/tst`.

To run the full regression suite, do

    python3 run_tests.py regression.suite

You can also pass a list of individual tests to the script, or create your own suite file.

## CI

We use the github CI for regression testing. The CI will not run if the PR is marked "Draft:" or
"WIP:". Removing these labels from the title will not automatically launch the CI. To launch the CI
with an empty commit, do

    git commit --allow-empty -m "trigger pipeline" && git push

A portion of the CI is run on LANL's internal Darwin platform. To launch this CI job, someone with
Darwin access (usually a LANL employee) must first create a Github Personal Access Token and store
it securely in their own environment as `ARTEMIS_GITHUB_TOKEN`, e.g. in their `~/.bashrc`:

    export ARTEMIS_GITHUB_TOKEN=[token]

and then log in to Darwin and manually launch the CI runner:

    cd artemis
    ./tst/launcher_ci_runner.py [Number of the github PR]

## Release

Artemis is released under the BSD 3-Clause License. For more details see the LICENSE.md
file.
