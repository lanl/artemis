# Artemis

Contributors: Adam Dempsey, Shengtai Li, Alex Long, Patrick Mullen, Ben Ryan, Ryan
Wollaeger

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
    power9-rhel7 (gpu)

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

There is a suite of tests in the `tst/` directory. To run the full regression suite, do

    python3 run_tests.py regression.suite

You can also pass a list of individual tests to the script, or create your own suite file. 

## CI

We use the gitlab CI for regression testing. The CI will not run if the PR is marked "Draft:" or
"WIP:". Removing these labels from the title will not automatically launch the CI. To launch the CI
with an empty commit, do

    git commit --allow-empty -m "trigger pipeline" && git push

## Release

Artemis is released under the BSD 3-Clause License. For more details see the LICENSE.md
file.
