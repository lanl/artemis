.. =======================================================================================
.. (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
..
.. This program was produced under U.S. Government contract 89233218CNA000001 for Los
.. Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
.. for the U.S. Department of Energy/National Nuclear Security Administration. All rights
.. in the program are reserved by Triad National Security, LLC, and the U.S. Department
.. of Energy/National Nuclear Security Administration. The Government is granted for
.. itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
.. license in this material to reproduce, prepare derivative works, distribute copies to
.. the public, perform publicly and display publicly, and to permit others to do so.
.. =======================================================================================

.. _running:

Running |code|
==============


Create the input file
^^^^^^^^^^^^^^^^^^^^^

All of the configuration options for an |code| simulation are contained in an input file called ``input.par`` for this example.
The input file contains all the necessary instructions for creating and executing the simulation.
The style of the input file follows that of Parthenon input files.
Input "blocks" (or "nodes") are denoted with ``<...>``.
Inside the blocks are ``key = value`` pairs that specify each parameter under that particular block.
The values can be integers, floats, strings, or even comma separated vectors.

At the very top of the input file, there is an ``<artemis>`` block that controls some global parameters such as the coordinate system and problem definition.

::

  <artemis>
  coordinates = cylindrical   # cartesian, cylindrical, spherical, axisymmetric
  problem = disk               # name of the problem generator

Next, the ``<parthenon/job>`` block specifies the base name of the simulation outputs.

::

  <parthenon/job>
  problem_id = name       # name of output files

The ``<parthenon/time>`` block controls the evolution of the simulation, including safeguards against crashing.

::

  <parthenon/time>
  nlim = -1           # Maximum number of cycles; -1 indicates unlimited cycles
  tlim = 62.8         # end time
  integrator = rk2    # rk1, vl2, rk2, rk3
  dt_min = 1e-10      # halt execution if dt < dt_min
  dt_max = 1.0        # halt execution if dt > dt_max

The parameters ``dt_min`` and ``dt_max`` will halt the simulation if the time step falls outside their range.

Next, the output blocks control when and what |code| writes to disk as the simulation is running.
Typically, there are three types of outputs specified here, history, hdf5 snapshots, and hdf5 restart files.

::

  <parthenon/output1>
  variables = gas.prim.density,  &
              gas.prim.velocity, &
              gas.prim.pressure
  file_type = hdf5  # HDF5 data dump
  dt        = 1.0   # time increment between outputs

  <parthenon/output2>
  file_type = rst
  dt = 6.28        # time between restarts

  <parthenon/output3>
  file_type = hst
  dt = 0.2         # time between history dumps


The next blocks define the simulation mesh dimensions, boundary conditions, and meshblock size.
This example sets up a 2D cylindrical mesh that spans the full :math:`2 \pi` in azimuth.
::

  <parthenon/mesh>
  nghost = 2
  refinement = static          # none, static, adaptive
  numlevel = 3                 # the maximum number of refinement levels

  nx1    = 256                 # Number of zones in X1-direction
  x1min  = 0.3                 # minimum value of X1
  x1max  = 3.0                 # maximum value of X1
  ix1_bc = ic                  # Inner-X1 boundary condition flag
  ox1_bc = ic                  # Outer-X1 boundary condition flag

  nx2    = 1024                # Number of zones in X2-direction
  x2min  = -3.141592653589793  # minimum value of X2
  x2max  = 3.141592653589793   # maximum value of X2
  ix2_bc = periodic            # Inner-X2 boundary condition flag
  ox2_bc = periodic            # Outer-X2 boundary condition flag

  nx3    = 1                   # Number of zones in X3-direction
  x3min  = -0.5                # minimum value of X3
  x3max  = 0.5                 # maximum value of X3
  ix3_bc = periodic            # Inner-X3 boundary condition flag
  ox3_bc = periodic            # Outer-X3 boundary condition flag

  <parthenon/meshblock>
  nx1 = 32
  nx2 = 32
  nx3 = 1

:ref:`bc` describes all the possible boundary conditions.

The ``refinement`` parameter in the ``<parthenon/mesh>`` block  controls how mesh refinement is handled.
If ``refinement = none``, no refinement occurs.
Setting ``refinement = adaptive`` activates adaptive mesh refinement.
The physics packages can control when and where to refine and de-refine the mesh.
See :ref:`physics` for a description of each package.
Lastly, if `refinement = static`, the input file controls where the mesh is refined.
There are additional input blocks specify where this refinement is located.
For example,

::

  <parthenon/static_refinement1>
  level = 3
  x1min = 0.9
  x1max = 1.1
  x2min = -0.5
  x2max = 0.5
  x3min = -1.0
  x3max = 1.0

Up until now, the input file has been mainly specifying parameters for Parthenon.
The following blocks deal with |code| specific parameters.
First, all of the active physics packages are listed.
The following example activates the ``gas``, ``gravity``, ``viscosity``, and ``rotating_frame`` packages.

::

  <physics>
  gas = true
  gravity = true
  viscosity = true
  rotating_frame = true

Each active physics package will have an associated input block specifying its parameters.
Following the example, simple ``<gas>``, ``<gas/viscosity>``, ``<gravity>``, and ``<rotating_frame>`` blocks are added.
For more details see the :ref:`physics` and :ref:`parameters` sections

::

  <gas>
  cfl = 0.3
  eos = ideal
  gamma = 1.4
  riemann = hllc
  reconstruct = plm

  <gas/viscosity>
  type = alpha
  alpha = 1e-3

  <gravity>
  type = binary
  gm = 1.0
  q = 1e-3
  a = 1.0
  sft2 = .06

  <rotating_frame>
  omega = 1.0


Finally, there is a ``<problem>`` block that contains the parameters controlling the problem initial conditions.
See :ref:`pgen` for a list of the possible ``<problem>`` blocks,

::

  <problem>
  h0 = .05
  rho0 = 1.0
  dslope = -0.5
  flare = 0.0



Run |code|
^^^^^^^^^^

|code| is an MPI + GPU executable.
The exact command to launch it depends on the system it is run on.
This example will assume a SLURM-like cluster.

To launch a fresh |code| on ``$NPROCS`` CPUs with ``srun``,

::

  srun -n $NPROCS artemis -i input.par

To restart a previous run, use the ``-r`` argument

::

  srun -n $NPROCS artemis -r name.final.rst

A modified input file can optionally still be passed with the ``-i`` argument.

Return codes
^^^^^^^^^^^^

When using batch submissions, it is possible to set up a self-restarting job.
The easiest way to do this is to take advantage of SLURM interrupt signals and the |code| return code.
|code|

An example batch submission script, ``run.sh``, would look like:

::

  #!/bin/bash
  #SBATCH -J name
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=128
  #SBATCH -t 16:00:00

  set -o pipefail

  if [ ! -f name.final.rst ]; then
    echo "Starting fresh"
    srun -n $SLURM_NPROCS artemis -i input.par -t 15:50:00
  else
    echo "Restarting"
    srun -n $SLURM_NPROCS artemis -r name.final.rst -t 15:50:00
  fi

  EXITCODE=$?

  set +o pipefail

  if [[ $EXITCODE -eq 2 ]]; then
   echo "Resubmitting"
   sbatch run.sh
  fi

This stops |code| 10 minutes before the job ends.
If the simulation has completed by then, |code| will return ``0``.
Instead if it hasn't reached its end time yet, it will return ``2``.
And if the simulation crashed for some reason, it will return ``1``.
If the return code is ``2``, the batch script will resubmit itself.


