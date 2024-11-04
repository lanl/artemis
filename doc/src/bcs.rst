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

.. _bc:

Boundary Conditions
===================

There are several types of boundary conditions available in |code|.
Some of these are inherited from ``parthenon``.

Choosing boundary conditions is done in the ``<parthenon/mesh>`` input block.
Each boundary in the problem can be set separately.

An example input block specify periodic boundaries in the ``x1``-direction and reflecting boundaries in the ``x2``-direction is:

::

  <parthenon/mesh>
  ....
  ix1_bc = periodic
  ox1_bc = periodic

  ...
  ix2_bc = reflecting
  ox2_bc = reflecting

  ...

Note that there are no ``user`` boundary conditions in |code|, as with some other Parthenon codes.
All boundary conditions are specified directly in the ``<parthenon/mesh>`` node.
Also note that some boundary conditions are only compatible with certain :ref:`pgen`.
Below is a complete listing of available boundary conditions.


periodic
--------

``periodic`` boundary conditions connect opposite ends of the domain.


reflecting
----------

``reflecting`` boundary conditions act as walls that try to keep the fluid inside the domain
by mirroring variables across the boundary, and negating the sign of the normal component of
vector quantities.

outflow
-------

``outflow`` boundary conditions attempt to let material freely leave the domain.
This is not always perfect, however, and reflections can happen.

ic
--
``ic`` boundaries keep the solution at the initial condition in the ghost zones.
Only the ``disk`` problem generator works with ``ic`` boundaries.

viscous
-------

``viscous`` boundary conditions are meant to provide continuous injection at the outer radial boundary and an outflow condition at the inner radial boundary of a disk calculation.
They arise from the solution of the 1D steady-state accretion disk equations which read,

.. math::
  \dot{M}  & = {\rm const} \\\\
  \partial_R ( F_\nu - \dot{M} \ell) & = 0

where :math:`\dot{M}` is the radial mass flux, :math:`\ell = R^2 \Omega` is the specific angular momentum, and :math:`F_\nu = 3 \pi \nu \Sigma \ell` is the viscous flux of angular momentum.
At the innner radial boundary we enforce:

.. math::
  F_\nu = \dot{M} \ell

which corresponds to keeping :math:`\nu \Sigma = {\rm const}` across the boundary (since :math:`\dot{M} = 3\pi \nu \Sigma`).
At the outer boundary, we set

.. math::
  \partial_R F_\nu = \dot{M} \partial_R \ell

by linearly expanding :math:`\ell` and :math:`F_nu`. This is, therefore, and extrapolation boundary at fixed :math:`\dot{M}`.
Once :math:`F_\nu` is set, the gas density is set, and from that and :math:`\dot{M}`, the radial velocity is set.

The value of the injection :math:`\dot{M}` is set in the ``<problem>`` block:

::

  <problem>
  ...
  mdot0 = 1e-6

Note that the derivation above was for 1D.
2D cylindrical simulations with planets have been shown to behave well with ``viscous`` boundary conditions.
3D simulations have been largely untested, however.

The ``viscous`` boundary condition is only allowed in the  ``disk`` problem generator, and only when viscosity is active.


extrapolate
-----------

``extrapolate`` boundaries are meant to provide an outflow boundary condition in situations where the fluid variables have non-trivial slopes.
Examples include:

 * maintaining vertical hydrostatic balance in the ``strat`` problem generator.
 * maintaining the background shear profile at the :math:`x` boundaries in the ``strat`` problem generator.
 * maintaining radial profiles of density, temperature, and velocity in the ``disk`` problem generator.

The ``extrapolate`` boundary condition is only allowed in the ``strat`` and ``disk`` problem generators.

conducting
----------

``conducting`` boundaries are meant to provide a constant thermal heat flux at one boundary, while fixing the temperature at the opposite boundary.
This allows the domain to relax to a steady-state, hydrostatic, solution with constant heat flux.
This is mainly used for internal testing.
When ``conducting`` boundaries are active, the ``flux`` parameter should be set under the ``<problem>`` block.
The ``conducting`` boundary condition is only allowed in the ``conduction`` problem generator.


inflow
------

``inflow`` boundaries are needed for the shearing-box (See :ref:`pgen`) in the ``y``-direction.
They are actually a combination of inflow and outflow depending on what side of :math:`x=0` the boundary is on.
On the inflow side, :math:`v_y` is set to :math:`- q \Omega x`, and the other fluid variables are set to their initial values.
On the outflow side, outflow conditions are set.
At the lower :math:`y`-boundary, inflow occurs for :math:`x>0`, whereas for the upper :math:`y`-boundary, inflow occurs for :math:`x<0`.
This boundary condition is only compatible with the ``strat`` problem generator.





