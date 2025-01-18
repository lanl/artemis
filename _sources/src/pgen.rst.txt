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

.. _pgen:

Problem Generators
==================

Problem generators in |code| control the initial and (to some extent) boundary conditions of a simulation.
The choice of problem generator is controlled by the ``problem`` parameter in the ``<artemis>`` input block



Below is a complete listing of all possible values of ``problem``.

disk
----

The ``disk`` problem generator sets up an axisymmetric sub-Keplerian disk in pressure-rotational equilibrium.
To use the ``disk`` problem generator set:

::

  <artemis>
  ...
  problem = disk


Both vertically isothermal and polytropic disks are supported.
Corrections for inner cavities, outer exponential cutoffs, and the presence of an inner binary, are also available.

The basic description of the disk starts with the aspect ratio profile:

.. math::
  \frac{H}{R} = h_0  \left(\frac{R}{R_0} \right)^{f}

where :math:`H` is the disk scale height and :math:`f` is the flaring index.
The following parameters control the disk scale height profile:

::

  <problem>
  ...
  r0 = 1.0      # reference radius
  h0 = .05      # aspect ratio at r0
  flare = 0.25  # flaring index

Once we specify the disk aspect ratio, we define the mid-plane density profile:

.. math::
  \rho_{\rm mid}(R) = \rho_0 \left(\frac{R}{R_0} \right)^{\mu} g_{\rm cav}(R) g_{\rm exp}(R)

This includes a power-law component and aforementioned corrections through the :math:`g(R)` functions.
The parameters controlling the mid-plane density are:

::

  <problem>
  ...
  rho0 = 1.0     # reference density
  dslope = -1.0  # density power-law
  rexp  = 5.0    # exponential cutoff
  rcav  = 0.5    # cavity radius
  l0    = 0.0    # optional binary accretion eigenvalue


The mid-plane density is then extended vertically depending on what polytropic index is defined.
The polytropic index, :math:`\Gamma` sets the temperature profile:

.. math::
  T(R,z) = \frac{H^2 \Omega^2}{\Gamma} \left( \frac{\rho}{\rho_0} \right)^{\Gamma - 1}

which for :math:`\Gamma = 1` corresponds to a vertically isothermal profile, and for any other :math:`\Gamma`, corresponds to a pressure profile that is polytropic, i.e., :math:`P \propto \rho^\Gamma`.
The polytropic index is controlled by the parameter

::

  <problem>
  ...
  polytropic_index = 1.0

With :math:`\Gamma` defined, the vertical distribution of the density profile follows:

.. math::
  \rho(R,z) & = \rho_{\rm mid}(R)  \exp\left[ - \frac{R^2}{H^2} \left(1 - \frac{R}{\sqrt{R^2 + z^2}} \right) \right]   \qquad {\rm if} \qquad \Gamma = 1 \\\\
            & = \rho_{\rm mid}(R)  \left[ 1 - (\Gamma-1) \frac{R^2}{H^2} \left(1 - \frac{R}{\sqrt{R^2 + z^2}} \right) \right]^{1/(\Gamma-1)} \qquad {\rm if} \qquad \Gamma \neq 1

Lastly, we directly compute the azimuthal disk velocity, :math:`v_\phi` by enforcing centrifugal balance,

.. math::
  \frac{v_\phi^2}{R} = \frac{v_K^2}{R} + \frac{1}{\rho} \frac{\partial P}{\partial R}


Disks with viscosity have additional options.
In particular,

::

  <problem>
  ...
  mdot0 = 1e-6          # initial mdot
  quiet_start = false   # set a non-zero vR

An initial constant :math:`\dot{M}` is enforced by using :math:`v_R = -3/2 \nu/R` and internally setting the reference density :math:`\rho_0 = \dot{M}/(3\pi\nu_0)`.
Therefore, if ``mdot0`` is set, then ``rho0`` will be calculated from it.
To start the disk with no radial velocity, set ``quiet_start = true``.

The ``disk`` problem allows the use of the ``viscous``, ``ic``, and ``extrapolate`` boundary conditions. See :ref:`bc` for a description.

``disk`` works in all coordinate systems by converting between the problem geometry and axisymmetric geometry.


strat
-----

The ``strat`` problem generator sets up a stratified shearing-box.
To use this the following nodes must be active,

::

  <artemis>
  coordinates = cartesian
  problem = strat

  <physics>
  rotating_frame = true

  <rotating_frame>
  omega = 1.0
  q = 1.5

Note that this requires cartesian coordinates and the rotating frame.

The initial profile is isothermal, :math:`T = H_0^2 \Omega_0^2`, with a gaussian density profile:

.. math::
  \rho(z) = \rho_0 \exp\left( -\frac{z^2}{2 H_0^2} \right)

where :math:`H_0` is the initial pressure scale height.
The relevant parameters that can be set under ``<problem>`` are:

::

  <problem>
  h = 1.0      # scale height
  rho0 = 1.0   # density normalization

The velocities are initially set to the steady-state background shear solution of :math:`v_y = - q \Omega_0 x`.

The ``strat`` problem requires the ``inflow`` boundaries for the :math:`y`-direction and allows for ``extrapolate`` boundaries for the :math:`x` and :math:`z` directions boundary conditions.
See  :ref:`bc` for a description of those.

Internal generators
-------------------

The remaining problem generators are mainly used for testing. Descriptions of their ``<problem>`` block parameters can be found in :ref:`parameters`.


