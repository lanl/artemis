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

.. _physics:

Physics Modules
===============


In general, to activate a physics capability, it must be set in the ``<physics>`` input block:

::

   <physics>
   gas = true
   dust = false
   viscosity = false
   conduction = false
   cooling = false
   gravity = false
   nbody = false
   rotating_frame = false
   radiation = false



Below is a complete listing of the physics capabilities available in |code|.


Fluid Models
------------

Gas
^^^
The main fluid type available in |code| is called "gas" and is controlled by the ``<gas>`` input block.
Gasses track the evolution of density, momentum, and energy.
Currently, they are restricted to an ideal gas equation of state.
|code| supports multiple gas species, controlled by the ``nspecies`` parameter in the ``<gas>`` block.
When setting up a ``<gas>`` input block, you need to specify the EOS, Riemann Solver, and reconstruction method

An example input block for a gas could read

::

  <gas>
  cfl = 0.3
  eos = ideal
  gamma = 1.4
  riemann = hllc     # llf, hlle, hllc Riemann solvers
  reconstruct = plm  # pcm, plm, ppm reconstructions

In order to safely model kinetic energy dominated flows, |code| uses a dual energy formalism controlled by the ``de_switch`` parameter under ``<gas>``.
When the fraction of the internal energy to the total energy in a cell is less than ``de_switch``, |code| will recover the internal energy from a separately evolved internal energy variable.
This is not conservative, but greatly enhances accuracy and stability in high Mach number flows.


For simulations where AMR is active, gasses can trigger refinement based on either the magnitudes or gradients of their densities or pressures.
These are controlled by the ``refine_field``, ``refine_type``, ``refine_thr``, and ``deref_thr`` parameters.
An example that triggers refinement when the gas density is larger than ``10.0`` and triggers de-refinement when the density falls below ``0.02`` is:

::

   <gas>
   ...
   refine_field = density
   refine_type = magnitude
   refine_thr = 10.0
   deref_thr = 0.02


Gasses support several microphysics models including `Cooling`_, `Viscosity`_, and `Conduction`_.
These are controlled by adding additional nodes under the ``<gas>`` node, e.g., ``<gas/viscosity>``.
We describe each of these below.

Cooling
"""""""

Adding a ``<gas/cooling>`` node and setting ``cooling = true`` under the ``<physics>`` node activates a simple heating/cooling functionality that brings the gas temperature toward a target temperature on some time scale.
This is done by adding the following source term to the internal energy equation,

.. math::
  \rho \frac{D e}{D t} = - C_v \frac{ T - T_0}{\tau_c}

where :math:`C_v` is the specific heat of the material, :math:`T_0` is a target temperature that can be spatially dependent, and :math:`\tau_c` is a cooling time that can also be spatially dependent.

|code| supports several types of :math:`T_0` profiles and cooling times.
These are controlled by the ``tref`` and ``type`` parameters, respectively.
Options for ``type`` are:

* ``type = constant``

This simply sets the cooling time to a constant value given by the parameter ``tau``. An example input block for constant cooling is

::

   <gas/cooling>
   ...
   type = constant
   tau = 1e-3

* ``type = beta``

:math:`beta`-cooling is a common cooling model in studies of accretion disks.
The cooling time is set to scale with the local rotation rate around the central object such that :math:`\beta = \tau_c \Omega_K`.
|code| allows for a vertically dependent :math:`beta` of the form:

.. math::
  \beta(z) = \beta_{\rm min} + \beta_0 \exp\left( - {\rm exp\_scale} * z^2/T_0 \right)

where :math:`T_0` is the reference temperature defined by ``tref``.
An example input with constant :math:`beta`-cooling is:

::

   <gas/cooling>
   ...
   type = beta
   beta0 = 10.0


Options for ``tref`` are:

* ``tref = powerlaw``

 This cools the gas temperature to a power-law profile in either cylindrical (:math:`R`) or spherical (:math:`r`) radius.

.. math::

   T_0(\mathbf{x}) = T_{\rm floor} + T_{\rm cyl} R^{{\rm cyl\_plaw}} +T_{\rm sph} r^{{\rm sph\_plaw}}

An example input block to cool to a cylindrical power-law profile with a floor is:

::

   <gas/cooling>
   ...
   tref = powerlaw
   tcyl = 0.0025
   cyl_plaw = -1.0
   tfloor = 1e-8


* ``tref = nbody``

The ``nbody`` option should be used in conjunction with an ``<nbody>`` node (see `N-Body Dynamics`_).
This cools the gas to a temperature profile of the form:

.. math::

   T_0(\mathbf{x}) =  T_{\rm sph} \left(-\frac{GM}{\Phi} \right)^{{\rm sph\_plaw}}

where :math:`Phi` is the total gravitational potential from all N-body particles in the problem.
Because the gravitational potential from a point-mass is :math:`\Phi = -GM/r`, we use :math:`-GM/Phi` as a proxy for :math:`r`.
This type of temperature profile is often used circumbinary disk studies.
An example input block for ``nbody`` is,

::

   <gas/cooling>
   ...
   tref = nbody
   tsph = 0.0025
   sph_plaw = -1.0




Viscosity
"""""""""

The only available stress model for gasses is viscous stress.
The viscous stress tensor, :math:`Pi` is defined as,

.. math::
   \Pi = \mu_s (\nabla \mathbf{v} + \nabla \mathbf{v}^T ) + \left( \mu_b -  \frac{2}{3} \mu_s \right) \nabla \cdot \mathbf{v}

and is controlled by a dynamic shear viscosity, :math:`\mu_s`, and dynamic bulk viscosity, :math:`\mu_b`.
These dynamic viscosities are related to the kinematic viscosities by the density, e.g., :math:`\mu_s = \rho \nu_s`.

To activate viscosity, set ``viscosity = true`` under the ``<physics>`` node and add a ``<gas/viscosity>`` node.
There are two models available for the viscosity coefficients: constant, and :math:`\alpha`-viscosity.
These are controlled by the ``type`` parameter.

* ``type = constant``

Constant viscosity sets the kinematic shear viscosity to a constant value determined by the ``nu`` parameter.
The bulk viscosity is set to be a constant multiple of the shear viscosity given by the ``eta_bulk`` parameter.
An example input block with constant non-zero shear viscosity and zero bulk viscosity is:

::

   <gas/viscosity>
   type = constant
   nu = 1e-6

* ``type = alpha``

A common viscosity model in the simulation of accretion disks is :math:`\alpha`-viscosity.
The dynamic shear viscosity in this model is set to,

.. math::
   \mu_s = \alpha \frac{B_s}{\Omega_{\rm K}} = \alpha \frac{\rho c_s^2}{\Omega_{\rm K}}

where :math:`\alpha` is constant and :math:`B_s` is the material's bulk modulus.
For a simple ideal gas with adiabatic index :math:`gamma`, the bulk modulus is equal to :math:`B_s = \gamma P`.
Note that because |code| formulates it's :math:`\alpha`-viscosity with the material's bulk modulus, there can be an implicit factor of :math:`\gamma` when comparing to other codes' values of :math:`\alpha`.

An example input block for :math:`\alpha`-viscosity with equal shear and bulk viscosities is:

::

   <gas/viscosity>
   type = alpha
   alpha = 1e-3
   eta_bulk = 1.0



Conduction
""""""""""

Thermal conduction transports thermal energy down temperature gradients with a flux, :math:`\mathbf{q} = - K \nabla T`, such that,

.. math::
   \rho \frac{D e}{D t}= \nabla \cdot \mathbf{q}

The strength of thermal conduction is set by the material's thermal conductivity :math:`K`, or, alternatively, the material's thermal diffusivity, :math:`\kappa = K/C_v`.

To add thermal conduction to a problem, there needs to be a `<gas/conductivity>` node and ``conduction = true`` under the ``<physics>`` node.

There are two types of conduction models in |code|: constant thermal conductivity or constant thermal diffusivity.
These are chosen by setting the ``type`` parameter:

 * ``type = constant``

This specifies a constant conductivity set by the ``cond`` parameter. An example input block for constant conductivity is,

::

   <gas/conductivity>
   type = constant
   cond = 1e-3

* ``type = diffusivity_constant``

This specifies a constant thermal diffusivity set by the `kappa` parameter. An example input block for constant thermal diffusivity is,

::

   <gas/conductivity>
   type = diffusivity_constant
   kappa = 1e-2



For both viscosity and conduction, |code| can average neighboring diffusion coefficients with either simple averaging where

.. math::
   D_{\rm face} = \frac{1}{2} \left( D_{\rm left} + D_{\rm right} \right)

or with harmonic averaging where

.. math::
   D_{\rm face} = 2 \frac{D_{\rm left}  D_{\rm right}}{D_{\rm left} + D_{\rm right} }

Harmonic averaging can be useful in situations where neighboring diffusion coefficients differ by orders of magnitude.

The ``averaging`` parameter can be set to ``arithmetic`` or ``harmonic`` to choose between these two models.
An example input block using harmonic averaging for viscosity is:

::

   <gas/viscosity>
   type = alpha
   alpha = 1e-3
   averaging = harmonic


Dust
^^^^

The ``<dust>`` input block adds a dust fluid to the simulation.
Dust in |code| is modeled as a pressure-less fluid. As such, it an EOS is not required.
Just as with gas fluids, dust fluids require specification of the number of species, Riemann solver, and reconstruction scheme.
An example dust input block reads:

::

   <dust>
   cfl = 0.3
   nspecies = 20
   riemann = hlle     # llf, hlle
   reconstruct = plm  # pcm, plm, ppm

By default, each dust species represents a dust fluid at a fixed particle size.


N-Body Dynamics
---------------

|code| allows the building of an arbitrarily complicated system of gravitating particles solely through the input file.
Particles interact with the fluids via gravity and accretion.
Once built, the evolution of the N-body system is handled by the REBOUND library.
|code| provides several parameters that directly control the execution of REBOUND.


All simulations using the ``nbody`` package need to set a few global parameters specifying how the N-body system evolves.
These include the maximum timestep, the type of N-body integrator, how to handle collisions of particles, and the frame of the N-body simulation in relation to the hydro system.

An example ``<nbody>`` input block would be:

::

   <nbody>
   dt =  0.01
   integrator = ias15  # ias15, none, whfast, leapfrog, janus, mercurius, saba, bs
   merge_on_collision = true
   frame = global      # global, local
   box_size = 10.0

Note that there are multiple options for integrators corresponding to some of the integrators available in REBOUND.
For almost all simulations, the 15th order ``ias15`` integrator should suffice.
While ``ias15`` is expensive compared to other integrators, it is still a small fraction of the cycle cost in |code|.
In certain situations, the ``none`` integrator can be useful.
For example, if there is only a single, stationary, particle in the domain or if the simulation is the rotating frame of a static binary.

The ``frame`` parameter is typically set to its default value of ``global``.
This means that the REBOUND simulation is in the global frame of reference.
Note that this can be a different frame than the |code| frame if the rotating frame is active.
Setting ``frame=local`` means that |code| artemis will not apply corrections to the REBOUND frame, but caution should be used with this option.

|code| can also make full use of the collision functionality in REBOUND.
Particles can be given radii defining their collisional cross-section.
When particles encounter other particles, REBOUND calls an |code| function.
In that function, we calculate the binding energy of the two particles, and if they are bound, and if ``merge_on_collision = true``, |code| combines the particles into one.
Alternatively, If ``merge_on_collision = false``, the particles pass through each other.



Building systems
^^^^^^^^^^^^^^^^

Building an N-Body system is an object-orientated system centered around the creation of particles.
Particles can be created one at a time by adding multiple ``<particle>`` blocks,

::

   <particle1>
   ...
   <particle2>
   ...
   etc.

, or multiple particles with similar parameters can be added at the same time using the aggregate blocks described in the subsequent sections.

Particles are characterized by their mass, size, initial position, initial velocity, and subgrid models.
Below we describe how to add particles individually or in collections.

Particles
^^^^^^^^^

The main parameters specifying a particle are:

::

   <particle1>
   mass = 1.0     # mass of the particle
   radius = 0.1   # physical size of the particle
   couple = 1     # Couple this particle to the fluid

The physical size of the particle is its ``radius`` (see the section on collisions below).
To turn add particle-fluid force to the fluids, the ``couple`` parameter must be set to ``1``.
To de-couple a particle from the fluid, set ``couple = 0``.

When particles are embedded in fluid, gravitational softening and mass removal are essential to control the dynamics very near the particles.
The ``<particle/soft>`` and ``<particle/sink>`` nodes control these two subgrid models for each particle.
One example is:

::

   <particle1>
   ....

   <particle1/soft>
   radius = 0.1
   type = spline    # none, spline, plummer

   <particle1/sink>
   radius = 0.1
   gamma = 30.0    # mass removal rate
   beta  = 0.0     # = torque-free mass removal

There are three softening models available: ``none`` does nothing, ``spline`` is the spline softening used in Gadget (Springel 2001), and ``plummer`` modifies :math:`r^2 \rightarrow r^2 + r_s^2`.
Spline softening is exactly Kepelerian outside the softening radius, whereas Plummer softening asymptoically approaches Keplerian outside the softening radius.
Roughly, the Plummer softening radius is :math:`\sim 2.8 \times` the spline softening radius.

Mass accretion is handled by a particle's ``<../sink>`` attribute.
Two types of removal rates are avaible.
The first, ``gamma``, sets the rate of mass removal for cells inside the particle's sink radius.
The second, ``beta``, controls the amount of momentum removed from the fluid.
Typically ``beta`` will either be zero -- corresponding to angular-momentum conserving mass removal -- or equal to ``gamma`` -- corresponding to isotropic momentum removal.


When adding particles one-by-one, the initial position and velocity can be (optionally) specified with an ``particle/initialize`` block.
For example, an input block that adds a stationary particle at the origin would be:
::

   <particle1/initialize>
   x = 0.0
   y = 0.0
   z = 0.0
   vx = 0.0
   vy = 0.0
   vz = 0.0

The momentum-conserving back-reaction fluid-particle force can be added to the particle equation of motion by setting the ``live`` parameter under each particle block.
There is an additional option to scale the total fluid force by a fixed amount determined by the ``mscale`` parameter.

An example input block that scales the fluid-particle force by ``1e-4`` and turns the particle "live" after a time of ``0.2`` is:
::

   <nbody>
   ...
   mscale = 1e-4

   <particle1>
   ....
   live = 1
   live_after = 0.2


When using AMR in |code|, additional refinement criteria related to the particles are available.
In particular, each active particle can refine the mesh within a specified distance to the maximum refinement level and de-refine the mesh outside another specified distance.
To activate this feature, the ``refine_type`` parameter should be set to ``distance``.
An example input block enrolling the particles in the AMR refinement criteria is:

::

   <nbody>
   ...
   refine_type = distance
   derefine_factor = 2.0
   <particle1>
   refine_distance = 0.6

One interesting feature that |code| allows is "Lagrangian" refinement regions by adding zero mass particles (i.e., test particles) with specified trajectories and refinement criteria.
These do not interact gravitationally with the fluid or other massive particles, but instead provide unique refinement regions.


Binaries
^^^^^^^^

There are two methods to add bound binaries.
An example that ties two previously defined particles together into a binary:

::

   <particle1>
   ...
   <particle2>
   ...
   <binary1>
   primary = 1
   secondary = 2
   a = 1.0             # Binary semi-major axis
   e = 0.0             # Binary eccentricity
   i = 0.0             # Binary inclination (in degrees)
   omega = 0.0         # Binary argument of pericenter (in degrees)
   Omega = 0.0         # Binary longitude of ascending node (in degrees)
   f = 0.0             # Binary true anomaly (in degrees)
   x = 0.0             # Center of mass x position
   y = 0.0             # Center of mass y position
   z = 0.0             # Center of mass z position
   vx = 0.0            # Center of mass x velocity
   vy = 0.0            # Center of mass y velocity
   vz = 0.0            # Center of mass z velocity

Here, we explicitly tell the ``<binary1>`` node which particles are the primary and secondary.
Note that all angles are in degrees.
This is to ensure that the :math:`sin` and :math:`cos` functions are exact.

Another example that creates a binary, including its particles, at once is:

::

   <binary2>
   couple = 1
   mass = 1.0          # Total binary mass
   q = 1.0             # Binary mass ratio
   a = 1.0             # Binary semi-major axis
   e = 0.0             # Binary eccentricity
   i = 0.0             # Binary inclination (in degrees)
   omega = 0.0         # Binary argument of pericenter (in degrees)
   Omega = 0.0         # Binary longitude of ascending node (in degrees)
   f = 0.0             # Binary true anomaly (in degrees)
   radius = 0.01
   rsoft = 0.01
   rsink = 0.01
   gamma = 30.0
   beta = 0.0
   stype = spline
   live = 0
   live_after = 0

This creates two particles with the same properties for softening, accretion, coupling, etc.

Triples
^^^^^^^

|code| supports the direct creation of hierarchical triples in a manner very similar to creating binaries.
Triples can be created by tying three particles together, or from scratch.
An example input block creating a triple from existing particles is,

::

   <particle1>
   ...
   <particle2>
   ...
   <particle3>
   ...
   <triple1>
   primary = 1
   secondary = 2
   tertiary = 3
   ao = 1.0            # Outer binary semi-major axis
   eo = 0.0            # Outer binary eccentricity
   ...                 # Additional outer binary orbital parameters
   a = 0.05            # Inner binary semi-major axis
   e = 0.0             # Inner binary eccentricity
   i = 0.0             # Inner binary inclination (in degrees)
   omega = 0.0         # Inner binary argument of pericenter (in degrees)
   Omega = 0.0         # Inner binary longitude of ascending node (in degrees)
   f = 0.0             # Inner binary true anomaly (in degrees)
   x = 0.0             # Center of mass x position
   y = 0.0             # Center of mass y position
   z = 0.0             # Center of mass z position
   vx = 0.0            # Center of mass x velocity
   vy = 0.0            # Center of mass y velocity
   vz = 0.0            # Center of mass z velocity

Note that there is full specification of the inner orbit (secondary + tertiary) and outer orbit (primary + (secondary + tertiary)).
Creating a triple system from scratch follows the ``<binary2>`` example above.


N-Body systems
^^^^^^^^^^^^^^

When the number of particles becomes large, specifying each particle by hand can become tedious.
To alleviate this, |code| can read an additional input file that lists the positions, velocities, and model parameters of a system of particles.

::

   <system1>
   couple = 1
   live = 1
   live_after = 0.0
   stype = spline    # none, plummer, spline
   input_file = sys.txt

With the corresponding input file ``sys.txt``:

::

  # sys.txt
  # m x y z vx vy vz rs gamma beta target radius
  1e-3	1.0	0.0	0	0	0	0	6.43e-4	1.0	0	4.8e-2	6.4e-3
  1e-5	0.0	1.0	0	0	0	0	6.77e-4	1.0	0	5.0e-2	6.7e-3
  2e-2   0.0	0.0	0	0	0	0	7.11e-4	1.0	0	5.3e-2	7.1e-3
  1e-4	1.0	1.0	0	0	0	0	7.45e-4	1.0	0	5.5e-2	7.4e-3


Planetary systems
^^^^^^^^^^^^^^^^^

Similarly to the ``<system>`` block, a planetary system can be easily added with a ``<planet>`` block.
Here, instead of specifying positions and velocities, the ``<planet>`` block specifies the orbital parameters of a collection of planets.
An example input block and configuration file to set up a system of 4 Jupiters is

::

   <planet1>
   couple = 1
   live = 1
   live_after = 0.0
   stype = spline    # none, plummer, spline
   input_file = psys.txt

With the corresponding input file ``psys.txt``:

::

  # psys.txt
  # q a e i f o O rs gamma beta target radius
  1e-3	1.0	0	0	0	0	0	6.43e-4	1.0	0	4.8e-2	6.4e-3
  1e-3	2.0	0	0	0	0	0	6.77e-4	1.0	0	5.0e-2	6.7e-3
  1e-3	3.0	0	0	0	0	0	7.11e-4	1.0	0	5.3e-2	7.1e-3
  1e-3	4.0	0	0	0	0	0	7.45e-4	1.0	0	5.5e-2	7.4e-3

All planets share the options set in the ``<planet>`` block.

Note that the central object has been left unspecified.
An additional particle can be added at the center of the system to represent the star, or a ``<binary>`` block could be added to model a circumbinary planetary sytem, or even a ``<system>`` block for a circum-cluster planetary system.




External Gravity
----------------

The ``<gravity>`` node adds external gravitational forces to every fluid.
Specifically, |code|, adds the body force,

.. math::
   \rho \frac{D \mathbf{v}}{D t} = \rho \mathbf{g}

To activate external gravity, there must be a ``<gravity>`` node and ``gravity = true`` under the ``<physics>`` node.
The ``<gravity>`` requires setting the parameter ``gm``, typically this is set to ``gm = 1.0``.

The specific model for the gravitational acceleration, :math:`\mathbf{g}`, is controlled by the ``type`` parameter.
Available options are:

* ``type = constant``

  This specifies a constant gravitational acceleration. The components of :math:`\mathbf{g}` in each direction are required.
  An example input block that sets :math:`\mathbf{g}=-1.0 \mathbf{\hat{z}}` is,

  ::

   <gravity>
   type = constant
   gm = 1.0
   gx1 = 0.0
   gx2 = 0.0
   gx3 = -1.0

* ``type = point``

  This adds the gravitational acceleration, :math:`\mathbf{g} = - GM /r^2`, from a point mass.
  The potential can be (optionally) softened by adding a fixed number to the distance of a cell to the point mass.
  Additionally, the point mass can remove mass from cells within a specified distance at a specified rate.
  An example input block with all available parameters for a point mass are:

  ::

   <gravity>
   type = point
   gm = 1.0
   x = 0.0
   y = 0.0
   z = 0.0
   soft = 1e-3        # Softening radius
   sink = 1e-3        # Mass accretion radius
   sink_rate = 30.0   # Mass removal rate

* ``type = binary``

  This adds two point masses in a fixed binary orbit.
  In addition to the softening and accretion prescriptions described above, there are also parameters that describe the binary orbit.
  An example input block that adds a Sun-Jupiter binary is:

  ::

   <gravity>
   type = binary
   gm = 1.0            # Binary total GM
   x = 0.0             # Binary x center of mass
   y = 0.0             # Binary y center of mass
   z = 0.0             # Binary z center of mass
   soft1 = 0.0         # Softening radius of the primary
   soft2 = 0.0         # Softening radius of the secondary
   sink1 = 0.0         # Mass accretion radius of the primary
   sink_rate1 = 0.0    # Mass removal rate of the primary
   sink2 = 0.01        # Mass accretion radius of the secondary
   sink_rate2 = 30.0   # Mass removal rate of the secondary
   q = 1-3             # Binary mass ratio
   a = 1.0             # Binary semi-major axis
   e = 0.0             # Binary eccentricity
   i = 0.0             # Binary inclination (in degrees)
   omega = 0.0         # Binary argument of pericenter (in degrees)
   Omega = 0.0         # Binary longitude of ascending node (in degrees)
   f = 0.0             # Binary true anomaly (in degrees)


* ``type = nbody``

  This indicates that the gravitational force will be calculated by the N-body system defined in the ``<nbody>`` input block.
  The only parameter required when ``type = nbody`` is the ``gm`` parameter.
  Note that the total mass of the system defined in the ``<nbody>`` block will be rescaled to ``gm``.
  An example input block would thus read:

  ::

   <gravity>
   type = nbody
   gm = 1.0

See `N-Body Dynamics`_ for a description of how to set up the N-body system.

Rotating Frame
--------------

|code| is capable of evolving the fluid in a non-inertial, rotating frame.
This adds the Coriolis and centrifugal forces,

.. math::
  \rho \frac{D \mathbf{v}}{D t} = - 2 \rho \mathbf{\Omega} \times \mathbf{v} - \rho \mathbf{\Omega} \times \mathbf{\Omega} \times \mathbf{r}

and the corresponding centrifugal work term to the total energy equation.

In order to use the rotating frame capability, there needs to be a ``<rotating_frame>`` input block and  ``rotating_frame = true`` under the ``<physics>`` block.

For simplicity, |code| assumes that the rotation vector, :math:`\mathbf{\Omega}`, is constant in time and aligned with the :math:`z`-axis.
Moreover, the ``<rotating_frame>`` block takes on different meanings in different coordinate systems.

In axisymmetric, cylindrical, or spherical geometries, :math:`\mathbf{\Omega} = \Omega_0 \mathbf{\hat{z}}`.
There is only parameter under ``<rotating_frame>``, ``omega``, that sets the rotation rate :math:`\Omega_0`.
An example input block that sets the rotation rate to unity is:

::

   <rotating_frame>
   omega = 1.0

In Cartesian problems, ``<rotating_frame>`` activates the "shearing-box" approximation.
The shearing-box is a local approximation of an accretion disk at some fixed radius.
The central potential is approximated as a constant value plus a linear shear:

.. math::
  \rho \frac{D \mathbf{v}}{D t} = - 2 \Omega_0 \rho \mathbf{\hat{z}} \times \mathbf{v} + \rho \Omega_0^2 ( 2 q x \mathbf{\hat{x}} - z \mathbf{\hat{z}} )

where :math:`q = -d \ln \Omega / d \ln r` is the shear rate.
An example input block for a Keplerian shearing-box is

::

   <rotating_frame>
   omega = 1.0
   q  = 1.5

Drag
----

|code| supports limited fluid-fluid interactions.
The simplest such interaction is drag -- specifically, drag between the gas and dust fluids.
To enable drag, ``drag`` needs to be set to ``true`` under the ``<physics>`` node.
In the case of a single gas species interacting with many dust species, |code| adds the following forces to the fluid momenta equations,

.. math::
   \frac{D \mathbf{v}_g}{Dt} = - \sum_j \alpha_j (\mathbf{v}_g - \mathbf{v}_j ) - \beta_g (\mathbf{v}_g - \mathbf{u}_g) \\\\
   \frac{D \mathbf{v}_j}{Dt} =  \alpha_j (\mathbf{v}_g - \mathbf{v}_j ) - \beta_j (\mathbf{v}_j - \mathbf{u}_j)

where :math:`\alpha_j` are the coupling rates between dust species :math:`j` and the gas.
|code| supports constant values of the coupling rates and a Stokes model where,

.. math::
   \alpha_j = \frac{\rho_g v_{th}}{\rho_{grain} a_{grain}}

where :math:`\rho_{grain}` is the grain density (not the fluid density), :math:`a_{grain}` is the grain size, and :math:`v_{th}` is the thermal speed in the gas.
In addition to the dust-gas drag, |code| also allows for self-drag that damps each fluid velocity to a target value (:math:`\mathbf{u}_j`) at the rate :math:`\beta_j`.

An example calculation that includes both a Stokes drag model and self-drag for both the gas and dust is:

::

  <gas>
   ....
  <dust>
  ....
  sizes = 1e-6, 1e-4, 1e-2  # grain sizes
  grain_density = 1.4       # grain density

  <dust/stopping_time>
  type =  stokes

  <gas/damping>
  inner_x1 = 0.45
  outer_x2 = 2.8
  inner_x1_rate = 30.0
  inner_x2_rate = 30.0

  <dust/damping>
  inner_x1 = 0.45
  outer_x2 = 2.8
  inner_x1_rate = 30.0
  inner_x2_rate = 30.0

  <drag>
  type = simple_dust

Note that the grain sizes and density must be in code units.

Radiation
---------

|code| supports gray photon transport via coupling to the |jaybenne| package.  The gray
representation of the radiation transport equation is

.. math::
    \partial_t I + c \mathbf{n} \cdot \nabla I = c \left(j - \alpha I \right),

where :math:`I` is the frequency integrated specific intensity, :math:`j` is the
frequency integrated emissivity, and :math:`\alpha` is the frequency integrated
absorptivity.  |jaybenne| evolves the transport equation via history-based, implicit Monte
Carlo.  Currently, |jaybenne| only supports emissivity/absorptivity decompositions into an
isotropic, elastic scattering component and a thermal emission/absorption component
(following the gray form of Kirchoff's law, hence assuming LTE).

The resolution element in |jaybenne| is the number of particles used to sample photon
phase space. Particles are handled via Parthenon's swarm API.  By default, Parthenon's
swarm API permits periodic and outflow boundary conditions on particles.  Additionally,
|jaybenne| supplies reflecting boundary conditions.  Boundary conditions are enrolled as
follows:

::

  <parthenon/swarm>
  ix1_bc = jaybenne_reflecting
  ox1_bc = jaybenne_reflecting
  ix2_bc = periodic
  ox2_bc = periodic
  ix3_bc = periodic
  ox3_bc = periodic

An |code| input file interfaces with |jaybenne| via a ``<jaybenne>`` node, e.g.,

::

  <jaybenne>
  dt = 1.0
  num_particles = 10000
  do_emission = true
  do_feedback = true

where ``dt`` defines a constant timestep by which radiation limits the global timestep (if
desired), ``num_particles`` sets the resolution element for IMC, ``do_emission`` enables
thermal emission, and ``do_feedback`` triggers coupling between the radiation field and
the relevant |code| energy field (momentum coupling is not yet implemented).  Remaining
``<jaybenne>`` runtime parameters will be defined in forthcoming ``jaybenne``
documentation.

Finally, |jaybenne| interfaces with |code| gas fields via absorption and scattering
opacities constructed with ``singularity-opac``.  A user interfaces with opacity models
via additional input nodes, e.g.,

::

  <gas/opacity/absorption>
  opacity_model = constant
  kappa_a = 1.0

  <gas/opacity/scattering>
  scattering_model = constant
  kappa_s = 1.0

where the above enrolls constant specific absorption and scattering opacities ``kappa_a``
and ``kappa_s``, respectively.  By default, ``jaybenne`` operates in assumed CGS units.
Often test problems invoke custom code units; see ``opacity_model = thermalization``
and/or ``opacity_model == shocktube_a`` implementations for custom units enrollment.

In a ``<parthenon/output...>`` block, a user can pass ``field.jaybenne.energy_tally`` to
dump the radiation energy density, as computed by |jaybenne|.
