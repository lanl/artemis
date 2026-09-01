.. artemis documentation master file, created by
   sphinx-quickstart on Tue Aug 27 07:51:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

artemis documentation
=====================

|code| is a performance-portable, multi-fluid, AMR hydrodynamics code built on top of the `Parthenon`_ mesh refinement library.
|code| solves the multi-fluid Navier-Stokes equations:

.. math::

   \frac{D \rho_i}{D t} & = -\rho \nabla \cdot \mathbf{v_i}   \\\\
   \rho_i \frac{D \mathbf{v}_i }{D t} & = -\nabla P_i + \rho_i \mathbf{g}_i + \nabla \cdot \mathbf{\Pi}_i + \sum_j \mathbf{R}_{ij}  \\\\
   \rho_i \frac{D e_i }{D t} & = - P_i \nabla \cdot \mathbf{v}_i + \mathbf{\Pi}_i : \nabla \mathbf{v}_i + \Gamma - \Lambda

where :math:`(\rho_i, e_i, \mathbf{v}_i)` are the material density, specific internal energy, and velocitity. 
The convective derivative is defined as :math:`D/Dt = \partial_t + \mathbf{v}_i \cdot \nabla`. 
Material temperature and pressure are specified by an equation of state EOS written in terms of the density and specific internal energy, :math:`T_i(\rho_i,e_i)` and :math:`P_i(\rho_i, e_i)`. 
Fluids interact via the collision operator, :math:`\mathbf{R}_{ij}`. 



Key Features
^^^^^^^^^^^^^
* Adaptive and static mesh refinement.
* Runs on CPUs or GPUs. 
* First- to third-order time stepping. 
* Full support for Cartesian or curvilinear coordinate systems, including axisymmetric, cylindrical, and spherical coordinates. 
* Various microphysics processes such as viscosity, cooling, and conduction.
* Dust coagulation models
* Coupling to arbitrarily complicated N-Body systems. 
* One executable design. No need to recompile. 

A full listing of the available physics in |code| can be found in :ref:`physics`.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :glob:

   src/*


.. _Parthenon: https://github.com/parthenon-hpc-lab/parthenon