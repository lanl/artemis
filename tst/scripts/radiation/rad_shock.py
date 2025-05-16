# ========================================================================================
#  (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

# Regression test based on the Lowrie&Edwards, Mach=3 Steady Radiation Shock
# Exact solutions are generated via the open source, Quokka script here:
# https://github.com/quokka-astro/quokka/blob/development/extern/LowrieEdwards/radshock.py

# Modules
import logging
import numpy as np
import os
import scipy.interpolate
import scripts.utils.artemis as artemis
import sys

logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to avoid issues with DISPLAY not being set
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.join(
        artemis.get_artemis_dir(),
        "external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools",
    ),
)
from phdf import phdf

# Plotting style
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Commands
_nranks = 1
_file_id = "shock"

# Constants
_kb = 1.3806488e-16
_ar = 7.5657308e-15
_c = 2.9979246e10
_amu = 1.6605389e-24

# Units
_time = 1.0e-10
_length = 0.01575
_temperature = 2.18e6
_density = 1.0
_mass = _density * _length**3.0
_velocity = _length / _time
_energy = _mass * _velocity**2.0

# Opacity
_ka0 = 577.0  # in CGS
_rho_exp = -1.0
_temp_exp = 0.0

# Constants
_tf = 6.0e-10 / _time
_gamma = 1.666666
_mu = 1.008

# L/R States
_rhol = 5.69 / _density
_rhor = 17.1 / _density
_vxl = 5.19e7 / _velocity
_vxr = 1.73e7 / _velocity
_tl = 2.18e6 / _temperature
_tr = 7.98e6 / _temperature
_xdisc = 0.01305 / _length

# Thresholds
# NOTE(@pdmullen): The particle count in this test is too low to get good statistics of
# the radiation energy density via a tally.  Future extensions of this test may get at the
# radiation temperature via a different means so that we can lower the trad threshold...
_thr_gas = 0.05
_thr_rad = 0.11


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + _file_id,
        "artemis/coordinates=cartesian",
        "artemis/physical_units=cgs",
        "artemis/unit_conversion=base",
        "artemis/time={:24.16e}".format(_time),
        "artemis/length={:24.16e}".format(_length),
        "artemis/mass={:24.16e}".format(_mass),
        "artemis/temperature={:24.16e}".format(_temperature),
        "gas/gamma={:24.16e}".format(_gamma),
        "gas/mu={:24.16e}".format(_mu),
        "gas/opacity/absorption/opacity_model=powerlaw",
        "gas/opacity/absorption/coef_kappa_a={:24.16e}".format(_ka0),
        "gas/opacity/absorption/rho_exp={:24.16e}".format(_rho_exp),
        "gas/opacity/absorption/temp_exp={:24.16e}".format(_temp_exp),
        "artemis/imc/num_particles=100000",
        "artemis/imc/use_ddmc=false",
        "problem/rhol={:24.16e}".format(_rhol),
        "problem/rhor={:24.16e}".format(_rhor),
        "problem/vxl={:24.16e}".format(_vxl),
        "problem/vxr={:24.16e}".format(_vxr),
        "problem/tl={:24.16e}".format(_tl),
        "problem/tr={:24.16e}".format(_tr),
        "problem/xdisc={:24.16e}".format(_xdisc),
    ]
    artemis.run(_nranks, "radiation/rad_shock.in", arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check that the correct equilibrium temperature
    # is obtained after reaching the final time.  Future extensions of this test could
    # test other jaybenne/dt to test how artemis fares for varying dt / (c rho kappa)^-1
    logger.debug("Analyzing test " + __name__)
    analyze_status = True

    # Derived constants
    gm1 = _gamma - 1.0
    cv = _kb / (_mu * _amu * gm1)

    # Grab Artemis datasets
    data = phdf(os.path.join(artemis.get_data_dir(), "shock.out1.final.phdf"))
    xc = 0.5 * (data.xng[0, 1:] + data.xng[0, :-1])
    sie = data.Get("gas.prim.sie_0", False, False)[0, 0, 0]
    erad = data.Get("field.jaybenne.energy_tally", False, False)[0, 0, 0]
    xc *= _length
    sie *= _energy / _mass
    erad *= _energy / _length**3.0
    tgas = np.array(sie / cv)
    trad = np.array((erad / _ar) ** 0.25)

    # Grab exact solution
    exact = np.loadtxt(
        os.path.join(artemis.get_artemis_dir(), "tst/scripts/radiation/rad_shock.dat"),
        unpack=True,
    )
    int_tg = scipy.interpolate.interp1d(exact[0], exact[1], kind="linear")
    int_tr = scipy.interpolate.interp1d(exact[0], exact[2], kind="linear")
    tgas_exact = int_tg(xc)
    trad_exact = int_tr(xc)

    # Plot results
    os.makedirs(artemis.get_fig_dir(), exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(xc, tgas, label="$T_\\mathrm{gas}$", lw=4, alpha=0.25, color=colors[0])
    ax1.scatter(xc, trad, label="$T_\\mathrm{rad}$", lw=4, alpha=0.25, color=colors[1])
    ax1.plot(xc, tgas_exact, label="$T_\\mathrm{gas}$", lw=2, color=colors[0])
    ax1.plot(xc, trad_exact, label="$T_\\mathrm{rad}$", lw=2, color=colors[1])
    ax1.set_xlabel("$x \\; (\\mathrm{cm})$")
    ax1.set_ylabel("$T \\; (\\mathrm{K})$")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.5)
    ax1.xaxis.set_ticks_position("both")
    ax1.yaxis.set_ticks_position("both")
    ax1.tick_params(which="both", direction="in")
    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(os.path.join(artemis.get_fig_dir(), "rad_shock.png"))

    # Check if solution errors are above threshold.  See notes above regarding thresholds.
    dx = xc[1] - xc[0]
    l1_tgas = dx * np.sum(np.abs(tgas - tgas_exact)) / _temperature / _length
    l1_trad = dx * np.sum(np.abs(trad - trad_exact)) / _temperature / _length
    print("l1_tgas: ", l1_tgas, "l1_trad: ", l1_trad)
    if l1_tgas > _thr_gas:
        logger.warning(
            "Error in gas temperature solution is greater than threshold: "
            "l1_tgas: {0} thr: {1}".format(l1_tgas, _thr_gas)
        )
        analyze_status = False
    if l1_trad > _thr_rad:
        logger.warning(
            "Error in radiation temperature solution is greater than threshold: "
            "l1_trad: {0} thr: {1} ".format(l1_trad, _thr_rad)
        )
        analyze_status = False

    return analyze_status
