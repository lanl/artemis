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

# Regression test based on the thermal relaxation of gas and radiation fields

# Modules
import logging
import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
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
_file_id = "thermalization"

# Constants
_kb = 1.3806488e-16
_ar = 7.5657308e-15
_c = 2.9979246e10
_amu = 1.6605389e-24

# Inputs
_rho = 1.0e-3
_ka = 2.0
_gamma = 1.4
_mu = 28.96
_tr0 = 5.0e5
_tg0 = 1.0e6
_tf = 6.0e-8
_thr = 5.0e-3


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    arguments = [
        "parthenon/job/problem_id=" + _file_id,
        "artemis/coordinates=cartesian",
        "artemis/physical_units=cgs",
        "artemis/unit_conversion=base",
        "parthenon/time/tlim={:24.16e}".format(_tf),
        "gas/gamma={:24.16e}".format(_gamma),
        "gas/mu={:24.16e}".format(_mu),
        "gas/opacity/absorption/opacity_model=constant",
        "gas/opacity/absorption/kappa_a={:24.16e}".format(_ka),
        "artemis/imc/dt=1.0e-10",
        "artemis/imc/num_particles=200000",
        "artemis/imc/use_ddmc=true",
        "problem/rho={:24.16e}".format(_rho),
        "problem/vx=0.0",
        "problem/tgas={:24.16e}".format(_tg0),
        "problem/trad={:24.16e}".format(_tr0),
    ]
    artemis.run(_nranks, "radiation/thermalization.in", arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check that the correct equilibrium temperature
    # is obtained after reaching the final time.  Future extensions of this test could
    # test other jaybenne/dt to test how artemis fares for varying dt / (c rho kappa)^-1
    logger.debug("Analyzing test " + __name__)
    analyze_status = True
    data_dir = artemis.get_data_dir()
    files = sorted(
        [
            os.path.abspath(os.path.join(data_dir, f))
            for f in os.listdir(data_dir)
            if f.startswith("thermalization") and not f.endswith(".xdmf")
        ]
    )

    # Derived constants
    gm1 = _gamma - 1.0
    cv = _kb / (_mu * _amu * gm1)

    # Grab Artemis datasets
    tt = []
    tgas = []
    trad = []
    for file in files:
        data = phdf(file)
        sie = data.Get("gas.prim.sie_0", False, False)[0, 0, 0]
        erad = data.Get("field.jaybenne.energy_tally", False, False)[0, 0, 0]
        tt.append(data.Time)
        tgas.append(sie / cv)
        trad.append((erad / _ar) ** 0.25)
    tgas = np.array(tgas)
    trad = np.array(trad)

    # Thermal equilibrium ODEs
    def rad_equil(t, y):
        tr, tg = y
        dtrdt = _c * _rho * _ka * (tg**4.0 - tr**4.0) / (4.0 * tr**3.0)
        dtgdt = -_c * _rho * _ka * (tg**4.0 - tr**4.0) / (_rho * cv / _ar)
        return [dtrdt, dtgdt]

    # Find exact temporal evolution (coupled ODEs) and equilibrium temperature (root find)
    tlim = [0, _tf]
    tsol = np.linspace(tlim[0], tlim[1], 1000)
    eg0 = _rho * cv * _tg0
    er0 = _ar * _tr0**4.0
    sol = solve_ivp(
        rad_equil,
        tlim,
        [_tr0, _tg0],
        t_eval=tsol,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    root = fsolve(
        lambda x: eg0 + er0 - _rho * cv * x - _ar * x**4.0, 0.5 * (_tr0 + _tg0)
    )

    # Plot results
    os.makedirs(artemis.get_fig_dir(), exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(
        tt, tgas[:, 0], label="$T_\\mathrm{gas}$", lw=4, alpha=0.25, color=colors[0]
    )
    ax1.plot(tt, tgas[:, 1], lw=4, alpha=0.25, color=colors[0])
    ax1.plot(tt, tgas[:, 2], lw=4, alpha=0.25, color=colors[0])
    ax1.plot(tt, tgas[:, 3], lw=4, alpha=0.25, color=colors[0])
    ax1.plot(
        tt, trad[:, 0], label="$T_\\mathrm{rad}$", lw=4, alpha=0.25, color=colors[1]
    )
    ax1.plot(tt, trad[:, 1], lw=4, alpha=0.25, color=colors[1])
    ax1.plot(tt, trad[:, 2], lw=4, alpha=0.25, color=colors[1])
    ax1.plot(tt, trad[:, 3], lw=4, alpha=0.25, color=colors[1])
    ax1.axhline(root, label="$T_\\mathrm{eq}$", lw=4, color=colors[2])
    ax1.plot(sol.t, sol.y[0], label="Exact", lw=2, color="black", ls="--")
    ax1.plot(sol.t, sol.y[1], lw=2, color="black", ls="--")
    ax1.set_xlabel("$t \\; (\\mathrm{s})$")
    ax1.set_ylabel("$T \\; (\\mathrm{K})$")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.5)
    ax1.xaxis.set_ticks_position("both")
    ax1.yaxis.set_ticks_position("both")
    ax1.tick_params(which="both", direction="in")
    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(os.path.join(artemis.get_fig_dir(), "thermalization.png"))

    # Check if any cell has temperature deviating from equilibrium temperature
    if np.any(np.abs(tgas[-1] - root) / root > _thr):
        logger.warning(
            "Gas temperature exceeds error threshold: "
            "tgas: {0} teq: {1} thr: {2} ".format(tgas[-1], root, _thr)
        )
        analyze_status = False
    if np.any(np.abs(trad[-1] - root) / root > _thr):
        logger.warning(
            "Radiation temperature exceeds error threshold: "
            "trad: {0} teq: {1} thr: {2} ".format(trad[-1], root, _thr)
        )
        analyze_status = False

    return analyze_status
