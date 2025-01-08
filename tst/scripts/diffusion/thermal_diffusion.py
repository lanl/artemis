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

# Regression to test steady-state conduction.

# Modules
import logging
import numpy as np
import os
import scripts.utils.artemis as artemis
from scipy.interpolate import interp1d


logger = logging.getLogger("artemis" + __name__[7:])  # set logger name
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import scripts.utils.analysis as analysis
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid issues with DISPLAY not being set

_nranks = 1
_file_id = "thermal_diffusion"

_geom = ["cartesian", "axisymmetric", "spherical"]
_flux = 0.01
_kcond = 0.1
_gtemp = 0.05
_gx1 = 0.0
_tol = 5e-3


# Run Artemis
def run(**kwargs):
    logger.debug("Runnning test " + __name__)
    for g in _geom:
        name = "{}_{}".format(_file_id, g[:3])
        arguments = [
            "parthenon/job/problem_id=" + name,
            "artemis/coordinates=" + g,
            "parthenon/time/tlim=50.0",
            "gas/conductivity/cond={:.8f}".format(_kcond),
            "gravity/uniform/gx1={:.8f}".format(_gx1),
            "problem/flux={:.8f}".format(_flux),
            "problem/gas_temp={:.8f}".format(_gtemp),
            "parthenon/meshblock/nx1={:d}".format(128 // _nranks),
        ]
        if g == "axisymmetric":
            arguments += [
                "parthenon/mesh/x2min={:.2f}".format(-0.5),
                "parthenon/mesh/x2max={:.2f}".format(0.5),
            ]
        elif g == "spherical":
            arguments += [
                "parthenon/mesh/x2min={:.8f}".format(np.pi / 2 - 0.5),
                "parthenon/mesh/x2max={:.8f}".format(np.pi / 2 + 0.5),
            ]

        artemis.run(_nranks, "diffusion/conduction.in", arguments)


def Tans(x, f=0.01, T0=0.05, x0=1.2, xi=0.2, d=0, k=0.1):
    f *= xi**d
    if d == 0:
        return T0 + (x - x0) * -f / k
    elif d == 1:
        return T0 + np.log(x / x0) * -f / k
    elif d == 2:
        return T0 + (1.0 / x - 1.0 / x0) * f / k


# Analyze outputs
def analyze():
    logger.debug("Analyzing test " + __name__)
    os.makedirs(artemis.get_fig_dir(), exist_ok=True)
    analyze_status = True

    fig, axes = plt.subplots(1, 3, figsize=(8 * 3, 6))
    dind = {"cartesian": 0, "axisymmetric": 1, "spherical": 2}
    errors = []
    for ax, g in zip(axes, _geom):
        name = "{}_{}".format(_file_id, g[:3])
        time, x, y, z, [d, u, v, w, T] = analysis.load_level(
            "final", dir=artemis.get_data_dir(), base="{}.out1".format(name)
        )
        xc = 0.5 * (x[1:] + x[:-1])
        ans = Tans(xc.ravel(), f=_flux, T0=_gtemp, x0=1.2, xi=0.2, d=dind[g], k=_kcond)
        temp = T[0, :].ravel()
        err = abs(temp / ans - 1.0)
        ax.plot(xc, ans, "--k")
        ax.plot(xc, temp)
        errors.append(err.mean())

    for ax in axes:
        ax.set_xlabel("$x$", fontsize=20)
        ax.set_ylabel("$T$", fontsize=20)
        ax.tick_params(labelsize=14)
        ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(
        os.path.join(artemis.get_fig_dir(), _file_id + "_temp.png"),
        bbox_inches="tight",
    )

    print(errors)
    analyze_status = all([err <= _tol for err in errors])

    return analyze_status
