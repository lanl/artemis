# ========================================================================================
#  (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

import numpy as np
import matplotlib.pyplot as plt


class Binary:
    # This class holds the data from a *.orb file
    def __init__(self, fname):
        with open(fname, "r") as f:
            f.readline()
            header = f.readline()
            self.cols = [_.split("=")[1] for _ in header[1:].strip().split()]
        self.data = np.loadtxt(fname)
        for i in range(self.data.shape[1]):
            setattr(self, self.cols[i], self.data[:, i])


class NBody:
    # This class holds the data from the *.reb file and all *.p_*.orb files for
    # a user defined primary (indexed as p).
    # There are several plotting utilities availabe
    # Note that this class is built for planetary systems, but can be used as the basis
    # for arbitrary systems
    def __init__(self, base="disk", primary=0):
        # The data files are e.g., disk.reb and disk.primary_*.orb
        self.orb = []
        d = np.loadtxt(base + ".reb")
        with open(base + ".reb", "r") as f:
            self.N = int(f.readline().split("=")[1].strip())
            header = f.readline()
            self.cols = [_.split("=")[1] for _ in header[1:].strip().split()]

        self.data = d
        self.rdata = d.reshape(d.shape[0] // self.N, self.N, d.shape[1])
        for i in range(d.shape[1]):
            setattr(self, self.cols[i], self.rdata[:, :, i])
        for i in range(1, self.N):
            fname = base + ".{:d}_{:d}.orb".format(primary, i)
            try:
                self.orb.append(Binary(fname))
            except:
                pass

    def plot(self, q="x", ax=None, istart=1, **kargs):
        # Plot the variable "q" for all planets.
        # By default, the first particle is omitted, but it can be included by setting istart=0
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        fig = ax.get_figure()
        for i in range(istart, self.N):
            active = np.argwhere(self.active[:, i] > 0)
            (l,) = ax.plot(
                self.time[:, i][active], getattr(self, q)[:, i][active], **kargs
            )
            ax.plot(self.time[0, i], getattr(self, q)[0, i], "o", c=l._color)
        ax.minorticks_on()
        ax.tick_params(labelsize=14)
        ax.set_ylabel(q, fontsize=20)
        ax.set_xlabel("$\\Omega_0 t$", fontsize=20)
        fig.tight_layout()
        return fig, ax

    def dist_plot(self, ax=None, istart=1, **kargs):
        # Plot distance to origin vs time for all particles
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        fig = ax.get_figure()
        for i in range(istart, self.N):
            active = np.argwhere(self.active[:, i] > 0)
            (l,) = ax.plot(
                self.time[:, i][active],
                np.sqrt(self.x**2 + self.y**2 + self.z**2)[:, i][active],
                **kargs
            )
            ax.plot(
                self.time[0, i],
                np.sqrt(self.x[0, i] ** 2 + self.y[0, i] ** 2 + self.z[0, i] ** 2),
                "o",
                c=l._color,
            )
        ax.minorticks_on()
        ax.tick_params(labelsize=14)
        ax.set_ylabel("$R/R_0$", fontsize=20)
        ax.set_xlabel("$\\Omega_0 t$", fontsize=20)
        fig.tight_layout()
        return fig, ax

    def orb_plot(self, axes=None, **kargs):
        # Plot a and e vs time for all particles
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(8 * 2, 6))
        fig = axes[0].get_figure()
        for i in range(self.N - 1):
            (l,) = axes[0].plot(self.orb[i].time, self.orb[i].ab, **kargs)
            axes[0].plot(self.orb[i].time[0], self.orb[i].ab[0], "o", c=l._color)
            (l,) = axes[1].plot(self.orb[i].time, self.orb[i].eb, **kargs)
            axes[1].plot(self.orb[i].time[0], self.orb[i].eb[0], "o", c=l._color)
        for ax in axes:
            ax.minorticks_on()
            ax.tick_params(labelsize=14)
            ax.set_xlabel("$\\Omega_0 t$", fontsize=20)
        axes[0].set_ylabel("$a_b$", fontsize=20)
        axes[1].set_ylabel("$e_b$", fontsize=20)
        fig.tight_layout()
        return fig, ax

    def dist_orb_plot(self, axes=None, istart=1, **kargs):
        # Combines the dist_plot and orb_plot plots
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(8 * 3, 6))
        fig = axes[0].get_figure()
        self.dist_plot(ax=axes[0], istart=istart, **kargs)
        self.orb_plot(axes=axes[1:], **kargs)
        return fig, axes


class Planet:
    # This class creates a Planet with all of the necessary input parameters for Artemis
    def __init__(
        self, m=1e-5, a=1.0, e=0, i=0, f=180.0, o=0, O=0, r=0, eps=0.02, gamma=1.0
    ):
        self.q = m
        self.a = a
        self.e = e
        self.i = i
        self.f = f
        self.o = o
        self.O = O
        self.rh = a * (m / 3.0) ** (1.0 / 3)
        self.rs = eps * self.rh
        self.gamma = gamma
        self.beta = 0.0
        self.targ = 1.5 * self.rh
        self.r = r

    def output(self):
        # Outputs the necessary parameters needed by artemis
        # q a e i f o O rs gamma beta target radius
        return "{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}".format(
            self.q,
            self.a,
            self.e,
            self.i,
            self.f,
            self.o,
            self.O,
            self.rs,
            self.gamma,
            self.beta,
            self.targ,
            self.r,
        )

    def plot(self, ax):
        ax.plot(self.a, self.f, "ok")


def generate_system(n=7, m=1e-4, eps=0.2):
    # This generates the file "planets.txt" for an n planetary system.
    # The output file can then be passed to Artemis

    # Some simple uniform spacing
    a = 1.0 + np.arange(n) / (1.0 * (n - 1))
    fig, ax = plt.subplots(figsize=(8, 6))
    with open("planets.txt", "w") as g:
        lines = "# q a e i f o O rs gamma beta target radius\n"
        for i in range(n):
            pl = Planet(
                a=a[i], e=0, m=m, f=0, o=0, r=eps * a[i] * (m / 3.0) ** (1.0 / 3)
            )
            lines += pl.output() + "\n"
            pl.plot(ax)
        g.write(lines)
