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

import sys
import os
import numpy as np
import argparse


class ahistory:
    def __init__(self, filename):
        # Save filename
        self.filename = filename

        # Identify only latest region of file in case of multiple runs appending to the same history
        start_lines = []
        n = 0
        with open(self.filename, "r") as history_file:
            lines = history_file.readlines()
            for n, line in enumerate(lines):
                if line.strip() == "#  History data":
                    start_lines.append(n)
                    labels = lines[n + 1].strip().split("[")[1:]
                n = n + 1
        assert len(start_lines) >= 1, "Not a history file!"
        if len(start_lines) > 1:
            print(
                f"Warning! Multiple histories ({len(start_lines)}) appended to this file! Using only last history."
            )
        start_line = start_lines[-1]

        # Get data
        data = np.loadtxt(self.filename, skiprows=start_line)

        # Parse into dictionary
        self.dict = {}
        for n, full_label in enumerate(labels):
            label = full_label.split("=")[1].strip()
            self.dict[label] = data[:, n]

    def Get(self, label):
        if label not in self.dict.keys():
            print(f'Error: key "{label}" not found! Known keys:')
            for key in self.dict.keys():
                print(f"  {key}")
            return None

        return self.dict[label]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parse input arguments
    parser = argparse.ArgumentParser(description="Plot Artemis history output")
    parser.add_argument("filename", type=str, help="Artemis history file to plot")
    parser.add_argument("label", type=str, help="Quantity to plot")
    args = parser.parse_args()

    # Load history file
    history = ahistory(args.filename)

    # Get x (time) and y data
    time = history.Get("time")
    data = history.Get(args.label)

    # Only plot if quantity exists
    if data is not None:
        # Plot history for this quantity against time
        fig, ax = plt.subplots(1, 1)
        ax.plot(time, data)
        ax.set_xlim(time[0])
        ax.set_xlabel("time")
        ax.set_ylabel(args.label)

        # Save figure
        savename = f"hst_{args.label}.png"
        print(f"Saving plot as {savename}")
        plt.savefig(savename, dpi=300)
