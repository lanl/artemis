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

# Provide path directly to phdf to avoid need to install parthenon_tools package
ahdf_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(
    0,
    os.path.join(
        ahdf_dir,
        "../external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/",
    ),
)
import phdf


# Class for opening an Artemis dump file and providing access to data.
# Wraps the Parthenon phdf class and provides additional Artemis-specific data,
# mainly related to the mesh.
class ahdf(phdf.phdf):
    def __init__(self, filename):
        super().__init__(filename)

        self.coordinates = self.Params["artemis/coord_sys"]

        # Get meshblock coordinate properties
        self.NX1 = self.MeshBlockSize[0]
        self.NX2 = self.MeshBlockSize[1]
        self.NX3 = self.MeshBlockSize[2]
        self.DX1 = np.zeros(self.NumBlocks)
        self.DX2 = np.zeros(self.NumBlocks)
        self.DX3 = np.zeros(self.NumBlocks)
        for b in range(self.NumBlocks):
            self.DX1[b] = (self.BlockBounds[b][1] - self.BlockBounds[b][0]) / self.NX1
            self.DX2[b] = (self.BlockBounds[b][3] - self.BlockBounds[b][2]) / self.NX2
            self.DX3[b] = (self.BlockBounds[b][5] - self.BlockBounds[b][4]) / self.NX3

        # Get node coordinates for each meshblock for plotting
        self.X1 = np.zeros([self.NumBlocks, self.NX3 + 1, self.NX2 + 1, self.NX1 + 1])
        self.X2 = np.zeros([self.NumBlocks, self.NX3 + 1, self.NX2 + 1, self.NX1 + 1])
        self.X3 = np.zeros([self.NumBlocks, self.NX3 + 1, self.NX2 + 1, self.NX1 + 1])
        for b in range(self.NumBlocks):
            for k in range(self.NX3 + 1):
                for j in range(self.NX2 + 1):
                    for i in range(self.NX1 + 1):
                        self.X1[b, k, j, i] = self.BlockBounds[b][0] + i * self.DX1[b]
                        self.X2[b, k, j, i] = self.BlockBounds[b][2] + j * self.DX2[b]
                        self.X3[b, k, j, i] = self.BlockBounds[b][4] + k * self.DX3[b]

        # Convert to Cartesian coordinates
        self.xmin = np.zeros(self.NumBlocks)
        self.xmax = np.zeros(self.NumBlocks)
        self.ymin = np.zeros(self.NumBlocks)
        self.ymax = np.zeros(self.NumBlocks)
        self.zmin = np.zeros(self.NumBlocks)
        self.zmax = np.zeros(self.NumBlocks)
        if self.coordinates == "cartesian":
            self.x = self.X1
            self.y = self.X2
            self.z = self.X3
        elif self.coordinates == "cylindrical":
            self.x = self.X1 * np.cos(self.X2)
            self.y = self.X1 * np.sin(self.X2)
            self.z = self.X3
        elif self.coordinates == "spherical":
            self.x = self.X1 * np.sin(self.X2) * np.cos(self.X3)
            self.y = self.X1 * np.sin(self.X2) * np.sin(self.X3)
            self.z = self.X1 * np.cos(self.X2)
        else:
            print(f'Coordinate system "{self.coordinates}" is unsupported!')

        # Store extents of each meshblock.
        for b in range(self.NumBlocks):
            self.xmin[b] = np.min(self.x[b, :, :, :])
            self.xmax[b] = np.max(self.x[b, :, :, :])
            self.ymin[b] = np.min(self.y[b, :, :, :])
            self.ymax[b] = np.max(self.y[b, :, :, :])
            self.zmin[b] = np.min(self.z[b, :, :, :])
            self.zmax[b] = np.max(self.z[b, :, :, :])

    # Returns data for a particular variable. Reports variables available in the dump
    # file if an invalid variable_name is provided.
    def Get(self, variable_name, flatten=False, report_available=True):
        variable = super().Get(variable_name, flatten)

        if variable is None and report_available:
            print("Variables contained in this dump file:")
            for name in self.Variables:
                if name not in [
                    "Blocks",
                    "Info",
                    "Input",
                    "Levels",
                    "Locations",
                    "LogicalLocations",
                    "Params",
                    "SparseInfo",
                    "VolumeLocations",
                ]:
                    print(f"  {name}")
            print("")

        return variable
