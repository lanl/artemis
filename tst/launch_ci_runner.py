#!/usr/bin/env python3
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

# This file was created in part or in whole by one of OpenAI's generative AI models

import subprocess
import socket
import fnmatch
import os
import sys
import subprocess

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ci_runner.py [PR number]")
        sys.exit(1)
    pr_number = sys.argv[1]

    # Check that we are on the right system
    hostname = socket.gethostname()
    if not fnmatch.fnmatch(hostname, "darwin-fe*"):
        print("ERROR script must be run from Darwin frontend node!")
        sys.exit(1)

    # sbatch_command = [
    #    'sbatch',
    #    '-A', 't24_ngpfc_g',
    #    '--job-name=ci_gpu_job',
    #    '--output=ci_gpu_job.out',
    #    '--error=ci_gpu_job.err',
    #    '--time=00:10:00',
    #    '-N', '1',
    #    '-p', 'gpu',
    #    '--qos=standard',
    #    '-C', 'gpu40',
    #    '--tasks-per-node=4',
    #    '--exclusive',
    #    '--mem=0',
    #    '--wrap',  # Wraps the following command as a single string
    #    f'python3 ci_runner.py {pr_number}'
    # ]
    # print(sbatch_command)
    sbatch_command = [
        "sbatch",
        "--job-name=artemis_ci_darwin_volta-x86",
        "--partition=volta-x86",
        "--time=04:00:00",
        "--wrap",
        f"python3 ci_runner.py {pr_number}",
    ]

    # Execute the sbatch command
    result = subprocess.run(sbatch_command, stdout=subprocess.PIPE, check=True)

    # Print the job ID
    # print(result)
    # for line in result.stdout.splitlines():
    #    if "Submitted batch job" in line:
    #        job_id = line.split()[-1]
    #        print(f"Job submitted with ID: {job_id}")

    # raise RuntimeError("Failed to submit Slurm job.")
