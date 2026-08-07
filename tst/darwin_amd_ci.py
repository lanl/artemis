#!/usr/bin/env python3
# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

import argparse
import fnmatch
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime

from launch_ci_runner import *

GITHUB_TOKEN = os.environ.get("ARTEMIS_GITHUB_TOKEN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CI tasks with optional Slurm submission."
    )
    parser.add_argument(
        "pr_number", type=int, help="Pull request number for the CI run."
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Flag to indicate the script is running as a Slurm submission job.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory created when launching submission script",
    )
    args = parser.parse_args()

    pr_info = get_pr_info(args.pr_number)
    head_repo = pr_info["head"]["repo"]["clone_url"]
    head_ref = pr_info["head"]["ref"]
    commit_sha = pr_info["head"]["sha"]

    context = "Continuous Integration / darwin_mi250"
    test_suite = "gpu.suite"
    suffix = "amd"

    if args.submission:
        update_status(commit_sha, "pending", "CI Slurm job running...", context)
        test_success = run_tests_in_temp_dir(
            args.pr_number, head_repo, head_ref, args.output_dir, test_suite, suffix
        )
        if test_success:
            update_status(commit_sha, "success", "All tests passed.", context)
        else:
            update_status(commit_sha, "failure", "Tests failed.", context)
    else:
        hostname = socket.gethostname()
        cluster = os.getenv("SLURM_CLUSTER_NAME")
        if not fnmatch.fnmatch(hostname, "darwin-fe*"):
            if cluster is None or cluster.lower() != "darwin":
                print("ERROR script must be run from Darwin!")
                sys.exit(1)

        try:
            job_name = f"artemis_ci_darwin_mi250_PR{args.pr_number}"
            squeue_command = (
                f"squeue --name={shlex.quote(job_name)} --user=$(whoami) "
                "--noheader --format=%i"
            )
            squeue_result = subprocess.run(
                squeue_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            job_ids = squeue_result.stdout.strip().split()
            if job_ids:
                print("Canceling jobs:")
                for job_id in job_ids:
                    print(f"  {job_id}")
                subprocess.run(
                    ["scancel"] + job_ids,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )

            current_date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            output_dir = os.path.join(
                "/usr",
                "projects",
                "jovian",
                "ci",
                "artemis",
                f"pr_{args.pr_number}",
                current_date_time,
            )
            subprocess.run(["mkdir", "-p", output_dir], check=True)

            sbatch_command = [
                "sbatch",
                f"--job-name={job_name}",
                f"--output={os.path.join(output_dir, job_name)}_%j.out",
                f"--error={os.path.join(output_dir, job_name)}_%j.out",
                "--partition=shared-gpu-amd-mi250",
                "--time=02:00:00",
                "--wrap",
                f"python3 {sys.argv[0]} {args.pr_number} --submission --output_dir {output_dir}",
            ]
            result = subprocess.run(
                sbatch_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                universal_newlines=True,
            )
            print(result.stdout.strip())
            update_status(commit_sha, "pending", "CI SLURM job submitted...", context)
        except Exception as err:
            update_status(
                commit_sha,
                "failure",
                "SLURM job submission failed with error: " + repr(err),
                context,
            )
        finally:
            update_status(
                commit_sha,
                "failure",
                "SLURM job submission didn't complete successfully",
                context,
            )
