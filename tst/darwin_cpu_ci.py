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
import requests
import sys
import json
import subprocess
import argparse
import tempfile
import shlex
from datetime import datetime
from launch_ci_runner import *

# The personal access token (PAT) with 'repo:status' permission
# Store your token securely and do not hardcode it in the script
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

    # Fetch PR information
    pr_info = get_pr_info(args.pr_number)
    head_repo = pr_info["head"]["repo"]["clone_url"]
    head_ref = pr_info["head"]["ref"]
    commit_sha = pr_info["head"]["sha"]

    # cpu context
    context = "Continuous Integration / darwin_skylake-gold"
    test_suite = "regression.suite"
    suffix = "cpu"

    if args.submission:
        # Update github PR status to indicate we have begun testing
        update_status(commit_sha, "pending", "CI Slurm job running...", context)

        # Run the tests in a temporary directory
        test_success = run_tests_in_temp_dir(
            args.pr_number, head_repo, head_ref, args.output_dir, test_suite, suffix
        )

        # Update github PR status to indicate that testing has concluded
        if test_success:
            update_status(commit_sha, "success", "All tests passed.", context)
        else:
            update_status(commit_sha, "failure", "Tests failed.", context)
    else:
        # Check that we are on the right system
        hostname = socket.gethostname()
        cluster = os.getenv("SLURM_CLUSTER_NAME")

        if not fnmatch.fnmatch(hostname, "darwin-fe*"):
            # if we are on a backend
            if cluster is None or cluster.lower() != "darwin":
                print("ERROR script must be run from Darwin!")
                sys.exit(1)

        # Execute the sbatch command
        try:
            # Submit batch job with ci_runner script that will checkout and build the code and run
            # tests
            job_name = f"artemis_ci_darwin_darwin_skylake-gold_PR{args.pr_number}"

            # Clean up existing jobs for same PR
            squeue_command = f"squeue --name={shlex.quote(job_name)} --user=$(whoami) --noheader  --format=%i"
            squeue_result = subprocess.run(
                squeue_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            job_ids = squeue_result.stdout.strip().split()
            if len(job_ids) >= 1:
                print("Canceling jobs:")
                for job_id in job_ids:
                    print(f"  {job_id}")

                # Use scancel to cancel the jobs
                scancel_command = ["scancel"] + job_ids
                scancel_result = subprocess.run(
                    scancel_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )

            # Build output path and create directory if necessary
            username = os.getenv("USER")
            current_date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            output_dir = os.path.join(
                "/usr",
                "projects",
                "jovian",
                "ci",
                f"pr_{args.pr_number}",
                current_date_time,
            )
            subprocess.run(["mkdir", "-p", output_dir], check=True)

            # Create subprocess command for submitting CI job, and submit
            sbatch_command = [
                "sbatch",
                f"--job-name={job_name}",
                f"--output={os.path.join(output_dir, job_name)}_%j.out",
                f"--error={os.path.join(output_dir, job_name)}_%j.out",
                "--partition=skylake-gold",
                "--time=04:00:00",
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

            # Update PR status that we have successfully submitted to SLURM job
            update_status(commit_sha, "pending", "CI SLURM job submitted...", context)
        except Exception as err:
            # Update PR status that we have failed to submit the SLURM job
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
                "SLURM job submission didn't complete sucessfully",
                context,
            )
