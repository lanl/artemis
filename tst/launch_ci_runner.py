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

# The personal access token (PAT) with 'repo:status' permission
# Store your token securely and do not hardcode it in the script
GITHUB_TOKEN = os.environ.get("ARTEMIS_GITHUB_TOKEN")


def get_pr_info(pr_number):
    url = f"https://api.github.com/repos/lanl/artemis/pulls/{pr_number}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching PR info: {response.status_code}")
        print(response.text)
        sys.exit(1)
    return response.json()


def update_status(commit_sha, state, description, context):
    url = f"https://api.github.com/repos/lanl/artemis/statuses/{commit_sha}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"state": state, "description": description, "context": context}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 201:
        print(f"Error setting status: {response.status_code}")
        print(response.text)
        sys.exit(1)


def run_tests_in_temp_dir(pr_number, head_repo, head_ref, output_dir, test_cmd, suffix):
    current_dir = os.getcwd()

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Clone the repository into the temporary directory
        subprocess.run(["git", "clone", head_repo, temp_dir], check=True)
        os.chdir(temp_dir)

        # Checkout the PR branch
        subprocess.run(["git", "pull", "--no-rebase", "origin", head_ref], check=True)

        # Update submodules
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"], check=True
        )

        # Run the tests
        os.chdir(os.path.join(temp_dir, "tst"))
        build_dir = os.path.join(temp_dir, "build")

        test_command = [
            "bash",
            "-c",
            "source ../env/bash && build_artemis -b "
            + build_dir
            + " -j 20 -f && cd "
            + os.path.join(temp_dir, "tst")
            + test_cmd
            + "--exe "
            + os.path.join(build_dir, "src", "artemis")
            + f" --output_dir={output_dir}"
            + " --log_file=darwin_log_" + suffix + ".txt"
            + " --erase_data",
    ]

        # Run subprocess command to compile code and launch run_tests.py
        try:
            ret = subprocess.run(test_command, check=True)
            result = ret.returncode == 0
        except:
            result = False

        # Set group ownership
        subprocess.run(
            ["chgrp", "-R", "jovian", output_dir],
            check=True,
        )

        # Set permissions for directories
        subprocess.run(
            ["find", output_dir, "-type", "d", "-exec", "chmod", "770", "{}", "+"],
            check=True,
        )

        # Set permissions for files
        subprocess.run(
            ["find", output_dir, "-type", "f", "-exec", "chmod", "660", "{}", "+"],
            check=True,
        )

        # Return true if the test script succeeded
        return result


def run_test(args, test_context, test_cmd, sbatch_cmd, suffix):
    if args.submission:
        # Update github PR status to indicate we have begun testing
        update_status(commit_sha, "pending", "CI Slurm job running...", context=test_context)

        # Run the tests in a temporary directory
        test_success = run_tests_in_temp_dir(
            args.pr_number, head_repo, head_ref, args.output_dir, test_cmd, suffix
        )

        return test_success

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
            job_name = f"artemis_ci_PR{args.pr_number}_" + suffix

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
                current_date_time + "_" + suffix
            )
            subprocess.run(["mkdir", "-p", output_dir], check=True)
    
            sbatch_command = [
                "sbatch",
                f"--job-name={job_name}",
                f"--output={os.path.join(output_dir, job_name)}_%j.out",
                f"--error={os.path.join(output_dir, job_name)}_%j.out",
                sbatch_cmd,
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
            update_status(commit_sha, "pending", "CI SLURM job submitted...", context=test_context)
        except Exception as err:
            # Update PR status that we have failed to submit the SLURM job
            update_status(
                commit_sha,
                "failure",
                "SLURM job submission failed with error: " + repr(err),
                context=test_context
            )

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

    # set test specific context and commands
    gpu_context = "Continuous Integration / darwin_volta-x86"
    test_cmd_gpu = " && python3 run_tests.py gpu.suite "
    sbatch_cmd_gpu = "--partition=volta-x86"


    cpu_context = "Continuous Integration / darwin_skylake-gold"
    test_cmd_cpu = " && python3 run_tests.py regression.suite "
    sbatch_cmd_cpu = "--partition=skylake-gold"

    # gpu tests
    gpu_test_success = run_test(args, test_context=gpu_context, test_cmd=test_cmd_gpu, sbatch_cmd=sbatch_cmd_gpu, suffix="gpu")

    # Update github PR status to indicate that gpu testing has concluded
    if gpu_test_success:
        update_status(commit_sha, "success", "All tests passed.", context=gpu_context)
    else:
        update_status(commit_sha, "failure", "Tests failed.", context=gpu_context)

    # cpu_tests
    cpu_test_success = run_test(args, test_context=cpu_context, test_cmd=test_cmd_cpu, sbatch_cmd=sbatch_cmd_cpu, suffix="cpu")

    # Update github PR status to indicate that cpu testing has concluded
    if cpu_test_success:
        update_status(commit_sha, "success", "All tests passed.", context=cpu_context)
    else:
        update_status(commit_sha, "failure", "Tests failed.", context=cpu_context)
