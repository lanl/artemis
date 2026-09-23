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


def run_tests_in_temp_dir(
    pr_number, head_repo, head_ref, output_dir, test_suite, suffix
):
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

        # Run subprocess command to compile code and launch run_tests.py
        test_command = [
            "bash",
            "-c",
            "source ../env/bash && build_artemis -b "
            + build_dir
            + " -j 20 -f && cd "
            + os.path.join(temp_dir, "tst")
            + " && python3 run_tests.py "
            + test_suite
            + " "
            + "--exe "
            + os.path.join(build_dir, "src", "artemis")
            + f" --output_dir={output_dir}"
            + " --log_file=darwin_log_"
            + suffix
            + ".txt"
            + " --erase_data",
        ]
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

    run_cmd = ["./darwin_cpu_ci.py", str(args.pr_number)]
    if args.output_dir:
        run_cmd.append("--output_dir=" + str(args.output_dir))

    subprocess.run(run_cmd)

    run_cmd = ["./darwin_amd_ci.py", str(args.pr_number)]
    if args.output_dir:
        run_cmd.append("--output_dir=" + str(args.output_dir))

    subprocess.run(run_cmd)

    run_cmd = ["./darwin_gpu_ci.py", str(args.pr_number)]
    if args.output_dir:
        run_cmd.append("--output_dir=" + str(args.output_dir))

    subprocess.run(run_cmd)
