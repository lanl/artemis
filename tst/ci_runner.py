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

import socket
import fnmatch
import os
import subprocess
import requests
import sys
import json
import tempfile

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


def update_status(
    commit_sha, state, description, context="Continuous Integration / darwin_volta-x86"
):
    url = f"https://api.github.com/repos/lanl/artemis/statuses/{commit_sha}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"state": state, "description": description, "context": context}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 201:
        print(f"Error setting status: {response.status_code}")
        print(response.text)
        sys.exit(1)


# def run_tests():
#    try:
#        subprocess.run(['python3', 'run_tests.py', 'regression.suite',
#                        '--save_build', '--make_nproc=4',
#                        '--cmake=-DCMAKE_C_COMPILER=gcc',
#                        '--cmake=-DCMAKE_CXX_COMPILER=g++',
#                        '--log_file=ci_cpu_log.txt'], check=True)
#        return True
#    except subprocess.CalledProcessError:
#        return False
def run_tests_in_temp_dir(pr_number, head_repo, head_ref, commit_sha):
    current_dir = os.getcwd()

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Clone the repository into the temporary directory
        subprocess.run(["git", "clone", head_repo, temp_dir], check=True)
        os.chdir(temp_dir)

        # Checkout the PR branch
        subprocess.run(["git", "fetch", "origin", head_ref], check=True)
        subprocess.run(["git", "checkout", head_ref], check=True)

        # Update submodules
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"], check=True
        )

        # Run the tests
        try:
            os.chdir(os.path.join(temp_dir, "tst"))
            build_dir = os.path.join(temp_dir, "build")
            test_command = [
                "bash",
                "-c",
                "source ../env/bash && build_artemis -b "
                + build_dir
                + " -j 4 -f && cd "
                + os.path.join(temp_dir, "tst")
                + " && python3 run_tests.py regression.suite "
                "--exe " + os.path.join(build_dir, "src", "artemis") + " "
                "--log_file=ci_cpu_log.txt",
            ]
            # print(test_command)
            subprocess.run(test_command, check=True)
            return True
            # Update the status to success
            # update_status(commit_sha, "success", "All tests passed.")
            # print("Tests passed.")
        except subprocess.CalledProcessError:
            return False
            # Update the status to failure
            # update_status(commit_sha, "failure", "Tests failed.")
            # print("Tests failed.")
            # sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: ci_runner.py [PR number]")
        sys.exit(1)

    pr_number = sys.argv[1]

    # Check that we are on the right system
    # hostname = socket.gethostname()
    # if not fnmatch.fnmatch(hostname, "ch-fe*"):
    #    print("ERROR script must be run from Chicoma frontend node!")
    #    sys.exit(1)

    # Fetch PR information
    pr_info = get_pr_info(pr_number)
    head_repo = pr_info["head"]["repo"]["clone_url"]
    head_ref = pr_info["head"]["ref"]
    commit_sha = pr_info["head"]["sha"]

    update_status(commit_sha, "pending", "CI Slurm job running...")

    # Run the tests in a temporary directory
    test_success = run_tests_in_temp_dir(pr_number, head_repo, head_ref, commit_sha)

    if test_success:
        update_status(commit_sha, "success", "All tests passed.")
    else:
        update_status(commit_sha, "failure", "Tests failed.")


if __name__ == "__main__":
    main()
