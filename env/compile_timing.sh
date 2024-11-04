#!/bin/bash
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
# This compile time Bash script was developed with the assistance of ChatGPT, an AI
# language model created by OpenAI.
#=========================================================================================

# Extract logfile path from input arguments
LOGFILE_PATH="$1"

# Ignore first argument from now on
shift

# Convert the command arguments to a string
filename="${@: -1}"

# Record the start time
start_time=$(date +%s)

# Run the compiler command passed to the script
"$@"

# Store whether compilation succeeded
compile_status=$?

# Report timing if compilation succeeded
if [ $compile_status -eq 0 ]; then

  # Record the end time
  end_time=$(date +%s)

  # Calculate duration
  duration=$((end_time - start_time))

  # Save the duration to the log file
  echo "Compilation of ${filename} took ${duration} s" >> "${LOGFILE_PATH}" #compile_times.log

  # Print the duration
  echo "Compiled ${filename} in ${duration} s"

else

  echo "Compilation of ${filename} failed!"

fi

exit $compile_status
