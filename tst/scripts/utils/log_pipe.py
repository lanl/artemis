# ========================================================================================
#  (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

# NOTE(@pdmullen): The following is largely borrowed from the open-source Athena++/AthenaK
# softwares.

# Provides LogPipe class to pipe output from subprocess to a log.
# Adapted from https://codereview.stackexchange.com/questions/6567

# Modules
import logging
import threading
import os


class LogPipe(threading.Thread):
    # Setup object with logger and a loglevel and start the thread
    def __init__(self, logger, level):
        super(LogPipe, self).__init__()
        # threading.Thread.__init__(self)
        self.logger = logging.getLogger(logger)
        self.daemon = False
        self.level = level
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()

    # Return the write file descriptor of the pipe
    def fileno(self):
        return self.fdWrite

    # Run the thread, logging everything.
    def run(self):
        for line in iter(self.pipeReader.readline, ""):
            self.logger.log(self.level, line.strip("\n"))
        self.pipeReader.close()

    # Close the write end of the pipe."""
    def close(self):
        os.close(self.fdWrite)
